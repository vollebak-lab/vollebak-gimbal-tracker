#pragma once
/**
 * @file hot_pixel_mask.hpp
 * @brief Hardware pixel masking infrastructure for Sony IMX636 (Phase 33.7a).
 *
 * Provides:
 * 1. File parser and validator for `hot_pixels.txt` (bounds checking, deduplication, hardware limit enforcement).
 * 2. Hardware mask programming via Metavision HAL facilities:
 *    - Primary: `Metavision::I_DigitalEventMask` (IMX636 / Gen4.1, 64 hardware mask registers);
 *    - Fallback: `Metavision::I_RoiPixelMask` (GenX320).
 * 3. File serializer for survey utilities.
 */

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <metavision/hal/device/device.h>
#include <metavision/hal/facilities/i_digital_event_mask.h>
#include <metavision/hal/facilities/i_roi_pixel_mask.h>

namespace predator {

/// A hot pixel record with spatial coordinates and optional measured firing rate.
struct HotPixel {
    uint16_t x{0};
    uint16_t y{0};
    double rate{0.0};
};

/// Hardware constraints for the sensor.
struct HotPixelMaskConfig {
    size_t max_masks{64};       ///< IMX636 Gen41DigitalEventMask provides 64 hardware mask registers
    uint16_t sensor_width{1280}; ///< IMX636 native width
    uint16_t sensor_height{720}; ///< IMX636 native height
};

/// Result of parsing and validating a hot_pixels.txt file.
struct HotPixelParseResult {
    bool success{false};
    std::vector<HotPixel> pixels;
    std::vector<std::string> warnings;
    std::string error;
};

/// Status of hardware mask application on the device.
struct HardwareMaskStatus {
    bool applied{false};
    std::string facility_name{"none"};
    size_t count{0};
    std::string message;
};

/**
 * @brief Parses and validates a hot pixel definition file.
 * @param filepath Path to hot_pixels.txt
 * @param cfg Sensor configuration and limits
 * @return HotPixelParseResult containing valid pixels, warnings, or error
 */
inline HotPixelParseResult parse_hot_pixels_file(
    const std::string& filepath,
    const HotPixelMaskConfig& cfg = HotPixelMaskConfig{})
{
    HotPixelParseResult res;
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
        res.error = "Cannot open hot pixels file: " + filepath;
        return res;
    }

    std::set<std::pair<uint16_t, uint16_t>> seen_coords;
    std::string line;
    size_t line_num = 0;

    while (std::getline(ifs, line)) {
        ++line_num;
        // Trim leading whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue; // blank line
        if (line[start] == '#') continue;        // comment line

        std::istringstream iss(line.substr(start));
        long long x_in = -1;
        long long y_in = -1;
        double rate_in = 0.0;

        if (!(iss >> x_in >> y_in)) {
            res.warnings.push_back("Line " + std::to_string(line_num) + ": Malformed line ignored: \"" + line + "\"");
            continue;
        }

        // Optional third column: rate (ev/s)
        if (!(iss >> rate_in)) {
            rate_in = 0.0;
        }

        // Coordinate bounds validation
        if (x_in < 0 || x_in >= cfg.sensor_width || y_in < 0 || y_in >= cfg.sensor_height) {
            res.warnings.push_back("Line " + std::to_string(line_num) + ": Coordinates out of bounds (" +
                                  std::to_string(x_in) + ", " + std::to_string(y_in) +
                                  ") for geometry " + std::to_string(cfg.sensor_width) + "x" +
                                  std::to_string(cfg.sensor_height));
            continue;
        }

        uint16_t x = static_cast<uint16_t>(x_in);
        uint16_t y = static_cast<uint16_t>(y_in);

        // Deduplication
        if (seen_coords.find({x, y}) != seen_coords.end()) {
            res.warnings.push_back("Line " + std::to_string(line_num) + ": Duplicate pixel (" +
                                  std::to_string(x) + ", " + std::to_string(y) + ") ignored");
            continue;
        }
        seen_coords.insert({x, y});

        // Hardware mask limit
        if (res.pixels.size() >= cfg.max_masks) {
            res.warnings.push_back("Line " + std::to_string(line_num) +
                                  ": Exceeded hardware mask capacity limit of " +
                                  std::to_string(cfg.max_masks) + " pixels. Pixel (" +
                                  std::to_string(x) + ", " + std::to_string(y) + ") truncated.");
            continue;
        }

        res.pixels.push_back({x, y, rate_in});
    }

    res.success = true;
    return res;
}

/**
 * @brief Writes a formatted hot pixel list to disk.
 * @param filepath Destination path
 * @param pixels Hot pixel vector
 * @param error_out Optional error message recipient
 * @return true on success
 */
inline bool write_hot_pixels_file(
    const std::string& filepath,
    const std::vector<HotPixel>& pixels,
    std::string* error_out = nullptr)
{
    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
        if (error_out) *error_out = "Cannot open output file: " + filepath;
        return false;
    }

    ofs << "# Predator IMX636 Hot Pixel Hardware Mask File\n"
        << "# Generated by hot_pixel_survey (Phase 33.7a)\n"
        << "# Columns: <x> <y> <rate_ev_s>\n";

    for (const auto& p : pixels) {
        ofs << p.x << " " << p.y << " " << std::fixed << std::setprecision(1) << p.rate << "\n";
    }

    return true;
}

/**
 * @brief Programs pixel masks into sensor hardware via OpenEB HAL facilities.
 * @param device OpenEB device instance
 * @param pixels List of hot pixels to mask
 * @return HardwareMaskStatus describing outcome and facility used
 */
inline HardwareMaskStatus apply_hardware_pixel_mask(
    Metavision::Device& device,
    const std::vector<HotPixel>& pixels)
{
    HardwareMaskStatus status;

    // 1. Primary facility for Sony IMX636 (Gen4.1 architecture): I_DigitalEventMask
    auto* dem = device.get_facility<Metavision::I_DigitalEventMask>();
    if (dem) {
        status.facility_name = "I_DigitalEventMask";
        const auto& masks = dem->get_pixel_masks();
        const size_t num_hw_masks = masks.size();
        const size_t n_to_apply = std::min(pixels.size(), num_hw_masks);

        for (size_t i = 0; i < num_hw_masks; ++i) {
            if (i < n_to_apply) {
                masks[i]->set_mask(pixels[i].x, pixels[i].y, true);
            } else {
                masks[i]->set_mask(0, 0, false);
            }
        }
        status.applied = true;
        status.count = n_to_apply;
        status.message = "Successfully programmed " + std::to_string(n_to_apply) +
                         " hardware pixel masks via I_DigitalEventMask (capacity: " +
                         std::to_string(num_hw_masks) + ")";
        return status;
    }

    // 2. Fallback facility for GenX320 devices: I_RoiPixelMask
    auto* rpm = device.get_facility<Metavision::I_RoiPixelMask>();
    if (rpm) {
        status.facility_name = "I_RoiPixelMask";
        rpm->reset_pixels();
        for (const auto& p : pixels) {
            rpm->set_pixel(p.x, p.y, true);
        }
        rpm->apply_pixels();
        status.applied = true;
        status.count = pixels.size();
        status.message = "Successfully programmed " + std::to_string(pixels.size()) +
                         " hardware pixel masks via I_RoiPixelMask";
        return status;
    }

    // 3. Graceful degradation: no hardware masking facility present
    status.applied = false;
    status.facility_name = "none";
    status.count = 0;
    status.message = "Device lacks hardware pixel mask facilities (neither I_DigitalEventMask nor I_RoiPixelMask present)";
    return status;
}

} // namespace predator
