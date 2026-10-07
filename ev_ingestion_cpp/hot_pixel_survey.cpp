/**
 * @file hot_pixel_survey.cpp
 * @brief Phase 33.7a Hot Pixel Survey & Hardware Mask Verification Utility.
 *
 * Discovers anomalous hot pixels on the IMX636 event sensor, exports them to hot_pixels.txt,
 * programs the IMX636 hardware registers (I_DigitalEventMask, 64-mask capacity), and runs
 * a verification pass to confirm zero emission from masked pixels and evaluate overall rate reduction.
 *
 * Usage:
 *   hot_pixel_survey [--seconds S] [--output PATH] [--min-rate HZ] [--median-factor K]
 *                    [--default-biases] [--diff-on N] [--diff-off N] [--refr N] [--fo N]
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <metavision/hal/decoders/evt21/evt21_decoder.h>
#include <metavision/hal/device/device.h>
#include <metavision/hal/device/device_discovery.h>
#include <metavision/hal/facilities/i_event_decoder.h>
#include <metavision/hal/facilities/i_events_stream.h>
#include <metavision/hal/facilities/i_hw_identification.h>
#include <metavision/hal/facilities/i_ll_biases.h>
#include <metavision/hal/utils/device_config.h>
#include <metavision/sdk/base/events/event_cd.h>

#include "hot_pixel_mask.hpp"

namespace {

constexpr uint32_t kWidth = 1280;
constexpr uint32_t kHeight = 720;

struct SurveyOptions {
    double seconds = 5.0;
    std::string output_path = "hot_pixels.txt";
    double min_rate = 1000.0;        // Absolute floor (1 kev/s)
    double median_factor = 1000.0;   // Factor K x median active pixel rate
    bool default_biases = false;
    int diff_on = 6;
    int diff_off = 6;
    int refr = 20;
    int fo = -8;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --seconds <double>      Capture duration per pass in seconds (default: 5.0)\n"
              << "  --output <path>         Output file path (default: hot_pixels.txt)\n"
              << "  --min-rate <double>     Absolute rate threshold in ev/s (default: 1000.0)\n"
              << "  --median-factor <double> Threshold multiplier K over median rate (default: 1000.0)\n"
              << "  --default-biases        Keep factory default biases (do not apply service tuning)\n"
              << "  --diff-on <int>         bias_diff_on (default: 6)\n"
              << "  --diff-off <int>        bias_diff_off (default: 6)\n"
              << "  --refr <int>            bias_refr (default: 20)\n"
              << "  --fo <int>              bias_fo (default: -8)\n"
              << "  --help                  Show this help\n";
}

bool parse_args(int argc, char** argv, SurveyOptions& opt) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        } else if (arg == "--seconds" && i + 1 < argc) {
            opt.seconds = std::atof(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            opt.output_path = argv[++i];
        } else if (arg == "--min-rate" && i + 1 < argc) {
            opt.min_rate = std::atof(argv[++i]);
        } else if (arg == "--median-factor" && i + 1 < argc) {
            opt.median_factor = std::atof(argv[++i]);
        } else if (arg == "--default-biases") {
            opt.default_biases = true;
        } else if (arg == "--diff-on" && i + 1 < argc) {
            opt.diff_on = std::atoi(argv[++i]);
        } else if (arg == "--diff-off" && i + 1 < argc) {
            opt.diff_off = std::atoi(argv[++i]);
        } else if (arg == "--refr" && i + 1 < argc) {
            opt.refr = std::atoi(argv[++i]);
        } else if (arg == "--fo" && i + 1 < argc) {
            opt.fo = std::atoi(argv[++i]);
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            print_usage(argv[0]);
            return false;
        }
    }
    return true;
}

// Runs a stream capture pass, accumulating per-pixel event counts
uint64_t run_capture_pass(
    Metavision::I_EventsStream& stream,
    double duration_s,
    std::vector<uint64_t>& pixel_counts,
    double& actual_duration_s)
{
    std::fill(pixel_counts.begin(), pixel_counts.end(), 0ULL);

    // Decoder sinks (EVT21 requires all 4 sinks to prevent segfaults on OTHERS/monitoring events)
    auto cd_sink = std::make_shared<Metavision::I_EventDecoder<Metavision::EventCD>>();
    auto trig_sink = std::make_shared<Metavision::I_EventDecoder<Metavision::EventExtTrigger>>();
    auto erc_sink = std::make_shared<Metavision::I_EventDecoder<Metavision::EventERCCounter>>();
    auto monitor_sink = std::make_shared<Metavision::I_EventDecoder<Metavision::EventMonitoring>>();

    Metavision::EVT21LegacyDecoder decoder(false, cd_sink, trig_sink, erc_sink, monitor_sink);

    uint64_t total_events = 0;
    cd_sink->add_event_buffer_callback([&](const Metavision::EventCD* b, const Metavision::EventCD* e) {
        for (; b != e; ++b) {
            if (b->x < kWidth && b->y < kHeight) {
                ++pixel_counts[b->y * kWidth + b->x];
            }
            ++total_events;
        }
    });

    const auto t_start = std::chrono::steady_clock::now();
    const auto t_end = t_start + std::chrono::duration<double>(duration_s);

    stream.start();
    while (std::chrono::steady_clock::now() < t_end) {
        if (stream.wait_next_buffer() < 0) {
            break;
        }
        auto buf = stream.get_latest_raw_data();
        if (!buf || buf.size() == 0) continue;
        decoder.decode(buf.data(), buf.data() + buf.size());
    }
    stream.stop();

    const auto t_finish = std::chrono::steady_clock::now();
    actual_duration_s = std::chrono::duration<double>(t_finish - t_start).count();
    return total_events;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    SurveyOptions opt;
    if (!parse_args(argc, argv, opt)) {
        return 2;
    }

    std::cout << "========================================================\n";
    std::cout << "  Predator IMX636 Hot Pixel Survey & Hardware Mask Tool \n";
    std::cout << "========================================================\n";

    std::vector<predator::HotPixel> hot_pixels;
    double rate1 = 0.0;
    double duration1 = 0.0;

    // =========================================================================
    // SESSION 1: Unmasked Baseline Capture
    // =========================================================================
    try {
        std::cout << "\n>>> SESSION 1: Opening camera for baseline unmasked capture...\n";
        Metavision::DeviceConfig cfg;
        cfg.set_format("EVT21");
        std::unique_ptr<Metavision::Device> device = Metavision::DeviceDiscovery::open("", cfg);
        if (!device) {
            std::cerr << "[ERROR] Could not open camera device. Ensure predator-camera.service is stopped.\n";
            return 3;
        }

        auto* hwid = device->get_facility<Metavision::I_HW_Identification>();
        auto* stream = device->get_facility<Metavision::I_EventsStream>();
        auto* biases = device->get_facility<Metavision::I_LL_Biases>();
        auto* dem = device->get_facility<Metavision::I_DigitalEventMask>();

        if (!hwid || !stream) {
            std::cerr << "[ERROR] Device missing I_HW_Identification or I_EventsStream.\n";
            return 3;
        }

        std::cout << "[INFO] Connected to sensor: " << hwid->get_serial()
                  << " (format: " << hwid->get_current_data_encoding_format() << ")\n";

        if (dem) {
            std::cout << "[INFO] Hardware mask facility detected: I_DigitalEventMask with "
                      << dem->get_pixel_masks().size() << " mask registers.\n";
            // Clear all mask registers for unmasked baseline
            for (auto& m : dem->get_pixel_masks()) {
                m->set_mask(0, 0, false);
            }
        } else {
            std::cerr << "[WARN] I_DigitalEventMask facility NOT detected on device!\n";
        }

        // Configure biases
        if (opt.default_biases) {
            std::cout << "[INFO] Operating with FACTORY DEFAULT biases (unmodified).\n";
        } else if (biases) {
            biases->set("bias_diff_on", opt.diff_on);
            biases->set("bias_diff_off", opt.diff_off);
            biases->set("bias_refr", opt.refr);
            biases->set("bias_fo", opt.fo);
            std::cout << "[INFO] Applied service biases: diff_on=" << opt.diff_on
                      << " diff_off=" << opt.diff_off << " refr=" << opt.refr
                      << " fo=" << opt.fo << "\n";
        }

        std::cout << "[INFO] Capturing baseline unmasked stream (" << opt.seconds << "s)...\n";
        std::vector<uint64_t> counts_pass1(kWidth * kHeight, 0ULL);
        uint64_t total1 = run_capture_pass(*stream, opt.seconds, counts_pass1, duration1);

        rate1 = duration1 > 0 ? total1 / duration1 : 0.0;
        std::cout << "[INFO] Baseline capture complete: " << total1 << " events in "
                  << std::fixed << std::setprecision(2) << duration1 << "s ("
                  << std::setprecision(1) << rate1 << " ev/s)\n";

        // Calculate distribution and median active rate
        std::vector<double> active_rates;
        active_rates.reserve(kWidth * kHeight);
        for (uint64_t c : counts_pass1) {
            if (c > 0) {
                active_rates.push_back(c / duration1);
            }
        }

        std::sort(active_rates.begin(), active_rates.end());
        double median_rate = active_rates.empty() ? 0.0 : active_rates[active_rates.size() / 2];
        double threshold = std::max(opt.min_rate, median_rate * opt.median_factor);

        std::cout << "[INFO] Active pixels: " << active_rates.size() << " / " << (kWidth * kHeight)
                  << " (" << std::setprecision(2) << (100.0 * active_rates.size() / (kWidth * kHeight)) << "%)\n"
                  << "[INFO] Median active pixel rate: " << std::setprecision(2) << median_rate << " ev/s\n"
                  << "[INFO] Hot pixel detection threshold: " << std::setprecision(1) << threshold << " ev/s "
                  << "(max of floor " << opt.min_rate << " ev/s and " << opt.median_factor << "x median)\n";

        // Identify hot pixels exceeding threshold
        for (uint16_t y = 0; y < kHeight; ++y) {
            for (uint16_t x = 0; x < kWidth; ++x) {
                uint64_t c = counts_pass1[y * kWidth + x];
                double r = c / duration1;
                if (r > threshold) {
                    hot_pixels.push_back({x, y, r});
                }
            }
        }

        std::sort(hot_pixels.begin(), hot_pixels.end(), [](const predator::HotPixel& a, const predator::HotPixel& b) {
            return a.rate > b.rate;
        });

        std::cout << "\n[RESULT] Identified " << hot_pixels.size() << " hot pixel(s) exceeding threshold:\n";
        std::cout << "  Rank |   X   |   Y   |  Firing Rate  | Share of Total Stream\n";
        std::cout << "  -----+-------+-------+---------------+----------------------\n";
        for (size_t i = 0; i < hot_pixels.size(); ++i) {
            double share = rate1 > 0 ? (100.0 * hot_pixels[i].rate / rate1) : 0.0;
            std::cout << "  " << std::setw(4) << (i + 1) << " | "
                      << std::setw(5) << hot_pixels[i].x << " | "
                      << std::setw(5) << hot_pixels[i].y << " | "
                      << std::setw(11) << std::fixed << std::setprecision(1) << hot_pixels[i].rate << " ev/s | "
                      << std::setw(8) << std::setprecision(3) << share << "%\n";
        }

        predator::HotPixelMaskConfig mask_cfg;
        if (hot_pixels.size() > mask_cfg.max_masks) {
            std::cout << "[WARN] Truncating hot pixels from " << hot_pixels.size()
                      << " down to hardware capacity (" << mask_cfg.max_masks << ").\n";
            hot_pixels.resize(mask_cfg.max_masks);
        }

        // Write to output file
        std::string err;
        if (predator::write_hot_pixels_file(opt.output_path, hot_pixels, &err)) {
            std::cout << "[INFO] Successfully exported hot pixels to " << opt.output_path << "\n";
        } else {
            std::cerr << "[ERROR] Could not write " << opt.output_path << ": " << err << "\n";
        }

        // Release device cleanly before session 2
        device.reset();

    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Session 1 Exception: " << e.what() << "\n";
        return 3;
    }

    if (hot_pixels.empty()) {
        std::cout << "\n[INFO] No hot pixels exceeded threshold. Sensor array is clean under these biases.\n";
        return 0;
    }

    // Give USB subsystem a brief moment to settle
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // =========================================================================
    // SESSION 2: Hardware Masked Verification Capture
    // =========================================================================
    try {
        std::cout << "\n>>> SESSION 2: Re-opening camera, applying hardware masks and verifying...\n";
        Metavision::DeviceConfig cfg;
        cfg.set_format("EVT21");
        std::unique_ptr<Metavision::Device> device = Metavision::DeviceDiscovery::open("", cfg);
        if (!device) {
            std::cerr << "[ERROR] Could not re-open camera device for Session 2.\n";
            return 3;
        }

        auto* stream = device->get_facility<Metavision::I_EventsStream>();
        auto* biases = device->get_facility<Metavision::I_LL_Biases>();
        if (!stream) {
            std::cerr << "[ERROR] Device missing I_EventsStream.\n";
            return 3;
        }

        // Configure biases
        if (opt.default_biases) {
            std::cout << "[INFO] Operating with FACTORY DEFAULT biases (unmodified).\n";
        } else if (biases) {
            biases->set("bias_diff_on", opt.diff_on);
            biases->set("bias_diff_off", opt.diff_off);
            biases->set("bias_refr", opt.refr);
            biases->set("bias_fo", opt.fo);
        }

        // Program hardware masks
        auto mask_status = predator::apply_hardware_pixel_mask(*device, hot_pixels);
        std::cout << "[INFO] " << mask_status.message << "\n";

        if (!mask_status.applied) {
            std::cerr << "[ERROR] Failed to apply hardware masks. Aborting verification.\n";
            return 1;
        }

        std::cout << "[INFO] Capturing masked verification stream (" << opt.seconds << "s)...\n";
        std::vector<uint64_t> counts_pass2(kWidth * kHeight, 0ULL);
        double duration2 = 0.0;
        uint64_t total2 = run_capture_pass(*stream, opt.seconds, counts_pass2, duration2);

        double rate2 = duration2 > 0 ? total2 / duration2 : 0.0;
        std::cout << "[INFO] Verification capture complete: " << total2 << " events in "
                  << std::fixed << std::setprecision(2) << duration2 << "s ("
                  << std::setprecision(1) << rate2 << " ev/s)\n";

        // Check each masked pixel's emissions
        std::cout << "\n[VERIFICATION] Emission check for masked pixels:\n";
        std::cout << "  Rank |   X   |   Y   | Unmasked Rate | Masked Rate | Suppression\n";
        std::cout << "  -----+-------+-------+---------------+-------------+------------\n";
        bool all_zero = true;
        for (size_t i = 0; i < hot_pixels.size(); ++i) {
            uint64_t c2 = counts_pass2[hot_pixels[i].y * kWidth + hot_pixels[i].x];
            double r2 = c2 / duration2;
            if (c2 != 0) all_zero = false;
            double supp = hot_pixels[i].rate > 0 ? 100.0 * (1.0 - r2 / hot_pixels[i].rate) : 100.0;
            std::cout << "  " << std::setw(4) << (i + 1) << " | "
                      << std::setw(5) << hot_pixels[i].x << " | "
                      << std::setw(5) << hot_pixels[i].y << " | "
                      << std::setw(11) << std::fixed << std::setprecision(1) << hot_pixels[i].rate << " ev/s | "
                      << std::setw(9) << std::setprecision(1) << r2 << " ev/s | "
                      << std::setw(8) << std::setprecision(2) << supp << "%\n";
        }

        double rate_reduction = rate1 > 0 ? 100.0 * (1.0 - rate2 / rate1) : 0.0;
        std::cout << "\n[SUMMARY]\n"
                  << "  Unmasked total stream rate : " << std::setprecision(1) << rate1 << " ev/s\n"
                  << "  Masked total stream rate   : " << std::setprecision(1) << rate2 << " ev/s\n"
                  << "  Array-wide rate reduction  : " << std::setprecision(2) << rate_reduction << "%\n";

        if (all_zero) {
            std::cout << "  [PASS] All masked hot pixels emitted EXACTLY ZERO events.\n";
        } else {
            std::cout << "  [FAIL] One or more masked hot pixels continued to emit events.\n";
        }

        device.reset();
        return all_zero ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Session 2 Exception: " << e.what() << "\n";
        return 3;
    }
}

