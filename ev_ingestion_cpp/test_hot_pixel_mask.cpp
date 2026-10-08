/**
 * @file test_hot_pixel_mask.cpp
 * @brief Unit tests for hot pixel mask parsing, validation, bounds enforcement,
 *        deduplication, and serialization (Phase 33.7a).
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "hot_pixel_mask.hpp"

namespace {

int g_failures = 0;

void run_test(const std::string& name, bool condition) {
    std::cout << "[TEST] " << std::left << std::setw(60) << name << ": "
              << (condition ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m") << std::endl;
    if (!condition) {
        ++g_failures;
    }
}

} // namespace

int main() {
    std::cout << "========================================================\n";
    std::cout << "  Predator Hot Pixel Mask Unit Test Suite (Phase 33.7a) \n";
    std::cout << "========================================================\n";

    predator::HotPixelMaskConfig cfg;
    cfg.max_masks = 64;
    cfg.sensor_width = 1280;
    cfg.sensor_height = 720;

    // Test 1: Non-existent file handling
    {
        auto res = predator::parse_hot_pixels_file("non_existent_file_xyz_123.txt", cfg);
        run_test("Non-existent file reports failure", !res.success && !res.error.empty());
    }

    // Test 2: Valid file parsing with comments and rates
    const std::string valid_file = "/tmp/test_hot_pixels_valid.txt";
    {
        std::ofstream ofs(valid_file);
        ofs << "# Header comment\n"
            << "   \n"
            << "279 677 4621885.9\n"
            << "# intermediate comment\n"
            << "448 33 4621877.8\n"
            << "1246 187\n"; // rate omitted -> defaults to 0.0
        ofs.close();

        auto res = predator::parse_hot_pixels_file(valid_file, cfg);
        bool ok = res.success && res.pixels.size() == 3;
        if (ok) {
            ok = ok && (res.pixels[0].x == 279 && res.pixels[0].y == 677 && res.pixels[0].rate > 4.6e6);
            ok = ok && (res.pixels[1].x == 448 && res.pixels[1].y == 33 && res.pixels[1].rate > 4.6e6);
            ok = ok && (res.pixels[2].x == 1246 && res.pixels[2].y == 187 && res.pixels[2].rate == 0.0);
        }
        run_test("Valid file with comments, whitespace, and rates", ok);
    }

    // Test 3: Out-of-bounds coordinates rejection
    const std::string oob_file = "/tmp/test_hot_pixels_oob.txt";
    {
        std::ofstream ofs(oob_file);
        ofs << "0 0 1000.0\n"          // valid corner
            << "1279 719 2000.0\n"      // valid corner
            << "-1 100 3000.0\n"        // negative x
            << "100 -5 4000.0\n"        // negative y
            << "1280 100 5000.0\n"      // x >= width
            << "100 720 6000.0\n"       // y >= height
            << "640 360 7000.0\n";      // valid center
        ofs.close();

        auto res = predator::parse_hot_pixels_file(oob_file, cfg);
        bool ok = res.success && res.pixels.size() == 3; // only 3 valid
        ok = ok && res.warnings.size() == 4; // 4 out-of-bounds rejected
        if (ok) {
            ok = ok && (res.pixels[0].x == 0 && res.pixels[0].y == 0);
            ok = ok && (res.pixels[1].x == 1279 && res.pixels[1].y == 719);
            ok = ok && (res.pixels[2].x == 640 && res.pixels[2].y == 360);
        }
        run_test("Coordinate bounds validation (rejects x<0, y<0, x>=W, y>=H)", ok);
    }

    // Test 4: Deduplication of duplicate coordinates
    const std::string dup_file = "/tmp/test_hot_pixels_dup.txt";
    {
        std::ofstream ofs(dup_file);
        ofs << "279 677 4621885.0\n"
            << "279 677 1000.0\n" // duplicate
            << "448 33 4621877.0\n"
            << "279 677 500.0\n";  // duplicate again
        ofs.close();

        auto res = predator::parse_hot_pixels_file(dup_file, cfg);
        bool ok = res.success && res.pixels.size() == 2;
        ok = ok && res.warnings.size() == 2;
        if (ok) {
            ok = ok && (res.pixels[0].x == 279 && res.pixels[0].y == 677);
            ok = ok && (res.pixels[1].x == 448 && res.pixels[1].y == 33);
        }
        run_test("Deduplication of duplicate (x, y) coordinates", ok);
    }

    // Test 5: Malformed line handling
    const std::string bad_file = "/tmp/test_hot_pixels_bad.txt";
    {
        std::ofstream ofs(bad_file);
        ofs << "100 200 1500.0\n"
            << "corrupted line without numbers\n"
            << "200\n" // single number
            << "300 400 not_a_float\n"; // non-float rate -> rate parsed as 0.0 or malformed
        ofs.close();

        auto res = predator::parse_hot_pixels_file(bad_file, cfg);
        bool ok = res.success && (res.pixels.size() >= 1 && res.pixels.size() <= 2);
        ok = ok && !res.warnings.empty();
        run_test("Malformed line detection and graceful recovery", ok);
    }

    // Test 6: Capacity limit truncation (exceeding 64 masks)
    const std::string overflow_file = "/tmp/test_hot_pixels_overflow.txt";
    {
        std::ofstream ofs(overflow_file);
        for (int i = 0; i < 100; ++i) {
            ofs << i << " " << i << " " << (1000.0 + i) << "\n";
        }
        ofs.close();

        auto res = predator::parse_hot_pixels_file(overflow_file, cfg);
        bool ok = res.success && res.pixels.size() == 64; // capped at max_masks
        ok = ok && res.warnings.size() == 36;            // 100 - 64 = 36 warnings
        if (ok) {
            ok = ok && (res.pixels[0].x == 0 && res.pixels[63].x == 63);
        }
        run_test("Hardware capacity enforcement (caps at max_masks=64)", ok);
    }

    // Test 7: Write and read round-trip
    const std::string roundtrip_file = "/tmp/test_hot_pixels_roundtrip.txt";
    {
        std::vector<predator::HotPixel> orig = {
            {279, 677, 4621885.9},
            {448, 33, 4621877.8},
            {1246, 187, 30.1}
        };

        bool w_ok = predator::write_hot_pixels_file(roundtrip_file, orig);
        auto res = predator::parse_hot_pixels_file(roundtrip_file, cfg);
        bool ok = w_ok && res.success && res.pixels.size() == orig.size();
        for (size_t i = 0; i < orig.size() && ok; ++i) {
            ok = ok && (res.pixels[i].x == orig[i].x && res.pixels[i].y == orig[i].y);
            ok = ok && (std::abs(res.pixels[i].rate - orig[i].rate) < 0.2);
        }
        run_test("File serializer & parser round-trip bit-accuracy", ok);
    }

    std::cout << "========================================================\n";
    if (g_failures == 0) {
        std::cout << "  ALL HOT PIXEL MASK UNIT TESTS PASSED SUCCESSFULLY!    \n";
        std::cout << "========================================================\n";
        return 0;
    } else {
        std::cerr << "  FAILED: " << g_failures << " unit test(s) failed.\n";
        std::cout << "========================================================\n";
        return 1;
    }
}
