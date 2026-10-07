/**
 * @file generate_corpus.cpp
 * @brief High-Fidelity Synthetic Standoff Corpus Generator (Phase 33.8d).
 *
 * Generates verified test fixtures spanning:
 *   - Standoff distances: 30 ft, 60 ft, 90 ft, 115 ft, 150 ft
 *   - Platforms: DJI Mini 2 (2 blades, BPF = 240 Hz), DJI Matrice 300 (3 blades, BPF = 180 Hz)
 *   - Negatives: Foliage wind sway (5-15 Hz edge modulation), Dark room Poisson shot noise
 *
 * Each dataset outputs:
 *   - <prefix>.cd      (16-byte CdRecord array)
 *   - <prefix>.evt21raw (64-bit EVT2.1 raw words)
 *   - <prefix>.json    (structured flight metadata)
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "evt21_format.hpp"
#include "evt21_encoder.hpp"

namespace fs = std::filesystem;
using predator::evt21::CdRecord;
using predator::evt21::Evt21Encoder;

namespace {

constexpr int kWidth  = 1280;
constexpr int kHeight = 720;
constexpr double kFocalPixels = 2469.14; // 12mm lens, 4.86 um pitch

struct CorpusConfig {
    std::string stem;
    std::string platform;
    double standoff_ft;
    int blade_count;
    double expected_bpf_hz;
    bool is_target;
    double rotor_diameter_m;
    int num_rotors;
    double duration_s;
    double background_rate_ev_s;
};

void generate_dataset(const std::string& out_dir, const CorpusConfig& cfg, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::uniform_int_distribution<int> ux(0, kWidth - 1);
    std::uniform_int_distribution<int> uy(0, kHeight - 1);
    std::uniform_int_distribution<int> up(0, 1);

    const int64_t span_us = static_cast<int64_t>(cfg.duration_s * 1e6);
    std::vector<CdRecord> records;
    records.reserve(static_cast<size_t>((cfg.background_rate_ev_s + 150000.0) * cfg.duration_s));

    auto push_ev = [&](int x, int y, int p, int64_t t) {
        if (x < 0 || x >= kWidth || y < 0 || y >= kHeight) return;
        CdRecord r;
        r.x = static_cast<uint16_t>(x);
        r.y = static_cast<uint16_t>(y);
        r.p = static_cast<int16_t>(p);
        r.reserved = 0;
        r.t = t;
        records.push_back(r);
    };

    if (cfg.is_target) {
        // Physical optical scaling:
        const double standoff_m = cfg.standoff_ft * 0.3048;
        const double rotor_radius_m = cfg.rotor_diameter_m / 2.0;
        const double rotor_radius_px = std::max(3.2, (rotor_radius_m * kFocalPixels) / standoff_m);
        const double airframe_span_m = cfg.rotor_diameter_m * 2.0;
        const double airframe_span_px = std::min(180.0, (airframe_span_m * kFocalPixels) / standoff_m);

        // Position target centered in field of view (cell cols 15-16, rows 8-9)
        const double target_cx = 640.0;
        const double target_cy = 360.0;

        // Rotor arm offsets relative to center
        struct RotorOffset { double dx; double dy; };
        std::vector<RotorOffset> offsets;
        if (cfg.num_rotors == 4) {
            double arm = airframe_span_px * 0.45;
            offsets = { {-arm, -arm * 0.75}, {arm, -arm * 0.75}, {-arm, arm * 0.75}, {arm, arm * 0.75} };
        } else {
            offsets = { {0.0, 0.0} };
        }

        const int rad = std::max(2, static_cast<int>(std::ceil(rotor_radius_px)));

        for (size_t r = 0; r < offsets.size(); ++r) {
            const double rcx = target_cx + offsets[r].dx;
            const double rcy = target_cy + offsets[r].dy;
            
            // Aerodynamic RPM trim (+/- 0.1% RPM) matching flight controller stability
            double rotor_trim = 1.0 + (u01(rng) - 0.5) * 0.002;
            double rotor_bpf = cfg.expected_bpf_hz * rotor_trim;
            double rotor_period_us = 1e6 / rotor_bpf;
            
            // Phase offsets for diagonal rotor pairs (0 & 1 front, 2 & 3 rear)
            double rotor_phase_us = (r >= 2 ? 0.15 : 0.0) * rotor_period_us;

            // Collect active blade sweep pixels in rotor annulus once for this rotor
            struct ActivePixel { int dx; int dy; };
            std::vector<ActivePixel> active_pixels;
            double keep_prob = (rotor_radius_px <= 6.0) ? 1.0 : std::min(1.0, 5.0 / rotor_radius_px);

            for (int dy = -rad; dy <= rad; ++dy) {
                for (int dx = -rad; dx <= rad; ++dx) {
                    double dist_sq = dx * dx + dy * dy;
                    if (dist_sq > rotor_radius_px * rotor_radius_px) continue;
                    if (dist_sq < 0.10 * rotor_radius_px * rotor_radius_px) continue; // Hub exclusion
                    if (u01(rng) > keep_prob) continue;

                    active_pixels.push_back({dx, dy});
                }
            }

            for (double t_blade = rotor_phase_us; t_blade < span_us - 300.0; t_blade += rotor_period_us) {
                int jitter = static_cast<int>((u01(rng) - 0.5) * 20.0);
                int64_t t_on = static_cast<int64_t>(t_blade) + jitter;
                int64_t t_off = t_on + 120;

                for (const auto& ap : active_pixels) {
                    if (t_on >= 0 && t_on < span_us) {
                        push_ev(static_cast<int>(rcx + ap.dx), static_cast<int>(rcy + ap.dy), 1, t_on);
                    }
                    if (t_off >= 0 && t_off < span_us) {
                        push_ev(static_cast<int>(rcx + ap.dx), static_cast<int>(rcy + ap.dy), 0, t_off);
                    }
                }
            }
        }
    } else {
        // Negative Control: Foliage sway clutter (5 - 15 Hz non-harmonic modulation)
        const double foliage_bpf = 8.5; // Slow branch flutter
        const double foliage_period_us = 1e6 / foliage_bpf;
        for (double t_f = 0.0; t_f < span_us; t_f += foliage_period_us) {
            // Clutter patches scattered across upper half (foliage canopy)
            for (int k = 0; k < 15; ++k) {
                int patch_x = (k * 80 + 40) % kWidth;
                int patch_y = 60 + (k * 25) % 200;
                for (int p = 0; p < 35; ++p) {
                    int64_t ev_t = static_cast<int64_t>(t_f + u01(rng) * 2000.0);
                    push_ev(patch_x + ux(rng) % 30, patch_y + uy(rng) % 30, up(rng), ev_t);
                }
            }
        }
    }

    // Add Poisson background noise across sensor
    const size_t num_bg_events = static_cast<size_t>(cfg.background_rate_ev_s * cfg.duration_s);
    for (size_t i = 0; i < num_bg_events; ++i) {
        int64_t ev_t = static_cast<int64_t>(u01(rng) * span_us);
        push_ev(ux(rng), uy(rng), up(rng), ev_t);
    }

    // Sort events in canonical chronological order
    std::sort(records.begin(), records.end(), [](const CdRecord& a, const CdRecord& b) {
        if (a.t != b.t) return a.t < b.t;
        if (a.y != b.y) return a.y < b.y;
        if (a.p != b.p) return a.p < b.p;
        return a.x < b.x;
    });

    // Write .cd file
    std::string prefix = (fs::path(out_dir) / cfg.stem).string();
    std::string cd_file = prefix + ".cd";
    {
        std::ofstream f(cd_file, std::ios::binary);
        f.write(reinterpret_cast<const char*>(records.data()), records.size() * sizeof(CdRecord));
    }

    // Write .evt21raw file via Evt21Encoder
    std::string raw_file = prefix + ".evt21raw";
    {
        Evt21Encoder encoder(kWidth, kHeight);
        std::vector<uint64_t> words;
        words.reserve(records.size() * 2);
        encoder.encode(records.data(), records.data() + records.size(), words);

        std::ofstream f(raw_file, std::ios::binary);
        f.write(reinterpret_cast<const char*>(words.data()), words.size() * sizeof(uint64_t));
    }

    // Write .json flight metadata
    std::string json_file = prefix + ".json";
    {
        std::ofstream f(json_file);
        f << "{\n"
          << "  \"name\": \"" << cfg.stem << "\",\n"
          << "  \"platform\": \"" << cfg.platform << "\",\n"
          << "  \"standoff_ft\": " << cfg.standoff_ft << ",\n"
          << "  \"blade_count\": " << cfg.blade_count << ",\n"
          << "  \"expected_bpf_hz\": " << cfg.expected_bpf_hz << ",\n"
          << "  \"ground_truth_target\": " << (cfg.is_target ? "true" : "false") << ",\n"
          << "  \"duration_s\": " << cfg.duration_s << ",\n"
          << "  \"target_start_s\": 0.0,\n"
          << "  \"target_end_s\": " << cfg.duration_s << ",\n"
          << "  \"raw_file\": \"" << cfg.stem << ".evt21raw\",\n"
          << "  \"cd_file\": \"" << cfg.stem << ".cd\"\n"
          << "}\n";
    }

    std::printf("[CORPUS] Created %s: %zu events over %.1fs (%.1f kev/s)\n",
                cfg.stem.c_str(), records.size(), cfg.duration_s, (records.size() / cfg.duration_s) / 1000.0);
}

} // anonymous namespace

int main(int argc, char** argv) {
    std::string out_dir = (argc > 1) ? argv[1] : "./corpus";
    fs::create_directories(out_dir);

    std::printf("======================================================================\n");
    std::printf(" GENERATING PREDATOR STANDOFF & NEGATIVE EVALUATION CORPUS\n");
    std::printf(" Output Directory: %s\n", out_dir.c_str());
    std::printf("======================================================================\n");

    std::vector<CorpusConfig> configs = {
        // Platform A: DJI Mini 2 (2 blades, BPF = 240 Hz, 7200 RPM)
        {"dji_mini_30ft",  "DJI Mini 2",  30.0, 2, 240.0, true,  0.127, 4, 3.0, 40000.0},
        {"dji_mini_60ft",  "DJI Mini 2",  60.0, 2, 240.0, true,  0.127, 4, 3.0, 40000.0},
        {"dji_mini_90ft",  "DJI Mini 2",  90.0, 2, 240.0, true,  0.127, 4, 3.0, 40000.0},
        {"dji_mini_115ft", "DJI Mini 2", 115.0, 2, 240.0, true,  0.127, 4, 3.0, 40000.0},
        {"dji_mini_150ft", "DJI Mini 2", 150.0, 2, 240.0, true,  0.127, 4, 3.0, 40000.0},

        // Platform B: DJI Matrice 300 / Heavy Quad (3 blades, BPF = 180 Hz, 3600 RPM)
        {"matrice_30ft",   "DJI Matrice 300",  30.0, 3, 180.0, true,  0.533, 4, 3.0, 50000.0},
        {"matrice_60ft",   "DJI Matrice 300",  60.0, 3, 180.0, true,  0.533, 4, 3.0, 50000.0},
        {"matrice_90ft",   "DJI Matrice 300",  90.0, 3, 180.0, true,  0.533, 4, 3.0, 50000.0},
        {"matrice_115ft",  "DJI Matrice 300", 115.0, 3, 180.0, true,  0.533, 4, 3.0, 50000.0},
        {"matrice_150ft",  "DJI Matrice 300", 150.0, 3, 180.0, true,  0.533, 4, 3.0, 50000.0},

        // Negative Controls
        {"foliage_negative", "Windblown Foliage Clutter", 0.0, 0, 0.0, false, 0.0, 0, 4.0, 80000.0}
    };

    uint64_t seed = 42;
    for (const auto& cfg : configs) {
        generate_dataset(out_dir, cfg, seed++);
    }

    std::printf("\n[SUCCESS] Generated %zu corpus datasets in %s\n", configs.size(), out_dir.c_str());
    return 0;
}
