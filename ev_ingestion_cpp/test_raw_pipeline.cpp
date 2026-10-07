/**
 * @file test_raw_pipeline.cpp
 * @brief Unit tests, parity verification, and microbenchmarks for RawPipeline (Phase 33.4b.c).
 */

#include "raw_pipeline.cuh"
#include "evt21_encoder.hpp"
#include "cuda_flicker_core.cuh"
#include "ego_motion.hpp"

#include <metavision/sdk/base/events/event_cd.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <thread>
#include <cassert>
#include <algorithm>

namespace {

using predator::RawPipeline;
using predator::RawPipelineConfig;
using predator::CudaFlickerCore;
using predator::ContinuousGyroWarper;
using predator::LensParameters;
using predator::Matrix3x3;
using predator::evt21::Evt21Encoder;
using predator::FlickerDetectionResult;

struct CdRecord {
    uint16_t x{0};
    uint16_t y{0};
    int16_t p{0};
    int16_t reserved{0};
    int64_t t{0};
};
static_assert(sizeof(CdRecord) == 16, "fixture record must be 16 bytes");

static int g_failures = 0;

void run_test(const std::string& name, bool passed, const std::string& extra = "") {
    std::cout << "[TEST] " << name << " : " << (passed ? "PASSED" : "FAILED");
    if (!extra.empty()) {
        std::cout << " (" << extra << ")";
    }
    std::cout << "\n";
    if (!passed) {
        g_failures++;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Test 1: Synthetic 200 Hz Drone Rotor Ingestion via Raw EVT2.1 Words
// ---------------------------------------------------------------------------------------------------------------
void test_synthetic_200hz_drone() {
    std::cout << "\n=== Test 1: Synthetic 200 Hz Drone Ingestion through Raw EVT2.1 Pipeline ===\n";

    LensParameters lens;
    lens.focal_length_mm = 12.0;
    ContinuousGyroWarper gyro_warper(lens);
    CudaFlickerCore cuda_core(1280, 720, 32, 18, 4000.0, 512);

    RawPipelineConfig cfg;
    cfg.sensor_width = 1280;
    cfg.sensor_height = 720;
    cfg.ring_slots = 4;
    cfg.max_words_per_slot = 32768;
    cfg.coalesce_words = 16384;
    cfg.enable_ego_warp = false;
    cfg.enable_ui_frame_gen = true;

    RawPipeline pipeline(cfg, cuda_core, gyro_warper);

    // Generate 200 Hz blade pass stream: pixel (640, 360) -> column 16, row 9
    // 100 cycles at 5000 us = 500 ms stream (spans 128 ms FFT window)
    std::vector<CdRecord> events;
    uint64_t t_now = 1000000;
    for (int cycle = 0; cycle < 100; ++cycle) {
        for (int b = 0; b < 5; ++b) {
            CdRecord ev;
            ev.x = 640;
            ev.y = 360;
            ev.p = 1;
            ev.t = static_cast<int64_t>(t_now + b * 50); // 50 us intra-burst
            events.push_back(ev);
        }
        t_now += 5000; // 200 Hz
    }

    // Encode events into EVT2.1 words
    Evt21Encoder encoder(1280, 720);
    std::vector<uint64_t> words;
    encoder.encode(events.data(), events.data() + events.size(), words);

    // Process raw words synchronously through GPU pipeline
    pipeline.process_raw_words_sync(words.data(), words.size());

    // Execute cuFFT spectral analysis
    std::vector<FlickerDetectionResult> raw_detections;
    cuda_core.execute_batched_spectral_analysis(0.0, raw_detections);

    // Check if cell (16, 9) detected 200 Hz blade pass
    bool found_200hz = false;
    float peak_snr = 0.0f;
    float peak_bpf = 0.0f;

    for (const auto& d : raw_detections) {
        if (d.patch_x == 16 && d.patch_y == 9) {
            peak_snr = static_cast<float>(d.peak_snr_db);
            peak_bpf = static_cast<float>(d.fundamental_bpf_hz);
            if (std::abs(d.fundamental_bpf_hz - 200.0) <= 5.0 && d.peak_snr_db >= 12.0) {
                found_200hz = true;
                break;
            }
        }
    }

    std::cout << "  -> Target cell (col=16, row=9): detected=" << (found_200hz ? "YES" : "NO")
              << ", BPF=" << peak_bpf << " Hz, SNR=" << peak_snr << " dB\n";

    run_test("Synthetic 200 Hz Drone Ingestion & cuFFT Detection", found_200hz,
             "BPF=" + std::to_string(peak_bpf) + " Hz, SNR=" + std::to_string(peak_snr) + " dB");
    assert(found_200hz);
}

// ---------------------------------------------------------------------------------------------------------------
// Test 2: UI Frame Synthesis Callback Verification (Phase 33.4b.e)
// ---------------------------------------------------------------------------------------------------------------
void test_ui_frame_synthesis() {
    std::cout << "\n=== Test 2: UI Frame Synthesis Callback Verification ===\n";

    LensParameters lens;
    ContinuousGyroWarper gyro_warper(lens);
    CudaFlickerCore cuda_core(1280, 720, 32, 18, 4000.0, 512);

    RawPipelineConfig cfg;
    cfg.sensor_width = 1280;
    cfg.sensor_height = 720;
    cfg.ui_fps = 1000.0; // Force immediate frame delivery
    cfg.enable_ui_frame_gen = true;

    RawPipeline pipeline(cfg, cuda_core, gyro_warper);

    bool frame_delivered = false;
    int received_w = 0, received_h = 0;
    uint8_t pixel_val = 0;

    pipeline.set_frame_callback([&](const uint8_t* data, int w, int h, uint64_t ts) {
        frame_delivered = true;
        received_w = w;
        received_h = h;
        // Check pixel at (640, 360)
        pixel_val = data[360 * w + 640];
    });

    std::vector<CdRecord> events;
    CdRecord ev{640, 360, 1, 0, 1000000};
    events.push_back(ev);

    Evt21Encoder encoder(1280, 720);
    std::vector<uint64_t> words;
    encoder.encode(events.data(), events.data() + events.size(), words);

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    pipeline.process_raw_words_sync(words.data(), words.size());

    run_test("GPU UI Frame Synthesis Delivery", frame_delivered && received_w == 1280 && received_h == 720,
             "delivered=" + std::to_string(frame_delivered) + ", pixel=" + std::to_string(pixel_val));
    assert(frame_delivered);
}

// ---------------------------------------------------------------------------------------------------------------
// Test 3: Fixture Parity (Full GPU Raw Pipeline vs CPU CD Ingestion)
// ---------------------------------------------------------------------------------------------------------------
void test_fixture_parity(const std::string& evt21raw_path, const std::string& cd_path) {
    std::cout << "\n=== Test 3: Fixture Parity (Raw EVT2.1 GPU Path vs Reference CD CPU Path) ===\n";

    std::ifstream raw_file(evt21raw_path, std::ios::binary);
    std::ifstream cd_file(cd_path, std::ios::binary);

    if (!raw_file.good() || !cd_file.good()) {
        std::cout << "[SKIP] Fixture files not found: " << evt21raw_path << " (skipping live fixture parity)\n";
        return;
    }

    // Read 20,000 raw words from fixture (~40,000 events)
    const size_t test_words = 20000;
    std::vector<uint64_t> raw_words(test_words);
    raw_file.read(reinterpret_cast<char*>(raw_words.data()), test_words * sizeof(uint64_t));
    size_t actual_words = static_cast<size_t>(raw_file.gcount() / sizeof(uint64_t));
    raw_words.resize(actual_words);

    if (actual_words == 0) {
        std::cout << "[SKIP] Empty fixture\n";
        return;
    }

    LensParameters lens;
    ContinuousGyroWarper gyro_warper(lens);

    // 1. Path A: Full GPU Pipeline (Raw words -> GpuEvt21Decoder -> GpuPeriodicitySieve -> CudaFlickerCore)
    CudaFlickerCore core_gpu(1280, 720, 32, 18, 4000.0, 512);
    RawPipelineConfig cfg;
    cfg.sensor_width = 1280;
    cfg.sensor_height = 720;
    cfg.max_words_per_slot = 32768;
    cfg.enable_ego_warp = false;
    cfg.enable_ui_frame_gen = false;
    RawPipeline pipeline(cfg, core_gpu, gyro_warper);

    pipeline.process_raw_words_sync(raw_words.data(), raw_words.size());
    uint64_t gpu_decoded = pipeline.total_decoded_events();

    std::vector<float> gpu_cell_totals;
    core_gpu.get_cell_total_events(gpu_cell_totals);

    // 2. Path B: CPU Reference Path (Decoded CD events -> ingest_event_batch)
    // Read the exact same number of events from cd_path
    std::vector<CdRecord> cd_records(gpu_decoded);
    cd_file.read(reinterpret_cast<char*>(cd_records.data()), gpu_decoded * sizeof(CdRecord));
    size_t actual_cd = static_cast<size_t>(cd_file.gcount() / sizeof(CdRecord));
    cd_records.resize(actual_cd);

    CudaFlickerCore core_cpu(1280, 720, 32, 18, 4000.0, 512);
    std::vector<Metavision::EventCD> cpu_events(actual_cd);
    for (size_t i = 0; i < actual_cd; ++i) {
        cpu_events[i].x = cd_records[i].x;
        cpu_events[i].y = cd_records[i].y;
        cpu_events[i].p = cd_records[i].p;
        cpu_events[i].t = cd_records[i].t;
    }

    uint64_t raw_c = 0, ret_c = 0;
    core_cpu.ingest_event_batch(cpu_events.data(), cpu_events.size(), Matrix3x3::identity(),
                                raw_c, ret_c, nullptr, 0.35f, true);

    std::vector<float> cpu_cell_totals;
    core_cpu.get_cell_total_events(cpu_cell_totals);

    // Compare all 1152 cells
    bool parity = true;
    int mismatch_cells = 0;
    for (size_t i = 0; i < 1152; ++i) {
        if (std::abs(gpu_cell_totals[i] - cpu_cell_totals[i]) > 1.0f) {
            parity = false;
            mismatch_cells++;
        }
    }

    std::cout << "  -> Raw Words: " << actual_words << ", GPU Decoded Events: " << gpu_decoded
              << ", CPU Events: " << actual_cd << "\n";
    std::cout << "  -> Cell Parity: " << (parity ? "PERFECT" : "MISMATCH")
              << " (" << mismatch_cells << "/1152 cells differing)\n";
    if (!parity) {
        for (size_t i = 0; i < 1152; ++i) {
            if (std::abs(gpu_cell_totals[i] - cpu_cell_totals[i]) > 1.0f) {
                std::cout << "    Cell " << i << ": GPU=" << gpu_cell_totals[i] << ", CPU=" << cpu_cell_totals[i] << "\n";
                break;
            }
        }
    }
    std::cout << std::flush;

    run_test("Fixture Ring Buffer Parity (GPU Raw vs CPU CD)", parity,
             "words=" + std::to_string(actual_words) + ", events=" + std::to_string(gpu_decoded));
    assert(parity);

}

// ---------------------------------------------------------------------------------------------------------------
// Test 4: End-to-End GPU Pipeline Microbenchmark (16,384 Words)
// ---------------------------------------------------------------------------------------------------------------
void benchmark_raw_pipeline() {
    std::cout << "\n=== Test 4: End-to-End GPU Pipeline Microbenchmark (16,384 Words) ===\n";

    LensParameters lens;
    ContinuousGyroWarper gyro_warper(lens);
    CudaFlickerCore cuda_core(1280, 720, 32, 18, 4000.0, 512);

    RawPipelineConfig cfg;
    cfg.sensor_width = 1280;
    cfg.sensor_height = 720;
    cfg.max_words_per_slot = 32768;
    cfg.enable_ego_warp = false;
    cfg.enable_ui_frame_gen = true;
    RawPipeline pipeline(cfg, cuda_core, gyro_warper);

    // Synthesize 16,384 words of realistic traffic
    const size_t batch_words = 16384;
    std::vector<CdRecord> events;
    events.reserve(batch_words * 2);

    int64_t t = 10000000;
    for (size_t i = 0; i < batch_words; ++i) {
        CdRecord e;
        e.x = static_cast<uint16_t>((i * 13) % 1280);
        e.y = static_cast<uint16_t>((i * 7) % 720);
        e.p = static_cast<int16_t>(i & 1);
        e.t = (t += 2);
        events.push_back(e);
    }

    Evt21Encoder encoder(1280, 720);
    std::vector<uint64_t> words;
    encoder.encode(events.data(), events.data() + events.size(), words);
    if (words.size() > batch_words) words.resize(batch_words);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        pipeline.process_raw_words_sync(words.data(), words.size());
    }

    const int iterations = 50;
    std::vector<double> timings_us;
    timings_us.reserve(iterations);

    for (int iter = 0; iter < iterations; ++iter) {
        auto t0 = std::chrono::steady_clock::now();
        pipeline.process_raw_words_sync(words.data(), words.size());
        auto t1 = std::chrono::steady_clock::now();
        timings_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    std::sort(timings_us.begin(), timings_us.end());
    double p50 = timings_us[timings_us.size() / 2];
    double p99 = timings_us[static_cast<size_t>(0.99 * (timings_us.size() - 1))];
    double min_us = timings_us.front();
    double max_us = timings_us.back();

    auto stats = pipeline.get_stats();
    double throughput_mev_s = (stats.last_batch_events / (p50 * 1e-6)) / 1e6;

    std::cout << "  Iterations: " << iterations << " batches\n";
    std::cout << "  Batch Size: " << words.size() << " words (" << stats.last_batch_events << " decoded events)\n";
    std::cout << "  End-to-End Latency: min=" << min_us << " us, p50=" << p50 << " us, p99=" << p99 << " us, max=" << max_us << " us\n";
    std::cout << "  Component Breakdown: Decode=" << stats.last_gpu_decode_us << " us, Sieve=" << stats.last_gpu_sieve_us
              << " us, Ingest=" << stats.last_gpu_ingest_us << " us\n";
    run_test("End-to-End GPU Pipeline Target (>= 20 MEv/s, p50 < 600 us)",
             (throughput_mev_s >= 20.0 && p50 < 600.0),
             "p50=" + std::to_string(p50) + " us, throughput=" + std::to_string(throughput_mev_s) + " MEv/s");
    assert(throughput_mev_s >= 20.0 && p50 < 600.0);
}

} // anonymous namespace

int main(int argc, char** argv) {
    std::cout << "========================================================\n";
    std::cout << "  Predator Raw Tap Ingestion Pipeline Verification      \n";
    std::cout << "  Phase 33.4b.c Zero-CPU GPU-Resident Ingestion         \n";
    std::cout << "========================================================\n";

    test_synthetic_200hz_drone();
    test_ui_frame_synthesis();

    std::string raw_path = "/home/orin/ev_deploy/fixtures/darkroom_evt21.evt21raw";
    std::string cd_path = "/home/orin/ev_deploy/fixtures/darkroom_evt21.cd";
    if (argc > 2) {
        raw_path = argv[1];
        cd_path = argv[2];
    }
    test_fixture_parity(raw_path, cd_path);

    benchmark_raw_pipeline();

    if (g_failures == 0) {
        std::cout << "\n========================================================\n";
        std::cout << "  ALL RAW TAP PIPELINE TESTS PASSED SUCCESSFULLY!       \n";
        std::cout << "========================================================\n";
        return 0;
    } else {
        std::cerr << "\n[FAILURE] " << g_failures << " tests failed!\n";
        return 1;
    }
}
