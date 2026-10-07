/**
 * @file test_gpu_sieve.cpp
 * @brief Unit tests, parity verification, and microbenchmarks for GpuPeriodicitySieve.
 *
 * Verifies 100% bit-exact parity against the CPU MicroNeighborhoodPeriodicitySieve across:
 *   1. Single drone rotor blade sweeps (140 Hz, 200 Hz, 250 Hz, 400 Hz)
 *   2. Ego-motion moving contrast edge rejection
 *   3. Foliage sway rejection
 *   4. Poisson random shot noise rejection
 *   5. Dense same-cell batches (20k events)
 *   6. Multi-rotor asynchronous concurrent stream
 *   7. Arbitrary batch splits and persistent state continuity
 *   8. Recorded live fixture parity (if available)
 *   9. 16k batch microbenchmark (GPU latency & MEv/s throughput)
 */

#include "gpu_sieve.cuh"
#include "flicker_dsp.hpp"

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>

namespace {

using predator::CudaRawEvent;
using predator::GpuPeriodicitySieve;
using predator::GpuSieveConfig;
using predator::MicroNeighborhoodPeriodicitySieve;

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

std::vector<CudaRawEvent> upload_and_process(
    GpuPeriodicitySieve& gpu_sieve,
    const std::vector<CudaRawEvent>& host_events,
    uint64_t* out_retained = nullptr)
{
    if (host_events.empty()) return {};

    CudaRawEvent* d_events = nullptr;
    size_t bytes = host_events.size() * sizeof(CudaRawEvent);
    cudaMalloc(&d_events, bytes);
    cudaMemcpy(d_events, host_events.data(), bytes, cudaMemcpyHostToDevice);

    gpu_sieve.process_batch(d_events, host_events.size(), nullptr, out_retained);

    std::vector<CudaRawEvent> result(host_events.size());
    cudaMemcpy(result.data(), d_events, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_events);

    return result;
}

std::vector<uint8_t> run_cpu_sieve(
    MicroNeighborhoodPeriodicitySieve& cpu_sieve,
    const std::vector<CudaRawEvent>& events)
{
    std::vector<uint8_t> hits(events.size());
    for (size_t i = 0; i < events.size(); ++i) {
        hits[i] = cpu_sieve.periodic_hits(events[i].x, events[i].y, events[i].t);
    }
    return hits;
}

bool verify_parity(
    const std::vector<CudaRawEvent>& gpu_events,
    const std::vector<uint8_t>& cpu_hits,
    size_t* out_mismatch_idx = nullptr)
{
    if (gpu_events.size() != cpu_hits.size()) return false;
    for (size_t i = 0; i < gpu_events.size(); ++i) {
        if (static_cast<uint8_t>(gpu_events[i].pad) != cpu_hits[i]) {
            if (out_mismatch_idx) *out_mismatch_idx = i;
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Test 1: Single Drone Rotor Blade Chops (140 Hz BPF / T = 7142 us) with Micro-Jitter
// ---------------------------------------------------------------------------------------------------------------
void test_single_drone_rotor() {
    MicroNeighborhoodPeriodicitySieve cpu_sieve(1280, 720, 70.0, 800.0);
    GpuPeriodicitySieve gpu_sieve;

    int drone_x = 640;
    int drone_y = 360;
    std::vector<CudaRawEvent> events;
    for (int i = 0; i < 50; ++i) {
        uint64_t t = 1000000 + i * 7142;
        int px = drone_x + (i % 2);
        int py = drone_y + ((i / 2) % 2);

        CudaRawEvent ev;
        ev.x = static_cast<uint16_t>(px);
        ev.y = static_cast<uint16_t>(py);
        ev.p = 1;
        ev.pad = 0;
        ev.t = t;
        events.push_back(ev);
    }

    auto cpu_hits = run_cpu_sieve(cpu_sieve, events);
    uint64_t gpu_retained = 0;
    auto gpu_events = upload_and_process(gpu_sieve, events, &gpu_retained);

    size_t mismatch = 0;
    bool parity = verify_parity(gpu_events, cpu_hits, &mismatch);

    int cpu_retained = 0;
    for (auto h : cpu_hits) if (h > 0) cpu_retained++;

    run_test("140 Hz Blade Chops: Parity vs CPU Sieve", parity,
             "CPU retained=" + std::to_string(cpu_retained) + ", GPU retained=" + std::to_string(gpu_retained));
    assert(parity && gpu_retained >= 48);
}

// ---------------------------------------------------------------------------------------------------------------
// Test 2: Multiple Frequencies (200 Hz, 250 Hz, 400 Hz)
// ---------------------------------------------------------------------------------------------------------------
void test_multi_frequencies() {
    double freqs[] = {200.0, 250.0, 400.0};
    for (double f : freqs) {
        MicroNeighborhoodPeriodicitySieve cpu_sieve(1280, 720, 70.0, 800.0);
        GpuPeriodicitySieve gpu_sieve;

        uint64_t period_us = static_cast<uint64_t>(1e6 / f);
        std::vector<CudaRawEvent> events;
        for (int i = 0; i < 60; ++i) {
            CudaRawEvent ev;
            ev.x = 500;
            ev.y = 300;
            ev.p = 1;
            ev.pad = 0;
            ev.t = 2000000 + i * period_us;
            events.push_back(ev);
        }

        auto cpu_hits = run_cpu_sieve(cpu_sieve, events);
        uint64_t gpu_retained = 0;
        auto gpu_events = upload_and_process(gpu_sieve, events, &gpu_retained);

        size_t mismatch = 0;
        bool parity = verify_parity(gpu_events, cpu_hits, &mismatch);
        run_test(std::to_string(static_cast<int>(f)) + " Hz Drone Blade Chops Parity", parity);
        assert(parity);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Test 3: Ego-Motion Moving Contrast Edge Rejection (20 deg/s sweep)
// ---------------------------------------------------------------------------------------------------------------
void test_ego_motion_edge() {
    MicroNeighborhoodPeriodicitySieve cpu_sieve(1280, 720, 70.0, 800.0);
    GpuPeriodicitySieve gpu_sieve;

    std::vector<CudaRawEvent> events;
    for (int i = 0; i < 100; ++i) {
        CudaRawEvent ev;
        ev.x = static_cast<uint16_t>(100 + i * 4); // steps to next tile each time
        ev.y = 200;
        ev.p = 1;
        ev.pad = 0;
        ev.t = 3000000 + i * 1000;
        events.push_back(ev);
    }

    auto cpu_hits = run_cpu_sieve(cpu_sieve, events);
    uint64_t gpu_retained = 0;
    auto gpu_events = upload_and_process(gpu_sieve, events, &gpu_retained);

    size_t mismatch = 0;
    bool parity = verify_parity(gpu_events, cpu_hits, &mismatch);
    run_test("Aperiodic Moving Edge 100% Rejection", parity && gpu_retained == 0,
             "retained=" + std::to_string(gpu_retained));
    assert(parity && gpu_retained == 0);
}

// ---------------------------------------------------------------------------------------------------------------
// Test 4: Foliage Sway Rejection (30 ms period > 14.3 ms cutoff)
// ---------------------------------------------------------------------------------------------------------------
void test_foliage_sway() {
    MicroNeighborhoodPeriodicitySieve cpu_sieve(1280, 720, 70.0, 800.0);
    GpuPeriodicitySieve gpu_sieve;

    std::vector<CudaRawEvent> events;
    for (int i = 0; i < 30; ++i) {
        CudaRawEvent ev;
        ev.x = 800;
        ev.y = 400;
        ev.p = 1;
        ev.pad = 0;
        ev.t = 4000000 + i * 30000; // 30ms period
        events.push_back(ev);
    }

    auto cpu_hits = run_cpu_sieve(cpu_sieve, events);
    uint64_t gpu_retained = 0;
    auto gpu_events = upload_and_process(gpu_sieve, events, &gpu_retained);

    size_t mismatch = 0;
    bool parity = verify_parity(gpu_events, cpu_hits, &mismatch);
    run_test("Foliage Sway (30 ms period) 100% Rejection", parity && gpu_retained == 0,
             "retained=" + std::to_string(gpu_retained));
    assert(parity && gpu_retained == 0);
}

// ---------------------------------------------------------------------------------------------------------------
// Test 5: Poisson Shot Noise Stream (10,000 random events)
// ---------------------------------------------------------------------------------------------------------------
void test_shot_noise() {
    MicroNeighborhoodPeriodicitySieve cpu_sieve(1280, 720, 70.0, 800.0);
    GpuPeriodicitySieve gpu_sieve;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> ux(0, 1279), uy(0, 719);
    std::exponential_distribution<double> gap(1.0); // 1 us avg gap

    std::vector<CudaRawEvent> events(10000);
    uint64_t cur_t = 5000000;
    for (size_t i = 0; i < events.size(); ++i) {
        cur_t += static_cast<uint64_t>(std::max(1.0, gap(rng) * 50.0));
        events[i].x = static_cast<uint16_t>(ux(rng));
        events[i].y = static_cast<uint16_t>(uy(rng));
        events[i].p = static_cast<int16_t>(rng() & 1);
        events[i].pad = 0;
        events[i].t = cur_t;
    }

    auto cpu_hits = run_cpu_sieve(cpu_sieve, events);
    uint64_t gpu_retained = 0;
    auto gpu_events = upload_and_process(gpu_sieve, events, &gpu_retained);

    size_t mismatch = 0;
    bool parity = verify_parity(gpu_events, cpu_hits, &mismatch);

    double pass_rate = (100.0 * gpu_retained) / events.size();
    run_test("Random Poisson Noise Stream Parity", parity && pass_rate < 2.0,
             "retained=" + std::to_string(gpu_retained) + "/" + std::to_string(events.size()) +
             " (" + std::to_string(pass_rate) + "%)");
    assert(parity && pass_rate < 2.0);
}

// ---------------------------------------------------------------------------------------------------------------
// Test 6: Dense Same-Cell Batch (20,000 events)
// ---------------------------------------------------------------------------------------------------------------
void test_dense_batch() {
    MicroNeighborhoodPeriodicitySieve cpu_sieve(1280, 720, 70.0, 800.0);
    GpuPeriodicitySieve gpu_sieve;

    const size_t n = 20000;
    std::vector<CudaRawEvent> events(n);
    for (size_t i = 0; i < n; ++i) {
        events[i].x = static_cast<uint16_t>(600 + (i * 7) % 20);
        events[i].y = static_cast<uint16_t>(400 + (i * 3) % 20);
        events[i].p = static_cast<int16_t>(i & 1);
        events[i].pad = 0;
        events[i].t = 6000000 + i * 5;
    }

    auto cpu_hits = run_cpu_sieve(cpu_sieve, events);
    uint64_t gpu_retained = 0;
    auto gpu_events = upload_and_process(gpu_sieve, events, &gpu_retained);

    size_t mismatch = 0;
    bool parity = verify_parity(gpu_events, cpu_hits, &mismatch);
    run_test("Dense Same-Cell 20k Event Batch Parity", parity);
    assert(parity);
}

// ---------------------------------------------------------------------------------------------------------------
// Test 7: Multi-Rotor Concurrent Stream with Clutter Interleaving
// ---------------------------------------------------------------------------------------------------------------
void test_multi_rotor_interleaved() {
    MicroNeighborhoodPeriodicitySieve cpu_sieve(1280, 720, 70.0, 800.0);
    GpuPeriodicitySieve gpu_sieve;

    struct RotorDef {
        int x, y;
        double bpf;
        uint64_t next_t;
    };
    RotorDef rotors[] = {
        {200, 150, 175.0, 7000000},
        {400, 500, 220.0, 7000000},
        {700, 250, 275.0, 7000000},
        {900, 600, 330.0, 7000000},
    };

    std::vector<CudaRawEvent> events;
    std::mt19937 rng(999);
    std::uniform_int_distribution<int> ux(0, 1279), uy(0, 719);

    uint64_t t_end = 7000000 + 100000; // 100 ms simulation
    uint64_t sim_t = 7000000;

    while (sim_t < t_end) {
        // Find next event among rotors or noise
        int best_rotor = -1;
        uint64_t min_rotor_t = UINT64_MAX;
        for (int r = 0; r < 4; ++r) {
            if (rotors[r].next_t < min_rotor_t) {
                min_rotor_t = rotors[r].next_t;
                best_rotor = r;
            }
        }

        if (best_rotor >= 0 && min_rotor_t <= sim_t + 100) {
            sim_t = min_rotor_t;
            // 4-event blade pass
            for (int k = 0; k < 4; ++k) {
                CudaRawEvent ev;
                ev.x = static_cast<uint16_t>(rotors[best_rotor].x + (k % 2));
                ev.y = static_cast<uint16_t>(rotors[best_rotor].y + ((k / 2) % 2));
                ev.p = 1;
                ev.pad = 0;
                ev.t = sim_t + k * 20;
                events.push_back(ev);
            }
            rotors[best_rotor].next_t += static_cast<uint64_t>(1e6 / rotors[best_rotor].bpf);
        } else {
            sim_t += 50;
            // Interleaved clutter event
            CudaRawEvent ev;
            ev.x = static_cast<uint16_t>(ux(rng));
            ev.y = static_cast<uint16_t>(uy(rng));
            ev.p = static_cast<int16_t>(rng() & 1);
            ev.pad = 0;
            ev.t = sim_t;
            events.push_back(ev);
        }
    }

    auto cpu_hits = run_cpu_sieve(cpu_sieve, events);
    uint64_t gpu_retained = 0;
    auto gpu_events = upload_and_process(gpu_sieve, events, &gpu_retained);

    size_t mismatch = 0;
    bool parity = verify_parity(gpu_events, cpu_hits, &mismatch);
    run_test("Multi-Rotor Interleaved Stream Parity", parity,
             "total_events=" + std::to_string(events.size()) + ", retained=" + std::to_string(gpu_retained));
    assert(parity);
}

// ---------------------------------------------------------------------------------------------------------------
// Test 8: Arbitrary Batch Splits and Persistent State Continuity
// ---------------------------------------------------------------------------------------------------------------
void test_arbitrary_batch_splits() {
    MicroNeighborhoodPeriodicitySieve cpu_sieve(1280, 720, 70.0, 800.0);
    GpuPeriodicitySieve gpu_single;
    GpuPeriodicitySieve gpu_chunked;

    // Create a continuous sequence of 8,000 events
    std::vector<CudaRawEvent> events(8000);
    for (size_t i = 0; i < events.size(); ++i) {
        events[i].x = static_cast<uint16_t>(320 + (i % 4));
        events[i].y = static_cast<uint16_t>(240 + ((i / 4) % 4));
        events[i].p = 1;
        events[i].pad = 0;
        events[i].t = 8000000 + i * 250; // periodic sweeps
    }

    auto cpu_hits = run_cpu_sieve(cpu_sieve, events);
    auto gpu_single_events = upload_and_process(gpu_single, events);

    // Now process with gpu_chunked across arbitrary random batch sizes
    std::mt19937 rng(777);
    std::vector<CudaRawEvent> gpu_chunked_events;
    gpu_chunked_events.reserve(events.size());

    size_t offset = 0;
    while (offset < events.size()) {
        size_t chunk_len = std::min(events.size() - offset, static_cast<size_t>(1 + (rng() % 350)));
        std::vector<CudaRawEvent> chunk(events.begin() + offset, events.begin() + offset + chunk_len);
        auto processed_chunk = upload_and_process(gpu_chunked, chunk);
        gpu_chunked_events.insert(gpu_chunked_events.end(), processed_chunk.begin(), processed_chunk.end());
        offset += chunk_len;
    }

    size_t mismatch_single = 0;
    bool parity_single = verify_parity(gpu_single_events, cpu_hits, &mismatch_single);

    size_t mismatch_chunked = 0;
    bool parity_chunked = verify_parity(gpu_chunked_events, cpu_hits, &mismatch_chunked);

    run_test("Arbitrary Batch Splits Parity (Single Batch == CPU)", parity_single);
    run_test("Arbitrary Batch Splits Parity (Chunked Batches == CPU)", parity_chunked);
    assert(parity_single && parity_chunked);
}

// ---------------------------------------------------------------------------------------------------------------
// Test 9: Recorded Dark-Room Fixture Parity (if available)
// ---------------------------------------------------------------------------------------------------------------
void test_fixture_parity(const std::string& cd_path) {
    std::ifstream file(cd_path, std::ios::binary);
    if (!file) {
        std::cout << "[SKIP] Fixture file not found: " << cd_path << " (skipping live fixture test)\n";
        return;
    }

    struct CdRecord {
        uint16_t x;
        uint16_t y;
        int16_t p;
        int16_t reserved;
        int64_t t;
    };

    const size_t max_test_events = 200000;
    std::vector<CdRecord> records;
    records.resize(max_test_events);
    file.read(reinterpret_cast<char*>(records.data()), max_test_events * sizeof(CdRecord));
    size_t read_count = static_cast<size_t>(file.gcount() / sizeof(CdRecord));
    records.resize(read_count);

    if (read_count == 0) {
        std::cout << "[SKIP] Fixture file empty\n";
        return;
    }

    std::vector<CudaRawEvent> events(read_count);
    for (size_t i = 0; i < read_count; ++i) {
        events[i].x = records[i].x;
        events[i].y = records[i].y;
        events[i].p = records[i].p;
        events[i].pad = 0;
        events[i].t = static_cast<uint64_t>(records[i].t);
    }

    MicroNeighborhoodPeriodicitySieve cpu_sieve(1280, 720, 70.0, 800.0);
    GpuPeriodicitySieve gpu_sieve;

    auto cpu_hits = run_cpu_sieve(cpu_sieve, events);
    uint64_t gpu_retained = 0;
    auto gpu_events = upload_and_process(gpu_sieve, events, &gpu_retained);

    size_t mismatch = 0;
    bool parity = verify_parity(gpu_events, cpu_hits, &mismatch);

    run_test("Dark-Room Recorded Fixture Parity (" + std::to_string(read_count) + " events)", parity,
             "mismatch_idx=" + (parity ? "none" : std::to_string(mismatch)));
    assert(parity);
}

// ---------------------------------------------------------------------------------------------------------------
// Test 10: 16k Batch Microbenchmark
// ---------------------------------------------------------------------------------------------------------------
void benchmark_16k_batches() {
    std::cout << "\n=== Running GPU Periodicity Sieve Microbenchmark (16,384-Event Batches) ===\n";

    GpuPeriodicitySieve gpu_sieve;
    const size_t batch_size = 16384;
    const int iterations = 100;

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> ux(0, 1279), uy(0, 719);

    std::vector<CudaRawEvent> host_batch(batch_size);
    uint64_t t = 10000000;
    for (size_t i = 0; i < batch_size; ++i) {
        host_batch[i].x = static_cast<uint16_t>(ux(rng));
        host_batch[i].y = static_cast<uint16_t>(uy(rng));
        host_batch[i].p = static_cast<int16_t>(i & 1);
        host_batch[i].pad = 0;
        host_batch[i].t = (t += 1);
    }

    CudaRawEvent* d_batch = nullptr;
    cudaMalloc(&d_batch, batch_size * sizeof(CudaRawEvent));
    cudaMemcpy(d_batch, host_batch.data(), batch_size * sizeof(CudaRawEvent), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        gpu_sieve.process_batch(d_batch, batch_size);
    }
    cudaDeviceSynchronize();

    std::vector<double> timings_us;
    timings_us.reserve(iterations);

    for (int iter = 0; iter < iterations; ++iter) {
        cudaEventRecord(start);
        gpu_sieve.process_batch(d_batch, batch_size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        timings_us.push_back(ms * 1000.0);
    }

    std::sort(timings_us.begin(), timings_us.end());
    double p50 = timings_us[timings_us.size() / 2];
    double p99 = timings_us[static_cast<size_t>(0.99 * (timings_us.size() - 1))];
    double min_us = timings_us.front();
    double max_us = timings_us.back();

    double throughput_mev_s = (batch_size / (p50 * 1e-6)) / 1e6;

    std::cout << "  Iterations: " << iterations << " batches\n";
    std::cout << "  Batch Size: " << batch_size << " events\n";
    std::cout << "  GPU Latency: min=" << min_us << " us, p50=" << p50 << " us, p99=" << p99 << " us, max=" << max_us << " us\n";
    std::cout << "  Throughput (at p50): " << throughput_mev_s << " MEv/s (" << (p50 * 1000.0 / batch_size) << " ns/event)\n";

    cudaFree(d_batch);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    run_test("Performance Target (p50 < 150 us at 16k events)", p50 < 150.0,
             "p50=" + std::to_string(p50) + " us, throughput=" + std::to_string(throughput_mev_s) + " MEv/s");
}

} // anonymous namespace

int main(int argc, char** argv) {
    std::cout << "========================================================\n";
    std::cout << "  Predator GPU Periodicity Sieve Parity & Benchmark     \n";
    std::cout << "  Phase 33.4b.d Order-Correct GPU Micro-Neighborhood    \n";
    std::cout << "========================================================\n";

    test_single_drone_rotor();
    test_multi_frequencies();
    test_ego_motion_edge();
    test_foliage_sway();
    test_shot_noise();
    test_dense_batch();
    test_multi_rotor_interleaved();
    test_arbitrary_batch_splits();

    std::string fixture_path = "/home/orin/ev_deploy/fixtures/darkroom_evt21.cd";
    if (argc > 1) {
        fixture_path = argv[1];
    }
    test_fixture_parity(fixture_path);

    benchmark_16k_batches();

    if (g_failures == 0) {
        std::cout << "\n========================================================\n";
        std::cout << "  ALL GPU PERIODICITY SIEVE TESTS PASSED SUCCESSFULLY!  \n";
        std::cout << "========================================================\n";
        return 0;
    } else {
        std::cerr << "\n[FAILURE] " << g_failures << " tests failed!\n";
        return 1;
    }
}
