#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <metavision/sdk/base/events/event_cd.h>

#include "flicker_dsp.hpp"
#include "cuda_flicker_core.cuh"
#include "spectral_combnet_trt.hpp"

void run_test(const std::string& name, bool condition) {
    std::cout << "[TEST] " << std::left << std::setw(55) << name << ": "
              << (condition ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m") << std::endl;
    if (!condition) {
        std::cerr << "Assertion failed in: " << name << std::endl;
        std::exit(1);
    }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "  Predator CUDA & cuFFT Acceleration Verification Suite \n";
    std::cout << "========================================================\n";

    predator::CudaFlickerCore cuda_core(1280, 720, 32, 18, 4000.0, 512);
    run_test("CUDA Core Initialization & Unified Allocation", cuda_core.is_initialized());

    // Test 1: Event Ingestion & SAE Periodicity Sieve on GPU
    // Synthetic drone blade chop: 200 Hz (T = 5000 us) at pixel (640, 360) across 100 streaming batches
    uint64_t total_raw = 0, total_retained = 0;
    predator::Matrix3x3 H_identity = predator::Matrix3x3::identity();
    uint64_t t_now = 1000000;

    for (int cycle = 0; cycle < 100; ++cycle) {
        std::vector<Metavision::EventCD> batch;
        for (int b = 0; b < 5; ++b) {
            Metavision::EventCD ev;
            ev.x = 640;
            ev.y = 360;
            ev.p = 1;
            ev.t = t_now + b * 50; // 50 us intra-burst
            batch.push_back(ev);
        }
        uint64_t raw_count = 0, retained_count = 0;
        cuda_core.ingest_event_batch(batch.data(), batch.size(), H_identity, raw_count, retained_count, nullptr, 0.35f, true);
        total_raw += raw_count;
        total_retained += retained_count;
        t_now += 5000; // 200 Hz blade pass
    }

    std::cout << "  -> Raw Events: " << total_raw << ", Retained (Periodic): " << total_retained
              << " (" << (100.0 * total_retained / total_raw) << "%)\n";
    run_test("GPU SAE Sieve: Retain 200 Hz Harmonic Blade Passes", total_retained >= (total_raw * 0.90));

    // Test 2: Reject Aperiodic Single-Shot Step Edge on GPU
    std::vector<Metavision::EventCD> aperiodic_events;
    for (int i = 0; i < 50; ++i) {
        Metavision::EventCD ev;
        ev.x = 100 + i * 2;
        ev.y = 200;
        ev.p = 1;
        ev.t = t_now + i * 1000; // Different non-repeating pixels
        aperiodic_events.push_back(ev);
    }
    uint64_t raw_count = 0, retained_count = 0;
    cuda_core.ingest_event_batch(aperiodic_events.data(), aperiodic_events.size(), H_identity, raw_count, retained_count, nullptr, 0.35f, true);
    run_test("GPU SAE Sieve: Reject Aperiodic Moving Edge", retained_count == 0);

    // Test 3: Batched 1152-Channel cuFFT & Peak Detection
    std::vector<predator::FlickerDetectionResult> candidates;
    // Warmup call to prime CUDA JIT and cuFFT kernel launch pipeline
    cuda_core.execute_batched_spectral_analysis(70.0, 800.0, 2.0, 8.0, 0.0, candidates);

    int iterations = 20;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it) {
        cuda_core.execute_batched_spectral_analysis(70.0, 800.0, 2.0, 8.0, 0.0, candidates);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_compute_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;

    std::cout << "  -> Steady-State Batched cuFFT Compute Time (1152 channels): " << avg_compute_ms << " ms\n";
    std::cout << "  -> Detected Candidates: " << candidates.size() << "\n";
    if (!candidates.empty()) {
        std::cout << "  -> Candidate 0: " << candidates[0].fundamental_bpf_hz << " Hz (SNR: "
                  << candidates[0].peak_snr_db << " dB, Conf: " << candidates[0].confidence << ")\n";
    }

    run_test("cuFFT Latency Performance (< 2.5 ms for 1152 cells)", avg_compute_ms < 2.5);
    run_test("cuFFT Peak Detection: Accurate 200 Hz Recovery",
             !candidates.empty() && std::abs(candidates[0].fundamental_bpf_hz - 200.0) < 5.0);
    run_test("DDHF Spectral Flatness Metric (gamma < 0.20 for harmonic comb)",
             !candidates.empty() && candidates[0].spectral_flatness < 0.20);

    // Test 4: Extreme Standoff Weak-Signal Detection (DDHF Spectral Flatness & Micro-Sieve Gate)
    // 1 event per blade pass across 140ms (35 passes @ 250 Hz, T = 4000 us) at pixel (320, 180)
    std::cout << "\n[TEST 4] Simulating Standoff Weak-Signal Target (1 event/pass @ 250 Hz across 140ms, 150m-300m equivalent)...\n";
    t_now = 2000000;
    for (int cycle = 0; cycle < 35; ++cycle) {
        std::vector<Metavision::EventCD> batch;
        Metavision::EventCD ev;
        ev.x = 320;
        ev.y = 180;
        ev.p = 1;
        ev.t = t_now;
        batch.push_back(ev);
        uint64_t raw_c = 0, ret_c = 0;
        cuda_core.ingest_event_batch(batch.data(), batch.size(), H_identity, raw_c, ret_c, nullptr, 0.35f, true);
        t_now += 4000; // 250 Hz blade chop
    }

    std::vector<predator::FlickerDetectionResult> standoff_cands;
    cuda_core.execute_batched_spectral_analysis(70.0, 800.0, 1.0, 5.0, 0.0, standoff_cands);
    std::cout << "  -> Standoff candidates detected: " << standoff_cands.size() << "\n";
    bool found_standoff = false;
    for (const auto& c : standoff_cands) {
        std::cout << "     Candidate: " << c.fundamental_bpf_hz << " Hz | Events: " << c.total_events
                  << " | SNR: " << c.peak_snr_db << " dB | Flatness: " << c.spectral_flatness
                  << " | Conf: " << c.confidence << " | Cell: (" << c.patch_x << ", " << c.patch_y << ")\n";
        if (std::abs(c.fundamental_bpf_hz - 250.0) < 12.0 && c.spectral_flatness < 0.30) {
            found_standoff = true;
            std::cout << "  -> Standoff Target Recovered: " << c.fundamental_bpf_hz << " Hz (Events: "
                      << c.total_events << ", Flatness: " << c.spectral_flatness << ", Conf: " << c.confidence << ")\n";
            break;
        }
    }
    run_test("Standoff Weak-Signal Detection via DDHF Flatness & Micro-Sieve", found_standoff);

    // Test 5: Extract 257-Bin Power Spectra for Active Cells
    std::cout << "\n[TEST 5] Extracting Active Cell Spectra for SpectralCombNet...\n";
    std::vector<int> active_cells;
    std::vector<std::vector<float>> active_spectra;
    cuda_core.get_active_cells_with_spectra(active_cells, active_spectra, 6.0f);

    std::cout << "  -> Active Cells Extracted: " << active_cells.size() << "\n";
    bool valid_spectra = !active_cells.empty() && !active_spectra.empty() && (active_spectra[0].size() == 257);
    run_test("GPU Extraction of 257-Bin Normalized Spectra", valid_spectra);

    // Test 6: SpectralCombNet TensorRT FP16 Inference
    std::cout << "\n[TEST 6] Benchmarking SpectralCombNet TensorRT FP16 Engine...\n";
    predator::SpectralCombNetEngine spectral_engine(128);
    std::string engine_path = "/home/orin/ev_deploy/models/spectral_combnet_fp16.engine";
    bool engine_loaded = spectral_engine.load_engine(engine_path);
    run_test("Load SpectralCombNet TensorRT Engine", engine_loaded);

    if (engine_loaded && !active_spectra.empty()) {
        std::vector<const float*> ptrs(active_spectra.size());
        for (size_t i = 0; i < active_spectra.size(); ++i) {
            ptrs[i] = active_spectra[i].data();
        }
        std::vector<predator::SpectralPrediction> preds;
        bool infer_ok = spectral_engine.infer_spectra(ptrs, active_cells, preds, 0.40f);
        run_test("SpectralCombNet Batched Inference Execution", infer_ok);
        std::cout << "  -> Neural Predictions Passing Threshold: " << preds.size() << "\n";
        bool found_target = false;
        for (const auto& p : preds) {
            std::cout << "     Cell #" << p.cell_idx << " | Drone Prob: " << std::fixed << std::setprecision(3)
                      << p.drone_prob << " | Freq: " << p.fund_freq_hz << " Hz | Purity: " << p.harmonic_purity << "\n";
            if (p.drone_prob >= 0.40f && p.fund_freq_hz >= 75.0f && p.fund_freq_hz <= 800.0f) {
                found_target = true;
            }
        }
        run_test("SpectralCombNet Target Recognition & Frequency Extracted", found_target);
    }

    std::cout << "\n========================================================\n";
    std::cout << "  ALL CUDA ACCELERATION UNIT TESTS PASSED SUCCESSFULLY! \n";
    std::cout << "========================================================\n";
    return 0;
}
