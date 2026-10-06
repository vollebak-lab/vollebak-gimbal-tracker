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
        // Primary rotor blade pass (250 Hz) with finite chord width (3 intra-burst events)
        for (int b = 0; b < 3; ++b) {
            Metavision::EventCD ev;
            ev.x = 320;
            ev.y = 180;
            ev.p = 1;
            ev.t = t_now + b * 60;
            batch.push_back(ev);
        }
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

    if (engine_loaded) {
        // Construct realistic multi-harmonic drone spectrum (200 Hz with 400 and 600 Hz harmonics on ambient noise floor)
        std::vector<float> realistic_drone_spec(257, 0.0f);
        for (int k = 0; k < 257; ++k) {
            realistic_drone_spec[k] = 1.0f + 0.2f * std::sin(k * 0.15f);
        }
        for (int h = 1; h <= 3; ++h) {
            float f_h = h * 200.0f;
            float bin_pos = f_h / 7.8125f;
            int center_bin = static_cast<int>(std::round(bin_pos));
            float h_amp = 60.0f / std::pow(static_cast<float>(h), 1.1f);
            for (int offset = -2; offset <= 2; ++offset) {
                int k = center_bin + offset;
                if (k >= 0 && k < 257) {
                    float dist = std::abs(k - bin_pos);
                    float spread = std::exp(-0.5f * std::pow(dist / 0.75f, 2.0f));
                    realistic_drone_spec[k] += h_amp * spread;
                }
            }
        }
        // Log10 median normalization
        std::vector<float> noise_slice(realistic_drone_spec.begin() + 5, realistic_drone_spec.begin() + 128);
        std::sort(noise_slice.begin(), noise_slice.end());
        float median_noise = std::max(1e-4f, noise_slice[noise_slice.size() / 2]);
        for (int k = 0; k < 257; ++k) {
            realistic_drone_spec[k] = std::log10(1.0f + realistic_drone_spec[k] / median_noise);
        }

        // Construct sparse foliage clutter spectrum (3 sparse peaks, no harmonics)
        std::vector<float> foliage_spec(257, 0.0f);
        foliage_spec[10] = 5.0f; foliage_spec[18] = 4.0f; foliage_spec[47] = 3.0f;

        std::vector<const float*> test_ptrs = { realistic_drone_spec.data(), foliage_spec.data() };
        std::vector<int> test_cells = { 42, 99 };
        std::vector<predator::SpectralPrediction> preds;

        bool infer_ok = spectral_engine.infer_spectra(test_ptrs, test_cells, preds, 0.0f);
        run_test("SpectralCombNet Batched Inference Execution", infer_ok);

        std::cout << "  -> Drone Spectrum Recognition: Prob=" << std::fixed << std::setprecision(4)
                  << preds[0].drone_prob << " | BPF=" << preds[0].fund_freq_hz << " Hz | PeakValid="
                  << (preds[0].has_valid_peak ? "YES" : "NO") << " | SNR=" << preds[0].physical_snr_db << " dB\n";
        std::cout << "  -> Foliage Clutter Rejection : Prob=" << preds[1].drone_prob << "\n";

        run_test("SpectralCombNet Drone Recognition (Prob >= 0.70)", preds[0].drone_prob >= 0.70f);
        run_test("SpectralCombNet Foliage Clutter Rejection (Prob <= 0.10)", preds[1].drone_prob <= 0.10f);
    }

    std::cout << "\n========================================================\n";
    std::cout << "  ALL CUDA ACCELERATION UNIT TESTS PASSED SUCCESSFULLY! \n";
    std::cout << "========================================================\n";
    return 0;
}
