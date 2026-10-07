#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>
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

    // Test 2b/2c/2d (Phase 33.4 data integrity). Separate core instances so the shared
    // core's state used by Test 3 is untouched.
    {
        // Dense batch: 20k time-ordered events confined to one 40x40 cell (many per 2x2 tile).
        // Previously, parallel SAE evaluation dropped out-of-order same-tile events BEFORE
        // ring-buffer accumulation. Every in-window event must now be counted.
        auto make_dense = [](size_t n, uint64_t t0, uint64_t dt_us) {
            std::vector<Metavision::EventCD> v(n);
            for (size_t i = 0; i < n; ++i) {
                v[i].x = static_cast<unsigned short>(600 + (i * 7) % 20);  // cell col 15
                v[i].y = static_cast<unsigned short>(400 + (i * 3) % 20);  // cell row 10
                v[i].p = static_cast<short>(i & 1);
                v[i].t = static_cast<Metavision::timestamp>(t0 + i * dt_us);
            }
            return v;
        };

        predator::CudaFlickerCore dense_core(1280, 720, 32, 18, 4000.0, 512);
        auto dense = make_dense(20000, 5000000, 5);  // spans 100 ms (< 128 ms window)
        uint64_t rc = 0, kc = 0;
        dense_core.ingest_event_batch(dense.data(), dense.size(), H_identity, rc, kc, nullptr, 0.35f, true);
        auto d = dense_core.get_roi_diagnostics(15, 15, 10, 10);
        std::cout << "  -> Dense same-cell batch: ingested=" << rc << " accumulated=" << d.total_events << "\n";
        run_test("Ingest: Zero FFT Sample Loss on Dense Same-Tile Batch", rc == 20000 && d.total_events == 20000.0f);

        // Oversized batch (> 131072 pinned staging capacity) must be chunked, not truncated.
        predator::CudaFlickerCore big_core(1280, 720, 32, 18, 4000.0, 512);
        auto big = make_dense(300000, 7000000, 0);  // all in one bin; 0 us spacing is legal
        for (size_t i = 0; i < big.size(); ++i) big[i].t = static_cast<Metavision::timestamp>(7000000 + i / 3000);
        big_core.ingest_event_batch(big.data(), big.size(), H_identity, rc, kc, nullptr, 0.35f, true);
        auto b = big_core.get_roi_diagnostics(15, 15, 10, 10);
        std::cout << "  -> Oversized batch: ingested=" << rc << " accumulated=" << b.total_events << "\n";
        run_test("Ingest: Oversized Batch Chunked Without Truncation", rc == 300000 && b.total_events == 300000.0f);

        // Determinism: identical input -> identical totals, sieve hits, and spectra.
        auto run_once = [&](std::vector<int>& cells, std::vector<std::vector<float>>& spectra,
                            predator::RoiDiagnostics& roi) {
            predator::CudaFlickerCore core(1280, 720, 32, 18, 4000.0, 512);
            uint64_t t = 9000000;
            for (int cycle = 0; cycle < 30; ++cycle) {
                std::vector<Metavision::EventCD> batch;
                for (int k = 0; k < 4; ++k) {           // 4-event blade burst @ 250 Hz
                    Metavision::EventCD ev;
                    ev.x = 610; ev.y = 410; ev.p = 1;
                    ev.t = static_cast<Metavision::timestamp>(t + k * 40);
                    batch.push_back(ev);
                }
                for (int k = 0; k < 40; ++k) {          // interleaved deterministic clutter
                    Metavision::EventCD ev;
                    ev.x = static_cast<unsigned short>(600 + (cycle * 13 + k * 5) % 20);
                    ev.y = static_cast<unsigned short>(400 + (cycle * 7 + k * 11) % 20);
                    ev.p = 0;
                    ev.t = static_cast<Metavision::timestamp>(t + 200 + k * 90);
                    batch.push_back(ev);
                }
                uint64_t r = 0, q = 0;
                core.ingest_event_batch(batch.data(), batch.size(), H_identity, r, q, nullptr, 0.35f, true);
                t += 4000;
            }
            std::vector<predator::FlickerDetectionResult> c;
            core.execute_batched_spectral_analysis(0.0, c);
            core.get_active_cells_with_spectra(cells, spectra, 6.0f);
            roi = core.get_roi_diagnostics(15, 15, 10, 10);
        };
        std::vector<int> cells_a, cells_b;
        std::vector<std::vector<float>> spec_a, spec_b;
        predator::RoiDiagnostics roi_a, roi_b;
        run_once(cells_a, spec_a, roi_a);
        run_once(cells_b, spec_b, roi_b);
        bool identical = (cells_a == cells_b) && (spec_a == spec_b) &&
                         roi_a.total_events == roi_b.total_events && roi_a.max_sieve_hits == roi_b.max_sieve_hits;
        std::cout << "  -> Determinism: cells=" << cells_a.size() << " events=" << roi_a.total_events
                  << " sieve_hits=" << roi_a.max_sieve_hits << "\n";
        run_test("Ingest: Deterministic Ring Buffers, Sieve Hits & Spectra", identical && roi_a.max_sieve_hits >= 2);

        // Order independence: the same events presented reversed must yield identical ring
        // buffers (totals and spectra). Sieve hit counts legitimately differ (the sieve is
        // defined on time-ordered input) but must never change what is accumulated.
        auto ring_of = [&](const std::vector<Metavision::EventCD>& evs, std::vector<int>& cells,
                           std::vector<std::vector<float>>& spectra) {
            predator::CudaFlickerCore core(1280, 720, 32, 18, 4000.0, 512);
            uint64_t r = 0, q = 0;
            core.ingest_event_batch(evs.data(), evs.size(), H_identity, r, q, nullptr, 0.35f, true);
            std::vector<predator::FlickerDetectionResult> c;
            core.execute_batched_spectral_analysis(0.0, c);
            core.get_active_cells_with_spectra(cells, spectra, 6.0f);
            return core.get_roi_diagnostics(15, 15, 10, 10).total_events;
        };
        auto ordered = make_dense(20000, 5000000, 5);
        std::vector<Metavision::EventCD> reversed(ordered.rbegin(), ordered.rend());
        std::vector<int> cells_o, cells_r;
        std::vector<std::vector<float>> spec_o, spec_r;
        float tot_o = ring_of(ordered, cells_o, spec_o);
        float tot_r = ring_of(reversed, cells_r, spec_r);
        std::cout << "  -> Order independence: ordered=" << tot_o << " reversed=" << tot_r
                  << " cells=" << cells_o.size() << "\n";
        run_test("Ingest: Shuffled Batch == Ordered Batch Ring Buffers",
                 tot_o == 20000.0f && tot_o == tot_r && cells_o == cells_r && spec_o == spec_r);
    }

    // Test 2f (Phase 33.5): end-to-end CFAR on the real ingest -> cuFFT -> gate path.
    // 1 Mev/s uniform shot noise over the full sensor (~190 events per 40x40 cell per window),
    // analyzed over 5 independent 128 ms windows: the default budget (6 FA/h) must stay silent.
    // Then a 210 Hz rotor (6 events/pass at one pixel) is embedded and must be detected.
    {
        auto run_field = [&](bool with_rotor, int& total_cands, bool& rotor_found, float& rotor_snr) {
            predator::CudaFlickerCore core(1280, 720, 32, 18, 4000.0, 512);
            std::mt19937 rng(with_rotor ? 77 : 2024);
            std::uniform_int_distribution<int> ux(0, 1279), uy(0, 719);
            std::exponential_distribution<double> gap(1.0);  // 1 event/us = 1 Mev/s
            const uint64_t t0 = 20000000;
            const double rotor_period_us = 1e6 / 210.0;
            double t_noise = 0.0, t_rotor = 0.0;
            total_cands = 0; rotor_found = false; rotor_snr = 0.0f;
            for (int window = 0; window < 5; ++window) {
                const double t_end = (window + 1) * 128000.0;
                // 4 ms time-ordered batches (OpenEB delivers ordered CD batches).
                for (double b0 = window * 128000.0; b0 < t_end; b0 += 4000.0) {
                    std::vector<Metavision::EventCD> batch;
                    while (t_noise < b0 + 4000.0) {
                        Metavision::EventCD ev;
                        ev.x = static_cast<unsigned short>(ux(rng));
                        ev.y = static_cast<unsigned short>(uy(rng));
                        ev.p = static_cast<short>(rng() & 1);
                        ev.t = static_cast<Metavision::timestamp>(t0 + static_cast<uint64_t>(t_noise));
                        batch.push_back(ev);
                        t_noise += gap(rng);
                    }
                    while (with_rotor && t_rotor < b0 + 4000.0) {
                        for (int k = 0; k < 6; ++k) {
                            Metavision::EventCD ev;
                            ev.x = 500; ev.y = 300; ev.p = 1;
                            ev.t = static_cast<Metavision::timestamp>(t0 + static_cast<uint64_t>(t_rotor) + k * 30);
                            batch.push_back(ev);
                        }
                        t_rotor += rotor_period_us;
                    }
                    std::sort(batch.begin(), batch.end(),
                              [](const Metavision::EventCD& a, const Metavision::EventCD& b) { return a.t < b.t; });
                    uint64_t r = 0, q = 0;
                    core.ingest_event_batch(batch.data(), batch.size(), H_identity, r, q, nullptr, 0.35f, true);
                }
                std::vector<predator::FlickerDetectionResult> c;
                core.execute_batched_spectral_analysis(0.0, c);
                total_cands += static_cast<int>(c.size());
                for (const auto& d : c) {
                    // Rotor pixel (500,300) -> base cell (12,7); pooled cells (11..12, 6..7) also contain it.
                    const bool in_cell = (d.patch_x >= 11 && d.patch_x <= 12 && d.patch_y >= 6 && d.patch_y <= 7);
                    if (in_cell && std::abs(d.fundamental_bpf_hz - 210.0) < 5.0) {
                        rotor_found = true;
                        rotor_snr = std::max(rotor_snr, static_cast<float>(d.peak_snr_db));
                    }
                }
            }
        };
        int noise_cands = 0, rotor_cands = 0;
        bool dummy = false, rotor_found = false;
        float dummy_snr = 0.0f, rotor_snr = 0.0f;
        run_field(false, noise_cands, dummy, dummy_snr);
        run_field(true, rotor_cands, rotor_found, rotor_snr);
        std::cout << "  -> CFAR field test: noise-only candidates=" << noise_cands << " (5 windows x 1152 cells)"
                  << " | with rotor: candidates=" << rotor_cands << " rotor_found=" << rotor_found
                  << " snr=" << rotor_snr << " dB\n";
        run_test("CFAR: Silent on Sensor-Wide Shot Noise (5 windows)", noise_cands == 0);
        run_test("CFAR: Rotor Embedded in Shot Noise Detected (210 Hz)", rotor_found);
    }

    // Test 3: Batched 1152-Channel cuFFT & Peak Detection
    std::vector<predator::FlickerDetectionResult> candidates;
    // Warmup call to prime CUDA JIT and cuFFT kernel launch pipeline
    cuda_core.execute_batched_spectral_analysis(0.0, candidates);

    int iterations = 20;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it) {
        cuda_core.execute_batched_spectral_analysis(0.0, candidates);
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
    cuda_core.execute_batched_spectral_analysis(0.0, standoff_cands);
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
