#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include <complex>
#include <stdexcept>
#include "flicker_dsp.hpp"
#include "spectral_gate.hpp"

// Unit test 1: Drone Blade Passage Frequency (4000 Hz sampling, 140 Hz BPF / 4200 RPM, 512 samples)
void test_pure_harmonic_signal() {
    std::cout << "[TEST 1] Testing Pure Harmonic Propeller Flicker Extraction (140 Hz BPF / 4200 RPM @ 4000 Hz, 512 samples)... ";
    
    double sample_rate = 4000.0; // 4000 Hz => 250us bins
    size_t window_size = 512;    // 128ms window
    predator::PropellerFlickerAnalyzer analyzer(sample_rate, window_size, 80.0, 1200.0, 15.0);

    double target_bpf = 140.0;
    std::vector<double> signal(window_size);
    for (size_t i = 0; i < window_size; ++i) {
        double t = static_cast<double>(i) / sample_rate;
        // Fundamental (140 Hz) + 2nd harmonic (280 Hz) + 3rd harmonic (420 Hz)
        signal[i] = 10.0 * std::sin(2.0 * M_PI * target_bpf * t) +
                     6.0 * std::sin(2.0 * M_PI * 2.0 * target_bpf * t) +
                     3.0 * std::sin(2.0 * M_PI * 3.0 * target_bpf * t) +
                    20.0; // DC offset
    }

    auto res = analyzer.analyze_time_series(signal, 2);

    assert(res.is_drone_detected == true);
    assert(std::abs(res.fundamental_bpf_hz - target_bpf) < 3.0); // within 3.0 Hz
    assert(std::abs(res.estimated_rpm - 4200.0) < 90.0);         // within 90 RPM
    assert(res.confidence > 0.60);
    assert(res.peak_snr_db > 10.0);

    std::cout << "PASSED! (Detected: " << res.fundamental_bpf_hz << " Hz | RPM: " << res.estimated_rpm << " | Confidence: " << res.confidence << ")\n";
}

// Unit test 2: High RPM Drone Propeller (400 Hz BPF / 12000 RPM with Harmonics)
void test_high_rpm_harmonic_comb() {
    std::cout << "[TEST 2] Testing High-RPM Propeller Extraction (400 Hz BPF / 12,000 RPM with Harmonics)... ";
    
    double sample_rate = 4000.0;
    size_t window_size = 512;
    predator::PropellerFlickerAnalyzer analyzer(sample_rate, window_size, 80.0, 1200.0, 15.0);

    double target_bpf = 400.0;
    std::vector<double> signal(window_size);
    for (size_t i = 0; i < window_size; ++i) {
        double t = static_cast<double>(i) / sample_rate;
        // Fundamental (400 Hz) + 2nd harmonic (800 Hz) + 3rd harmonic (1200 Hz)
        signal[i] = 12.0 * std::sin(2.0 * M_PI * target_bpf * t) +
                     7.0 * std::sin(2.0 * M_PI * 2.0 * target_bpf * t) +
                     4.0 * std::sin(2.0 * M_PI * 3.0 * target_bpf * t) +
                    25.0;
    }

    auto res = analyzer.analyze_time_series(signal, 2);

    assert(res.is_drone_detected == true);
    assert(std::abs(res.fundamental_bpf_hz - target_bpf) < 5.0);
    assert(std::abs(res.estimated_rpm - 12000.0) < 150.0);
    assert(res.confidence > 0.65);

    std::cout << "PASSED! (Detected: " << res.fundamental_bpf_hz << " Hz | RPM: " << res.estimated_rpm << " | Confidence: " << res.confidence << ")\n";
}

// Unit test 3: Low-Light Night-Time Photon Starved Rotor (Attenuated Harmonics)
void test_low_light_sparse_rotor() {
    std::cout << "[TEST 3] Testing Low-Light / Night-Time Sparse Rotor Extraction (Harmonics rolled off by analog front-end)... ";
    
    double sample_rate = 4000.0;
    size_t window_size = 512;
    predator::PropellerFlickerAnalyzer analyzer(sample_rate, window_size, 80.0, 1200.0, 15.0);

    std::mt19937 rng(1337);
    std::normal_distribution<double> noise_dist(0.0, 1.2);

    double target_bpf = 180.0; // 5400 RPM
    std::vector<double> signal(window_size);
    for (size_t i = 0; i < window_size; ++i) {
        double t = static_cast<double>(i) / sample_rate;
        // Strong fundamental (180 Hz) but heavily attenuated harmonics due to pixel RC cutoff + dark noise
        signal[i] = 3.5 * std::sin(2.0 * M_PI * target_bpf * t) +
                    0.4 * std::sin(2.0 * M_PI * 2.0 * target_bpf * t) +
                    noise_dist(rng) + 5.0; // sparse count
    }

    auto res = analyzer.analyze_time_series(signal, 2);

    assert(res.is_drone_detected == true);
    assert(std::abs(res.fundamental_bpf_hz - target_bpf) < 4.0);
    assert(res.confidence >= 0.45);

    std::cout << "PASSED! (Night-time sparse fundamental locked: " << res.fundamental_bpf_hz << " Hz | SNR: " << res.peak_snr_db << " dB)\n";
}

// Unit test 4: Low-Frequency Ego-Motion Rejection
void test_ego_motion_rejection() {
    std::cout << "[TEST 4] Testing Ego-Motion & Random Clutter Rejection (5-15 Hz walking sway + white noise)... ";
    
    double sample_rate = 4000.0;
    size_t window_size = 512;
    predator::PropellerFlickerAnalyzer analyzer(sample_rate, window_size, 80.0, 1200.0, 15.0);

    std::mt19937 rng(42);
    std::normal_distribution<double> noise_dist(0.0, 3.0);

    std::vector<double> signal(window_size);
    for (size_t i = 0; i < window_size; ++i) {
        double t = static_cast<double>(i) / sample_rate;
        // Low frequency human gait motion (2 Hz, 6 Hz) + random noise
        signal[i] = 20.0 * std::sin(2.0 * M_PI * 2.5 * t) +
                    15.0 * std::sin(2.0 * M_PI * 6.0 * t) +
                    noise_dist(rng) + 50.0;
    }

    auto res = analyzer.analyze_time_series(signal, 2);

    // Should NOT trigger drone detection
    assert(res.is_drone_detected == false);
    assert(res.confidence < 0.35);

    std::cout << "PASSED! (Properly rejected clutter, Detection: FALSE | Confidence: " << res.confidence << ")\n";
}

// Unit test 5: Ambient Powerline Light Flicker Rejection (50 Hz / 60 Hz)
void test_powerline_flicker_rejection() {
    std::cout << "[TEST 5] Testing AC Powerline Light Flicker Rejection (50-60 Hz room lighting)... ";
    
    double sample_rate = 4000.0;
    size_t window_size = 512;
    predator::PropellerFlickerAnalyzer analyzer(sample_rate, window_size, 80.0, 1200.0, 15.0);

    std::vector<double> signal(window_size);
    for (size_t i = 0; i < window_size; ++i) {
        double t = static_cast<double>(i) / sample_rate;
        // 60 Hz ambient lighting sine wave
        signal[i] = 25.0 * std::sin(2.0 * M_PI * 60.0 * t) + 30.0;
    }

    auto res = analyzer.analyze_time_series(signal, 2);

    // 60 Hz is below the 80 Hz cutoff -> must be rejected
    assert(res.is_drone_detected == false);

    std::cout << "PASSED! (60 Hz room light rejected, Detection: FALSE)\n";
}

// Unit test 6: Activity Density Gate (Dark Background Noise)
void test_low_activity_density_gate() {
    std::cout << "[TEST 6] Testing Low-Activity Density Gate (Sparse dark noise)... ";
    
    double sample_rate = 4000.0;
    size_t window_size = 512;
    predator::PropellerFlickerAnalyzer analyzer(sample_rate, window_size, 80.0, 1200.0, 8.0);

    // Only 5 total events over 128ms
    std::vector<double> signal(window_size, 0.0);
    signal[10] = 2.0;
    signal[40] = 1.0;
    signal[100] = 2.0;

    auto res = analyzer.analyze_time_series(signal, 2);

    // Activity < 8 events -> immediately rejected
    assert(res.is_drone_detected == false);

    std::cout << "PASSED! (Low event count rejected before FFT, Detection: FALSE)\n";
}

// Unit test 7: Spatial 2x2 Pooling Grid Verification
void test_spatial_2x2_pooling() {
    std::cout << "[TEST 7] Testing Spatial 2x2 Cell Pooling & Ingestion... ";
    
    predator::SpatialPatchGrid grid(32, 18, 4000.0, 512);
    
    // Ingest events into cell (2, 2) and (3, 3)
    uint64_t base_ts = 1000000;
    for (int i = 0; i < 50; ++i) {
        grid.ingest_event(85, 85, base_ts + i * 200); // lands in col 2, row 2
        grid.ingest_event(125, 125, base_ts + i * 200); // lands in col 3, row 3
    }
    grid.advance_temporal_bin();

    assert(grid.is_pooled_patch_active(2, 2, 50.0) == true);
    auto pooled = grid.get_pooled_patch_history(2, 2);
    double total_pooled = std::accumulate(pooled.begin(), pooled.end(), 0.0);
    assert(total_pooled >= 100.0); // Both cells pooled together!

    double cx, cy;
    grid.get_pooled_patch_center(2, 2, cx, cy);
    assert(std::abs(cx - 3.0 * 40.0) < 1e-4);
    assert(std::abs(cy - 3.0 * 40.0) < 1e-4);

    std::cout << "PASSED! (Pooled 2x2 patch correctly aggregated events and computed center)\n";
}

// Unit test 8: Global Common-Mode & M-of-N Track State Machine
void test_global_common_mode_and_tracking() {
    std::cout << "[TEST 8] Testing Global Common-Mode Spatial Rejection & M-of-N Track Lifecycle... ";
    
    predator::SpatialFlickerClusterer::reset_tracker();

    std::vector<predator::FlickerDetectionResult> raw_detections;

    // Candidates as emitted by the Phase 33.5 GPU detector: every one passed the CFAR gate, so
    // SNR >= ~13 dB and confidence ~0.7; patch indices are consistent with the centroid (40 px grid).
    auto make_candidate = [](double f_hz, int cx, int cy) {
        predator::FlickerDetectionResult d;
        d.is_drone_detected = true;
        d.fundamental_bpf_hz = f_hz;
        d.centroid_px_x = cx;
        d.centroid_px_y = cy;
        d.patch_x = cx / 40;
        d.patch_y = cy / 40;
        d.peak_snr_db = 14.0;
        d.confidence = 0.70;
        d.spectral_flatness = 0.30;
        return d;
    };

    // Simulate 20 patches across the entire screen all reporting ~120 Hz (e.g. 120Hz LED ambient light spanning 600px)
    for (int i = 0; i < 20; ++i) {
        raw_detections.push_back(make_candidate(120.0 + (i % 2) * 0.5,
                                                100 + (i % 5) * 200,    // Spans 100 to 900 px
                                                100 + (i / 5) * 150));  // Spans 100 to 550 px
    }

    // Add 2 overlapping pooled patches from the same localized drone rotor at 260 Hz (20 px apart)
    raw_detections.push_back(make_candidate(260.0, 400, 300));
    raw_detections.push_back(make_candidate(261.0, 420, 310));

    // Frame 1: Tentative (0 confirmed)
    auto f1 = predator::SpatialFlickerClusterer::filter_and_cluster(raw_detections, 6);
    assert(f1.empty());

    // Frame 2: Tentative (0 confirmed)
    auto f2 = predator::SpatialFlickerClusterer::filter_and_cluster(raw_detections, 6);
    assert(f2.empty());

    // Frame 3: Confirmed! (3-hit confirmation validated, common-mode rejected, overlapping merged)
    auto f3 = predator::SpatialFlickerClusterer::filter_and_cluster(raw_detections, 6);
    assert(f3.size() == 1);
    assert(std::abs(f3[0].fundamental_bpf_hz - 260.0) < 2.0);

    // Frame 4: Test track coasting across a 1-frame sparse dropout (empty input)
    auto coasted = predator::SpatialFlickerClusterer::filter_and_cluster({}, 6);
    assert(coasted.size() == 1); // Confirmed track successfully coasted!
    assert(std::abs(coasted[0].fundamental_bpf_hz - 260.0) < 2.0);

    std::cout << "PASSED! (Spatial dispersion rejected ambient light, 3-hit confirmation validated, and coasted across dropout)\n";
}

// Unit test 9: Lens Bearing Geometry (12mm f/2.0 M12 Lens)
void test_lens_bearing_geometry() {
    std::cout << "[TEST 9] Testing 12mm f/2.0 M12 Lens Bearing Geometry... ";
    
    predator::LensParameters lens;
    lens.focal_length_mm = 12.0;
    lens.pixel_pitch_um = 4.86;
    lens.sensor_width = 1280;
    lens.sensor_height = 720;

    double az, el;
    // Center pixel (640, 360) -> Optical Axis (0, 0 deg)
    lens.pixel_to_angles(640.0, 360.0, az, el);
    assert(std::abs(az) < 1e-4);
    assert(std::abs(el) < 1e-4);

    // Top-Right corner (1280, 0) -> HFOV/2 = ~14.53 deg, VFOV/2 = ~8.29 deg
    lens.pixel_to_angles(1280.0, 0.0, az, el);
    assert(az > 13.5 && az < 15.5); // Expect ~14.53 deg
    assert(el > 7.5 && el < 9.0);   // Expect ~8.29 deg

    std::cout << "PASSED! (Center: [0, 0] deg | Corner: [" << az << ", " << el << "] deg)\n";
}

// Unit test 10: Drone Target Beneath 120 Hz Building Floodlight (Multi-Candidate Peak Extraction)
void test_floodlight_drone_coexistence() {
    std::cout << "[TEST 10] Testing Drone Detection Directly Beneath 120 Hz AC Building Floodlight... ";
    
    predator::SpatialFlickerClusterer::reset_tracker();

    double sample_rate = 4000.0;
    size_t window_size = 512;
    predator::PropellerFlickerAnalyzer analyzer(sample_rate, window_size, 80.0, 1200.0, 15.0);

    // 1. Drone Patch Signal: Strong 120 Hz Floodlight ripple + 175 Hz Drone Propeller (5250 RPM)
    std::vector<double> drone_patch_signal(window_size);
    for (size_t i = 0; i < window_size; ++i) {
        double t = static_cast<double>(i) / sample_rate;
        drone_patch_signal[i] = 20.0 * std::sin(2.0 * M_PI * 120.0 * t) + // 120 Hz floodlight carrier
                                10.0 * std::sin(2.0 * M_PI * 175.0 * t) + // 175 Hz drone propeller
                                30.0;
    }

    // 2. Extract multi-candidates from drone patch
    auto drone_cands = analyzer.analyze_time_series_candidates(drone_patch_signal, 2, 2);
    assert(drone_cands.size() == 2); // Both 120 Hz floodlight AND 175 Hz drone extracted!

    // 3. Simulate 8 background patches illuminated by the same 120 Hz floodlight (spread across 800px)
    std::vector<predator::FlickerDetectionResult> raw_all;
    for (int p = 0; p < 8; ++p) {
        predator::FlickerDetectionResult bg_det;
        bg_det.is_drone_detected = true;
        bg_det.fundamental_bpf_hz = 120.0;
        bg_det.centroid_px_x = p * 120; // 0 to 840 px
        bg_det.centroid_px_y = 100;
        raw_all.push_back(bg_det);
    }

    // Add drone patch candidates (at pixel 600, 350)
    for (auto& c : drone_cands) {
        c.centroid_px_x = 600;
        c.centroid_px_y = 350;
        raw_all.push_back(c);
    }

    // Pass 3 frames through clusterer to confirm (M=3)
    predator::SpatialFlickerClusterer::filter_and_cluster(raw_all, 4);
    predator::SpatialFlickerClusterer::filter_and_cluster(raw_all, 4);
    auto confirmed = predator::SpatialFlickerClusterer::filter_and_cluster(raw_all, 4);

    // Verified: The 120 Hz floodlight was suppressed across all patches, and the 175 Hz drone was confirmed!
    assert(confirmed.size() == 1);
    assert(std::abs(confirmed[0].fundamental_bpf_hz - 175.0) < 3.0);
    assert(confirmed[0].centroid_px_x == 600);

    std::cout << "PASSED! (120 Hz floodlight rejected across scene, 175 Hz drone confirmed beneath light)\n";
}

// Unit test 11: Shaded Quadcopter 4-Rotor Multi-Scale Airframe Fusion (12 hits -> 1 unified airframe lock)
void test_shaded_multirotor_fusion() {
    std::cout << "[TEST 11] Testing Shaded Quadcopter Multi-Rotor Airframe Fusion (12 candidate hits -> 1 unified target)... ";
    
    predator::SpatialFlickerClusterer::reset_tracker();

    // Simulate 4 rotors + multi-scale pooled cells generating 12 hits all at ~183 Hz (5500 RPM Mavic Air 2)
    // Located at bottom-center of FOV (X=640, Y=580)
    std::vector<predator::FlickerDetectionResult> quad_hits;
    int base_x = 640;
    int base_y = 580;

    for (int rotor = 0; rotor < 4; ++rotor) {
        int rx = base_x + ((rotor % 2 == 0) ? -25 : 25);
        int ry = base_y + ((rotor < 2) ? -25 : 25);
        
        // Each rotor generates 1 single-cell hit + 2 pooled-cell hits
        for (int k = 0; k < 3; ++k) {
            predator::FlickerDetectionResult h;
            h.is_drone_detected = true;
            h.fundamental_bpf_hz = 183.0 + (rotor * 0.4);
            h.centroid_px_x = rx + (k * 4 - 4);
            h.centroid_px_y = ry + (k * 4 - 4);
            h.confidence = 0.60;
            h.peak_snr_db = 11.5;
            quad_hits.push_back(h);
        }
    }

    assert(quad_hits.size() == 12);

    // Pass 3 frames for M=3 confirmation
    auto f1 = predator::SpatialFlickerClusterer::filter_and_cluster(quad_hits, 6);
    assert(f1.empty());

    auto f2 = predator::SpatialFlickerClusterer::filter_and_cluster(quad_hits, 6);
    assert(f2.empty());

    // Frame 3: Confirmed!
    auto f3 = predator::SpatialFlickerClusterer::filter_and_cluster(quad_hits, 6);
    // Must NOT be suppressed by common-mode filter (spatial dispersion <= 140px)
    // Must fuse all 12 hits into 1 unified airframe target
    assert(f3.size() == 1);
    assert(std::abs(f3[0].fundamental_bpf_hz - 183.6) < 2.0);
    assert(std::abs(f3[0].centroid_px_x - base_x) < 15);
    assert(std::abs(f3[0].centroid_px_y - base_y) < 15);
    // Multi-rotor fusion boost elevates confidence to >= 0.90!
    assert(f3[0].confidence >= 0.90);

    std::cout << "PASSED! (12 multi-scale rotor hits fused into 1 target | BPF: " << f3[0].fundamental_bpf_hz 
              << " Hz | Centroid: [" << f3[0].centroid_px_x << ", " << f3[0].centroid_px_y 
              << "] | Unified Confidence: " << f3[0].confidence << ")\n";
}

// Unit test 12: Micro-Neighborhood Recurrent Periodicity Sieve (HelixTrack / FrequencyCam Core)
void test_micro_neighborhood_periodicity_sieve() {
    std::cout << "[TEST 12] Testing Micro-Neighborhood Periodicity Sieve (Rotor vs Ego-Motion Edge vs Wind Trees)... ";
    
    predator::MicroNeighborhoodPeriodicitySieve sieve(1280, 720, 70.0, 800.0);

    // 1. Test Drone Rotor Blade Chops (140 Hz BPF => T = 7142 us)
    int drone_x = 640;
    int drone_y = 360;
    int drone_passed = 0;
    int drone_total = 50; // 50 blade passages = ~350ms
    for (int i = 0; i < drone_total; ++i) {
        uint64_t t = 1000000 + i * 7142;
        // Blade tip micro-jitter (+- 1 pixel)
        int px = drone_x + (i % 2);
        int py = drone_y + ((i + 1) % 2);
        if (sieve.is_periodic_event(px, py, t)) {
            drone_passed++;
        }
    }
    // After 1 initial priming event, 100% of periodic blade sweeps must pass
    assert(drone_passed >= 48);

    // 2. Test Single-Shot Ego-Motion Moving Contrast Edge (Sweeping across sensor at 20 deg/s)
    sieve.reset();
    int edge_passed = 0;
    int edge_total = 100;
    for (int i = 0; i < edge_total; ++i) {
        // Edge moves horizontally: pixel changes every 1ms (1000 us)
        int ex = 100 + i * 4; // moves to next tile
        int ey = 200;
        uint64_t t = 2000000 + i * 1000;
        // Each tile sees only 1 edge entry
        if (sieve.is_periodic_event(ex, ey, t)) {
            edge_passed++;
        }
    }
    // Moving edge non-repeating steps must be 100% rejected!
    assert(edge_passed == 0);

    // 3. Test Windblown Foliage Sway (3 Hz sway => T = 333,333 us)
    sieve.reset();
    int tree_passed = 0;
    int tree_total = 30;
    for (int i = 0; i < tree_total; ++i) {
        int tx = 800;
        int ty = 400;
        uint64_t t = 3000000 + i * 333333; // 3 Hz leaf flutter
        if (sieve.is_periodic_event(tx, ty, t)) {
            tree_passed++;
        }
    }
    // Slow foliage sway (> 14.3ms period) must be 100% rejected!
    assert(tree_passed == 0);

    // 4. Test Sparse Random Thermal Shot Noise
    sieve.reset();
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> rand_x(0, 1279);
    std::uniform_int_distribution<int> rand_y(0, 719);
    std::uniform_int_distribution<uint64_t> rand_dt(50, 50000);
    int noise_passed = 0;
    int noise_total = 1000;
    uint64_t cur_t = 4000000;
    for (int i = 0; i < noise_total; ++i) {
        cur_t += rand_dt(rng);
        if (sieve.is_periodic_event(rand_x(rng), rand_y(rng), cur_t)) {
            noise_passed++;
        }
    }
    // Random shot noise should have < 2% accidental pass rate
    assert(noise_passed < 20);

    std::cout << "PASSED! (Drone Pass: " << drone_passed << "/" << drone_total 
              << " | Ego-Motion Edge Rej: " << (edge_total - edge_passed) << "/" << edge_total
              << " | Tree Sway Rej: " << (tree_total - tree_passed) << "/" << tree_total
              << " | Noise Rej: " << (noise_total - noise_passed) << "/" << noise_total << ")\n";
}

// ---------------------------------------------------------------------------------------------
// Phase 33.5: CFAR gate statistics. Spectra are produced exactly like the runtime: 512 bins of
// 250 us event counts -> symmetric Hann (N-1 = 511) -> rFFT -> |X|^2.
// ---------------------------------------------------------------------------------------------
namespace cfar_test {

/// In-place iterative radix-2 complex FFT (n must be a power of two).
static void fft_inplace(std::vector<std::complex<double>>& a) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = -2.0 * M_PI / static_cast<double>(len);
        const std::complex<double> wl(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (size_t k = 0; k < len / 2; ++k) {
                const std::complex<double> u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wl;
            }
        }
    }
}

/// Power spectrum (257 bins) of a 512-sample count series, runtime-identical windowing.
static std::vector<float> power_spectrum(const std::vector<double>& counts) {
    std::vector<std::complex<double>> a(512);
    for (int i = 0; i < 512; ++i) {
        const double w = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / 511.0));
        a[i] = std::complex<double>(counts[i] * w, 0.0);
    }
    fft_inplace(a);
    std::vector<float> p(predator::kSpectrumBins);
    for (int k = 0; k < predator::kSpectrumBins; ++k) p[k] = static_cast<float>(std::norm(a[k]));
    return p;
}

/// Poisson background with optional rate modulation lambda(t) = bg + sum_h amp_h * (1 + cos(2 pi h f t)).
static std::vector<double> poisson_counts(std::mt19937& rng, double bg_per_bin, double f_hz,
                                          const std::vector<double>& harmonic_amps) {
    std::vector<double> c(512);
    for (int i = 0; i < 512; ++i) {
        const double t = i * 250e-6;
        double lam = bg_per_bin;
        for (size_t h = 0; h < harmonic_amps.size(); ++h) {
            lam += harmonic_amps[h] * (1.0 + std::cos(2.0 * M_PI * (h + 1) * f_hz * t));
        }
        std::poisson_distribution<int> pd(lam);
        c[i] = pd(rng);
    }
    return c;
}

}  // namespace cfar_test

void test_cfar_gate_statistics() {
    std::cout << "[TEST 13] CFAR Gate: False-Alarm Bound & Detection (Phase 33.5)... ";
    using namespace predator;

    // (a) Default derivation: 6 FA/h over 1152 cells x bins 9..128 at 4 kHz / 512. The OS-CFAR
    // threshold must exceed the known-noise-mean threshold ln(N/Pfa) (estimator loss is real) and
    // stay within a few dB of it.
    const SpectralGateConfig def = derive_spectral_gate(SpectralGateConfig{}, 4000.0, 512, 1152);
    assert(def.min_bin == 9 && def.max_bin == 128);
    const double known_noise_eta = std::log(1152.0 * 120.0 / (6.0 * 0.128 / 3600.0));
    assert(def.cfar_threshold > known_noise_eta);
    assert(def.cfar_threshold_db < 10.0 * std::log10(known_noise_eta) + 4.0);

    // Invalid configurations are rejected, not silently clamped.
    bool threw = false;
    try { SpectralGateConfig bad; bad.false_alarms_per_hour = 0.0f; derive_spectral_gate(bad, 4000.0, 512, 1152); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);

    // (b) Empirical FA vs analytic bound. Budget chosen so the per-cell bound is 5% (measurable):
    // per-cell P(pass) <= N_bins * exp(-eta) = 0.05 with min_sharpness = 1 (pure CFAR statistic).
    std::mt19937 rng(1234);
    const int n_trials = 4000;
    SpectralGateConfig loose;
    loose.min_sharpness = 1.0f;
    loose.false_alarms_per_hour = static_cast<float>(0.05 * 3600.0 / 0.128);  // per-window Pfa = 0.05
    loose = derive_spectral_gate(loose, 4000.0, 512, 1);
    int loose_pass = 0, default_pass = 0, legacy_5p5db = 0;
    for (int i = 0; i < n_trials; ++i) {
        const auto p = cfar_test::power_spectrum(cfar_test::poisson_counts(rng, 0.5, 0.0, {}));
        const auto gl = evaluate_spectral_gate(p.data(), 256.0f, 0, 0.0f, loose);
        if (gl.verdict == GateVerdict::Pass) ++loose_pass;
        if (gl.snr_db >= 5.5f) ++legacy_5p5db;  // Phase 32 effective threshold (D3)
        if (evaluate_spectral_gate(p.data(), 256.0f, 0, 0.0f, def).verdict == GateVerdict::Pass) ++default_pass;
    }
    const double loose_rate = static_cast<double>(loose_pass) / n_trials;
    const double sigma = std::sqrt(0.05 * 0.95 / n_trials);
    assert(loose_rate <= 0.05 + 3.0 * sigma);   // union bound holds on real Hann/Poisson spectra
    assert(default_pass == 0);                  // default budget: silent on pure noise
    assert(legacy_5p5db > n_trials / 2);        // documents why Phase 32 flooded with candidates

    // Expected periodogram SNR for lambda(t) = bg + A (1 + cos 2 pi f t), Hann (sum w = 255.5,
    // sum w^2 = 191.6): peak ~ (A/2 * 255.5)^2 * scallop (0.81 worst case at 0.4-0.6 bin offset),
    // noise mean ~ (bg + A) * 191.6.

    // (c) Detection: 200 Hz, A = 1.5 on bg 0.5 -> ~29700 / ~383 = ~77 (~19 dB) vs ~14 dB threshold.
    std::mt19937 rng2(99);
    int detected = 0;
    double max_err_hz = 0.0;
    for (int i = 0; i < 50; ++i) {
        const auto p = cfar_test::power_spectrum(cfar_test::poisson_counts(rng2, 0.5, 200.0, {1.5}));
        const auto g = evaluate_spectral_gate(p.data(), 1000.0f, 0, 0.0f, def);
        if (g.verdict == GateVerdict::Pass) {
            ++detected;
            max_err_hz = std::max(max_err_hz, std::abs(g.fundamental_hz - 200.0));
        }
    }
    assert(detected >= 48 && max_err_hz < 4.0);

    // Single-window limit (printed, not asserted): A = 0.25 gives ~7 dB, below any threshold that
    // meets the false-alarm budget. Such targets need multi-window integration (track-before-detect).
    std::mt19937 rng_weak(5);
    int weak_detected = 0;
    for (int i = 0; i < 50; ++i) {
        const auto p = cfar_test::power_spectrum(cfar_test::poisson_counts(rng_weak, 0.5, 200.0, {0.25}));
        if (evaluate_spectral_gate(p.data(), 400.0f, 0, 0.0f, def).verdict == GateVerdict::Pass) ++weak_detected;
    }

    // (d) 2nd harmonic dominant (common for 2-blade props at off-axis views). 150 Hz (bin 19.2,
    // scallop ~0.95) A1 = 1.1; 300 Hz (bin 38.4, scallop ~0.81) A2 = 1.6. Harmonic power is ~1.8x
    // the fundamental (strongest line = detection statistic, ~17 dB); the fundamental line itself
    // is ~14 dB (>= 10 dB floor) and ~0.55x the peak (>= 0.04), so it must be reported as 150 Hz.
    std::mt19937 rng3(7);
    int harm_ok = 0;
    std::vector<SpectralGateResult> harm_results;
    for (int i = 0; i < 50; ++i) {
        const auto p = cfar_test::power_spectrum(cfar_test::poisson_counts(rng3, 0.5, 150.0, {1.1, 1.6}));
        const auto g = evaluate_spectral_gate(p.data(), 1000.0f, 0, 0.0f, def);
        harm_results.push_back(g);
        if (g.verdict == GateVerdict::Pass && std::abs(g.fundamental_hz - 150.0) < 4.0) ++harm_ok;
    }
    if (harm_ok < 45) {
        std::cerr << "\n  harmonic-dominant diagnostics (" << harm_ok << "/50 ok):\n";
        for (size_t i = 0; i < 12 && i < harm_results.size(); ++i) {
            const auto& g = harm_results[i];
            std::cerr << "    " << gate_verdict_name(g.verdict) << " peak_bin=" << g.peak_bin
                      << " fund_bin=" << g.fund_bin << " f0=" << g.fundamental_hz << " Hz snr=" << g.snr_db
                      << " dB sharp=" << g.sharpness << "\n";
        }
    }
    assert(harm_ok >= 45);

    std::cout << "PASSED! (eta=" << def.cfar_threshold_db << " dB | noise FA: loose " << loose_rate
              << " <= 0.05 bound, default " << default_pass << "/" << n_trials
              << ", legacy 5.5 dB would pass " << legacy_5p5db << "/" << n_trials
              << " | 200 Hz det " << detected << "/50 (7 dB case " << weak_detected
              << "/50) | harmonic-dominant " << harm_ok << "/50)\n";
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "  Predator — Frequency-Domain DSP Unit Verification     \n";
    std::cout << "========================================================\n";

    test_pure_harmonic_signal();
    test_high_rpm_harmonic_comb();
    test_low_light_sparse_rotor();
    test_ego_motion_rejection();
    test_powerline_flicker_rejection();
    test_low_activity_density_gate();
    test_spatial_2x2_pooling();
    test_global_common_mode_and_tracking();
    test_lens_bearing_geometry();
    test_floodlight_drone_coexistence();
    test_shaded_multirotor_fusion();
    test_micro_neighborhood_periodicity_sieve();
    test_cfar_gate_statistics();

    std::cout << "========================================================\n";
    std::cout << "  ALL 13 MATHEMATICAL DSP UNIT TESTS PASSED SUCCESSFULLY!\n";
    std::cout << "========================================================\n";
    return 0;
}
