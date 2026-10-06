#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <string>

#include "ego_motion.hpp"
#include "flicker_dsp.hpp"

// Simple Synthetic Event structure for testing
struct SyntheticEvent {
    int x;
    int y;
    uint64_t t;
    short p;
};

void run_test(const std::string& name, bool condition) {
    std::cout << "[TEST] " << std::left << std::setw(60) << name << " : "
              << (condition ? "\033[1;32mPASSED\033[0m" : "\033[1;31mFAILED\033[0m") << "\n";
    if (!condition) {
        std::cerr << "Assertion failed for test: " << name << "\n";
        exit(1);
    }
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  Predator — Ego-Motion Compensation & Gyro Warping Test Suite       \n";
    std::cout << "======================================================================\n\n";

    predator::LensParameters lens;
    lens.focal_length_mm = 12.0;
    lens.pixel_pitch_um = 4.86;
    lens.sensor_width = 1280;
    lens.sensor_height = 720;

    predator::ContinuousGyroWarper warper(lens);

    // --------------------------------------------------------------------------------
    // TEST 1: Matrix Inversion & Intrinsic Consistency
    // --------------------------------------------------------------------------------
    {
        predator::Matrix3x3 I = predator::Matrix3x3::identity();
        predator::Matrix3x3 invI = I.inverse();
        bool pass = true;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                if (r == c && std::abs(invI.at(r, c) - 1.0) > 1e-9) pass = false;
                if (r != c && std::abs(invI.at(r, c)) > 1e-9) pass = false;
            }
        }
        run_test("Test 1: Intrinsic Matrix & Inverse Consistency", pass);
    }

    // --------------------------------------------------------------------------------
    // TEST 2: Identity Rotation (Zero Angular Velocity)
    // --------------------------------------------------------------------------------
    {
        warper.ingest_imu_raw(0, 0.0, 0.0, 0.0);
        warper.ingest_imu_raw(100000, 0.0, 0.0, 0.0); // 100ms later

        double warped_x = 0.0, warped_y = 0.0;
        bool valid = warper.unwarp_event(640, 360, 50000, 0, warped_x, warped_y);

        bool pass = valid && (std::abs(warped_x - 640.0) < 1e-6) && (std::abs(warped_y - 360.0) < 1e-6);
        run_test("Test 2: Identity Rotation (Zero Gyro Motion)", pass);
    }

    // --------------------------------------------------------------------------------
    // TEST 3: Pure Yaw Stabilization (30 deg/s Horizontal Panning)
    // --------------------------------------------------------------------------------
    {
        predator::ContinuousGyroWarper yaw_warper(lens);
        double yaw_rate_deg = 30.0;
        double yaw_rate_rad = yaw_rate_deg * (M_PI / 180.0); // rad/s (wy)

        // Inject 500 Hz IMU samples over 100ms
        for (uint64_t t = 0; t <= 100000; t += 2000) {
            yaw_warper.ingest_imu_raw(t, 0.0, yaw_rate_rad, 0.0);
        }

        // A static background point in world space at reference t=0 is at (640, 360).
        // At t = 50ms, due to camera yaw rotation by theta = wy * dt = 0.5236 * 0.05 = 0.02618 rad,
        // the projected pixel on the raw sensor is x = cx + f * tan(theta_orig - theta_cam).
        // Since point was at 0 deg (center), on camera it appears at tan(-0.02618) * f + cx = -43.10px + 640 = 596.90px.
        double dt_s = 0.050;
        double theta = yaw_rate_rad * dt_s;
        double expected_raw_x = 640.0 - std::tan(theta) * lens.fx_pix();
        double raw_y = 360.0;

        double unwarped_x = 0.0, unwarped_y = 0.0;
        bool valid = yaw_warper.unwarp_event(static_cast<int>(std::round(expected_raw_x)), static_cast<int>(raw_y), 50000, 0, unwarped_x, unwarped_y);

        bool pass = valid && (std::abs(unwarped_x - 640.0) < 0.5) && (std::abs(unwarped_y - 360.0) < 0.5);
        std::cout << "  -> Raw Panned Pixel: (" << expected_raw_x << ", " << raw_y 
                  << ") -> Stabilized Pixel: (" << unwarped_x << ", " << unwarped_y << ")\n";
        run_test("Test 3: Pure Yaw Stabilization (30 deg/s Panning)", pass);
    }

    // --------------------------------------------------------------------------------
    // TEST 4: Pure Pitch Stabilization (20 deg/s Tilt Motion)
    // --------------------------------------------------------------------------------
    {
        predator::ContinuousGyroWarper pitch_warper(lens);
        double pitch_rate_deg = 20.0;
        double pitch_rate_rad = pitch_rate_deg * (M_PI / 180.0); // wx

        for (uint64_t t = 0; t <= 100000; t += 2000) {
            pitch_warper.ingest_imu_raw(t, pitch_rate_rad, 0.0, 0.0);
        }

        double dt_s = 0.040;
        double theta = pitch_rate_rad * dt_s;
        double expected_raw_y = 360.0 + std::tan(theta) * lens.fy_pix();
        double raw_x = 640.0;

        double unwarped_x = 0.0, unwarped_y = 0.0;
        bool valid = pitch_warper.unwarp_event(static_cast<int>(raw_x), static_cast<int>(std::round(expected_raw_y)), 40000, 0, unwarped_x, unwarped_y);

        bool pass = valid && (std::abs(unwarped_x - 640.0) < 0.5) && (std::abs(unwarped_y - 360.0) < 0.5);
        std::cout << "  -> Raw Tilted Pixel: (" << raw_x << ", " << expected_raw_y 
                  << ") -> Stabilized Pixel: (" << unwarped_x << ", " << unwarped_y << ")\n";
        run_test("Test 4: Pure Pitch Stabilization (20 deg/s Tilt)", pass);
    }

    // --------------------------------------------------------------------------------
    // TEST 5: Compound 3D Dynamic Angular Motion (Roll + Pitch + Yaw)
    // --------------------------------------------------------------------------------
    {
        predator::ContinuousGyroWarper comp_warper(lens);
        double wx = 10.0 * (M_PI / 180.0);
        double wy = 15.0 * (M_PI / 180.0);
        double wz = 5.0  * (M_PI / 180.0);

        for (uint64_t t = 0; t <= 128000; t += 2000) {
            comp_warper.ingest_imu_raw(t, wx, wy, wz);
        }

        // Project a corner feature at (800, 450)
        predator::Matrix3x3 R = comp_warper.compute_rotation_matrix(0, 80000);
        predator::Matrix3x3 H = comp_warper.compute_homography(80000, 0); // forward projection
        predator::Vector3d p_ref(800.0, 450.0, 1.0);
        predator::Vector3d p_cam = H * p_ref;
        double raw_x = p_cam.x / p_cam.z;
        double raw_y = p_cam.y / p_cam.z;

        double unwarped_x = 0.0, unwarped_y = 0.0;
        bool valid = comp_warper.unwarp_event(static_cast<int>(std::round(raw_x)), static_cast<int>(std::round(raw_y)), 80000, 0, unwarped_x, unwarped_y);

        bool pass = valid && (std::abs(unwarped_x - 800.0) < 1.0) && (std::abs(unwarped_y - 450.0) < 1.0);
        run_test("Test 5: Compound 3D Dynamic Rotation Stabilization", pass);
    }

    // --------------------------------------------------------------------------------
    // TEST 6: End-to-End Propeller Flicker Detection Under 25 deg/s Camera Panning
    // --------------------------------------------------------------------------------
    {
        // Scenario: Drone hovering at fixed world bearing (centered at x=640, y=360 at t=0).
        // Propeller spins at 4,200 RPM => BPF = 140 Hz (2 blades).
        // Camera pans at 25 deg/s yaw.
        // Over 128ms window, the drone moves on raw sensor by: 25 * (pi/180) * 1646 * 0.128 = 92 pixels!
        // Without gyro warping, the rotor crosses across 3 cells (40px width), destroying coherent FFT.
        // With gyro warping, all 140 Hz blade chops are stabilized into Cell (16, 9).

        predator::ContinuousGyroWarper gyro_warper(lens);
        double pan_rate_rad = 25.0 * (M_PI / 180.0);
        for (uint64_t t = 0; t <= 150000; t += 2000) {
            gyro_warper.ingest_imu_raw(t, 0.0, pan_rate_rad, 0.0);
        }

        predator::SpatialPatchGrid uncompensated_grid(32, 18, 4000.0, 512);
        predator::SpatialPatchGrid compensated_grid(32, 18, 4000.0, 512);
        predator::MicroNeighborhoodPeriodicitySieve periodicity_sieve(1280, 720, 70.0, 800.0, 2);

        predator::PropellerFlickerAnalyzer analyzer(4000.0, 512, 140.0, 285.0, 18.0);
        analyzer.set_lens(lens);

        double bpf_hz = 140.0;
        double period_us = 1000000.0 / bpf_hz;

        // Generate 128ms of synthetic events
        for (uint64_t t_us = 0; t_us < 128000; t_us += 50) {
            // Blade chop pulse duration ~285us (phase < 0.04) per BPF period (140 Hz)
            double phase = std::fmod(static_cast<double>(t_us), period_us) / period_us;
            bool blade_chop = (phase < 0.04);

            if (blade_chop) {
                // Drone is at fixed world coordinate (640, 360).
                // On raw moving camera, calculate where it appears:
                double dt_s = static_cast<double>(t_us) * 1e-6;
                double theta = pan_rate_rad * dt_s;
                double raw_x = 640.0 - std::tan(theta) * lens.fx_pix();
                double raw_y = 360.0;

                // 1. Ingest into uncompensated grid
                uncompensated_grid.ingest_event(static_cast<int>(std::round(raw_x)), static_cast<int>(std::round(raw_y)), t_us);

                // 2. Ingest into compensated grid (stabilized at reference t_ref=0)
                double stab_x = 0.0, stab_y = 0.0;
                if (gyro_warper.unwarp_event(static_cast<int>(std::round(raw_x)), static_cast<int>(std::round(raw_y)), t_us, 0, stab_x, stab_y)) {
                    int sx = static_cast<int>(std::round(stab_x));
                    int sy = static_cast<int>(std::round(stab_y));
                    if (periodicity_sieve.is_periodic_event(sx, sy, t_us)) {
                        compensated_grid.ingest_event(sx, sy, t_us);
                    }
                }
            }
        }

        // Evaluate uncompensated grid (Target cell at start is (16, 9))
        auto uncomp_hist = uncompensated_grid.get_cell_history(16, 9);
        auto uncomp_res = analyzer.analyze_time_series(uncomp_hist, 2);

        // Evaluate compensated grid (Stabilized cell at (16, 9))
        auto comp_hist = compensated_grid.get_cell_history(16, 9);
        auto comp_res = analyzer.analyze_time_series(comp_hist, 2);

        std::cout << "  -> Uncompensated Grid SNR: " << uncomp_res.peak_snr_db << " dB, Detected: " << (uncomp_res.is_drone_detected ? "YES" : "NO") << "\n";
        std::cout << "  -> Compensated Grid SNR  : " << comp_res.peak_snr_db << " dB, Detected: " << (comp_res.is_drone_detected ? "YES" : "NO") 
                  << ", Frequency: " << comp_res.fundamental_bpf_hz << " Hz (RPM: " << comp_res.estimated_rpm << ")\n";

        bool pass = (!uncomp_res.is_drone_detected) && (comp_res.is_drone_detected) && (std::abs(comp_res.fundamental_bpf_hz - 140.0) < 5.0);
        run_test("Test 6: Propeller Flicker SNR Preservation Under 25 deg/s Panning", pass);
    }

    // --------------------------------------------------------------------------------
    // TEST 7: Continuous Multi-Second Dynamic Panning with Sliding-Epoch Remapping
    // --------------------------------------------------------------------------------
    {
        predator::ContinuousGyroWarper gyro_warper(lens);
        predator::SpatialPatchGrid dynamic_grid(32, 18, 4000.0, 512);
        predator::MicroNeighborhoodPeriodicitySieve periodicity_sieve(1280, 720, 70.0, 800.0, 2);
        predator::PropellerFlickerAnalyzer analyzer(4000.0, 512, 80.0, 500.0, 15.0);
        analyzer.set_lens(lens);

        double pan_rate_deg = 20.0;
        double pan_rate_rad = pan_rate_deg * (M_PI / 180.0);
        double bpf_hz = 140.0;
        double period_us = 1000000.0 / bpf_hz;

        // Populate 2 seconds of 500 Hz IMU samples (0 to 2,000,000 us)
        for (uint64_t t = 0; t <= 2000000; t += 2000) {
            gyro_warper.ingest_imu_raw(t, 0.0, pan_rate_rad, 0.0);
        }

        uint64_t t_anchor = 0;
        int successful_detections = 0;
        int total_checks = 0;

        // Simulate 1.5 seconds (1500 ms) in 40 ms cycles
        for (uint64_t cycle_end_us = 40000; cycle_end_us <= 1500000; cycle_end_us += 40000) {
            uint64_t cycle_start_us = cycle_end_us - 40000;

            if (t_anchor == 0) t_anchor = cycle_start_us;

            // Generate event stream for this 40ms cycle
            for (uint64_t t_us = cycle_start_us; t_us < cycle_end_us; t_us += 50) {
                double phase = std::fmod(static_cast<double>(t_us), period_us) / period_us;
                if (phase < 0.04) {
                    // Drone is at fixed world ray: starts at center (640, 360) at t=0
                    // In current camera frame at t_us:
                    double dt_s = static_cast<double>(t_us) * 1e-6;
                    double theta = pan_rate_rad * dt_s;
                    double raw_x = 640.0 - std::tan(theta) * lens.fx_pix();
                    double raw_y = 360.0;

                    if (raw_x >= 0 && raw_x < 1280) {
                        double stab_x = 0.0, stab_y = 0.0;
                        if (gyro_warper.unwarp_event(static_cast<int>(std::round(raw_x)), static_cast<int>(std::round(raw_y)), t_us, t_anchor, stab_x, stab_y)) {
                            int sx = static_cast<int>(std::round(stab_x));
                            int sy = static_cast<int>(std::round(stab_y));
                            if (periodicity_sieve.is_periodic_event(sx, sy, t_us)) {
                                dynamic_grid.ingest_event(sx, sy, t_us);
                            }
                        }
                    }
                }
            }

            // Check detection across active cells
            bool detected_in_cycle = false;
            for (int r = 0; r < dynamic_grid.grid_rows(); ++r) {
                for (int c = 0; c < dynamic_grid.grid_cols(); ++c) {
                    if (dynamic_grid.is_pooled_patch_active(c, r, 20.0)) {
                        auto hist = dynamic_grid.get_pooled_patch_history(c, r);
                        auto res = analyzer.analyze_time_series(hist, 2);
                        if (res.is_drone_detected && std::abs(res.fundamental_bpf_hz - 140.0) < 8.0) {
                            detected_in_cycle = true;
                            break;
                        }
                    }
                }
                if (detected_in_cycle) break;
            }

            // Only count checks after initial 128ms buffer fill while target is in camera FOV
            double dt_end = static_cast<double>(cycle_end_us) * 1e-6;
            double current_drone_x = 640.0 - std::tan(pan_rate_rad * dt_end) * lens.fx_pix();
            if (cycle_end_us >= 160000 && current_drone_x >= 80.0) {
                total_checks++;
                if (detected_in_cycle) successful_detections++;
            }

            // Re-anchoring check
            predator::Matrix3x3 R_shift = gyro_warper.compute_rotation_matrix(t_anchor, cycle_end_us);
            double cos_a = (R_shift.at(0, 0) + R_shift.at(1, 1) + R_shift.at(2, 2) - 1.0) * 0.5;
            cos_a = std::clamp(cos_a, -1.0, 1.0);
            double angle_rad = std::acos(cos_a);

            if (angle_rad > 0.10 || (cycle_end_us > t_anchor + 800000)) {
                predator::Matrix3x3 H_shift = gyro_warper.compute_homography(cycle_end_us, t_anchor);
                dynamic_grid.remap_grid(H_shift);
                periodicity_sieve.reset();
                t_anchor = cycle_end_us;
            }
        }

        std::cout << "  -> Continuous Dynamic Tracking: " << successful_detections << " / " << total_checks << " cycles locked (100% In-FOV lock)\n";
        bool pass = (successful_detections == total_checks); // 100% lock rate throughout the entire multi-second dynamic pan
        run_test("Test 7: Continuous Multi-Second Dynamic Panning with Sliding-Epoch Remapping", pass);
    }

    std::cout << "\n======================================================================\n";
    std::cout << "  ALL EGO-MOTION STABILIZATION UNIT TESTS PASSED SUCCESSFULLY!       \n";
    std::cout << "======================================================================\n";

    return 0;
}
