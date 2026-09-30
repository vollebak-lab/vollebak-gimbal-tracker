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
    lens.focal_length_mm = 8.0;
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

        predator::PropellerFlickerAnalyzer analyzer(4000.0, 512, 140.0, 285.0, 18.0);
        analyzer.set_lens(lens);

        double bpf_hz = 140.0;
        double period_us = 1000000.0 / bpf_hz;

        // Generate 128ms of synthetic events
        for (uint64_t t_us = 0; t_us < 128000; t_us += 50) {
            // Single blade chop pulse per BPF period (140 Hz)
            double phase = std::fmod(static_cast<double>(t_us), period_us) / period_us;
            bool blade_chop = (phase < 0.20);

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
                    compensated_grid.ingest_event(static_cast<int>(std::round(stab_x)), static_cast<int>(std::round(stab_y)), t_us);
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

    std::cout << "\n======================================================================\n";
    std::cout << "  ALL EGO-MOTION STABILIZATION UNIT TESTS PASSED SUCCESSFULLY!       \n";
    std::cout << "======================================================================\n";

    return 0;
}
