// ---
// VOLLEBAK_ENGINEERING_METADATA:
//   PROJECT_ID: PREDATOR-01
//   TRACK: ECOSYSTEM
//   PHASE: CHALLENGE_HUB
//   CONTRIBUTOR_ID: Antigravity AI
//   THE_DELTA: 2-axis self-leveling PID controller for mast base using Maxon EC 20 flat + GP 22 3-stage gearhead
//   FAILURE_MODE: Without mast stabilization, operator body movement (pitch/roll) consumes gimbal range and corrupts parallax correction
//   IP_STATUS: VOLLEBAK_PROPRIETARY
//   DEPENDENCIES: [predator-messages]
// ---

//! # Mast Self-Leveling Stabilization Controller
//!
//! Maintains vertical orientation of the mast independent of operator
//! body movement (pitch and roll). Yaw is handled by gimbal azimuth.
//!
//! ## Hardware
//!
//! - 2× Maxon EC 20 flat 3W brushless motors
//! - 2× GP 22 planetary gearhead, 3-stage (~350:1)
//! - Output torque: ~1.0 Nm per axis
//! - Total mass: ~186g (30g motors + 136g gearheads + 20g hardware)
//!
//! ## Bandwidth Partitioning
//!
//! | Motion | Frequency | Handler |
//! |--------|-----------|---------|
//! | Body pitch/roll | 0–5 Hz | **This controller** (100Hz loop) |
//! | Operator yaw | 0–3 Hz | Gimbal azimuth (360° travel) |
//! | Target tracking | 0–20 Hz | Gimbal PID (200Hz) |
//! | Beam fine-pointing | 0–200 Hz | Fast Steer Mirror |

use predator_messages::{
    MastStabilizationMode, MastStabilizationStatus,
};
use serde::{Deserialize, Serialize};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the mast stabilization PID controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MastStabilizationConfig {
    /// Enable/disable stabilization. When false, controller stays in Idle.
    pub enabled: bool,

    /// PID gains for the pitch axis.
    pub pitch_gains: StabilizationPidGains,

    /// PID gains for the roll axis.
    pub roll_gains: StabilizationPidGains,

    /// Maximum correction angle in degrees (±). Based on EC 20 + GP 22
    /// torque budget: ~1.0 Nm supports ±25° with 500g turret at 0.5m.
    pub max_correction_angle_deg: f64,

    /// Leveling threshold in degrees. Both pitch and roll residual
    /// error must be below this for `is_leveled == true`.
    /// Default: 0.5° — at 100m range, 0.5° = 0.87m error, within
    /// the 2-5W beam dwell spot.
    pub leveling_threshold_deg: f64,

    /// Motor overcurrent threshold (amps). Exceeding triggers fault.
    pub overcurrent_threshold_amps: f64,

    /// Motor stall detection: if error > threshold for N consecutive
    /// samples with motor at max current, declare stall fault.
    pub stall_error_threshold_deg: f64,
    pub stall_sample_count: u32,

    /// Controller update rate in Hz.
    pub update_rate_hz: u32,
}

impl Default for MastStabilizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            pitch_gains: StabilizationPidGains::default(),
            roll_gains: StabilizationPidGains::default(),
            max_correction_angle_deg: 25.0,
            leveling_threshold_deg: 0.5,
            overcurrent_threshold_amps: 0.8,
            stall_error_threshold_deg: 5.0,
            stall_sample_count: 50, // 0.5s at 100Hz
            update_rate_hz: 100,
        }
    }
}

/// PID gains for a single stabilization axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilizationPidGains {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    /// Anti-windup integral clamp.
    pub max_integral: f64,
    /// Max output in Nm (clamped to actuator torque budget).
    pub max_output_nm: f64,
}

impl Default for StabilizationPidGains {
    fn default() -> Self {
        Self {
            kp: 3.0,   // Aggressive for fast leveling
            ki: 0.5,   // Steady-state error elimination
            kd: 0.15,  // Damping to prevent oscillation
            max_integral: 5.0,
            max_output_nm: 1.0, // GP 22 3-stage output limit
        }
    }
}

// ---------------------------------------------------------------------------
// Single-Axis PID
// ---------------------------------------------------------------------------

/// Single-axis PID state for stabilization.
#[derive(Debug, Clone)]
struct AxisPid {
    gains: StabilizationPidGains,
    integral: f64,
    last_error: f64,
    last_time: Instant,
}

impl AxisPid {
    fn new(gains: StabilizationPidGains) -> Self {
        Self {
            gains,
            integral: 0.0,
            last_error: 0.0,
            last_time: Instant::now(),
        }
    }

    /// Compute PID output (torque command in Nm) given error in degrees.
    fn update(&mut self, error_deg: f64) -> f64 {
        let now = Instant::now();
        let dt = now.duration_since(self.last_time).as_secs_f64();
        self.last_time = now;

        if dt <= 0.0 || dt > 0.1 {
            // Guard: stale or zero delta
            self.last_error = error_deg;
            return self.gains.kp * error_deg;
        }

        // Proportional
        let p = self.gains.kp * error_deg;

        // Integral with anti-windup
        self.integral += error_deg * dt;
        self.integral = self
            .integral
            .clamp(-self.gains.max_integral, self.gains.max_integral);
        let i = self.gains.ki * self.integral;

        // Derivative
        let d = self.gains.kd * (error_deg - self.last_error) / dt;
        self.last_error = error_deg;

        // Clamp output to actuator torque budget
        (p + i + d).clamp(-self.gains.max_output_nm, self.gains.max_output_nm)
    }

    /// Reset integrator and derivative state.
    fn reset(&mut self) {
        self.integral = 0.0;
        self.last_error = 0.0;
        self.last_time = Instant::now();
    }
}

// ---------------------------------------------------------------------------
// Stabilization Controller
// ---------------------------------------------------------------------------

/// Fault codes for the stabilization controller.
pub const FAULT_NONE: u16 = 0;
pub const FAULT_PITCH_OVERCURRENT: u16 = 1;
pub const FAULT_ROLL_OVERCURRENT: u16 = 2;
pub const FAULT_PITCH_STALL: u16 = 3;
pub const FAULT_ROLL_STALL: u16 = 4;
pub const FAULT_IMU_INVALID: u16 = 5;

/// 2-axis self-leveling mast stabilization controller.
///
/// Runs a 100Hz PID loop that reads body orientation (pitch, roll)
/// from IMU data and commands the mast base motors to maintain
/// vertical turret orientation.
///
/// The controller publishes [`MastStabilizationStatus`] with the
/// current leveling state. The orchestrator uses `is_leveled` to
/// gate engagement authorization.
pub struct MastStabilizationController {
    config: MastStabilizationConfig,
    mode: MastStabilizationMode,
    pitch_pid: AxisPid,
    roll_pid: AxisPid,

    // Current state
    current_pitch_deg: f64,
    current_roll_deg: f64,
    pitch_error_deg: f64,
    roll_error_deg: f64,

    // Motor feedback (simulated from PID output for now)
    motor_current_pitch: f64,
    motor_current_roll: f64,

    // Fault tracking
    fault_code: u16,
    pitch_stall_counter: u32,
    roll_stall_counter: u32,
}

impl MastStabilizationController {
    /// Create a new stabilization controller with the given config.
    pub fn new(config: MastStabilizationConfig) -> Self {
        let pitch_pid = AxisPid::new(config.pitch_gains.clone());
        let roll_pid = AxisPid::new(config.roll_gains.clone());

        Self {
            config,
            mode: MastStabilizationMode::Idle,
            pitch_pid,
            roll_pid,
            current_pitch_deg: 0.0,
            current_roll_deg: 0.0,
            pitch_error_deg: 0.0,
            roll_error_deg: 0.0,
            motor_current_pitch: 0.0,
            motor_current_roll: 0.0,
            fault_code: FAULT_NONE,
            pitch_stall_counter: 0,
            roll_stall_counter: 0,
        }
    }

    /// Set the operating mode. Resets PID state on mode transitions.
    pub fn set_mode(&mut self, mode: MastStabilizationMode) {
        if mode != self.mode {
            self.pitch_pid.reset();
            self.roll_pid.reset();
            self.fault_code = FAULT_NONE;
            self.pitch_stall_counter = 0;
            self.roll_stall_counter = 0;
            self.mode = mode;
        }
    }

    /// Process one stabilization cycle.
    ///
    /// # Arguments
    ///
    /// * `body_pitch_deg` — Current body pitch from IMU (degrees from vertical)
    /// * `body_roll_deg` — Current body roll from IMU (degrees from vertical)
    /// * `imu_valid` — Whether the IMU data is fresh
    /// * `motor_current_pitch` — Feedback: pitch motor current (amps)
    /// * `motor_current_roll` — Feedback: roll motor current (amps)
    ///
    /// # Returns
    ///
    /// Tuple of (pitch_torque_nm, roll_torque_nm) motor commands.
    pub fn update(
        &mut self,
        body_pitch_deg: f64,
        body_roll_deg: f64,
        imu_valid: bool,
        motor_current_pitch: f64,
        motor_current_roll: f64,
    ) -> (f64, f64) {
        self.motor_current_pitch = motor_current_pitch;
        self.motor_current_roll = motor_current_roll;

        match self.mode {
            MastStabilizationMode::Idle => {
                self.current_pitch_deg = body_pitch_deg;
                self.current_roll_deg = body_roll_deg;
                self.pitch_error_deg = body_pitch_deg;
                self.roll_error_deg = body_roll_deg;
                (0.0, 0.0)
            }
            MastStabilizationMode::Stabilize => {
                if !imu_valid {
                    self.fault_code = FAULT_IMU_INVALID;
                    return (0.0, 0.0);
                }

                // Check for faults
                self.check_faults(motor_current_pitch, motor_current_roll);
                if self.fault_code != FAULT_NONE {
                    return (0.0, 0.0);
                }

                self.current_pitch_deg = body_pitch_deg;
                self.current_roll_deg = body_roll_deg;

                // Target: vertical (0° pitch, 0° roll)
                // Clamp input to max correction range
                let clamped_pitch = body_pitch_deg.clamp(
                    -self.config.max_correction_angle_deg,
                    self.config.max_correction_angle_deg,
                );
                let clamped_roll = body_roll_deg.clamp(
                    -self.config.max_correction_angle_deg,
                    self.config.max_correction_angle_deg,
                );

                // PID error: how far from vertical
                let pitch_torque = self.pitch_pid.update(clamped_pitch);
                let roll_torque = self.roll_pid.update(clamped_roll);

                // After correction, the residual is the uncorrected portion.
                // In real hardware, residual comes from encoder feedback.
                // Here we estimate: if torque is at limit, residual persists.
                let pitch_correction_ratio = if clamped_pitch.abs() > 0.01 {
                    (pitch_torque / self.config.pitch_gains.max_output_nm).abs().min(1.0)
                } else {
                    1.0
                };
                let roll_correction_ratio = if clamped_roll.abs() > 0.01 {
                    (roll_torque / self.config.roll_gains.max_output_nm).abs().min(1.0)
                } else {
                    1.0
                };

                // Residual error: portion the motor couldn't correct
                // In real hardware, this comes from encoder - commanded position
                self.pitch_error_deg = clamped_pitch * (1.0 - pitch_correction_ratio * 0.95);
                self.roll_error_deg = clamped_roll * (1.0 - roll_correction_ratio * 0.95);

                (pitch_torque, roll_torque)
            }
            MastStabilizationMode::Override => {
                // Manual setpoints — PID targets the override value
                self.current_pitch_deg = body_pitch_deg;
                self.current_roll_deg = body_roll_deg;
                self.pitch_error_deg = 0.0;
                self.roll_error_deg = 0.0;
                (0.0, 0.0) // Override commands bypass PID
            }
        }
    }

    /// Check whether the mast is leveled within threshold.
    pub fn is_leveled(&self) -> bool {
        self.mode == MastStabilizationMode::Stabilize
            && self.fault_code == FAULT_NONE
            && self.pitch_error_deg.abs() < self.config.leveling_threshold_deg
            && self.roll_error_deg.abs() < self.config.leveling_threshold_deg
    }

    /// Generate a status message for the orchestrator.
    pub fn status(&self, timestamp_us: u64) -> MastStabilizationStatus {
        MastStabilizationStatus {
            pitch_deg: self.current_pitch_deg,
            roll_deg: self.current_roll_deg,
            pitch_error_deg: self.pitch_error_deg,
            roll_error_deg: self.roll_error_deg,
            is_leveled: self.is_leveled(),
            motor_current_pitch_amps: self.motor_current_pitch,
            motor_current_roll_amps: self.motor_current_roll,
            fault_code: self.fault_code,
            mode: self.mode,
            timestamp_us,
        }
    }

    /// Get the current fault code.
    pub fn fault_code(&self) -> u16 {
        self.fault_code
    }

    /// Get the current operating mode.
    pub fn mode(&self) -> MastStabilizationMode {
        self.mode
    }

    /// Get the residual pitch error after stabilization (degrees).
    pub fn residual_pitch_deg(&self) -> f64 {
        self.pitch_error_deg
    }

    /// Get the residual roll error after stabilization (degrees).
    pub fn residual_roll_deg(&self) -> f64 {
        self.roll_error_deg
    }

    /// Clear a fault and reset PID state.
    pub fn clear_fault(&mut self) {
        self.fault_code = FAULT_NONE;
        self.pitch_stall_counter = 0;
        self.roll_stall_counter = 0;
        self.pitch_pid.reset();
        self.roll_pid.reset();
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    /// Check motor current and stall conditions.
    fn check_faults(&mut self, pitch_current: f64, roll_current: f64) {
        // Overcurrent check
        if pitch_current > self.config.overcurrent_threshold_amps {
            self.fault_code = FAULT_PITCH_OVERCURRENT;
            return;
        }
        if roll_current > self.config.overcurrent_threshold_amps {
            self.fault_code = FAULT_ROLL_OVERCURRENT;
            return;
        }

        // Stall detection: high error + high current for sustained period
        if self.pitch_error_deg.abs() > self.config.stall_error_threshold_deg
            && pitch_current > self.config.overcurrent_threshold_amps * 0.8
        {
            self.pitch_stall_counter += 1;
            if self.pitch_stall_counter >= self.config.stall_sample_count {
                self.fault_code = FAULT_PITCH_STALL;
                return;
            }
        } else {
            self.pitch_stall_counter = 0;
        }

        if self.roll_error_deg.abs() > self.config.stall_error_threshold_deg
            && roll_current > self.config.overcurrent_threshold_amps * 0.8
        {
            self.roll_stall_counter += 1;
            if self.roll_stall_counter >= self.config.stall_sample_count {
                self.fault_code = FAULT_ROLL_STALL;
                return;
            }
        } else {
            self.roll_stall_counter = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_controller() -> MastStabilizationController {
        MastStabilizationController::new(MastStabilizationConfig::default())
    }

    #[test]
    fn test_idle_mode_no_output() {
        let mut ctrl = default_controller();
        ctrl.set_mode(MastStabilizationMode::Idle);

        let (pitch, roll) = ctrl.update(10.0, 5.0, true, 0.0, 0.0);
        assert!((pitch).abs() < 1e-9, "Idle should produce zero pitch torque");
        assert!((roll).abs() < 1e-9, "Idle should produce zero roll torque");
        assert!(!ctrl.is_leveled(), "Idle mode should not report leveled");
    }

    #[test]
    fn test_stabilize_produces_corrective_torque() {
        let mut ctrl = default_controller();
        ctrl.set_mode(MastStabilizationMode::Stabilize);

        // 15° pitch lean forward → should produce positive corrective torque
        let (pitch_torque, _roll_torque) = ctrl.update(15.0, 0.0, true, 0.1, 0.0);
        assert!(
            pitch_torque > 0.0,
            "Should produce positive pitch torque for forward lean: {}",
            pitch_torque
        );
    }

    #[test]
    fn test_stabilize_roll_corrective_torque() {
        let mut ctrl = default_controller();
        ctrl.set_mode(MastStabilizationMode::Stabilize);

        // 10° roll right → should produce positive corrective roll torque
        let (_pitch_torque, roll_torque) = ctrl.update(0.0, 10.0, true, 0.0, 0.1);
        assert!(
            roll_torque > 0.0,
            "Should produce positive roll torque for right lean: {}",
            roll_torque
        );
    }

    #[test]
    fn test_leveled_when_error_within_threshold() {
        let mut ctrl = default_controller();
        ctrl.set_mode(MastStabilizationMode::Stabilize);

        // Small tilt → should be leveled after correction
        ctrl.update(0.1, 0.1, true, 0.01, 0.01);
        assert!(
            ctrl.is_leveled(),
            "Small tilt should report leveled, pitch_err={}, roll_err={}",
            ctrl.residual_pitch_deg(),
            ctrl.residual_roll_deg()
        );
    }

    #[test]
    fn test_not_leveled_during_large_tilt() {
        let mut ctrl = default_controller();
        ctrl.set_mode(MastStabilizationMode::Stabilize);

        // Large tilt → residual should exceed threshold
        ctrl.update(20.0, 20.0, true, 0.1, 0.1);
        assert!(
            !ctrl.is_leveled(),
            "Large tilt should not report leveled, pitch_err={}, roll_err={}",
            ctrl.residual_pitch_deg(),
            ctrl.residual_roll_deg()
        );
    }

    #[test]
    fn test_imu_invalid_sets_fault() {
        let mut ctrl = default_controller();
        ctrl.set_mode(MastStabilizationMode::Stabilize);

        ctrl.update(5.0, 5.0, false, 0.0, 0.0);
        assert_eq!(ctrl.fault_code(), FAULT_IMU_INVALID);
        assert!(!ctrl.is_leveled());
    }

    #[test]
    fn test_overcurrent_fault() {
        let mut ctrl = default_controller();
        ctrl.set_mode(MastStabilizationMode::Stabilize);

        // First call with normal current
        ctrl.update(5.0, 0.0, true, 0.1, 0.0);
        assert_eq!(ctrl.fault_code(), FAULT_NONE);

        // Exceed overcurrent threshold (default 0.8A)
        ctrl.update(5.0, 0.0, true, 1.0, 0.0);
        assert_eq!(ctrl.fault_code(), FAULT_PITCH_OVERCURRENT);
    }

    #[test]
    fn test_clear_fault_resets_state() {
        let mut ctrl = default_controller();
        ctrl.set_mode(MastStabilizationMode::Stabilize);

        ctrl.update(5.0, 0.0, false, 0.0, 0.0);
        assert_eq!(ctrl.fault_code(), FAULT_IMU_INVALID);

        ctrl.clear_fault();
        assert_eq!(ctrl.fault_code(), FAULT_NONE);
    }

    #[test]
    fn test_mode_transition_resets_pid() {
        let mut ctrl = default_controller();
        ctrl.set_mode(MastStabilizationMode::Stabilize);
        ctrl.update(10.0, 10.0, true, 0.1, 0.1);

        // Transition to idle and back → should reset PID integrator
        ctrl.set_mode(MastStabilizationMode::Idle);
        ctrl.set_mode(MastStabilizationMode::Stabilize);

        let (pitch, _roll) = ctrl.update(0.0, 0.0, true, 0.0, 0.0);
        // After reset with zero error, output should be very small
        assert!(
            pitch.abs() < 0.1,
            "After reset + zero error, torque should be near zero: {}",
            pitch
        );
    }

    #[test]
    fn test_status_message_generation() {
        let mut ctrl = default_controller();
        ctrl.set_mode(MastStabilizationMode::Stabilize);
        ctrl.update(0.1, -0.1, true, 0.01, 0.02);

        let status = ctrl.status(1_000_000);
        assert!((status.pitch_deg - 0.1).abs() < 1e-9);
        assert!((status.roll_deg - (-0.1)).abs() < 1e-9);
        assert_eq!(status.mode, MastStabilizationMode::Stabilize);
        assert_eq!(status.timestamp_us, 1_000_000);
    }

    #[test]
    fn test_clamped_correction_range() {
        let mut ctrl = default_controller();
        ctrl.set_mode(MastStabilizationMode::Stabilize);

        // 40° lean exceeds max_correction_angle_deg (25°)
        let (pitch, _) = ctrl.update(40.0, 0.0, true, 0.1, 0.0);
        // Output should be at max (clamped input)
        assert!(
            pitch <= ctrl.config.pitch_gains.max_output_nm,
            "Torque should not exceed max: {}",
            pitch
        );
    }

    #[test]
    fn test_fault_blocks_output() {
        let mut ctrl = default_controller();
        ctrl.set_mode(MastStabilizationMode::Stabilize);

        // Create fault via invalid IMU
        ctrl.update(5.0, 5.0, false, 0.0, 0.0);
        assert_ne!(ctrl.fault_code(), FAULT_NONE);

        // Subsequent update should still return zero (fault blocks output)
        let (_pitch, _roll) = ctrl.update(5.0, 5.0, true, 0.0, 0.0);
        assert_eq!(_pitch, 0.0);
        assert_eq!(_roll, 0.0);
        // Note: fault persists until cleared — but IMU invalid overrides
        // Actually fault_code stays as IMU_INVALID, need to clear first
        assert!(!ctrl.is_leveled());
    }
}
