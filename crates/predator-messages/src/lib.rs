// ---
// VOLLEBAK_ENGINEERING_METADATA:
//   PROJECT_ID: PREDATOR-01
//   TRACK: ECOSYSTEM
//   PHASE: CHALLENGE_HUB
//   CONTRIBUTOR_ID: Antigravity AI
//   THE_DELTA: Shared binary-serializable Zenoh message types for cross-language Python↔Rust pipeline communication
//   FAILURE_MODE: JSON serialization on timing-critical Zenoh topics adds 10-25ms aggregate latency
//   IP_STATUS: VOLLEBAK_PROPRIETARY
//   USER_FEEDBACK_REF: Language Selection Audit — Binary Zenoh payloads
//   DEPENDENCIES: [serde, rmp-serde]
// ---

//! # Predator Messages
//!
//! Shared Zenoh message types used across the Predator C-UAS pipeline.
//! All messages are serializable via both JSON (for debugging) and
//! MessagePack (for production latency-critical paths).
//!
//! ## Topic Namespace
//!
//! | Topic | Direction | Payload |
//! |-------|-----------|---------|
//! | `predator/layer1/detection` | Python → Rust | [`Layer1Detection`] |
//! | `predator/radar/track` | Python → Rust | [`RadarTrackUpdate`] |
//! | `predator/radar/classification` | Python → Rust | [`ThreatClassification`] |
//! | `predator/imu/gimbal_feedforward` | Python → Rust | [`ImuFeedForward`] |
//! | `predator/system/state` | Rust → Python | [`SystemStateMsg`] |
//! | `predator/engagement/pre_slew` | Rust → Rust | [`PreSlewCommand`] |
//! | `predator/engagement/command` | Rust → Rust | [`GimbalCommand`] |
//! | `predator/arm/command` | Rust → HW | [`ArmCommand`] |
//! | `predator/arm/status` | HW → Rust | [`ArmStatus`] |
//! | `predator/mast/stabilization/command` | Rust → HW | [`MastStabilizationCommand`] |
//! | `predator/mast/stabilization/status` | HW → Rust | [`MastStabilizationStatus`] |

pub mod topics;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Zenoh topic constants
// ---------------------------------------------------------------------------

/// Zenoh topic namespace constants.
pub mod zenoh_topics {
    pub const LAYER1_DETECTION: &str = "predator/layer1/detection";
    pub const RADAR_TRACK: &str = "predator/radar/track";
    pub const RADAR_CLASSIFICATION: &str = "predator/radar/classification";
    pub const IMU_FEEDFORWARD: &str = "predator/imu/gimbal_feedforward";
    pub const SYSTEM_STATE: &str = "predator/system/state";
    pub const ENGAGEMENT_PRE_SLEW: &str = "predator/engagement/pre_slew";
    pub const ENGAGEMENT_COMMAND: &str = "predator/engagement/command";
    pub const ENGAGEMENT_STATUS: &str = "predator/engagement/status";
    pub const ARM_COMMAND: &str = "predator/arm/command";
    pub const ARM_STATUS: &str = "predator/arm/status";
    pub const MAST_STABILIZATION_COMMAND: &str = "predator/mast/stabilization/command";
    pub const MAST_STABILIZATION_STATUS: &str = "predator/mast/stabilization/status";
}

// ---------------------------------------------------------------------------
// System State
// ---------------------------------------------------------------------------

/// Predator engagement pipeline states.
///
/// Mirrors the Python `SystemState` enum exactly for cross-language
/// compatibility. Transition rules are enforced by the Rust orchestrator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SystemState {
    /// No threat detected. Radar silent. Arm stowed.
    Silent,
    /// Layer 1 detection received. Arm deploying. Fast Steer Mirror pre-aiming.
    Alert,
    /// Radar authorized and searching for target.
    RadarActive,
    /// Radar has confirmed track. Threat scoring in progress.
    Tracking,
    /// Threat confirmed. Laser engagement authorized (radar-guided).
    Engagement,
    /// L1-only passive engagement (Waiter mode). Zero RF emission.
    L1Engagement,
    /// Post-engagement battle damage assessment.
    Bda,
}

impl std::fmt::Display for SystemState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Silent => write!(f, "SILENT"),
            Self::Alert => write!(f, "ALERT"),
            Self::RadarActive => write!(f, "RADAR_ACTIVE"),
            Self::Tracking => write!(f, "TRACKING"),
            Self::Engagement => write!(f, "ENGAGEMENT"),
            Self::L1Engagement => write!(f, "L1_ENGAGEMENT"),
            Self::Bda => write!(f, "BDA"),
        }
    }
}

// ---------------------------------------------------------------------------
// Layer 1 — Neuromorphic Detection
// ---------------------------------------------------------------------------

/// A propeller flicker detection from the DVXplorer Micro array.
///
/// Published by the Python `propeller_detector.py` on
/// `predator/layer1/detection`. Consumed by the Rust orchestrator
/// to drive state machine transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer1Detection {
    /// Source camera identifier (0-3).
    pub camera_id: u8,
    /// Global bearing to detection in degrees (0 = forward, CW).
    pub bearing_deg: f64,
    /// Detection confidence (0.0–1.0).
    pub confidence: f64,
    /// Hardware timestamp in microseconds.
    pub timestamp_us: u64,
    /// Elevation angle in degrees (0=horizon, negative=below).
    /// Required for Waiter mode auto-detect.
    pub elevation_deg: Option<f64>,
    /// Vertical angular rate from L1 tracker (deg/s).
    /// Positive = ascending (ground-launch Waiter).
    pub centroid_vy_degps: Option<f64>,
}

// ---------------------------------------------------------------------------
// Layer 2 — Radar
// ---------------------------------------------------------------------------

/// A radar track update from the aerial target tracker.
///
/// Published by the Python `aerial_target_tracker.py` on
/// `predator/radar/track`. Contains kinematic state for the
/// orchestrator's threat scoring and gimbal cueing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadarTrackUpdate {
    /// Tracker-assigned target ID.
    pub track_id: u32,
    /// Target range in meters.
    pub range_m: f64,
    /// Target azimuth in degrees.
    pub azimuth_deg: f64,
    /// Target elevation in degrees.
    pub elevation_deg: f64,
    /// Target speed in m/s (magnitude of velocity vector).
    pub speed_mps: f64,
    /// Radial velocity (closing rate) in m/s. Negative = closing.
    pub radial_velocity_mps: f64,
    /// Target altitude AGL in meters.
    pub altitude_m: f64,
    /// Radar cross-section in dBsm.
    pub rcs_dbsm: f64,
    /// Timestamp in microseconds.
    pub timestamp_us: u64,
}

/// Threat classification result from the Python threat classifier.
///
/// Published on `predator/radar/classification`. May be a kinematic
/// primer (fast, no micro-Doppler) or a full assessment (with STFT).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatClassification {
    /// Track ID this classification applies to.
    pub track_id: u32,
    /// Aggregated threat score (0.0–1.0).
    pub threat_score: f64,
    /// Whether this is a kinematic primer (true) or full assessment (false).
    pub is_primer: bool,
    /// Classified target type (e.g., "rotary_uas", "fixed_wing", "bird").
    pub target_class: String,
    /// Flicker confidence from Layer 1.
    pub flicker_confidence: f64,
    /// Micro-Doppler confidence (0.0 if primer, >0 if full).
    pub doppler_confidence: f64,
    /// Timestamp in microseconds.
    pub timestamp_us: u64,
}

// ---------------------------------------------------------------------------
// IMU
// ---------------------------------------------------------------------------

/// IMU feed-forward correction for gimbal stabilization.
///
/// Published by the Python `motion_compensator.py` on
/// `predator/imu/gimbal_feedforward`. Consumed by both the Rust
/// orchestrator and the gimbal PID controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImuFeedForward {
    /// Azimuth correction to subtract (degrees).
    pub delta_az_deg: f64,
    /// Elevation correction to subtract (degrees).
    pub delta_el_deg: f64,
    /// Roll correction (degrees). Positive = operator tilted right.
    pub delta_roll_deg: f64,
    /// Operator heading rotation rate (deg/s).
    pub angular_rate_az_dps: f64,
    /// Operator pitch rotation rate (deg/s).
    pub angular_rate_el_dps: f64,
    /// Operator roll rotation rate (deg/s).
    pub angular_rate_roll_dps: f64,
    /// Whether the IMU data is fresh enough for feed-forward.
    pub is_valid: bool,
}

impl Default for ImuFeedForward {
    fn default() -> Self {
        Self {
            delta_az_deg: 0.0,
            delta_el_deg: 0.0,
            delta_roll_deg: 0.0,
            angular_rate_az_dps: 0.0,
            angular_rate_el_dps: 0.0,
            angular_rate_roll_dps: 0.0,
            is_valid: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Engagement — Commands
// ---------------------------------------------------------------------------

/// Pre-slew command for mast deployment and Fast Steer Mirror coarse aim.
///
/// Published by the orchestrator on `predator/engagement/pre_slew`
/// upon first Layer 1 detection. The mast extends vertically from
/// Z-fold stow to full height above the operator's head.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreSlewCommand {
    /// Target bearing for Fast Steer Mirror coarse aim (degrees, 0 = fwd, CW).
    pub bearing_deg: f64,
    /// Timestamp in microseconds.
    pub timestamp_us: u64,
}

/// Arm/mast actuator command.
///
/// Published on `predator/arm/command`. The mast extends vertically
/// as a single-axis Z-fold from the rear plate carrier mount.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmCommand {
    /// Command type.
    pub action: ArmAction,
    /// Timestamp in microseconds.
    pub timestamp_us: u64,
}

/// Mast action types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArmAction {
    /// Extend mast from Z-fold stow to full vertical position.
    Deploy,
    /// Retract mast to Z-fold stow position on plate carrier.
    Stow,
}

/// Mast deployment state — reported by mast motor controller.
///
/// The orchestrator gates laser engagement authorization on
/// `Deployed` + `is_stable == true`. Any other state blocks firing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArmDeploymentState {
    /// Mast is in Z-fold stow position on plate carrier. Locked flat.
    Stowed,
    /// Mast is actively extending vertically (Z-fold unfolding).
    Deploying,
    /// Mast is fully extended to vertical position. Locked and stable.
    Deployed,
    /// Mast is retracting to Z-fold stow position.
    Stowing,
    /// Mast actuator fault — motor overcurrent, jam, encoder loss.
    Fault,
}

impl std::fmt::Display for ArmDeploymentState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stowed => write!(f, "STOWED"),
            Self::Deploying => write!(f, "DEPLOYING"),
            Self::Deployed => write!(f, "DEPLOYED"),
            Self::Stowing => write!(f, "STOWING"),
            Self::Fault => write!(f, "FAULT"),
        }
    }
}

/// Mast status feedback from the motor controller.
///
/// Published on `predator/arm/status` by the mast motor controller
/// at ~50Hz. The orchestrator subscribes to this to gate engagement
/// authorization — laser fire is blocked until the mast reports
/// `Deployed` and `is_stable == true`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmStatus {
    /// Current deployment state.
    pub deployment_state: ArmDeploymentState,
    /// Whether the mast is mechanically stable (settled, no vibration).
    /// Determined by the motor controller when joint encoders show
    /// position error < threshold for N consecutive samples.
    pub is_stable: bool,
    /// Actual measured mast extension height in meters.
    /// Derived from encoder positions. 0.0 = fully stowed.
    pub mast_height_m: f64,
    /// Joint angles in degrees for each of the 3 Z-fold segments.
    /// Index 0 = base (plate carrier hinge), 1 = mid, 2 = top.
    pub joint_angles_deg: Vec<f64>,
    /// Deployment progress (0.0 = stowed, 1.0 = fully deployed).
    pub deployment_progress: f64,
    /// Motor current draw in amps (for thermal monitoring).
    pub motor_current_amps: f64,
    /// Motor temperature in Celsius.
    pub motor_temp_celsius: f64,
    /// Fault code (0 = no fault). Vendor-specific.
    pub fault_code: u16,
    /// Timestamp in microseconds.
    pub timestamp_us: u64,
}

// ---------------------------------------------------------------------------
// Body-Frame Geometry
// ---------------------------------------------------------------------------

/// Physical geometry of the Predator system in the body-centered
/// coordinate frame.
///
/// Origin: center of torso (sternum/spine midpoint).
/// +X = forward (operator facing), +Y = left, +Z = up.
///
/// These dimensions are calibrated per-operator during plate carrier
/// fitting. Defaults based on ANSUR 50th percentile male.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyFrameGeometry {
    /// Front radar module position [x, y, z] in meters from torso center.
    /// Default: top of front armor plate, centered laterally.
    pub radar_front_offset: [f64; 3],
    /// Rear radar module position [x, y, z] in meters from torso center.
    /// Default: top of rear armor plate, centered laterally.
    pub radar_rear_offset: [f64; 3],
    /// Mast base height (Z) above torso center in meters.
    /// Where the Z-fold hinge attaches to the rear plate carrier.
    pub mast_base_height_m: f64,
    /// Fully extended mast length in meters.
    /// 3-segment Z-fold: ~167mm per segment, 500mm total.
    /// Must clear IHPS helmet + IR beacon + 50mm margin.
    pub mast_length_m: f64,
}

impl BodyFrameGeometry {
    /// Compute the turret position in body frame when mast is fully deployed.
    ///
    /// The mast is vertical, centered on the body X/Y plane.
    pub fn turret_position(&self) -> [f64; 3] {
        [0.0, 0.0, self.mast_base_height_m + self.mast_length_m]
    }

    /// Compute turret position at a given actual mast extension height.
    ///
    /// Used when the mast is partially deployed (deploying/stowing).
    pub fn turret_position_at_height(&self, mast_height_m: f64) -> [f64; 3] {
        [0.0, 0.0, self.mast_base_height_m + mast_height_m]
    }
}

impl Default for BodyFrameGeometry {
    /// ANSUR 50th percentile male defaults.
    fn default() -> Self {
        Self {
            // Front radar: top of front armor plate, centered.
            // +0.12m forward, 0.0 lateral, +0.25m above torso center.
            radar_front_offset: [0.12, 0.0, 0.25],
            // Rear radar: top of rear armor plate, centered.
            // -0.12m rearward, 0.0 lateral, +0.25m above torso center.
            radar_rear_offset: [-0.12, 0.0, 0.25],
            // Mast base at top of rear plate.
            mast_base_height_m: 0.25,
            // 3-segment Z-fold, 500mm total.
            // Clears IHPS + MS2000 strobe + 50mm margin.
            mast_length_m: 0.50,
        }
    }
}

impl BodyFrameGeometry {
    /// Compute the turret position in world frame given body orientation.
    ///
    /// When the mast stabilization is active, the turret stays close
    /// to vertical even as the operator moves. The `residual_pitch_deg`
    /// and `residual_roll_deg` represent how far from vertical the mast
    /// actually is after stabilization correction.
    ///
    /// When stabilization is perfect (residual = 0), this returns
    /// the same as `turret_position()` rotated into world frame.
    pub fn turret_position_world(
        &self,
        body_pitch_deg: f64,
        body_roll_deg: f64,
        residual_pitch_deg: f64,
        residual_roll_deg: f64,
    ) -> [f64; 3] {
        // Decomposition:
        // 1. Mast base is rigidly attached to body → rotates with body
        // 2. Mast extension is stabilized → only rotates by residual error
        // turret_world = R_body × base_offset + R_residual × mast_extension

        // Body rotation: R = Ry(pitch) × Rx(roll)
        let bp = body_pitch_deg.to_radians();
        let br = body_roll_deg.to_radians();
        let cbp = bp.cos();
        let sbp = bp.sin();
        let cbr = br.cos();
        let sbr = br.sin();

        // Mast base in body frame: [0, 0, mast_base_height]
        // R_body × [0, 0, h] = [sbp*h, -sbr*cbp*h, cbr*cbp*h]
        let h = self.mast_base_height_m;
        let base_x = sbp * h;
        let base_y = -sbr * cbp * h;
        let base_z = cbr * cbp * h;

        // Residual rotation: R = Ry(res_pitch) × Rx(res_roll)
        let rp = residual_pitch_deg.to_radians();
        let rr = residual_roll_deg.to_radians();
        let crp = rp.cos();
        let srp = rp.sin();
        let crr = rr.cos();
        let srr = rr.sin();

        // Mast extension in stabilized frame: [0, 0, mast_length]
        // R_residual × [0, 0, L] = [srp*L, -srr*crp*L, crr*crp*L]
        let l = self.mast_length_m;
        let ext_x = srp * l;
        let ext_y = -srr * crp * l;
        let ext_z = crr * crp * l;

        // Turret world position = body-rotated base + residual-rotated extension
        [base_x + ext_x, base_y + ext_y, base_z + ext_z]
    }

    /// Rotate a radar position from body frame to world frame.
    pub fn radar_position_world(
        &self,
        radar_offset: [f64; 3],
        body_pitch_deg: f64,
        body_roll_deg: f64,
    ) -> [f64; 3] {
        let bp = body_pitch_deg.to_radians();
        let br = body_roll_deg.to_radians();

        let cp = bp.cos();
        let sp = bp.sin();
        let cr = br.cos();
        let sr = br.sin();

        let x = cp * radar_offset[0] + sp * radar_offset[2];
        let y = sr * sp * radar_offset[0] + cr * radar_offset[1] - sr * cp * radar_offset[2];
        let z = -cr * sp * radar_offset[0] + sr * radar_offset[1] + cr * cp * radar_offset[2];

        [x, y, z]
    }
}

// ---------------------------------------------------------------------------
// Mast Stabilization
// ---------------------------------------------------------------------------

/// Mast stabilization operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MastStabilizationMode {
    /// Motors unpowered. Used when mast is stowed or system is in SILENT.
    Idle,
    /// Active self-leveling using IMU feedback. Normal operation.
    Stabilize,
    /// Manual pitch/roll setpoint for calibration and testing.
    Override,
}

impl std::fmt::Display for MastStabilizationMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle => write!(f, "IDLE"),
            Self::Stabilize => write!(f, "STABILIZE"),
            Self::Override => write!(f, "OVERRIDE"),
        }
    }
}

/// Command to the mast stabilization controller.
///
/// Published on `predator/mast/stabilization/command` by the
/// orchestrator. The stabilization controller runs a 100Hz PID
/// loop on pitch and roll axes using 2× Maxon EC 20 flat motors
/// with 3-stage GP 22 gearheads (~350:1, ~1.0 Nm output).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MastStabilizationCommand {
    /// Desired operating mode.
    pub mode: MastStabilizationMode,
    /// Manual pitch setpoint (degrees). Only used in Override mode.
    pub override_pitch_deg: f64,
    /// Manual roll setpoint (degrees). Only used in Override mode.
    pub override_roll_deg: f64,
    /// Timestamp in microseconds.
    pub timestamp_us: u64,
}

/// Status feedback from the mast stabilization controller.
///
/// Published on `predator/mast/stabilization/status` at ~100Hz.
/// The orchestrator uses `is_leveled` for engagement gating.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MastStabilizationStatus {
    /// Current mast base pitch (degrees from vertical). + = forward tilt.
    pub pitch_deg: f64,
    /// Current mast base roll (degrees from vertical). + = right tilt.
    pub roll_deg: f64,
    /// Residual pitch error after stabilization (degrees).
    pub pitch_error_deg: f64,
    /// Residual roll error after stabilization (degrees).
    pub roll_error_deg: f64,
    /// True when both pitch and roll residual error < leveling threshold.
    pub is_leveled: bool,
    /// Pitch axis motor current (amps). For thermal monitoring.
    pub motor_current_pitch_amps: f64,
    /// Roll axis motor current (amps). For thermal monitoring.
    pub motor_current_roll_amps: f64,
    /// Fault code (0 = no fault). Vendor-specific.
    pub fault_code: u16,
    /// Current operating mode.
    pub mode: MastStabilizationMode,
    /// Timestamp in microseconds.
    pub timestamp_us: u64,
}

/// Gimbal command — target bearing for the PID controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GimbalCommand {
    /// Target azimuth in degrees.
    pub azimuth_deg: f64,
    /// Target elevation in degrees.
    pub elevation_deg: f64,
    /// Maximum slew rate in degrees/second.
    pub slew_rate_dps: f64,
    /// Target range for scan pattern sizing.
    pub target_range_m: f64,
    /// Timestamp in microseconds.
    pub timestamp_us: u64,
}

/// Gimbal status published by the PID controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GimbalStatus {
    pub current_az_deg: f64,
    pub current_el_deg: f64,
    pub commanded_az_deg: f64,
    pub commanded_el_deg: f64,
    pub az_error_deg: f64,
    pub el_error_deg: f64,
    pub is_on_target: bool,
    pub safety_veto: bool,
    pub timestamp_us: u64,
}

// ---------------------------------------------------------------------------
// System State Message
// ---------------------------------------------------------------------------

/// System state broadcast from the orchestrator.
///
/// Published on `predator/system/state` whenever the state machine
/// transitions. Python components subscribe to this for coordination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStateMsg {
    /// Current system state.
    pub state: SystemState,
    /// Previous state (for transition logging).
    pub previous_state: SystemState,
    /// What triggered the transition.
    pub trigger: String,
    /// Whether radar emission is authorized.
    pub radar_authorized: bool,
    /// Whether laser engagement is authorized.
    pub engagement_authorized: bool,
    /// Whether the arm is deployed.
    pub arm_deployed: bool,
    /// Current threat score (0.0–1.0).
    pub threat_score: f64,
    /// Timestamp in microseconds.
    pub timestamp_us: u64,
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

/// Serialize a message to MessagePack bytes (binary, compact).
///
/// Use for timing-critical Zenoh topics to avoid JSON overhead.
pub fn to_msgpack<T: Serialize>(msg: &T) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    rmp_serde::to_vec_named(msg)
}

/// Deserialize a message from MessagePack bytes.
pub fn from_msgpack<'a, T: Deserialize<'a>>(bytes: &'a [u8]) -> Result<T, rmp_serde::decode::Error> {
    rmp_serde::from_slice(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer1_detection_roundtrip_msgpack() {
        let det = Layer1Detection {
            camera_id: 2,
            bearing_deg: 45.0,
            confidence: 0.92,
            timestamp_us: 1_000_000,
            elevation_deg: Some(-5.0),
            centroid_vy_degps: None,
        };
        let bytes = to_msgpack(&det).unwrap();
        let decoded: Layer1Detection = from_msgpack(&bytes).unwrap();
        assert_eq!(decoded.camera_id, 2);
        assert!((decoded.bearing_deg - 45.0).abs() < 1e-9);
        assert!((decoded.confidence - 0.92).abs() < 1e-9);
        // MessagePack is much smaller than JSON
        assert!(bytes.len() < 150, "msgpack should be compact: {} bytes", bytes.len());
    }

    #[test]
    fn test_system_state_serialization() {
        let msg = SystemStateMsg {
            state: SystemState::Tracking,
            previous_state: SystemState::RadarActive,
            trigger: "radar_acquisition".to_string(),
            radar_authorized: true,
            engagement_authorized: false,
            arm_deployed: true,
            threat_score: 0.65,
            timestamp_us: 2_000_000,
        };
        let bytes = to_msgpack(&msg).unwrap();
        let decoded: SystemStateMsg = from_msgpack(&bytes).unwrap();
        assert_eq!(decoded.state, SystemState::Tracking);
        assert_eq!(decoded.previous_state, SystemState::RadarActive);
        assert!(decoded.radar_authorized);
        assert!(!decoded.engagement_authorized);
    }

    #[test]
    fn test_pre_slew_mast_deploy() {
        let cmd = PreSlewCommand {
            bearing_deg: 90.0,
            timestamp_us: 500_000,
        };
        assert!((cmd.bearing_deg - 90.0).abs() < 1e-9);

        // Verify msgpack roundtrip
        let bytes = to_msgpack(&cmd).unwrap();
        let decoded: PreSlewCommand = from_msgpack(&bytes).unwrap();
        assert!((decoded.bearing_deg - 90.0).abs() < 1e-9);
    }

    #[test]
    fn test_arm_command_stow() {
        let cmd = ArmCommand {
            action: ArmAction::Stow,
            timestamp_us: 1_000_000,
        };
        let bytes = to_msgpack(&cmd).unwrap();
        let decoded: ArmCommand = from_msgpack(&bytes).unwrap();
        assert_eq!(decoded.action, ArmAction::Stow);
    }

    #[test]
    fn test_body_frame_geometry_defaults() {
        let geom = BodyFrameGeometry::default();
        // Front radar: forward, centered, at plate top
        assert!(geom.radar_front_offset[0] > 0.0, "front radar should be forward");
        assert!((geom.radar_front_offset[1]).abs() < 1e-9, "centered laterally");
        // Rear radar: rearward
        assert!(geom.radar_rear_offset[0] < 0.0, "rear radar should be rearward");
        // Mast: 3-segment, 0.50m
        assert!((geom.mast_length_m - 0.50).abs() < 1e-9);
        // Turret position
        let turret = geom.turret_position();
        assert!((turret[2] - 0.75).abs() < 1e-9, "turret at mast_base + mast_length");
        assert!((turret[0]).abs() < 1e-9, "turret centered X");
        assert!((turret[1]).abs() < 1e-9, "turret centered Y");
    }

    #[test]
    fn test_turret_position_at_partial_height() {
        let geom = BodyFrameGeometry::default();
        let partial = geom.turret_position_at_height(0.25);
        assert!((partial[2] - 0.50).abs() < 1e-9, "partial height = base + 0.25");
    }

    #[test]
    fn test_kinematic_primer_flag() {
        let primer = ThreatClassification {
            track_id: 1,
            threat_score: 0.72,
            is_primer: true,
            target_class: "rotary_uas".to_string(),
            flicker_confidence: 0.8,
            doppler_confidence: 0.0,
            timestamp_us: 3_000_000,
        };
        assert!(primer.is_primer);
        assert!((primer.doppler_confidence - 0.0).abs() < 1e-9);

        let full = ThreatClassification {
            is_primer: false,
            doppler_confidence: 0.85,
            ..primer
        };
        assert!(!full.is_primer);
        assert!(full.doppler_confidence > 0.0);
    }

    #[test]
    fn test_imu_feed_forward_has_roll() {
        let ff = ImuFeedForward {
            delta_az_deg: 1.0,
            delta_el_deg: 2.0,
            delta_roll_deg: 3.0,
            angular_rate_az_dps: 10.0,
            angular_rate_el_dps: 20.0,
            angular_rate_roll_dps: 30.0,
            is_valid: true,
        };
        let bytes = to_msgpack(&ff).unwrap();
        let decoded: ImuFeedForward = from_msgpack(&bytes).unwrap();
        assert!((decoded.delta_roll_deg - 3.0).abs() < 1e-9);
        assert!((decoded.angular_rate_roll_dps - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_mast_stabilization_command_roundtrip() {
        let cmd = MastStabilizationCommand {
            mode: MastStabilizationMode::Stabilize,
            override_pitch_deg: 0.0,
            override_roll_deg: 0.0,
            timestamp_us: 100_000,
        };
        let bytes = to_msgpack(&cmd).unwrap();
        let decoded: MastStabilizationCommand = from_msgpack(&bytes).unwrap();
        assert_eq!(decoded.mode, MastStabilizationMode::Stabilize);
    }

    #[test]
    fn test_mast_stabilization_status_leveled() {
        let status = MastStabilizationStatus {
            pitch_deg: 0.3,
            roll_deg: -0.2,
            pitch_error_deg: 0.3,
            roll_error_deg: 0.2,
            is_leveled: true,
            motor_current_pitch_amps: 0.05,
            motor_current_roll_amps: 0.04,
            fault_code: 0,
            mode: MastStabilizationMode::Stabilize,
            timestamp_us: 200_000,
        };
        assert!(status.is_leveled);
        assert_eq!(status.fault_code, 0);

        let bytes = to_msgpack(&status).unwrap();
        let decoded: MastStabilizationStatus = from_msgpack(&bytes).unwrap();
        assert!(decoded.is_leveled);
        assert_eq!(decoded.mode, MastStabilizationMode::Stabilize);
    }

    #[test]
    fn test_turret_position_world_zero_residual() {
        let geom = BodyFrameGeometry::default();
        // Zero body tilt + zero residual → same as static turret position
        let world = geom.turret_position_world(0.0, 0.0, 0.0, 0.0);
        let static_pos = geom.turret_position();
        assert!((world[0] - static_pos[0]).abs() < 1e-6, "X: {} vs {}", world[0], static_pos[0]);
        assert!((world[1] - static_pos[1]).abs() < 1e-6, "Y: {} vs {}", world[1], static_pos[1]);
        assert!((world[2] - static_pos[2]).abs() < 1e-6, "Z: {} vs {}", world[2], static_pos[2]);
    }
}
