// ---
// VOLLEBAK_ENGINEERING_METADATA:
//   PROJECT_ID: PREDATOR-01
//   TRACK: ECOSYSTEM
//   PHASE: CHALLENGE_HUB
//   CONTRIBUTOR_ID: Antigravity AI
//   THE_DELTA: Rust port of the Python PredatorStateMachine — deterministic sub-1ms state transitions without GIL contention
//   FAILURE_MODE: Python state machine contends with PyTorch inference threads for GIL, adding 5-15ms jitter on critical transitions
//   IP_STATUS: VOLLEBAK_PROPRIETARY
//   USER_FEEDBACK_REF: Language Selection Audit — Hard Rule violation
//   DEPENDENCIES: [predator-messages, tracing]
// ---

//! # Predator State Machine (Rust)
//!
//! Real-time engagement pipeline state machine ported from Python.
//! Runs inside the orchestrator's async event loop with sub-millisecond
//! transition latency and zero GIL contention.
//!
//! ## State Diagram
//!
//! ```text
//! SILENT → ALERT → RADAR_ACTIVE → TRACKING → ENGAGEMENT → BDA → SILENT
//!   ↑        │           ↑             │           │
//!   └────────┘           └─────────────┘           │
//!   (timeout/stow)       (coast timeout)           │
//!   ↑                                              │
//!   └──────────────────────────────────────────────┘
//!   (kill confirmed / bda timeout)
//! ```

use predator_messages::{
    ArmAction, ArmCommand, ArmDeploymentState, GimbalCommand,
    MastStabilizationMode, PreSlewCommand, SystemState, SystemStateMsg,
};
use std::time::{Duration, Instant};
use tracing::info;

/// Configuration for the state machine.
///
/// Note: "SM" = State Machine (software). Not to be confused with
/// "FSM" = Fast Steer Mirror (hardware optical element).
#[derive(Debug, Clone)]
pub struct SmConfig {
    /// Detections needed to confirm alert (default: 1).
    pub alert_confirm_count: usize,
    /// Confidence above which a single detection fast-tracks
    /// directly to RADAR_ACTIVE (skipping ALERT).
    pub high_confidence_threshold: f64,
    /// When true, emit arm deploy + Fast Steer Mirror pre-slew on first detection.
    pub pre_slew_on_alert: bool,
    /// Time before ALERT reverts to SILENT (seconds).
    pub alert_timeout: Duration,
    /// Time before RADAR_ACTIVE reverts to ALERT (seconds).
    pub radar_search_timeout: Duration,
    /// Time without radar updates before losing track (seconds).
    pub tracking_coast_timeout: Duration,
    /// Maximum engagement duration (seconds).
    pub engagement_timeout: Duration,
    /// Post-engagement BDA observation window (seconds).
    pub bda_observation: Duration,
    /// Minimum threat score to authorize engagement.
    pub min_threat_score: f64,

    // Waiter mode (L1-only passive engagement) parameters
    /// Whether Waiter mode auto-detection is enabled.
    pub waiter_mode_enabled: bool,
    /// Minimum confidence for L1-only engagement.
    pub l1_engagement_confidence: f64,
    /// Maximum range for L1-only engagement (meters).
    pub l1_max_range_m: f64,
    /// Operator camera height for range estimation (meters).
    pub operator_camera_height_m: f64,
}

impl Default for SmConfig {
    fn default() -> Self {
        Self {
            alert_confirm_count: 1,
            high_confidence_threshold: 0.8,
            pre_slew_on_alert: true,
            alert_timeout: Duration::from_secs(5),
            radar_search_timeout: Duration::from_secs(15),
            tracking_coast_timeout: Duration::from_secs(3),
            engagement_timeout: Duration::from_secs(10),
            bda_observation: Duration::from_secs(5),
            min_threat_score: 0.7,
            waiter_mode_enabled: true,
            l1_engagement_confidence: 0.95,
            l1_max_range_m: 50.0,
            operator_camera_height_m: 1.85,
        }
    }
}

/// State transition record.
#[derive(Debug, Clone)]
pub struct StateTransition {
    pub from_state: SystemState,
    pub to_state: SystemState,
    pub trigger: String,
    pub timestamp_us: u64,
}

/// Commands emitted by the state machine for the orchestrator to publish.
///
/// The state machine is a pure logic component — it does not own Zenoh
/// sessions. Instead, it returns commands that the orchestrator publishes.
#[derive(Debug, Clone)]
pub enum SmCommand {
    /// Publish a system state change.
    StateChange(SystemStateMsg),
    /// Deploy the arm to a shoulder.
    ArmDeploy(ArmCommand),
    /// Pre-slew the Fast Steer Mirror and arm.
    PreSlew(PreSlewCommand),
    /// Stow the arm.
    ArmStow(ArmCommand),
    /// Authorize radar emission.
    RadarAuthorize { bearing_deg: f64 },
    /// Send gimbal command to PID controller.
    GimbalCue(GimbalCommand),
}

/// The Predator engagement state machine.
///
/// Pure logic — no I/O, no async, no Zenoh. The orchestrator calls
/// input methods and collects emitted [`SmCommand`]s for publishing.
///
/// Note: "SM" = State Machine (software). "FSM" = Fast Steer Mirror (hardware).
pub struct PredatorStateMachine {
    config: SmConfig,
    state: SystemState,
    state_entry: Instant,

    // Alert accumulation
    alert_detections: Vec<Layer1DetectionLocal>,
    alert_camera_ids: std::collections::HashSet<u8>,

    // Pre-slew / mast state
    arm_deployed: bool,
    arm_deployment_confirmed: bool,
    arm_state: ArmDeploymentState,
    arm_stable: bool,
    mast_height_m: f64,
    pre_slew_bearing_deg: f64,

    // Mast stabilization state
    mast_leveled: bool,
    stabilization_fault: bool,
    stabilization_mode: MastStabilizationMode,
    stabilization_residual_pitch_deg: f64,
    stabilization_residual_roll_deg: f64,

    // Tracking state
    has_radar_track: bool,
    last_radar_update: Instant,
    current_threat_score: f64,

    // Engagement state
    engagement_start: Instant,
    laser_fired: bool,

    // BDA state
    bda_start: Instant,
    flicker_ceased: bool,
    doppler_lost: bool,
    l1_only_bda: bool,

    // Output buffer — drained by orchestrator after each call
    pending_commands: Vec<SmCommand>,

    // History
    transition_log: Vec<StateTransition>,
}

/// Local copy of detection data (avoids importing full message struct
/// into the FSM logic — keeps it pure).
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Layer1DetectionLocal {
    camera_id: u8,
    bearing_deg: f64,
    confidence: f64,
    timestamp_us: u64,
    elevation_deg: Option<f64>,
    centroid_vy_degps: Option<f64>,
}

impl PredatorStateMachine {
    /// Create a new state machine with the given configuration.
    pub fn new(config: SmConfig) -> Self {
        let now = Instant::now();
        Self {
            config,
            state: SystemState::Silent,
            state_entry: now,
            alert_detections: Vec::new(),
            alert_camera_ids: std::collections::HashSet::new(),
            arm_deployed: false,
            arm_deployment_confirmed: false,
            arm_state: ArmDeploymentState::Stowed,
            arm_stable: false,
            mast_height_m: 0.0,
            pre_slew_bearing_deg: 0.0,
            mast_leveled: false,
            stabilization_fault: false,
            stabilization_mode: MastStabilizationMode::Idle,
            stabilization_residual_pitch_deg: 0.0,
            stabilization_residual_roll_deg: 0.0,
            has_radar_track: false,
            last_radar_update: now,
            current_threat_score: 0.0,
            engagement_start: now,
            laser_fired: false,
            bda_start: now,
            flicker_ceased: false,
            doppler_lost: false,
            l1_only_bda: false,
            pending_commands: Vec::new(),
            transition_log: Vec::new(),
        }
    }

    /// Current system state.
    pub fn state(&self) -> SystemState {
        self.state
    }

    /// Whether radar emission is authorized.
    pub fn is_radar_authorized(&self) -> bool {
        matches!(
            self.state,
            SystemState::RadarActive
                | SystemState::Tracking
                | SystemState::Engagement
                | SystemState::Bda
        )
    }

    /// Whether laser engagement is authorized.
    pub fn is_engagement_authorized(&self) -> bool {
        matches!(
            self.state,
            SystemState::Engagement | SystemState::L1Engagement
        )
    }

    /// Whether the current engagement is L1-only (Waiter mode).
    pub fn is_l1_only_engagement(&self) -> bool {
        self.state == SystemState::L1Engagement || self.l1_only_bda
    }

    /// Whether the arm is currently deployed.
    pub fn is_arm_deployed(&self) -> bool {
        self.arm_deployed
    }

    /// Current threat score.
    pub fn threat_score(&self) -> f64 {
        self.current_threat_score
    }

    /// Drain all pending commands emitted since last drain.
    pub fn drain_commands(&mut self) -> Vec<SmCommand> {
        std::mem::take(&mut self.pending_commands)
    }

    /// Whether the arm is confirmed deployed and stable.
    pub fn is_arm_deployment_confirmed(&self) -> bool {
        self.arm_deployment_confirmed
    }

    /// Whether the mast stabilization reports leveled.
    pub fn is_mast_leveled(&self) -> bool {
        self.mast_leveled
    }

    /// Whether stabilization is in fault state.
    pub fn is_stabilization_faulted(&self) -> bool {
        self.stabilization_fault
    }

    /// Get stabilization residual errors for dynamic parallax.
    pub fn stabilization_residuals(&self) -> (f64, f64) {
        (self.stabilization_residual_pitch_deg, self.stabilization_residual_roll_deg)
    }

    /// Current arm deployment state.
    pub fn arm_deployment_state(&self) -> ArmDeploymentState {
        self.arm_state
    }

    /// Get the full transition log.
    pub fn transition_log(&self) -> &[StateTransition] {
        &self.transition_log
    }

    // -------------------------------------------------------------------
    // State transition
    // -------------------------------------------------------------------

    fn transition_to(&mut self, new_state: SystemState, trigger: &str) {
        let old_state = self.state;
        let now_us = self.state_entry.elapsed().as_micros() as u64;

        let transition = StateTransition {
            from_state: old_state,
            to_state: new_state,
            trigger: trigger.to_string(),
            timestamp_us: now_us,
        };

        info!(
            "STATE: {} → {} (trigger: {})",
            old_state, new_state, trigger
        );

        self.state = new_state;
        self.state_entry = Instant::now();
        self.transition_log.push(transition);

        // Emit state change command
        self.pending_commands.push(SmCommand::StateChange(SystemStateMsg {
            state: new_state,
            previous_state: old_state,
            trigger: trigger.to_string(),
            radar_authorized: self.is_radar_authorized(),
            engagement_authorized: self.is_engagement_authorized(),
            arm_deployed: self.arm_deployed,
            threat_score: self.current_threat_score,
            timestamp_us: now_us,
        }));
    }

    // -------------------------------------------------------------------
    // Input: Layer 1 detection
    // -------------------------------------------------------------------

    /// Process a Layer 1 neuromorphic detection.
    ///
    /// Implements:
    /// - **Waiter mode**: Auto-detect L1-only engagement when elevation,
    ///   confidence, and range conditions are met (below horizon, ≥0.95,
    ///   ≤50m). Transitions SILENT → L1Engagement directly, bypassing
    ///   radar entirely. Zero RF emission.
    /// - High confidence fast-track (skip ALERT → RADAR_ACTIVE)
    /// - Pre-slew on first detection (arm deploy + FSM coarse aim)
    /// - Single-detection confirmation (alert_confirm_count = 1)
    ///
    /// # Arguments
    ///
    /// * `camera_id` — Source camera identifier.
    /// * `bearing_deg` — Global bearing to detection centroid.
    /// * `confidence` — Detection confidence (0.0–1.0).
    /// * `timestamp_us` — Detection timestamp.
    /// * `elevation_deg` — Optional elevation angle (0°=horizon, neg=below).
    /// * `centroid_vy_degps` — Optional vertical angular rate (°/s, pos=ascending).
    pub fn on_layer1_detection(
        &mut self,
        camera_id: u8,
        bearing_deg: f64,
        confidence: f64,
        timestamp_us: u64,
        elevation_deg: Option<f64>,
        centroid_vy_degps: Option<f64>,
    ) {
        let det = Layer1DetectionLocal {
            camera_id,
            bearing_deg,
            confidence,
            timestamp_us,
            elevation_deg,
            centroid_vy_degps,
        };

        // ---------------------------------------------------------------
        // Waiter mode gate: L1-only engagement for close-range ambush drones
        //
        // Conditions (all must be true):
        //   1. waiter_mode_enabled in config
        //   2. Current state is Silent (cold start) or Alert (already alerted)
        //   3. Elevation data present AND below horizon (< 0°)
        //   4. Confidence >= l1_engagement_confidence (0.95)
        //   5. Geometric range estimate <= l1_max_range_m (50m)
        //
        // When met, transitions directly to L1Engagement, bypassing
        // ALERT/RADAR_ACTIVE/TRACKING. Radar stays deep-sleep.
        // ---------------------------------------------------------------
        if self.config.waiter_mode_enabled
            && matches!(self.state, SystemState::Silent | SystemState::Alert)
        {
            if let Some(el_deg) = elevation_deg {
                if el_deg < 0.0
                    && confidence >= self.config.l1_engagement_confidence
                {
                    // Geometric range: camera_height / tan(|elevation|)
                    let el_abs_rad = el_deg.abs().to_radians();
                    if el_abs_rad > 0.001 {
                        // Guard against near-horizon division
                        let range_est =
                            self.config.operator_camera_height_m / el_abs_rad.tan();

                        if range_est <= self.config.l1_max_range_m && range_est > 0.0 {
                            info!(
                                "WAITER MODE: conf={:.3}, el={:.1}°, range_est={:.1}m, \
                                 vy={:.1}°/s → L1_ENGAGEMENT",
                                confidence,
                                el_deg,
                                range_est,
                                centroid_vy_degps.unwrap_or(0.0),
                            );

                            // Pre-slew mast if not already deployed
                            if self.config.pre_slew_on_alert && !self.arm_deployed {
                                self.request_mast_deploy(bearing_deg);
                            }

                            self.l1_only_bda = false;
                            self.transition_to(
                                SystemState::L1Engagement,
                                "waiter_mode_auto_detect",
                            );

                            // Emit immediate gimbal cue for L1-sensed bearing/elevation
                            self.pending_commands.push(SmCommand::GimbalCue(
                                GimbalCommand {
                                    azimuth_deg: bearing_deg,
                                    elevation_deg: el_deg,
                                    slew_rate_dps: 180.0, // Max sweep rate
                                    target_range_m: range_est,
                                    timestamp_us,
                                },
                            ));

                            return; // Waiter fast-path exits here
                        }
                    }
                }
            }
        }

        // ---------------------------------------------------------------
        // Standard engagement path (no Waiter mode conditions met)
        // ---------------------------------------------------------------
        match self.state {
            SystemState::Silent => {
                self.alert_detections.clear();
                self.alert_camera_ids.clear();
                self.alert_detections.push(det);
                self.alert_camera_ids.insert(camera_id);

                // Pre-slew: begin arm deploy + FSM coarse aim
                if self.config.pre_slew_on_alert {
                    self.request_mast_deploy(bearing_deg);
                }

                // Fast-track: high confidence → skip ALERT → RADAR_ACTIVE
                if confidence >= self.config.high_confidence_threshold {
                    info!(
                        "HIGH-CONF fast-track: conf={:.3} >= {:.3} → RADAR_ACTIVE",
                        confidence, self.config.high_confidence_threshold
                    );
                    self.transition_to(SystemState::RadarActive, "high_confidence_detection");
                    self.pending_commands.push(SmCommand::RadarAuthorize {
                        bearing_deg,
                    });
                } else {
                    self.transition_to(SystemState::Alert, "layer1_detection");
                }
            }
            SystemState::Alert => {
                self.alert_detections.push(det);
                self.alert_camera_ids.insert(camera_id);

                // Update pre-slew bearing with latest
                if self.config.pre_slew_on_alert {
                    self.pre_slew_bearing_deg = bearing_deg;
                }

                // Check confirmation
                if self.alert_detections.len() >= self.config.alert_confirm_count {
                    let consensus = self.circular_mean_deg();
                    info!(
                        "CONFIRMED: {} detections, consensus bearing {:.1}°",
                        self.alert_detections.len(),
                        consensus
                    );
                    self.transition_to(SystemState::RadarActive, "confirmed_detection");
                }
            }
            _ => {
                // Detections in other states are logged but don't drive transitions
            }
        }
    }

    // -------------------------------------------------------------------
    // Input: Radar track
    // -------------------------------------------------------------------

    /// Process a confirmed radar track acquisition.
    pub fn on_radar_track_acquired(
        &mut self,
        track_id: u32,
        range_m: f64,
        azimuth_deg: f64,
        elevation_deg: f64,
    ) {
        if self.state == SystemState::RadarActive {
            self.has_radar_track = true;
            self.last_radar_update = Instant::now();

            info!(
                "RADAR TRACK: id={}, range={:.1}m, az={:.1}°, el={:.1}°",
                track_id, range_m, azimuth_deg, elevation_deg
            );

            self.transition_to(SystemState::Tracking, "radar_acquisition");

            // Emit gimbal cue command
            self.pending_commands.push(SmCommand::GimbalCue(GimbalCommand {
                azimuth_deg,
                elevation_deg,
                slew_rate_dps: 360.0,
                target_range_m: range_m,
                timestamp_us: Instant::now().elapsed().as_micros() as u64,
            }));
        }
    }

    /// Update tracking state with a new radar measurement.
    pub fn on_radar_track_update(&mut self, _track_id: u32) {
        if matches!(
            self.state,
            SystemState::Tracking | SystemState::Engagement
        ) {
            self.last_radar_update = Instant::now();
        }
    }

    // -------------------------------------------------------------------
    // Input: Threat classification
    // -------------------------------------------------------------------

    /// Process a threat score update (from kinematic primer or full assessment).
    ///
    /// Transitions TRACKING → ENGAGEMENT only when:
    /// 1. Threat score >= min_threat_score
    /// 2. Arm deployment is confirmed (deployed + stable)
    /// 3. Mast is leveled OR stabilization is faulted (degraded mode)
    ///
    /// In degraded mode (stabilization fault), engagement proceeds
    /// with gimbal-only compensation. A warning is logged.
    pub fn on_threat_score_update(&mut self, threat_score: f64) {
        self.current_threat_score = threat_score;

        let engagement_ready = self.arm_deployment_confirmed
            && (self.mast_leveled || self.stabilization_fault);

        if self.state == SystemState::Tracking
            && threat_score >= self.config.min_threat_score
            && engagement_ready
        {
            if self.stabilization_fault {
                info!(
                    "DEGRADED ENGAGEMENT: stabilization fault — gimbal-only compensation active. score={:.3}",
                    threat_score
                );
            }
            info!(
                "THREAT CONFIRMED: score={:.3} >= {:.3}, arm={}, leveled={}",
                threat_score, self.config.min_threat_score, self.arm_state, self.mast_leveled
            );
            self.transition_to(SystemState::Engagement, "threat_score_exceeded");
            self.engagement_start = Instant::now();
        } else if self.state == SystemState::Tracking
            && threat_score >= self.config.min_threat_score
            && !engagement_ready
        {
            info!(
                "THREAT SCORE {:.3} exceeds threshold but engagement blocked (arm={}, leveled={}, stab_fault={})",
                threat_score, self.arm_state, self.mast_leveled, self.stabilization_fault
            );
        }
    }

    // -------------------------------------------------------------------
    // Input: Arm status feedback
    // -------------------------------------------------------------------

    /// Process mast motor controller status feedback.
    ///
    /// Updates internal mast tracking and re-evaluates the engagement
    /// gate if the mast just became deployment-confirmed while a
    /// threat score was already above threshold.
    pub fn on_arm_status(
        &mut self,
        deployment_state: ArmDeploymentState,
        is_stable: bool,
        mast_height_m: f64,
    ) {
        let was_confirmed = self.arm_deployment_confirmed;
        self.arm_state = deployment_state;
        self.arm_stable = is_stable;
        self.mast_height_m = mast_height_m;

        // Mast is confirmed deployed when: state == Deployed AND stable
        self.arm_deployment_confirmed =
            deployment_state == ArmDeploymentState::Deployed && is_stable;

        // Handle fault
        if deployment_state == ArmDeploymentState::Fault {
            info!("MAST FAULT detected — blocking engagement");
            self.arm_deployment_confirmed = false;
        }

        // Re-evaluate engagement gate if mast just became confirmed
        // and threat score is already above threshold
        let engagement_ready = self.arm_deployment_confirmed
            && (self.mast_leveled || self.stabilization_fault);

        if !was_confirmed
            && engagement_ready
            && self.state == SystemState::Tracking
            && self.current_threat_score >= self.config.min_threat_score
        {
            info!(
                "MAST DEPLOYED + STABLE — releasing engagement gate (score={:.3}, height={:.3}m, leveled={})",
                self.current_threat_score, mast_height_m, self.mast_leveled
            );
            self.transition_to(SystemState::Engagement, "arm_deployment_confirmed");
            self.engagement_start = Instant::now();
        }
    }

    // -------------------------------------------------------------------
    // Input: Mast stabilization status
    // -------------------------------------------------------------------

    /// Process mast stabilization controller status feedback.
    ///
    /// Updates leveling state and re-evaluates the engagement gate.
    /// In degraded mode (fault), engagement proceeds with gimbal-only
    /// compensation — a warning is logged but engagement is not blocked.
    pub fn on_mast_stabilization_status(
        &mut self,
        is_leveled: bool,
        fault_code: u16,
        mode: MastStabilizationMode,
        residual_pitch_deg: f64,
        residual_roll_deg: f64,
    ) {
        let was_leveled = self.mast_leveled;
        self.mast_leveled = is_leveled;
        self.stabilization_fault = fault_code != 0;
        self.stabilization_mode = mode;
        self.stabilization_residual_pitch_deg = residual_pitch_deg;
        self.stabilization_residual_roll_deg = residual_roll_deg;

        if fault_code != 0 && !self.stabilization_fault {
            info!(
                "STABILIZATION FAULT: code={} — switching to degraded gimbal-only mode",
                fault_code
            );
        }

        // Re-evaluate engagement gate if leveling just became confirmed
        let engagement_ready = self.arm_deployment_confirmed
            && (self.mast_leveled || self.stabilization_fault);

        if !was_leveled
            && engagement_ready
            && self.state == SystemState::Tracking
            && self.current_threat_score >= self.config.min_threat_score
        {
            info!(
                "MAST LEVELED — releasing engagement gate (score={:.3}, leveled={}, degraded={})",
                self.current_threat_score, self.mast_leveled, self.stabilization_fault
            );
            self.transition_to(SystemState::Engagement, "mast_leveled_confirmed");
            self.engagement_start = Instant::now();
        }
    }

    // -------------------------------------------------------------------
    // Input: Engagement events
    // -------------------------------------------------------------------

    /// Record that the laser has been fired.
    pub fn on_laser_fired(&mut self) {
        if self.state == SystemState::Engagement {
            self.laser_fired = true;
            self.transition_to(SystemState::Bda, "laser_fire_command");
            self.bda_start = Instant::now();
            self.flicker_ceased = false;
            self.doppler_lost = false;
        }
    }

    /// Layer 1 reports propeller flicker has stopped.
    pub fn on_flicker_ceased(&mut self) {
        self.flicker_ceased = true;
    }

    /// Layer 2 reports micro-Doppler signature lost.
    pub fn on_doppler_lost(&mut self) {
        self.doppler_lost = true;
    }

    // -------------------------------------------------------------------
    // Tick — called on each orchestrator cycle
    // -------------------------------------------------------------------

    /// Process timeouts and automatic state transitions.
    ///
    /// Must be called at the orchestrator's tick rate (typically 200Hz).
    pub fn tick(&mut self) {
        let elapsed = self.state_entry.elapsed();

        match self.state {
            SystemState::Alert => {
                if elapsed > self.config.alert_timeout {
                    // False positive — stow arm
                    self.stow_arm();
                    self.transition_to(SystemState::Silent, "alert_timeout");
                }
            }
            SystemState::RadarActive => {
                if elapsed > self.config.radar_search_timeout {
                    self.transition_to(SystemState::Alert, "radar_search_timeout");
                }
            }
            SystemState::Tracking => {
                let coast = self.last_radar_update.elapsed();
                if coast > self.config.tracking_coast_timeout {
                    self.has_radar_track = false;
                    self.transition_to(SystemState::RadarActive, "track_coast_timeout");
                }
            }
            SystemState::Engagement => {
                if elapsed > self.config.engagement_timeout {
                    self.transition_to(SystemState::Bda, "engagement_timeout");
                    self.bda_start = Instant::now();
                }
            }
            SystemState::L1Engagement => {
                // L1-only engagement timeout (reuse engagement_timeout)
                if elapsed > self.config.engagement_timeout {
                    self.l1_only_bda = true;
                    self.transition_to(SystemState::Bda, "l1_engagement_timeout");
                    self.bda_start = Instant::now();
                }
            }
            SystemState::Bda => {
                if self.l1_only_bda {
                    // L1-only BDA: flicker cessation alone = kill
                    if self.flicker_ceased {
                        info!("BDA (L1-ONLY): KILL CONFIRMED — flicker ceased");
                        self.stow_arm();
                        self.l1_only_bda = false;
                        self.transition_to(SystemState::Silent, "l1_kill_confirmed");
                    } else if elapsed > self.config.bda_observation {
                        info!("BDA (L1-ONLY): TIMEOUT — flicker_ceased={}", self.flicker_ceased);
                        self.stow_arm();
                        self.l1_only_bda = false;
                        self.transition_to(SystemState::Silent, "l1_bda_timeout");
                    }
                } else {
                    // Standard BDA: flicker + doppler
                    if self.flicker_ceased && self.doppler_lost {
                        info!("BDA: KILL CONFIRMED — flicker ceased + Doppler lost");
                        self.stow_arm();
                        self.transition_to(SystemState::Silent, "kill_confirmed");
                    } else if elapsed > self.config.bda_observation {
                        if self.flicker_ceased || self.doppler_lost {
                            info!("BDA: PROBABLE KILL — partial signature loss");
                        } else {
                            info!("BDA: MISS — no signature change observed");
                        }
                        self.stow_arm();
                        self.transition_to(SystemState::Silent, "bda_timeout");
                    }
                }
            }
            SystemState::Silent => {} // No timeouts in SILENT
        }
    }

    // -------------------------------------------------------------------
    // Mast deploy / stow
    // -------------------------------------------------------------------

    fn request_mast_deploy(&mut self, bearing_deg: f64) {
        self.arm_deployed = true;
        self.pre_slew_bearing_deg = bearing_deg;

        info!(
            "MAST DEPLOY: extending vertical, Fast Steer Mirror coarse aim → {:.1}°",
            bearing_deg
        );

        let now_us = Instant::now().elapsed().as_micros() as u64;

        // Emit mast deploy command
        self.pending_commands.push(SmCommand::ArmDeploy(ArmCommand {
            action: ArmAction::Deploy,
            timestamp_us: now_us,
        }));

        // Emit Fast Steer Mirror pre-slew command
        self.pending_commands.push(SmCommand::PreSlew(PreSlewCommand {
            bearing_deg,
            timestamp_us: now_us,
        }));
    }

    fn stow_arm(&mut self) {
        if self.arm_deployed {
            self.arm_deployed = false;
            self.mast_height_m = 0.0;
            info!("MAST: retracting to Z-fold stow");

            self.pending_commands.push(SmCommand::ArmStow(ArmCommand {
                action: ArmAction::Stow,
                timestamp_us: Instant::now().elapsed().as_micros() as u64,
            }));
        }
    }

    // -------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------

    fn circular_mean_deg(&self) -> f64 {
        if self.alert_detections.is_empty() {
            return 0.0;
        }
        let (sin_sum, cos_sum) = self.alert_detections.iter().fold(
            (0.0_f64, 0.0_f64),
            |(s, c), d| {
                let rad = d.bearing_deg.to_radians();
                (s + rad.sin(), c + rad.cos())
            },
        );
        let n = self.alert_detections.len() as f64;
        let mean_rad = (sin_sum / n).atan2(cos_sum / n);
        mean_rad.to_degrees().rem_euclid(360.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use predator_messages::{ArmDeploymentState, MastStabilizationMode};

    /// Create a state machine with arm deployment pre-confirmed
    /// (simulates the arm motor controller reporting Deployed+Stable).
    /// Most tests need this to exercise the full kill chain.
    fn make_sm() -> PredatorStateMachine {
        let mut sm = PredatorStateMachine::new(SmConfig::default());
        // Pre-confirm arm deployment so engagement gate is open
        sm.on_arm_status(ArmDeploymentState::Deployed, true, 0.50);
        // Pre-confirm mast stabilization leveled
        sm.on_mast_stabilization_status(true, 0, MastStabilizationMode::Stabilize, 0.1, 0.1);
        sm
    }

    /// Create a state machine WITHOUT arm deployment confirmed.
    fn make_sm_no_arm() -> PredatorStateMachine {
        PredatorStateMachine::new(SmConfig::default())
    }

    #[test]
    fn test_initial_state_is_silent() {
        let sm = make_sm_no_arm();
        assert_eq!(sm.state(), SystemState::Silent);
        assert!(!sm.is_radar_authorized());
        assert!(!sm.is_arm_deployed());
        assert!(!sm.is_arm_deployment_confirmed());
        assert_eq!(sm.arm_deployment_state(), ArmDeploymentState::Stowed);
    }

    #[test]
    fn test_low_confidence_triggers_alert() {
        let mut sm = make_sm();
        sm.on_layer1_detection(0, 45.0, 0.5, 1_000, None, None);
        assert_eq!(sm.state(), SystemState::Alert);
        assert!(sm.is_arm_deployed()); // Pre-slew deploys arm
    }

    #[test]
    fn test_high_confidence_fast_tracks() {
        let mut sm = make_sm();
        sm.on_layer1_detection(0, 45.0, 0.9, 1_000, None, None);
        assert_eq!(sm.state(), SystemState::RadarActive);
        assert!(sm.is_radar_authorized());
        assert!(sm.is_arm_deployed());
    }

    #[test]
    fn test_mast_pre_slew_right_bearing() {
        let mut sm = make_sm();
        sm.on_layer1_detection(0, 90.0, 0.5, 1_000, None, None);
        let commands = sm.drain_commands();
        let pre_slew = commands.iter().find(|c| matches!(c, SmCommand::PreSlew(_)));
        assert!(pre_slew.is_some());
        if let Some(SmCommand::PreSlew(cmd)) = pre_slew {
            assert!((cmd.bearing_deg - 90.0).abs() < 1e-9);
        }
        // Mast deploy command should also be present
        assert!(commands.iter().any(|c| matches!(c, SmCommand::ArmDeploy(_))));
    }

    #[test]
    fn test_mast_pre_slew_rear_bearing() {
        let mut sm = make_sm();
        sm.on_layer1_detection(0, 270.0, 0.5, 1_000, None, None);
        let commands = sm.drain_commands();
        let pre_slew = commands.iter().find(|c| matches!(c, SmCommand::PreSlew(_)));
        assert!(pre_slew.is_some());
        if let Some(SmCommand::PreSlew(cmd)) = pre_slew {
            assert!((cmd.bearing_deg - 270.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_alert_timeout_stows_arm() {
        let mut sm = PredatorStateMachine::new(SmConfig {
            alert_timeout: Duration::from_millis(1),
            ..Default::default()
        });
        sm.on_layer1_detection(0, 45.0, 0.5, 1_000, None, None);
        assert_eq!(sm.state(), SystemState::Alert);
        assert!(sm.is_arm_deployed());

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(5));
        sm.tick();
        assert_eq!(sm.state(), SystemState::Silent);
        assert!(!sm.is_arm_deployed());
    }

    #[test]
    fn test_radar_track_triggers_tracking() {
        let mut sm = make_sm();
        // High-confidence fast-track to RADAR_ACTIVE
        sm.on_layer1_detection(0, 45.0, 0.9, 1_000, None, None);
        assert_eq!(sm.state(), SystemState::RadarActive);

        // Radar acquires track
        sm.on_radar_track_acquired(1, 150.0, 45.0, 10.0);
        assert_eq!(sm.state(), SystemState::Tracking);
    }

    #[test]
    fn test_alert_confirm_then_radar() {
        let mut sm = make_sm();
        // Low confidence → ALERT
        sm.on_layer1_detection(0, 45.0, 0.5, 1_000, None, None);
        assert_eq!(sm.state(), SystemState::Alert);

        // Second detection confirms (count=1, so first in ALERT confirms)
        sm.on_layer1_detection(1, 46.0, 0.6, 2_000, None, None);
        assert_eq!(sm.state(), SystemState::RadarActive);
    }

    // -------------------------------------------------------------------
    // ARM DEPLOYMENT GATE TESTS
    // -------------------------------------------------------------------

    #[test]
    fn test_threat_score_blocked_without_arm_confirmation() {
        let mut sm = make_sm_no_arm();
        sm.on_layer1_detection(0, 45.0, 0.9, 1_000, None, None);
        sm.on_radar_track_acquired(1, 100.0, 45.0, 5.0);
        assert_eq!(sm.state(), SystemState::Tracking);

        // Threat score exceeds threshold but arm is NOT confirmed
        sm.on_threat_score_update(0.85);
        assert_eq!(
            sm.state(),
            SystemState::Tracking,
            "Engagement MUST be blocked when arm is not confirmed deployed+stable"
        );
    }

    #[test]
    fn test_threat_score_triggers_engagement_with_arm() {
        let mut sm = make_sm(); // arm pre-confirmed
        sm.on_layer1_detection(0, 45.0, 0.9, 1_000, None, None);
        sm.on_radar_track_acquired(1, 100.0, 45.0, 5.0);
        assert_eq!(sm.state(), SystemState::Tracking);

        sm.on_threat_score_update(0.85);
        assert_eq!(sm.state(), SystemState::Engagement);
    }

    #[test]
    fn test_arm_confirmation_releases_gate() {
        let mut sm = make_sm_no_arm();
        // Pre-set stabilization as leveled (arm not yet confirmed)
        sm.on_mast_stabilization_status(true, 0, MastStabilizationMode::Stabilize, 0.1, 0.1);
        sm.on_layer1_detection(0, 45.0, 0.9, 1_000, None, None);
        sm.on_radar_track_acquired(1, 100.0, 45.0, 5.0);
        sm.on_threat_score_update(0.85);
        // Blocked — arm not confirmed
        assert_eq!(sm.state(), SystemState::Tracking);

        // Mast reports deployed + stable → gate releases
        sm.on_arm_status(ArmDeploymentState::Deployed, true, 0.50);
        assert_eq!(
            sm.state(),
            SystemState::Engagement,
            "Engagement should trigger once arm confirms deployed+stable"
        );
    }

    #[test]
    fn test_arm_deploying_does_not_release_gate() {
        let mut sm = make_sm_no_arm();
        sm.on_layer1_detection(0, 45.0, 0.9, 1_000, None, None);
        sm.on_radar_track_acquired(1, 100.0, 45.0, 5.0);
        sm.on_threat_score_update(0.85);

        // Mast reports deploying (not yet deployed)
        sm.on_arm_status(ArmDeploymentState::Deploying, false, 0.15);
        assert_eq!(sm.state(), SystemState::Tracking);
        assert!(!sm.is_arm_deployment_confirmed());
    }

    #[test]
    fn test_arm_deployed_but_unstable_blocks_gate() {
        let mut sm = make_sm_no_arm();
        sm.on_layer1_detection(0, 45.0, 0.9, 1_000, None, None);
        sm.on_radar_track_acquired(1, 100.0, 45.0, 5.0);
        sm.on_threat_score_update(0.85);

        // Mast reports deployed but NOT stable (still settling)
        sm.on_arm_status(ArmDeploymentState::Deployed, false, 0.50);
        assert_eq!(
            sm.state(),
            SystemState::Tracking,
            "Deployed but unstable arm must block engagement"
        );
        assert!(!sm.is_arm_deployment_confirmed());
    }

    #[test]
    fn test_arm_fault_blocks_engagement() {
        let mut sm = make_sm(); // start with confirmed arm
        assert!(sm.is_arm_deployment_confirmed());

        // Mast reports fault
        sm.on_arm_status(ArmDeploymentState::Fault, false, 0.0);
        assert!(!sm.is_arm_deployment_confirmed());
        assert_eq!(sm.arm_deployment_state(), ArmDeploymentState::Fault);
    }

    #[test]
    fn test_arm_status_stowed() {
        let mut sm = make_sm_no_arm();
        assert_eq!(sm.arm_deployment_state(), ArmDeploymentState::Stowed);
        assert!(!sm.is_arm_deployment_confirmed());

        // Report stowed explicitly
        sm.on_arm_status(ArmDeploymentState::Stowed, true, 0.0);
        assert!(!sm.is_arm_deployment_confirmed());
    }

    // -------------------------------------------------------------------
    // KILL CHAIN TESTS (with arm gate)
    // -------------------------------------------------------------------

    #[test]
    fn test_laser_fire_triggers_bda() {
        let mut sm = make_sm();
        sm.on_layer1_detection(0, 45.0, 0.9, 1_000, None, None);
        sm.on_radar_track_acquired(1, 100.0, 45.0, 5.0);
        sm.on_threat_score_update(0.85);
        assert_eq!(sm.state(), SystemState::Engagement);

        sm.on_laser_fired();
        assert_eq!(sm.state(), SystemState::Bda);
    }

    #[test]
    fn test_kill_confirmed() {
        let mut sm = make_sm();
        sm.on_layer1_detection(0, 45.0, 0.9, 1_000, None, None);
        sm.on_radar_track_acquired(1, 100.0, 45.0, 5.0);
        sm.on_threat_score_update(0.85);
        sm.on_laser_fired();
        assert_eq!(sm.state(), SystemState::Bda);

        sm.on_flicker_ceased();
        sm.on_doppler_lost();
        sm.tick();
        assert_eq!(sm.state(), SystemState::Silent);
        assert!(!sm.is_arm_deployed());
    }

    #[test]
    fn test_full_kill_chain_commands() {
        let mut sm = make_sm();

        // Detection → pre-slew + state change commands
        sm.on_layer1_detection(0, 90.0, 0.9, 1_000, None, None);
        let cmds = sm.drain_commands();
        assert!(cmds.iter().any(|c| matches!(c, SmCommand::ArmDeploy(_))));
        assert!(cmds.iter().any(|c| matches!(c, SmCommand::PreSlew(_))));
        assert!(cmds.iter().any(|c| matches!(c, SmCommand::StateChange(_))));
        assert!(cmds.iter().any(|c| matches!(c, SmCommand::RadarAuthorize { .. })));

        // Radar acquisition → gimbal cue
        sm.on_radar_track_acquired(1, 100.0, 90.0, 5.0);
        let cmds = sm.drain_commands();
        assert!(cmds.iter().any(|c| matches!(c, SmCommand::GimbalCue(_))));

        // Engagement → BDA → kill → stow
        sm.on_threat_score_update(0.85);
        sm.drain_commands();
        sm.on_laser_fired();
        sm.drain_commands();
        sm.on_flicker_ceased();
        sm.on_doppler_lost();
        sm.tick();
        let cmds = sm.drain_commands();
        assert!(cmds.iter().any(|c| matches!(c, SmCommand::ArmStow(_))));
    }

    // -------------------------------------------------------------------
    // MAST STABILIZATION GATE TESTS
    // -------------------------------------------------------------------

    #[test]
    fn test_not_leveled_blocks_engagement() {
        let mut sm = make_sm_no_arm();
        // Arm confirmed but stabilization NOT leveled
        sm.on_arm_status(ArmDeploymentState::Deployed, true, 0.50);
        // Stabilization not leveled
        sm.on_mast_stabilization_status(false, 0, MastStabilizationMode::Stabilize, 5.0, 3.0);

        sm.on_layer1_detection(0, 45.0, 0.9, 1_000, None, None);
        sm.on_radar_track_acquired(1, 100.0, 45.0, 5.0);
        sm.on_threat_score_update(0.85);

        assert_eq!(
            sm.state(),
            SystemState::Tracking,
            "Not-leveled stabilization should block engagement"
        );
    }

    #[test]
    fn test_degraded_mode_allows_engagement() {
        let mut sm = make_sm_no_arm();
        // Arm confirmed
        sm.on_arm_status(ArmDeploymentState::Deployed, true, 0.50);
        // Stabilization FAULTED → degraded mode allows engagement
        sm.on_mast_stabilization_status(false, 1, MastStabilizationMode::Stabilize, 10.0, 5.0);

        sm.on_layer1_detection(0, 45.0, 0.9, 1_000, None, None);
        sm.on_radar_track_acquired(1, 100.0, 45.0, 5.0);
        sm.on_threat_score_update(0.85);

        assert_eq!(
            sm.state(),
            SystemState::Engagement,
            "Degraded mode (stabilization fault) should allow engagement"
        );
    }

    #[test]
    fn test_leveling_releases_gate() {
        let mut sm = make_sm_no_arm();
        // Arm confirmed
        sm.on_arm_status(ArmDeploymentState::Deployed, true, 0.50);
        // Not leveled yet
        sm.on_mast_stabilization_status(false, 0, MastStabilizationMode::Stabilize, 5.0, 3.0);

        sm.on_layer1_detection(0, 45.0, 0.9, 1_000, None, None);
        sm.on_radar_track_acquired(1, 100.0, 45.0, 5.0);
        sm.on_threat_score_update(0.85);
        assert_eq!(sm.state(), SystemState::Tracking);

        // Now stabilization reports leveled → gate should release
        sm.on_mast_stabilization_status(true, 0, MastStabilizationMode::Stabilize, 0.3, 0.2);
        assert_eq!(
            sm.state(),
            SystemState::Engagement,
            "Leveling should release the engagement gate"
        );
    }

    #[test]
    fn test_stabilization_residuals_exposed() {
        let mut sm = make_sm();
        sm.on_mast_stabilization_status(true, 0, MastStabilizationMode::Stabilize, 0.4, 0.3);
        let (rp, rr) = sm.stabilization_residuals();
        assert!((rp - 0.4).abs() < 1e-9);
        assert!((rr - 0.3).abs() < 1e-9);
        assert!(sm.is_mast_leveled());
        assert!(!sm.is_stabilization_faulted());
    }

    // =======================================================================
    // Waiter Mode (L1-Only Engagement) Tests
    // =======================================================================

    /// Helper: detection params that satisfy all Waiter mode conditions.
    /// - elevation: -10° (below horizon)
    /// - confidence: 0.98 (above 0.95 threshold)
    /// - Geometric range at 1.85m camera height, -10°:
    ///   range = 1.85 / tan(10°) ≈ 10.5m (well within 50m)
    fn waiter_detection() -> (u8, f64, f64, u64, Option<f64>, Option<f64>) {
        (0, 45.0, 0.98, 1_000, Some(-10.0), Some(5.0))
    }

    #[test]
    fn test_waiter_mode_triggers_l1_engagement_from_silent() {
        let mut sm = make_sm();
        let (cam, brg, conf, ts, el, vy) = waiter_detection();
        sm.on_layer1_detection(cam, brg, conf, ts, el, vy);
        assert_eq!(sm.state(), SystemState::L1Engagement);
        assert!(sm.is_l1_only_engagement());
        assert!(sm.is_engagement_authorized());
    }

    #[test]
    fn test_waiter_mode_triggers_l1_engagement_from_alert() {
        let mut sm = make_sm();
        // First: low confidence → ALERT
        sm.on_layer1_detection(0, 45.0, 0.4, 500, None, None);
        assert_eq!(sm.state(), SystemState::Alert);

        // Second: waiter-qualifying detection while in ALERT
        let (cam, brg, conf, ts, el, vy) = waiter_detection();
        sm.on_layer1_detection(cam, brg, conf, ts, el, vy);
        assert_eq!(sm.state(), SystemState::L1Engagement);
    }

    #[test]
    fn test_waiter_mode_rejects_above_horizon() {
        let mut sm = make_sm();
        // Elevation +5° (above horizon) → should NOT trigger Waiter mode
        sm.on_layer1_detection(0, 45.0, 0.98, 1_000, Some(5.0), Some(0.0));
        // Falls through to standard path: high conf → RadarActive
        assert_eq!(sm.state(), SystemState::RadarActive);
        assert!(!sm.is_l1_only_engagement());
    }

    #[test]
    fn test_waiter_mode_rejects_far_range() {
        let mut sm = make_sm();
        // Elevation -1° → range = 1.85 / tan(1°) ≈ 106m → exceeds 50m
        sm.on_layer1_detection(0, 45.0, 0.98, 1_000, Some(-1.0), None);
        // Falls through to standard path
        assert_eq!(sm.state(), SystemState::RadarActive);
        assert!(!sm.is_l1_only_engagement());
    }

    #[test]
    fn test_waiter_mode_rejects_near_horizon_division_guard() {
        let mut sm = make_sm();
        // Elevation -0.001° → near-horizon guard (el_abs_rad < 0.001)
        sm.on_layer1_detection(0, 45.0, 0.98, 1_000, Some(-0.0005), None);
        // Should NOT trigger Waiter (division guard), falls to standard
        assert_eq!(sm.state(), SystemState::RadarActive);
    }

    #[test]
    fn test_waiter_mode_rejects_low_confidence() {
        let mut sm = make_sm();
        // Below horizon, close range, but confidence 0.80 < 0.95
        sm.on_layer1_detection(0, 45.0, 0.80, 1_000, Some(-10.0), Some(5.0));
        // Falls to standard: 0.80 >= high_confidence_threshold (0.8) → RadarActive
        assert_eq!(sm.state(), SystemState::RadarActive);
        assert!(!sm.is_l1_only_engagement());
    }

    #[test]
    fn test_waiter_mode_disabled_falls_to_standard() {
        let mut sm = PredatorStateMachine::new(SmConfig {
            waiter_mode_enabled: false,
            ..Default::default()
        });
        // Pre-set arm as deployed + confirmed for standard path
        sm.on_arm_status(ArmDeploymentState::Deployed, true, 0.5);

        let (cam, brg, conf, ts, el, vy) = waiter_detection();
        sm.on_layer1_detection(cam, brg, conf, ts, el, vy);
        // With waiter_mode disabled, high confidence → standard fast-track
        assert_eq!(sm.state(), SystemState::RadarActive);
        assert!(!sm.is_l1_only_engagement());
    }

    #[test]
    fn test_waiter_mode_no_elevation_falls_to_standard() {
        let mut sm = make_sm();
        // High confidence but no elevation data → standard path
        sm.on_layer1_detection(0, 45.0, 0.98, 1_000, None, None);
        assert_eq!(sm.state(), SystemState::RadarActive);
        assert!(!sm.is_l1_only_engagement());
    }

    #[test]
    fn test_waiter_mode_emits_gimbal_cue() {
        let mut sm = make_sm();
        let (cam, brg, conf, ts, el, vy) = waiter_detection();
        sm.on_layer1_detection(cam, brg, conf, ts, el, vy);
        assert_eq!(sm.state(), SystemState::L1Engagement);

        let commands = sm.drain_commands();
        let gimbal_cue = commands.iter().find(|c| matches!(c, SmCommand::GimbalCue(_)));
        assert!(gimbal_cue.is_some(), "L1Engagement must emit GimbalCue");

        // Verify gimbal command parameters
        if let Some(SmCommand::GimbalCue(gc)) = gimbal_cue {
            assert!((gc.azimuth_deg - 45.0).abs() < 1e-9, "Azimuth must match bearing");
            assert!((gc.elevation_deg - (-10.0)).abs() < 1e-9, "Elevation must match L1");
            assert!((gc.slew_rate_dps - 180.0).abs() < 1e-9, "Max sweep rate");
            assert!(gc.target_range_m > 0.0 && gc.target_range_m <= 50.0, "Range within bounds");
        }
    }

    #[test]
    fn test_l1_only_bda_kill_confirmed() {
        let mut sm = make_sm();
        let (cam, brg, conf, ts, el, vy) = waiter_detection();
        sm.on_layer1_detection(cam, brg, conf, ts, el, vy);
        assert_eq!(sm.state(), SystemState::L1Engagement);

        // Simulate engagement timeout → BDA
        sm.config.engagement_timeout = Duration::from_millis(1);
        std::thread::sleep(Duration::from_millis(5));
        sm.tick();
        assert_eq!(sm.state(), SystemState::Bda);
        assert!(sm.is_l1_only_engagement(), "BDA should be L1-only mode");

        // Flicker cessation alone = kill confirmed in L1-only BDA
        sm.on_flicker_ceased();
        sm.tick();
        assert_eq!(sm.state(), SystemState::Silent, "L1 BDA: flicker cessation = kill");
    }

    #[test]
    fn test_l1_only_bda_timeout() {
        let mut sm = make_sm();
        let (cam, brg, conf, ts, el, vy) = waiter_detection();
        sm.on_layer1_detection(cam, brg, conf, ts, el, vy);
        assert_eq!(sm.state(), SystemState::L1Engagement);

        // Force L1Engagement → BDA via timeout
        sm.config.engagement_timeout = Duration::from_millis(1);
        std::thread::sleep(Duration::from_millis(5));
        sm.tick();
        assert_eq!(sm.state(), SystemState::Bda);

        // No flicker cessation → BDA times out
        sm.config.bda_observation = Duration::from_millis(1);
        std::thread::sleep(Duration::from_millis(5));
        sm.tick();
        assert_eq!(sm.state(), SystemState::Silent, "L1 BDA timeout → SILENT");
    }
}
