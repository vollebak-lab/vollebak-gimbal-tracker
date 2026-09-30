// ---
// VOLLEBAK_ENGINEERING_METADATA:
//   PROJECT_ID: PREDATOR-01
//   TRACK: ECOSYSTEM
//   PHASE: CHALLENGE_HUB
//   CONTRIBUTOR_ID: Antigravity AI
//   THE_DELTA: Async Zenoh orchestrator wiring state machine, kinematic primer, and all pipeline components via binary message bus
//   FAILURE_MODE: Python GIL contention between ML inference and state machine ticks adds 5-15ms jitter; no pipeline orchestrator existed
//   IP_STATUS: VOLLEBAK_PROPRIETARY
//   USER_FEEDBACK_REF: Latency Audit Gap 5 — No pipeline orchestrator
//   DEPENDENCIES: [zenoh, tokio, predator-messages, tracing]
// ---

//! # Predator Orchestrator
//!
//! Real-time async event loop that:
//!
//! 1. Subscribes to Python-published Zenoh topics (L1 detection, radar tracks, IMU)
//! 2. Drives the Rust state machine on each incoming message
//! 3. Runs the kinematic primer on radar track updates
//! 4. Publishes engagement commands (pre-slew, gimbal cue, arm deploy/stow)
//! 5. Publishes system state for Python components to subscribe to
//!
//! ## Architecture
//!
//! ```text
//! Python Processes ──Zenoh──► Rust Orchestrator ──Zenoh──► Rust Gimbal PID
//!   (L1, L2, IMU)              (SM + Primer)               (200Hz servo)
//! ```

pub mod kinematic_primer;
pub mod mast_stabilization;
pub mod parallax;
pub mod state_machine;

use kinematic_primer::{compute_kinematic_primer, PrimerConfig};
use mast_stabilization::MastStabilizationConfig;
use predator_messages::{
    topics, from_msgpack, to_msgpack, ArmStatus, BodyFrameGeometry, Layer1Detection,
    MastStabilizationStatus, RadarTrackUpdate, ThreatClassification,
};
use state_machine::{SmCommand, SmConfig, PredatorStateMachine};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{error, info, warn};

/// Orchestrator configuration.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// State machine configuration.
    pub sm: SmConfig,
    /// Kinematic primer configuration.
    pub primer: PrimerConfig,
    /// Body-frame geometry (radar positions, mast dimensions).
    pub geometry: BodyFrameGeometry,
    /// Mast stabilization controller configuration.
    pub stabilization: MastStabilizationConfig,
    /// Tick rate in Hz for the state machine timer loop.
    pub tick_rate_hz: u32,
    /// Whether to use MessagePack (true) or JSON (false) for Zenoh payloads.
    pub use_binary_payloads: bool,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            sm: SmConfig::default(),
            primer: PrimerConfig::default(),
            geometry: BodyFrameGeometry::default(),
            stabilization: MastStabilizationConfig::default(),
            tick_rate_hz: 200,
            use_binary_payloads: true,
        }
    }
}

/// The main orchestrator runtime.
///
/// Owns the Zenoh session, state machine, and kinematic primer.
/// All Zenoh I/O is handled here — the state machine is pure logic.
pub struct Orchestrator {
    config: OrchestratorConfig,
    sm: Arc<Mutex<PredatorStateMachine>>,
    /// Latest flicker confidence from L1 (for primer computation).
    latest_flicker_confidence: Arc<Mutex<f64>>,
}

impl Orchestrator {
    pub fn new(config: OrchestratorConfig) -> Self {
        let sm = PredatorStateMachine::new(config.sm.clone());
        Self {
            config,
            sm: Arc::new(Mutex::new(sm)),
            latest_flicker_confidence: Arc::new(Mutex::new(0.0)),
        }
    }

    /// Run the orchestrator event loop.
    ///
    /// This is the main entry point. Opens a Zenoh session, subscribes
    /// to all input topics, and runs the state machine tick loop.
    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Predator Orchestrator starting...");

        // Open Zenoh session
        let session = zenoh::open(zenoh::Config::default()).await?;
        let session = Arc::new(session);
        info!("Zenoh session opened");

        // Declare publishers
        let state_pub = session.declare_publisher(topics::SYSTEM_STATE).await?;
        let pre_slew_pub = session.declare_publisher(topics::ENGAGEMENT_PRE_SLEW).await?;
        let gimbal_pub = session.declare_publisher(topics::ENGAGEMENT_COMMAND).await?;
        let arm_pub = session.declare_publisher(topics::ARM_COMMAND).await?;

        // Subscribe to Layer 1 detections
        let sm_l1 = self.sm.clone();
        let sub_l1 = session.declare_subscriber(topics::LAYER1_DETECTION).await?;
        let gimbal_pub_l1 = session.declare_publisher(topics::ENGAGEMENT_COMMAND).await?;
        let state_pub_l1 = session.declare_publisher(topics::SYSTEM_STATE).await?;
        let arm_pub_l1 = session.declare_publisher(topics::ARM_COMMAND).await?;
        let pre_slew_pub_l1 = session.declare_publisher(topics::ENGAGEMENT_PRE_SLEW).await?;
        let use_binary_l1 = self.config.use_binary_payloads;
        let _l1_handle = tokio::spawn(async move {
            loop {
                match sub_l1.recv_async().await {
                    Ok(sample) => {
                        let payload = sample.payload().to_bytes();
                        // Try msgpack first, fall back to JSON
                        let det: Option<Layer1Detection> =
                            from_msgpack(&payload).ok().or_else(|| {
                                serde_json::from_slice(&payload).ok()
                            });

                        if let Some(det) = det {
                            let mut sm = sm_l1.lock().await;
                            sm.on_layer1_detection(
                                det.camera_id,
                                det.bearing_deg,
                                det.confidence,
                                det.timestamp_us,
                                det.elevation_deg,
                                det.centroid_vy_degps,
                            );

                            // Immediate command drain for L1 engagements
                            // (Optimization #4: bypass tick-wait latency for
                            // time-critical Waiter mode prosecutions)
                            if sm.is_l1_only_engagement() {
                                let commands = sm.drain_commands();
                                drop(sm); // Release lock before async I/O
                                for cmd in commands {
                                    let result = match &cmd {
                                        SmCommand::StateChange(msg) => {
                                            let p = if use_binary_l1 {
                                                to_msgpack(msg).unwrap_or_default()
                                            } else {
                                                serde_json::to_vec(msg).unwrap_or_default()
                                            };
                                            state_pub_l1.put(p).await
                                        }
                                        SmCommand::GimbalCue(gc) => {
                                            let p = if use_binary_l1 {
                                                to_msgpack(gc).unwrap_or_default()
                                            } else {
                                                serde_json::to_vec(gc).unwrap_or_default()
                                            };
                                            gimbal_pub_l1.put(p).await
                                        }
                                        SmCommand::ArmDeploy(ac) | SmCommand::ArmStow(ac) => {
                                            let p = if use_binary_l1 {
                                                to_msgpack(ac).unwrap_or_default()
                                            } else {
                                                serde_json::to_vec(ac).unwrap_or_default()
                                            };
                                            arm_pub_l1.put(p).await
                                        }
                                        SmCommand::PreSlew(ps) => {
                                            let p = if use_binary_l1 {
                                                to_msgpack(ps).unwrap_or_default()
                                            } else {
                                                serde_json::to_vec(ps).unwrap_or_default()
                                            };
                                            pre_slew_pub_l1.put(p).await
                                        }
                                        SmCommand::RadarAuthorize { .. } => {
                                            // No-op for L1 engagements (radar stays asleep)
                                            Ok(())
                                        }
                                    };
                                    if let Err(e) = result {
                                        error!("L1 immediate publish error: {}", e);
                                    }
                                }
                            }
                        } else {
                            warn!("Failed to deserialize L1 detection");
                        }
                    }
                    Err(e) => {
                        error!("L1 subscriber error: {}", e);
                        break;
                    }
                }
            }
        });

        // Subscribe to radar track updates
        let sm_radar = self.sm.clone();
        let primer_config = self.config.primer.clone();
        let flicker_ref = self.latest_flicker_confidence.clone();
        let sub_radar = session.declare_subscriber(topics::RADAR_TRACK).await?;
        let _radar_handle = tokio::spawn(async move {
            loop {
                match sub_radar.recv_async().await {
                    Ok(sample) => {
                        let payload = sample.payload().to_bytes();
                        let track: Option<RadarTrackUpdate> =
                            from_msgpack(&payload).ok().or_else(|| {
                                serde_json::from_slice(&payload).ok()
                            });

                        if let Some(track) = track {
                            let flicker = *flicker_ref.lock().await;

                            // Run kinematic primer
                            let primer = compute_kinematic_primer(
                                track.speed_mps,
                                track.radial_velocity_mps,
                                track.altitude_m,
                                flicker,
                                &primer_config,
                            );

                            let mut sm = sm_radar.lock().await;

                            // First radar track → acquire
                            if sm.state() == predator_messages::SystemState::RadarActive {
                                sm.on_radar_track_acquired(
                                    track.track_id,
                                    track.range_m,
                                    track.azimuth_deg,
                                    track.elevation_deg,
                                );
                            } else {
                                sm.on_radar_track_update(track.track_id);
                            }

                            // Feed primer score to state machine
                            sm.on_threat_score_update(primer.score);
                        }
                    }
                    Err(e) => {
                        error!("Radar subscriber error: {}", e);
                        break;
                    }
                }
            }
        });

        // Subscribe to threat classification (full assessment replaces primer)
        let sm_class = self.sm.clone();
        let sub_class = session.declare_subscriber(topics::RADAR_CLASSIFICATION).await?;
        let _class_handle = tokio::spawn(async move {
            loop {
                match sub_class.recv_async().await {
                    Ok(sample) => {
                        let payload = sample.payload().to_bytes();
                        let class: Option<ThreatClassification> =
                            from_msgpack(&payload).ok().or_else(|| {
                                serde_json::from_slice(&payload).ok()
                            });

                        if let Some(class) = class {
                            if !class.is_primer {
                                // Full assessment overrides primer
                                let mut sm = sm_class.lock().await;
                                sm.on_threat_score_update(class.threat_score);
                                info!(
                                    "Full assessment received: score={:.3}, class={}",
                                    class.threat_score, class.target_class
                                );
                            }
                        }
                    }
                    Err(e) => {
                        error!("Classification subscriber error: {}", e);
                        break;
                    }
                }
            }
        });

        // Subscribe to arm status feedback
        let sm_arm = self.sm.clone();
        let sub_arm = session.declare_subscriber(topics::ARM_STATUS).await?;
        let _arm_handle = tokio::spawn(async move {
            loop {
                match sub_arm.recv_async().await {
                    Ok(sample) => {
                        let payload = sample.payload().to_bytes();
                        let status: Option<ArmStatus> =
                            from_msgpack(&payload).ok().or_else(|| {
                                serde_json::from_slice(&payload).ok()
                            });

                        if let Some(status) = status {
                            let mut sm = sm_arm.lock().await;
                            sm.on_arm_status(
                                status.deployment_state,
                                status.is_stable,
                                status.mast_height_m,
                            );
                        } else {
                            warn!("Failed to deserialize arm status");
                        }
                    }
                    Err(e) => {
                        error!("Arm status subscriber error: {}", e);
                        break;
                    }
                }
            }
        });

        // Subscribe to mast stabilization status feedback
        let sm_stab = self.sm.clone();
        let sub_stab = session
            .declare_subscriber(topics::MAST_STABILIZATION_STATUS)
            .await?;
        let _stab_handle = tokio::spawn(async move {
            loop {
                match sub_stab.recv_async().await {
                    Ok(sample) => {
                        let payload = sample.payload().to_bytes();
                        let status: Option<MastStabilizationStatus> =
                            from_msgpack(&payload).ok().or_else(|| {
                                serde_json::from_slice(&payload).ok()
                            });

                        if let Some(status) = status {
                            let mut sm = sm_stab.lock().await;
                            sm.on_mast_stabilization_status(
                                status.is_leveled,
                                status.fault_code,
                                status.mode,
                                status.pitch_error_deg,
                                status.roll_error_deg,
                            );
                        } else {
                            warn!("Failed to deserialize stabilization status");
                        }
                    }
                    Err(e) => {
                        error!("Stabilization status subscriber error: {}", e);
                        break;
                    }
                }
            }
        });

        // Tick loop — runs at configured rate (default 200Hz)
        let tick_interval =
            std::time::Duration::from_micros(1_000_000 / self.config.tick_rate_hz as u64);
        let sm_tick = self.sm.clone();
        let use_binary = self.config.use_binary_payloads;

        info!(
            "Orchestrator running — tick rate {}Hz, binary payloads: {}",
            self.config.tick_rate_hz, use_binary
        );

        loop {
            tokio::time::sleep(tick_interval).await;

            let commands = {
                let mut sm = sm_tick.lock().await;
                sm.tick();
                sm.drain_commands()
            };

            // Publish all pending commands
            for cmd in commands {
                match cmd {
                    SmCommand::StateChange(msg) => {
                        let payload = if use_binary {
                            to_msgpack(&msg).unwrap_or_default()
                        } else {
                            serde_json::to_vec(&msg).unwrap_or_default()
                        };
                        if let Err(e) = state_pub.put(payload).await {
                            error!("Failed to publish state: {}", e);
                        }
                    }
                    SmCommand::PreSlew(cmd) => {
                        let payload = if use_binary {
                            to_msgpack(&cmd).unwrap_or_default()
                        } else {
                            serde_json::to_vec(&cmd).unwrap_or_default()
                        };
                        if let Err(e) = pre_slew_pub.put(payload).await {
                            error!("Failed to publish pre-slew: {}", e);
                        }
                    }
                    SmCommand::ArmDeploy(cmd) | SmCommand::ArmStow(cmd) => {
                        let payload = if use_binary {
                            to_msgpack(&cmd).unwrap_or_default()
                        } else {
                            serde_json::to_vec(&cmd).unwrap_or_default()
                        };
                        if let Err(e) = arm_pub.put(payload).await {
                            error!("Failed to publish arm command: {}", e);
                        }
                    }
                    SmCommand::GimbalCue(cmd) => {
                        let payload = if use_binary {
                            to_msgpack(&cmd).unwrap_or_default()
                        } else {
                            serde_json::to_vec(&cmd).unwrap_or_default()
                        };
                        if let Err(e) = gimbal_pub.put(payload).await {
                            error!("Failed to publish gimbal cue: {}", e);
                        }
                    }
                    SmCommand::RadarAuthorize { bearing_deg } => {
                        info!("Radar authorized at bearing {:.1}°", bearing_deg);
                        // Radar authorization is implicit via state change
                    }
                }
            }
        }
    }
}

fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    info!("Predator Orchestrator v0.1.0");
    info!("Language: Rust (Hard Rule: long-running daemon + soft real-time + Zenoh-native)");
    info!("");
    info!("Zenoh subscriptions:");
    info!("  ← {}", topics::LAYER1_DETECTION);
    info!("  ← {}", topics::RADAR_TRACK);
    info!("  ← {}", topics::RADAR_CLASSIFICATION);
    info!("  ← {}", topics::IMU_FEEDFORWARD);
    info!("");
    info!("Zenoh publications:");
    info!("  → {}", topics::SYSTEM_STATE);
    info!("  → {}", topics::ENGAGEMENT_PRE_SLEW);
    info!("  → {}", topics::ENGAGEMENT_COMMAND);
    info!("  → {}", topics::ARM_COMMAND);

    let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
    let orchestrator = Orchestrator::new(OrchestratorConfig::default());

    rt.block_on(async {
        if let Err(e) = orchestrator.run().await {
            error!("Orchestrator error: {}", e);
        }
    });
}
