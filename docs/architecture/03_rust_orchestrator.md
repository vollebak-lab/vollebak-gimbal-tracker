# Predator — Rust Orchestrator & Message Bus

## Rust Workspace

```toml
[workspace]
members = [
    "crates/predator-messages",      # Shared Zenoh message types
    "crates/predator-orchestrator",   # Real-time event loop + SM
    "src/layer4_engagement/gimbal_controller",  # 200 Hz PID (src dir missing)
]
```

**Dependencies**: zenoh 1.0, tokio (full), serde + serde_json, rmp-serde 1.3, nalgebra 0.33, tracing, chrono.

---

## `predator-messages` Crate (lib.rs: 780 lines)

### Purpose
Shared binary-serializable Zenoh message types for Python↔Rust IPC.

### Serialization
- **Production**: MessagePack (via rmp-serde) — `to_msgpack()` / `from_msgpack()`
- **Debug**: JSON (via serde_json)

### Enums
| Enum | Variants |
|---|---|
| `SystemState` | Silent, Alert, RadarActive, Tracking, Engagement, Bda |
| `ArmAction` | Deploy, Stow |
| `ArmDeploymentState` | Stowed, Deploying, Deployed, Stowing, Fault |
| `MastStabilizationMode` | Idle, Stabilize, Override |

### Key Structs
| Struct | Fields |
|---|---|
| `Layer1Detection` | camera_id, bearing_deg, confidence, timestamp_us |
| `RadarTrackUpdate` | track_id, range_m, azimuth_deg, elevation_deg, speed_mps, radial_velocity_mps, altitude_m, rcs_dbsm |
| `ThreatClassification` | track_id, threat_score, is_primer, target_class, flicker_confidence, doppler_confidence |
| `PreSlewCommand` | bearing_deg |
| `ArmCommand` | action, timestamp |
| `ArmStatus` | deployment_state, is_stable, mast_height_m, joint_angles, deployment_progress, motor_current, motor_temp, fault_code |
| `GimbalCommand` | azimuth_deg, elevation_deg, slew_rate_dps, target_range_m |
| `GimbalStatus` | current/commanded az/el, errors, is_on_target, safety_veto |
| `SystemStateMsg` | state, previous_state, trigger, radar_authorized, engagement_authorized, arm_deployed, threat_score |
| `ImuFeedForward` | delta_az/el/roll_deg, angular_rate_az/el/roll_dps, is_valid |
| `MastStabilizationCommand` | mode, override_pitch/roll_deg |
| `MastStabilizationStatus` | pitch/roll_deg, pitch/roll_error_deg, is_leveled, motor_currents, fault_code, mode |
| `BodyFrameGeometry` | radar_front/rear_offset, mast_base_height_m=0.25, mast_length_m=0.50 |

### Body Frame Convention
- **Origin**: Torso center
- **+X**: Forward, **+Y**: Left, **+Z**: Up
- **Turret position**: `[0, 0, mast_base_height + mast_length]` = `[0, 0, 0.75]`
- **Defaults**: ANSUR 50th percentile

### Zenoh Topics (topics.rs)
| Topic | Direction | Content |
|---|---|---|
| `predator/layer1/detection` | Python → Rust | L1 bearing + confidence |
| `predator/radar/track` | Python → Rust | 3D track update |
| `predator/radar/classification` | Python → Rust | Threat score + class |
| `predator/imu/gimbal_feedforward` | Python → Rust | Feed-forward corrections |
| `predator/system/state` | Rust → Python | System state broadcast |
| `predator/engagement/pre_slew` | Rust → Python | Pre-slew bearing command |
| `predator/engagement/command` | Rust → Gimbal | Gimbal cue |
| `predator/engagement/status` | Gimbal → Rust | On-target status |
| `predator/arm/command` | Rust → Arm | Deploy/stow |
| `predator/arm/status` | Arm → Rust | Deployment state |
| `predator/mast/stabilization/command` | Rust → Mast | Stabilize/idle/override |
| `predator/mast/stabilization/status` | Mast → Rust | Level status + faults |

---

## `predator-orchestrator` Crate

### `main.rs` (408 lines) — Event Loop

**Architecture**:
```
Python Processes ──Zenoh──► Rust Orchestrator ──Zenoh──► Rust Gimbal PID
  (L1, L2, IMU)              (SM + Primer)               (200 Hz servo)
```

**Config**: `OrchestratorConfig { sm, primer, geometry, stabilization, tick_rate_hz=200, use_binary_payloads=true }`

**Subscribers**:
1. `LAYER1_DETECTION` → `sm.on_layer1_detection(camera_id, bearing, confidence, timestamp)`
2. `RADAR_TRACK` → `compute_kinematic_primer()` + `sm.on_radar_track_acquired/update()`
3. `RADAR_CLASSIFICATION` → `sm.on_threat_score_update()` (only if `!is_primer`)
4. `ARM_STATUS` → `sm.on_arm_status(state, stable, height)`
5. `MAST_STABILIZATION_STATUS` → `sm.on_mast_stabilization_status(leveled, fault, mode, residuals)`

**Publishers**: `SYSTEM_STATE`, `ENGAGEMENT_PRE_SLEW`, `ENGAGEMENT_COMMAND`, `ARM_COMMAND`

**Tick Loop**: 200 Hz via `tokio::time::sleep` → `sm.tick()` → `sm.drain_commands()` → publish all

**Deserialization**: Try MessagePack first → fallback to JSON (dual-mode for debug/production)

### `state_machine.rs` (1070 lines)

**Struct**: `PredatorStateMachine` — pure logic, no I/O. All Zenoh I/O handled by `main.rs`.

**Config** (`SmConfig`):
| Parameter | Default | Purpose |
|---|---|---|
| `alert_confirm_count` | 1 | Detections needed to confirm ALERT |
| `high_conf_thresh` | 0.8 | Fast-track threshold |
| `pre_slew_on_alert` | true | Pre-slew gimbal on first detection |
| `alert_timeout_s` | 5.0 | ALERT → SILENT if no confirmation |
| `radar_search_timeout_s` | 15.0 | RADAR_ACTIVE → SILENT if no acquisition |
| `coast_timeout_s` | 3.0 | TRACKING → SILENT if track lost |
| `engagement_timeout_s` | 10.0 | ENGAGEMENT → BDA timeout |
| `bda_observation_s` | 5.0 | BDA observation window |
| `min_threat_score` | 0.7 | Engagement authorization threshold |

**Inputs**: `on_layer1_detection`, `on_radar_track_acquired`, `on_radar_track_update`, `on_threat_score_update`, `on_arm_status`, `on_mast_stabilization_status`, `on_laser_fired`, `on_flicker_ceased`, `on_doppler_lost`

**Outputs** (`SmCommand`): `StateChange`, `ArmDeploy`, `PreSlew`, `ArmStow`, `RadarAuthorize`, `GimbalCue`

**Kill Chain**: `ENGAGEMENT → on_laser_fired → BDA → (flicker_ceased + doppler_lost) → kill_confirmed → SILENT + stow`

### `kinematic_primer.rs` (242 lines)

**Purpose**: Immediate threat score without waiting for 500 ms STFT micro-Doppler classification.

**Function**: `compute_kinematic_primer(speed, radial_vel, altitude, flicker_conf, &config) → PrimerResult`

**Weights**: speed=0.30, closing_rate=0.30, altitude=0.05, flicker=0.35

**Latency**: <1 µs execution

### `mast_stabilization.rs` (623 lines)

**Purpose**: 2-axis active stabilization of the mast-mounted turret.

**Hardware**: 2× Maxon EC20 flat (3 W) + GP22 3-stage gearhead (~350:1) → ~1.0 Nm per axis, 186 g total

**Bandwidth allocation**:
| Frequency | Controller | Rate |
|---|---|---|
| 0–5 Hz (body pitch/roll) | Mast stabilization PID | 100 Hz |
| 0–3 Hz (yaw) | Gimbal azimuth | 200 Hz |
| 0–20 Hz (tracking) | Gimbal PID | 200 Hz |
| 0–200 Hz (fine) | Fast Steering Mirror (FSM) | Hardware |

**PID**: `kp=3.0, ki=0.5, kd=0.15, max_integral=5.0, max_output_nm=1.0` with anti-windup clamping

**Faults**: `PITCH_OVERCURRENT(1)`, `ROLL_OVERCURRENT(2)`, `PITCH_STALL(3)`, `ROLL_STALL(4)`, `IMU_INVALID(5)`

**Leveled condition**: `mode==Stabilize AND fault==0 AND |pitch_err| < threshold AND |roll_err| < threshold`

### `parallax.rs` (355 lines)

**Purpose**: Radar-to-turret parallax correction (radar is chest-mounted, turret is mast-mounted 0.75 m above).

**Functions**:
- `radar_to_turret(radar_az, radar_el, range, radar_pos, turret_pos)` → static correction
- `correct_for_parallax_dynamic(+ body_pitch, body_roll, residuals)` → IMU-aware correction

**Impact by range**:
| Range | Parallax Error | Significance |
|---|---|---|
| 200 m | 0.14° | Negligible |
| 50 m | 0.57° | Moderate |
| 20 m | 1.43° | Significant |
| 10 m | 2.86° | **CRITICAL** |

**Fast path**: If residual < 0.1° and body tilt < 0.1° → use static correction (skip dynamic)

---

## Terminology

> **FSM** = **Fast Steering Mirror** (optical beam steering component on the gimbal), NOT Finite State Machine.
> The engagement pipeline state machine is referred to as the "state machine" or "engagement state machine."
