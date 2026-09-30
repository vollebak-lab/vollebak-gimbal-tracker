# Predator — Layer-by-Layer Technical Specification

## Layer 1: Neuromorphic Passive Tripwire

### Purpose
Continuous passive surveillance using event cameras that detect propeller flicker signatures. Provides bearing-only cueing to Layer 2 radar. Zero RF emission.

### Components

#### `dvx_event_streamer.py` (418 lines)
- **Class**: `DvxEventStreamer`
- **Sensors**: 4× DVXplorer Micro (640×480, USB 3.1)
- **Threading**: 1 `CameraReaderThread` per camera (daemon) → `Queue(maxsize=10)` → main consumer
- **Batching**: 10 ms event batches
- **Voxelization**: `time_bin=100 µs` voxel grid → SpMiniUNet inference
- **Bearing**: Per-camera azimuth offset (0°/90°/180°/270°) + 55° FOV → global bearing
- **Failmode**: USB dropout → camera health monitoring + auto-reconnect stub

#### `event_aggregator.py` (185 lines)
- **Class**: `EventAggregator`
- **Function**: `compute_pixel_bearings(x_pixels, fov_h, az_offset)` → global bearing degrees
- **Output**: Enriched `AggregatedEventBatch` → Zenoh topic `predator/events/aggregated`

#### `propeller_detector.py` (537 lines)
- **Model**: `SpMiniUNet` (sparse 3D CNN via spconv)
- **Input**: Voxelized events | **Output**: Detection logits
- **Tracker**: `SimpleTracker` FSM: `IDLE → SEARCHING → TRACKING → LOST`
- **Thresholds**: `det_thresh_logit=1.0`, `centre_thresh_logit=0.0`, `max_misses=3`
- **Multi-cam**: All 4 cameras processed per cycle; detections tagged with `camera_id` + `bearing_deg`
- **Publishes**: `predator/layer1/detection` → `{camera_id, bearing_deg, confidence, timestamp_us}`

### Data Flow
```
DVXplorer USB → CameraReaderThread → Queue → Voxelize → SpMiniUNet → Detection
  → SimpleTracker FSM → DetectionAlert(camera_id, bearing, confidence)
  → Zenoh: predator/layer1/detection
```

---

## Layer 2: Active Radar

### Purpose
3D localization and micro-Doppler classification of airborne targets. Activated only on L1 cue (Silent-to-Active). Supports cognitive duty cycling.

### Components

#### `radar_interface.py` (375 lines)
- **ABC**: `RadarBackend` — `start()`, `stop()`, `get_frame()`, `get_doppler_time_series()`, `set_power_mode()`, `set_cfar_config()`
- **Impl**: `SimulatedRadarBackend` (all L2 currently simulated — `uhnder_bridge/` is EMPTY, blocked on S80 SDK)
- **Power Modes**: `RadarPowerMode` enum: `DEEP_SLEEP` (~50 mW), `SECTOR_SEARCH` (~7–9 W), `FULL_TRACK` (9.5 W)
- **CFAR Config**: `CFARConfig(pfa, guard_cells, training_cells)`
- **Dataclasses**:
  - `RadarDetection(range_m, azimuth_deg, elevation_deg, doppler_mps, rcs_dbsm, snr_db, hcr_margin_db, cfar_bypassed, point_count)`
  - `RadarFrame(detections[], timestamp_us, frame_id)`
  - `DopplerTimeSeries(timestamps_us, doppler_mps, range_m, azimuth_deg)` → feeds micro-Doppler

#### `micro_doppler_classifier.py` (312 lines)
- **Class**: `MicroDopplerClassifier`
- **Pipeline**: STFT (window=50 ms, overlap=0.75) → power spectrum → harmonic peak detection → blade rate
- **Classes**: `TargetClass(ROTARY_UAS, FIXED_WING, BIRD, CLUTTER, UNKNOWN)`
- **Harmonics**: Fundamental (50–500 Hz) + 2nd/3rd harmonic strength ratio → confidence
- **Latency**: ~500 ms (STFT window) — **kinematic primer bypasses this for fast engagement**

#### Uhnder S80 Hardware Specifications
- **Chip**: Uhnder S80, 76–81 GHz, PMCW-DCM architecture
- **Per chip**: 12 Tx, 16 Rx, 192 virtual channels
- **Arrays**: Front (2 chips, 384 virtual, eth0) + Rear (2 chips, 384 virtual, eth1)
- **System**: 4 chips, 768 virtual channels, 48 W total, 360° coverage
- **Beam steering**: Wide scan (10 Hz), Narrow track (50 Hz)

### Cognitive Duty Cycling
| State | Power Mode | Behavior |
|---|---|---|
| `SILENT` | `DEEP_SLEEP` | RF Tx/Rx power-gated, clock only (~50 mW) |
| `ALERT` | `SECTOR_SEARCH` | Aggressive wake, beam-steered to L1 bearing (~7–9 W) |
| `TRACKING+` | `FULL_TRACK` | All channels, max PRF, continuous waveform (9.5 W) |
| `BDA→SILENT` | `DEEP_SLEEP` | Return to standby |

---

## Layer 3: Fusion & Decision

### Purpose
Multi-target tracking, threat classification, engagement state management, and safety gating. This is the brain of the system.

### Tracking Pipeline (3-Layer Architecture)

```
┌─────────────────────────────────────────────┐
│  1. MOTION MODEL — IMM                      │
│     "How does the target move?"             │
│     4 parallel models: CV, CJ, MSM, STS    │
├─────────────────────────────────────────────┤
│  2. DATA ASSOCIATION — JPDA                 │
│     "Which measurement → which track?"      │
│     + Neuromorphic bearing as Bayesian prior│
├─────────────────────────────────────────────┤
│  3. HYPOTHESIS MANAGEMENT — MHT            │
│     "Across time, which tracks are real?"   │
│     N-scan-back pruning, swarm persistence  │
└─────────────────────────────────────────────┘
```

#### `imm_tracker.py` (NEW — 25 KB)
- **Class**: `IMMTrack` — 4-model bank running in parallel
- **Models**:
  - **CV** (Constant Velocity, σ_a=0.5): Cruise/transit
  - **CJ** (Constant Jerk): FPV kamikaze aggressive pitch
  - **MSM** (Move-Stop-Move): Hover/loiter
  - **STS** (Sprint-to-Stop): Rapid decel to observation hover
- **Cycle**: Mix → Predict → Update → Combine (Markov transition matrix)
- **Output**: Per-model probability weights for downstream threat scoring

#### `jpda_associator.py` (NEW — 17 KB)
- **Class**: `JPDAAssociator` — replaces Hungarian assignment entirely
- **Logic**: For each track, compute probability each validated measurement (within Mahalanobis gate) originated from true target vs. clutter
- **Neuromorphic fusion**: L1 bearing injected as Bayesian prior — measurements correlating with neuromorphic bearing get boosted association weight
- **CFAR bypass**: When neuromorphic detection corroborates radar return (same bearing ±10°), bypass CFAR threshold

#### `hypothesis_tracker.py` (NEW — 16 KB)
- **Class**: `MultipleHypothesisTracker` — wraps JPDA for temporal decisions
- **Purpose**: Deferred-decision track association for swarm scenarios
- **Pruning**: N-scan-back (configurable, default N=3), max 100 hypotheses
- **Activation**: Engages only when track density exceeds threshold

#### `aerial_target_tracker.py` (413 lines)
- **Class**: `AerialMultiTargetTracker`
- **Model**: 9-state EKF [x, y, z, vx, vy, vz, ax, ay, az] (legacy) or IMM (flag-gated)
- **Config**: `AerialTrackerConfig(dt=0.1, max_coast_frames=5, use_imm=True, use_jpda=True)`
- **Track lifecycle**: Spawn → coast (max frames) → prune
- **Output**: `TargetState(target_id, position, velocity, acceleration, rcs_dbsm, track_age_frames)`

#### `predator_state_machine.py` (532 lines)
- **Class**: `PredatorStateMachine`
- **States**: `SystemState(SILENT, ALERT, RADAR_ACTIVE, TRACKING, ENGAGEMENT, BDA)`
- **Config**: `FSMConfig(alert_confirm_count=2, high_conf_thresh=0.8, alert_timeout=5s, radar_search_timeout=15s, coast_timeout=3s, engagement_timeout=10s, bda_obs=5s)`
- **Cognitive duty cycling**: Emits `RadarPowerMode` commands on state transitions
- **Pre-slew**: On first detection → arm deploy + pre-slew to bearing
- **Engagement gate**: `threat_score ≥ 0.7` + arm + mast gate
- **BDA**: `flicker_ceased AND doppler_lost` → kill confirmed → SILENT + stow

#### `threat_classifier.py` (322 lines)
- **Class**: `ThreatClassifier`
- **Kinematic Primer**: Fast score without 500 ms STFT wait — `speed + closing_rate + altitude + flicker → immediate threat estimate`
- **Full Assessment**: Primer + Doppler classification → composite threat score
- **Weights**: speed=0.20, altitude=0.15, flicker=0.30, doppler=0.35 (redistributed in primer mode)
- **HCR Integration**: Small-RCS targets with >25 dB HCR margin near large reflectors → confidence boost
- **Threshold**: 0.7 for engagement authorization

#### `safety_manager.py` (164 lines)
- **Class**: `SafetyManager`
- **Method**: `check_engagement(bearing, elevation, range)` → `SafetyCheck(authorized, keep_out_clear, range_clear, elevation_clear)`

---

## Layer 4: Engagement

### Purpose
Gimbal slew, laser scan pattern generation, and Battle Damage Assessment.

#### `slew_to_cue.py` (136 lines)
- **Class**: `SlewToCue`
- **Logic**: Cartesian EKF track → spherical (az, el, range) + lead angle compensation
- **Output**: `GimbalCmd(azimuth_deg, elevation_deg, target_range_m)` → Rust gimbal PID

#### `lissajous_scanner.py` (176 lines)
- **Class**: `LissajousScanner`
- **Patterns**: Lissajous (primary), Rosette (alternative), Raster (fallback)
- **Purpose**: Distribute laser energy across target surface for maximum dwell coverage
- **Config**: `ScanConfig(num_points=100, angular_extent_deg=2.0)`

#### `bda_assessor.py` (181 lines)
- **Class**: `BDAAssessor`
- **Indicators**: Flicker cessation, Doppler loss, RCS trajectory (falling)
- **Outcomes**: `KILL_CONFIRMED` (all), `PROBABLE_KILL` (partial), `MISS` (none), `ASSESSMENT_TIMEOUT`
- **Timeline**: Post-laser → 5s observation window → multi-indicator fusion → verdict

---

## Layer 5: Cooperative (Stub → Active)

#### `cooperative_tracker.py` (NEW — 12 KB)
- **Class**: `CooperativeTrackerNode`
- **Purpose**: P2P leader election for multi-Predator networks
- **Logic**: Node with best target geometry + highest battery → `FULL_TRACK`; subordinates → `DEEP_SLEEP`
- **Transport**: Zenoh pub/sub (placeholder — Zenoh transport integration pending)

---

## IMU Subsystem (Cross-cutting)

#### `imu_provider.py` (534 lines)
- **Sources**: External BHI260AP (authority, 200 Hz, Zenoh) + 4× DVXplorer ICM-42688 (fallback)
- **Fusion**: External authority → DVX cross-validation → degraded mode
- **Motion classification**: Stationary (<0.3 m/s), Walking (<2.5 m/s), Running
- **Output**: `ImuState(quaternion, gyro, accel, heading, ego_velocity, timestamp)`

#### `motion_compensator.py` (255 lines)
- **Radar**: Body-to-world rotation + Doppler ego correction
- **Gimbal**: Feed-forward corrections (delta az/el/roll + rates)
- **Degraded**: Static cluster median Doppler subtraction (no IMU fallback)
