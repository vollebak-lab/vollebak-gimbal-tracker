# Predator — Order of Operations & Data Flow

## Sensor-to-Engagement Pipeline

This document formalizes the complete order of operations from initial passive detection through engagement and kill assessment.

---

## Phase 1: Passive Surveillance (SILENT State)

**Active components**: Layer 1 only. Radar in `DEEP_SLEEP`. Zero RF emission.

```
1. DVXplorer Micro cameras continuously stream events (10 ms batches)
2. CameraReaderThread (1 per camera, daemon) → Queue(maxsize=10)
3. Events voxelized (time_bin=100 µs) into sparse 3D tensor
4. SpMiniUNet inference → detection logits
5. SimpleTracker FSM per camera: IDLE → SEARCHING → TRACKING
6. If detection logit > 1.0:
   → DetectionAlert(camera_id, bearing_deg, confidence, timestamp_us)
   → Published to Zenoh: predator/layer1/detection
```

**Latency budget**: ~5 ms (event batch) + ~2 ms (inference) = ~7 ms total

---

## Phase 2: Alert & Radar Wake (ALERT / RADAR_ACTIVE)

**Trigger**: L1 detection received by Rust orchestrator.

### Low-Confidence Path (confidence < 0.8)
```
1. Orchestrator receives L1 detection on LAYER1_DETECTION topic
2. SM.on_layer1_detection(camera_id, bearing, confidence, timestamp)
3. State: SILENT → ALERT
4. Radar command: DEEP_SLEEP → SECTOR_SEARCH(bearing, ±15° sector)
5. Arm deploy requested, pre-slew gimbal to bearing
6. Wait for alert_confirm_count detections (default=2) OR multi-camera consensus
7. On confirmation: ALERT → RADAR_ACTIVE
8. Radar: SECTOR_SEARCH → FULL_TRACK
9. Timeout: 5s with no confirmation → revert to SILENT + DEEP_SLEEP
```

### High-Confidence Fast Track (confidence ≥ 0.8)
```
1. Orchestrator receives L1 detection with confidence ≥ 0.8
2. FAST TRACK: SILENT → RADAR_ACTIVE (skip ALERT entirely)
3. Radar: DEEP_SLEEP → SECTOR_SEARCH at L1 bearing
4. Simultaneously: ArmDeploy + PreSlew + RadarAuthorize
5. All three commands emitted in single tick cycle
```

---

## Phase 3: Radar Acquisition (RADAR_ACTIVE → TRACKING)

**Active components**: L1 (continuous) + L2 (radar at SECTOR_SEARCH or FULL_TRACK)

```
1. SimulatedRadarBackend (or S80 via uhnder_bridge) generates RadarFrame
2. Each RadarFrame contains: [RadarDetection(range, az, el, doppler, rcs, snr, hcr_margin)]
3. Motion compensator applies ego-motion correction:
   a. ImuProvider → body-to-world rotation matrix
   b. Doppler ego-velocity subtraction
   c. Degraded fallback: static cluster median Doppler
4. Corrected detections fed to AerialMultiTargetTracker.update()
5. Tracking pipeline (if use_imm=True, use_jpda=True):
   a. IMM predict: 4 models (CV, CJ, MSM, STS) predict independently
   b. JPDA associate: probabilistic measurement-to-track assignment
      - Neuromorphic bearing injected as Bayesian prior
      - CFAR-bypassed detections get higher initial weight
   c. IMM update: each model updated with weighted innovation
   d. IMM combine: model probabilities mixed via Markov chain
   e. MHT: hypothesis tree maintained for swarm scenarios
6. First track → published to Zenoh: predator/radar/track
7. Orchestrator receives RadarTrackUpdate:
   a. compute_kinematic_primer(speed, radial_vel, altitude, flicker_conf)
   b. SM.on_radar_track_acquired(track_id, range, az, el)
   c. SM.on_threat_score_update(primer.score)
   d. State: RADAR_ACTIVE → TRACKING
8. Micro-Doppler classifier begins STFT processing (500 ms window)
```

---

## Phase 4: Threat Assessment (TRACKING)

**Active components**: L1 + L2 + L3 tracking + micro-Doppler + threat classifier

```
1. Continuous radar track updates → SM.on_radar_track_update()
2. After ~500 ms: MicroDopplerClassifier.classify(DopplerTimeSeries)
   → ClassificationResult(target_class, confidence)
3. ThreatClassifier.assess() computes full composite score:
   - Flicker confidence (0.30 weight)
   - Doppler classification (0.35 weight)
   - Speed score (0.20 weight)
   - Altitude band score (0.15 weight)
   - HCR margin boost (if applicable)
4. ThreatClassification published → Zenoh: predator/radar/classification
5. Orchestrator receives full assessment (is_primer=false)
   → SM.on_threat_score_update(full_score) — overrides kinematic primer
6. Parallax correction computed:
   - radar_to_turret() for static geometry
   - correct_for_parallax_dynamic() if IMU data available
7. If threat_score ≥ 0.7 AND engagement gate satisfied:
   → State: TRACKING → ENGAGEMENT
```

**Engagement Gate**:
```
engagement_authorized = (
    arm_deployment_confirmed (state=Deployed, is_stable=true)
    AND (mast_leveled OR stabilization_fault)
)
```

---

## Phase 5: Engagement (ENGAGEMENT)

**Active components**: All layers + gimbal + laser

```
1. SlewToCue.compute_command(x, y, z) → GimbalCmd(az, el, range)
   - Applies lead angle compensation for fast movers
   - Parallax-corrected coordinates
2. GimbalCmd published → Zenoh: predator/engagement/command
3. Rust gimbal PID controller tracks at 200 Hz
4. IMU feed-forward provides inertial stabilization:
   - ImuProvider → GimbalFeedForward(delta_az, delta_el, rates)
   - Low-pass filtered at 20 Hz cutoff
5. Mast stabilization controller (100 Hz):
   - Body pitch/roll compensation via Maxon EC20 motors
   - Leveling threshold: ±0.5°
6. When gimbal reports is_on_target:
   - LissajousScanner.generate_pattern() → scan points
   - Laser fires along Lissajous/Rosette/Raster pattern
   - Energy distributed across target surface for max dwell
7. On laser fire → SM.on_laser_fired()
8. State: ENGAGEMENT → BDA
9. Engagement timeout: 10s → auto-transition to BDA
```

---

## Phase 6: Battle Damage Assessment (BDA)

```
1. Post-laser observation window: 5 seconds
2. BDAAssessor.assess(flicker_trend, doppler_trend, rcs_trajectory)
3. Three kill indicators monitored:
   a. Flicker cessation — propeller signature disappears
   b. Doppler loss — rotor micro-Doppler signature gone
   c. RCS trajectory — RCS decreasing (target falling)
4. Outcomes:
   - KILL_CONFIRMED: all 3 indicators positive
   - PROBABLE_KILL: 2 of 3 indicators
   - MISS: 0–1 indicators
   - ASSESSMENT_TIMEOUT: window expired
5. On kill confirmed:
   - SM.on_flicker_ceased() + SM.on_doppler_lost()
   - State: BDA → SILENT
   - Arm stow command emitted
   - Radar: FULL_TRACK → DEEP_SLEEP
6. On timeout/miss:
   - State: BDA → SILENT
   - Arm stow, radar sleep
   - System returns to passive surveillance
```

---

## Complete Timing Budget

| Phase | Duration | Bottleneck |
|---|---|---|
| L1 detection | ~7 ms | SpMiniUNet inference |
| Radar wake (DEEP_SLEEP → SECTOR_SEARCH) | ~50–100 ms (configurable) | S80 hardware wake latency |
| Radar acquisition | 100–500 ms | First detection at beam-steered bearing |
| Kinematic primer | <1 µs | Pure arithmetic |
| Micro-Doppler STFT | ~500 ms | STFT window accumulation |
| Full threat assessment | <1 ms | Score computation |
| Gimbal slew | 100–500 ms | Depends on angular distance |
| Engagement dwell | 1–10 s | Configurable, target-dependent |
| BDA observation | 5 s | Fixed observation window |
| **Total (best case, fast track)** | **~1.2 s** | L1 → laser on target |
| **Total (worst case, full pipeline)** | **~12 s** | Full confirmation + STFT + engagement |

---

## Known Technical Debt

| Item | Status | Blocker |
|---|---|---|
| `uhnder_bridge/` EMPTY | Blocked | S80 SDK delivery (VLB-52 blocked on VLB-46) |
| Layer 5 cooperative | Stub → basic impl | Zenoh transport integration pending |
| `gimbal_controller` crate | Declared in Cargo.toml, src dir missing | Hardware not yet integrated |
| Python SM parallel to Rust SM | Potential drift | Both exist; Rust is authority in production |
| DvxImuSource quaternion | Hardcoded identity | No onboard orientation estimation |
| Cognitive power allocation | Stub in radar_interface | Requires S80 SDK for Tx power control |
| HCR margin, CFAR config | Simulated only | Real values require S80 SDK |

---

## Test Coverage

| Suite | Count | Framework |
|---|---|---|
| Python unit tests | 48 | pytest |
| Rust unit tests | ~45 | `#[cfg(test)]` in-crate |
| **Total** | **~93** | |

### Python Test Classes
| Class | Tests | Coverage |
|---|---|---|
| TestPredatorStateMachine | 13 | State transitions, fast track, arm gate, kill chain |
| TestKinematicPrimer | 6 | Fast/slow mover, closing rate, full assessment |
| TestImuProvider | 5 | Degraded mode, rotation, quaternion |
| TestMotionCompensator | 4 | Passthrough, empty, degraded, gimbal |
| TestAerialTracker | 4 | Spawn, persist, coast, prune |
| TestSafetyManager | 5 | Safe, keep-out, range, elevation |
| TestMicroDopplerClassifier | 2 | Rotary UAS, clutter |
| TestSlewToCue | 2 | Command generation |
| TestLissajousScanner | 2 | Pattern generation |
| TestSimulatedRadar | 2 | Frame generation |
| TestEventAggregator | 2 | Bearing computation |
