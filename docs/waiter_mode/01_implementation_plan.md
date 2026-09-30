# L1-Only Engagement Path ("Waiter Mode") — v3 FINAL

Fully passive engagement pipeline bypassing radar. Detects, targets, and prosecutes close-range ambush drones using neuromorphic cameras + Q-switched pulsed laser vertical sweep. Zero RF emission. ~400 ms total kill chain.

## Design Decisions (User Approved)

- **Activation**: Auto-detect only from geometry (below-horizon + range ≤ 50 m + confidence ≥ 0.95)
- **Max range**: 50 m
- **Confidence**: 0.95 minimum
- **Engagement**: Vertical sweep at max gimbal rate, not precision pointing
- **Laser physics**: Q-switched nanosecond pulses → mechanical lattice shatter (not thermal dwell). Single pulse entering drone camera lens = permanent CMOS damage via 100,000× optical gain

## Pulsed Laser Engagement Model

The Thor Dynamics laser operates as a Q-switched pulsed system:

| Parameter | Value |
|---|---|
| Average power | 2–5 W |
| Pulse width | ~10 ns (nanosecond) |
| Peak power | Megawatt-class (energy compression) |
| Damage mechanism | Dielectric breakdown → lattice shatter → line damage → sensor kill |
| LIDT (pulsed, CMOS) | 78.9 mJ/cm² |
| Optical gain (drone lens) | Up to 100,000× (lens acts as energy funnel) |
| Damage progression | Point damage → line damage → sensor kill |

**Engagement implication**: At <50 m, a single pulse crossing the drone's camera lens causes irreversible CMOS damage. Sweep strategy maximizes angular coverage probability, NOT thermal dwell accumulation.

### Sweep Timing (Revised for Pulsed Physics)

| Parameter | Value |
|---|---|
| Sweep rate | 180°/s (gimbal max — speed is king) |
| Passes | 2–3 (probability coverage, not dwell) |
| 10° sweep time | 55 ms/pass |
| Total engagement | ~150–200 ms |
| **Full kill chain** | **~400 ms** (7 ms detect + 200 ms slew + 200 ms sweep) |

---

## Proposed Changes

### Layer 1 — Neuromorphic Detection

#### [MODIFY] [event_aggregator.py](file:///c:/Users/snowd/OneDrive/Documents/Vollebak/predator/src/layer1_neuromorphic/event_aggregator.py)

Add elevation computation from pixel Y (same math as azimuth from pixel X).

- Add `compute_pixel_elevations(y_pixels, fov_v_deg, elevation_offset_deg)` method
- Add `resolution_h: int = 480` to `__init__`
- Add `global_elevations_deg` and `centroid_elevation_deg` to `AggregatedEventBatch`
- Modify `aggregate()` to compute elevation alongside bearing

#### [MODIFY] [propeller_detector.py](file:///c:/Users/snowd/OneDrive/Documents/Vollebak/predator/src/layer1_neuromorphic/propeller_detector.py)

Add elevation + centroid velocity tracking to `DetectionAlert`.

- Add `elevation_deg`, `centroid_vx_degps`, `centroid_vy_degps`, `bounding_box` to `DetectionAlert`
- Add velocity tracking in `PerCameraTracker` (Δcentroid/Δt across sequential TRACK frames)
- Add `fov_v_deg`, `elevation_offset_deg` params to `PerCameraTracker`

#### [MODIFY] [dvxplorer_array.yaml](file:///c:/Users/snowd/OneDrive/Documents/Vollebak/predator/config/dvxplorer_array.yaml)

Add `fov_v_deg: 41.0` per camera.

---

### Layer 3 — Fusion, State Machine, Safety

#### [MODIFY] [predator_state_machine.py](file:///c:/Users/snowd/OneDrive/Documents/Vollebak/predator/src/layer3_fusion/predator_state_machine.py)

Add `L1_ENGAGEMENT` state and auto-detect path.

- Add `L1_ENGAGEMENT` to `SystemState` enum
- Add waiter config fields to `FSMConfig`
- Modify `on_layer1_detection()` — new params `elevation_deg`, `centroid_vy_degps`:
  - If `elevation < 0` AND `confidence >= 0.95` AND `range_est <= 50m`:
    - `SILENT → L1_ENGAGEMENT` (skip ALERT/RADAR_ACTIVE/TRACKING)
    - Emit ArmDeploy + GimbalCue (bearing only, sweep mode)
    - Radar stays DEEP_SLEEP
  - Else: normal ALERT/RADAR_ACTIVE path
- Add `L1_ENGAGEMENT` to `tick()` timeout handling
- Modify BDA: L1_ENGAGEMENT origin → flicker cessation alone = KILL_CONFIRMED
- `_emit_radar_power_command()`: no-op for L1_ENGAGEMENT

#### [MODIFY] [safety_manager.py](file:///c:/Users/snowd/OneDrive/Documents/Vollebak/predator/src/layer3_fusion/safety_manager.py)

Add sweep-bounds safety check.

- Add `L1SafetyConfig(min_elevation=-20°, max_range=50m, min_range=3m)`
- Add `check_sweep_engagement(bearing, el_min, el_max, range_est)` method

#### [NEW] [waiter_range_estimator.py](file:///c:/Users/snowd/OneDrive/Documents/Vollebak/predator/src/layer3_fusion/waiter_range_estimator.py)

Range estimation + velocity vector + sweep bounds computation.

- `WaiterRangeEstimator(camera_height_m, ansur_bounds, pixel_angular_res)`
- `estimate_range(elevation_deg) → RangeEstimate(range_m, uncertainty_m, is_valid, posture)`
- `update_posture(imu_pitch_deg)` — standing/crouching/prone height adjustment
- `compute_sweep_bounds(elevation, range, velocity_vy) → SweepBounds`:
  - Zone 1 (PRIMARY): ground level ± margin
  - Zone 2 (EXCLUSION): horizon band (skip)
  - Zone 3 (PURSUIT): above horizon if ascending velocity detected

---

### Layer 4 — Engagement

#### [MODIFY] [slew_to_cue.py](file:///c:/Users/snowd/OneDrive/Documents/Vollebak/predator/src/layer4_engagement/slew_to_cue.py)

Add `compute_bearing_command(azimuth, sweep_center_el, sweep_half_width)` for L1-only mode. No Cartesian conversion, no lead-angle.

#### [MODIFY] [lissajous_scanner.py](file:///c:/Users/snowd/OneDrive/Documents/Vollebak/predator/src/layer4_engagement/lissajous_scanner.py)

Add `VerticalSweepPattern` — zone-aware 1D raster at max gimbal rate.

- `VerticalSweepConfig(sweep_rate_dps=180.0, num_passes=3, azimuth_fixed_deg)`
- `generate_pattern(sweep_bounds) → list[ScanPoint]`
- Multi-pass at max rate for probabilistic angular coverage
- Zone-aware: sweep Zone 1 first, Zone 3 if ascending, skip exclusion band

#### [MODIFY] [bda_assessor.py](file:///c:/Users/snowd/OneDrive/Documents/Vollebak/predator/src/layer4_engagement/bda_assessor.py)

Add `l1_only_mode` to BDAConfig. Flicker cessation alone = KILL_CONFIRMED.

---

### Rust Orchestrator (Mirror)

#### [MODIFY] [lib.rs](file:///c:/Users/snowd/OneDrive/Documents/Vollebak/predator/crates/predator-messages/src/lib.rs)

Add `L1Engagement` to `SystemState`, add `elevation_deg` + `centroid_vy_degps` to `Layer1Detection`.

#### [MODIFY] [state_machine.rs](file:///c:/Users/snowd/OneDrive/Documents/Vollebak/predator/crates/predator-orchestrator/src/state_machine.rs)

Mirror L1_ENGAGEMENT path: waiter config, auto-detect, L1-only BDA.

#### [MODIFY] [main.rs](file:///c:/Users/snowd/OneDrive/Documents/Vollebak/predator/crates/predator-orchestrator/src/main.rs)

Parse elevation/velocity from L1 payload, skip primer in L1Engagement.

---

### Configuration

#### [MODIFY] [predator_system.yaml](file:///c:/Users/snowd/OneDrive/Documents/Vollebak/predator/config/predator_system.yaml)

```yaml
waiter_mode:
  enabled: true
  l1_engagement_confidence: 0.95
  max_range_m: 50.0

operator:
  eye_height_m: 1.75
  helmet_camera_offset_m: 0.10
  ansur_min_height_m: 1.37
  ansur_max_height_m: 1.79

laser:
  mode: "pulsed"                 # "pulsed" | "cw"
  pulse_width_ns: 10             # Q-switched nanosecond pulse
  average_power_w: 5.0
  peak_power_class: "megawatt"   # Energy compression
  lidt_cmos_mj_cm2: 78.9        # Permanent damage threshold
  optical_gain_factor: 100000    # Drone lens funnel effect
```

---

### Tests (~15 new)

**WaiterRangeEstimator (4)**: range computation, above-horizon rejection, ANSUR uncertainty, posture adjustment

**L1 Velocity Vector (2)**: ascending detection, stationary detection

**State Machine L1_ENGAGEMENT (4)**: auto-detect engagement, above-horizon rejection, far-range rejection, flicker-only BDA

**Vertical Sweep (3)**: ground sweep bounds, ascending velocity zone activation, multi-pass coverage

**Safety (2)**: sweep authorized, keep-out veto

---

## Execution Order

1. Config: `dvxplorer_array.yaml` + `predator_system.yaml`
2. L1 elevation + velocity: `event_aggregator.py` → `propeller_detector.py`
3. Range estimator: `waiter_range_estimator.py` (new)
4. State machine (Python): `predator_state_machine.py`
5. Safety: `safety_manager.py`
6. Engagement: `slew_to_cue.py` → `lissajous_scanner.py` → `bda_assessor.py`
7. Rust mirror: `lib.rs` → `state_machine.rs` → `main.rs`
8. Tests: `test_predator_pipeline.py`

## Verification

```bash
python -m pytest tests/test_predator_pipeline.py -v
cargo test --workspace
```
