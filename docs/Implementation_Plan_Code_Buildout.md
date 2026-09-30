# Project Predator — Code Buildout Plan

## What We Can Build NOW vs What's Blocked

| Layer | SDK Available? | Build Status |
| -- | -- | -- |
| **Layer 1 — Neuromorphic** | ✅ `dv_processing` + overlab-kevin repo | **BUILD NOW** — full pipeline |
| **Layer 2 — Radar** | ❌ Blocked on Uhnder S80 SDK | **STUB INTERFACES** — define contracts |
| **Layer 3 — Fusion** | ✅ Pure Python (numpy/scipy) | **BUILD NOW** — uses radar stubs |
| **Layer 4 — Engagement** | ✅ Math is pure Python; Rust gimbal scaffolding | **BUILD NOW** (Python); **SCAFFOLD** (Rust) |
| **Layer 5 — Cooperative** | ✅ Zenoh (reuse radar_belt_engine patterns) | **SCAFFOLD** — extend swarm_network |

---

## Build Order

### 1\. Layer 1: Neuromorphic Detection Pipeline (Full Implementation)

#### `src/layer1_neuromorphic/dvx_event_streamer.py`

* Multi-camera USB event ingestion using `dv.io.CameraCapture`
* Opens 4× DVXplorer Micro by USB path from `dvxplorer_array.yaml`
* Yields `(events_tensor, camera_id, hw)` per batch
* Thread-per-camera architecture (matches `RadarReaderThread` pattern from `radar_belt_engine`)
* Graceful USB disconnect/reconnect

#### `src/layer1_neuromorphic/event_aggregator.py`

* Consumes streams from `dvx_event_streamer.py`
* Converts per-camera pixel → global bearing using azimuth offsets
* Publishes aggregated events to Zenoh

#### `src/layer1_neuromorphic/propeller_detector.py`

* Adapts overlab-kevin `track_aedat.py` SimpleTracker FSM
* Per-camera voxelization → SpMiniUNet inference → detection alert
* Publishes to Zenoh `predator/layer1/detection`
* Imports model/voxelization from vendor submodule

---

### 2\. Layer 2: Radar Interface Stubs

#### `src/layer2_radar/radar_interface.py`

* Abstract base class defining the radar API contract
* `get_point_cloud() → RadarPointCloud` dataclass
* `get_detections() → List[RadarDetection]` dataclass
* Simulated radar backend for testing (generates synthetic point clouds)

#### `src/layer2_radar/micro_doppler_classifier.py`

* STFT-based blade frequency extraction (pure scipy — no SDK needed)
* Classifier: extract fundamental + harmonics → classify target type
* Can be tested with synthetic Doppler data now

#### `src/layer2_radar/coordinate_transformer.py`

* Radar-to-global coordinate transforms (pure numpy)
* IMU ego-motion compensation (reuse from `ego_motion_compensator.py`)

---

### 3\. Layer 3: Sensor Fusion & State Machine (Full Implementation)

#### `src/layer3_fusion/predator_state_machine.py`

* `SILENT → ALERT → RADAR_ACTIVE → TRACKING → ENGAGEMENT → BDA`
* Subscribes to Layer 1 detection + Layer 2 radar (or stubs)
* Publishes system state to Zenoh

#### `src/layer3_fusion/aerial_target_tracker.py`

* Fork of `target_tracker.py` KalmanTrack + Hungarian assignment
* State vector extended for aerial kinematics: `[x, y, z, vx, vy, vz, ax, ay, az]`
* Higher velocity gates (drones move faster than humans)
* 3D maneuver model instead of constant-velocity

#### `src/layer3_fusion/sensor_fusion.py`

* EKF fusing neuromorphic bearing (azimuth only) + radar 3D (range/az/el/doppler)
* DVXplorer IMU for head motion compensation

#### `src/layer3_fusion/threat_classifier.py`

* Aggregates: micro-Doppler class + flicker confidence + radar RCS → threat score

#### `src/layer3_fusion/safety_manager.py`

* Keep-out zone enforcement (30° cone around operator head)
* Range gating (10m min, 300m max)
* Engagement authorization logic

---

### 4\. Layer 4: Engagement (Python Math + Rust Scaffold)

#### `src/layer4_engagement/slew_to_cue.py`

* Converts radar track `(x, y, z)` → gimbal `(azimuth, elevation)` commands
* Accounts for gimbal mounting offset and operator body frame

#### `src/layer4_engagement/lissajous_scanner.py`

* Generates Lissajous/Rosette scan patterns parameterized by target range and RCS
* Outputs az/el deflection sequences at configurable dwell time

#### `src/layer4_engagement/bda_assessor.py`

* Kill assessment: monitors flicker cessation + Doppler signature loss + ballistic trajectory onset
* Publishes BDA result to Zenoh

#### `src/layer4_engagement/engagement_log.py`

* Append-only JSON-lines engagement chronicle

#### `src/layer4_engagement/gimbal_controller/` (Rust)

* Cargo project structure with PID controller stub
* Zenoh subscriber for target bearing commands
* Safety interlock enforcement in firmware

---

### Reuse Map from `radar_belt_engine`

| Source File | Predator Target | Adaptation |
| -- | -- | -- |
| `target_tracker.py` KalmanTrack | `aerial_target_tracker.py` | Expand state to 9-dim, constant-accel, wider gates |
| `target_tracker.py` TrackerConfig | `aerial_target_tracker.py` | Re-tune `gate_threshold`, `velocity_gate_deg` |
| `telemetry_publisher.py` Zenoh patterns | All layers | Same pub/sub pattern, new topics |
| `ego_motion_compensator.py` | `coordinate_transformer.py` | Adapt for DVXplorer IMU instead of belt IMU |
| `point_cloud_filter.py` | `radar_interface.py` | Range/SNR filtering for aerial targets |

---

## Verification Plan

### Unit Tests (can run without hardware)

* `test_propeller_detector.py` — feed recorded AEDAT4 → validate SpMiniUNet outputs
* `test_micro_doppler.py` — synthetic Doppler signals → validate STFT + classification
* `test_state_machine.py` — inject detection/tracking events → validate FSM transitions
* `test_aerial_tracker.py` — synthetic 3D trajectories → validate Kalman tracking
* `test_safety_manager.py` — sweep gimbal angles → verify keep-out enforcement
* `test_lissajous.py` — validate scan pattern geometry
* `test_slew_to_cue.py` — known XYZ → expected az/el

### Integration (requires hardware stubs)

* Layer 1 → Layer 3: detection alert → FSM transition → simulated radar → tracking
* Full pipeline with radar stubs: detect → track → engage → BDA