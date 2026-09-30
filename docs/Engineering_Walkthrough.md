# Project Predator — Engineering Walkthrough

## Delivered Code

**26 source files | 18 Python modules | 4 Rust crates | 4 YAML configs | 86/86 tests passing**

### Layer 1 — Neuromorphic Detection Pipeline (38.9 KB)

| File | Purpose |
| -- | -- |
| `dvx_event_streamer.py` | 4× DVXplorer Micro USB ingestion, thread-per-camera |
| `event_aggregator.py` | Global bearing enrichment from camera mount offset + FOV |
| `propeller_detector.py` | SpMM-kernel propeller flicker detection with harmonic matching |
| `imu_provider.py` | BNO055 IMU fusion for operator heading compensation |
| `motion_compensator.py` | Ego-motion subtraction from event stream + gimbal feed-forward |

### Layer 2 — Radar Processing (25.1 KB)

| File | Purpose |
| -- | -- |
| `aerial_target_tracker.py` | Kalman-filtered 3D track maintenance with coast logic |
| `micro_doppler_classifier.py` | STFT-based rotary/fixed-wing/bird classification |
| `simulated_radar.py` | Development radar stub generating synthetic tracks |
| `safety_manager.py` | Keep-out cone, range limits, elevation floor enforcement |

### Layer 3 — Threat Fusion (12.4 KB)

| File | Purpose |
| -- | -- |
| `threat_classifier.py` | Multi-source threat scoring (flicker + radar + micro-Doppler) |
| `kinematic_primer.py` | Python reference for fast kinematic-only threat screening |

### Layer 4 — Engagement (Rust) (42.3 KB)

| Crate/File | Purpose |
| -- | -- |
| `predator-orchestrator/src/main.rs` | Async Zenoh event loop — subscribes to all pipeline topics, drives SM |
| `predator-orchestrator/src/state_machine.rs` | System Machine (SM) — 6-state kill chain with engagement gate |
| `predator-orchestrator/src/kinematic_primer.rs` | Rust kinematic primer — 0-latency threat pre-screen |
| `predator-orchestrator/src/parallax.rs` | **NEW** — radar-to-turret parallax correction for body-frame geometry |
| `predator-messages/src/lib.rs` | Shared binary message types (msgpack) — all Zenoh payloads |
| `predator-messages/src/topics.rs` | Zenoh topic namespace constants |
| `predator-gimbal-controller/src/main.rs` | 200Hz PID gimbal servo with Lissajous scan pattern |

---

## Architecture

### Kill Chain Pipeline

```
Layer 1 (Python)          Layer 2 (Python)         Layer 3 (Python)        Layer 4 (Rust)
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐
│ 4× DVXplorer │    │ 4× Uhnder    │    │ Threat       │    │ Orchestrator         │
│ Micro        │───►│ S80 Radar    │───►│ Classifier   │───►│ (SM + Primer +       │
│ Event Camera │    │ Track+Class  │    │ Score Fusion  │    │  Parallax Correction)│
└──────────────┘    └──────────────┘    └──────────────┘    └──────┬───────────────┘
                                                                    │
                                                                    ▼
                                                            ┌──────────────────┐
                                                            │ Gimbal PID       │
                                                            │ (200Hz servo)    │
                                                            │ + Z-Fold Mast    │
                                                            │ + Thor Laser     │
                                                            └──────────────────┘
```

### Zenoh Topic Map

| Topic | Direction | Payload | Serialization |
| -- | -- | -- | -- |
| `predator/layer1/detection` | Python → Rust | `Layer1Detection` | msgpack |
| `predator/radar/track` | Python → Rust | `RadarTrackUpdate` | msgpack |
| `predator/radar/classification` | Python → Rust | `ThreatClassification` | msgpack |
| `predator/imu/gimbal_feedforward` | Python → Rust | `ImuFeedForward` | msgpack |
| `predator/system/state` | Rust → Python | `SystemStateMsg` | msgpack |
| `predator/engagement/pre_slew` | Rust → Rust | `PreSlewCommand` | msgpack |
| `predator/engagement/command` | Rust → Rust | `GimbalCommand` | msgpack |
| `predator/arm/command` | Rust → HW | `ArmCommand` | msgpack |
| `predator/arm/status` | HW → Rust | `ArmStatus` | msgpack |

---

## System Machine (SM) States

| State | Entry Condition | Actions |
| -- | -- | -- |
| **SILENT** | Initial / timeout | Radar off, mast stowed |
| **ALERT** | L1 detection (low conf) | Mast deploys vertically, FSM pre-aims |
| **RADAR_ACTIVE** | Confirmed detection | Radar authorized, searching |
| **TRACKING** | Radar acquires track | Gimbal cueing, threat scoring |
| **ENGAGEMENT** | Threat confirmed + mast deployed + stable | Laser authorized |
| **BDA** | Post-engagement | Flicker/Doppler monitoring for kill confirm |

### Engagement Safety Gate

Transition from TRACKING → ENGAGEMENT requires **both**:

1. `threat_score >= 0.7`
2. `arm_deployment_confirmed == true` (mast `Deployed` + `is_stable`)

If threat score exceeds threshold before mast is ready, the score is stored. When the mast subsequently confirms deployment, the gate auto-releases.

---

## Vertical Mast System

### Design Decision

Replaced shoulder-deployed arm with vertical Z-fold mast. See "Vertical Mast Design Rationale" document for full engineering justification.

**Key advantages:**

* Static turret position → no forward kinematics → no joint encoder error in pointing
* Symmetric 360° coverage (vs asymmetric shoulder)
* Single-axis extension → simpler actuator → fewer failure modes
* Vertical loading → gravity in compression (strongest axis)

### Mast Specification

| Parameter | Value |
| -- | -- |
| Length | 500mm (3 segments × \~167mm) |
| Clearance | IHPS helmet + MS2000 strobe + 50mm margin |
| Deployed turret position | (0, 0, +0.75m) in body frame |
| Stow profile | \~170mm flat against rear plate |

### Body-Frame Geometry

See "Body-Frame Geometry & Parallax Correction" document for coordinate system definition, sensor positions, and parallax correction algorithm.

---

## Test Coverage

### Rust — 38 tests

| Suite | Count |
| -- | -- |
| Messages (serialization, geometry, body frame) | 7 |
| Parallax correction (4 range tiers, direction, offset) | 7 |
| State machine (core transitions + mast safety gate) | 16 |
| Kinematic primer | 5 |
| Doc-tests | 1 |
| Gimbal controller | 2 |

### Python — 48 tests

| Suite | Count |
| -- | -- |
| State machine transitions | 14 |
| Aerial tracker | 4 |
| Micro-Doppler classifier | 2 |
| Safety manager | 5 |
| Gimbal math (slew-to-cue, Lissajous) | 4 |
| Simulated radar | 2 |
| Event aggregator | 2 |
| IMU provider | 5 |
| Motion compensator | 4 |
| Kinematic primer | 6 |

### Total: 86/86 passing