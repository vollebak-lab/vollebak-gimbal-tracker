# Project Predator: Autonomous Counter-UAS Neuromorphic & Radar Soft-Kill System

[![Tests](https://img.shields.io/badge/Python%20Tests-48%2F48%20Passing-brightgreen)]()
[![Rust](https://img.shields.io/badge/Rust%20Tests-72%2F72%20Passing-brightgreen)]()
[![C++ DSP](https://img.shields.io/badge/C%2B%2B%2FCUDA%20DSP-17%2F17%20Passing-brightgreen)]()
[![Platform](https://img.shields.io/badge/Target-NVIDIA%20Jetson%20Orin%20Nano-blue)]()

Project Predator is an ultra-low-latency, tactical Counter-UAS (C-UAS) defense system combining passive neuromorphic vision, 4D PMCW radar, kinematic threat assessment, mast self-leveling stabilization, and directed-energy soft-kill laser engagement.

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    LAYER 1: PASSIVE NEUROMORPHIC                            │
│  Sony IMX636 (1280×720 @ >10 MEv/s) / iniVation DVXplorer                                   │
│  ├── Tier 1: Continuous Gyro Homography Warper (Coordinate stabilization under ego-motion)   │
│  ├── Tier 2: UZH RSS 2026 Anticipatory ConvGRU + TensorRT FP16 Motion Mask Suppression       │
│  └── Tier 3: 4000 Hz Temporal Binning + 512-FFT + Harmonic Product Spectrum (HPS) Peak DSP    │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │ Fast-Track Cue (<5ms)
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    LAYER 2: ACTIVE 4D RADAR                                 │
│  Uhnder S80 Cascaded PMCW Array (768 Virtual Channels, 300m+ Range)                         │
│  ├── Micro-Doppler signature extraction (rotary blade chop vs micro-turbulences)            │
│  └── Range-Doppler-Azimuth-Elevation pointcloud centroiding                                 │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    LAYER 3: TRACKING & FUSION                               │
│  Rust Orchestrator (`predator-orchestrator`) + Python Fusion Engine                         │
│  ├── Extended Kalman Filter (EKF) / Interacting Multiple Model (IMM) Target Tracking         │
│  ├── Mast Self-Leveling PID Torque Control & Body-Frame Dynamic Parallax Solver             │
│  ├── Kinematic Primer: Instantaneous closing-rate threat scoring                            │
│  └── Silent-to-Active Finite State Machine with Safety Veto Interlocks                      │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 LAYER 4 & 5: ENGAGEMENT & FABRIC                            │
│  Thor Dynamics Directed-Energy Laser + Real-Time Rust Gimbal Daemon                         │
│  ├── Slew-to-Cue with sub-milliradian pointing precision                                    │
│  ├── High-frequency Lissajous optical beam scanning for sensor saturation soft-kill        │
│  ├── Closed-Loop Battle Damage Assessment (BDA) via flicker cessation & Doppler loss        │
│  └── Zenoh Distributed Multi-Node Cooperative Tracking Fabric                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Key Subsystems & Features

### 1. Neuromorphic Event Camera Ingestion Core (`ev_ingestion_cpp`)
- **Native OpenEB 5.2.0 HAL Driver**: Custom Cypress CX3 Treuzell board plugin (`0x1409:0x8e00`) for native IDS UE-39B0XCP-E (Sony IMX636) direct bulk streaming at **>9.47 Million events/second**.
- **Continuous Gyro Homography Warper (`ego_motion.hpp`)**: Analytical spherical homography $\mathbf{K} \mathbf{R}(t_{\text{ref}}, t_i) \mathbf{K}^{-1}$ with Rodrigues quaternion integration, canceling platform rotational blur (verified up to $30^\circ/\text{s}$).
- **Anticipatory Motion Suppression (`event_suppression_trt.hpp`)**: UZH RSS 2026 ConvGRU + Attention-based Time Conditioning (ATC) dynamic mask engine compiled to TensorRT 10.3 FP16 (**$14.0\text{ ms}$ GPU compute, 71.0 FPS**).
- **Hardened Multi-Gate Frequency DSP (`flicker_dsp.hpp`)**:
  - $4000\text{ Hz}$ temporal binning ($250\ \mu\text{s}$ resolution, $2000\text{ Hz}$ Nyquist).
  - 512-sample coherent integration ($128\text{ ms}$ sliding window, $+3\text{ dB}$ processing gain).
  - Harmonic Product Spectrum (HPS)comb filtering for Blade Passage Frequency ($f_{\text{BPF}}$) and RPM extraction.
  - Wideband noise floor estimator ($40\text{--}1000\text{ Hz}$) suppressing Poisson extreme-value noise.
  - Multi-candidate peak extraction resolving multi-carrier AC floodlight modulation ($100/120\text{ Hz}$) from rotor signatures.
  - Spatial dispersion bounding-box filter distinguishing localized quadcopter rotors from global ambient lighting.
  - M-of-N temporal confirmation ($M=3$) and track coasting up to $200\text{ ms}$.
- **Web UI & Telemetry Endpoint**: Live HTTP visualizer (`http://<target-ip>:8080/`) streaming 30 FPS polarity-colored overlays and `/flicker_stats` JSON metrics.

### 2. High-Assurance Rust Orchestrator (`crates/predator-orchestrator`)
- Real-time Silent-to-Active state machine managing zero-RF-signature standby to active laser soft-kill prosecution.
- Mast self-leveling stabilization loop with dual-axis PID and overcurrent/IMU fault handling.
- Dynamic body-frame 3D parallax correction reconciling radar, camera, and laser turret coordinate offsets.
- High-throughput Zero-Copy message serialization (`predator-messages`) powered by Zenoh and MessagePack.

### 3. Verification & Outdoor Benchmarks
- **Range & Tracking**: Verified on DJI Mavic Air 2 in direct sunlight and semi-shade at 50ft, 75ft, and 100ft ($30.5\text{m}$) with zero false alarms.
- **Test Coverage**:
  - `python -m pytest`: 48/48 passed.
  - `cargo test`: 72/72 passed.
  - `ev_ingestion_cpp` unit tests: 17/17 passed.

---

## 🛠️ Repository Layout

```
predator/
├── config/                      # System & sensor configuration YAMLs
│   ├── dvxplorer_array.yaml
│   └── predator_system.yaml
├── crates/                      # Rust workspace crates
│   ├── predator-messages/       # Shared message types, topics, and serialization
│   └── predator-orchestrator/   # Real-time state machine, mast stabilization, & parallax
├── docs/                        # Architecture specs, kill-chain latency audits, research papers
│   ├── architecture/
│   ├── research/
│   └── waiter_mode/
├── ev_ingestion_cpp/            # High-performance C++/CUDA/TensorRT neuromorphic engine
│   ├── ego_motion.hpp           # Analytical continuous gyro homography warper
│   ├── event_suppression_trt.hpp# UZH RSS 2026 TensorRT FP16 dynamic suppression
│   ├── flicker_dsp.hpp          # Harmonic comb & 4000Hz frequency-domain DSP
│   ├── ev_flicker_detector.cpp  # Main live detector & Web HUD daemon
│   ├── ev_web_viewer.cpp        # Low-overhead web visualizer
│   └── test_*.cpp               # C++ DSP and ego-motion test suites
├── models/                      # Deep learning suppression network export & ONNX definitions
│   └── export_suppression_model.py
├── src/                         # Python C-UAS layers
│   ├── imu/                     # IMU providers & motion compensation
│   ├── layer1_neuromorphic/     # Python event streamers & aggregators
│   ├── layer2_radar/            # Uhnder PMCW radar interface & micro-Doppler classifier
│   ├── layer3_fusion/           # EKF/IMM tracking, JPDA associator, safety manager
│   ├── layer4_engagement/       # Laser slew-to-cue, Lissajous scanner, gimbal controller
│   └── layer5_cooperative/      # Distributed cooperative multi-node tracker
├── tests/                       # Python end-to-end integration test suite
├── brain_updates.md             # Detailed engineering log, RCA findings, and hardware profiles
└── task.md                      # Phased milestone and implementation tracking
```

---

## 🚀 Building & Running

### 1. Python Pipeline Tests
```bash
python -m pytest
```

### 2. Rust Workspace
```bash
cargo check
cargo test
```

### 3. C++ Neuromorphic Engine (Jetson Orin Nano / Linux)
```bash
cd ev_ingestion_cpp
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
./test_flicker_dsp
./test_ego_motion
./ev_flicker_detector
```

---

## 🔒 Security & Safety Controls
- **Keep-Out Cones**: Software and hardware interlocks strictly veto laser emission towards operator forward zones, non-cleared airspace, or ground elevations $< 5^\circ$.
- **Zero-RF Silent Stance**: Layer 1 passive neuromorphic vision operates with zero RF emission until confirmed drone rotor harmonics trigger active cueing.
- **Fail-Safe Interlocks**: Automatic laser shutdown and mast stowage upon track loss, BDA confirmation, or IMU telemetry degradation.
