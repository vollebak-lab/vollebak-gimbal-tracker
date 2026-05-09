# Project Predator

Tactical helmet-mounted Counter-UAS system: Silent-to-Active sensor pipeline with directed-energy prosecution.

## Architecture

```
Layer 1: PASSIVE TRIPWIRE         Layer 2: ACTIVE RADAR             Layer 3: FUSION
4× DVXplorer Micro (iniVation)    4× Uhnder S80 (2 cascade pairs)  EKF + Silent-to-Active FSM
~200m range, ~220° FOV            ~300m+ range, 360° coverage       Bearing + 3D → unified track
Propeller flicker (SpMiniUNet)    Micro-Doppler classification      Threat scoring
        │                                  │                                │
        └──────────── ALERT ───────────────┘                                │
                                                                            ▼
                                                              Layer 4: ENGAGEMENT
                                                              Rust gimbal controller
                                                              Slew-to-cue + Lissajous scan
                                                              Thor Dynamics 2-5W soft-kill laser
                                                              BDA (flicker cessation + Doppler loss)
```

## Hardware Stack

| Component | Quantity | Role |
|-----------|----------|------|
| **DVXplorer Micro** (iniVation) | 4 | Neuromorphic event cameras, 640×480, USB 3.1, onboard IMU |
| **Uhnder S80** | 4 (2×2 cascaded) | PMCW 4D radar, 12Tx/16Rx per chip, 768 total virtual channels |
| **Jetson Orin NX** | 1 | Primary compute: SNN inference + radar processing |
| **Thor Dynamics Laser** | 1 | 2-5W optical, soft-kill to 10km, mounted on robotic arm/gimbal |

## Language Standards

- **Rust**: High-frequency I/O daemons (gimbal controller, cooperative engagement)
- **Python**: ML inference (SpMiniUNet), radar signal processing (micro-Doppler FFT)
- **C++**: Uhnder SDK bridge (thin wrapper only)

## Sensor Pipeline

Layer 1 (neuromorphic) detects propeller flicker → cues Layer 2 (radar) with bearing estimate →
radar provides 3D localization + micro-Doppler classification → sensor fusion produces unified
track → gimbal slews to target → Lissajous scan pattern saturates target → BDA assesses kill.

## Dependencies

- [event-cam-prop-tracker](https://github.com/overlab-kevin/event-cam-prop-tracker) — SpMiniUNet for propeller detection (git submodule)
- [dv_processing](https://gitlab.com/inivation/dv/dv-processing) — iniVation event camera SDK
- [spconv](https://github.com/traveller59/spconv) — Sparse 3D convolutions for PyTorch
- [zenoh](https://zenoh.io/) — Pub/sub telemetry fabric

## Linear

Project tracked at: [Project Predator](https://linear.app/ghost-lab/project/project-predator-5bed061157e6)
