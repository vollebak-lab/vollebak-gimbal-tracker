# Predator System Architecture — Overview

> **Project**: PREDATOR-01 | **Track**: Ecosystem | **Phase**: Challenge Hub
> **Classification**: VOLLEBAK PROPRIETARY

## 1. Mission Statement

Predator is a dual helmet and plate carrier mounted Counter-UAS (C-UAS) system employing a **Silent-to-Active** sensor pipeline with directed-energy prosecution. The system remains electromagnetically silent until a passive neuromorphic tripwire detects a threat, then either:

- **Standard path**: Activates radar for precision 3D tracking → laser engagement
- **Waiter path** *(new)*: Engages directly from L1 geometry alone — zero RF emission throughout the kill chain

## 2. System Context

| Attribute       | Value                                                                       |
| --------------- | --------------------------------------------------------------------------- |
| **Form Factor** | Helmet-worn, plate-carrier mounted                                          |
| **Compute**     | NVIDIA Jetson Orin NX (16 GB)                                               |
| **Languages**   | Python (ML/signal processing), Rust (real-time daemons), C++ (SDK bridge)   |
| **IPC**         | Zenoh pub/sub (MessagePack binary + JSON fallback)                          |
| **Effector**    | Thor Dynamics Q-switched pulsed laser, 2–5 W average, megawatt-class peak   |
| **Target Set**  | Multi-rotor UAS, FPV kamikazes, fixed-wing sUAS, ground-level ambush drones |

## 3. Five-Layer Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                        PREDATOR PIPELINE                            │
│                                                                      │
│  Layer 1: PASSIVE TRIPWIRE (Neuromorphic)                           │
│  4× DVXplorer Micro → SpMiniUNet propeller detection                │
│  → bearing + elevation + velocity vector                            │
│                              │                                       │
│              ┌───────────────┴───────────────┐                      │
│              │                               │                       │
│     elevation < 0° AND              standard detection              │
│     conf ≥ 0.95 AND                 path (conf < 0.95              │
│     range ≤ 50 m                    OR above horizon)               │
│              │                               │                       │
│              ▼                               ▼                       │
│  WAITER PATH (L1_ENGAGEMENT)     STANDARD PATH                      │
│  Zero RF. VerticalSweep +        Layer 2: ACTIVE RADAR              │
│  pulsed laser. ~400 ms kill      2× Zadar ZPulse (front + rear)    │
│  chain.                          → micro-Doppler classification     │
│              │                               │                       │
│              │                   Radar Track + Classification        │
│              │                               │                       │
│              │                               ▼                       │
│              │                   Layer 3: FUSION & DECISION         │
│              │                   IMM/JPDA/MHT tracking │ State      │
│              │                   Machine │ Threat Scoring │ Safety  │
│              │                               │                       │
│              │                      Engagement Command               │
│              └───────────────┬───────────────┘                      │
│                              ▼                                       │
│  Layer 4: ENGAGEMENT                                                 │
│  Slew-to-Cue → Gimbal (200 Hz Rust PID)                            │
│  → VerticalSweep / Lissajous / Rosette / Raster → BDA              │
│                              │                                       │
│                        Kill Assessment                               │
│                              ▼                                       │
│  Layer 5: COOPERATIVE                                               │
│  Squad Zenoh mesh → 360° coverage → optimal laser selection         │
└──────────────────────────────────────────────────────────────────────┘
```

## 4. Hardware Stack

| Component                       | Qty | Role                                                                                           | Interface                 |
| ------------------------------- | --- | ---------------------------------------------------------------------------------------------- | ------------------------- |
| **DVXplorer Micro** (iniVation) | 4   | Neuromorphic event cameras, 640×480, 9 µm pixel, 110 dB DR, onboard ICM-42688 IMU              | USB 3.1                   |
| **Zadar Labs ZPulse**           | 2   | 4D FMCW SDIR™ radar, 77 GHz, 120°×90° FOV per unit; front plate carrier + rear plate carrier   | Automotive Ethernet / PoE |
| **Jetson Orin NX**              | 1   | Primary compute: SNN + radar DSP + orchestrator                                                | —                         |
| **Thor Dynamics Laser**         | 1   | Q-switched pulsed, 2–5 W average, ~10 ns pulse, megawatt-class peak power, 10 km claimed range | GPIO/serial               |
| **Gimbal Assembly**             | 1   | 2-axis + FSM, 200 Hz servo rate, Lissajous/Rosette/Raster/VerticalSweep patterns               | Rust daemon               |
| **Mast Assembly**               | 1   | Z-fold 3-segment, 500 mm deployed, rear plate carrier                                          | Maxon EC20                |
| **BHI260AP IMU**                | 1   | External authority IMU, 200 Hz                                                                 | Zenoh                     |
| **ICM-42688**                   | 4   | Per-camera onboard IMU (DVXplorer Micro integrated)                                            | dv_processing             |

### Radar Coverage Model — Per-Operator and Squad-Level

The ZPulse's 120°×90° FOV means no single unit covers 360°. Coverage is achieved at two scales:

| Scale            | Mechanism                                    | Coverage                                             |
| ---------------- | -------------------------------------------- | ---------------------------------------------------- |
| **Per operator** | 2× ZPulse (front + rear plate carrier)       | ~240° combined (with overlap at flanks)              |
| **Squad**        | Zenoh mesh detection events across all nodes | Full 360° — no two operators face the same direction |

When a detection event arrives over the squad mesh, the **cooperative layer automatically selects the optimal Predator** for engagement based on geometry (bearing to target), laser availability, and engagement gate status. No operator input required — selection and engagement handoff are autonomous.

**Detection basis**: The ZPulse does not need to resolve drone RCS directly. Between the L1 neuromorphic propeller flicker (visual) and the ZPulse 4D Doppler return (RF flicker from rotating blades), the sensor fusion has two independent high-confidence cues — range, azimuth, elevation, and radial velocity — before any engagement decision.

**SDK**: Zadar Ethernet Sensor SDK (MIT open-source, C++/Python/ROS). Replaces the previously planned Uhnder SDK (which was proprietary, undelivered, and the vendor was unresponsive). The `uhnder_bridge/` directory is now stale and will be renamed to `zpulse_bridge/` when the integration work begins.

### Laser Physics — Q-Switched Pulsed Operation

The Thor Dynamics laser operates as a **Q-switched pulsed system**, not a CW thermal dwell weapon:

| Parameter                 | Value                                                            |
| ------------------------- | ---------------------------------------------------------------- |
| Average power             | 2–5 W                                                            |
| Pulse width               | ~10 ns                                                           |
| Peak power class          | Megawatt (energy compression via Q-switching)                    |
| Pulse rate                | 5 Hz                                                             |
| Damage mechanism          | Dielectric breakdown → lattice shatter (mechanical, not thermal) |
| CMOS LIDT (pulsed)        | 78.9 mJ/cm²                                                      |
| Optical gain (drone lens) | Up to 100,000× (lens funnels beam onto CMOS)                     |

**Engagement implication**: A single pulse entering the drone's camera lens causes irreversible CMOS damage. The VerticalSweep pattern maximizes the probability of aperture hit across passes — not thermal dwell accumulation.

## 5. Engagement States

```
SILENT ──(L1 detection)──► ALERT ──(confirmed)──► RADAR_ACTIVE
           │                 │                        │
     (waiter trigger)  (conf ≥ 0.8 = FAST TRACK)     │
           │                 └────────────────────────┘
           │                                          │
           ▼                               (radar acquires)
   L1_ENGAGEMENT                                      ▼
   (zero RF, VerticalSweep            TRACKING
    pulsed laser, ~400 ms)                │
           │                    (threat ≥ 0.7 + gate)
           │                              ▼
           │                         ENGAGEMENT
           │                              │
           │                        (laser fired)
           └────────────────┬─────────────┘
                            ▼
                           BDA
                            │
                   (kill confirmed)
                            ▼
                          SILENT
```

### Standard Path

- **Alert confirm**: 1 detection (single-detection fast-track; cameras are physically spaced, making simultaneous multi-camera hits on the same target unlikely)
- **Fast Track**: If L1 confidence ≥ 0.8, skip ALERT → jump directly to RADAR_ACTIVE + deploy arm + pre-slew
- **Engagement Gate**: `arm_deployed AND stable AND (mast_leveled OR stabilization_fault)`. On stabilization fault → degraded gimbal-only engagement permitted

### Waiter Path (L1_ENGAGEMENT — Zero RF)

Activated automatically when all three geometric conditions are met from a single L1 detection:

| Condition      | Threshold                                |
| -------------- | ---------------------------------------- |
| Elevation      | < 0° (target below horizon)              |
| Confidence     | ≥ 0.95                                   |
| Range estimate | ≤ 50 m (geometric: `camera_height / tan( |

- Radar stays in `DEEP_SLEEP` throughout. No RF emission.
- Gimbal slews to bearing at max rate, executes **3-zone VerticalSweep** at 180°/s
- BDA: flicker cessation alone = `KILL_CONFIRMED` (no Doppler available)
- Total kill chain: **~400 ms** (mast pre-deployed) / **~1.5–2.5 s** (cold start)

#### Three-Zone Vertical Sweep

| Zone               | Elevation   | Purpose                                    |
| ------------------ | ----------- | ------------------------------------------ |
| Zone 1 (Primary)   | −15° to −2° | Ground-level drone body                    |
| Zone 2 (Exclusion) | −2° to +3°  | Horizon band — skipped                     |
| Zone 3 (Pursuit)   | +3° to +10° | Ascending drone (triggered by positive vy) |

## 6. Safety Constraints

| Constraint           | Value                                                           | Enforcement                        |
| -------------------- | --------------------------------------------------------------- | ---------------------------------- |
| Keep-out cone        | ±30° around operator head                                       | `SafetyManager.check_engagement()` |
| Min engagement range | 10 m (standard) / 3 m (L1 sweep)                                | Veto if closer                     |
| Max engagement range | 300 m (standard) / 50 m (L1 sweep)                              | Veto if farther                    |
| Min elevation        | 5° above horizon (standard) / −20° floor (L1)                   | Prevents ground engagement         |
| Engagement timeout   | 10 s                                                            | Auto-transition to BDA             |
| BDA observation      | 5 s                                                             | Post-laser observation window      |
| L1 sweep bounds      | `check_l1_sweep_engagement(bearing, el_min, el_max, range_est)` | `SafetyManager`                    |

## 7. Codebase Structure

```
predator/
├── config/
│   ├── predator_system.yaml        # System-wide config (incl. waiter_mode + operator calibration)
│   ├── dvxplorer_array.yaml        # 4-camera array setup (incl. fov_v_deg per camera)
│   ├── radar_cascaded_s80.yaml     # STALE — was Uhnder S80 cascade config; to be replaced
│   └── imu_config.yaml             # Multi-source IMU fusion
├── crates/
│   ├── predator-messages/          # Shared Zenoh message types (Rust)
│   │   └── src/{lib.rs, topics.rs}
│   └── predator-orchestrator/      # Real-time event loop (Rust)
│       └── src/{main.rs, state_machine.rs, kinematic_primer.rs,
│                mast_stabilization.rs, parallax.rs}
├── src/
│   ├── layer1_neuromorphic/        # Passive detection (Python)
│   │   ├── dvx_event_streamer.py
│   │   ├── event_aggregator.py     # Now outputs elevation_deg alongside bearing
│   │   └── propeller_detector.py   # DetectionAlert now includes elevation_deg,
│   │                               #   centroid_vy_degps, bounding_box
│   ├── layer2_radar/               # Active radar (Python)
│   │   ├── radar_interface.py
│   │   ├── micro_doppler_classifier.py
│   │   └── uhnder_bridge/          # EMPTY — STALE name; to be renamed zpulse_bridge/
│   ├── layer3_fusion/              # Tracking & decision (Python)
│   │   ├── aerial_target_tracker.py
│   │   ├── imm_tracker.py
│   │   ├── jpda_associator.py
│   │   ├── hypothesis_tracker.py
│   │   ├── predator_state_machine.py  # Now includes L1_ENGAGEMENT state + Waiter path
│   │   ├── waiter_range_estimator.py  # NEW — geometric range + 3-zone sweep bounds
│   │   ├── safety_manager.py          # Now includes check_l1_sweep_engagement()
│   │   └── threat_classifier.py
│   ├── layer4_engagement/          # Engagement (Python + Rust)
│   │   ├── slew_to_cue.py          # Now includes compute_l1_command() for L1-only mode
│   │   ├── lissajous_scanner.py    # Now includes VerticalSweepPattern (5 Hz pulse-synced)
│   │   ├── bda_assessor.py         # Now includes l1_only_mode (flicker cessation = kill)
│   │   ├── engagement_log.py
│   │   └── gimbal_controller/      # Rust crate
│   ├── layer5_cooperative/         # Squad mesh: 360° coverage + optimal laser selection
│   │   └── cooperative_tracker.py
│   └── imu/                        # IMU subsystem (Python)
│       ├── imu_provider.py
│       └── motion_compensator.py
├── tests/
│   └── test_predator_pipeline.py   # 48 Python unit tests (93 total incl. Rust)
├── docs/
│   ├── architecture/               # This document + layer specs, Rust orchestrator, ops order
│   ├── waiter_mode/                # Waiter Mode implementation plan + walkthrough
│   ├── Body-Frame Geometry & Parallax Correction.md
│   ├── Mast Motor Controller & Safety Gate Specification.md
│   ├── Mast Self-Leveling Stabilization — Engineering Specification.md
│   ├── Radar Fire Control RoC Research.md
│   └── Vertical Mast Design Rationale — Arm Architecture Decision.md
├── Cargo.toml                      # Rust workspace
└── README.md
```

## 8. Key Architecture Decisions

| Decision                                       | Rationale                                                                                                                                                                                                                             |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Vertical Z-fold mast** vs. shoulder arm      | Static offset geometry (compile-time constant) eliminates joint kinematics entirely from laser pointing. Shoulder arm propagated joint errors → 2.6–10.5 m miss at 100 m.                                                             |
| **DVXplorer Micro** vs. GenX320                | GenX320 required OpenMV bridge boards and incompatible SDK. DVXplorer Micro: direct USB + native `dv_processing` support.                                                                                                             |
| **Zadar ZPulse** vs. Uhnder S80                | Uhnder were unresponsive; vendor viability uncertain. ZPulse is a production unit with open-source MIT SDK (C++/Python/ROS), IP68, and a known integration path. Architecture adapts: 2 units per operator instead of 4-chip cascade. |
| **2-unit per operator + squad mesh** for 360°  | No single affordable unit covers 360°. Squad members naturally face different directions. Zenoh mesh aggregates detection events across all nodes; cooperative layer selects optimal laser automatically.                             |
| **Propeller flicker as primary detection cue** | Drone RCS is too small for reliable standalone radar detection. L1 visual flicker + L2 RF Doppler blade return give two independent high-confidence cues. Radar confirms range/velocity; L1 confirms bearing/elevation.               |
| **Single-detection fast-track**                | Cameras are physically spaced 90° apart — simultaneous multi-camera detection of the same target is geometrically unlikely. Single detection is sufficient.                                                                           |
| **Pulsed (Q-switched) laser** vs. CW           | Single nanosecond pulse entering drone lens = lattice shatter. Sweep for hit probability, not thermal dwell.                                                                                                                          |
| **Waiter Mode (L1_ENGAGEMENT)**                | Ambush drones at ground level (<50 m) can be prosecuted faster and with zero RF signature if engagement is triggered purely from L1 geometry.                                                                                         |
| **Rust orchestrator** as authority             | Sub-millisecond tick latency, no GIL, deterministic real-time behaviour. Python SM exists for unit testing only.                                                                                                                      |
