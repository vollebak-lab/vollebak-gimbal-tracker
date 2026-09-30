**Decision: Replace shoulder-deployed robotic arm with vertical Z-fold mast.**  
**Status: Implemented and tested (86/86 tests passing).**

---

## 1. Problem Statement

The Predator C-UAS system requires a robotic arm to physically position the laser turret (gimbal + Fast Steer Mirror + laser) clear of the operator's body for line-of-sight engagement. The original design deployed the arm from the rear plate carrier to either the left or right shoulder based on detection bearing.

This document records the engineering rationale for transitioning to a vertical mast design.

---

## 2. Why Mast Is Superior to Shoulder Deployment

### 2.1 Coordinate Transform Complexity

**Shoulder deployment** creates a **dynamic kinematics** problem:

- Turret position = f(shoulder, joint_angle₀, joint_angle₁, ..., joint_angleₙ)
- Each joint has encoder uncertainty, backlash, and flex
- Position must be computed at 200Hz (state machine tick rate)
- Any joint error propagates as laser pointing error

**Vertical mast** creates a **static offset** problem:

- Turret position = (0, 0, mast_base_height + mast_length) — a compile-time constant
- No kinematics computation needed
- No encoder feedback required for pointing (only for deployment state)
- Zero kinematic error contribution to laser pointing

### 2.2 Pointing Error Analysis

At 100m range, the relationship between position error and angular miss:

| Joint uncertainty | Shoulder → angular error                | Mast → angular error            |
| ----------------- | --------------------------------------- | ------------------------------- |
| 0.5° per joint    | 1.5–3.0° (3 joints) → **2.6–5.2m miss** | 0° (no joints in pointing path) |
| 1.0° per joint    | 3.0–6.0° → **5.2–10.5m miss**           | 0°                              |
| 2.0° per joint    | 6.0–12.0° → catastrophic                | 0°                              |

The mast design removes joint kinematics entirely from the pointing equation. The only pointing components are the gimbal PID controller and the Fast Steer Mirror — both purpose-built for sub-millidegree precision.

### 2.3 Decision Latency

**Shoulder deployment** required bearing → shoulder routing:

1. Layer 1 detection at bearing θ
2. Normalize bearing to [0°, 360°)
3. If θ < 180° → deploy to right shoulder
4. If θ ≥ 180° → deploy to left shoulder
5. If threat moves across centerline → wrong shoulder → must re-deploy

**Vertical mast** eliminates shoulder selection entirely:

1. Layer 1 detection at any bearing
2. Deploy mast vertically
3. Fast Steer Mirror aims at bearing
4. Full 360° coverage from single mast position

### 2.4 Coverage Symmetry

- **Shoulder:** asymmetric — left/right shoulder geometry differs due to operator handedness, kit placement, and arm flex. Each shoulder had different occlusion zones.
- **Mast:** symmetric — turret is on the body centerline above the head. 360° unobstructed coverage (minus helmet shadow below horizon).

### 2.5 Mechanical Simplicity

- **Shoulder:** required shoulder selection servo, multi-axis articulation, and variable stow positions.
- **Mast:** single-axis extension (fold/unfold along Z). Three identical hinged segments. Simpler actuator, fewer failure modes.

### 2.6 Stability

- **Shoulder:** arm extended horizontally creates a long moment arm. Wind loading, operator movement, and arm flex all contribute to turret oscillation.
- **Mast:** vertical extension is mechanically superior — gravity loads the structure in compression (strongest axis). Shorter moment arm for wind loading. The gimbal PID compensates for remaining oscillation.

---

## 3. Mast Clearance Geometry

### 3.1 Requirement

The mast must extend the laser turret above the operator's head, clearing:

1. The tallest combat helmet in current US military inventory
2. Equipment mounted on top of the helmet (IR beacons, NVG battery packs)
3. Safety margin for head movement during operation

### 3.2 Helmet Inventory Considered

| Helmet   | Era   | Profile  | Notes                                                                          |
| -------- | ----- | -------- | ------------------------------------------------------------------------------ |
| **MICH** | 2001+ | Mid-cut  | Modular Integrated Communications Helmet                                       |
| **ACH**  | 2003+ | Mid-cut  | Advanced Combat Helmet (successor to MICH)                                     |
| **ECH**  | 2013+ | Mid-cut  | Enhanced Combat Helmet (UHMWPE + thermoplastic)                                |
| **IHPS** | 2019+ | Full-cut | Integrated Head Protection System — tallest profile with mandible guard option |

The IHPS represents the worst-case (tallest) helmet profile and was used as the design reference.

### 3.3 Dimensional Stack (Sources)

All measurements referenced against ANSUR (Anthropometric Survey of U.S. Army Personnel) 50th percentile male data and published equipment specifications.

| Component                           | Height (mm) | Source                                                                                     |
| ----------------------------------- | ----------- | ------------------------------------------------------------------------------------------ |
| **Torso center → top of shoulders** | 200         | ANSUR 50th %ile male, menton-to-sellion → acromion height                                  |
| **Shoulders → crown of head**       | 230         | ANSUR head height: 232mm chin-to-crown (50th %ile male)                                    |
| **Crown → top of helmet shell**     | 130–150     | ACH/IHPS: shell + 19mm pad standoff + suspension height. Source: TM 10-8470-204-10         |
| **Helmet shell → IR beacon peak**   | 30–75       | MS2000 strobe: 28–33mm profile height (ACR Electronics spec sheet). Velcro mount adds ~5mm |
| **Safety clearance margin**         | 50–75       | Accounts for head tilt ±15° during tactical movement                                       |
| **Total from torso center**         | **640–730** | Sum of all components                                                                      |

### 3.4 Mast Base Position

The mast base mounts at the **top of the rear armor plate** on the plate carrier.

- Position relative to torso center: approximately (−0.12m, 0.0m, +0.25m)
- The −0.12m X offset is rearward (behind sternum)
- This is where the Z-fold hinge attaches

### 3.5 Required Mast Extension

```
Required clearance above torso center:     730mm (worst case)
Mast base height above torso center:       250mm
Required mast extension:                   480mm → rounded to 500mm
```

### 3.6 Z-Fold Segment Design

| Parameter              | Value                                           |
| ---------------------- | ----------------------------------------------- |
| **Total mast length**  | 500mm                                           |
| **Number of segments** | 3                                               |
| **Segment length**     | ~167mm each                                     |
| **Stow height**        | ~170mm (segments folded flat + hinge clearance) |
| **Stow profile**       | Fits flat against rear plate carrier panel      |

### 3.7 Derived Turret Position

When fully deployed:

- **Turret position in body frame:** (0.0, 0.0, +0.75m) above torso center
- This is mast_base_height (0.25m) + mast_length (0.50m)
- The turret is on the body centerline (X=0, Y=0)

---

## 4. Motor Controller Feedback

### 4.1 What the Motor Controller Reports

The mast motor controller publishes `ArmStatus` at ~50Hz on `predator/arm/status`:

| Field                 | Type     | Purpose                                                                  |
| --------------------- | -------- | ------------------------------------------------------------------------ |
| `deployment_state`    | enum     | Stowed / Deploying / Deployed / Stowing / Fault                          |
| `is_stable`           | bool     | Position settled, no vibration (encoder error < threshold for N samples) |
| `mast_height_m`       | f64      | Actual measured extension height from encoders                           |
| `joint_angles_deg`    | [f64; 3] | Angle of each Z-fold segment (0=base, 1=mid, 2=top)                      |
| `deployment_progress` | f64      | 0.0 = stowed, 1.0 = fully deployed                                       |
| `motor_current_amps`  | f64      | For thermal monitoring and stall detection                               |
| `motor_temp_celsius`  | f64      | Thermal protection threshold                                             |
| `fault_code`          | u16      | Vendor-specific error code (0 = no fault)                                |

### 4.2 Engagement Safety Gate

The orchestrator **blocks laser engagement** unless both conditions are met:

1. `deployment_state == Deployed`
2. `is_stable == true`

This is a hard gate — no overrides. If the mast faults during engagement, the `arm_deployment_confirmed` flag is immediately revoked and laser fire is blocked.

### 4.3 Deployment States

```
STOWED ──deploy──► DEPLOYING ──lock──► DEPLOYED
                                          │
STOWED ◄──lock──── STOWING ◄──stow───────┘
                                          │
                   FAULT ◄──fault─────────┘
```

### 4.4 Fault Handling

Any transition to `Fault` state:

1. Immediately clears `arm_deployment_confirmed`
2. Blocks engagement regardless of threat score
3. Logs `MAST FAULT detected — blocking engagement`
4. Requires manual reset or successful re-deploy to clear

---

## 5. Implementation Status

### 5.1 Code Changes

| File                                         | Change                                                                 |
| -------------------------------------------- | ---------------------------------------------------------------------- |
| `predator-messages/src/lib.rs`               | Removed `ArmShoulder`, simplified arm types, added `BodyFrameGeometry` |
| `predator-orchestrator/src/parallax.rs`      | **NEW** — radar-to-turret parallax correction                          |
| `predator-orchestrator/src/state_machine.rs` | Removed shoulder logic, renamed to `request_mast_deploy()`             |
| `predator-orchestrator/src/main.rs`          | Added `BodyFrameGeometry` config, updated arm status handler           |

### 5.2 Test Coverage

| Suite                                    | Count  | Status       |
| ---------------------------------------- | ------ | ------------ |
| Rust messages (serialization, geometry)  | 7      | ✅            |
| Rust parallax correction (4 range tiers) | 7      | ✅            |
| Rust state machine (core + mast gate)    | 16     | ✅            |
| Rust kinematic primer                    | 5      | ✅            |
| Rust doc-tests                           | 1      | ✅            |
| Python pipeline                          | 48     | ✅            |
| **Total**                                | **86** | **All pass** |

---

## 6. Configurable Parameters

All dimensions are stored in `BodyFrameGeometry` and calibrated per-operator:

```
radar_front_offset: [+0.12, 0.0, +0.25]  // top of front armor plate
radar_rear_offset:  [-0.12, 0.0, +0.25]  // top of rear armor plate
mast_base_height_m: 0.25                  // rear plate top
mast_length_m:      0.50                  // 3-segment Z-fold
```

Operators with different body sizes or helmet configurations adjust these values during plate carrier fitting. A tall operator with GPNVG-18 battery pack may need `mast_length_m: 0.55`.
