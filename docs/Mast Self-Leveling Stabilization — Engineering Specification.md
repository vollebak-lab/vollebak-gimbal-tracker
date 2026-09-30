## 1. Problem Statement

When the operator moves (leans, crouches, runs), the rigidly-attached mast tilts with the body. This causes:

1. **Gimbal range saturation** — the gimbal uses its limited ±elevation budget compensating for body tilt instead of tracking targets
2. **Roll coupling** — a 2-axis gimbal (az/el) cannot fully decouple 3-DOF body rotation (yaw/pitch/roll). Roll-induced cross-axis errors corrupt the pointing solution.
3. **Dynamic parallax corruption** — turret position shifts in world frame, invalidating the static parallax correction that assumes a vertically-positioned turret

## 2. Solution: 2-Axis Self-Leveling Mast Base

Two Maxon EC 20 flat motors with 3-stage GP 22 gearheads (~350:1) at the mast base maintain vertical turret orientation independent of operator body movement.

### 2.1 Actuator Specification

| Component                                 | Qty | Unit Mass | Total     | Source                                                                                             |
| ----------------------------------------- | --- | --------- | --------- | -------------------------------------------------------------------------------------------------- |
| Maxon EC 20 flat 3W brushless             | 2   | 15g       | 30g       | [Maxon 339257](https://www.maxongroup.com/maxon/view/product/motor/ecmotor/ecflat/ecflat20/339257) |
| GP 22 planetary gearhead 3-stage (~350:1) | 2   | ~68g      | 136g      | Maxon GP 22 catalog                                                                                |
| Mounting hardware + bearings              | 2   | ~10g      | 20g       | Estimate                                                                                           |
| **Total stabilization subsystem**         |     |           | **~186g** |                                                                                                    |

**Output torque per axis:** ~1.0 Nm (3.07 mNm × 350:1 × ~93% efficiency)

**Required torque at 30° lean:** ~1.2 Nm for 500g turret at 0.5m arm → adequate for ±25° correction range

> **Note:** EC 20 flat is NRND (Not Recommended for New Designs). Production should migrate to modern equivalent (e.g., EC-i 22).

### 2.2 Bandwidth Partitioning

| Motion Component               | Frequency | Handler                            | Notes                       |
| ------------------------------ | --------- | ---------------------------------- | --------------------------- |
| Body lean/crouch (pitch, roll) | 0–5 Hz    | **Mast stabilization** (100Hz PID) | Large angle, slow           |
| Operator turning (yaw)         | 0–3 Hz    | **Gimbal azimuth**                 | Gimbal has 360° az range    |
| Target tracking                | 0–20 Hz   | **Gimbal PID** (200Hz)             | High bandwidth, small angle |
| Beam fine-pointing             | 0–200 Hz  | **Fast Steer Mirror**              | Sub-millidegree precision   |

The mast stabilization only handles **pitch and roll**. Yaw compensation stays in the gimbal azimuth axis, since the gimbal already has full 360° azimuth travel and yaw doesn't reduce the gimbal's elevation budget.

## 3. Design Rationale: Why Not Gimbal-Only?

| Criterion               | Gimbal-Only             | Mast Self-Leveling           |
| ----------------------- | ----------------------- | ---------------------------- |
| Roll compensation       | ❌ Cannot decouple       | ✅ Independent axis           |
| Gimbal range budget     | Consumed by body motion | Reserved for target tracking |
| Parallax correction     | Dynamic, complex        | Static when leveled          |
| Single point of failure | Gimbal overloaded       | Separated concerns           |
| Mass penalty            | 0g                      | ~186g                        |

**Decision:** Option B (self-leveling) adopted. The 186g mass penalty is negligible compared to the gimbal saturation risk.

## 4. Control Architecture

### 4.1 Stabilization PID Controller (100Hz)

- **Target:** 0° pitch, 0° roll (vertical)
- **Input:** IMU orientation (pitch, roll) from `ImuFeedForward`
- **Output:** Motor torque commands for pitch/roll axes
- **Leveling threshold:** 0.5° residual error for `is_leveled = true`
  - At 100m range: 0.5° = 0.87m error → within 2-5W beam dwell spot
- **Max correction:** ±25° (based on actuator torque budget)

### 4.2 Fault Detection

| Fault       | Detection                          | Response                     |
| ----------- | ---------------------------------- | ---------------------------- |
| Overcurrent | Motor current > 0.8A               | Fault code, motors unpowered |
| Stall       | High error + high current for 0.5s | Fault code, motors unpowered |
| IMU invalid | `is_valid == false`                | Fault code, hold position    |

### 4.3 Operating Modes

- **Idle:** Motors unpowered (mast stowed or system SILENT)
- **Stabilize:** Active leveling using IMU feedback (normal operation)
- **Override:** Manual pitch/roll setpoint (calibration/testing)

## 5. Engagement Gate Integration

The laser engagement gate now requires **three** conditions:

1. ✅ Arm deployed + stable
2. ✅ Threat score ≥ threshold
3. ✅ Mast leveled **OR** stabilization fault (degraded mode)

### 5.1 Degraded Mode

When stabilization is faulted, the system **allows engagement** with gimbal-only compensation. This is a conscious decision:

- A stabilization fault should not prevent self-defense
- The gimbal can partially compensate (reduced capability)
- A warning is logged for post-engagement review
- Dynamic parallax correction automatically uses the degraded path

## 6. Dynamic Parallax Correction

When the operator tilts, both radar and turret positions shift in world frame:

1. Radar position: `radar_world = R_body × radar_body` (tilts with body)
2. Turret position: `turret_world = R_body × base + R_residual × extension` (stabilized)
3. When residual < 0.1° and body tilt < 0.1°: falls through to fast static path

The turret position decomposition:

- **Mast base** (0.25m above torso center) rotates with body
- **Mast extension** (0.50m Z-fold) rotates only by stabilization residual error

## 7. Zenoh Message Schema

### New Topics

| Topic                                 | Direction         | Type                       | Rate      |
| ------------------------------------- | ----------------- | -------------------------- | --------- |
| `predator/mast/stabilization/command` | Orchestrator → HW | `MastStabilizationCommand` | On-demand |
| `predator/mast/stabilization/status`  | HW → Orchestrator | `MastStabilizationStatus`  | ~100Hz    |

### ImuFeedForward Extension

Added `delta_roll_deg` and `angular_rate_roll_dps` fields to enable 3-axis operator motion tracking.

## 8. Test Coverage

| Module                                   | Tests  | Status     |
| ---------------------------------------- | ------ | ---------- |
| predator-messages                        | 11     | ✅ All pass |
| mast_stabilization                       | 11     | ✅ All pass |
| parallax (incl. dynamic)                 | 10     | ✅ All pass |
| state_machine (incl. stabilization gate) | 20     | ✅ All pass |
| kinematic_primer                         | 5      | ✅ All pass |
| Doc-tests                                | 1      | ✅ Pass     |
| **Total**                                | **62** | ✅          |


