**System: 3-Segment Z-Fold Vertical Mast**  
**Interface: Zenoh topic** `predator/arm/status` @ ~50Hz

---

## 1. Mechanical Specification

### Z-Fold Mast Parameters

| Parameter                 | Value                    | Notes                                 |
| ------------------------- | ------------------------ | ------------------------------------- |
| **Total extended length** | 500mm (0.50m)            | Clears IHPS + IR beacon + 50mm margin |
| **Number of segments**    | 3                        | Equal-length segments                 |
| **Segment length**        | ~167mm each              | 500mm / 3                             |
| **Stow profile height**   | ~170mm                   | Folded flat + hinge clearance         |
| **Stow location**         | Rear plate carrier panel | Flat against back plate               |
| **Extension axis**        | Vertical (Z only)        | Single-axis deployment                |
| **Mast base height**      | 250mm above torso center | Top of rear armor plate               |

### Clearance Budget

The 500mm mast length was derived from a worst-case clearance stack analysis:

**Source data:**

- **ANSUR (Anthropometric Survey of U.S. Army Personnel)** — 50th percentile male body dimensions
- **TM 10-8470-204-10** — ACH operator's manual (helmet fit dimensions)
- **ACR Electronics** — MS2000 strobe physical specifications
- **PEO Soldier** — IHPS (Integrated Head Protection System) program reference

**Clearance stack:**

| Layer                         | From Torso Center (mm) | Dimension (mm)        | Source                                  |
| ----------------------------- | ---------------------- | --------------------- | --------------------------------------- |
| Torso center → shoulder top   | 0 → 200                | 200                   | ANSUR 50th %ile male                    |
| Shoulder → head crown         | 200 → 430              | 230                   | ANSUR head height (232mm)               |
| Head crown → helmet shell top | 430 → 580              | 130–150               | ACH/IHPS shell + 19mm pad standoff      |
| Helmet → IR beacon peak       | 580 → 613              | 28–33                 | MS2000: 28–33mm profile (ACR spec)      |
| IR beacon → safety clearance  | 613 → 663              | 50                    | Head tilt ±15° during tactical movement |
| **Total clearance needed**    |                        | **663mm**             |                                         |
| **Mast base position**        |                        | **250mm**             | Top of rear plate                       |
| **Required extension**        |                        | **413–480mm → 500mm** | Rounded up for margin                   |

### Helmet Models Considered

| Helmet   | Variant               | Used By                            |
| -------- | --------------------- | ---------------------------------- |
| **MICH** | TC-2000               | Legacy US Army/USMC                |
| **ACH**  | Standard              | US Army standard issue             |
| **ECH**  | Standard              | USMC primary                       |
| **IHPS** | Full-cut (worst case) | Next-gen US Army — tallest profile |

The IHPS with optional mandible guard represents the tallest helmet profile in current US military inventory and was used as the design reference.

### IR Beacon Models Considered

| Beacon                  | Height  | Mount Style                           |
| ----------------------- | ------- | ------------------------------------- |
| **ACR MS2000**          | 28–33mm | Top-mount velcro, rectangular profile |
| **S&S Precision Manta** | ~15mm   | Low-profile, conforms to helmet       |
| **Phoenix IR**          | ~20mm   | Flat profile, velcro mount            |

The MS2000 is the worst-case (tallest) and was used as the reference.

---

## 2. Motor Controller Interface

### 2.1 Zenoh Topics

| Topic                  | Direction                       | Rate      | Payload                |
| ---------------------- | ------------------------------- | --------- | ---------------------- |
| `predator/arm/command` | Orchestrator → Motor Controller | On-demand | `ArmCommand` (msgpack) |
| `predator/arm/status`  | Motor Controller → Orchestrator | ~50Hz     | `ArmStatus` (msgpack)  |

### 2.2 ArmCommand (Orchestrator → Controller)

```
struct ArmCommand {
    action: ArmAction,      // Deploy or Stow
    timestamp_us: u64,
}

enum ArmAction {
    Deploy,   // Extend mast from Z-fold to full vertical
    Stow,     // Retract mast to Z-fold stow
}
```

No shoulder parameter — the mast always extends vertically.

### 2.3 ArmStatus (Controller → Orchestrator)

Published at ~50Hz by the motor controller:

```
struct ArmStatus {
    deployment_state: ArmDeploymentState,  // Stowed/Deploying/Deployed/Stowing/Fault
    is_stable: bool,                       // Vibration settled
    mast_height_m: f64,                    // Actual encoder-measured extension
    joint_angles_deg: Vec<f64>,            // [base, mid, top] segment angles
    deployment_progress: f64,              // 0.0 → 1.0
    motor_current_amps: f64,               // Thermal monitoring
    motor_temp_celsius: f64,               // Thermal protection
    fault_code: u16,                       // 0 = no fault
    timestamp_us: u64,                     // Hardware timestamp
}
```

### 2.4 Deployment State Machine

```
          ┌─────────────────────────────────────┐
          ▼                                     │
       STOWED ──deploy──► DEPLOYING ──lock──► DEPLOYED
          ▲                                     │
          │                                     │
       STOWED ◄──lock──── STOWING ◄──stow──────┘
                              │
                              ├──fault──► FAULT
                              │
       DEPLOYING ─────────────┘
```

### 2.5 Stability Determination

The `is_stable` flag is set by the motor controller when:

1. All three joint encoders report position error < threshold
2. This condition has held for N consecutive samples (settling time)
3. The deployment state is `Deployed`

Stability is cleared (false) when:

- The mast is in any state other than `Deployed`
- Motor vibration exceeds threshold
- Any joint encoder reports drift

---

## 3. Engagement Safety Gate

### 3.1 Gate Logic

The orchestrator enforces a **hard engagement gate** that blocks the transition from `TRACKING` to `ENGAGEMENT`:

```
ENGAGEMENT requires BOTH:
  1. threat_score >= 0.7 (min_threat_score)
  2. arm_deployment_confirmed == true
     └── requires: deployment_state == Deployed AND is_stable == true
```

This gate is **not bypassable**. There is no override mechanism.

### 3.2 Late Confirmation Flow

If the threat score exceeds threshold before the mast is ready:

1. Threat score arrives → stored in `current_threat_score`
2. State remains `TRACKING` (gate blocked)
3. Warning logged: "Threat score X exceeds threshold but mast not confirmed"
4. Mast subsequently reports `Deployed + Stable`
5. `on_arm_status()` re-evaluates gate
6. If score still >= threshold → immediate transition to `ENGAGEMENT`

### 3.3 Fault Revocation

If a fault occurs at any time:

1. `arm_deployment_confirmed` immediately set to `false`
2. Any pending or active engagement is blocked
3. If currently in `ENGAGEMENT`, the BDA timeout will expire and the system returns to safe state
4. Cannot re-enter `ENGAGEMENT` until fault is cleared and mast re-deploys successfully

### 3.4 Test Coverage

| Test                                                 | Scenario                              | Expected             |
| ---------------------------------------------------- | ------------------------------------- | -------------------- |
| `test_threat_score_blocked_without_arm_confirmation` | Threat score high, mast not confirmed | Stays in TRACKING    |
| `test_threat_score_triggers_engagement_with_arm`     | Threat score high, mast confirmed     | Enters ENGAGEMENT    |
| `test_arm_confirmation_releases_gate`                | Score stored, then mast confirms      | Immediate ENGAGEMENT |
| `test_arm_deploying_does_not_release_gate`           | Mast deploying (not deployed)         | Stays in TRACKING    |
| `test_arm_deployed_but_unstable_blocks_gate`         | Deployed but vibrating                | Stays in TRACKING    |
| `test_arm_fault_blocks_engagement`                   | Confirmed → fault                     | Confirmation revoked |
| `test_arm_status_stowed`                             | Stowed (even if stable)               | Not confirmed        |

All 7 gate tests pass.

---

## 4. Thermal Protection

The motor controller monitors two thermal signals:

### Motor Current

- **Field:** `motor_current_amps`
- **Purpose:** Detect stall conditions (motor jammed, mast obstructed)
- **Action:** If sustained overcurrent → transition to `Fault` state

### Motor Temperature

- **Field:** `motor_temp_celsius`
- **Purpose:** Prevent thermal damage to actuator coils
- **Action:** If temperature exceeds vendor limit → transition to `Fault` state
- **Note:** Multiple rapid deploy/stow cycles in hot environments may trigger thermal protection

---

## 5. Integration Requirements

### For Motor Controller Firmware Team

The motor controller must:

1. Publish `ArmStatus` at ~50Hz on `predator/arm/status` using MessagePack serialization
2. Report `Deployed + is_stable=true` only when all encoder positions are within threshold
3. Report `Fault` immediately on overcurrent, encoder loss, or thermal limit
4. Accept `ArmCommand` on `predator/arm/command` (Deploy/Stow only)
5. Implement hardware limit switches at full extension and full retraction
6. Provide `mast_height_m` from encoder readout (not derived from deploy command)
