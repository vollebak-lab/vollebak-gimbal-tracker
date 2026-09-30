**Purpose: Define the spatial relationship between radar sensors and laser turret to enable accurate target handoff.**

---

## 1. Body-Frame Coordinate System

The Predator system uses a body-centered coordinate frame for all sensor-to-turret calculations.

### Origin & Axes

- **Origin:** Center of torso (sternum/spine midpoint at mid-chest height)
- **+X:** Forward (operator facing direction)
- **+Y:** Left (operator's left side)
- **+Z:** Up (vertical, aligned with gravity when operator is upright)

### Physical Layout

```
                    TURRET ← (0, 0, +0.75m)
                      │
                      │ 500mm mast (3 segments)
                      │
                  ────┬────  ← MAST BASE (0, 0, +0.25m)
                 /    │    \
        REAR    /     │     \    FRONT
        RADAR  ←  ── ─┼─ ── → RADAR
       (-0.12,      (0,0,0)    (+0.12,
        0,+0.25)  TORSO CENTER  0,+0.25)
                      │
```

### Sensor Positions

| Component             | Position [X, Y, Z] (m) | Notes                              |
| --------------------- | ---------------------- | ---------------------------------- |
| **Torso center**      | [0.0, 0.0, 0.0]        | Origin — sternum/spine midpoint    |
| **Front radar**       | [+0.12, 0.0, +0.25]    | Top of front armor plate, centered |
| **Rear radar**        | [−0.12, 0.0, +0.25]    | Top of rear armor plate, centered  |
| **Mast base**         | [0.0, 0.0, +0.25]      | Z-fold hinge on rear plate carrier |
| **Turret (deployed)** | [0.0, 0.0, +0.75]      | Mast base + 500mm mast extension   |

### Radar Module Placement

Both IWR6843AOP radar modules are mounted with the top of the module flush with the top of the armor plate region. This places them at approximately +0.25m above torso center.

The front radar faces forward (+X direction), the rear radar faces rearward (−X direction). This creates a 180° azimuth rotation between their local coordinate frames, which is handled as a known fixed rotation in the detection pipeline.

---

## 2. The Parallax Problem

### Why It Matters

The radar detects a target at (azimuth, elevation, range) **relative to the radar's position**. The laser turret needs to point at the same target **from a different physical position**. The angular difference between these two observations is **parallax**.

### Geometry

```
         TARGET at (az_r, el_r, range) from radar
            *
           /|\
          / | \
         /  |  \
        /   |   \
RADAR /    |    \ TURRET
  [R]      |      [T]
            |
    offset = R − T = Δx, Δy, Δz
```

The radar sees the target at one angle. The turret, displaced by (Δx, Δy, Δz), needs to point at a **different angle** to hit the same target.

### Magnitude at Range

For the Predator system, the turret is approximately 0.50m above the front radar:

| Range to Target | Turret-Radar Offset | Angular Error (uncorrected) | Impact                                      |
| --------------- | ------------------- | --------------------------- | ------------------------------------------- |
| 200m            | 0.50m               | 0.14°                       | Negligible — well within beam divergence    |
| 100m            | 0.50m               | 0.29°                       | Marginal — acceptable for 2-5W beam         |
| 50m             | 0.50m               | 0.57°                       | Noticeable — edge of beam coverage          |
| 20m             | 0.50m               | 1.43°                       | **Significant — must correct**              |
| 10m             | 0.50m               | 2.86°                       | **Critical — laser misses target entirely** |

**At close range (<20m), parallax correction is mandatory.** Without it, the laser fires at where the radar thinks the target is, not where the turret can actually hit it.

---

## 3. Correction Algorithm

Implemented in `predator-orchestrator/src/parallax.rs`.

### Step 1: Spherical → Cartesian (Radar Frame)

Convert the radar's (az, el, range) observation to a Cartesian target position in body frame:

```
Convention: azimuth 0° = forward (+X), clockwise looking down.
90° = right = −Y in body frame.

target_x = radar_x + range × cos(el) × cos(az)
target_y = radar_y − range × cos(el) × sin(az)
target_z = radar_z + range × sin(el)
```

### Step 2: Turret-to-Target Vector

Subtract the turret position to get the direction vector from turret to target:

```
Δx = target_x − turret_x    (turret_x = 0)
Δy = target_y − turret_y    (turret_y = 0)
Δz = target_z − turret_z    (turret_z = 0.75)
```

### Step 3: Cartesian → Spherical (Turret Frame)

Convert back to (az, el) for the gimbal command:

```
corrected_az = atan2(−Δy, Δx)        // CW convention
corrected_el = atan2(Δz, √(Δx² + Δy²))
```

### API

```
pub fn correct_for_parallax(
    radar_az_deg: f64,
    radar_el_deg: f64,
    range_m: f64,
    is_front_radar: bool,
    geometry: &BodyFrameGeometry,
) -> CorrectedPointing
```

Returns corrected (az, el) plus the delta corrections applied.

---

## 4. Elevation Correction Direction

An important physical intuition: the turret is **above** the radar. For a target at low positive elevation (e.g., 5° above horizon):

- The radar looks *up* at the target
- The turret, being higher, needs to look *less up* (or more down) to hit the same point
- Therefore: **corrected elevation < radar elevation**

This is verified by the `test_elevation_correction_direction` unit test.

---

## 5. Configuration

All geometry is stored in `BodyFrameGeometry` — a serializable struct with `Default` implementation for ANSUR 50th percentile male:

```
BodyFrameGeometry {
    radar_front_offset: [0.12, 0.0, 0.25],   // meters
    radar_rear_offset: [-0.12, 0.0, 0.25],    // meters
    mast_base_height_m: 0.25,                 // meters
    mast_length_m: 0.50,                      // meters
}
```

These values are:

- **Loaded at orchestrator startup** from configuration file
- **Calibrated per-operator** during plate carrier fitting
- **Adjustable at runtime** if needed for different load configurations

### Turret Position Helpers

```
// Turret position when fully deployed
geometry.turret_position()  // → [0.0, 0.0, 0.75]

// Turret position at partial mast extension (deploying)
geometry.turret_position_at_height(0.25)  // → [0.0, 0.0, 0.50]
```

---

## 6. Test Coverage

| Test                                     | Range | Assertion                                  |
| ---------------------------------------- | ----- | ------------------------------------------ |
| `test_long_range_negligible_parallax`    | 200m  | Az correction < 0.2°, El correction < 0.5° |
| `test_medium_range_moderate_parallax`    | 50m   | El correction < 2.0°                       |
| `test_close_range_significant_parallax`  | 20m   | El correction > 0.3°                       |
| `test_very_close_critical_parallax`      | 10m   | El correction > 1.0°                       |
| `test_rear_radar_different_offset`       | 50m   | Front/rear produce different corrections   |
| `test_elevation_correction_direction`    | 50m   | Corrected el < radar el (turret above)     |
| `test_broadside_target_azimuth`          | 50m   | 90° target stays approximately 90°         |
| `test_body_frame_geometry_defaults`      | —     | ANSUR defaults correct                     |
| `test_turret_position_at_partial_height` | —     | Partial deployment math correct            |
