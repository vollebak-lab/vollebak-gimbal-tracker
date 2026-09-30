// ---
// VOLLEBAK_ENGINEERING_METADATA:
//   PROJECT_ID: PREDATOR-01
//   TRACK: ECOSYSTEM
//   PHASE: CHALLENGE_HUB
//   CONTRIBUTOR_ID: Antigravity AI
//   THE_DELTA: Radar-to-turret parallax correction for body-frame coordinate transform
//   FAILURE_MODE: At close range (<20m), uncorrected parallax causes 1-3° pointing error → laser miss
//   IP_STATUS: VOLLEBAK_PROPRIETARY
//   DEPENDENCIES: [predator-messages]
// ---

//! # Radar-to-Turret Parallax Correction
//!
//! Corrects for the physical offset between radar modules and the
//! turret (laser) at the top of the vertical mast.
//!
//! ## Coordinate Frame
//!
//! Body-centered frame with origin at torso center:
//! - +X = forward (operator facing direction)
//! - +Y = left (operator's left)
//! - +Z = up (vertical)
//!
//! ## When Parallax Matters
//!
//! | Range | Offset (0.5m) | Angular Error |
//! |-------|---------------|---------------|
//! | 200m  | 0.14°         | Negligible    |
//! | 50m   | 0.57°         | Marginal      |
//! | 20m   | 1.43°         | Significant   |
//! | 10m   | 2.86°         | **Critical**  |

use predator_messages::BodyFrameGeometry;

/// Result of a parallax-corrected pointing computation.
#[derive(Debug, Clone, Copy)]
pub struct CorrectedPointing {
    /// Corrected azimuth for the turret (degrees, 0 = fwd, CW).
    pub azimuth_deg: f64,
    /// Corrected elevation for the turret (degrees, +up).
    pub elevation_deg: f64,
    /// Parallax correction magnitude in azimuth (degrees).
    pub delta_az_deg: f64,
    /// Parallax correction magnitude in elevation (degrees).
    pub delta_el_deg: f64,
}

/// Convert a radar detection to corrected turret pointing angles.
///
/// Given a target observed by a radar at `radar_pos` in body frame,
/// compute the azimuth and elevation that the turret at `turret_pos`
/// needs to point to hit the same target.
///
/// # Arguments
///
/// * `radar_az_deg` — Azimuth as seen by the radar (0° = fwd, CW)
/// * `radar_el_deg` — Elevation as seen by the radar (+up)
/// * `range_m` — Slant range from radar to target
/// * `radar_pos` — Radar position [x, y, z] in body frame (meters)
/// * `turret_pos` — Turret position [x, y, z] in body frame (meters)
///
/// # Returns
///
/// [`CorrectedPointing`] with corrected azimuth/elevation for the turret
/// and the delta corrections applied.
pub fn radar_to_turret(
    radar_az_deg: f64,
    radar_el_deg: f64,
    range_m: f64,
    radar_pos: [f64; 3],
    turret_pos: [f64; 3],
) -> CorrectedPointing {
    // Convert radar observation from spherical to Cartesian.
    //
    // Convention: azimuth 0° = +X (forward), 90° = +Y? No.
    // Convention: azimuth 0° = forward (+X), CW looking down.
    // So 90° = right = -Y in body frame.
    //
    //   x = range * cos(el) * cos(az)
    //   y = -range * cos(el) * sin(az)   [CW azimuth → -Y for right]
    //   z = range * sin(el)
    let az_rad = radar_az_deg.to_radians();
    let el_rad = radar_el_deg.to_radians();

    let cos_el = el_rad.cos();
    let target_x = radar_pos[0] + range_m * cos_el * az_rad.cos();
    let target_y = radar_pos[1] - range_m * cos_el * az_rad.sin();
    let target_z = radar_pos[2] + range_m * el_rad.sin();

    // Vector from turret to target.
    let dx = target_x - turret_pos[0];
    let dy = target_y - turret_pos[1];
    let dz = target_z - turret_pos[2];

    let ground_range = (dx * dx + dy * dy).sqrt();

    // Corrected azimuth: atan2(-dy, dx) to maintain CW convention.
    // Because our convention maps positive azimuth → -Y,
    // we invert: az = atan2(-dy, dx).
    let corrected_az_rad = (-dy).atan2(dx);
    let corrected_el_rad = dz.atan2(ground_range);

    let corrected_az_deg = corrected_az_rad.to_degrees();
    let corrected_el_deg = corrected_el_rad.to_degrees();

    CorrectedPointing {
        azimuth_deg: corrected_az_deg,
        elevation_deg: corrected_el_deg,
        delta_az_deg: corrected_az_deg - radar_az_deg,
        delta_el_deg: corrected_el_deg - radar_el_deg,
    }
}

/// Convenience: compute corrected pointing using a `BodyFrameGeometry`
/// and a flag indicating which radar observed the target.
pub fn correct_for_parallax(
    radar_az_deg: f64,
    radar_el_deg: f64,
    range_m: f64,
    is_front_radar: bool,
    geometry: &BodyFrameGeometry,
) -> CorrectedPointing {
    let radar_pos = if is_front_radar {
        geometry.radar_front_offset
    } else {
        geometry.radar_rear_offset
    };
    let turret_pos = geometry.turret_position();
    radar_to_turret(radar_az_deg, radar_el_deg, range_m, radar_pos, turret_pos)
}

/// IMU-aware parallax correction accounting for operator body tilt
/// and mast stabilization residual error.
///
/// When stabilization is active and `residual_pitch_deg` /
/// `residual_roll_deg` are near zero, this produces the same result
/// as the static `correct_for_parallax()`. When the operator leans
/// (and stabilization has non-zero residual), both the radar and
/// turret positions are rotated into world frame before computing
/// the pointing correction.
///
/// # Arguments
///
/// * `radar_az_deg` — Azimuth as seen by the radar (body frame)
/// * `radar_el_deg` — Elevation as seen by the radar (body frame)
/// * `range_m` — Slant range from radar to target
/// * `is_front_radar` — Which radar module observed the target
/// * `geometry` — Body-frame geometry configuration
/// * `body_pitch_deg` — Current body pitch from IMU (degrees from vertical)
/// * `body_roll_deg` — Current body roll from IMU (degrees from vertical)
/// * `residual_pitch_deg` — Stabilization residual pitch error (degrees)
/// * `residual_roll_deg` — Stabilization residual roll error (degrees)
pub fn correct_for_parallax_dynamic(
    radar_az_deg: f64,
    radar_el_deg: f64,
    range_m: f64,
    is_front_radar: bool,
    geometry: &BodyFrameGeometry,
    body_pitch_deg: f64,
    body_roll_deg: f64,
    residual_pitch_deg: f64,
    residual_roll_deg: f64,
) -> CorrectedPointing {
    // If residual is negligible, use the fast static path
    if residual_pitch_deg.abs() < 0.1 && residual_roll_deg.abs() < 0.1
        && body_pitch_deg.abs() < 0.1 && body_roll_deg.abs() < 0.1
    {
        return correct_for_parallax(radar_az_deg, radar_el_deg, range_m, is_front_radar, geometry);
    }

    // Rotate radar position into world frame (radar tilts with body)
    let radar_body = if is_front_radar {
        geometry.radar_front_offset
    } else {
        geometry.radar_rear_offset
    };
    let radar_world = geometry.radar_position_world(radar_body, body_pitch_deg, body_roll_deg);

    // Turret position in world frame (stabilized, but with residual)
    let turret_world = geometry.turret_position_world(
        body_pitch_deg,
        body_roll_deg,
        residual_pitch_deg,
        residual_roll_deg,
    );

    radar_to_turret(radar_az_deg, radar_el_deg, range_m, radar_world, turret_world)
}


#[cfg(test)]
mod tests {
    use super::*;

    fn default_geometry() -> BodyFrameGeometry {
        BodyFrameGeometry::default()
    }

    #[test]
    fn test_long_range_negligible_parallax() {
        // At 200m, parallax should be < 0.2°
        let result = correct_for_parallax(0.0, 5.0, 200.0, true, &default_geometry());
        assert!(
            result.delta_az_deg.abs() < 0.2,
            "Az parallax at 200m should be negligible: {:.4}°",
            result.delta_az_deg
        );
        assert!(
            result.delta_el_deg.abs() < 0.5,
            "El parallax at 200m should be small: {:.4}°",
            result.delta_el_deg
        );
    }

    #[test]
    fn test_medium_range_moderate_parallax() {
        // At 50m, parallax should be noticeable but < 1°
        let result = correct_for_parallax(0.0, 5.0, 50.0, true, &default_geometry());
        assert!(
            result.delta_el_deg.abs() < 2.0,
            "El parallax at 50m: {:.4}°",
            result.delta_el_deg
        );
    }

    #[test]
    fn test_close_range_significant_parallax() {
        // At 20m, parallax should be > 0.5° — must correct
        let result = correct_for_parallax(0.0, 5.0, 20.0, true, &default_geometry());
        assert!(
            result.delta_el_deg.abs() > 0.3,
            "El parallax at 20m should be significant: {:.4}°",
            result.delta_el_deg
        );
    }

    #[test]
    fn test_very_close_critical_parallax() {
        // At 10m, parallax should be > 1° — laser misses without correction
        let result = correct_for_parallax(0.0, 10.0, 10.0, true, &default_geometry());
        assert!(
            result.delta_el_deg.abs() > 1.0,
            "El parallax at 10m should be critical: {:.4}°",
            result.delta_el_deg
        );
    }

    #[test]
    fn test_rear_radar_different_offset() {
        let geom = default_geometry();
        // Use 30° azimuth so the X-offset difference between
        // front (+0.12m) and rear (-0.12m) radar creates a
        // measurable azimuth correction difference.
        let front = correct_for_parallax(30.0, 5.0, 50.0, true, &geom);
        let rear = correct_for_parallax(30.0, 5.0, 50.0, false, &geom);
        // Front and rear radars at different X positions should give
        // different azimuth corrections at non-zero azimuth.
        assert!(
            (front.azimuth_deg - rear.azimuth_deg).abs() > 0.001,
            "Front and rear radar should produce different az: front={:.4}° rear={:.4}°",
            front.azimuth_deg, rear.azimuth_deg
        );
    }

    #[test]
    fn test_elevation_correction_direction() {
        // Turret is above radar — for a target at low elevation,
        // the turret needs to look DOWN more than the radar does.
        // So corrected elevation should be less than radar elevation.
        let result = correct_for_parallax(0.0, 5.0, 50.0, true, &default_geometry());
        assert!(
            result.elevation_deg < 5.0,
            "Turret above radar should look down more: corrected_el={:.4}° vs radar_el=5.0°",
            result.elevation_deg
        );
    }

    #[test]
    fn test_broadside_target_azimuth() {
        // Target at 90° (right side), 50m
        let result = correct_for_parallax(90.0, 0.0, 50.0, true, &default_geometry());
        // Corrected azimuth should still be approximately 90°
        assert!(
            (result.azimuth_deg - 90.0).abs() < 2.0,
            "Broadside target az correction should be small: {:.4}°",
            result.azimuth_deg
        );
    }

    // -----------------------------------------------------------------------
    // Dynamic parallax tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dynamic_zero_tilt_matches_static() {
        let geom = default_geometry();
        let static_result = correct_for_parallax(10.0, 5.0, 50.0, true, &geom);
        let dynamic_result = correct_for_parallax_dynamic(
            10.0, 5.0, 50.0, true, &geom,
            0.0, 0.0,  // no body tilt
            0.0, 0.0,  // no residual
        );
        assert!(
            (static_result.azimuth_deg - dynamic_result.azimuth_deg).abs() < 0.01,
            "Zero-tilt dynamic should match static: static={:.4}° dynamic={:.4}°",
            static_result.azimuth_deg, dynamic_result.azimuth_deg
        );
        assert!(
            (static_result.elevation_deg - dynamic_result.elevation_deg).abs() < 0.01,
            "Zero-tilt dynamic el should match static: static={:.4}° dynamic={:.4}°",
            static_result.elevation_deg, dynamic_result.elevation_deg
        );
    }

    #[test]
    fn test_dynamic_with_residual_differs_from_static() {
        let geom = default_geometry();
        let static_result = correct_for_parallax(10.0, 5.0, 20.0, true, &geom);
        // 15° body lean, 2° stabilization residual
        let dynamic_result = correct_for_parallax_dynamic(
            10.0, 5.0, 20.0, true, &geom,
            15.0, 0.0,  // body pitched forward 15°
            2.0, 0.0,   // 2° residual (stabilization corrected 13°)
        );
        // At close range (20m), the body tilt + residual should produce
        // a measurably different correction
        let el_diff = (static_result.elevation_deg - dynamic_result.elevation_deg).abs();
        assert!(
            el_diff > 0.01,
            "Body lean with residual should differ from static: diff={:.4}°",
            el_diff
        );
    }

    #[test]
    fn test_dynamic_large_lean_azimuth_shift() {
        let geom = default_geometry();
        let static_result = correct_for_parallax(45.0, 5.0, 30.0, true, &geom);
        // 20° roll right → turret shifts laterally
        let dynamic_result = correct_for_parallax_dynamic(
            45.0, 5.0, 30.0, true, &geom,
            0.0, 20.0,  // 20° roll right
            0.0, 3.0,   // 3° residual roll
        );
        // Roll should shift the azimuth correction
        let az_diff = (static_result.azimuth_deg - dynamic_result.azimuth_deg).abs();
        assert!(
            az_diff > 0.01,
            "Roll should shift azimuth correction: diff={:.4}°",
            az_diff
        );
    }
}
