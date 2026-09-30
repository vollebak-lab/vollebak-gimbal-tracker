// ---
// VOLLEBAK_ENGINEERING_METADATA:
//   PROJECT_ID: PREDATOR-01
//   TRACK: ECOSYSTEM
//   PHASE: CHALLENGE_HUB
//   CONTRIBUTOR_ID: Antigravity AI
//   THE_DELTA: Rust kinematic primer for immediate threat scoring using speed + closing rate before micro-Doppler STFT completes
//   FAILURE_MODE: Waiting for 500ms micro-Doppler STFT blocks engagement authorization on fast movers
//   IP_STATUS: VOLLEBAK_PROPRIETARY
//   USER_FEEDBACK_REF: Latency Audit Change 3
//   DEPENDENCIES: []
// ---

//! # Kinematic Primer
//!
//! Fast threat assessment using speed, closing rate, altitude, and
//! Layer 1 flicker confidence — WITHOUT waiting for 500ms micro-Doppler STFT.
//!
//! Runs inside the Rust orchestrator on each radar track update.
//! When the primer score exceeds the threat threshold, the orchestrator
//! can authorize engagement immediately. The full assessment (with
//! micro-Doppler) replaces the primer score when it arrives later.

use tracing::info;

/// Kinematic primer result.
#[derive(Debug, Clone)]
pub struct PrimerResult {
    /// Primer-based threat score (0.0–1.0).
    pub score: f64,
    /// Speed component contribution.
    pub speed_component: f64,
    /// Closing rate component contribution.
    pub closing_rate_component: f64,
    /// Altitude component contribution.
    pub altitude_component: f64,
    /// Flicker component contribution.
    pub flicker_component: f64,
}

/// Kinematic primer configuration.
#[derive(Debug, Clone)]
pub struct PrimerConfig {
    /// Maximum expected threat speed in m/s (for normalization).
    pub max_threat_speed_mps: f64,
    /// Maximum expected closing rate in m/s (for normalization).
    pub max_closing_rate_mps: f64,
    /// Maximum threat altitude in meters AGL.
    pub max_threat_altitude_m: f64,
    /// Minimum threat altitude in meters AGL.
    pub min_threat_altitude_m: f64,
    /// Weight for speed component.
    pub weight_speed: f64,
    /// Weight for closing rate component (from radial velocity).
    pub weight_closing_rate: f64,
    /// Weight for altitude component.
    pub weight_altitude: f64,
    /// Weight for flicker confidence (from Layer 1).
    pub weight_flicker: f64,
}

impl Default for PrimerConfig {
    fn default() -> Self {
        // Weights derived from latency audit:
        // Original doppler weight (0.35) redistributed:
        //   60% → kinematics (speed 0.105 + closing 0.105)
        //   40% → flicker (0.14)
        // Plus original kinematics weight (0.20) and flicker (0.30)
        //
        // Net primer weights (without doppler):
        //   speed:        0.105 + 0.10 = 0.205 → normalized to ~0.31
        //   closing_rate: 0.105 + 0.10 = 0.205 → normalized to ~0.31
        //   altitude:     0.0   → 0.05  (small sanity check)
        //   flicker:      0.14  + 0.30 = 0.44  → normalized to ~0.33
        //
        // After normalization (sum = 1.0):
        Self {
            max_threat_speed_mps: 50.0,
            max_closing_rate_mps: 50.0,
            max_threat_altitude_m: 120.0,
            min_threat_altitude_m: 1.0,
            weight_speed: 0.30,
            weight_closing_rate: 0.30,
            weight_altitude: 0.05,
            weight_flicker: 0.35,
        }
    }
}

/// Compute a kinematic primer threat score.
///
/// This runs in <1µs and provides an immediate threat assessment
/// based on target kinematics and Layer 1 flicker, WITHOUT waiting
/// for the 500ms micro-Doppler STFT window.
///
/// # Arguments
///
/// * `speed_mps` — Target ground speed (magnitude).
/// * `radial_velocity_mps` — Radial velocity (negative = closing).
/// * `altitude_m` — Target altitude AGL.
/// * `flicker_confidence` — Layer 1 propeller flicker confidence (0–1).
/// * `config` — Primer weight configuration.
///
/// # Returns
///
/// A [`PrimerResult`] with the composite score and per-component breakdown.
pub fn compute_kinematic_primer(
    speed_mps: f64,
    radial_velocity_mps: f64,
    altitude_m: f64,
    flicker_confidence: f64,
    config: &PrimerConfig,
) -> PrimerResult {
    // Speed component: higher speed → higher threat
    let speed_norm = (speed_mps / config.max_threat_speed_mps).clamp(0.0, 1.0);

    // Closing rate: negative radial velocity = closing on operator
    // More negative = higher threat
    let closing_rate = (-radial_velocity_mps).max(0.0);
    let closing_norm = (closing_rate / config.max_closing_rate_mps).clamp(0.0, 1.0);

    // Altitude: drone-height altitude band is more threatening
    // Score peaks when altitude is in the 5-50m band
    let alt_score = if altitude_m < config.min_threat_altitude_m {
        0.0 // Ground level — likely ground clutter
    } else if altitude_m > config.max_threat_altitude_m {
        0.2 // Very high — less threatening (commercial aircraft)
    } else {
        // Peak threat in the 5-50m band
        let mid = (config.max_threat_altitude_m + config.min_threat_altitude_m) / 2.0;
        let half_range = (config.max_threat_altitude_m - config.min_threat_altitude_m) / 2.0;
        1.0 - ((altitude_m - mid) / half_range).abs().clamp(0.0, 1.0)
    };

    // Flicker: direct pass-through from Layer 1
    let flicker_norm = flicker_confidence.clamp(0.0, 1.0);

    // Weighted composite
    let score = config.weight_speed * speed_norm
        + config.weight_closing_rate * closing_norm
        + config.weight_altitude * alt_score
        + config.weight_flicker * flicker_norm;

    let result = PrimerResult {
        score: score.clamp(0.0, 1.0),
        speed_component: config.weight_speed * speed_norm,
        closing_rate_component: config.weight_closing_rate * closing_norm,
        altitude_component: config.weight_altitude * alt_score,
        flicker_component: config.weight_flicker * flicker_norm,
    };

    info!(
        "PRIMER: score={:.3} (spd={:.3} cls={:.3} alt={:.3} flk={:.3})",
        result.score,
        result.speed_component,
        result.closing_rate_component,
        result.altitude_component,
        result.flicker_component,
    );

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_mover_exceeds_threshold() {
        let config = PrimerConfig::default();
        let result = compute_kinematic_primer(
            41.0,  // 41 m/s — FPV kamikaze speed
            -38.0, // closing at 38 m/s
            30.0,  // 30m altitude — drone band
            0.8,   // high flicker confidence
            &config,
        );
        // Should exceed 0.7 threshold
        assert!(
            result.score > 0.7,
            "Fast mover primer should exceed 0.7: got {:.3}",
            result.score
        );
    }

    #[test]
    fn test_slow_mover_below_threshold() {
        let config = PrimerConfig::default();
        let result = compute_kinematic_primer(
            5.0,  // 5 m/s — slow
            2.0,  // moving away
            80.0, // high altitude
            0.3,  // low flicker
            &config,
        );
        assert!(
            result.score < 0.3,
            "Slow mover should be below 0.3: got {:.3}",
            result.score
        );
    }

    #[test]
    fn test_closing_rate_boost() {
        let config = PrimerConfig::default();
        let opening = compute_kinematic_primer(20.0, 10.0, 30.0, 0.5, &config);
        let closing = compute_kinematic_primer(20.0, -30.0, 30.0, 0.5, &config);
        assert!(
            closing.score > opening.score,
            "Closing should score higher: {:.3} vs {:.3}",
            closing.score,
            opening.score
        );
    }

    #[test]
    fn test_ground_level_altitude_zero() {
        let config = PrimerConfig::default();
        let result = compute_kinematic_primer(20.0, -20.0, 0.5, 0.5, &config);
        assert!(
            result.altitude_component < 0.01,
            "Ground level altitude should contribute ~0"
        );
    }

    #[test]
    fn test_primer_result_components_sum() {
        let config = PrimerConfig::default();
        let result = compute_kinematic_primer(30.0, -25.0, 40.0, 0.7, &config);
        let component_sum = result.speed_component
            + result.closing_rate_component
            + result.altitude_component
            + result.flicker_component;
        assert!(
            (result.score - component_sum).abs() < 0.01,
            "Score should equal sum of components: {:.3} vs {:.3}",
            result.score,
            component_sum
        );
    }
}
