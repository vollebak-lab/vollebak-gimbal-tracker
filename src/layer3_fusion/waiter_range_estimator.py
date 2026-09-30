# ---
# VOLLEBAK_ENGINEERING_METADATA:
#   PROJECT_ID: PREDATOR-01
#   TRACK: ECOSYSTEM
#   PHASE: CHALLENGE_HUB
#   CONTRIBUTOR_ID: Antigravity AI
#   THE_DELTA: Ground-plane range estimation from camera height + elevation for L1-only passive engagement of Waiter drones
#   FAILURE_MODE: Without range estimation, L1-only engagements cannot be safety-gated (no radar range data available)
#   IP_STATUS: VOLLEBAK_PROPRIETARY
#   USER_FEEDBACK_REF: Passive Neuromorphic Engagement Architecture v3
#   DEPENDENCIES: [numpy]
# ---
"""
Waiter Range Estimator — Ground-Plane Geometry Engine.

Estimates range to ground-level targets using camera height + elevation
angle trigonometry when radar is not active (L1-only passive engagement).

Also computes:
- Posture-adjusted camera height from IMU pitch data
- ANSUR-bounded uncertainty on range estimates
- L1 velocity vector classification (ascending/descending/stationary)
- Intelligent 3-zone vertical sweep bounds for pulsed laser engagement

Kill chain timing (mast stowed): ~1.5–2.5 s
Kill chain timing (mast pre-deployed): ~400–700 ms

Pulsed laser physics: Q-switched nanosecond pulses produce megawatt
peak power. Single pulse entering drone camera lens = permanent CMOS
damage via 100,000× optical gain. Sweep maximizes probability of
lens aperture hit, NOT thermal dwell accumulation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RangeEstimate:
    """Ground-plane range estimate from camera geometry.

    Attributes:
        range_m: Estimated range to target in meters.
        uncertainty_m: Range uncertainty from ANSUR height bounds.
        is_valid: False if elevation >= 0 (above horizon — range
            indeterminate from ground-plane geometry).
        posture: Operator posture used for height ("standing",
            "crouching", "prone").
        camera_height_m: Active camera height used in computation.
    """
    range_m: float
    uncertainty_m: float
    is_valid: bool
    posture: str
    camera_height_m: float


@dataclass(frozen=True, slots=True)
class VelocityEstimate:
    """L1 centroid velocity classification.

    Attributes:
        vx_degps: Azimuth angular rate (°/s).
        vy_degps: Elevation angular rate (°/s).
        launch_vector: Classification of vertical motion:
            "ascending" — drone lifting off from ground
            "descending" — drone dropping from rooftop
            "stationary" — drone on ground, spinning up
    """
    vx_degps: float
    vy_degps: float
    launch_vector: str


@dataclass(frozen=True, slots=True)
class SweepBounds:
    """Vertical sweep zones for pulsed laser engagement.

    Zone 1 (PRIMARY): Ground-level sweep — highest priority.
    Zone 2 (EXCLUSION): Horizon band — skip (low value target region).
    Zone 3 (PURSUIT): Above-horizon — sweep if drone ascending.

    Attributes:
        zone1_el_min: Primary zone lower bound (degrees, negative).
        zone1_el_max: Primary zone upper bound (degrees, negative).
        zone3_el_min: Pursuit zone lower bound (degrees, positive).
            None if no pursuit zone needed.
        zone3_el_max: Pursuit zone upper bound (degrees, positive).
            None if no pursuit zone needed.
        exclusion_el_min: Exclusion band lower bound.
        exclusion_el_max: Exclusion band upper bound.
        azimuth_deg: Fixed azimuth for entire sweep column.
    """
    zone1_el_min: float
    zone1_el_max: float
    zone3_el_min: float | None
    zone3_el_max: float | None
    exclusion_el_min: float
    exclusion_el_max: float
    azimuth_deg: float


# ---------------------------------------------------------------------------
# Posture Constants
# ---------------------------------------------------------------------------

# IMU pitch thresholds for posture classification
_POSTURE_STANDING_MAX_PITCH = 15.0   # ±15° from upright
_POSTURE_CROUCHING_MAX_PITCH = 45.0  # 15°–45° forward lean
# Beyond 45° → prone

# Camera height multipliers relative to configured eye height
_POSTURE_HEIGHTS = {
    "standing": 1.0,    # Full height
    "crouching": 0.60,  # ~60% of standing height
    "prone": 0.20,      # ~20% of standing height (ground level)
}


# ---------------------------------------------------------------------------
# WaiterRangeEstimator
# ---------------------------------------------------------------------------

class WaiterRangeEstimator:
    """Estimate range to ground-level targets from camera geometry.

    Uses the fundamental relationship:
        range = camera_height / tan(|elevation_angle|)

    The camera height is the operator's configured eye height plus
    helmet camera offset, adjusted for posture via IMU pitch.

    ANSUR anthropometric bounds (5th–95th percentile with 10% buffer)
    provide uncertainty quantification on the range estimate.

    Args:
        camera_height_m: Configured camera height (eye + helmet offset).
        ansur_min_height_m: ANSUR lower bound (5th percentile × 0.9).
        ansur_max_height_m: ANSUR upper bound (95th percentile × 1.1).
        pixel_angular_res_deg: Angular resolution per pixel for
            elevation uncertainty estimation.
    """

    def __init__(
        self,
        camera_height_m: float = 1.85,
        ansur_min_height_m: float = 1.37,
        ansur_max_height_m: float = 1.79,
        pixel_angular_res_deg: float = 0.0854,
    ) -> None:
        self._configured_height_m = camera_height_m
        self._ansur_min_m = ansur_min_height_m
        self._ansur_max_m = ansur_max_height_m
        self._pixel_angular_res = pixel_angular_res_deg

        # Active camera height (adjusted by posture)
        self._active_height_m = camera_height_m
        self._posture = "standing"

    @property
    def camera_height_m(self) -> float:
        """Active camera height in meters (posture-adjusted)."""
        return self._active_height_m

    @property
    def posture(self) -> str:
        """Current operator posture classification."""
        return self._posture

    # -------------------------------------------------------------------
    # Range Estimation
    # -------------------------------------------------------------------

    def estimate_range(self, elevation_deg: float) -> RangeEstimate:
        """Estimate range to ground-level target from elevation angle.

        Uses: range = camera_height / tan(|elevation|)

        Rejects positive elevation (above horizon) — range is
        indeterminate from ground-plane geometry alone.

        Args:
            elevation_deg: Target elevation angle in degrees.
                Negative = below horizon (expected for ground targets).

        Returns:
            RangeEstimate with range, uncertainty, and validity.
        """
        # Reject above-horizon detections — ground-plane geometry
        # does not apply (target is not on the ground plane)
        if elevation_deg >= 0.0:
            return RangeEstimate(
                range_m=float("inf"),
                uncertainty_m=float("inf"),
                is_valid=False,
                posture=self._posture,
                camera_height_m=self._active_height_m,
            )

        # Reject near-zero elevation — tan approaches infinity,
        # range estimate becomes unreliable (> 1 km)
        abs_el = abs(elevation_deg)
        if abs_el < 0.1:
            return RangeEstimate(
                range_m=float("inf"),
                uncertainty_m=float("inf"),
                is_valid=False,
                posture=self._posture,
                camera_height_m=self._active_height_m,
            )

        # Core trigonometry: range = h / tan(|el|)
        el_rad = math.radians(abs_el)
        range_m = self._active_height_m / math.tan(el_rad)

        # Uncertainty from ANSUR height bounds
        # Compute range at min and max possible heights
        posture_factor = _POSTURE_HEIGHTS.get(self._posture, 1.0)
        h_min = self._ansur_min_m * posture_factor
        h_max = self._ansur_max_m * posture_factor
        range_min = h_min / math.tan(el_rad)
        range_max = h_max / math.tan(el_rad)
        uncertainty_m = (range_max - range_min) / 2.0

        return RangeEstimate(
            range_m=range_m,
            uncertainty_m=uncertainty_m,
            is_valid=True,
            posture=self._posture,
            camera_height_m=self._active_height_m,
        )

    # -------------------------------------------------------------------
    # Posture Adjustment
    # -------------------------------------------------------------------

    def update_posture(self, imu_pitch_deg: float) -> None:
        """Adjust camera height based on IMU-derived operator posture.

        Classifies posture from IMU pitch angle:
        - Standing: |pitch| < 15°  →  100% of configured height
        - Crouching: 15° ≤ |pitch| < 45°  →  60% of configured height
        - Prone: |pitch| ≥ 45°  →  20% of configured height

        Args:
            imu_pitch_deg: IMU pitch angle in degrees. 0° = upright,
                positive = forward lean.
        """
        abs_pitch = abs(imu_pitch_deg)

        if abs_pitch < _POSTURE_STANDING_MAX_PITCH:
            self._posture = "standing"
        elif abs_pitch < _POSTURE_CROUCHING_MAX_PITCH:
            self._posture = "crouching"
        else:
            self._posture = "prone"

        factor = _POSTURE_HEIGHTS[self._posture]
        self._active_height_m = self._configured_height_m * factor

        logger.debug(
            "Posture: %s (pitch=%.1f°, h=%.2fm)",
            self._posture, imu_pitch_deg, self._active_height_m,
        )

    # -------------------------------------------------------------------
    # Velocity Vector Analysis
    # -------------------------------------------------------------------

    @staticmethod
    def classify_velocity(
        centroid_vy_degps: float,
        threshold_degps: float = 5.0,
    ) -> VelocityEstimate:
        """Classify L1 centroid vertical velocity as launch vector.

        Positive vy = ascending (ground-placed Waiter lifting off).
        Negative vy = descending (rooftop-placed Waiter dropping).
        Near-zero vy = stationary (spinning up, not yet airborne).

        Args:
            centroid_vy_degps: Elevation angular rate from L1 tracker.
            threshold_degps: Minimum rate to classify as moving.

        Returns:
            VelocityEstimate with launch_vector classification.
        """
        if centroid_vy_degps > threshold_degps:
            launch_vector = "ascending"
        elif centroid_vy_degps < -threshold_degps:
            launch_vector = "descending"
        else:
            launch_vector = "stationary"

        return VelocityEstimate(
            vx_degps=0.0,  # Azimuth rate not used for sweep bounds
            vy_degps=centroid_vy_degps,
            launch_vector=launch_vector,
        )

    # -------------------------------------------------------------------
    # Sweep Bounds Computation
    # -------------------------------------------------------------------

    def compute_sweep_bounds(
        self,
        azimuth_deg: float,
        elevation_deg: float,
        range_est_m: float,
        velocity_vy_degps: float = 0.0,
        sweep_margin_deg: float = 5.0,
    ) -> SweepBounds:
        """Compute intelligent vertical sweep zones from geometry + velocity.

        Zone 1 (PRIMARY): Ground level ± margin. Sweep first.
        Zone 2 (EXCLUSION): Horizon band. Skip — low value.
        Zone 3 (PURSUIT): Above horizon, activated if drone ascending.

        At Waiter ranges (<50 m), the pulsed laser needs only angular
        coverage (not dwell time). Q-switched nanosecond pulses at
        megawatt peak power cause irreversible CMOS damage on a single
        hit through the drone's camera lens (100,000× optical gain).

        Args:
            azimuth_deg: Fixed azimuth for sweep column.
            elevation_deg: L1 estimated elevation to target (negative).
            range_est_m: Ground-plane range estimate.
            velocity_vy_degps: L1 centroid vertical rate (°/s).
            sweep_margin_deg: Margin around estimated elevation.

        Returns:
            SweepBounds with zone definitions.
        """
        # Zone 1: Primary sweep around detected elevation
        zone1_center = elevation_deg
        zone1_el_min = max(zone1_center - sweep_margin_deg, -20.0)
        zone1_el_max = min(zone1_center + sweep_margin_deg, -0.5)

        # Exclusion band: near-horizon region (low value)
        exclusion_el_min = -0.5
        exclusion_el_max = 2.0

        # Zone 3: Pursuit zone (above horizon), only if ascending
        velocity = self.classify_velocity(velocity_vy_degps)
        if velocity.launch_vector == "ascending":
            # Drone is lifting off — extend sweep above horizon
            # Estimated max drone altitude at close range: ~3m above ground
            # At range_est_m, that's: arctan(3 / range_est_m) above horizon
            if range_est_m > 0:
                max_drone_el = math.degrees(
                    math.atan(3.0 / range_est_m)
                )
            else:
                max_drone_el = 10.0

            zone3_el_min = exclusion_el_max
            zone3_el_max = min(max_drone_el + sweep_margin_deg, 15.0)
        else:
            zone3_el_min = None
            zone3_el_max = None

        return SweepBounds(
            zone1_el_min=zone1_el_min,
            zone1_el_max=zone1_el_max,
            zone3_el_min=zone3_el_min,
            zone3_el_max=zone3_el_max,
            exclusion_el_min=exclusion_el_min,
            exclusion_el_max=exclusion_el_max,
            azimuth_deg=azimuth_deg,
        )
