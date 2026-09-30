# ---
# VOLLEBAK_ENGINEERING_METADATA:
#   PROJECT_ID: PREDATOR-01
#   TRACK: ECOSYSTEM
#   PHASE: CHALLENGE_HUB
#   CONTRIBUTOR_ID: Antigravity AI
#   THE_DELTA: Multi-factor threat scoring with kinematic primer — speed and trajectory feed initial threat score BEFORE micro-Doppler results arrive, enabling faster engagement authorization for fast movers
#   FAILURE_MODE: Single-sensor classification produces excessive false positives; micro-Doppler 500ms data requirement blocks fast engagement of 41m/s FPV drones
#   IP_STATUS: VOLLEBAK_PROPRIETARY
#   USER_FEEDBACK_REF: Latency audit Q4 — keep weights in place, adjust with testing data
#   DEPENDENCIES: [numpy]
# ---
"""
Threat Classifier — Multi-Sensor Threat Score Aggregation.

Combines Layer 1 (neuromorphic flicker), Layer 2 (radar micro-Doppler
class + RCS), and kinematics into a single threat score.

Kinematic Primer
~~~~~~~~~~~~~~~~
Speed and trajectory are used as a *primer* to micro-Doppler input.
The kinematic primer provides an initial threat assessment using
only speed, closing rate, and altitude — without waiting for the
500ms micro-Doppler STFT window.

When micro-Doppler results arrive, the full weighted score replaces
the primer. This decouples engagement authorization from the
micro-Doppler latency for fast-moving targets.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.layer2_radar.micro_doppler_classifier import TargetClass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ThreatAssessment:
    """Aggregated threat assessment for a tracked target.

    Attributes:
        target_id: Tracker-assigned target ID.
        threat_score: Final weighted threat score (0.0–1.0).
        flicker_score: Neuromorphic propeller flicker component.
        doppler_score: micro-Doppler classification component.
        kinematics_score: Speed + altitude + trajectory component.
        rcs_score: Radar cross-section component.
        kinematic_primer_score: Speed + trajectory pre-assessment
            (available before micro-Doppler).
        is_threat: Whether score exceeds engagement threshold.
        is_primer_only: True if micro-Doppler data was NOT available
            when this assessment was computed.
    """
    target_id: int
    threat_score: float
    flicker_score: float
    doppler_score: float
    kinematics_score: float
    rcs_score: float
    kinematic_primer_score: float
    is_threat: bool
    is_primer_only: bool


@dataclass(slots=True)
class ThreatClassifierConfig:
    """Weights and thresholds for threat scoring.

    Attributes:
        w_flicker: Weight for neuromorphic flicker confidence.
        w_doppler: Weight for micro-Doppler classification.
        w_kinematics: Weight for kinematics (speed + altitude).
        w_rcs: Weight for radar cross-section.
        engagement_threshold: Minimum total score to authorize engagement.
        min_speed_mps: Speed below which kinematics score is zero.
        max_rcs_dbsm: Upper bound of drone-like RCS.
        min_rcs_dbsm: Lower bound of drone-like RCS.
        primer_closing_rate_boost: Extra score for targets closing
            toward the operator (negative radial velocity).
        primer_speed_scale_mps: Speed at which speed_factor saturates
            to 1.0. Set lower to credit fast movers more aggressively.
    """
    w_flicker: float = 0.30
    w_doppler: float = 0.35
    w_kinematics: float = 0.20
    w_rcs: float = 0.15
    engagement_threshold: float = 0.7
    min_speed_mps: float = 0.5
    max_rcs_dbsm: float = 0.0
    min_rcs_dbsm: float = -30.0
    primer_closing_rate_boost: float = 0.15
    primer_speed_scale_mps: float = 20.0


class ThreatClassifier:
    """Multi-factor threat score aggregator with kinematic primer.

    The primer enables a fast initial threat assessment using only
    speed, closing rate, and altitude — without waiting for the
    micro-Doppler STFT window to complete. When micro-Doppler
    arrives, the full weighted score is computed.

    Args:
        config: Scoring weights and thresholds.
    """

    def __init__(self, config: Optional[ThreatClassifierConfig] = None) -> None:
        self._config = config or ThreatClassifierConfig()
        self.assessments_total: int = 0
        self.primer_assessments: int = 0

    def compute_kinematic_primer(
        self,
        target_id: int,
        speed_mps: float,
        radial_velocity_mps: float,
        altitude_m: float,
        flicker_confidence: float,
        rcs_dbsm: float,
    ) -> ThreatAssessment:
        """Compute fast kinematic-only threat assessment (NO micro-Doppler).

        This is the speed/trajectory primer that runs immediately when
        radar acquires a target, without waiting for the 500ms
        micro-Doppler STFT window.

        The primer uses speed, closing rate (negative radial velocity =
        approaching), altitude, and flicker confidence to generate a
        preliminary threat score. This score can authorize engagement
        for fast-moving threats before micro-Doppler is available.

        Args:
            target_id: Tracker-assigned target ID.
            speed_mps: Target ground speed in m/s.
            radial_velocity_mps: Radial velocity (negative = closing).
            altitude_m: Target altitude above operator.
            flicker_confidence: Layer 1 flicker detection confidence.
            rcs_dbsm: Radar cross-section in dBsm.

        Returns:
            ThreatAssessment with ``is_primer_only=True``.
        """
        cfg = self._config

        # --- Flicker component (same as full assess) ---
        flicker_score = float(np.clip(flicker_confidence, 0.0, 1.0))

        # --- Kinematic primer score ---
        # Speed factor: saturates at primer_speed_scale_mps
        speed_factor = 0.0
        if speed_mps >= cfg.min_speed_mps:
            speed_factor = min(1.0, speed_mps / cfg.primer_speed_scale_mps)

        # Altitude factor: low-altitude flight is more threatening
        alt_factor = min(1.0, max(0.0, altitude_m / 50.0))

        # Closing rate boost: target approaching operator
        closing_boost = 0.0
        if radial_velocity_mps < -1.0:
            # Negative radial velocity = approaching. Scale 0–1.
            closing_boost = min(
                cfg.primer_closing_rate_boost,
                cfg.primer_closing_rate_boost * abs(radial_velocity_mps) / 30.0,
            )

        kinematics_score = float(np.clip(
            0.4 * speed_factor + 0.3 * alt_factor + 0.3 * closing_boost / max(cfg.primer_closing_rate_boost, 1e-6),
            0.0, 1.0,
        ))

        # --- RCS component (same as full assess) ---
        rcs_score = self._compute_rcs_score(rcs_dbsm)

        # --- Primer total: redistribute Doppler weight to kinematics ---
        # Since micro-Doppler is not available, its weight (0.35) is
        # redistributed: 60% to kinematics, 40% to flicker.
        doppler_redistrib_to_kinematics = cfg.w_doppler * 0.6
        doppler_redistrib_to_flicker = cfg.w_doppler * 0.4

        primer_score = float(np.clip(
            (cfg.w_flicker + doppler_redistrib_to_flicker) * flicker_score
            + (cfg.w_kinematics + doppler_redistrib_to_kinematics) * kinematics_score
            + cfg.w_rcs * rcs_score
            + closing_boost,
            0.0, 1.0,
        ))

        self.primer_assessments += 1
        self.assessments_total += 1

        return ThreatAssessment(
            target_id=target_id,
            threat_score=primer_score,
            flicker_score=flicker_score,
            doppler_score=0.0,  # Not available yet
            kinematics_score=kinematics_score,
            rcs_score=rcs_score,
            kinematic_primer_score=primer_score,
            is_threat=primer_score >= cfg.engagement_threshold,
            is_primer_only=True,
        )

    def assess(
        self,
        target_id: int,
        flicker_confidence: float,
        doppler_class: TargetClass,
        doppler_confidence: float,
        speed_mps: float,
        altitude_m: float,
        rcs_dbsm: float,
        radial_velocity_mps: float = 0.0,
    ) -> ThreatAssessment:
        """Compute full threat score including micro-Doppler.

        This is the authoritative assessment that replaces the kinematic
        primer once micro-Doppler STFT data is available.

        Args:
            target_id: Tracker-assigned target ID.
            flicker_confidence: Layer 1 flicker detection confidence.
            doppler_class: micro-Doppler classification result.
            doppler_confidence: micro-Doppler classification confidence.
            speed_mps: Target ground speed in m/s.
            altitude_m: Target altitude above operator.
            rcs_dbsm: Radar cross-section in dBsm.
            radial_velocity_mps: Radial velocity (negative = closing).

        Returns:
            ThreatAssessment with ``is_primer_only=False``.
        """
        cfg = self._config

        flicker_score = float(np.clip(flicker_confidence, 0.0, 1.0))

        # Doppler score from micro-Doppler classification
        doppler_score = 0.0
        if doppler_class == TargetClass.ROTARY_UAS:
            doppler_score = doppler_confidence
        elif doppler_class == TargetClass.FIXED_WING:
            doppler_score = doppler_confidence * 0.5

        # Kinematics score (enhanced with closing rate)
        kinematics_score = 0.0
        if speed_mps >= cfg.min_speed_mps:
            speed_factor = min(1.0, speed_mps / cfg.primer_speed_scale_mps)
            alt_factor = min(1.0, max(0.0, altitude_m / 50.0))

            # Closing rate adds to kinematics
            closing_factor = 0.0
            if radial_velocity_mps < -1.0:
                closing_factor = min(1.0, abs(radial_velocity_mps) / 30.0)

            kinematics_score = (
                0.35 * speed_factor
                + 0.30 * alt_factor
                + 0.35 * closing_factor
            )

        # RCS score
        rcs_score = self._compute_rcs_score(rcs_dbsm)

        # Full weighted score
        threat_score = float(np.clip(
            cfg.w_flicker * flicker_score
            + cfg.w_doppler * doppler_score
            + cfg.w_kinematics * kinematics_score
            + cfg.w_rcs * rcs_score,
            0.0, 1.0,
        ))

        # Also compute what the primer would have been for comparison
        primer_score = float(np.clip(
            (cfg.w_flicker + cfg.w_doppler * 0.4) * flicker_score
            + (cfg.w_kinematics + cfg.w_doppler * 0.6) * kinematics_score
            + cfg.w_rcs * rcs_score,
            0.0, 1.0,
        ))

        self.assessments_total += 1

        return ThreatAssessment(
            target_id=target_id,
            threat_score=threat_score,
            flicker_score=flicker_score,
            doppler_score=doppler_score,
            kinematics_score=kinematics_score,
            rcs_score=rcs_score,
            kinematic_primer_score=primer_score,
            is_threat=threat_score >= cfg.engagement_threshold,
            is_primer_only=False,
        )

    def _compute_rcs_score(
        self,
        rcs_dbsm: float,
        hcr_margin_db: float = 0.0,
    ) -> float:
        """Score radar cross-section for drone-like signature.

        With HCR (High Contrast Resolution) exploitation from PMCW
        code-domain sidelobe suppression. When a small-RCS target is
        detected with high HCR margin near a large reflector, the
        detection is almost certainly real (not a sidelobe artifact).

        RCS Bands:
            Drone-like (-30 to 0 dBsm): 0.8 base score.
            Very small (< -30 dBsm): 0.3 base score.
            Too large (> 0 dBsm): 0.1 base score.

        HCR Boost:
            High HCR margin (>25 dB) on a small-RCS target near
            large reflectors → +0.15 confidence boost. This exploits
            PMCW's ~35dB sidelobe suppression to confirm drone
            detections in urban near-structure scenarios.

        Args:
            rcs_dbsm: Radar cross-section in dBsm.
            hcr_margin_db: Delta between target correlation peak and
                nearest large reflector sidelobe (from PMCW processing).
                Higher = more confident the target is distinct from clutter.

        Returns:
            RCS score (0.0–1.0).
        """
        cfg = self._config

        # Base RCS band scoring
        if cfg.min_rcs_dbsm <= rcs_dbsm <= cfg.max_rcs_dbsm:
            base_score = 0.8
        elif rcs_dbsm < cfg.min_rcs_dbsm:
            base_score = 0.3
        else:
            base_score = 0.1

        # HCR exploitation: high sidelobe suppression margin on
        # drone-sized targets provides confident near-structure
        # detection. PMCW RoC spec: ~35dB sidelobe suppression.
        hcr_boost = 0.0
        if hcr_margin_db > 25.0 and rcs_dbsm <= cfg.max_rcs_dbsm:
            # Scale boost: 25dB → 0.0, 35dB → 0.15
            hcr_boost = min(0.15, (hcr_margin_db - 25.0) * 0.015)
            logger.debug(
                "HCR boost: %.1f dB margin → +%.3f RCS confidence",
                hcr_margin_db, hcr_boost,
            )

        return float(np.clip(base_score + hcr_boost, 0.0, 1.0))
