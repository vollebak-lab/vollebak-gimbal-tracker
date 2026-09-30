# ---
# VOLLEBAK_ENGINEERING_METADATA:
#   PROJECT_ID: PREDATOR-01
#   TRACK: ECOSYSTEM
#   PHASE: CHALLENGE_HUB
#   CONTRIBUTOR_ID: Antigravity AI
#   THE_DELTA: Operator keep-out cone and range gate enforcement prevents laser engagement within 30-deg of operator head or below minimum safe range
#   FAILURE_MODE: Without safety gates the gimbal could point the laser at the operator during rapid slew maneuvers
#   IP_STATUS: VOLLEBAK_PROPRIETARY
#   USER_FEEDBACK_REF: None
#   DEPENDENCIES: [numpy]
# ---
"""
Safety Manager — Engagement Authorization and Keep-Out Enforcement.

Enforces safety constraints before laser engagement is authorized:
    1. Keep-out cone around operator head (configurable, default 30°)
    2. Range gating (min 10m, max 300m)
    3. Elevation floor (no ground engagement)
    4. Engagement rate limiting
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SafetyConfig:
    """Safety constraint parameters."""
    keep_out_cone_deg: float = 30.0
    min_engagement_range_m: float = 10.0
    max_engagement_range_m: float = 300.0
    min_elevation_deg: float = 5.0
    max_engagements_per_minute: int = 10
    cooldown_between_engagements_s: float = 2.0

    # L1-only sweep engagement constraints (Waiter mode)
    l1_sweep_min_elevation_deg: float = -20.0   # Floor for vertical sweep
    l1_sweep_max_elevation_deg: float = 15.0    # Ceiling for vertical sweep
    l1_sweep_max_range_m: float = 50.0          # L1-only range limit


@dataclass(frozen=True, slots=True)
class SafetyCheck:
    """Result of a safety authorization check."""
    authorized: bool
    reason: str
    target_bearing_deg: float
    target_elevation_deg: float
    target_range_m: float
    keep_out_clear: bool
    range_clear: bool
    elevation_clear: bool
    rate_clear: bool


class SafetyManager:
    """Engagement safety gate.

    All engagement commands must pass through this manager before
    the laser fire command is issued. Any failed check vetoes
    engagement immediately.

    Args:
        config: Safety constraint parameters.
    """

    def __init__(self, config: Optional[SafetyConfig] = None) -> None:
        self._config = config or SafetyConfig()
        self._engagement_timestamps: list[float] = []
        self._last_engagement_time: float = 0.0
        self.checks_total: int = 0
        self.vetoes_total: int = 0

    def check_engagement(
        self,
        bearing_deg: float,
        elevation_deg: float,
        range_m: float,
        operator_heading_deg: float = 0.0,
    ) -> SafetyCheck:
        """Evaluate whether engagement is safe.

        Args:
            bearing_deg: Target global bearing (0=fwd, CW).
            elevation_deg: Target elevation above horizon.
            range_m: Slant range to target.
            operator_heading_deg: Operator facing direction.

        Returns:
            SafetyCheck with authorization result and details.
        """
        now = time.monotonic()
        cfg = self._config

        # --- Keep-out cone check ---
        # Compute angular separation between target and operator head
        # Operator head is at bearing=operator_heading, elevation=0
        delta_bearing = self._angular_diff(bearing_deg, operator_heading_deg)
        angular_sep = np.sqrt(delta_bearing**2 + elevation_deg**2)
        keep_out_clear = angular_sep > cfg.keep_out_cone_deg

        # --- Range check ---
        range_clear = cfg.min_engagement_range_m <= range_m <= cfg.max_engagement_range_m

        # --- Elevation check ---
        elevation_clear = elevation_deg >= cfg.min_elevation_deg

        # --- Rate limiting ---
        # Prune old timestamps
        cutoff = now - 60.0
        self._engagement_timestamps = [
            t for t in self._engagement_timestamps if t > cutoff
        ]
        rate_clear = len(self._engagement_timestamps) < cfg.max_engagements_per_minute
        cooldown_clear = (now - self._last_engagement_time) >= cfg.cooldown_between_engagements_s
        rate_clear = rate_clear and cooldown_clear

        authorized = keep_out_clear and range_clear and elevation_clear and rate_clear

        self.checks_total += 1
        if not authorized:
            self.vetoes_total += 1
            reasons = []
            if not keep_out_clear:
                reasons.append(f"keep-out cone ({angular_sep:.1f}° < {cfg.keep_out_cone_deg}°)")
            if not range_clear:
                reasons.append(f"range ({range_m:.1f}m outside [{cfg.min_engagement_range_m}, {cfg.max_engagement_range_m}])")
            if not elevation_clear:
                reasons.append(f"elevation ({elevation_deg:.1f}° < {cfg.min_elevation_deg}°)")
            if not rate_clear:
                reasons.append("rate limit / cooldown")
            reason = "VETOED: " + "; ".join(reasons)
            logger.warning("Safety veto: %s", reason)
        else:
            reason = "AUTHORIZED"

        return SafetyCheck(
            authorized=authorized,
            reason=reason,
            target_bearing_deg=bearing_deg,
            target_elevation_deg=elevation_deg,
            target_range_m=range_m,
            keep_out_clear=keep_out_clear,
            range_clear=range_clear,
            elevation_clear=elevation_clear,
            rate_clear=rate_clear,
        )

    def record_engagement(self) -> None:
        """Record that an engagement was executed (for rate limiting)."""
        now = time.monotonic()
        self._engagement_timestamps.append(now)
        self._last_engagement_time = now

    def check_l1_sweep_engagement(
        self,
        bearing_deg: float,
        sweep_el_min: float,
        sweep_el_max: float,
        estimated_range_m: float,
        operator_heading_deg: float = 0.0,
    ) -> SafetyCheck:
        """Evaluate whether an L1-only vertical sweep engagement is safe.

        L1-only (Waiter mode) engagements target ground-level drones
        with a vertical sweep pattern. Standard elevation checks
        (min 5° above horizon) do not apply — instead we enforce:

        1. Keep-out cone: sweep column must be outside operator cone.
        2. Elevation floor: sweep must not go below -20°.
        3. Elevation ceiling: sweep must not exceed +15°.
        4. Range gate: estimated range must be within [0, 50 m].
        5. Rate limiting (same as standard).

        Args:
            bearing_deg: Fixed azimuth for the vertical sweep.
            sweep_el_min: Lower elevation bound (degrees, negative).
            sweep_el_max: Upper elevation bound (degrees).
            estimated_range_m: Ground-plane range estimate.
            operator_heading_deg: Operator facing direction.

        Returns:
            SafetyCheck with authorization result.
        """
        now = time.monotonic()
        cfg = self._config

        # Keep-out cone: use sweep column bearing
        delta_bearing = self._angular_diff(bearing_deg, operator_heading_deg)
        keep_out_clear = delta_bearing > cfg.keep_out_cone_deg

        # Elevation bounds: sweep must stay within safety floor/ceiling
        el_floor_clear = sweep_el_min >= cfg.l1_sweep_min_elevation_deg
        el_ceil_clear = sweep_el_max <= cfg.l1_sweep_max_elevation_deg
        elevation_clear = el_floor_clear and el_ceil_clear

        # Range gate: L1-only range limit
        range_clear = 0.0 <= estimated_range_m <= cfg.l1_sweep_max_range_m

        # Rate limiting
        cutoff = now - 60.0
        self._engagement_timestamps = [
            t for t in self._engagement_timestamps if t > cutoff
        ]
        rate_clear = (
            len(self._engagement_timestamps)
            < cfg.max_engagements_per_minute
        )
        cooldown_clear = (
            (now - self._last_engagement_time)
            >= cfg.cooldown_between_engagements_s
        )
        rate_clear = rate_clear and cooldown_clear

        authorized = (
            keep_out_clear and range_clear
            and elevation_clear and rate_clear
        )

        self.checks_total += 1
        if not authorized:
            self.vetoes_total += 1
            reasons = []
            if not keep_out_clear:
                reasons.append(
                    f"keep-out cone ({delta_bearing:.1f}° < "
                    f"{cfg.keep_out_cone_deg}°)"
                )
            if not el_floor_clear:
                reasons.append(
                    f"sweep floor ({sweep_el_min:.1f}° < "
                    f"{cfg.l1_sweep_min_elevation_deg}°)"
                )
            if not el_ceil_clear:
                reasons.append(
                    f"sweep ceiling ({sweep_el_max:.1f}° > "
                    f"{cfg.l1_sweep_max_elevation_deg}°)"
                )
            if not range_clear:
                reasons.append(
                    f"L1 range ({estimated_range_m:.1f}m > "
                    f"{cfg.l1_sweep_max_range_m}m)"
                )
            if not rate_clear:
                reasons.append("rate limit / cooldown")
            reason = "L1 SWEEP VETOED: " + "; ".join(reasons)
            logger.warning("Safety veto: %s", reason)
        else:
            reason = "L1 SWEEP AUTHORIZED"

        # Use sweep midpoint for elevation in result
        sweep_midpoint_el = (sweep_el_min + sweep_el_max) / 2.0

        return SafetyCheck(
            authorized=authorized,
            reason=reason,
            target_bearing_deg=bearing_deg,
            target_elevation_deg=sweep_midpoint_el,
            target_range_m=estimated_range_m,
            keep_out_clear=keep_out_clear,
            range_clear=range_clear,
            elevation_clear=elevation_clear,
            rate_clear=rate_clear,
        )

    @staticmethod
    def _angular_diff(a: float, b: float) -> float:
        """Compute shortest angular difference in degrees."""
        diff = (a - b + 180.0) % 360.0 - 180.0
        return abs(diff)
