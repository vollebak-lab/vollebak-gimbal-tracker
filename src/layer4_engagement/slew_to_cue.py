# ---
# VOLLEBAK_ENGINEERING_METADATA:
#   PROJECT_ID: PREDATOR-01
#   TRACK: ECOSYSTEM
#   PHASE: CHALLENGE_HUB
#   CONTRIBUTOR_ID: Antigravity AI
#   THE_DELTA: Cartesian-to-gimbal transform with operator body frame offset and lead-angle prediction for moving targets
#   FAILURE_MODE: Static pointing without lead compensation misses fast-moving drones during gimbal slew latency
#   IP_STATUS: VOLLEBAK_PROPRIETARY
#   USER_FEEDBACK_REF: None
#   DEPENDENCIES: [numpy]
# ---
"""
Slew-to-Cue — Target Position to Gimbal Command Converter.

Converts radar track (x, y, z) in global frame to gimbal azimuth/elevation
commands, accounting for gimbal mounting offset and target lead angle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GimbalCommand:
    """Command for the gimbal controller.

    Attributes:
        azimuth_deg: Commanded azimuth (0=fwd, CW positive).
        elevation_deg: Commanded elevation (0=horizon, up positive).
        slew_rate_dps: Requested slew rate in degrees/second.
        target_range_m: Range to target for focus/power control.
        is_lead_compensated: Whether lead angle was applied.
    """
    azimuth_deg: float
    elevation_deg: float
    slew_rate_dps: float
    target_range_m: float
    is_lead_compensated: bool


@dataclass(slots=True)
class SlewConfig:
    """Slew-to-cue configuration.

    Attributes:
        gimbal_az_offset_deg: Gimbal mounting azimuth offset from body forward.
        gimbal_el_offset_deg: Gimbal mounting elevation offset from body horizon.
        max_slew_rate_dps: Maximum gimbal slew rate.
        lead_time_s: How far ahead to predict target position.
        min_slew_threshold_deg: Minimum angular error to command a slew.
    """
    gimbal_az_offset_deg: float = 0.0
    gimbal_el_offset_deg: float = 0.0
    max_slew_rate_dps: float = 180.0
    lead_time_s: float = 0.1
    min_slew_threshold_deg: float = 0.5


class SlewToCue:
    """Converts target track state to gimbal commands.

    Args:
        config: Slew configuration parameters.
    """

    def __init__(self, config: Optional[SlewConfig] = None) -> None:
        self._config = config or SlewConfig()
        self.commands_total: int = 0

    def compute_command(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        target_vx: float = 0.0,
        target_vy: float = 0.0,
        target_vz: float = 0.0,
    ) -> GimbalCommand:
        """Compute gimbal command from target state.

        Applies lead-angle compensation using target velocity and
        subtracts gimbal mounting offset.

        Args:
            target_x/y/z: Target position in meters (global frame).
            target_vx/vy/vz: Target velocity in m/s.

        Returns:
            GimbalCommand with azimuth, elevation, and slew rate.
        """
        cfg = self._config
        lead_t = cfg.lead_time_s

        # Predicted position with lead compensation
        px = target_x + target_vx * lead_t
        py = target_y + target_vy * lead_t
        pz = target_z + target_vz * lead_t

        has_lead = (lead_t > 0 and (target_vx != 0 or target_vy != 0 or target_vz != 0))

        # Convert to spherical
        range_m = float(np.sqrt(px**2 + py**2 + pz**2))
        range_m = max(range_m, 1e-6)

        azimuth_deg = float(np.degrees(np.arctan2(px, py)))
        elevation_deg = float(np.degrees(np.arcsin(pz / range_m)))

        # Subtract gimbal mounting offsets
        azimuth_deg -= cfg.gimbal_az_offset_deg
        elevation_deg -= cfg.gimbal_el_offset_deg

        # Wrap azimuth to [-180, 180] for gimbal command
        azimuth_deg = ((azimuth_deg + 180.0) % 360.0) - 180.0

        # Compute required slew rate from angular error magnitude
        angular_error = np.sqrt(azimuth_deg**2 + elevation_deg**2)
        slew_rate = min(angular_error * 10.0, cfg.max_slew_rate_dps)

        self.commands_total += 1

        return GimbalCommand(
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            slew_rate_dps=slew_rate,
            target_range_m=range_m,
            is_lead_compensated=has_lead,
        )

    def compute_l1_command(
        self,
        bearing_deg: float,
        elevation_deg: float,
        estimated_range_m: float = 25.0,
    ) -> GimbalCommand:
        """Compute gimbal command from L1-only bearing/elevation.

        Used in Waiter mode (L1_ENGAGEMENT) where no radar track
        data is available. The gimbal slews at max rate to the
        bearing, then the vertical sweep pattern takes over.

        No lead-angle compensation is applied — Waiter drones
        are stationary on the ground at the moment of detection.
        The sweep pattern handles positional uncertainty.

        Args:
            bearing_deg: L1 detection bearing (0=fwd, CW).
            elevation_deg: L1 detection elevation (negative=below).
            estimated_range_m: Range estimate from WaiterRangeEstimator.

        Returns:
            GimbalCommand with max slew rate, no lead compensation.
        """
        cfg = self._config

        # Apply gimbal mounting offsets
        az = bearing_deg - cfg.gimbal_az_offset_deg
        el = elevation_deg - cfg.gimbal_el_offset_deg

        # Wrap azimuth
        az = ((az + 180.0) % 360.0) - 180.0

        # Max slew rate — time is critical in Waiter mode
        slew_rate = cfg.max_slew_rate_dps

        self.commands_total += 1

        logger.info(
            "L1 SLEW: az=%.1f° el=%.1f° range=%.1fm (max rate)",
            az, el, estimated_range_m,
        )

        return GimbalCommand(
            azimuth_deg=az,
            elevation_deg=el,
            slew_rate_dps=slew_rate,
            target_range_m=estimated_range_m,
            is_lead_compensated=False,
        )
