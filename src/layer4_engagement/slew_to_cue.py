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
