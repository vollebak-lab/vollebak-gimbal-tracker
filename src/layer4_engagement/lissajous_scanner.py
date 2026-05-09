# ---
# VOLLEBAK_ENGINEERING_METADATA:
#   PROJECT_ID: PREDATOR-01
#   TRACK: ECOSYSTEM
#   PHASE: CHALLENGE_HUB
#   CONTRIBUTOR_ID: Antigravity AI
#   THE_DELTA: Parametric Lissajous and Rosette scan patterns sized to target angular extent for efficient energy delivery
#   FAILURE_MODE: Fixed-point laser engagement only hits center of target; scanning pattern ensures coverage of propeller disc and optics
#   IP_STATUS: VOLLEBAK_PROPRIETARY
#   USER_FEEDBACK_REF: None
#   DEPENDENCIES: [numpy]
# ---
"""
Lissajous/Rosette Scan Pattern Generator.

Generates 2D scan patterns for the gimbal to execute during laser
engagement. The pattern ensures the beam sweeps across the target's
propeller disc and optics for maximum soft-kill probability.

Supported patterns:
    - Lissajous: x = A·sin(a·t + δ), y = B·sin(b·t)
    - Rosette: r = A·cos(k·θ)
    - Raster: row-by-row sweep
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

logger = logging.getLogger(__name__)


class ScanPattern(Enum):
    """Available scan pattern types."""
    LISSAJOUS = auto()
    ROSETTE = auto()
    RASTER = auto()


@dataclass(frozen=True, slots=True)
class ScanPoint:
    """A single point in the scan pattern."""
    az_offset_deg: float   # Azimuth offset from target center
    el_offset_deg: float   # Elevation offset from target center
    time_s: float          # Time within scan cycle


@dataclass(slots=True)
class ScanConfig:
    """Scan pattern configuration.

    Attributes:
        pattern: Scan pattern type.
        angular_extent_deg: Half-angle of the scan pattern.
        scan_rate_hz: Pattern repetition frequency.
        num_points: Points per scan cycle.
        lissajous_a: Lissajous frequency ratio (horizontal).
        lissajous_b: Lissajous frequency ratio (vertical).
        lissajous_delta: Lissajous phase offset.
        rosette_k: Rosette petal count parameter.
    """
    pattern: ScanPattern = ScanPattern.LISSAJOUS
    angular_extent_deg: float = 0.5
    scan_rate_hz: float = 10.0
    num_points: int = 200
    lissajous_a: int = 3
    lissajous_b: int = 2
    lissajous_delta: float = np.pi / 2
    rosette_k: int = 5


class LissajousScanner:
    """Generates scan patterns for laser engagement.

    The pattern is sized based on target range and assumed angular
    extent. The gimbal controller applies these offsets on top of
    the base slew-to-cue pointing.

    Args:
        config: Scan pattern parameters.
    """

    def __init__(self, config: ScanConfig | None = None) -> None:
        self._config = config or ScanConfig()

    def generate_pattern(
        self,
        target_range_m: float = 100.0,
        target_size_m: float = 0.3,
    ) -> list[ScanPoint]:
        """Generate a complete scan pattern cycle.

        The angular extent is auto-sized to the target's apparent
        angular size if smaller than the configured extent.

        Args:
            target_range_m: Slant range to target.
            target_size_m: Physical target size (propeller disc diameter).

        Returns:
            List of ScanPoint for one complete cycle.
        """
        # Auto-size angular extent to target
        if target_range_m > 0:
            target_angular_deg = float(
                np.degrees(np.arctan(target_size_m / target_range_m))
            )
            extent = max(target_angular_deg, self._config.angular_extent_deg)
        else:
            extent = self._config.angular_extent_deg

        cfg = self._config
        n = cfg.num_points
        period = 1.0 / cfg.scan_rate_hz
        t = np.linspace(0, period, n, endpoint=False)

        if cfg.pattern == ScanPattern.LISSAJOUS:
            return self._lissajous(t, extent, cfg)
        elif cfg.pattern == ScanPattern.ROSETTE:
            return self._rosette(t, extent, cfg)
        elif cfg.pattern == ScanPattern.RASTER:
            return self._raster(t, extent, n)
        else:
            return self._lissajous(t, extent, cfg)

    @staticmethod
    def _lissajous(
        t: np.ndarray, extent: float, cfg: ScanConfig,
    ) -> list[ScanPoint]:
        """Generate Lissajous pattern."""
        omega = 2 * np.pi * cfg.scan_rate_hz
        az = extent * np.sin(cfg.lissajous_a * omega * t + cfg.lissajous_delta)
        el = extent * np.sin(cfg.lissajous_b * omega * t)
        return [
            ScanPoint(float(az[i]), float(el[i]), float(t[i]))
            for i in range(len(t))
        ]

    @staticmethod
    def _rosette(
        t: np.ndarray, extent: float, cfg: ScanConfig,
    ) -> list[ScanPoint]:
        """Generate Rosette pattern."""
        theta = 2 * np.pi * cfg.scan_rate_hz * t
        r = extent * np.cos(cfg.rosette_k * theta)
        az = r * np.cos(theta)
        el = r * np.sin(theta)
        return [
            ScanPoint(float(az[i]), float(el[i]), float(t[i]))
            for i in range(len(t))
        ]

    @staticmethod
    def _raster(
        t: np.ndarray, extent: float, n: int,
    ) -> list[ScanPoint]:
        """Generate raster (row-by-row) pattern."""
        rows = int(np.sqrt(n))
        cols = n // rows
        points = []
        el_vals = np.linspace(-extent, extent, rows)
        az_vals = np.linspace(-extent, extent, cols)
        idx = 0
        for r, el in enumerate(el_vals):
            row_az = az_vals if r % 2 == 0 else az_vals[::-1]
            for az in row_az:
                if idx >= n:
                    break
                points.append(ScanPoint(float(az), float(el), float(t[idx])))
                idx += 1
        return points
