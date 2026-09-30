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
Lissajous/Rosette/VerticalSweep Scan Pattern Generator.

Generates 2D scan patterns for the gimbal to execute during laser
engagement. The pattern ensures the beam sweeps across the target's
propeller disc and optics for maximum soft-kill probability.

Supported patterns:
    - Lissajous: x = A·sin(a·t + δ), y = B·sin(b·t)
    - Rosette: r = A·cos(k·θ)
    - Raster: row-by-row sweep
    - VerticalSweep: fixed-azimuth vertical column sweep for
      Waiter mode (L1-only pulsed laser engagement). Scan points
      are synced to pulsed laser rate (5 Hz Q-switched).
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
    VERTICAL_SWEEP = auto()  # L1-only Waiter mode pulsed laser sweep


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

    # VerticalSweep parameters (Waiter mode)
    sweep_rate_dps: float = 180.0    # Gimbal sweep rate (°/s)
    sweep_passes: int = 3            # Multi-pass for coverage probability
    sweep_pulse_rate_hz: float = 5.0 # Q-switched pulse rate
    sweep_az_jitter_deg: float = 0.5 # Azimuth dither between passes


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
        elif cfg.pattern == ScanPattern.VERTICAL_SWEEP:
            # For VERTICAL_SWEEP, use generate_vertical_sweep() instead
            logger.warning(
                "VERTICAL_SWEEP via generate_pattern() — use "
                "generate_vertical_sweep() for proper zone support"
            )
            return self._vertical_sweep(
                el_min=-extent, el_max=extent,
                az_center=0.0, cfg=cfg,
            )
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

    def generate_vertical_sweep(
        self,
        sweep_bounds: object,
    ) -> list[ScanPoint]:
        """Generate multi-zone vertical sweep for Waiter mode.

        Produces scan points for pulsed laser engagement of
        ground-level ambush drones. The sweep covers Zone 1 (primary,
        below horizon) and optionally Zone 3 (pursuit, above horizon),
        skipping the exclusion band near the horizon.

        Scan points are spaced by the angular distance the gimbal
        covers between laser pulses:
            Δθ = sweep_rate_dps / pulse_rate_hz

        At 180°/s and 5 Hz: Δθ = 36° per pulse. Each pulse is a
        nanosecond Q-switched hammer delivering megawatt peak power
        through the drone lens (100,000× optical gain).

        Args:
            sweep_bounds: SweepBounds instance with zone definitions.
                Expected attributes: zone1_el_min, zone1_el_max,
                zone3_el_min, zone3_el_max, azimuth_deg.

        Returns:
            List of ScanPoint for the complete multi-pass sweep.
        """
        cfg = self._config
        points = []

        # Zone 1: Primary (below horizon)
        z1_points = self._vertical_sweep(
            el_min=sweep_bounds.zone1_el_min,
            el_max=sweep_bounds.zone1_el_max,
            az_center=sweep_bounds.azimuth_deg,
            cfg=cfg,
        )
        points.extend(z1_points)

        # Zone 3: Pursuit (above horizon), if present
        if (
            sweep_bounds.zone3_el_min is not None
            and sweep_bounds.zone3_el_max is not None
        ):
            z3_points = self._vertical_sweep(
                el_min=sweep_bounds.zone3_el_min,
                el_max=sweep_bounds.zone3_el_max,
                az_center=sweep_bounds.azimuth_deg,
                cfg=cfg,
            )
            points.extend(z3_points)

        logger.info(
            "VERTICAL SWEEP: %d points, %d passes, "
            "zone1=[%.1f°, %.1f°] az=%.1f°",
            len(points), cfg.sweep_passes,
            sweep_bounds.zone1_el_min, sweep_bounds.zone1_el_max,
            sweep_bounds.azimuth_deg,
        )

        return points

    @staticmethod
    def _vertical_sweep(
        el_min: float,
        el_max: float,
        az_center: float,
        cfg: ScanConfig,
    ) -> list[ScanPoint]:
        """Generate vertical sweep scan points for a single zone.

        Points are spaced by the angular distance the gimbal covers
        between consecutive laser pulses. Each pulse is independently
        lethal (nanosecond Q-switched lattice shattering). Multi-pass
        provides probabilistic lens aperture coverage, not thermal
        dwell accumulation.

        Args:
            el_min: Lower elevation bound (degrees).
            el_max: Upper elevation bound (degrees).
            az_center: Fixed azimuth for the sweep column.
            cfg: Scan configuration with sweep parameters.

        Returns:
            List of ScanPoint for multi-pass sweep of this zone.
        """
        # Angular spacing between pulses
        # Δθ = sweep_rate / pulse_rate
        if cfg.sweep_pulse_rate_hz > 0:
            delta_deg = cfg.sweep_rate_dps / cfg.sweep_pulse_rate_hz
        else:
            delta_deg = 1.0  # Fallback

        # Generate sweep points for one pass (bottom → top)
        sweep_range = el_max - el_min
        if sweep_range <= 0:
            return []

        n_points = max(1, int(sweep_range / delta_deg) + 1)
        el_values = np.linspace(el_min, el_max, n_points)

        # Time per pass
        time_per_pass = sweep_range / cfg.sweep_rate_dps

        points: list[ScanPoint] = []
        t = 0.0

        for pass_idx in range(cfg.sweep_passes):
            # Alternate sweep direction: up → down → up
            if pass_idx % 2 == 0:
                sweep_els = el_values
            else:
                sweep_els = el_values[::-1]

            # Apply small azimuth jitter between passes for coverage
            az_jitter = (
                (pass_idx - cfg.sweep_passes // 2)
                * cfg.sweep_az_jitter_deg
            )

            dt = time_per_pass / n_points if n_points > 0 else 0.0

            for el in sweep_els:
                points.append(ScanPoint(
                    az_offset_deg=az_center + az_jitter,
                    el_offset_deg=float(el),
                    time_s=t,
                ))
                t += dt

        return points
