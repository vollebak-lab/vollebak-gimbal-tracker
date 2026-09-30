# ---
# VOLLEBAK_ENGINEERING_METADATA:
#   PROJECT_ID: PREDATOR-01
#   TRACK: ECOSYSTEM
#   PHASE: CHALLENGE_HUB
#   CONTRIBUTOR_ID: Antigravity AI
#   THE_DELTA: Abstract radar API with simulated backend enables full pipeline testing before Uhnder SDK availability
#   FAILURE_MODE: Hard coupling to SDK blocks all downstream development until hardware arrives
#   IP_STATUS: VOLLEBAK_PROPRIETARY
#   USER_FEEDBACK_REF: None
#   DEPENDENCIES: [numpy]
# ---
"""
Radar Interface — Abstract API and Simulated Backend.

Defines the contract for radar data ingestion from the Uhnder S80
cascaded dual-pair radar system. Provides a simulated backend that
generates synthetic 4D point clouds and Doppler returns for pipeline
testing without hardware.

The real Uhnder S80 backend will implement the same ABC when the
SDK is available (VLB-52 blocked on VLB-46).
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Radar Power & Configuration Enums
# ---------------------------------------------------------------------------

class RadarPowerMode(Enum):
    """Radar power states for cognitive duty cycling.

    Supports 3-tier power model per Radar Fire Control RoC
    cognitive power gating architecture.

    DEEP_SLEEP: RF Tx/Rx and DSPs power-gated. Clock only.
        Typical draw ~50mW. Used in SILENT engagement state.
    SECTOR_SEARCH: High duty cycle, beam steered to neuromorphic
        cue bearing. Aggressive wake for acquisition after L1 trigger.
        Typical draw ~7-9W.
    FULL_TRACK: All 12Tx/16Rx, max PRF, continuous waveform.
        Track-quality updates at full rate. Typical draw ~9.5W.
    """
    DEEP_SLEEP = auto()
    SECTOR_SEARCH = auto()
    FULL_TRACK = auto()


@dataclass(frozen=True, slots=True)
class CFARConfig:
    """Constant False Alarm Rate detector configuration.

    Configures the adaptive detection threshold for environment-
    specific sensitivity tuning. When neuromorphic detection
    corroborates a radar return, CFAR may be bypassed entirely
    (see ``cfar_bypassed`` field on RadarDetection).

    Attributes:
        pfa: Probability of false alarm. Lower = fewer false
            alarms but reduced detection probability.
            - Open field: 1e-6 (conservative)
            - Urban/congested: 1e-4 (permissive)
        guard_cells: Number of guard cells around CUT (cell under test).
        training_cells: Number of training cells for noise estimation.
    """
    pfa: float = 1e-6
    guard_cells: int = 4
    training_cells: int = 16


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

class RadarArrayId(Enum):
    """Identifier for the front or rear radar cascade pair."""
    FRONT = auto()
    REAR = auto()


@dataclass(frozen=True, slots=True)
class RadarDetection:
    """A single radar detection from the 4D point cloud.

    Attributes:
        range_m: Range to target in meters.
        azimuth_deg: Azimuth angle in degrees (0=boresight, CW positive).
        elevation_deg: Elevation angle in degrees (0=horizon, up positive).
        doppler_mps: Radial Doppler velocity in m/s (positive=approaching).
        snr_db: Signal-to-noise ratio in dB.
        rcs_dbsm: Radar cross-section in dBsm.
        x: Cartesian X in meters (derived from spherical).
        y: Cartesian Y in meters (forward from radar).
        z: Cartesian Z in meters (up from radar).
        array_id: Which radar array produced this detection.
    """
    range_m: float
    azimuth_deg: float
    elevation_deg: float
    doppler_mps: float
    snr_db: float
    rcs_dbsm: float
    x: float
    y: float
    z: float
    array_id: RadarArrayId
    hcr_margin_db: float = 0.0
    cfar_bypassed: bool = False


@dataclass(slots=True)
class RadarFrame:
    """A complete radar frame from one or both cascade pairs.

    Attributes:
        timestamp_us: Frame timestamp in microseconds.
        detections: List of RadarDetection objects.
        array_id: Which array produced this frame (or None for fused).
        frame_number: Sequential frame counter.
    """
    timestamp_us: int
    detections: list[RadarDetection]
    array_id: Optional[RadarArrayId]
    frame_number: int


@dataclass(frozen=True, slots=True)
class DopplerTimeSeries:
    """Time series of Doppler returns for micro-Doppler analysis.

    Attributes:
        timestamps_us: (T,) array of timestamps.
        doppler_mps: (T,) array of Doppler velocity values.
        range_m: Range bin center.
        azimuth_deg: Azimuth of the target.
    """
    timestamps_us: np.ndarray
    doppler_mps: np.ndarray
    range_m: float
    azimuth_deg: float


# ---------------------------------------------------------------------------
# Abstract Radar Backend
# ---------------------------------------------------------------------------

class RadarBackend(ABC):
    """Abstract base class for radar data sources.

    All radar backends (Uhnder SDK, simulated, replay) must implement
    this interface. This ensures the downstream pipeline (micro-Doppler
    classifier, coordinate transformer, fusion) is decoupled from the
    specific hardware SDK.
    """

    @abstractmethod
    def start(self) -> None:
        """Initialize and start the radar data stream."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the radar data stream and release resources."""

    @abstractmethod
    def get_frame(self, timeout_s: float = 0.1) -> Optional[RadarFrame]:
        """Get the next radar frame.

        Args:
            timeout_s: Maximum wait time in seconds.

        Returns:
            RadarFrame, or None if no frame available within timeout.
        """

    @abstractmethod
    def get_doppler_time_series(
        self,
        range_m: float,
        azimuth_deg: float,
        duration_s: float = 0.5,
    ) -> Optional[DopplerTimeSeries]:
        """Extract a Doppler time series for a specific range-azimuth cell.

        Used for micro-Doppler analysis (propeller blade-rate extraction).

        Args:
            range_m: Center of the range bin.
            azimuth_deg: Center azimuth of the target.
            duration_s: Length of the time series to accumulate.

        Returns:
            DopplerTimeSeries, or None if insufficient data.
        """

    @abstractmethod
    def set_power_mode(
        self,
        mode: RadarPowerMode,
        sector_bearing_deg: Optional[float] = None,
        sector_width_deg: Optional[float] = None,
    ) -> None:
        """Set the radar power/duty cycle mode.

        Implements cognitive duty cycling per RoC research.
        Neuromorphic camera provides the cue bearing — radar wakes
        aggressively and beam-steers to that sector.

        Args:
            mode: Target power mode.
            sector_bearing_deg: Bearing for SECTOR_SEARCH beam steering.
                Required when mode is SECTOR_SEARCH.
            sector_width_deg: Search sector half-width in degrees.
                Defaults to ±15° if not specified.
        """

    @abstractmethod
    def set_cfar_config(self, config: CFARConfig) -> None:
        """Update CFAR detection threshold parameters.

        Allows mission-configurable sensitivity:
        - Low Pfa (1e-6): Open field — minimize false alarms
        - High Pfa (1e-4): Urban — accept more clutter for higher Pd

        Args:
            config: CFAR detector parameters.
        """

    @abstractmethod
    def set_tx_power_allocation(
        self,
        power_fraction: float,
        beam_indices: Optional[list[int]] = None,
    ) -> None:
        """Set transmit power allocation for cognitive power management.

        Reduces Tx power on close/high-SNR targets to conserve battery.
        Requires S80 SDK — this is a stub interface until SDK is available.

        Args:
            power_fraction: Fraction of max Tx power (0.0–1.0).
            beam_indices: Optional subset of beam channels to adjust.
                None = apply globally to all Tx channels.
        """


# ---------------------------------------------------------------------------
# Simulated Radar Backend
# ---------------------------------------------------------------------------

class SimulatedRadarBackend(RadarBackend):
    """Simulated radar backend for pipeline testing.

    Generates synthetic drone targets with configurable parameters.
    Produces realistic point clouds with:
    - Range + azimuth + elevation spread
    - Blade-rate Doppler modulation (micro-Doppler signature)
    - Clutter and noise detections
    - HCR margin simulation for near-structure scenarios
    - Power mode state tracking for duty cycle testing

    Args:
        update_rate_hz: Frame rate for simulated updates.
        num_clutter: Number of clutter detections per frame.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        update_rate_hz: float = 10.0,
        num_clutter: int = 5,
        seed: int = 42,
    ) -> None:
        self._update_rate_hz = update_rate_hz
        self._num_clutter = num_clutter
        self._rng = np.random.default_rng(seed)
        self._running = False
        self._frame_counter = 0
        self._start_time_us = 0

        # Simulated targets (configurable)
        self._targets: list[dict] = []

        # Power mode state (cognitive duty cycling)
        self._power_mode: RadarPowerMode = RadarPowerMode.DEEP_SLEEP
        self._sector_bearing_deg: Optional[float] = None
        self._sector_width_deg: float = 15.0
        self._simulated_power_w: float = 0.05  # Deep sleep default

        # CFAR config state
        self._cfar_config: CFARConfig = CFARConfig()

        # Tx power allocation state
        self._tx_power_fraction: float = 1.0

    def add_simulated_target(
        self,
        range_m: float = 150.0,
        azimuth_deg: float = 10.0,
        elevation_deg: float = 15.0,
        velocity_mps: float = -5.0,
        rcs_dbsm: float = -10.0,
        blade_rate_hz: float = 120.0,
        num_blades: int = 2,
    ) -> None:
        """Add a simulated drone target.

        Args:
            range_m: Target range.
            azimuth_deg: Target azimuth.
            elevation_deg: Target elevation.
            velocity_mps: Radial approach velocity.
            rcs_dbsm: Radar cross section.
            blade_rate_hz: Propeller blade-pass frequency.
            num_blades: Number of propeller blades.
        """
        self._targets.append({
            "range_m": range_m,
            "azimuth_deg": azimuth_deg,
            "elevation_deg": elevation_deg,
            "velocity_mps": velocity_mps,
            "rcs_dbsm": rcs_dbsm,
            "blade_rate_hz": blade_rate_hz,
            "num_blades": num_blades,
        })

    def start(self) -> None:
        """Start the simulated radar."""
        self._running = True
        self._start_time_us = int(time.monotonic() * 1e6)
        self._frame_counter = 0
        logger.info("SimulatedRadarBackend started")

    def stop(self) -> None:
        """Stop the simulated radar."""
        self._running = False
        self._power_mode = RadarPowerMode.DEEP_SLEEP
        self._simulated_power_w = 0.05
        logger.info("SimulatedRadarBackend stopped")

    def get_frame(self, timeout_s: float = 0.1) -> Optional[RadarFrame]:
        """Generate a synthetic radar frame.

        Returns:
            Simulated RadarFrame with target + clutter detections.
        """
        if not self._running:
            return None

        # In DEEP_SLEEP mode, radar produces no frames
        if self._power_mode == RadarPowerMode.DEEP_SLEEP:
            return None

        time.sleep(1.0 / self._update_rate_hz)

        now_us = int(time.monotonic() * 1e6) - self._start_time_us
        detections: list[RadarDetection] = []

        # Generate target detections
        for target in self._targets:
            # In SECTOR_SEARCH mode, only return targets within sector
            if self._power_mode == RadarPowerMode.SECTOR_SEARCH:
                if self._sector_bearing_deg is not None:
                    bearing_delta = abs(
                        target["azimuth_deg"] - self._sector_bearing_deg
                    )
                    # Handle wraparound
                    bearing_delta = min(bearing_delta, 360.0 - bearing_delta)
                    if bearing_delta > self._sector_width_deg:
                        continue

            # Add micro-Doppler modulation
            t_sec = now_us / 1e6
            blade_doppler = (
                0.5
                * np.sin(2 * np.pi * target["blade_rate_hz"] * t_sec)
            )
            total_doppler = target["velocity_mps"] + blade_doppler

            # Simulate HCR margin — PMCW provides ~35dB sidelobe
            # suppression. Targets near large reflectors still get
            # good HCR margin due to code-domain isolation.
            hcr_margin = 35.0 + self._rng.normal(0, 3.0)

            # Scale SNR by Tx power fraction (cognitive power mgmt)
            effective_snr = 20.0 + self._rng.normal(0, 2)
            if self._tx_power_fraction < 1.0:
                effective_snr += 10.0 * np.log10(
                    max(self._tx_power_fraction, 0.01)
                )

            det = self._make_detection(
                range_m=target["range_m"] + self._rng.normal(0, 0.3),
                azimuth_deg=target["azimuth_deg"] + self._rng.normal(0, 0.5),
                elevation_deg=target["elevation_deg"] + self._rng.normal(0, 0.5),
                doppler_mps=total_doppler + self._rng.normal(0, 0.1),
                snr_db=effective_snr,
                rcs_dbsm=target["rcs_dbsm"] + self._rng.normal(0, 1),
                array_id=RadarArrayId.FRONT,
                hcr_margin_db=hcr_margin,
            )
            detections.append(det)

        # Generate clutter detections (only in search/track modes)
        clutter_count = self._num_clutter
        if self._power_mode == RadarPowerMode.SECTOR_SEARCH:
            # Fewer clutter returns in sector mode (narrower beam)
            clutter_count = max(1, self._num_clutter // 3)

        for _ in range(clutter_count):
            clutter_az = self._rng.uniform(-90, 90)
            # In sector mode, clutter only within sector
            if self._power_mode == RadarPowerMode.SECTOR_SEARCH:
                if self._sector_bearing_deg is not None:
                    clutter_az = (
                        self._sector_bearing_deg
                        + self._rng.uniform(
                            -self._sector_width_deg,
                            self._sector_width_deg,
                        )
                    )

            det = self._make_detection(
                range_m=self._rng.uniform(5, 400),
                azimuth_deg=clutter_az,
                elevation_deg=self._rng.uniform(-5, 5),
                doppler_mps=self._rng.normal(0, 0.3),
                snr_db=self._rng.uniform(5, 12),
                rcs_dbsm=self._rng.uniform(-20, 0),
                array_id=RadarArrayId.FRONT,
                hcr_margin_db=self._rng.uniform(5, 15),  # Low HCR for clutter
            )
            detections.append(det)

        self._frame_counter += 1

        return RadarFrame(
            timestamp_us=now_us,
            detections=detections,
            array_id=None,  # Fused
            frame_number=self._frame_counter,
        )

    def get_doppler_time_series(
        self,
        range_m: float,
        azimuth_deg: float,
        duration_s: float = 0.5,
    ) -> Optional[DopplerTimeSeries]:
        """Generate a synthetic Doppler time series for a target.

        Produces a signal with blade-pass frequency harmonics.

        Returns:
            Synthetic DopplerTimeSeries.
        """
        # Find closest target
        if not self._targets:
            return None

        target = min(
            self._targets,
            key=lambda t: abs(t["range_m"] - range_m) + abs(t["azimuth_deg"] - azimuth_deg),
        )

        # Generate time-domain Doppler signal
        fs = 1000.0  # 1kHz sample rate
        num_samples = int(duration_s * fs)
        t = np.arange(num_samples) / fs

        # Base velocity + blade-pass modulation + harmonics
        signal = target["velocity_mps"] * np.ones(num_samples)
        for harmonic in range(1, 4):
            amplitude = 0.5 / harmonic
            signal += amplitude * np.sin(
                2 * np.pi * target["blade_rate_hz"] * harmonic * t
            )

        # Add noise
        signal += self._rng.normal(0, 0.05, num_samples)

        timestamps = np.arange(num_samples) * int(1e6 / fs)

        return DopplerTimeSeries(
            timestamps_us=timestamps.astype(np.int64),
            doppler_mps=signal.astype(np.float32),
            range_m=range_m,
            azimuth_deg=azimuth_deg,
        )

    def set_power_mode(
        self,
        mode: RadarPowerMode,
        sector_bearing_deg: Optional[float] = None,
        sector_width_deg: Optional[float] = None,
    ) -> None:
        """Set simulated radar power mode.

        Updates internal power state and simulated power draw.
        In SECTOR_SEARCH mode, limits detection returns to the
        specified sector around the neuromorphic cue bearing.
        """
        old_mode = self._power_mode
        self._power_mode = mode

        if sector_bearing_deg is not None:
            self._sector_bearing_deg = sector_bearing_deg
        if sector_width_deg is not None:
            self._sector_width_deg = sector_width_deg

        # Simulated power draw
        power_map = {
            RadarPowerMode.DEEP_SLEEP: 0.05,
            RadarPowerMode.SECTOR_SEARCH: 8.0,
            RadarPowerMode.FULL_TRACK: 9.5,
        }
        self._simulated_power_w = power_map.get(mode, 9.5)

        logger.info(
            "SimRadar power: %s → %s (%.1fW, sector=%.1f°±%.1f°)",
            old_mode.name, mode.name, self._simulated_power_w,
            self._sector_bearing_deg or 0.0, self._sector_width_deg,
        )

    def set_cfar_config(self, config: CFARConfig) -> None:
        """Update simulated CFAR parameters."""
        self._cfar_config = config
        logger.info(
            "SimRadar CFAR: Pfa=%.1e, guard=%d, training=%d",
            config.pfa, config.guard_cells, config.training_cells,
        )

    def set_tx_power_allocation(
        self,
        power_fraction: float,
        beam_indices: Optional[list[int]] = None,
    ) -> None:
        """Set simulated Tx power allocation.

        Adjusts simulated SNR proportionally to power fraction.
        """
        self._tx_power_fraction = max(0.01, min(1.0, power_fraction))
        logger.info(
            "SimRadar Tx power: %.0f%% (beams: %s)",
            self._tx_power_fraction * 100,
            beam_indices or "all",
        )

    @property
    def power_mode(self) -> RadarPowerMode:
        """Current radar power mode."""
        return self._power_mode

    @property
    def simulated_power_w(self) -> float:
        """Current simulated power draw in watts."""
        return self._simulated_power_w

    @staticmethod
    def _make_detection(
        range_m: float,
        azimuth_deg: float,
        elevation_deg: float,
        doppler_mps: float,
        snr_db: float,
        rcs_dbsm: float,
        array_id: RadarArrayId,
        hcr_margin_db: float = 0.0,
        cfar_bypassed: bool = False,
    ) -> RadarDetection:
        """Build a RadarDetection from spherical coordinates."""
        az_rad = np.radians(azimuth_deg)
        el_rad = np.radians(elevation_deg)

        x = range_m * np.cos(el_rad) * np.sin(az_rad)
        y = range_m * np.cos(el_rad) * np.cos(az_rad)
        z = range_m * np.sin(el_rad)

        return RadarDetection(
            range_m=range_m,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            doppler_mps=doppler_mps,
            snr_db=snr_db,
            rcs_dbsm=rcs_dbsm,
            x=float(x),
            y=float(y),
            z=float(z),
            array_id=array_id,
            hcr_margin_db=hcr_margin_db,
            cfar_bypassed=cfar_bypassed,
        )
