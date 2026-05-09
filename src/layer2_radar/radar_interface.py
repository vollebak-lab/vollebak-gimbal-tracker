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
        logger.info("SimulatedRadarBackend stopped")

    def get_frame(self, timeout_s: float = 0.1) -> Optional[RadarFrame]:
        """Generate a synthetic radar frame.

        Returns:
            Simulated RadarFrame with target + clutter detections.
        """
        if not self._running:
            return None

        time.sleep(1.0 / self._update_rate_hz)

        now_us = int(time.monotonic() * 1e6) - self._start_time_us
        detections: list[RadarDetection] = []

        # Generate target detections
        for target in self._targets:
            # Add micro-Doppler modulation
            t_sec = now_us / 1e6
            blade_doppler = (
                0.5
                * np.sin(2 * np.pi * target["blade_rate_hz"] * t_sec)
            )
            total_doppler = target["velocity_mps"] + blade_doppler

            # Add measurement noise
            det = self._make_detection(
                range_m=target["range_m"] + self._rng.normal(0, 0.3),
                azimuth_deg=target["azimuth_deg"] + self._rng.normal(0, 0.5),
                elevation_deg=target["elevation_deg"] + self._rng.normal(0, 0.5),
                doppler_mps=total_doppler + self._rng.normal(0, 0.1),
                snr_db=20.0 + self._rng.normal(0, 2),
                rcs_dbsm=target["rcs_dbsm"] + self._rng.normal(0, 1),
                array_id=RadarArrayId.FRONT,
            )
            detections.append(det)

        # Generate clutter detections
        for _ in range(self._num_clutter):
            det = self._make_detection(
                range_m=self._rng.uniform(5, 400),
                azimuth_deg=self._rng.uniform(-90, 90),
                elevation_deg=self._rng.uniform(-5, 5),
                doppler_mps=self._rng.normal(0, 0.3),
                snr_db=self._rng.uniform(5, 12),
                rcs_dbsm=self._rng.uniform(-20, 0),
                array_id=RadarArrayId.FRONT,
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

    @staticmethod
    def _make_detection(
        range_m: float,
        azimuth_deg: float,
        elevation_deg: float,
        doppler_mps: float,
        snr_db: float,
        rcs_dbsm: float,
        array_id: RadarArrayId,
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
        )
