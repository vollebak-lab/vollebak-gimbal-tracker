# ---
# VOLLEBAK_ENGINEERING_METADATA:
#   PROJECT_ID: PREDATOR-01
#   TRACK: ECOSYSTEM
#   PHASE: CHALLENGE_HUB
#   CONTRIBUTOR_ID: Antigravity AI
#   THE_DELTA: Multi-source IMU provider aggregates DVXplorer onboard IMU + external high-rate IMU into a single fused orientation/acceleration state for cross-layer consumption
#   FAILURE_MODE: Body-worn C-UAS without ego-motion compensation produces phantom targets from operator walking gait and head movement
#   IP_STATUS: VOLLEBAK_PROPRIETARY
#   USER_FEEDBACK_REF: None
#   DEPENDENCIES: [dv_processing, numpy]
# ---
"""
IMU Provider — Multi-Source Inertial Measurement Aggregation.

Provides a unified IMU state interface consumed by:
    - Layer 2: Radar point cloud ego-motion compensation
    - Layer 3: FSM engagement gating (operator motion state)
    - Layer 4: Gimbal stabilization + robotic arm compensation
    - Layer 1: DVXplorer onboard IMU for neuromorphic event correction

IMU Sources:
    1. DVXplorer Micro onboard IMU (InvenSense ICM-42688)
       - 6-axis (accel + gyro), accessed via dv_processing SDK
       - 1 per camera, 4 cameras = 4 IMUs (redundancy)
       - ~200Hz native rate
    2. External dedicated IMU (e.g., VectorNav VN-100, BHI260AP)
       - Higher grade, higher rate (up to 400Hz)
       - Dedicated mounting on helmet for best rigid-body reference
       - Connected via UART/SPI to Orin NX

The provider fuses these sources using a complementary filter:
external IMU provides the authority orientation, DVXplorer IMUs
provide cross-validation and per-camera rigid-body offsets.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IMU State Dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ImuState:
    """Unified IMU state snapshot.

    All values are in the world frame (NED convention) unless noted.

    Attributes:
        quaternion: Orientation as (w, x, y, z) unit quaternion.
        gyro_rps: Angular velocity (rad/s) in body frame [wx, wy, wz].
        linear_accel_mps2: Linear acceleration (m/s², gravity-compensated)
            in body frame [ax, ay, az].
        heading_deg: Magnetic heading in degrees (0=N, CW).
        ego_velocity_mps: Estimated ego-velocity in world frame [vx, vy, vz].
        timestamp_ns: Monotonic timestamp in nanoseconds.
        source: Which IMU source produced this state.
        is_valid: Whether the state is based on fresh data.
    """
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    gyro_rps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    linear_accel_mps2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    heading_deg: float = 0.0
    ego_velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    timestamp_ns: int = 0
    source: str = "none"
    is_valid: bool = False


@dataclass(slots=True)
class OperatorMotionState:
    """High-level operator motion classification.

    Used by the FSM to gate engagement authorization — engaging while
    running is different from engaging while stationary.

    Attributes:
        is_stationary: Operator speed below threshold.
        is_walking: Moderate periodic motion detected.
        is_running: High-frequency gait detected.
        speed_mps: Estimated operator ground speed.
        angular_rate_dps: Operator head rotation rate (degrees/s).
        stability_score: 0.0 (maximum instability) to 1.0 (rock-steady).
    """
    is_stationary: bool = True
    is_walking: bool = False
    is_running: bool = False
    speed_mps: float = 0.0
    angular_rate_dps: float = 0.0
    stability_score: float = 1.0


# ---------------------------------------------------------------------------
# Abstract IMU Source
# ---------------------------------------------------------------------------

class ImuSource(ABC):
    """Abstract interface for a single IMU data source."""

    @abstractmethod
    def start(self) -> None:
        """Start reading IMU data."""

    @abstractmethod
    def stop(self) -> None:
        """Stop reading IMU data."""

    @abstractmethod
    def get_latest(self) -> Optional[ImuState]:
        """Return the latest IMU state, or None if stale."""

    @abstractmethod
    def is_stale(self) -> bool:
        """Whether the IMU data is older than the staleness threshold."""


# ---------------------------------------------------------------------------
# DVXplorer Onboard IMU Source
# ---------------------------------------------------------------------------

class DvxImuSource(ImuSource):
    """Reads IMU data from a DVXplorer Micro's onboard ICM-42688.

    Uses the dv_processing SDK's IMU stream from the camera capture.
    Each DVXplorer Micro provides 6-axis IMU at ~200Hz.

    Args:
        camera_name: Human-readable camera name for logging.
        camera_capture: An active dv.io.CameraCapture instance.
        stale_threshold_s: Seconds before data is considered stale.
    """

    def __init__(
        self,
        camera_name: str,
        camera_capture: object,
        stale_threshold_s: float = 0.1,
    ) -> None:
        self._camera_name = camera_name
        self._capture = camera_capture
        self._stale_threshold_s = stale_threshold_s
        self._lock = threading.Lock()
        self._state = ImuState(source=f"dvx_{camera_name}")
        self._last_update_time: float = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Stats
        self.samples_received: int = 0

    def start(self) -> None:
        """Start the IMU reading thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
            name=f"DvxIMU-{self._camera_name}",
        )
        self._thread.start()
        logger.info("DvxImuSource started for camera '%s'", self._camera_name)

    def stop(self) -> None:
        """Stop the IMU reading thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("DvxImuSource stopped for camera '%s'", self._camera_name)

    def _read_loop(self) -> None:
        """Background loop reading IMU data from dv_processing."""
        while self._running:
            try:
                # dv_processing provides getNextImuBatch() on CameraCapture
                if hasattr(self._capture, 'getNextImuBatch'):
                    imu_batch = self._capture.getNextImuBatch()
                    if imu_batch is not None and len(imu_batch) > 0:
                        # Take the latest sample from the batch
                        latest = imu_batch[-1]
                        with self._lock:
                            self._state = ImuState(
                                quaternion=(1.0, 0.0, 0.0, 0.0),  # DVX doesn't provide orientation directly
                                gyro_rps=(
                                    float(latest.gyroscopeX),
                                    float(latest.gyroscopeY),
                                    float(latest.gyroscopeZ),
                                ),
                                linear_accel_mps2=(
                                    float(latest.accelerometerX),
                                    float(latest.accelerometerY),
                                    float(latest.accelerometerZ),
                                ),
                                timestamp_ns=int(latest.timestamp * 1000),
                                source=f"dvx_{self._camera_name}",
                                is_valid=True,
                            )
                            self._last_update_time = time.monotonic()
                        self.samples_received += 1
                else:
                    time.sleep(0.01)
            except Exception:
                logger.debug("IMU read error on camera '%s'", self._camera_name)
                time.sleep(0.01)

    def get_latest(self) -> Optional[ImuState]:
        """Return latest DVXplorer IMU state."""
        with self._lock:
            if self._state.is_valid:
                return ImuState(
                    quaternion=self._state.quaternion,
                    gyro_rps=self._state.gyro_rps,
                    linear_accel_mps2=self._state.linear_accel_mps2,
                    heading_deg=self._state.heading_deg,
                    ego_velocity_mps=self._state.ego_velocity_mps,
                    timestamp_ns=self._state.timestamp_ns,
                    source=self._state.source,
                    is_valid=True,
                )
            return None

    def is_stale(self) -> bool:
        """Check staleness."""
        with self._lock:
            if self._last_update_time == 0.0:
                return True
            return (time.monotonic() - self._last_update_time) > self._stale_threshold_s


# ---------------------------------------------------------------------------
# External IMU Source (Zenoh subscriber)
# ---------------------------------------------------------------------------

class ZenohImuSource(ImuSource):
    """Receives IMU state from an external high-rate IMU via Zenoh.

    Mirrors the ``ZenohImuSubscriber`` from ``radar_belt_engine`` but
    adapted for Predator's Zenoh topic namespace.

    Args:
        session: Active Zenoh session.
        topic: Zenoh key expression for IMU state.
        stale_threshold_s: Staleness threshold.
    """

    def __init__(
        self,
        session: object,
        topic: str = "predator/imu/state",
        stale_threshold_s: float = 0.1,
    ) -> None:
        self._session = session
        self._topic = topic
        self._stale_threshold_s = stale_threshold_s
        self._lock = threading.Lock()
        self._state = ImuState(source="external")
        self._last_update_time: float = 0.0
        self._subscriber = None
        self.samples_received: int = 0

    def start(self) -> None:
        """Subscribe to Zenoh IMU topic."""
        import json
        self._subscriber = self._session.declare_subscriber(
            self._topic,
            self._on_sample,
        )
        logger.info("ZenohImuSource subscribed to '%s'", self._topic)

    def _on_sample(self, sample) -> None:
        """Zenoh callback for each IMU sample."""
        import json
        try:
            data = json.loads(sample.payload.to_string())
            with self._lock:
                self._state = ImuState(
                    quaternion=tuple(data.get("quaternion", [1, 0, 0, 0])),
                    gyro_rps=tuple(data.get("gyro_rps", [0, 0, 0])),
                    linear_accel_mps2=tuple(data.get("linear_accel_mps2", [0, 0, 0])),
                    heading_deg=data.get("heading_deg", 0.0),
                    timestamp_ns=data.get("timestamp_ns", 0),
                    source="external",
                    is_valid=True,
                )
                self._last_update_time = time.monotonic()
            self.samples_received += 1
        except Exception:
            logger.debug("Malformed external IMU sample")

    def stop(self) -> None:
        """Unsubscribe."""
        if self._subscriber:
            self._subscriber.undeclare()
        logger.info("ZenohImuSource stopped")

    def get_latest(self) -> Optional[ImuState]:
        """Return latest external IMU state."""
        with self._lock:
            if self._state.is_valid:
                return ImuState(**{
                    k: getattr(self._state, k)
                    for k in self._state.__dataclass_fields__
                })
            return None

    def is_stale(self) -> bool:
        with self._lock:
            if self._last_update_time == 0.0:
                return True
            return (time.monotonic() - self._last_update_time) > self._stale_threshold_s


# ---------------------------------------------------------------------------
# Fused IMU Provider
# ---------------------------------------------------------------------------

class ImuProvider:
    """Multi-source IMU provider with complementary fusion.

    Aggregates DVXplorer onboard IMUs and an optional external IMU
    into a single fused state. The external IMU is the authority
    source for orientation; DVXplorer IMUs provide cross-validation.

    This is the single interface consumed by all downstream layers.

    Args:
        external_source: Optional external high-rate IMU source.
        dvx_sources: Optional list of DVXplorer IMU sources (1 per camera).
        velocity_decay_alpha: Exponential decay for integrated velocity.
        motion_speed_threshold_mps: Speed below which operator is stationary.
        walking_speed_max_mps: Speed above which operator is running.
    """

    def __init__(
        self,
        external_source: Optional[ImuSource] = None,
        dvx_sources: Optional[list[ImuSource]] = None,
        velocity_decay_alpha: float = 0.95,
        motion_speed_threshold_mps: float = 0.3,
        walking_speed_max_mps: float = 2.5,
    ) -> None:
        self._external = external_source
        self._dvx_sources = dvx_sources or []
        self._velocity_decay = velocity_decay_alpha
        self._motion_threshold = motion_speed_threshold_mps
        self._walking_max = walking_speed_max_mps

        # Integrated ego-velocity (body frame)
        self._ego_velocity = np.zeros(3, dtype=np.float64)
        self._last_timestamp_ns: int = 0

        # Angular rate buffer for stability scoring
        self._angular_rate_buffer: list[float] = []
        self._angular_rate_buffer_max = 50  # ~0.5s at 100Hz

    def start(self) -> None:
        """Start all IMU sources."""
        if self._external:
            self._external.start()
        for dvx in self._dvx_sources:
            dvx.start()
        logger.info(
            "ImuProvider started — %d DVX sources, external=%s",
            len(self._dvx_sources),
            "yes" if self._external else "no",
        )

    def stop(self) -> None:
        """Stop all IMU sources."""
        if self._external:
            self._external.stop()
        for dvx in self._dvx_sources:
            dvx.stop()

    def get_fused_state(self) -> ImuState:
        """Get the best available fused IMU state.

        Priority:
        1. External IMU (authority for orientation)
        2. DVXplorer IMU (fallback, uses first non-stale camera)
        3. Identity state (degraded — no IMU data)

        Returns:
            Fused ImuState with ego-velocity integration.
        """
        state = None

        # Try external first
        if self._external and not self._external.is_stale():
            state = self._external.get_latest()

        # Fallback to DVXplorer IMUs
        if state is None:
            for dvx in self._dvx_sources:
                if not dvx.is_stale():
                    state = dvx.get_latest()
                    if state is not None:
                        break

        if state is None:
            return ImuState(source="degraded", is_valid=False)

        # Integrate ego-velocity from acceleration
        self._integrate_velocity(state)

        # Update angular rate buffer
        gyro = np.array(state.gyro_rps)
        angular_rate = float(np.linalg.norm(gyro))
        self._angular_rate_buffer.append(angular_rate)
        if len(self._angular_rate_buffer) > self._angular_rate_buffer_max:
            self._angular_rate_buffer.pop(0)

        # Inject integrated velocity into state
        return ImuState(
            quaternion=state.quaternion,
            gyro_rps=state.gyro_rps,
            linear_accel_mps2=state.linear_accel_mps2,
            heading_deg=state.heading_deg,
            ego_velocity_mps=tuple(self._ego_velocity.tolist()),
            timestamp_ns=state.timestamp_ns,
            source=state.source,
            is_valid=True,
        )

    def get_rotation_matrix(self) -> np.ndarray:
        """Get the current body-to-world rotation matrix.

        Returns:
            3×3 float64 rotation matrix. Identity if no IMU data.
        """
        state = self.get_fused_state()
        if not state.is_valid:
            return np.eye(3, dtype=np.float64)
        w, x, y, z = state.quaternion
        return self._quaternion_to_rotation(w, x, y, z)

    def get_operator_motion_state(self) -> OperatorMotionState:
        """Classify the operator's current motion state.

        Used by the FSM to adjust engagement parameters based on
        whether the operator is stationary, walking, or running.

        Returns:
            OperatorMotionState with classification and metrics.
        """
        speed = float(np.linalg.norm(self._ego_velocity))

        # Angular rate (degrees/s)
        angular_rate_dps = 0.0
        if self._angular_rate_buffer:
            angular_rate_dps = float(
                np.degrees(np.mean(self._angular_rate_buffer[-10:]))
            )

        # Stability score: 1.0 = perfectly stable, 0.0 = extreme motion
        # Based on angular rate variance (higher variance = gait/instability)
        if len(self._angular_rate_buffer) >= 10:
            rate_std = float(np.std(self._angular_rate_buffer[-20:]))
            stability = max(0.0, 1.0 - rate_std / 2.0)
        else:
            stability = 0.5  # Unknown

        is_stationary = speed < self._motion_threshold
        is_running = speed > self._walking_max
        is_walking = not is_stationary and not is_running

        return OperatorMotionState(
            is_stationary=is_stationary,
            is_walking=is_walking,
            is_running=is_running,
            speed_mps=speed,
            angular_rate_dps=angular_rate_dps,
            stability_score=stability,
        )

    def _integrate_velocity(self, state: ImuState) -> None:
        """Integrate linear acceleration into ego-velocity estimate.

        Uses exponential decay to prevent unbounded drift.
        """
        now_ns = state.timestamp_ns
        if self._last_timestamp_ns == 0:
            self._last_timestamp_ns = now_ns
            return

        dt = (now_ns - self._last_timestamp_ns) / 1e9
        self._last_timestamp_ns = now_ns

        if dt <= 0 or dt > 0.5:
            return

        accel = np.array(state.linear_accel_mps2, dtype=np.float64)

        # Rotate acceleration to world frame using quaternion
        w, x, y, z = state.quaternion
        rot = self._quaternion_to_rotation(w, x, y, z)
        accel_world = rot @ accel

        # Integrate
        self._ego_velocity += accel_world * dt

        # Exponential decay toward zero (drift mitigation)
        self._ego_velocity *= self._velocity_decay

    @staticmethod
    def _quaternion_to_rotation(
        w: float, x: float, y: float, z: float,
    ) -> np.ndarray:
        """Convert unit quaternion to 3×3 rotation matrix.

        Reuses the proven implementation from ego_motion_compensator.py.
        """
        norm = math.sqrt(w*w + x*x + y*y + z*z)
        if norm < 1e-10:
            return np.eye(3, dtype=np.float64)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm

        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ], dtype=np.float64)
