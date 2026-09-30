# ---
# VOLLEBAK_ENGINEERING_METADATA:
#   PROJECT_ID: PREDATOR-01
#   TRACK: ECOSYSTEM
#   PHASE: CHALLENGE_HUB
#   CONTRIBUTOR_ID: Antigravity AI
#   THE_DELTA: Unified ego-motion compensator for radar point cloud + gimbal feed-forward, forked from radar_belt_engine ego_motion_compensator.py with aerial target extensions
#   FAILURE_MODE: Body-worn 4D radar without ego-motion compensation produces phantom drone targets correlated to operator gait and head turning
#   IP_STATUS: VOLLEBAK_PROPRIETARY
#   USER_FEEDBACK_REF: None
#   DEPENDENCIES: [numpy]
# ---
"""
Motion Compensator — Ego-Motion Correction for Radar + Gimbal.

Applies IMU-derived orientation and ego-velocity corrections to:
    1. Radar 4D point clouds (rotate body→world, correct Doppler)
    2. Gimbal feed-forward (compensate for operator motion in real-time)
    3. Neuromorphic event timestamps (DVXplorer onboard IMU correction)

Forked from ``radar_belt_engine/src/ego_motion_compensator.py`` with
extensions for the aerial-target use case and gimbal stabilization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.imu.imu_provider import ImuProvider, ImuState

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CompensationConfig:
    """Configuration for motion compensation.

    Attributes:
        enabled: Master enable.
        compensate_radar: Apply to radar point clouds.
        compensate_gimbal: Generate gimbal feed-forward corrections.
        static_cluster_doppler_max: Max Doppler (m/s) for static
            reflector identification (fallback ego-velocity estimation).
    """
    enabled: bool = True
    compensate_radar: bool = True
    compensate_gimbal: bool = True
    static_cluster_doppler_max: float = 0.3


@dataclass(frozen=True, slots=True)
class GimbalFeedForward:
    """Feed-forward correction for gimbal stabilization.

    The gimbal controller subtracts these offsets from its PID
    command to compensate for operator body motion in real-time.

    Attributes:
        delta_az_deg: Azimuth correction (degrees) to subtract.
        delta_el_deg: Elevation correction (degrees) to subtract.
        delta_roll_deg: Roll correction (degrees). Positive = operator tilted right.
            Used by the mast stabilization controller, not the gimbal PID.
        angular_rate_az_dps: Operator heading rotation rate (deg/s).
        angular_rate_el_dps: Operator pitch rotation rate (deg/s).
        angular_rate_roll_dps: Operator roll rotation rate (deg/s).
        is_valid: Whether IMU data is fresh enough for feed-forward.
    """
    delta_az_deg: float = 0.0
    delta_el_deg: float = 0.0
    delta_roll_deg: float = 0.0
    angular_rate_az_dps: float = 0.0
    angular_rate_el_dps: float = 0.0
    angular_rate_roll_dps: float = 0.0
    is_valid: bool = False


class MotionCompensator:
    """Unified ego-motion compensation for all Predator subsystems.

    Sits between the IMU provider and the radar/gimbal/neuromorphic
    consumers. Each pipeline cycle:
        1. Reads latest IMU state from ``ImuProvider``
        2. Computes body→world rotation matrix
        3. Applies rotation to radar point cloud XYZ
        4. Corrects Doppler velocities for ego-motion
        5. Computes gimbal feed-forward for stabilization

    Args:
        imu_provider: The system-wide ``ImuProvider`` instance.
        config: Compensation parameters.
    """

    def __init__(
        self,
        imu_provider: ImuProvider,
        config: Optional[CompensationConfig] = None,
    ) -> None:
        self._imu = imu_provider
        self._config = config or CompensationConfig()

        # Cache latest rotation for gimbal feed-forward
        self._last_rotation = np.eye(3, dtype=np.float64)
        self._last_euler_deg = np.zeros(3, dtype=np.float64)  # [yaw, pitch, roll]

        # Stats
        self.frames_compensated: int = 0
        self.frames_degraded: int = 0

    def compensate_radar_pointcloud(
        self, points: np.ndarray,
    ) -> np.ndarray:
        """Apply ego-motion compensation to a radar point cloud.

        Transforms points from body frame to world frame and corrects
        Doppler velocities by subtracting the ego-velocity radial
        projection.

        Args:
            points: (N, 4+) float32 array. First 4 columns must be
                [x, y, z, doppler]. Additional columns are preserved.

        Returns:
            (N, 4+) float32 array in world frame.
        """
        if not self._config.enabled or not self._config.compensate_radar:
            return points

        if points.shape[0] == 0:
            return points

        imu_state = self._imu.get_fused_state()

        if not imu_state.is_valid:
            self.frames_degraded += 1
            # Fallback: Doppler-based ego-velocity estimation
            return self._compensate_degraded(points)

        rotation = self._imu.get_rotation_matrix()
        self._last_rotation = rotation
        self._last_euler_deg = self._rotation_to_euler_deg(rotation)

        result = points.copy()

        # Step 1: Rotate XYZ to world frame
        xyz_body = points[:, :3].astype(np.float64)
        xyz_world = (rotation @ xyz_body.T).T
        result[:, :3] = xyz_world.astype(np.float32)

        # Step 2: Correct Doppler for ego-velocity
        ego_vel = np.array(imu_state.ego_velocity_mps, dtype=np.float64)

        if np.linalg.norm(ego_vel) > 0.01:
            ranges = np.linalg.norm(xyz_world, axis=1, keepdims=True)
            safe_ranges = np.where(ranges > 1e-6, ranges, 1.0)
            unit_vectors = xyz_world / safe_ranges
            ego_radial = (unit_vectors @ ego_vel).astype(np.float32)
            result[:, 3] -= ego_radial

        self.frames_compensated += 1
        return result

    def get_gimbal_feed_forward(self) -> GimbalFeedForward:
        """Compute gimbal feed-forward correction.

        Returns the current operator orientation change that the
        gimbal must counteract to maintain target pointing.

        The gimbal PID loop subtracts these corrections from its
        commanded position before driving the servos.

        Returns:
            GimbalFeedForward with az/el corrections and rates.
        """
        if not self._config.compensate_gimbal:
            return GimbalFeedForward()

        imu_state = self._imu.get_fused_state()

        if not imu_state.is_valid:
            return GimbalFeedForward(is_valid=False)

        euler = self._last_euler_deg  # [yaw, pitch, roll]

        # Gimbal feed-forward: negate operator motion
        # Yaw → azimuth correction, Pitch → elevation correction
        # Roll → mast stabilization (not gimbal PID)
        delta_az = -euler[0]  # Counteract yaw
        delta_el = -euler[1]  # Counteract pitch
        delta_roll = euler[2]  # Roll: positive = tilted right

        # Angular rates from gyro (body frame)
        gyro = np.array(imu_state.gyro_rps)
        az_rate_dps = float(np.degrees(gyro[2]))   # Z-axis = yaw
        el_rate_dps = float(np.degrees(gyro[1]))    # Y-axis = pitch
        roll_rate_dps = float(np.degrees(gyro[0]))  # X-axis = roll

        return GimbalFeedForward(
            delta_az_deg=float(delta_az),
            delta_el_deg=float(delta_el),
            delta_roll_deg=float(delta_roll),
            angular_rate_az_dps=az_rate_dps,
            angular_rate_el_dps=el_rate_dps,
            angular_rate_roll_dps=roll_rate_dps,
            is_valid=True,
        )

    def _compensate_degraded(self, points: np.ndarray) -> np.ndarray:
        """Fallback compensation using static-cluster Doppler estimation.

        When no IMU is available, estimate ego-velocity from static
        reflectors in the radar point cloud.
        """
        if points.shape[0] < 5:
            return points

        doppler = points[:, 3]
        threshold = self._config.static_cluster_doppler_max * 3

        static_mask = np.abs(doppler) < threshold
        static_points = points[static_mask]

        if static_points.shape[0] < 3:
            return points

        median_doppler = float(np.median(static_points[:, 3]))

        result = points.copy()
        result[:, 3] -= median_doppler  # Subtract median static Doppler

        self.frames_degraded += 1
        return result

    @staticmethod
    def _rotation_to_euler_deg(R: np.ndarray) -> np.ndarray:
        """Extract Euler angles (yaw, pitch, roll) from rotation matrix.

        Args:
            R: 3×3 rotation matrix.

        Returns:
            (3,) array [yaw, pitch, roll] in degrees.
        """
        # Yaw (heading) from rotation matrix
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        # Pitch
        pitch = np.degrees(np.arcsin(-np.clip(R[2, 0], -1.0, 1.0)))
        # Roll
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))

        return np.array([yaw, pitch, roll], dtype=np.float64)
