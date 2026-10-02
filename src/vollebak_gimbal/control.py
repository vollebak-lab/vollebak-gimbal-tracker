from __future__ import annotations

from dataclasses import dataclass

from .config import GimbalConfig, TrackingConfig
from .models import Angles


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass(slots=True)
class SafeAngleController:
    gimbal: GimbalConfig
    tracking: TrackingConfig
    current: Angles | None = None

    def reset(self, angles: Angles | None = None) -> None:
        self.current = angles

    def update(self, desired: Angles) -> Angles:
        desired = Angles(
            clamp(desired.pan, self.gimbal.pan_min, self.gimbal.pan_max),
            clamp(desired.tilt, self.gimbal.tilt_min, self.gimbal.tilt_max),
        )
        if self.current is None:
            self.current = Angles(self.gimbal.home_pan, self.gimbal.home_tilt)

        alpha = self.tracking.smoothing_alpha
        smoothed_pan = self.current.pan + alpha * (desired.pan - self.current.pan)
        smoothed_tilt = self.current.tilt + alpha * (desired.tilt - self.current.tilt)

        pan_delta = smoothed_pan - self.current.pan
        tilt_delta = smoothed_tilt - self.current.tilt
        if abs(pan_delta) < self.tracking.deadband_deg:
            pan_delta = 0.0
        if abs(tilt_delta) < self.tracking.deadband_deg:
            tilt_delta = 0.0

        step = self.tracking.max_step_deg
        next_angles = Angles(
            clamp(self.current.pan + clamp(pan_delta, -step, step), self.gimbal.pan_min, self.gimbal.pan_max),
            clamp(self.current.tilt + clamp(tilt_delta, -step, step), self.gimbal.tilt_min, self.gimbal.tilt_max),
        )
        self.current = next_angles
        return next_angles
