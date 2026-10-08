from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .models import Angles, Detection


@dataclass(slots=True)
class ExtrinsicAlignmentConfig:
    """Extrinsic baseline and trim calibration between fixed camera and gimbal laser.

    Defaults reflect measured geometry:
      - Laser mounted on gimbal to the right (+X) of camera: 7.0 inches = 0.1778 m
      - Laser mounted on gimbal above (+Y) camera: 1.0 inch = 0.0254 m
      - Target standoff distance: nominal 12.0 feet = 3.6576 m
    """

    enabled: bool = True
    auto_range: bool = True
    offset_x_m: float = 0.1778  # 7.0 inches
    offset_y_m: float = 0.0254  # 1.0 inch
    offset_z_m: float = 0.0     # coplanar mount
    nominal_distance_m: float = 3.6576  # 12.0 ft
    min_distance_m: float = 1.0
    max_distance_m: float = 15.0
    human_height_m: float = 1.70  # standard human height for pinhole range estimation
    camera_vfov_deg: float = 52.0  # Logitech MX Brio vertical FOV
    trim_pan_deg: float = 0.0      # manual live fine-tune trim
    trim_tilt_deg: float = 0.0     # manual live fine-tune trim

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtrinsicAlignmentConfig:
        return cls(
            enabled=bool(data.get("enabled", True)),
            auto_range=bool(data.get("auto_range", True)),
            offset_x_m=float(data.get("offset_x_m", 0.1778)),
            offset_y_m=float(data.get("offset_y_m", 0.0254)),
            offset_z_m=float(data.get("offset_z_m", 0.0)),
            nominal_distance_m=float(data.get("nominal_distance_m", 3.6576)),
            min_distance_m=float(data.get("min_distance_m", 1.0)),
            max_distance_m=float(data.get("max_distance_m", 15.0)),
            human_height_m=float(data.get("human_height_m", 1.70)),
            camera_vfov_deg=float(data.get("camera_vfov_deg", 52.0)),
            trim_pan_deg=float(data.get("trim_pan_deg", 0.0)),
            trim_tilt_deg=float(data.get("trim_tilt_deg", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "auto_range": self.auto_range,
            "offset_x_m": round(self.offset_x_m, 4),
            "offset_y_m": round(self.offset_y_m, 4),
            "offset_z_m": round(self.offset_z_m, 4),
            "offset_x_in": round(self.offset_x_m / 0.0254, 2),
            "offset_y_in": round(self.offset_y_m / 0.0254, 2),
            "nominal_distance_m": round(self.nominal_distance_m, 2),
            "nominal_distance_ft": round(self.nominal_distance_m / 0.3048, 1),
            "min_distance_m": round(self.min_distance_m, 2),
            "max_distance_m": round(self.max_distance_m, 2),
            "human_height_m": round(self.human_height_m, 2),
            "camera_vfov_deg": round(self.camera_vfov_deg, 1),
            "trim_pan_deg": round(self.trim_pan_deg, 2),
            "trim_tilt_deg": round(self.trim_tilt_deg, 2),
        }


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    angles: Angles
    distance_m: float
    distance_source: str
    parallax_pan_deg: float
    parallax_tilt_deg: float
    trim_pan_deg: float
    trim_tilt_deg: float


class ExtrinsicParallaxModel:
    """Calculates geometric parallax compensation between stationary camera and gimbal laser.

    When the camera optical center and gimbal pivot are separated by (dX, dY, dZ),
    the gimbal must aim along a convergent vector rather than the parallel optical ray.
    """

    def __init__(self, config: ExtrinsicAlignmentConfig | None = None) -> None:
        self.config = config or ExtrinsicAlignmentConfig()

    def estimate_distance(
        self,
        target: Detection | None,
        frame_height_px: int,
    ) -> tuple[float, str]:
        """Estimate target distance using the pinhole model on the detected bounding box."""
        if not self.config.auto_range or target is None or target.height <= 8.0:
            return self.config.nominal_distance_m, "nominal"

        # Focal length in pixels from vertical FOV
        vfov_rad = math.radians(max(10.0, min(120.0, self.config.camera_vfov_deg)))
        f_y = (frame_height_px / 2.0) / math.tan(vfov_rad / 2.0)

        # Distance Z = (f * H_real) / H_pixels
        raw_z = (f_y * self.config.human_height_m) / max(1.0, target.height)
        clamped_z = max(self.config.min_distance_m, min(self.config.max_distance_m, raw_z))
        return clamped_z, "auto_bbox"

    def compute_compensated_angles(
        self,
        cam_angles: Angles,
        target: Detection | None = None,
        frame_height_px: int = 480,
        apply_trim: bool = True,
    ) -> AlignmentResult:
        """Apply extrinsic parallax correction and manual trim to camera-space angles.

        Args:
            cam_angles: Angles(pan, tilt) computed from camera pixel coordinates.
            target: Optional target detection (used for bounding-box height range estimation).
            frame_height_px: Camera frame height in pixels.
            apply_trim: Whether to include trim_pan_deg and trim_tilt_deg.

        Returns:
            AlignmentResult with final commanded Angles, estimated distance, and deltas.
        """
        trim_p = self.config.trim_pan_deg if apply_trim else 0.0
        trim_t = self.config.trim_tilt_deg if apply_trim else 0.0

        if not self.config.enabled:
            final_pan = cam_angles.pan + trim_p
            final_tilt = cam_angles.tilt + trim_t
            return AlignmentResult(
                angles=Angles(final_pan, final_tilt),
                distance_m=self.config.nominal_distance_m,
                distance_source="disabled",
                parallax_pan_deg=0.0,
                parallax_tilt_deg=0.0,
                trim_pan_deg=trim_p,
                trim_tilt_deg=trim_t,
            )

        distance_m, source = self.estimate_distance(target, frame_height_px)

        # Target position in camera frame
        cam_pan_rad = math.radians(cam_angles.pan)
        cam_tilt_rad = math.radians(cam_angles.tilt)

        x_c = distance_m * math.tan(cam_pan_rad)
        y_c = distance_m * math.tan(cam_tilt_rad)
        z_c = distance_m

        # Target position in gimbal/laser frame
        # Laser is displaced by (offset_x_m, offset_y_m, offset_z_m)
        x_g = x_c - self.config.offset_x_m
        y_g = y_c - self.config.offset_y_m
        z_g = max(0.2, z_c - self.config.offset_z_m)

        # Convergent gimbal pan/tilt angles
        gimbal_pan_deg = math.degrees(math.atan2(x_g, z_g))
        gimbal_tilt_deg = math.degrees(math.atan2(y_g, z_g))

        parallax_pan = gimbal_pan_deg - cam_angles.pan
        parallax_tilt = gimbal_tilt_deg - cam_angles.tilt

        # Add user fine-tune trim if requested
        final_pan = gimbal_pan_deg + trim_p
        final_tilt = gimbal_tilt_deg + trim_t

        return AlignmentResult(
            angles=Angles(final_pan, final_tilt),
            distance_m=distance_m,
            distance_source=source,
            parallax_pan_deg=parallax_pan,
            parallax_tilt_deg=parallax_tilt,
            trim_pan_deg=trim_p,
            trim_tilt_deg=trim_t,
        )
