from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import Angles


@dataclass(slots=True)
class Calibration:
    matrix: np.ndarray
    camera_width: int
    camera_height: int
    rms_error_deg: float = 0.0
    samples: list[dict[str, list[float]]] | None = None

    def map_pixel(self, x: float, y: float) -> Angles:
        result = self.matrix @ np.array([x, y, 1.0], dtype=float)
        if abs(result[2]) < 1e-9:
            raise ValueError("Calibration maps this pixel to infinity")
        return Angles(float(result[0] / result[2]), float(result[1] / result[2]))

    def save(self, path: str | Path) -> None:
        payload = {
            "version": 1,
            "camera_width": self.camera_width,
            "camera_height": self.camera_height,
            "matrix": self.matrix.tolist(),
            "rms_error_deg": self.rms_error_deg,
            "samples": self.samples or [],
        }
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        matrix = np.asarray(payload["matrix"], dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("Calibration matrix must be a finite 3x3 array")
        return cls(
            matrix=matrix,
            camera_width=int(payload["camera_width"]),
            camera_height=int(payload["camera_height"]),
            rms_error_deg=float(payload.get("rms_error_deg", 0.0)),
            samples=payload.get("samples", []),
        )

    @classmethod
    def fit(
        cls,
        pixels: list[tuple[float, float]],
        angles: list[tuple[float, float]],
        camera_width: int,
        camera_height: int,
    ) -> Calibration:
        if len(pixels) != len(angles) or len(pixels) < 4:
            raise ValueError("At least four matching pixel/angle samples are required")

        rows: list[list[float]] = []
        values: list[float] = []
        for (x, y), (pan, tilt) in zip(pixels, angles, strict=True):
            rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -pan * x, -pan * y])
            values.append(pan)
            rows.append([0.0, 0.0, 0.0, x, y, 1.0, -tilt * x, -tilt * y])
            values.append(tilt)

        coefficients, _, rank, _ = np.linalg.lstsq(
            np.asarray(rows, dtype=float), np.asarray(values, dtype=float), rcond=None
        )
        if rank < 8:
            raise ValueError("Calibration points are degenerate; spread them across the image")
        matrix = np.append(coefficients, 1.0).reshape(3, 3)
        calibration = cls(matrix, camera_width, camera_height)
        errors = []
        for pixel, expected in zip(pixels, angles, strict=True):
            actual = calibration.map_pixel(*pixel)
            errors.append((actual.pan - expected[0]) ** 2 + (actual.tilt - expected[1]) ** 2)
        calibration.rms_error_deg = float(np.sqrt(np.mean(errors)))
        calibration.samples = [
            {"pixel": [float(x), float(y)], "angles": [float(pan), float(tilt)]}
            for (x, y), (pan, tilt) in zip(pixels, angles, strict=True)
        ]
        return calibration
