from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Detection:
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0
    label: str = "target"
    custom_center: tuple[float, float] | None = None

    @property
    def center(self) -> tuple[float, float]:
        if self.custom_center is not None:
            return self.custom_center
        return self.x + self.width / 2.0, self.y + self.height / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class Angles:
    pan: float
    tilt: float
