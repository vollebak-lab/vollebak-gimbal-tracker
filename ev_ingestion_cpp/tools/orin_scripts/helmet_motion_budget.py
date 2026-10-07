"""Helmet ego-motion budget: rotation vs translation (bounce) per analysis window.

Approximate head kinematics (order-of-magnitude literature values; verify
with our own helmet IMU logs):
  walking: vertical bounce ~5 cm p-p at ~2 Hz, head rotation ~10-30 deg/s
  running: vertical bounce ~10 cm p-p at ~2.8 Hz, head rotation ~30-60 deg/s
  head turn (voluntary): 100-300 deg/s
Translation shift of a target at range R is d/R (needs R, unknown to a
monocular passive sensor); rotation shift is range-independent.
"""
import math

PITCH = 4.86e-6
WINDOW_S = 0.033
LENSES_MM = (12, 16)
RANGES_M = (10, 30, 100, 300)

GAITS = {
    # name: (bounce p-p [m], step freq [Hz], representative rotation [deg/s])
    "walk": (0.05, 2.0, 20.0),
    "run": (0.10, 2.8, 45.0),
}


def px_per_window(rate_rad_s: float, f_mm: float) -> float:
    """Pixel displacement accumulated over one analysis window."""
    return rate_rad_s * WINDOW_S / (PITCH / (f_mm / 1000.0))


if __name__ == "__main__":
    for gait, (pp, fz, rot_dps) in GAITS.items():
        v_peak = 2 * math.pi * fz * pp / 2  # m/s, sinusoidal bounce velocity amplitude
        rot = math.radians(rot_dps)
        for f_mm in LENSES_MM:
            trans = " ".join(f"{r}m:{px_per_window(v_peak / r, f_mm):5.2f}" for r in RANGES_M)
            print(f"{gait:4s} {f_mm}mm | rotation {rot_dps:>4.0f} dps -> "
                  f"{px_per_window(rot, f_mm):6.1f} px/win | bounce v={v_peak:4.2f} m/s -> px/win {trans}")
