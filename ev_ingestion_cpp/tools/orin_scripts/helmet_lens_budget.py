"""Helmet vs stationary lens trade study for IMX636 (1280x720, 4.86 um pitch).

Outputs per focal length: FOV, IFOV, range at which a rotor spans 3 px
(platform classes), head-rate tolerance (1 px smear per 33 ms window
without compensation), required gyro<->event sync for 0.5 px at a given
head rate, and hyperfocal distance at f/1.4, f/2.0, f/2.5 (CoC = 1 px).
"""
import math

PITCH = 4.86e-6
W_PX, H_PX = 1280, 720
SENSOR_W, SENSOR_H = W_PX * PITCH, H_PX * PITCH
ROTORS_M = {"5in FPV": 0.127, "Mavic-class": 0.182, "M300-class": 0.533}
PX_REQ = 3.0           # rotor pixels needed (literature/our 3 px design point)
WINDOW_S = 0.033       # DDHF-style per-pixel window
HEAD_RATES = (2.0, 30.0)  # deg/s: quiet standing, walking (approx.)


def row(f_mm: float) -> str:
    f = f_mm / 1000.0
    hfov = math.degrees(2 * math.atan(SENSOR_W / 2 / f))
    vfov = math.degrees(2 * math.atan(SENSOR_H / 2 / f))
    ifov = PITCH / f  # rad/px
    ranges = " ".join(f"{k}={d / (PX_REQ * ifov):5.0f}m" for k, d in ROTORS_M.items())
    rate_1px = math.degrees(ifov / WINDOW_S)  # deg/s for 1 px smear per window
    sync = " ".join(
        f"{r:g}dps:{0.5 * ifov / math.radians(r) * 1e3:5.2f}ms" for r in HEAD_RATES
    )
    hyp = " ".join(f"f/{n}:{f * f / (n * PITCH):5.1f}m" for n in (1.4, 2.0, 2.5))
    return (f"{f_mm:>4.0f}mm HFOV {hfov:5.1f} VFOV {vfov:4.1f} IFOV {ifov * 1e3:5.3f}mrad | "
            f"3px@ {ranges} | 1px/33ms @ {rate_1px:4.2f}dps | sync0.5px {sync} | H {hyp}")


if __name__ == "__main__":
    for fl in (6, 8, 12, 16, 20, 25, 35, 50):
        print(row(fl))
