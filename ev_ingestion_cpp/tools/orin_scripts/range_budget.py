"""Optical range budget for IMX636 (4.86 um, 1280x720) vs focal length and platform.

Platform-agnostic classes (rotor diameter, blade chord, airframe span):
  small  : 5" FPV         prop 0.127 m, chord 0.012 m, span 0.22 m
  medium : Mavic Air 2    prop 0.183 m, chord 0.018 m, span 0.35 m
  large  : Matrice-class  prop 0.533 m, chord 0.045 m, span 0.90 m
Blade-pass modulation per pixel (sub-pixel regime) ~ (blade area inside pixel footprint)/(footprint area)
  ~ min(1, chord/GSD) * min(1, rotor_radius/GSD)   (upper bound, perfect focus, contrast=1)
"""
import math

PITCH = 4.86e-6
W, H = 1280, 720
PLAT = {"small 5in": (0.127, 0.012, 0.22), "medium Mavic": (0.183, 0.018, 0.35), "large M300": (0.533, 0.045, 0.90)}
RANGES = [35, 100, 200, 300]
for f_mm in (8, 12, 16, 25, 35, 50):
    f = f_mm / 1000
    ifov = PITCH / f
    hfov = math.degrees(2 * math.atan(W * PITCH / 2 / f))
    vfov = math.degrees(2 * math.atan(H * PITCH / 2 / f))
    print(f"\n== {f_mm} mm: IFOV {ifov*1e3:.3f} mrad  HFOV {hfov:.1f} deg  VFOV {vfov:.1f} deg")
    for name, (prop, chord, span) in PLAT.items():
        row = []
        for R in RANGES:
            gsd = ifov * R
            rotor_px = prop / gsd
            span_px = span / gsd
            mod = min(1.0, chord / gsd) * min(1.0, (prop / 2) / gsd)
            row.append(f"{R}m: rotor {rotor_px:5.1f}px span {span_px:5.1f}px mod<={mod:5.3f}")
        print(f"  {name:13s} | " + " | ".join(row))
