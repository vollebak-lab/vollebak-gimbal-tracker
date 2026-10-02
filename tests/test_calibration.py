import numpy as np

from vollebak_gimbal.calibration import Calibration


def test_fit_and_map_affine_calibration():
    pixels = [(0, 0), (640, 0), (640, 480), (0, 480), (320, 240)]
    angles = [(-45, 30), (45, 30), (45, -30), (-45, -30), (0, 0)]
    calibration = Calibration.fit(pixels, angles, 640, 480)

    mapped = calibration.map_pixel(160, 120)
    assert np.isclose(mapped.pan, -22.5, atol=1e-6)
    assert np.isclose(mapped.tilt, 15.0, atol=1e-6)
    assert calibration.rms_error_deg < 1e-6


def test_fit_rejects_degenerate_points():
    pixels = [(0, 0), (1, 0), (2, 0), (3, 0)]
    angles = [(0, 0), (1, 0), (2, 0), (3, 0)]

    try:
        Calibration.fit(pixels, angles, 640, 480)
    except ValueError as exc:
        assert "degenerate" in str(exc)
    else:
        raise AssertionError("Expected degenerate calibration to fail")
