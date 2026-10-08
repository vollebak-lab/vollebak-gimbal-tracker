from __future__ import annotations

import math
from dataclasses import replace

import pytest

from vollebak_gimbal.alignment import (
    ExtrinsicAlignmentConfig,
    ExtrinsicParallaxModel,
)
from vollebak_gimbal.config import load_config
from vollebak_gimbal.models import Angles, Detection
from vollebak_gimbal.web import DashboardEngine


def test_extrinsic_alignment_config_serialization():
    config = ExtrinsicAlignmentConfig(
        offset_x_m=0.1778,
        offset_y_m=0.0254,
        nominal_distance_m=3.6576,
        trim_pan_deg=1.2,
        trim_tilt_deg=-0.4,
    )
    d = config.to_dict()
    assert d["offset_x_in"] == 7.0
    assert d["offset_y_in"] == 1.0
    assert round(d["nominal_distance_ft"], 1) == 12.0
    assert d["trim_pan_deg"] == 1.2
    assert d["trim_tilt_deg"] == -0.4

    restored = ExtrinsicAlignmentConfig.from_dict(d)
    assert math.isclose(restored.offset_x_m, 0.1778, rel_tol=1e-3)
    assert math.isclose(restored.offset_y_m, 0.0254, rel_tol=1e-3)
    assert restored.trim_pan_deg == 1.2
    assert restored.trim_tilt_deg == -0.4


def test_extrinsic_parallax_calculation():
    # Laser is 7 inches (0.1778m) to the right, 1 inch (0.0254m) above camera center
    # Target is on optical axis at nominal 12.0 ft (3.6576m)
    config = ExtrinsicAlignmentConfig(
        offset_x_m=0.1778,
        offset_y_m=0.0254,
        nominal_distance_m=3.6576,
        trim_pan_deg=0.0,
        trim_tilt_deg=0.0,
    )
    model = ExtrinsicParallaxModel(config)
    res = model.compute_compensated_angles(Angles(0.0, 0.0))

    # Because laser is to the right (+X), gimbal must pan to the left (negative angle) to hit target
    expected_pan = math.degrees(math.atan2(-0.1778, 3.6576))
    assert math.isclose(res.angles.pan, expected_pan, abs_tol=1e-2)
    assert res.angles.pan < 0.0  # inward convergence

    # Because laser is higher (+Y), gimbal must tilt slightly downward (negative angle)
    expected_tilt = math.degrees(math.atan2(-0.0254, 3.6576))
    assert math.isclose(res.angles.tilt, expected_tilt, abs_tol=1e-2)
    assert res.angles.tilt < 0.0


def test_parallax_scales_with_target_distance():
    config = ExtrinsicAlignmentConfig(
        offset_x_m=0.1778,
        offset_y_m=0.0254,
        nominal_distance_m=3.6576,
    )
    model = ExtrinsicParallaxModel(config)

    # Far target (10m) -> small parallax
    far_target = Detection(x=270, y=100, width=100, height=80)  # small bbox -> far
    res_far = model.compute_compensated_angles(Angles(0.0, 0.0), target=far_target)

    # Close target (2m) -> large bbox -> large parallax
    close_target = Detection(x=200, y=50, width=240, height=350)  # large bbox -> close
    res_close = model.compute_compensated_angles(Angles(0.0, 0.0), target=close_target)

    assert res_close.distance_m < res_far.distance_m
    assert abs(res_close.parallax_pan_deg) > abs(res_far.parallax_pan_deg)


def test_auto_range_estimation_and_clamping():
    config = ExtrinsicAlignmentConfig(
        min_distance_m=1.0,
        max_distance_m=10.0,
        nominal_distance_m=3.5,
    )
    model = ExtrinsicParallaxModel(config)

    # None target falls back to nominal
    dist, src = model.estimate_distance(None, 480)
    assert dist == 3.5
    assert src == "nominal"

    # Extremely tall bbox clamps to min_distance_m
    giant_target = Detection(x=0, y=0, width=640, height=2000)
    dist_min, src_min = model.estimate_distance(giant_target, 480)
    assert dist_min == 1.0
    assert src_min == "auto_bbox"

    # Extremely tiny bbox clamps to max_distance_m
    tiny_target = Detection(x=0, y=0, width=10, height=12)
    dist_max, src_max = model.estimate_distance(tiny_target, 480)
    assert dist_max == 10.0
    assert src_max == "auto_bbox"


def test_manual_trim_offsets():
    config = ExtrinsicAlignmentConfig(
        offset_x_m=0.0,
        offset_y_m=0.0,
        trim_pan_deg=2.5,
        trim_tilt_deg=-1.5,
    )
    model = ExtrinsicParallaxModel(config)
    res = model.compute_compensated_angles(Angles(5.0, 10.0))
    assert math.isclose(res.angles.pan, 7.5, abs_tol=1e-3)
    assert math.isclose(res.angles.tilt, 8.5, abs_tol=1e-3)
    assert res.trim_pan_deg == 2.5
    assert res.trim_tilt_deg == -1.5


def test_dashboard_engine_alignment_methods(monkeypatch):
    try:
        import cv2  # noqa: F401
    except ImportError:
        from unittest.mock import MagicMock
        mock_cv2 = MagicMock()
        monkeypatch.setattr("vollebak_gimbal.camera.require_cv2", lambda: mock_cv2)
        monkeypatch.setattr("vollebak_gimbal.web.require_cv2", lambda: mock_cv2)

    config = load_config("config/dev.yaml")
    config.alignment.offset_x_m = 0.1778
    config.alignment.offset_y_m = 0.0254
    engine = DashboardEngine(config, force_demo=True)
    try:
        initial = engine.get_alignment()
        assert initial["offset_x_in"] == 7.0
        assert initial["offset_y_in"] == 1.0

        updated = engine.update_alignment({
            "trim_pan_deg": -1.2,
            "trim_tilt_deg": 0.8,
            "nominal_distance_ft": 15.0,
        })
        assert updated["trim_pan_deg"] == -1.2
        assert updated["trim_tilt_deg"] == 0.8
        assert round(updated["nominal_distance_ft"], 1) == 15.0

        state = engine.state()
        assert state["alignment"]["trim_pan_deg"] == -1.2
        assert state["alignment_live"]["trim_pan_deg"] == -1.2
    finally:
        engine.close()


def test_lock_current_aim_exact_trim_transfer(monkeypatch):
    try:
        import cv2  # noqa: F401
    except ImportError:
        from unittest.mock import MagicMock
        mock_cv2 = MagicMock()
        monkeypatch.setattr("vollebak_gimbal.camera.require_cv2", lambda: mock_cv2)
        monkeypatch.setattr("vollebak_gimbal.web.require_cv2", lambda: mock_cv2)

    config = load_config("config/dev.yaml")
    engine = DashboardEngine(config, force_demo=True)
    try:
        # Simulate target detected at center of frame
        engine._latest_target = Detection(x=100.0, y=70.0, width=120.0, height=100.0, label="person")
        engine._latest_frame_shape = (240, 320)

        # Operator manually aims laser at target
        target_pan, target_tilt = 12.5, -4.2
        engine.move(target_pan, target_tilt)
        assert engine._manual_aim_active is True

        # Lock current aim
        res = engine.lock_current_aim()
        assert engine._manual_aim_active is False

        # Verify that computing compensated angles with trim now yields EXACT target angles
        cam_angles = engine.calibration.map_pixel(
            160.0 * engine.calibration.camera_width / 320,
            120.0 * engine.calibration.camera_height / 240,
        )
        comp = engine.alignment_model.compute_compensated_angles(
            cam_angles, target=engine._latest_target, frame_height_px=240, apply_trim=True
        )
        assert math.isclose(comp.angles.pan, target_pan, abs_tol=1e-2)
        assert math.isclose(comp.angles.tilt, target_tilt, abs_tol=1e-2)
        assert math.isclose(res["trim_pan_deg"], engine.alignment_model.config.trim_pan_deg, abs_tol=1e-2)
    finally:
        engine.close()


def test_seamless_handoff_on_set_tracking(monkeypatch):
    try:
        import cv2  # noqa: F401
    except ImportError:
        from unittest.mock import MagicMock
        mock_cv2 = MagicMock()
        monkeypatch.setattr("vollebak_gimbal.camera.require_cv2", lambda: mock_cv2)
        monkeypatch.setattr("vollebak_gimbal.web.require_cv2", lambda: mock_cv2)

    config = load_config("config/dev.yaml")
    config.tracking.start_enabled = False
    engine = DashboardEngine(config, force_demo=True)
    try:
        engine._latest_target = Detection(x=120.0, y=80.0, width=80.0, height=80.0, label="person")
        engine._latest_frame_shape = (240, 320)

        # Operator nudges gimbal
        engine.move(7.5, -2.5)
        assert engine._manual_aim_active is True

        # Operator clicks "START TRACKING"
        engine.set_tracking(True)
        assert engine.state()["tracking_enabled"] is True
        assert engine._manual_aim_active is False

        # Trim was automatically locked
        assert engine.alignment_model.config.trim_pan_deg != 0.0 or engine.alignment_model.config.trim_tilt_deg != 0.0
    finally:
        engine.close()


def test_lock_current_aim_requires_target(monkeypatch):
    try:
        import cv2  # noqa: F401
    except ImportError:
        from unittest.mock import MagicMock
        mock_cv2 = MagicMock()
        monkeypatch.setattr("vollebak_gimbal.camera.require_cv2", lambda: mock_cv2)
        monkeypatch.setattr("vollebak_gimbal.web.require_cv2", lambda: mock_cv2)

    config = load_config("config/dev.yaml")
    engine = DashboardEngine(config, force_demo=True)
    try:
        engine._latest_target = None
        with pytest.raises(ValueError, match="No target currently detected"):
            engine.lock_current_aim()
    finally:
        engine.close()

