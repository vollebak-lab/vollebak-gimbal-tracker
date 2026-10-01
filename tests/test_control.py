from vollebak_gimbal.config import GimbalConfig, TrackingConfig
from vollebak_gimbal.control import SafeAngleController
from vollebak_gimbal.models import Angles


def test_controller_clamps_and_step_limits():
    controller = SafeAngleController(
        GimbalConfig(pan_min=-10, pan_max=10, tilt_min=-5, tilt_max=5),
        TrackingConfig(smoothing_alpha=1.0, deadband_deg=0.0, max_step_deg=2.0),
    )

    first = controller.update(Angles(99, -99))
    second = controller.update(Angles(99, -99))

    assert first == Angles(2.0, -2.0)
    assert second == Angles(4.0, -4.0)


def test_controller_deadband_holds_position():
    controller = SafeAngleController(
        GimbalConfig(),
        TrackingConfig(smoothing_alpha=1.0, deadband_deg=1.0, max_step_deg=5.0),
    )
    controller.reset(Angles(3.0, 4.0))
    assert controller.update(Angles(3.5, 3.5)) == Angles(3.0, 4.0)
