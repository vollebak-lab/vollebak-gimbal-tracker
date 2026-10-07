from dataclasses import replace

import pytest

from vollebak_gimbal.config import LaserTestConfig, load_config
from vollebak_gimbal.laser_test import LaserTestController
from vollebak_gimbal.web import DashboardEngine


def test_laser_test_pulse_is_brief_and_ends_low():
    commands: list[list[str]] = []
    now = [10.0]

    def run(command, **_kwargs):
        commands.append(command)

    controller = LaserTestController(
        LaserTestConfig(enabled=True, gpio=17, pulse_s=0.1, cooldown_s=2.0),
        runner=run,
        sleeper=lambda duration: now.__setitem__(0, now[0] + duration),
        clock=lambda: now[0],
        pinctrl_path="/usr/bin/pinctrl",
    )

    controller.pulse()

    assert [command[-1] for command in commands] == ["dl", "dl", "dh", "dl"]
    assert controller.snapshot()["active"] is False
    assert controller.snapshot()["cooldown_remaining_s"] == 2.0


def test_laser_test_cooldown_blocks_repeat_pulse():
    now = [10.0]
    controller = LaserTestController(
        LaserTestConfig(enabled=True),
        runner=lambda *_args, **_kwargs: None,
        sleeper=lambda duration: now.__setitem__(0, now[0] + duration),
        clock=lambda: now[0],
        pinctrl_path="/usr/bin/pinctrl",
    )
    controller.pulse()

    with pytest.raises(ValueError, match="cooldown"):
        controller.pulse()


def test_dashboard_rejects_laser_pulse_while_tracking():
    config = load_config("config/dev.yaml")
    config.laser_test = replace(config.laser_test, enabled=False)
    engine = DashboardEngine(config, force_demo=True)
    try:
        assert engine.state()["tracking_enabled"] is True
        with pytest.raises(ValueError, match="Pause tracking"):
            engine.pulse_laser_test()
    finally:
        engine.close()
