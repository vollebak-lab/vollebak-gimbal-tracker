from __future__ import annotations

import argparse
import importlib.util
import logging
import platform
import sys
from pathlib import Path

from .calibration import Calibration
from .camera import OpenCVCamera, require_cv2
from .config import AppConfig, load_config
from .control import clamp
from .drivers import build_driver

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gimbal-tracker")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "manual", "calibrate", "run", "auto-track", "ui"):
        command = subparsers.add_parser(name)
        command.add_argument("-c", "--config", default="config/dev.yaml")
    subparsers.choices["run"].add_argument("--preview", action="store_true")
    subparsers.choices["run"].add_argument("--max-frames", type=int)
    subparsers.choices["auto-track"].add_argument("--preview", action="store_true")
    subparsers.choices["auto-track"].add_argument("--max-frames", type=int)
    subparsers.choices["calibrate"].add_argument("--output")
    subparsers.choices["ui"].add_argument("--host", default="127.0.0.1")
    subparsers.choices["ui"].add_argument("--port", type=int, default=8080)
    subparsers.choices["ui"].add_argument(
        "--demo", action="store_true", help="force the synthetic moving-target camera"
    )
    subparsers.choices["ui"].add_argument(
        "--no-demo-fallback",
        action="store_true",
        help="show a camera fault instead of falling back to the synthetic feed",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = load_config(args.config)
        if args.command == "doctor":
            return doctor(config)
        if args.command == "manual":
            return manual(config)
        if args.command == "calibrate":
            return calibrate(config, args.output)
        if args.command == "run":
            from .tracker import run_tracker

            run_tracker(config, args.preview, args.max_frames)
            return 0
        if args.command == "auto-track":
            from .tracker import run_autonomous_tracker

            run_autonomous_tracker(config, args.preview, args.max_frames)
            return 0
        if args.command == "ui":
            from .web import serve_dashboard

            serve_dashboard(
                config,
                host=args.host,
                port=args.port,
                force_demo=args.demo,
                fallback_demo=not args.no_demo_fallback,
            )
            return 0
    except (RuntimeError, TypeError, ValueError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 2
    return 0


def doctor(config: AppConfig) -> int:
    print(f"Python:      {platform.python_version()} ({platform.machine()})")
    print(f"Platform:    {platform.platform()}")
    print(f"OpenCV:      {'installed' if importlib.util.find_spec('cv2') else 'MISSING'}")
    calibration = config.resolve(config.calibration.path)
    print(f"Calibration: {calibration} ({'found' if calibration.exists() else 'MISSING'})")
    print(f"Driver:      {config.gimbal.driver}")
    from serial.tools import list_ports

    ports = list(list_ports.comports())
    print("Serial:      " + (", ".join(port.device for port in ports) if ports else "none found"))
    if importlib.util.find_spec("cv2"):
        cv2 = require_cv2()
        discovered = []
        for index in range(4):
            capture = cv2.VideoCapture(index)
            if capture.isOpened():
                discovered.append(str(index))
            capture.release()
        print("Cameras:     " + (", ".join(discovered) if discovered else "none found"))
    print("Config validation: OK")
    return 0


def manual(config: AppConfig) -> int:
    print("Absolute manual control. Enter: PAN TILT  (or: home, quit)")
    print("Keep the mechanism clear and be ready to remove 12 V servo power.")
    driver = build_driver(config.gimbal)
    try:
        while True:
            raw = input("gimbal> ").strip().lower()
            if raw in {"q", "quit", "exit"}:
                break
            if raw == "home":
                pan, tilt = config.gimbal.home_pan, config.gimbal.home_tilt
            else:
                try:
                    pan, tilt = (float(value) for value in raw.replace(",", " ").split())
                except ValueError:
                    print("Enter two numbers, 'home', or 'quit'.")
                    continue
            pan = clamp(pan, config.gimbal.pan_min, config.gimbal.pan_max)
            tilt = clamp(tilt, config.gimbal.tilt_min, config.gimbal.tilt_max)
            driver.move(pan, tilt, config.gimbal.speed, config.gimbal.acceleration)
            print(f"commanded pan={pan:+.1f}, tilt={tilt:+.1f}")
    finally:
        driver.close()
    return 0


def calibrate(config: AppConfig, output: str | None) -> int:
    cv2 = require_cv2()
    driver = build_driver(config.gimbal)
    pan, tilt = config.gimbal.home_pan, config.gimbal.home_tilt
    selected: list[tuple[int, int] | None] = [None]
    pixels: list[tuple[float, float]] = []
    angles: list[tuple[float, float]] = []
    window = "Fixed-camera calibration"

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected[0] = (x, y)

    print("Aim the gimbal at a visible physical point, then click that point in the camera image.")
    print("W/S tilt, A/D pan, shift+key for 5 degrees, C capture, R reset, Enter save, Q cancel.")
    try:
        with OpenCVCamera(config.camera) as camera:
            cv2.namedWindow(window)
            cv2.setMouseCallback(window, on_mouse)
            driver.move(pan, tilt, config.gimbal.speed, config.gimbal.acceleration)
            while True:
                frame = camera.read()
                for point in pixels:
                    cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
                if selected[0]:
                    cv2.drawMarker(frame, selected[0], (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(
                    frame,
                    f"pan={pan:+.1f} tilt={tilt:+.1f} samples={len(pixels)}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (50, 255, 50),
                    2,
                )
                cv2.imshow(window, frame)
                key = cv2.waitKey(20) & 0xFF
                if key in (ord("q"), 27):
                    print("Calibration cancelled.")
                    return 1
                if key in (13, 10):
                    if len(pixels) < 4:
                        print("Need at least four samples; six or more is recommended.")
                        continue
                    height, width = frame.shape[:2]
                    result = Calibration.fit(pixels, angles, width, height)
                    output_path = Path(output) if output else config.resolve(config.calibration.path)
                    result.save(output_path)
                    print(f"Saved {len(pixels)} samples to {output_path}")
                    print(f"Calibration RMS error: {result.rms_error_deg:.3f} degrees")
                    return 0
                if key == ord("r"):
                    pixels.clear()
                    angles.clear()
                    selected[0] = None
                    continue
                if key == ord("c"):
                    if selected[0] is None:
                        print("Click the corresponding point in the camera image first.")
                    else:
                        pixels.append((float(selected[0][0]), float(selected[0][1])))
                        angles.append((pan, tilt))
                        print(f"sample {len(pixels)}: pixel={selected[0]} -> ({pan:+.1f}, {tilt:+.1f})")
                        selected[0] = None
                    continue
                delta = 5.0 if key in (ord("W"), ord("A"), ord("S"), ord("D")) else 1.0
                moved = True
                if key in (ord("a"), ord("A")):
                    pan -= delta
                elif key in (ord("d"), ord("D")):
                    pan += delta
                elif key in (ord("s"), ord("S")):
                    tilt -= delta
                elif key in (ord("w"), ord("W")):
                    tilt += delta
                else:
                    moved = False
                if moved:
                    pan = clamp(pan, config.gimbal.pan_min, config.gimbal.pan_max)
                    tilt = clamp(tilt, config.gimbal.tilt_min, config.gimbal.tilt_max)
                    driver.move(pan, tilt, config.gimbal.speed, config.gimbal.acceleration)
    finally:
        driver.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
