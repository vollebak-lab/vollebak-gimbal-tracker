#!/usr/bin/env python3
"""Low-latency, loopback-only MJPEG bridge for a Windows USB camera."""

from __future__ import annotations

import argparse
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


class CameraBridge:
    def __init__(self, source: int, width: int, height: int, fps: int, quality: int) -> None:
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(source, backend)
        if hasattr(cv2, "VideoWriter_fourcc"):
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera source {source}")

        self.quality = quality
        self.condition = threading.Condition()
        self.frame: bytes | None = None
        self.frame_id = 0
        self.actual_width = width
        self.actual_height = height
        self.measured_fps = 0.0
        self.stopped = threading.Event()

    def run(self) -> None:
        previous = time.monotonic()
        while not self.stopped.is_set():
            ok, frame = self.capture.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            ok, encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            )
            if not ok:
                continue
            now = time.monotonic()
            instantaneous = 1.0 / max(now - previous, 1e-6)
            previous = now
            self.measured_fps = (
                instantaneous
                if self.measured_fps == 0.0
                else self.measured_fps * 0.9 + instantaneous * 0.1
            )
            self.actual_height, self.actual_width = frame.shape[:2]
            with self.condition:
                self.frame = encoded.tobytes()
                self.frame_id += 1
                self.condition.notify_all()

    def close(self) -> None:
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()
        self.capture.release()


def make_handler(bridge: CameraBridge):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/health":
                payload = json.dumps(
                    {
                        "ok": bridge.frame is not None,
                        "fps": round(bridge.measured_fps, 1),
                        "width": bridge.actual_width,
                        "height": bridge.actual_height,
                    }
                ).encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if self.path != "/stream.mjpg":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            previous_id = -1
            try:
                while not bridge.stopped.is_set():
                    with bridge.condition:
                        bridge.condition.wait_for(
                            lambda: bridge.frame_id != previous_id or bridge.stopped.is_set(),
                            timeout=2.0,
                        )
                        previous_id = bridge.frame_id
                        frame = bridge.frame
                    if frame is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return

        def log_message(self, *_args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--quality", type=int, default=70)
    args = parser.parse_args()

    bridge = CameraBridge(args.source, args.width, args.height, args.fps, args.quality)
    worker = threading.Thread(target=bridge.run, name="rgb-capture", daemon=True)
    worker.start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(bridge))
    server.daemon_threads = True
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        bridge.close()
        worker.join(timeout=2.0)


if __name__ == "__main__":
    main()
