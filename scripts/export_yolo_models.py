#!/usr/bin/env python3
"""Download and export YOLOv8/v11 models to ONNX for Raspberry Pi 5 edge execution."""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("export_yolo")

MODEL_URLS = {
    "yolov8n": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    "yolov8n-pose": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt",
}


def download_file(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 100_000:
        LOGGER.info("File already exists: %s", dest)
        return dest
    LOGGER.info("Downloading %s -> %s ...", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    LOGGER.info("Downloaded %d bytes", dest.stat().st_size)
    return dest


def export_to_onnx(model_name: str, output_dir: Path, imgsz: int = 320) -> Path:
    from ultralytics import YOLO

    pt_url = MODEL_URLS[model_name]
    pt_path = output_dir / f"{model_name}.pt"
    download_file(pt_url, pt_path)

    LOGGER.info("Loading PyTorch model: %s", pt_path)
    model = YOLO(str(pt_path))

    LOGGER.info("Exporting to ONNX format (imgsz=%d)...", imgsz)
    onnx_path_str = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,
        simplify=True,
    )
    onnx_path = Path(onnx_path_str)
    target_path = output_dir / f"{model_name}.onnx"
    if onnx_path.resolve() != target_path.resolve() and onnx_path.exists():
        import shutil
        shutil.move(str(onnx_path), str(target_path))
    LOGGER.info("Successfully exported: %s (%d bytes)", target_path, target_path.stat().st_size)
    return target_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLOv8 models to ONNX for Pi 5")
    parser.add_argument(
        "--model",
        choices=["yolov8n", "yolov8n-pose", "all"],
        default="yolov8n",
        help="Model variant to export",
    )
    parser.add_argument("--imgsz", type=int, default=320, help="Input resolution (default 320 for Pi 5)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models",
        help="Target output directory",
    )
    args = parser.parse_args()

    models = ["yolov8n", "yolov8n-pose"] if args.model == "all" else [args.model]
    for m in models:
        try:
            export_to_onnx(m, args.output_dir, imgsz=args.imgsz)
        except Exception as exc:
            LOGGER.error("Failed to export %s: %s", m, exc)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
