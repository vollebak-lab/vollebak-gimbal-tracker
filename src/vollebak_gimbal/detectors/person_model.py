from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..camera import require_cv2
from ..models import Detection

LOGGER = logging.getLogger(__name__)


def compute_anatomical_com(
    x: float,
    y: float,
    width: float,
    height: float,
    keypoints: list[tuple[float, float, float]] | None = None,
    mode: str = "torso_chest",
    torso_y_ratio: float = 0.38,
    keypoint_conf_thresh: float = 0.3,
) -> tuple[float, float]:
    """Compute human center-of-mass from pose keypoints or anthropometric torso offsets.

    Keypoints indexing follows the standard COCO 17-keypoint layout:
      5: left_shoulder, 6: right_shoulder, 11: left_hip, 12: right_hip
    """
    if mode == "pose" and keypoints is not None and len(keypoints) >= 17:
        ls = keypoints[5]
        rs = keypoints[6]
        lh = keypoints[11]
        rh = keypoints[12]

        shoulders_valid = ls[2] >= keypoint_conf_thresh and rs[2] >= keypoint_conf_thresh
        hips_valid = lh[2] >= keypoint_conf_thresh and rh[2] >= keypoint_conf_thresh

        if shoulders_valid and hips_valid:
            # Centroid of the torso quadrilateral (midpoint of shoulder line and hip line)
            cx = (ls[0] + rs[0] + lh[0] + rh[0]) / 4.0
            cy = (ls[1] + rs[1] + lh[1] + rh[1]) / 4.0
            return float(cx), float(cy)
        elif shoulders_valid:
            # Chest midpoint offset downwards towards sternum
            cx = (ls[0] + rs[0]) / 2.0
            cy = (ls[1] + rs[1]) / 2.0 + 0.15 * height
            return float(cx), float(cy)
        elif hips_valid:
            # Pelvis midpoint offset upwards towards solar plexus
            cx = (lh[0] + rh[0]) / 2.0
            cy = (lh[1] + rh[1]) / 2.0 - 0.15 * height
            return float(cx), float(cy)

    if mode == "bbox_center":
        return float(x + width / 2.0), float(y + height / 2.0)

    # Default: Anthropometric human center-of-mass at sternum/thorax
    # Standard biomechanical center of mass for an upright human is located
    # approximately 38% down from the vertex of the head inside the torso.
    com_x = x + width * 0.50
    com_y = y + height * torso_y_ratio
    return float(com_x), float(com_y)


def decode_yolo_output(
    output_tensor: np.ndarray,
    orig_shape: tuple[int, int],
    input_shape: tuple[int, int],
    conf_thresh: float = 0.40,
    nms_thresh: float = 0.45,
    person_class_id: int = 0,
    mode: str = "torso_chest",
    torso_y_ratio: float = 0.38,
    min_area: float = 400.0,
) -> list[Detection]:
    """Decode raw YOLOv8/v11 output tensor into person detections with center-of-mass."""
    cv2 = require_cv2()
    tensor = output_tensor
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]  # Remove batch dim: (C, N) or (N, C)

    # Transpose to (N, C)
    if tensor.shape[1] in (56, 84):
        pass  # Already (N, C)
    elif tensor.shape[0] in (56, 84):
        tensor = tensor.T  # (C, N) -> (N, C)
    elif tensor.shape[0] <= 85 and tensor.shape[1] != tensor.shape[0]:
        # If axis 0 is a known feature dimension (e.g. 5, 6, 84) and axis 1 is candidates
        tensor = tensor.T

    n_candidates, channels = tensor.shape
    if channels < 5:
        return []

    orig_h, orig_w = orig_shape
    net_w, net_h = input_shape
    scale_x = orig_w / float(net_w)
    scale_y = orig_h / float(net_h)

    boxes: list[list[int]] = []
    float_boxes: list[tuple[float, float, float, float]] = []
    confidences: list[float] = []
    com_coords: list[tuple[float, float]] = []
    keypoints_list: list[list[tuple[float, float, float]] | None] = []

    is_pose = (channels == 56)  # 4 bbox + 1 conf + 17 * 3 keypoints

    for row in tensor:
        if is_pose:
            # YOLO Pose format: [cx, cy, w, h, box_score, kpt0_x, kpt0_y, kpt0_conf, ...]
            score = float(row[4])
            if score < conf_thresh:
                continue
            cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            
            # Extract keypoints
            raw_kpts = row[5:].reshape(-1, 3)
            kpts = []
            for kx, ky, kc in raw_kpts:
                kpts.append((float(kx * scale_x), float(ky * scale_y), float(kc)))
            parsed_keypoints = kpts
        else:
            # Standard YOLO detection format: [cx, cy, w, h, class0_prob, class1_prob, ...]
            cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            if channels > 4:
                class_scores = row[4:]
                if person_class_id < len(class_scores):
                    score = float(class_scores[person_class_id])
                else:
                    score = float(np.max(class_scores))
            else:
                score = 1.0

            if score < conf_thresh:
                continue
            parsed_keypoints = None

        # Convert center x/y/w/h in network resolution to corner x/y/w/h in original frame
        x1 = (cx - w / 2.0) * scale_x
        y1 = (cy - h / 2.0) * scale_y
        box_w = w * scale_x
        box_h = h * scale_y

        if (box_w * box_h) < min_area:
            continue

        # Compute anatomical center of mass
        com = compute_anatomical_com(
            x1, y1, box_w, box_h,
            keypoints=parsed_keypoints,
            mode=mode,
            torso_y_ratio=torso_y_ratio,
        )

        boxes.append([int(x1), int(y1), int(box_w), int(box_h)])
        float_boxes.append((float(x1), float(y1), float(box_w), float(box_h)))
        confidences.append(float(score))
        com_coords.append(com)
        keypoints_list.append(parsed_keypoints)

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
    if len(indices) == 0:
        return []

    selected_indices = indices.flatten() if hasattr(indices, "flatten") else [i[0] for i in indices]

    results: list[Detection] = []
    for idx in selected_indices:
        fb = float_boxes[idx]
        conf = confidences[idx]
        com = com_coords[idx]
        results.append(
            Detection(
                x=fb[0],
                y=fb[1],
                width=fb[2],
                height=fb[3],
                confidence=conf,
                label="person",
                custom_center=com,
            )
        )
    return results


class PersonModelDetector:
    """Production neural person detector running on Raspberry Pi 5.

    Supports ONNX Runtime and OpenCV DNN (cv2.dnn) backends with YOLOv8/v11
    inference and anatomical Center-of-Mass (CoM) extraction.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.cv2 = require_cv2()
        self.config = config
        self.model_path = config.get("model_path")
        self.input_width = int(config.get("input_width", 640))
        self.input_height = int(config.get("input_height", 640))
        self.conf_thresh = float(config.get("confidence_threshold", 0.40))
        self.nms_thresh = float(config.get("nms_threshold", 0.45))
        self.person_class_id = int(config.get("person_class_id", 0))
        self.mode = str(config.get("center_of_mass_mode", "torso_chest"))
        self.torso_y_ratio = float(config.get("torso_y_offset_ratio", 0.38))
        self.min_area = float(config.get("min_area", 400.0))
        self.backend = str(config.get("backend", "auto")).lower()

        self._session: Any = None
        self._net: Any = None
        self._active_backend = "mock"

        self._initialize_backend()

    def _initialize_backend(self) -> None:
        if not self.model_path:
            LOGGER.info("No model_path specified; PersonModelDetector running in mock/synthetic mode")
            return

        resolved_path = Path(self.model_path)
        if not resolved_path.exists():
            LOGGER.warning("Model file not found at %s; operating in mock mode", resolved_path)
            return

        path_str = str(resolved_path.resolve())

        # Attempt ONNX Runtime first if requested or auto
        if self.backend in ("auto", "onnxruntime"):
            try:
                import onnxruntime as ort

                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = int(self.config.get("threads", 4))
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                self._session = ort.InferenceSession(
                    path_str, sess_options, providers=["CPUExecutionProvider"]
                )
                self._input_name = self._session.get_inputs()[0].name
                self._output_name = self._session.get_outputs()[0].name
                self._active_backend = "onnxruntime"
                LOGGER.info("PersonModelDetector initialized with ONNX Runtime: %s", path_str)
                return
            except (ImportError, Exception) as exc:
                LOGGER.warning("ONNX Runtime initialization failed (%s); falling back to cv2.dnn", exc)

        # Attempt OpenCV DNN
        if self.backend in ("auto", "opencv") or self._session is None:
            try:
                self._net = self.cv2.dnn.readNetFromONNX(path_str)
                self._net.setPreferableBackend(self.cv2.dnn.DNN_BACKEND_OPENCV)
                self._net.setPreferableTarget(self.cv2.dnn.DNN_TARGET_CPU)
                self._active_backend = "opencv"
                LOGGER.info("PersonModelDetector initialized with OpenCV DNN: %s", path_str)
            except Exception as exc:
                LOGGER.error("Failed to load model with OpenCV DNN: %s", exc)
                self._active_backend = "mock"

    @property
    def active_backend(self) -> str:
        return self._active_backend

    def detect(self, frame: Any) -> list[Detection]:
        if frame is None or frame.size == 0:
            return []

        orig_h, orig_w = frame.shape[:2]

        if self._active_backend == "mock":
            # Passthrough if mock or test mode
            return []

        # Preprocess frame: resize to (input_width, input_height), RGB normalize
        blob = self.cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=(self.input_width, self.input_height),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )

        if self._active_backend == "onnxruntime":
            outputs = self._session.run([self._output_name], {self._input_name: blob})
            raw_output = outputs[0]
        elif self._active_backend == "opencv":
            self._net.setInput(blob)
            raw_output = self._net.forward()
        else:
            return []

        return decode_yolo_output(
            raw_output,
            orig_shape=(orig_h, orig_w),
            input_shape=(self.input_width, self.input_height),
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
            person_class_id=self.person_class_id,
            mode=self.mode,
            torso_y_ratio=self.torso_y_ratio,
            min_area=self.min_area,
        )
