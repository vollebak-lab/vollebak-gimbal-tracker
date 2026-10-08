from __future__ import annotations

import numpy as np
import pytest

from vollebak_gimbal.detectors.person_model import (
    PersonModelDetector,
    compute_anatomical_com,
    decode_yolo_output,
)


def test_compute_anatomical_com_torso_chest():
    x, y, w, h = 100.0, 50.0, 80.0, 200.0
    com = compute_anatomical_com(x, y, w, h, mode="torso_chest", torso_y_ratio=0.38)
    expected_x = 100.0 + 40.0  # 140.0
    expected_y = 50.0 + 200.0 * 0.38  # 126.0
    assert pytest.approx(com[0]) == expected_x
    assert pytest.approx(com[1]) == expected_y


def test_compute_anatomical_com_bbox_center():
    x, y, w, h = 100.0, 50.0, 80.0, 200.0
    com = compute_anatomical_com(x, y, w, h, mode="bbox_center")
    assert pytest.approx(com[0]) == 140.0
    assert pytest.approx(com[1]) == 150.0


def test_compute_anatomical_com_pose_four_points():
    # 17 keypoints layout. Keypoints: [x, y, conf]
    kpts = [(0.0, 0.0, 0.0)] * 17
    # Shoulders
    kpts[5] = (120.0, 80.0, 0.95)   # left shoulder
    kpts[6] = (160.0, 80.0, 0.95)   # right shoulder
    # Hips
    kpts[11] = (130.0, 160.0, 0.90) # left hip
    kpts[12] = (150.0, 160.0, 0.90) # right hip

    com = compute_anatomical_com(100.0, 50.0, 80.0, 200.0, keypoints=kpts, mode="pose")
    expected_x = (120.0 + 160.0 + 130.0 + 150.0) / 4.0  # 140.0
    expected_y = (80.0 + 80.0 + 160.0 + 160.0) / 4.0    # 120.0
    assert pytest.approx(com[0]) == expected_x
    assert pytest.approx(com[1]) == expected_y


def test_compute_anatomical_com_pose_shoulders_only():
    kpts = [(0.0, 0.0, 0.0)] * 17
    kpts[5] = (120.0, 80.0, 0.95)
    kpts[6] = (160.0, 80.0, 0.95)
    # Hips occluded (conf < 0.3)
    kpts[11] = (130.0, 160.0, 0.1)
    kpts[12] = (150.0, 160.0, 0.1)

    h = 200.0
    com = compute_anatomical_com(100.0, 50.0, 80.0, h, keypoints=kpts, mode="pose")
    expected_x = 140.0
    expected_y = 80.0 + 0.15 * 200.0  # 110.0
    assert pytest.approx(com[0]) == expected_x
    assert pytest.approx(com[1]) == expected_y


def test_decode_yolo_output_standard_detection():
    # Construct synthetic YOLOv8 detection output: shape (1, 84, 10)
    # 84 channels: cx, cy, w, h, class0 (person), class1, ... class79
    raw = np.zeros((1, 84, 10), dtype=np.float32)
    # Candidate 0: Person in center
    # Network input 640x640, image 1280x720 (scale_x = 2.0, scale_y = 1.125)
    raw[0, 0, 0] = 320.0  # cx
    raw[0, 1, 0] = 320.0  # cy
    raw[0, 2, 0] = 100.0  # w
    raw[0, 3, 0] = 200.0  # h
    raw[0, 4, 0] = 0.88   # class 0 (person) conf

    # Candidate 1: Low confidence person (should be filtered out)
    raw[0, 0, 1] = 100.0
    raw[0, 1, 1] = 100.0
    raw[0, 2, 1] = 50.0
    raw[0, 3, 1] = 50.0
    raw[0, 4, 1] = 0.20   # < conf_thresh

    # Candidate 2: High confidence car (class 2) with low person score
    raw[0, 0, 2] = 200.0
    raw[0, 1, 2] = 200.0
    raw[0, 2, 2] = 80.0
    raw[0, 3, 2] = 80.0
    raw[0, 4, 2] = 0.05   # person = 0.05
    raw[0, 6, 2] = 0.95   # car = 0.95

    detections = decode_yolo_output(
        raw,
        orig_shape=(720, 1280),
        input_shape=(640, 640),
        conf_thresh=0.40,
        nms_thresh=0.45,
        person_class_id=0,
        mode="torso_chest",
        torso_y_ratio=0.38,
    )

    assert len(detections) == 1
    det = detections[0]
    assert det.label == "person"
    assert pytest.approx(det.confidence) == 0.88

    # Scale factors: scale_x = 1280/640 = 2.0, scale_y = 720/640 = 1.125
    expected_w = 100.0 * 2.0  # 200.0
    expected_h = 200.0 * 1.125 # 225.0
    expected_x = (320.0 - 50.0) * 2.0  # 540.0
    expected_y = (320.0 - 100.0) * 1.125 # 247.5
    assert pytest.approx(det.x) == expected_x
    assert pytest.approx(det.y) == expected_y
    assert pytest.approx(det.width) == expected_w
    assert pytest.approx(det.height) == expected_h

    # Anatomical CoM check:
    expected_com_x = expected_x + expected_w * 0.50
    expected_com_y = expected_y + expected_h * 0.38
    assert pytest.approx(det.center[0]) == expected_com_x
    assert pytest.approx(det.center[1]) == expected_com_y


def test_decode_yolo_output_pose():
    # Construct synthetic YOLOv8-pose output: shape (1, 56, 5)
    # 56 channels: cx, cy, w, h, box_score, 17 * (kx, ky, kc)
    raw = np.zeros((1, 56, 5), dtype=np.float32)
    raw[0, 0, 0] = 300.0 # cx
    raw[0, 1, 0] = 300.0 # cy
    raw[0, 2, 0] = 100.0 # w
    raw[0, 3, 0] = 200.0 # h
    raw[0, 4, 0] = 0.92  # box_score

    # Set keypoints in network coordinate space (scale = 1.0 for 640x640 -> 640x640)
    # Left shoulder (index 5)
    raw[0, 5 + 5 * 3, 0] = 280.0
    raw[0, 5 + 5 * 3 + 1, 0] = 240.0
    raw[0, 5 + 5 * 3 + 2, 0] = 0.95
    # Right shoulder (index 6)
    raw[0, 5 + 6 * 3, 0] = 320.0
    raw[0, 5 + 6 * 3 + 1, 0] = 240.0
    raw[0, 5 + 6 * 3 + 2, 0] = 0.95
    # Left hip (index 11)
    raw[0, 5 + 11 * 3, 0] = 290.0
    raw[0, 5 + 11 * 3 + 1, 0] = 330.0
    raw[0, 5 + 11 * 3 + 2, 0] = 0.95
    # Right hip (index 12)
    raw[0, 5 + 12 * 3, 0] = 310.0
    raw[0, 5 + 12 * 3 + 1, 0] = 330.0
    raw[0, 5 + 12 * 3 + 2, 0] = 0.95

    detections = decode_yolo_output(
        raw,
        orig_shape=(640, 640),
        input_shape=(640, 640),
        conf_thresh=0.50,
        mode="pose",
    )

    assert len(detections) == 1
    det = detections[0]
    expected_cx = (280.0 + 320.0 + 290.0 + 310.0) / 4.0 # 300.0
    expected_cy = (240.0 + 240.0 + 330.0 + 330.0) / 4.0 # 285.0
    assert pytest.approx(det.center[0]) == expected_cx
    assert pytest.approx(det.center[1]) == expected_cy


def test_person_model_detector_mock_mode():
    config = {
        "model_path": None,
        "input_width": 320,
        "input_height": 320,
        "confidence_threshold": 0.40,
    }
    detector = PersonModelDetector(config)
    assert detector.active_backend == "mock"
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detector.detect(dummy_frame) == []


def test_person_model_detector_real_onnx_inference():
    from pathlib import Path

    model_path = Path("models/yolov8n.onnx")
    if not model_path.exists():
        pytest.skip("models/yolov8n.onnx not present")

    config = {
        "model_path": str(model_path),
        "backend": "opencv",
        "input_width": 320,
        "input_height": 320,
        "confidence_threshold": 0.25,
    }
    detector = PersonModelDetector(config)
    assert detector.active_backend == "opencv"

    # Test forward pass on real RGB image array
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw simple synthetic figure to exercise non-empty detection pipeline
    frame[100:350, 250:390] = (200, 180, 160)
    detections = detector.detect(frame)
    # The forward pass must succeed without exception and return a list
    assert isinstance(detections, list)
