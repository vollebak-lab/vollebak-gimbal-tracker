# System design

## Data flow

```text
stationary USB camera
        |
        v
motion / color / person detector
        |
        v
largest target center (pixel x, y)
        |
        v
calibrated pixel-to-angle homography
        |
        v
smoothing -> deadband -> step/range/rate limits
        |
        v
Waveshare T=133 JSON command over 115200-baud serial
        |
        +---- local dashboard state + MJPEG feed + digital twin
```

The detector and hardware driver are deliberately separate. A stronger detector can replace the baseline OpenCV implementation without touching motion control, and the mock driver allows the full vision pipeline to run without energized hardware.

The dashboard is served directly by the Python process with no cloud or JavaScript build step. It exposes read-only state and MJPEG video plus local control endpoints for pause/resume, home, manual angles, and calibrated click-to-aim. Binding to `127.0.0.1` keeps it on one machine; binding to `0.0.0.0` makes it available on the local network and should only be done on a trusted network.

## Fixed-camera geometry

The 3x3 homography maps image coordinates to absolute pan/tilt angles. It is an empirical approximation, not a full 3D reconstruction. It is a good fit when:

- the camera and gimbal do not move after calibration;
- the working area is approximately planar; or
- subjects stay within a narrow range of distance.

If subjects move through substantial depth, translation between camera and gimbal creates parallax. The next upgrade should be one of:

1. Mount the observing camera coaxially on the gimbal and use image-center visual servoing.
2. Add depth (stereo/depth camera) and calibrate both camera and gimbal in a shared 3D coordinate frame.
3. Define multiple calibrations for known depth zones and choose one from another range sensor.

## Target selection

The initial policy follows the largest detection. This avoids identity management and is deterministic, but it can switch targets. For multiple people, add a detector/tracker with persistent IDs and expose the selected ID through a small web UI or API.

## Failure behavior

- Detection absent briefly: hold the last commanded pose.
- Detection absent through `park_after_s`: walk toward the configured home pose using the normal step limit.
- Invalid/missing calibration: refuse to start.
- Camera read failure or process signal: close serial and exit. Servo power remains a hardware concern.
