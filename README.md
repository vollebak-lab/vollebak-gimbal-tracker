# Predator Observation Console

This repository combines Bart's complete [Vollebak Predator](https://github.com/vollebak-lab/predator) source tree at commit `032be3d` with a working Raspberry Pi 5 camera-to-gimbal application.

The runnable Pi profile is deliberately scoped to passive observation, target tracking, and two-axis pointing. The regular-camera path uses OpenCV MOG2 background subtraction to find moving regions, rejects small contours, selects the largest remaining moving region, and maps its center through the saved camera-to-gimbal calibration. A rate-limited controller smooths that pan/tilt demand before the Waveshare driver sends serial commands. This is motion following, not object-identity recognition: any sufficiently large moving object can become the target.

The GUI also exposes the readiness of Bart's layered stack without pretending disconnected hardware is live:

- L1: the Logitech RGB camera and connected IDS UE-39B0XCP/Sony IMX636 event camera are selectable live feeds. The verified CPU FFT/HPS detector publishes event rate, tracks, bearing, BPF, RPM, and SNR. Bart's latest source additionally includes corrected Nicla IMU scaling, active ego-warp, foliage/standoff co-gating, hover-lock recovery, CUDA processing, and a TensorRT FP16 SpectralCombNet classifier. The dashboard preserves confirmed targets, tentative tracks, ROI diagnostics, and accelerator status when the detector exposes them.
- L2: Uhnder radar is shown as not connected; its real backend in Bart's current tree is a stub.
- L3: the Pi profile provides one bearing-only visual track; Bart's full IMM/JPDA fusion source remains in the repository.
- L4: mock or Waveshare gimbal pointing only. Engagement is hard-disabled and no engagement command endpoint exists.

See [Bart integration audit](docs/bart-integration-audit.md) for the component-by-component review.

## One-click handoff

For the prepared Windows laptop and Raspberry Pi pair, daily operation is now:

1. Connect and power the hardware.
2. Double-click the **Vollebak Gimbal Dashboard** desktop shortcut. Install it once with `INSTALL_DASHBOARD_SHORTCUT.cmd` if needed.
3. Confirm camera health in the browser, clear the gimbal area, and press **START TRACKING**.
4. Double-click `STOP_SYSTEM.cmd` when finished.

The launcher restores the Logitech bridge, WSL IMX636 detector, USB/IP attachment, both SSH tunnels, Pi dashboard, and browser automatically. A background hot-plug watchdog scans for the IMX636 every three seconds, attaches it to WSL, and starts or restarts Bart's detector, so the event camera may be connected after the launcher is already running. It always starts tracking paused and homes the gimbal. Run `SETUP_HANDOFF.cmd` once as Administrator when provisioning another operator or laptop; Windows must share the USB device once before automatic attachment can work. See [system handoff](docs/HANDOFF.md) for prerequisites and troubleshooting.

## Run it now

```powershell
cd C:\Users\gutie\Downloads\vollebak-gimbal-tracker
.\.venv\Scripts\Activate.ps1
gimbal-tracker ui -c config/dev.yaml
```

Open <http://127.0.0.1:8080>. The Windows bridge selects the DirectShow device named `MX Brio`; only if named-device enumeration is unavailable does it probe indexes for the fastest camera. This prevents the laptop webcam from silently replacing the external tracking camera when USB indexes change. If no camera is available, the app falls back to a synthetic target and probes again every five seconds.

The development dashboard also consumes Bart's event-camera service at <http://127.0.0.1:8081>. On this Windows workstation the IDS camera is passed into WSL with `usbipd`; both event services cap the sensor in hardware at 10 MEv/s to stay within the virtual USB bridge's practical limit. Use the feed selector in the dashboard to switch between `LOGITECH RGB` and `IMX636 EVENT`.

Start the already-built CPU detector after attaching USB bus `2-14`:

```powershell
usbipd attach --wsl --busid 2-14
wsl -d Ubuntu -- bash /mnt/c/Users/gutie/Downloads/vollebak-gimbal-tracker/scripts/run_event_detector_wsl.sh
```

The bus ID can change after reconnecting the camera; confirm it with `usbipd list`. `run_event_viewer_wsl.sh` is a lower-CPU raw-event fallback. The newest CUDA/TensorRT detector must be rebuilt on compatible NVIDIA hardware (the upstream CMake profile targets Jetson Orin, CUDA architecture 87); it cannot run natively on the Raspberry Pi 5 GPU. The existing CPU build remains available for this workstation/WSL setup.

Useful checks:

```powershell
gimbal-tracker doctor -c config/dev.yaml
python -m pytest
ruff check src/vollebak_gimbal tests/test_calibration.py tests/test_control.py tests/test_predator_observer.py tests/test_waveshare.py tests/test_web.py
```

## Raspberry Pi 5

Start with 64-bit Raspberry Pi OS Bookworm:

```bash
sudo bash scripts/install_pi.sh
source .venv/bin/activate
cp config/pi.example.yaml config/pi.yaml
gimbal-tracker doctor -c config/pi.yaml
gimbal-tracker ui -c config/pi.yaml --host 0.0.0.0
```

Keep `gimbal.driver: mock` until the Waveshare assembly is mechanically centered, powered from its specified external 12 V supply, and clear to move. The Pi must not power the servos. USB serial is the simplest first connection; expected servo IDs are tilt `1` and pan `2`.

Before enabling `waveshare_serial`:

1. Verify the board's serial port with `gimbal-tracker doctor`.
2. Make only 2–5 degree manual movements with the payload removed.
3. Rigidly mount the stationary camera and gimbal.
4. Run `gimbal-tracker calibrate -c config/pi.yaml` and collect at least six well-spaced points.
5. Keep a physical servo-power disconnect within reach during initial tracking.

Detailed instructions are in [Pi bring-up](docs/pi-bring-up.md) and [system design](docs/system-design.md).

## Repository map

```text
src/vollebak_gimbal/       Runnable Pi camera, calibration, control, UI, and driver
src/layer1_neuromorphic/   Bart's Python event-camera pipeline
src/layer2_radar/          Bart's radar and micro-Doppler layer
src/layer3_fusion/         Bart's trackers, association, and fusion source
src/layer4_engagement/     Upstream research source; not connected to the Pi runtime
src/layer5_cooperative/    Bart's cooperative tracking source
ev_ingestion_cpp/          Jetson/OpenEB/CUDA event-camera engine and its web HUD
crates/                    Bart's Rust messages and orchestrator workspace
config/                    Pi configs plus upstream sensor configs
docs/                      Pi guides and Bart's architecture/research corpus
tests/                     Pi tests plus Bart's 48-test Python suite
```

## Safety boundary

`config/predator_system.yaml` is the active integration policy and fixes the runtime to observation-only. The dashboard health response and telemetry both report `engagement_enabled: false`. The app has no target-linked laser, firing, scanning, or engagement API. Bart's upstream engagement-oriented source is retained for provenance and analysis but is not imported by the Pi application.

An optional bench-only laser alignment diagnostic is disabled by default. When explicitly enabled in the Pi configuration, it provides a single operator-confirmed pulse capped at 100 ms, enforces a two-second cooldown, refuses requests while tracking is active, and forces the GPIO output low at startup, tracking activation, and shutdown. It requires a physical arm switch and a fixed matte beam stop; it is never activated by target detection.

Bart's original system config and README are preserved as `config/predator_system.upstream.yaml` and `docs/upstream/bart-README.md`; neither is loaded by the Pi application.

The checked-in Pi configs use the mock driver. Motion commands are range-, step-, rate-, and deadband-limited, but software limits are not a substitute for physical clearance and a servo-power emergency stop.

## Vendor references

- [Waveshare product wiki](https://www.waveshare.com/wiki/2-Axis_Pan-Tilt_Camera_Module)
- [Waveshare assembly and calibration guide](https://www.waveshare.com/wiki/2-Axis_Pan-Tilt_Camera_Module_Assembly_and_Configuration_Guide)
- [Waveshare host-control reference](https://github.com/waveshareteam/ugv_rpi)
