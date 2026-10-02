# Bart integration audit

## Source reviewed

- Repository: `vollebak-lab/predator`
- Integrated main-branch commit: `ea46468`
- Main-branch history reviewed: four Bart commits through 2026-09-30
- Imported tracked files: 104
- Python verification: 48 upstream tests passed after installing SciPy
- Native verification: OpenEB 5.2.0 was built under WSL with the IDS USB identifiers from Bart's integration notes. The camera test, live viewer, CPU flicker detector, all 11 flicker-DSP tests, and all 6 ego-motion tests ran successfully against the connected IMX636. A sensor-side 10 MEv/s cap makes the full detector stable through WSL USB/IP; native USB remains preferred for deployment.

### Commit history assessment

- `e8c4c34` established the original five-layer Python pipeline and its 48-test suite.
- `b197dc5` is the substantive expansion: 91 files changed and roughly 37,500 lines added, including the C++/CUDA event pipeline, Rust workspace, research corpus, additional fusion/tracking modules, and revised hardware architecture.
- `3239ec2` recorded the repository release in the engineering log; it did not introduce another runtime subsystem.
- `ea46468` added the final architecture note formally superseding SpMiniUNet with the frequency-domain DSP core.

Every tracked file from the reviewed commit was copied into this repository. The Pi application in `src/vollebak_gimbal` remains the executable integration layer because it already supports the connected Logitech camera, fixed-camera calibration, the Waveshare serial protocol, and a hardware-free mock driver.

The two upstream files replaced by integration-specific versions are preserved verbatim as `docs/upstream/bart-README.md` and `config/predator_system.upstream.yaml`.

## Findings by component

### Layer 1: neuromorphic sensing

Bart's newest implementation makes frequency-domain DSP the authoritative propeller detector. `ev_ingestion_cpp/flicker_dsp.hpp` uses a 4 kHz analysis rate, 512-sample windows, FFT/harmonic-product-spectrum scoring, spatial patch clustering, common AC-carrier suppression, and M-of-N track persistence. The C++ daemon combines that path with an OpenEB event stream, gyro motion compensation, and optional TensorRT suppression.

This path is designed for a Sony IMX636/DVX-class event camera, not for a conventional Logitech frame camera. The connected IDS UE-39B0XCP was identified as a 1280x720 IMX636 and verified with OpenEB. The dashboard displays its real MJPEG event surface and full detector telemetry alongside the Logitech feed. Under WSL, a 10 MEv/s sensor-side ERC cap avoids overwhelming USB/IP. BPF/RPM/SNR remain empty when the detector has no qualifying target instead of being fabricated.

### Layer 2: radar

The simulated radar backend and micro-Doppler classification logic are implemented and covered by Python tests. The real Uhnder S80 backend is still a stub and no radar is connected here. The Pi GUI reports L2 as `NOT CONNECTED`; radar simulation is off by default so camera results cannot be mistaken for fused radar observations.

### Layer 3: tracking and fusion

The repository includes event aggregation, motion compensation, IMM tracking, Hungarian/JPDA association, hypothesis management, state-machine logic, safety-zone logic, classification, and cooperative tracking. These modules account for much of the 48-test Python suite and all those tests pass locally.

The current Pi has one conventional camera and no range sensor, so the live adapter exposes a single calibrated bearing/elevation track. It does not claim range, micro-Doppler, multi-sensor covariance, or radar confirmation. Bart's fuller fusion modules remain available for a future sensor-equipped compute node.

### Rust workspace and gimbal controller

The Rust workspace contains shared messages, an orchestrator, mast stabilization, parallax compensation, and an engagement-oriented state machine. Its gimbal binary has useful PID and IMU feed-forward primitives, but `main.rs` explicitly leaves Zenoh subscription, the 200 Hz loop, status publication, and the hardware servo interface as TODOs. It is not a working Waveshare driver.

The existing Python Waveshare driver is therefore the correct hardware path for tomorrow's board. It sends the General Driver for Robots JSON command at 115200 baud and is protected by mechanical limits, update-rate limits, step limits, smoothing, and a mock-by-default configuration.

### C++ web HUD

Bart's event-camera daemons embed live MJPEG/status pages with event rate, inference suppression, ego-motion, flicker, RPM, SNR, and track telemetry. Those concepts were brought into the main Pi dashboard as a four-layer readiness panel. Unsupported values remain `N/A` or `NOT CONNECTED`; live RGB tracking and gimbal state come from the Pi process.

## Runtime mapping

```text
Logitech USB camera
        |
        v
OpenCV motion/color/person detector              L1 ACTIVE (RGB fallback)
        |
        v
largest target + fixed-camera calibration
        |
        v
bearing/elevation track                          L3 BEARING-ONLY
        |
        v
smoothing + deadband + range/step/rate limits
        |
        +--> mock driver --> browser digital twin
        |
        `--> Waveshare serial driver (after bring-up) L4 POINTING ONLY

IDS IMX636 event surface + rate telemetry        L1 LIVE (selectable)
Event camera FFT/HPS detector                    L1 LIVE (CPU / WSL)
Uhnder radar                                     L2 NOT CONNECTED
Directed-energy/engagement path                  HARD DISABLED
```

## Deliberate integration boundaries

- The Pi runtime imports none of Bart's engagement modules.
- No HTTP endpoint can enable engagement or issue a non-gimbal actuator command.
- `config/predator_system.yaml` fixes `engagement_enabled: false` and `POINTING_ONLY` capability.
- The browser and `/api/health` display the observation-only state explicitly.
- Bart's engagement-oriented files are retained as upstream provenance, not an executable Pi path.

## Remaining hardware work

1. Assemble and mechanically center the Waveshare module.
2. Identify its Pi serial device and cautiously verify servo direction/limits.
3. Create a real fixed-camera calibration for the final mount and expected target depth.
4. Move the verified event-camera build to the Pi/native Linux host so the same live FFT/HPS service runs without the USB/IP bridge; the GUI telemetry adapter and stream proxy are already integrated.
5. If radar is later added, replace the S80 stub with its supported SDK transport and validate timestamps/extrinsics before claiming sensor fusion.
