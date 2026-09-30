# Task List: Frequency-Domain Propeller Flicker Detection Engine

- [x] **Phase 1: Mathematical DSP Core (C++)**
  - [x] Implement High-Pass Temporal Filter ($f_c = 40\text{ Hz}$) for ego-motion and DC rejection.
  - [x] Implement Temporal Event Rate Profiler & Sliding-Window NDFT/FFT ($50\text{--}100\text{ ms}$ sliding window, microsecond timestamps).
  - [x] Implement Harmonic Product Spectrum (HPS) & Harmonic Comb Peak Detector to extract fundamental Blade Passage Frequency ($f_{\text{BPF}}$) and RPM.
  - [x] Implement Spectral Purity / SNR confidence metric.

- [x] **Phase 2: Live Stream Integration & Pipeline Architecture**
  - [x] Integrate DSP core into real-time OpenEB event callback pipeline.
  - [x] Grid-based / coarse patch spatial partitioning ($40 \times 24$ patch grid covering $1280 \times 720$).
  - [x] Deploy C++ binary (`ev_flicker_detector`) on Jetson Orin Nano with systemd service.

- [x] **Phase 3: Verification & Benchmarking**
  - [x] Automated unit tests (`test_flicker_dsp`) for harmonic event streams ($120\text{ Hz}$ BPF / $3600\text{ RPM}$), ego-motion rejection ($5\text{--}15\text{ Hz}$), and 8mm M12 lens bearing geometry.
  - [x] Live benchmark on Jetson Orin Nano with physical IDS IMX636 camera streaming at $9.45\text{ MEv/s}$.

- [x] **Phase 4: Telemetry & Middleware Integration**
  - [x] Formulate detection alerts (`DetectionAlert` with $f_{\text{BPF}}$, RPM, confidence, bearing Az/El, pixel centroid).
  - [x] Real-time JSON telemetry endpoint (`/flicker_stats`) and interactive HUD at `http://10.0.0.34:8080/`.

- [x] **Phase 5: Outdoor Field Optimization & Stability Hardening**
  - [x] Eliminate `NonContinuousTimeHigh` log storms by removing `MV_FLAGS_EVT3_ROBUST_DECODER` and setting `MV_FLAGS_EVT3_UNSAFE_DECODER=1` with `MV_LOG_LEVEL=ERROR`.
  - [x] Upgrade temporal binning to $4000\text{ Hz}$ ($250\ \mu\text{s}$ bins, $2000\text{ Hz}$ Nyquist) to capture high-RPM harmonics up to $36,000\text{ RPM}$.
  - [x] Implement $2 \times 2$ sliding hierarchical cell pooling on a $32 \times 18$ base grid ($80 \times 80\text{ px}$ effective receptive fields) to prevent boundary splitting for $52\text{ px}$ rotors at 4m+.
  - [x] Implement Non-Maximum Suppression (NMS) in `SpatialFlickerClusterer` to merge overlapping pooled detections.
- [x] **Phase 6: Outdoor Field Verification @ 50ft & False Alarm Suppression**
  - [x] Identify root cause of false alarms (Poisson shot noise periodogram peak extreme-value distribution).
  - [x] Implement wideband noise floor estimation ($40\text{--}1000\text{ Hz}$, $>120\text{ bins}$) for low-variance background estimator.
  - [x] Implement hardened multi-gate pipeline: activity density gate ($60/90\text{ events}$), absolute peak power gate ($P_{\text{peak}} \ge 4.0$), spectral sharpness ($Q \ge 2.5$), SNR ($\ge 10.5\text{ dB}$).
  - [x] Implement robust M-of-N temporal confirmation ($M=3$ consecutive frames, $60\text{ px} / \pm 8\text{ Hz}$ association, border perimeter suppression).
- [x] **Phase 7: Long-Range Verification @ 75ft & 100ft**
  - [x] Geometric & optical scaling analysis for 100ft detection envelope ($N_{\text{rotor}} \approx 6.86\text{ px}$).
  - [x] Calibrate activity thresholds to $25\text{ events}$ (single cell) / $40\text{ events}$ (pooled cell) to capture sparse 100ft blade sweeps.
  - [x] Expand track coasting to 5 frames ($200\text{ ms}$) across sparse blade phase dropouts.
  - [x] Verify continuous positive target lock at 100ft ($30.5\text{m}$) on Jetson Orin Nano hardware with zero false alarms.

- [x] **Phase 8: High-Rate Gyro Stabilization & Homography Warping Core (C++/CUDA)**
  - [x] Implement `ContinuousGyroWarper` with spherical homography rotation matrix $\mathbf{K} \mathbf{R}(t_{\text{ref}}, t_i) \mathbf{K}^{-1}$ and quaternion angular integration.
  - [x] Implement spatial bilinear event unwarping to lock background contrast edges into stabilized coordinates.
  - [x] Implement parametric background velocity rejection / angular rate ego-motion gate.
  - [x] Build automated unit test suite `test_ego_motion` verifying $30^\circ/\text{s}$ rotation stabilization and drone rotor signature preservation.

- [x] **Phase 9: Anticipatory Motion Suppression & TensorRT Integration (C++/TensorRT FP16)**
  - [x] Implement 2-bin temporal stack accumulator from stabilized event stream ($B \times 2 \times H \times W$).
  - [x] Implement UZH RSS 2026 ConvGRU + Attention-based Time Conditioning (ATC) forward flow & dynamic mask inference engine in TensorRT.
  - [x] Implement backward mask flow warping kernel $\tilde{M}_t(\mathbf{x}) = \hat{M}_{t-\Delta t_d}(\mathbf{x} - \boldsymbol{\psi}_t(\mathbf{x}))$.
  - [x] Implement asynchronous binary event gating operator $S_{\tilde{M}_t}(E)$ to filter out background ego-motion events before spatial patch grid ingestion.

- [x] **Phase 10: Live Pipeline Deployment & Jetson Orin Nano Hardware Verification**
  - [x] Integrate Tier 1 Gyro Warper and Tier 2 TensorRT Suppression Engine into `ev_flicker_detector.cpp`.
  - [x] Verify build on Jetson Orin Nano with CUDA 12.6 and TensorRT 10.3.
  - [x] Benchmark end-to-end latency ($14.0\text{ ms}$ TensorRT GPU compute, 71.0 FPS) and MEv/s throughput under simulated and live camera ego-motion.
  - [x] Update Web HUD and `/flicker_stats` JSON telemetry with ego-motion status, angular velocity, and suppressed event metrics.

- [x] **Phase 11: Arduino Nicla Sense ME Live IMU Streaming & Hardware Lock**
  - [x] Develop binary protocol firmware (`nicla_predator_imu.ino`) streaming 200 Hz 32-byte gyro/accel packets over USB CDC serial.
  - [x] Apply coordinate frame transformations on Nicla for rear-mount optical axis alignment ($\omega_x^{\text{cam}} = +\omega_y^{\text{nicla}}$, $\omega_y^{\text{cam}} = +\omega_x^{\text{nicla}}$, $\omega_z^{\text{cam}} = -\omega_z^{\text{nicla}}$).
  - [x] Configure OpenOCD CMSIS-DAP flashing permissions (`/dev/hidraw0`) and flash Nicla Sense ME over USB from Orin Nano.
  - [x] Build and verify live standalone C++ receiver (`test_nicla_live`) on Jetson Orin hardware.
  - [x] Integrate threaded non-blocking `NiclaSerialReader` into `ev_flicker_detector` service with live 200 Hz ingestion into `ContinuousGyroWarper`.
  - [x] Update Web UI and JSON telemetry endpoint with live IMU lock status and angular rate metrics.

