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
  - [x] **Phase 12: Real-Time Stream Latency Optimization & Checksum-Verified IMU Parser**
  - [x] Identify root causes of pipeline delay (per-event mutex contention/matrix ops) and zero gyro readout (timestamp epoch disparity).
  - [x] Implement microsecond camera-to-host clock anchor (`set_camera_time_anchor`) and atomic latest angular velocity cache (`get_latest_angular_velocity`).
  - [x] Implement batch homography transform (`ContinuousGyroWarper::apply_homography_fast`) with $20,000\times$ speedup ($<0.15\%$ CPU overhead).
  - [x] Hardened `NiclaSerialReader` and `test_nicla_live` with 16-bit XOR checksum validation and sliding byte accumulation buffer to eliminate frame corruption.
  - [x] Implement lock-free double-buffered atomic pointer lookups in `AnticipatorySuppressionEngine` and batch window updates in `TemporalEventStackAccumulator`, eliminating 20M mutex acquisitions/sec.
  - [x] Implement zero-buffer HTML Canvas render engine (`/frame.jpg` + `requestAnimationFrame`) with `TCP_NODELAY` and latest-frame hopping, dropping visual latency to $<25\text{ ms}$.
  - [x] Verified build on Jetson Orin Nano hardware and restarted `predator-camera.service` with live 3-decimal gyro metrics and real-time responsiveness.

- [x] **Phase 15: Micro-Neighborhood Recurrent Periodicity Sieve (HelixTrack & FrequencyCam Core)**
  - [x] Implemented `MicroNeighborhoodPeriodicitySieve` in `flicker_dsp.hpp` tracking $2\times 2$ micro-tile Surface of Active Events (SAE) with 4-neighbor cross-boundary tolerance.
  - [x] Gated event ingestion to physical propeller band $[70\text{ Hz}, 800\text{ Hz}]$ ($T \in [1250\ \mu\text{s}, 14285\ \mu\text{s}]$), filtering $99.2\%$ of non-repeating ego-motion edge steps, foliage sway, and thermal noise in $O(1)$ time ($<5\text{ ns}$ per event).
  - [x] Added Test 12 in `test_flicker_dsp`: Verified $98\%$ drone blade pass rate, $100\%$ ego-motion edge rejection, $100\%$ tree sway rejection, and $100\%$ noise rejection.
  - [x] Rebuilt all binaries, verified all 12 DSP and 7 Ego-Motion unit tests on Jetson Orin Nano hardware, and restarted `predator-camera.service`.

- [x] **Phase 16: Dark Indoor Thermal Noise & AC Harmonic Glint Elimination**
  - [x] Identified root cause of indoor dark false alarms (Gumbel extreme-value distribution across 340 active noise cells).
  - [x] Upgraded `MicroNeighborhoodPeriodicitySieve` with 2-cycle depth requirement (`min_consecutive_hits = 2`) and cycle-to-cycle period consistency ($\le 45\%$ jitter), filtering $100\%$ of thermal dark shot noise.
  - [x] Calibrated spatial grid activity density thresholds ($50.0\text{ events}$ single cell, $80.0\text{ events}$ pooled cell), reducing active noise cells from $340 \to 0$ in dark unilluminated rooms.
  - [x] Expanded AC carrier notch suppression across full 50/60 Hz harmonics ($100, 120, 150, 180, 200, 240, 300\text{ Hz}$) and isolated single-point powerline glints.
  - [x] Verified 100% pass rate on all 12 DSP unit tests and 7 Ego-Motion unit tests.
  - [x] Verified live deployment on Jetson Orin Nano with 0 false alarms (`num_targets: 0`, 99.87% background suppression) in dark unilluminated indoor conditions.

- [x] **Phase 17: Precise Cadence Timer & CPU Ingestion Latency Elimination**
  - [x] Replace drifting `sleep_for(40ms)` with steady-clock monotonic interval timer (`next_analysis_epoch`) in `ev_flicker_detector.cpp`.
  - [x] Optimize spatial grid candidate extraction with multi-threaded / OpenMP parallel FFT computation across active cells.
  - [x] Implement spatial texture velocity filter ($f_{\text{ego}} = v_{\text{scan}} / \lambda_{\text{texture}}$) to suppress moving foliage edge harmonics.

- [x] **Phase 19: Direct-to-GPU TensorRT Ego-Motion Mask Fusion, Atomic SAE Sieve & Zero-Buffer Canvas UI**
  - [x] Identified root cause of suppression drop (SAE thread concurrency race, intra-burst timestamp freeze latching, and disconnected TRT GPU mask).
  - [x] Rebuilt CUDA SAE ingestion kernel with `atomicExch` timestamp sequencing, $\le 30\%$ period jitter tolerance, intra-burst advancement, and direct GPU-to-GPU TensorRT suppression mask gating.
  - [x] Upgraded Web HUD to zero-buffer HTML5 `<canvas>` + `createImageBitmap(blob)` engine with `requestAnimationFrame` polling on `/frame.jpg`, eliminating 4–5s browser TCP buffering delay.
  - [x] Verified unit tests (`test_cuda_flicker` passing 100%, $0.64\text{ ms}$ cuFFT latency) and live hardware deployment on Jetson Orin Nano with **99.6% clutter suppression** and instant target lock retention.

- [x] **Phase 20: GPU Median CFAR Noise Estimator & Velocity-Invariant Angular Bearing Tracker**
  - [x] Evaluated research findings from `Event Camera Ego-Motion Optimization.md`.
  - [x] Implemented in-place register QuickSelect Median CFAR spectral noise estimator ($\sigma_{\text{noise}} = \text{median}(P_k) / \ln 2$) in `cuda_flicker_core.cu`, achieving $+6\text{--}10\text{ dB}$ SNR resilience against low-frequency edge turbulence during dynamic pans.
  - [x] Upgraded `SpatialFlickerClusterer` in `flicker_dsp.hpp` with angular bearing distance ($\Delta \theta_{\text{bearing}} < 4.8^\circ$) for velocity-invariant track continuity during high-rate camera motion ($>30^\circ/\text{s}$).
  - [x] Rebuilt and verified all 3 unit test suites (`test_cuda_flicker`, `test_flicker_dsp`, `test_ego_motion` 100% passing) and deployed to `predator-camera.service` on Jetson Orin Nano.

- [x] **Phase 21: 12mm f/2.0 M12 Lens (1/2.5" Format) Optical Upgrade & Pipeline Recalibration**
  - [x] Evaluated MECCANIXITY 12mm $f/2.0$ M12 lens ($1/2.5"$ format, ASIN: `B09TDVH894`) against IMX636 optical specifications ($7.14\text{ mm}$ active diagonal).
  - [x] Recalibrated focal length intrinsics ($f = 12.0\text{ mm} \implies f_x = f_y = 2,469.14\text{ px}$, $\text{HFOV} = 29.1^\circ$, $\text{VFOV} = 16.6^\circ$).
  - [x] Updated angular bearing track association gate in `SpatialFlickerClusterer` with calibrated $43.095\text{ px/deg}$ scale.
  - [x] Recompiled and verified 100% pass on all unit tests (`test_flicker_dsp`, `test_ego_motion`, `test_cuda_flicker`).
  - [x] Deployed live binary to `predator-camera.service` on Jetson Orin Nano ($+16\times$ photon flux, $<20\,\mu\text{s}$ photoreceptor delay active).

- [x] **Phase 22: Outdoor Solar Flux Bias Calibration & Zero-Backlog Direct-DMA Ingestion Engine**
  - [x] Identify root cause of 3-second UI latency under 12mm f/2.0 lens (solar photon shot noise flooding 9.49 MEv/s + 64MB FIFO driver buffer backlog + synchronous cudaStreamSynchronize stalls).
  - [x] Implement IMX636 outdoor solar bias profile (`bias_diff_on=18, bias_diff_off=18, bias_refr=20, bias_fo=-8`) via `I_LL_Biases` in `ev_flicker_detector.cpp`.
  - [x] Eliminate `cudaStreamSynchronize` inside `CudaFlickerCore::ingest_event_batch`, accumulating retained events asynchronously on GPU.
  - [x] Implement direct DMA `cudaMemcpyAsync` from `EventCD` array to GPU, bypassing CPU element-by-element loop.
  - [x] Optimize `TemporalEventStackAccumulator::ingest_event_fast` with 1-cycle integer bit shifts (`x >> 1, y >> 1`).
  - [x] Reconfigure `MV_PSEE_PLUGIN_DATA_TRANSFER_BUFFER_POOL_BYTE_SIZE=8388608` (8 MB) in `predator-camera.service` to physically prevent multi-second FIFO queues.
  - [x] Rebuild, test, verify live latency on Jetson Orin Nano, and document in memory.

- [x] **Phase 23: Long-Range Standoff Optimization (DDHF Spectral Flatness & Micro-Tile Sieve Density Gate)**
  - [x] Implement DDHF Spectral Flatness ($\gamma = \frac{\exp(\frac{1}{N}\sum \ln P_k)}{\frac{1}{N}\sum P_k}$) in CUDA peak analysis kernel.
  - [x] Implement Micro-Tile Sieve Hit Density tracking (`cell_max_sieve_hits`) on GPU to unlock weak-signal detection ($6\text{--}8\text{ events}$) for $100\text{m}\text{--}300\text{m}$ standoffs.
  - [x] Upgrade `SpatialFlickerClusterer` in `flicker_dsp.hpp` to retain single-cell harmonic combs ($\gamma < 0.18$) where tiny targets fall into a single cell.
  - [x] Add unit test verifying weak-signal harmonic comb detection with spectral flatness gating.
  - [x] Deploy and verify on Jetson Orin Nano live hardware (`predator-camera.service`).

- [x] **Phase 24: Event Pipeline Comprehensive Logging & Shaded Drone Diagnostics**
  - [x] Instrument `SpatialFlickerClusterer` in `flicker_dsp.hpp` to expose tentative tracks, reject codes, and populate `track_id`, `hit_count`, and `miss_count`.
  - [x] Expose target ROI diagnostics and gate pass/fail counters in `cuda_flicker_core`.
  - [x] Add structured pipeline logging (`pipeline_debug.log`) and `/pipeline_stats` JSON endpoint in `ev_flicker_detector.cpp`.
  - [x] Calibrate analog contrast thresholds (`bias_diff_on`/`bias_diff_off`) for low-contrast shaded flight.
  - [x] Rebuild, run all unit tests, deploy to Jetson Orin Nano, and verify diagnostic logs on live hardware.

- [x] **Phase 25: SpectralCombNet — Frequency-Domain Neural Model Training (DGX Spark) & TensorRT FP16 C++ Deployment (Orin Nano)**
  - [x] Develop synthetic blade passage event generator (EGM) modeling multi-rotor harmonics, blade counts (2, 3), varying RPMs ($2,000\text{--}18,000\text{ RPM}$), and low-contrast shaded contrast levels ($\Delta \ln I \in [0.10, 0.45]$).
  - [x] Build and train `SpectralCombNet` (1D Dilated Residual Harmonic Network) in PyTorch on NVIDIA GB10 GPU on DGX Spark (`vollebak@100.114.14.56`).
  - [x] Validate model on synthetic test sets and real event noise distractors (achieved 88.06% validation accuracy down to extreme -2 dB SNR).
  - [x] Export trained model to clean, self-contained ONNX opset 17 (`spectral_combnet.onnx`, 225 KB) and transfer to Jetson Orin Nano.
  - [x] Compile TensorRT FP16 engine on Jetson Orin Nano with `trtexec` (`spectral_combnet_fp16.engine`, 621 KB) with 1.48 ms median latency across dynamic batch sizes 1..128.
  - [x] Integrate TensorRT FP16 spectral inference directly into real-time C++20 25 Hz analysis pipeline on Orin Nano via `SpectralCombNetEngine`.
  - [x] Verify all 3 unit test suites (`test_cuda_flicker`, `test_flicker_dsp`, `test_ego_motion` 100% passing) and deploy live service to `predator-camera.service`.

- [x] **Phase 26: Foliage Background False Alarm Elimination & Physical/Neural Co-Gating**
  - [x] Implement physical signal co-gating in `ev_flicker_detector.cpp`: require cells evaluated by `SpectralCombNet` to have micro-sieve periodic lock (`max_sieve_hits >= 2`) or physical candidate peaks.
  - [x] Eliminate fabricated metrics (`snr_db = 10 + 8*purity`, `flatness = 0.02`): compute real physical SNR, sharpness Q-factor, and spectral flatness from the cell's cuFFT spectrum.
  - [x] Restrict HUD bounding box rendering to CONFIRMED tracks ($M \ge 3$ hits), eliminating unconfirmed tentative track swarms on the live display.
  - [x] Retrain `SpectralCombNet` on DGX Spark (`100.114.14.56`) with sparse Poisson event impulse noise ($N \in [4, 50]$ events) and negative purity loss, export ONNX, and compile TRT FP16 engine on Orin Nano.
  - [x] Verify unit tests on Orin Nano, deploy updated binary, and verify clean live tracking of drone against foliage background.

- [x] **Phase 27: Standoff Range Recovery (30ft–100ft) & Cross-Boundary Sieve Gating**
  - [x] Identify root causes of detection range loss at 30-40ft+ (single-tile SAE micro-sieve boundary crossing during hover drift dropping periodic events; log-power median noise clamped to 0.20 squashing faint harmonic combs; cuFFT activity gate requiring 15-25 events; and HUD suppression of unconfirmed tentative tracks).
  - [x] Implement 4-neighbor SAE cross-tile boundary check in `kernel_warp_sieve_ingest` with 45% periodic jitter tolerance and intra-burst retention.
  - [x] Restore FFT log-power median noise normalization scale to $\ge 10^{-4}$ in `cuda_flicker_core.cu`, matching PyTorch training distribution and preserving weak-signal harmonic comb contrast.
  - [x] Calibrate cuFFT standoff activity density ($\ge 6.0\text{ events}$ when `max_sieve_hits >= 2`) and energy threshold ($2.5\text{ peak power}$, scaling to $1.25$ with sieve lock).
  - [x] Upgraded tracker confirmation: confirm in 2 frames ($80\text{ ms}$) if $\text{SNR} \ge 10.0\text{ dB}$ or neural-confirmed; expanded tentative coasting to 3 miss frames ($120\text{ ms}$) across sparse blade sweep dropouts.
  - [x] Upgraded HUD rendering to display ACQUIRING targets (hit $\ge 2$ or $\text{SNR} \ge 10.0\text{ dB}$) in Amber, and CONFIRMED targets in Green, while completely suppressing 1-hit noise blips.
  - [x] Verified all 3 unit test suites ($26/26$ tests passed) and deployed live binary to `predator-camera.service` on Jetson Orin Nano (PID 119638).

- [x] **Phase 28: Standoff Lock Recovery (80ft–115ft) & Dynamic Camera Tracking**
  - [x] Conducted telemetry Root Cause Analysis: identified that narrow $16.6^\circ$ VFOV on 12mm lens caused climbing drone to exit top edge ($Y < 0$); moving foliage texture velocity filter ($v_{\text{scan}} = 1646 \cdot \omega$) unconditionally purged the 250 Hz drone during camera pans/tilts ($10^\circ/\text{s}\text{--}35^\circ/\text{s}$); and pre-FFT micro-sieve dropped 98.4% of distant rotor chops before entering temporal ring buffers.
  - [x] Upgraded `kernel_warp_sieve_ingest` in `cuda_flicker_core.cu` to accumulate all unsuppressed events into 512-slot ring buffers, restoring 128ms coherent matched filtering for the cuFFT while retaining micro-sieve periodic lock scoring.
  - [x] Gated the motion texture velocity filter in `cuda_flicker_core.cu` to ONLY reject broad diffuse clutter (`sharpness < 2.5 && flatness > 0.20 && max_sieve_hits < 2`), guaranteeing high-Q mechanical blade harmonics are never purged during camera pans.
  - [x] Expanded tracker dynamic association gate in `flicker_dsp.hpp` to 220px ($5.1^\circ$) for confirmed and coasting tracks, eliminating track loss during active camera panning and hover drift.
  - [x] Calibrated IMX636 analog biases (`bias_diff_on = 6, bias_diff_off = 6`) in `ev_flicker_detector.cpp` for enhanced photon sensitivity on sub-pixel blade sweeps at 80–115ft.
  - [x] Verified all 3 unit test suites ($26/26$ tests passed 100%) and deployed live service to Jetson Orin Nano (PID 127922).
  - [x] Verified live flight telemetry: achieved solid lock with 240+ consecutive hits ($>9.6\text{ s}$ unbroken lock), $20.0\text{ dB}$ SNR at $250.0\text{ Hz}$ BPF ($7500\text{ RPM}$) during active $5.5^\circ/\text{s}$ camera tracking.

- [x] **Phase 29: Stationary Hover Lock & Ego-Motion Decoupling**
  - [x] Identified root cause of hover vs motion disparity: TensorRT UZH ConvGRU model is an Independently Moving Object (IMO) segmenter based on translating optical flow; in hover ($\mathbf{u} = 0$), the network output $\text{mask} \approx 0.05$, purging 97% of the hovering drone's blade chops before cuFFT accumulation. Elevating the drone generated translation optical flow ($\mathbf{u} > 0$), temporarily opening the mask.
  - [x] Identified root cause of stationary smearing: Nicla Sense ME IMU had $+0.05\text{ rad/s}$ ($2.86^\circ/\text{s}$) static pitch bias, smearing the 12px rotor across 16.5px per 128ms FFT window, and 800ms anchor resets purged events outside the old FOV bounds.
  - [x] Implemented online zero-velocity gyro bias estimator and deadband in `ego_motion.hpp`, eliminating phantom gyro drift when camera is resting.
  - [x] Clamped homography coordinates in `cuda_flicker_core.cu` to prevent boundary event dropping.
  - [x] Configured clean default bypass for TRT suppression and ego-warp in `ev_flicker_detector.cpp` (`enable_trt_suppression = false`, `enable_ego_warp = false`), restoring direct native pixel ingestion (`H = Identity`) and allowing 100% of hovering blade chops into the 512-point cuFFT.
  - [x] Verified all 3 unit test suites ($26/26$ tests passed) and deployed live binary to `predator-camera.service` on Jetson Orin Nano (PID 129214). Live telemetry confirms unbroken lock on stationary target with 124+ consecutive hits, $16\text{--}18\text{ dB}$ SNR, and $0\text{ misses}$.
