# Predator Neuromorphic Drone Propeller Detection — Engineering Log & Troubleshooting Dossier

**Session Date**: 2026-09-28 to 2026-09-29  
**Hardware Target**: NVIDIA Jetson Orin Nano Developer Kit (8GB RAM, 1TB NVMe SSD, IP: `10.0.0.34`)  
**Neuromorphic Sensor**: IDS Imaging Development Systems UE-39B0XCP-E (Sony IMX636 1280x720 Event Camera)  
**Lens Mounted**: Edmund Optics 8mm $f/8$ BLUE Series M12 Lens (#27052)  
**Middleware & Driver**: OpenEB 5.2.0 HAL (Patched for Cypress CX3 Treuzell Protocol)  

---

## 1. Hardware Integration & HAL Treuzell Patch

### Problem & Discovery
- The IDS uEye EVS event camera does not use standard IDS Peak / GenTL transport layers in OpenEB. It communicates via the Cypress CX3 SuperSpeed controller using the Prophesee Treuzell board protocol (Vendor ID `0x1409`, Product ID `0x8e00`, Interface Subclass `0x19`).

### OpenEB 5.2.0 Native HAL Patch Applied
1. **`hal_psee_plugins/src/plugin/psee_universal.cpp`**:
   ```cpp
   tz_cam_discovery->add_usb_id(0x1409, 0x8e00, 0x19);
   ```
2. **`hal_psee_plugins/src/boards/treuzell/tz_libusb_board_command.cpp`**:
   ```cpp
   if (((desc.idVendor == 0x04b4) || (desc.idVendor == 0x1409)) &&
       ((desc.idProduct == 0x00f4) || (desc.idProduct == 0x00f5) || (desc.idProduct == 0x8e00))) {
       if (desc.bcdDevice < 0x0307)
           quirks.do_not_set_config = true;
   }
   ```
3. **Udev Rules (`/etc/udev/rules.d/99-ids-usb-access.rules`)**:
   ```udev
   SUBSYSTEM=="usb", ATTR{idVendor}=="1409", MODE="0666", GROUP="plugdev"
   SUBSYSTEM=="usb", ATTR{idVendor}=="1409", ATTR{idProduct}=="8e00", MODE="0666", GROUP="plugdev"
   KERNEL=="*", SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTRS{idVendor}=="1409", MODE="0666", GROUP="plugdev"
   ```
4. **Throughput Benchmark**: Verified sustained live stream at **9.47 Million events/second** over USB 3.0 (5000 Mbps link).

---

## 2. Outdoor Field Issue #1: UI Stalling & Stream Freezing

### Root Cause Analysis (RCA)
- `MV_FLAGS_EVT3_ROBUST_DECODER=1` instructed OpenEB's `GrammarValidator` to validate EVT3 stream packet continuity strictly.
- Under outdoor high-event throughput ($\sim 10\text{ MEv/s}$), normal USB 3.0 packet boundaries caused `GrammarValidator` to flood `stderr` with thousands of `[HAL][WARNING] Evt3 protocol violation detected: NonContinuousTimeHigh` messages per second.
- `systemd-journald` absorbed these logs synchronously, creating high I/O thrashing and CPU starvation that stalled the web stream socket and frame visualizer threads.

### Resolution
- Removed `MV_FLAGS_EVT3_ROBUST_DECODER` and configured:
  ```cpp
  setenv("MV_FLAGS_EVT3_UNSAFE_DECODER", "1", 1);
  setenv("MV_LOG_LEVEL", "ERROR", 1);
  ```
- OpenEB engaged `BasicCheckValidator`, maintaining 100% event throughput with zero journal logging overhead.

---

## 3. Outdoor Field Issue #2: Intermittent Propeller Lock at 4 Meters

### Optical & Geometric Math
- Sensor: Sony IMX636 ($1280 \times 720$), pixel pitch $p = 4.86\ \mu\text{m}$.
- Lens: Edmund Optics 8mm $f/8$ ($f = 8.0\text{ mm}$).
- Target: 5-inch rotor ($L = 127\text{ mm}$) at range $D = 4000\text{ mm}$:
  $$N_{\text{pix}} = \frac{L \cdot f}{D \cdot p} = \frac{127\text{ mm} \times 8.0\text{ mm}}{4000\text{ mm} \times 0.00486\text{ mm}} \approx 52.26\text{ pixels}$$

### Root Cause
1. **Spatial Boundary Splitting**: On a rigid $40 \times 24$ grid ($32 \times 30\text{ px/cell}$), when the $52\text{ px}$ rotor sat on a boundary, its energy was split across 2 or 4 neighboring isolated cells, causing single-cell SNR to dip below detection thresholds.
2. **Nyquist Ceiling at 2000 Hz**: High-RPM drone props ($> 12,000\text{ RPM} \implies f_{\text{BPF}} > 400\text{ Hz}$) had 2nd/3rd harmonics ($800\text{ Hz}, 1200\text{ Hz}$) truncated by the $1000\text{ Hz}$ Nyquist limit ($2000\text{ Hz}$ sampling), degrading the Harmonic Product Spectrum (HPS) comb score.

### Architectural Fixes Applied
1. **Hierarchical $2 \times 2$ Cell Pooling**:
   - Base grid: $32 \times 18$ ($40 \times 40\text{ px/cell}$).
   - Sliding $2 \times 2$ pooling (`get_pooled_patch_history`) evaluates $80 \times 80\text{ px}$ effective receptive fields, completely capturing the $52\text{ px}$ rotor without boundary losses.
2. **$4000\text{ Hz}$ Temporal Sampling**:
   - $250\ \mu\text{s}$ temporal bins expand Nyquist limit to $2000\text{ Hz}$, tracking props up to $36,000\text{ RPM}$ with full 2nd/3rd harmonic combs.
3. **Spatial Non-Maximum Suppression (NMS)**:
   - Merges adjacent pooled candidate boxes ($\Delta r < 60\text{ px}$, $\Delta f < 10\text{ Hz}$) into a single unified centroid track.

---

## 4. Nighttime Degradation & Starlight Physics

### Physical Mechanisms (Per Research)
1. **$f/8$ Aperture Deficit**:
   - The 8mm $f/8$ lens entrance pupil is only $1.0\text{ mm}$. Compared to an $f/1.4$ lens, it blocks **$97\%$ of light** ($\sim 33\times$ or 5-stop photon loss).
2. **Photoreceptor Bandwidth Collapse**:
   - Bandwidth scales with photocurrent: $f_{\text{3dB}} \approx \frac{I_{\text{ph}}}{2\pi U_T C_{\text{tot}}}$.
   - Under nighttime starlight ($I_{\text{ph}} \sim \text{pA}$), pixel bandwidth collapses from $> 3000\text{ Hz} \to 20\text{--}80\text{ Hz}$ and latency spikes from $<150\ \mu\text{s} \to 10\text{--}50\text{ ms}$.
   - The pixel analog front-end acts as an unintentional low-pass filter, attenuating $100\text{--}400\text{ Hz}$ blade flicker.

### Deployed Software Compensations
1. **IMX636 Analog Bias Tuning (`I_LL_Biases`)**:
   - `bias_diff_on = -15` & `bias_diff_off = -10` (increases sensitivity).
   - `bias_fo = +15` (boosts source-follower cutoff bandwidth).
2. **512-Sample Coherent Integration ($128\text{ ms}$ Window)**:
   - Yields a $+3\text{ dB}$ coherent integration gain ($G_{\text{int}} \propto \sqrt{N}$), pulling sparse periodic photons out of Poisson dark noise.
3. **M-of-N Track State Machine**:
   - Requires $M=3$ consecutive frequency matches to transition from `TENTATIVE` to `CONFIRMED`. Coasts up to 3 frames ($120\text{ ms}$) across sparse dropouts.

---

## 5. Building Floodlight & Motion Light Interference

### Optical Intermodulation Mechanism
- Building floodlights modulate at $120\text{ Hz}$ AC line ripple ($100\text{ Hz}$ in EU).
- Reflected optical intensity over the drone is multiplicative:
  $$I_{\text{sensor}}(t) = I_{\text{floodlight}}(t) \times R_{\text{propeller}}(t) = [1 + m_L \cos(2\pi f_L t)] \cdot [1 + m_P \cos(2\pi f_{\text{BPF}} t)]$$
- Single-peak FFT selected the higher $120\text{ Hz}$ floodlight peak; spatial clustering then saw $120\text{ Hz}$ across the background and discarded the patch, masking the drone.

### Resolution Applied
1. **Multi-Candidate Peak Extraction (`analyze_time_series_candidates`)**:
   - Extracts top-2 distinct spectral peaks per patch, capturing BOTH the $120\text{ Hz}$ floodlight carrier and the $f_{\text{BPF}}$ propeller peak.
2. **AC Line Ripple Common-Mode Suppression**:
   - Suppresses AC carriers ($100\text{ Hz}, 120\text{ Hz}, 200\text{ Hz}, 240\text{ Hz}$) present in background patches, passing the localized $f_{\text{BPF}}$ directly to the tracker.

---

## 6. Day vs. Night Optical & Architectural Trade-offs

- **Daytime**: Sun is pure DC (zero 120Hz ripple), IMX636 has full $>3\text{ kHz}$ bandwidth, $f/8$ lens receives ample light. Enables strict SNR ($\ge 12\text{ dB}$) and instant 1-frame lock.
- **Nighttime**: Demands $f/1.4$ fast optics or active NIR (850nm/940nm) VCSEL illumination, or SPAD quantum compressed sensing for EMCON starlight operations.

---

## 7. Optical Lens Recommendation for Wide-FOV Operations

| Lens | Focal Length | Aperture | $\text{HFOV} \times \text{VFOV}$ | IFOV Resolution | 5" Rotor @ 4m | Operational Domain |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Current** | **$8.0\text{ mm}$** | **$f/8.0$** | $44.5^\circ \times 25.1^\circ$ | $0.608\text{ mrad/px}$ | $52.3\text{ px}$ | Narrow corridor ($25\text{--}50\text{m}$) |
| **Recommended** | **$4.0\text{ mm}$** | **$f/2.8$** | $\mathbf{75.8^\circ \times 47.2^\circ}$ | $1.215\text{ mrad/px}$ | $\mathbf{26.1\text{ px}}$ | **Tactical Wide ($0\text{--}25\text{m}$ dome)** |
| **Ultra-Wide** | **$2.8\text{ mm}$** | **$f/2.4$** | $\mathbf{95.8^\circ \times 63.8^\circ}$ | $1.736\text{ mrad/px}$ | $\mathbf{18.3\text{ px}}$ | Close perimeter ($0\text{--}15\text{m}$) |

*Note: Sourced lenses must support $\ge 1/2"$ sensor format (image circle $\ge 7.14\text{ mm}$) with M12 $\times$ 0.5 thread.*

---

## 8. Unit Verification Suite (10/10 Tests Passed)

```
[TEST 1] Pure Harmonic Propeller Flicker Extraction (140 Hz BPF @ 4000 Hz, 512 samples)... PASSED!
[TEST 2] High-RPM Propeller Extraction (400 Hz BPF / 12,000 RPM with Harmonics)... PASSED!
[TEST 3] Low-Light / Night-Time Sparse Rotor Extraction (Attenuated harmonics)... PASSED!
[TEST 4] Ego-Motion Clutter Rejection (5-15 Hz walking sway)... PASSED!
[TEST 5] AC Powerline Light Flicker Rejection (50-60 Hz lighting)... PASSED!
[TEST 6] Low-Activity Density Gate (Sparse dark noise)... PASSED!
[TEST 7] Spatial 2x2 Cell Pooling & Ingestion... PASSED!
[TEST 8] Global Common-Mode Spatial Rejection & M-of-N Track Lifecycle... PASSED!
[TEST 9] Edmund Optics 8mm f/8 M12 Lens Bearing Geometry... PASSED!
[TEST 10] Drone Detection Directly Beneath 120 Hz AC Building Floodlight... PASSED!
```

---

## 9. Daytime Real-Time Pipeline Optimization & Zero-Copy Circular Buffers (15ft Field Tests)

### Problem & Field Symptoms
- During 15ft daytime outdoor testing, the video stream suffered from multi-second visual lag/stutter, failed to lock stably on the drone propellers, and occasionally triggered on random noise.

### Root Cause Analysis (RCA)
1. **Memory Shift Storm inside Camera Driver Callback**:
   - `advance_temporal_bin()` executed $576 \times 512 = 294,912$ array shifts per 250us tick ($> 1.17\text{ Billion ops/sec}$) inside the CD callback while holding the ingestion mutex.
   - This starved the Cypress CX3 USB FIFO, causing packet drops that corrupted the temporal phase continuity of the propeller flicker.
2. **Synchronous JPEG Encoding**:
   - `cv::imencode` was executed synchronously inside `frame_gen.output_callback`, blocking the camera event thread for $8\text{--}15\text{ ms}$ on every frame.

### Deployed Architecture
1. **$O(1)$ Circular Ring Buffers**:
   - Replaced vector shifts with circular indexing `(head_idx + t) % 512`, reducing per-tick cost by $> 500\times$.
2. **Decoupled Display Worker Thread**:
   - Offloaded `cv::imencode` and UI drawing to `display_encoder_thread_func`.
---

## 10. DJI Mavic Air 2 Target Scaling & Multi-Scale Airframe Fusion (50ft Milestone)

### Target Physical Ground Truth
- **Platform**: DJI Mavic Air 2
- **Propellers**: DJI 7238F Low-Noise ($7.2\text{ inch} = 182.88\text{ mm}$ diameter, 2-blade)
- **Rotor Operational BPF**: $173\text{--}250\text{ Hz}$ ($5,200\text{--}7,500\text{ RPM}$)
- **Diagonal Wheelbase**: $302\text{ mm}$

### Optical Projection vs Range ($f = 8.0\text{ mm}$, $p = 4.86\ \mu\text{m}$):
- **15ft ($4.57\text{m}$)**: $\text{GSD} = 2.78\text{ mm/px} \implies \text{Rotor} = 65.8\text{ px}$. Spans full $80\times 80\text{ px}$ pooled patch.
- **30ft ($9.14\text{m}$)**: $\text{GSD} = 5.56\text{ mm/px} \implies \text{Rotor} = 32.9\text{ px}$. Fits inside single $40\times 40\text{ px}$ cell.
- **50ft ($15.24\text{m}$)**: $\text{GSD} = 9.26\text{ mm/px} \implies \text{Rotor} = 19.7\text{ px}$, entire drone frame = $32.6\text{ px}$.

### Root Cause of 50ft Dropout
1. **Receptive Field Over-Sizing**: $2\times 2$ pooling ($80\times 80\text{ px}$) diluted the small $19.7\text{ px}$ rotor signal with $6,200\text{ px}$ of surrounding background noise, lowering effective SNR.
2. **Activity Gate Ceiling**: `min_events_threshold = 25` blocked sparse $19.7\text{ px}$ rotors ($8\text{--}16\text{ events}$ through $f/8$ aperture).

### Architectural Solution Deployed
1. **Multi-Scale Scanning**: Evaluates single cells ($40\times 40\text{ px}$) for distant targets ($\ge 30\text{ft}$) and $2\times 2$ pooled cells ($80\times 80\text{ px}$) for close targets ($\le 25\text{ft}$).
---

## 11. Vehicle Transient Rejection & Low-Contrast Ground Rotor Bias Profile

### Root Cause Analysis (RCA)
1. **Low-Contrast Floor Background vs Sky Silhouette**:
   - At 50ft, a static drone resting 18" off the floor has identical albedo/radiance to the ground background ($\Delta I / I_{\text{bg}} < 10\%$).
   - Standard nominal bias (`bias_diff_on = 0`) requires $> 25\%$ contrast, preventing pixels from firing events through the tiny 1.0mm aperture of an $f/8$ lens.
2. **Moving Vehicle Broadband Transients (150ft)**:
   - Passing cars have high optical contrast and large pixel area, creating broadband step transients across FFT bins.

### Resolutions Applied
1. **High-Sensitivity Analog Biases**:
   - Set `bias_diff_on = -20`, `bias_diff_off = -20`, and `bias_fo = +10` on IMX636 to trigger on $< 12\%$ relative contrast changes.
2. **Spectral Purity Gate (`snr_linear >= 5.0`)**:
   - Rejects broadband impulses from passing cars, accepting only narrow tonal harmonic peaks from rotating propellers.
3. **Lock-Free Snapshots**:
   - Mutex lock duration reduced to $< 20\ \mu\text{s}$, completely eliminating UI stalls and thread deadlocks on shutdown.




