# Predator: Handoff Plan for the Next Session (written 2026-10-07 02:40 UTC)

Read this file first, then `task.md` (Phase 33) and `brain_updates.md` Sections 39 to 41. Those three files are the
source of truth. Do not redo work marked `[x]`.

## 0. Ground rules (from the user; enforced)

**Code standards**
- No stubs, placeholders or TODOs. Code must be complete and production-grade.
- Runtime stack is C++17/CUDA 12.6 on the Orin. Python is only used for DGX training.
- Platform-agnostic: nothing may be tuned to the Mavic or any single drone.
- Head/ego-motion compensation is out of scope (ego-warp stays off).
- SpectralCombNet is retrained in 33.6, not removed.

**Working method**
- Work one component at a time, depth-first. Verify each on the Orin before starting the next.
- Debugging: state the root cause first. No monkey patches.
- Two-strike rule: after one failed fix, research (web or docs) before trying again.
- Make surgical edits, match the existing style, and preserve unrelated comments.
- Every new core-logic component gets a test, added to the deploy gate when it is fast.

**Records and communication**
- Before marking any task `[x]`, append a dated delta to `brain_updates.md`: commands that worked, measurements, and failure states.
- Push back on non-viable directions; the user wants this.
- Keep responses concise and use file links.

**Security**
- Never store or echo the Orin sudo password. Passwordless sudo exists only for
  `systemctl start|stop|restart|status predator-camera.service`, with no extra args.
- Remind the user to rotate the Orin password.

## 1. Environment

**Repo and hosts**

| Item | Location |
|---|---|
| Repo (Windows) | `c:\Users\snowd\OneDrive\Documents\Vollebak\predator` |
| Runtime code | `ev_ingestion_cpp/` |
| Orin | `orin@10.0.0.34`, ssh alias `orin-nano`, Jetson Orin Nano 8 GB, unified memory (NO separate VRAM) |
| DGX Spark | `vollebak@100.114.14.56:~/predator_spectral` (33.6 only) |

**Hardware:** IDS UE-39B0XCP-E camera (Sony IMX636, 1280x720, serial 4110044079) with a 12 mm f/2.5 M12 lens. The camera
and Orin are in a dark room.

**Deploy:** run `powershell -ExecutionPolicy Bypass -File ev_ingestion_cpp\deploy\deploy.ps1 [-NoRestart]`.
- It copies every *.cpp/.hpp/.cu/.cuh plus CMakeLists.txt.
- It builds, runs the test gate (`test_flicker_dsp`, `test_ego_motion`, `test_cuda_flicker`, `test_evt21_decoder`), installs, restarts, and verifies the build id.
- The last full run was `-NoRestart` at 02:31 UTC; the installed binary differs from the running one only by a header move.

**Orin build directories**
- Source `~/ev_deploy/src`, build `~/ev_deploy/build`.
- Reconfigure: `cmake -S ~/ev_deploy/src -B ~/ev_deploy/build`.
- Build one target: `cmake --build ~/ev_deploy/build --target <t> -j6`.

**Orin helper scripts** (in `~`; repo copies, including `cd_profile.cpp` and `mem_topology.cu`, are in
`ev_ingestion_cpp/tools/orin_scripts/`):

| Script | Purpose |
|---|---|
| `build_dec.sh` | build the decoder test and dependent targets |
| `run_dec_test.sh [--fixture <prefix>]` | run the decoder parity test |
| `run_capture.sh <secs> [prefix]` | stops the service, runs `evt21_capture`, restarts the service on exit |
| `run_bench.sh` | ingest benchmark |
| `monitor_fa.sh <secs>` | false-alarm monitor |
| `check_live.sh` | live service check |

**Service:** check status with `SYSTEMD_PAGER= sudo -n systemctl status predator-camera.service`.

**Fixture:** `~/ev_deploy/fixtures/darkroom_evt21.{evt21raw,cd}` (766 MB / 1.52 GB, 95.1M events, default biases).
`.cd` records are `{u16 x, u16 y, i16 p, i16 reserved, i64 t}`, 16 B each.

**OpenEB 5.2.0 source:** `~/openeb`. HAL headers are under `~/openeb/hal/cpp/include/metavision/hal/`.

**Shell pitfalls**
- PowerShell ssh one-liners with quotes, braces, `$?` or `$(...)` break.
- Write a bash script locally, scp it, run `sed -i 's/\r$//'` on it, then execute it.
- `/usr/bin/time` is not installed; use bash `time`.

## 2. State at handoff

- **Done:** 33.1, 33.2, 33.3, 33.4, 33.5; 33.4b.a; **33.4b.b** (GPU EVT2.1 decoder bit-exact vs OpenEB on 95.1M live events).
- **Uncommitted:** everything since HEAD `2f428d7`. **Ask the user to commit before starting new work.**

**Key numbers**

| Item | Value |
|---|---|
| GPU decode, per 16384-word batch | 48 us CPU enqueue, 70 us GPU; 7.1 ns/event |
| Old SDK path | ~20 us fixed CUDA API cost per ~300-event callback; cannot sustain 10 Mev/s |
| Dark-room service CPU | ~44% of one core |
| Unidentified thread | ~28% of a core, independent of event rate |

## 3. Ordered plan

### Step 1: 33.7a Hot-pixel hardware mask (user approved, do first)

1. **Exact coordinates.** Compute per-pixel event counts from `darkroom_evt21.cd`. The scratch tool `cd_profile.cpp` did this; rewrite it if missing as a small opt-in target `cd_profile`. Record the exact (x, y) of the two hot pixels; so far we only know "row 677" and "column 279".
2. **API.** Read `~/openeb/hal/cpp/include/metavision/hal/facilities/i_roi_pixel_mask.h`, plus the IMX636 implementation under `~/openeb/hal_psee_plugins`. Verify the method names (expected: `set_pixel(x, y, enable)`, `apply_pixels()`, `reset_pixels()`, max pixel count) and the hardware limit; the IMX636 is believed to support 64 masked pixels. Do not guess signatures.
3. **Survey tool** `hot_pixel_survey.cpp` (opt-in target; HAL only):
   - Opens the camera with the service's biases (read them from the service config or code; do not hardcode).
   - Records N seconds and flags pixels with rate > max(absolute floor, K x array median). Make the floor and K CLI parameters, defaulting to 1 kev/s and 1000x.
   - Writes `hot_pixels.txt` (one `x y rate` per line), capped at the hardware limit.
   - Runs a second pass with the mask applied and reports the before/after rate per pixel and in total.
4. **Service.** `ev_flicker_detector` loads `hot_pixels.txt` (path from env/config) at startup and applies it through `I_RoiPixelMask` on the device it already opens.
   - Validate the file: bounds, count limit, malformed lines rejected with a logged error.
   - Expose the masked list in `/pipeline_stats`.
   - If the facility is missing, log a warning and continue (graceful degradation).
5. **Tests.** Unit-test the file parser and validator (add it to the gate). Live acceptance: masked pixels emit 0 events, the total rate drops by at least 90% at default biases, and the service rate at its own biases does not rise.
6. Update `task.md` 33.7a and append a `brain_updates.md` delta.

### Step 2: identify the ~28% rate-independent thread (before 33.4b.e; diagnosis only)

- Locate it with `top -H -p <pid>`, then map the TID to a thread through `/proc/<pid>/task/*/comm` and gdb `thread apply all bt` (or `perf top -t <tid>` if perf is installed).
- Candidates: UI JPEG encoding / frame generation, the 25 Hz analysis loop, journal ROI logging.
- Record the root cause and measurements. Do not fix it until the cause is known.

### Step 3: 33.4b.d Order-correct GPU sieve

- Port the CPU `MicroNeighborhoodPeriodicitySieve` semantics exactly:
  1. Stable radix sort of the decoded batch by micro-tile key (CUB `DeviceRadixSort::SortPairs` on key = tile id, values = event index; stability keeps time order inside a tile).
  2. A per-tile sequential scan (one thread or warp per tile run) with persistent per-tile state in device memory across batches.
  3. Write the hit count into `CudaRawEvent::pad`.
- **Do NOT use a naive parallel SAE sieve** (it is a recorded failure state; order dependence breaks it).
- Parity: GPU sieve retained/hit output must equal the CPU sieve for the inputs of tests 2b-2e plus decoded fixture slices, for any batch split.
- Report GPU us per batch at 16k words.

### Step 4: 33.4b.c Raw tap in `ev_flicker_detector`

- Open the HAL device with `DeviceConfig::set_format("EVT21")`. Keep bias setup through `I_LL_Biases`, plus the step 1 mask.
- Loop on `I_EventsStream::wait_next_buffer()` / `get_latest_raw_data()`, copying into a `cudaHostAllocMapped` ring of fixed slots.
- **Coalesce per ~2 ms or N words** into one decode, then sieve, then accumulate launch set on one stream. Never launch per buffer or per callback.
- Ring overrun policy: count and log dropped buffers, never block the USB thread indefinitely.
- Remove the SDK CD callback / per-event CPU path from the hot path.
- Check whether a CUDA Graph for the fixed launch sequence cuts the ~50 us enqueue cost.
- Acceptance:
  - Replaying the fixture through the full GPU path gives the same FFT ring contents as the CPU path (parity test).
  - Live: the 10-minute dark-room `monitor_fa.sh 600` still shows 0 confirmed targets.
  - Service CPU% is measured before and after.

### Step 5: 33.4b.e UI frame from GPU accumulators

- Replace the per-event `PeriodicFrameGenerationAlgorithm` with a frame built from GPU accumulators.
- Run a live CPU% A/B and close 33.4b.

### Later (in `task.md` order)

- **33.6:** SpectralCombNet v3 on the DGX (Python reference of the runtime preprocessing first, parity <= 1e-3).
- **33.7:** bias sweep, after the user refocuses the lens at infinity.
- **33.8:** recorded corpus + replay harness.
- **Mempalace:** add a drawer (wing `predator`) summarising Sections 40-41.

## 4. Failure states (never repeat)

**Hardware and platform**
- Claiming the Orin has dedicated VRAM, or using managed memory for streaming rings (`concurrentManagedAccess=0`).
- Proposing 25-35 mm lenses for the helmet configuration (FOV and size are not viable; stationary installs only).

**Pipeline performance**
- Assuming SDK decode is the dominant CPU cost (it is the smallest, 12-15 ns/event).
- Issuing CUDA work per SDK callback (~20 us fixed cost each).

**OpenEB usage**
- Constructing OpenEB EVT21/EVT3 decoders without all four sinks (CD, ExtTrigger, ERCCounter, Monitoring); it segfaults on OTHERS words.

**Detection logic**
- Tests compiled with NDEBUG (asserts vanish; the gate uses `-UNDEBUG`).
- CFAR thresholds tuned on known noise.
- Using CombNet as a veto/pruner before v3 passes validation.
- A naive parallel GPU SAE sieve.

**Shell and sudo**
- PowerShell ssh one-liners with complex quoting.
- Passing extra args to the whitelisted sudo commands.
