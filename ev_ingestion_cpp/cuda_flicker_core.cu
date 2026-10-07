#include "cuda_flicker_core.cuh"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <metavision/sdk/base/events/event_cd.h>

namespace predator {

// CUDA Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while (0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult res = call; \
        if (res != CUFFT_SUCCESS) { \
            std::cerr << "[cuFFT ERROR] code " << res \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while (0)

// CUDA Constant Memory for Hanning Window
__constant__ float c_hanning_window[512];

/**
 * @brief CUDA Kernel: Unwarp events via Homography H, apply the optional TensorRT suppression mask,
 * and atomically accumulate every in-window event into the spatial temporal ring buffers.
 *
 * Phase 33.4 (data integrity): ring-buffer accumulation is UNCONDITIONAL for every event that
 * lands inside the stabilized sensor area. The periodicity sieve no longer runs here: it is
 * evaluated on the CPU in timestamp order (MicroNeighborhoodPeriodicitySieve) and its per-event
 * hit count arrives in CudaRawEvent::pad. Previously a parallel SAE evaluation saw per-tile
 * events out of order and `return`ed before accumulation, silently dropping FFT samples.
 */
__global__ void kernel_warp_sieve_ingest(
    const CudaRawEvent* __restrict__ events,
    size_t count,
    const float* __restrict__ H,
    const float* __restrict__ suppression_mask,
    float suppression_threshold,
    int width, int height,
    float* __restrict__ ring_buffers,
    float* __restrict__ cell_total_events,
    uint32_t* __restrict__ cell_max_sieve_hits,
    size_t head_idx,
    uint64_t head_bin_start_us,
    uint32_t bin_duration_us) {

    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count) return;

    const CudaRawEvent& ev = events[idx];
    float x = static_cast<float>(ev.x);
    float y = static_cast<float>(ev.y);

    // 1. Fast 3x3 Homography Projection (8 float operations)
    float denom = H[6] * x + H[7] * y + H[8];
    if (fabsf(denom) < 1e-7f) return;

    float inv_denom = 1.0f / denom;
    float stab_x = (H[0] * x + H[1] * y + H[2]) * inv_denom;
    float stab_y = (H[3] * x + H[4] * y + H[5]) * inv_denom;

    // Events warped outside the stabilized field are discarded. (Clamping them, as before,
    // piled them onto border cells and produced persistent border false tracks.)
    if (stab_x < -0.5f || stab_x > static_cast<float>(width) - 0.5f ||
        stab_y < -0.5f || stab_y > static_cast<float>(height) - 0.5f) {
        return;
    }

    // 2. Direct-to-GPU TensorRT Ego-Motion Suppression Gating
    if (suppression_mask != nullptr) {
        int mx = __float2int_rn(stab_x * (640.0f / static_cast<float>(width)));
        int my = __float2int_rn(stab_y * (360.0f / static_cast<float>(height)));
        if (mx >= 0 && mx < 640 && my >= 0 && my < 360) {
            float mask_val = suppression_mask[my * 640 + mx];
            if (mask_val < suppression_threshold) {
                // Suppressed by TensorRT Ego-Motion Forecaster
                return;
            }
        }
    }

    int sx = min(width - 1, max(0, __float2int_rn(stab_x)));
    int sy = min(height - 1, max(0, __float2int_rn(stab_y)));

    // 3. Periodicity sieve verdict computed on the CPU (0 = not periodic, else consecutive hits)
    const uint32_t sieve_hits = static_cast<uint32_t>(static_cast<uint16_t>(ev.pad));
    const bool is_periodic = (sieve_hits > 0);

    // 4. Temporal Ring Buffer Bin Calculation:
    // Map event timestamp ev.t directly to its exact temporal slot relative to head_bin_start_us
    uint64_t ev_t = ev.t;
    size_t event_slot = head_idx;
    bool in_window = true;

    if (ev_t >= head_bin_start_us) {
        event_slot = head_idx;
    } else {
        uint64_t dt_back = head_bin_start_us - ev_t;
        uint64_t bins_back = (dt_back + bin_duration_us - 1) / bin_duration_us;
        if (bins_back >= 512) {
            in_window = false;
        } else {
            event_slot = (head_idx + 512 - (bins_back % 512)) % 512;
        }
    }

    if (in_window) {
        // Accumulate into base spatial grid (32 cols x 18 rows, 40x40 px cells)
        int col = sx / 40;
        int row = sy / 40;
        if (col < 0) col = 0; if (col >= 32) col = 31;
        if (row < 0) row = 0; if (row >= 18) row = 17;

        int cell_idx = row * 32 + col;
        atomicAdd(&ring_buffers[cell_idx * 512 + event_slot], 1.0f);
        atomicAdd(&cell_total_events[cell_idx], 1.0f);
        if (cell_max_sieve_hits != nullptr && is_periodic) {
            atomicMax(&cell_max_sieve_hits[cell_idx], sieve_hits);
        }

        // 5. Also accumulate into 2x2 pooled cells (576..1151)
        for (int pr = max(0, row - 1); pr <= min(17, row); ++pr) {
            for (int pc = max(0, col - 1); pc <= min(31, col); ++pc) {
                int pooled_idx = 576 + (pr * 32 + pc);
                atomicAdd(&ring_buffers[pooled_idx * 512 + event_slot], 1.0f);
                atomicAdd(&cell_total_events[pooled_idx], 1.0f);
                if (cell_max_sieve_hits != nullptr && is_periodic) {
                    atomicMax(&cell_max_sieve_hits[pooled_idx], sieve_hits);
                }
            }
        }
    }
}

/**
 * @brief CUDA Kernel: Zeroes out expiring slots in ring buffers and updates cell totals
 */
__global__ void kernel_advance_temporal_bins(
    float* __restrict__ ring_buffers,
    float* __restrict__ cell_total_events,
    uint32_t* __restrict__ cell_max_sieve_hits,
    size_t old_head_idx,
    size_t steps,
    int num_cells) {

    int cell_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell_idx >= num_cells) return;

    float expired_sum = 0.0f;
    for (size_t s = 0; s < steps; ++s) {
        size_t slot = (old_head_idx + 1 + s) % 512;
        size_t idx = cell_idx * 512 + slot;
        expired_sum += ring_buffers[idx];
        ring_buffers[idx] = 0.0f;
    }

    if (expired_sum > 0.0f) {
        float cur_tot = cell_total_events[cell_idx];
        float new_tot = fmaxf(0.0f, cur_tot - expired_sum);
        cell_total_events[cell_idx] = new_tot;
        if (cell_max_sieve_hits != nullptr && new_tot <= 0.0f) {
            cell_max_sieve_hits[cell_idx] = 0;
        }
    }
}

/**
 * @brief CUDA Kernel: Prepares Hanning-windowed float time series for batched cuFFT across 1152 cells
 */
__global__ void kernel_prepare_fft_window(
    const float* __restrict__ ring_buffers,
    size_t head_idx,
    float* __restrict__ fft_input) {

    int cell_idx = blockIdx.x; // 0..1151
    int t = threadIdx.x;       // 0..511

    // Chronological order: oldest sample is at (head_idx + 1) % 512, newest is at head_idx
    size_t ring_k = (head_idx + 1 + t) % 512;
    float raw_val = ring_buffers[cell_idx * 512 + ring_k];
    fft_input[cell_idx * 512 + t] = raw_val * c_hanning_window[t];
}

/**
 * @brief CUDA Kernel: per-cell spectral detection for all 1152 cells (576 base + 576 pooled).
 *
 * Computes |X_k|^2, optionally exports it for diagnostics / SpectralCombNet, then runs the shared
 * CFAR gate chain (spectral_gate.hpp) so the GPU detector and host diagnostics are identical.
 */
__global__ void kernel_analyze_spectral_peaks(
    const cufftComplex* __restrict__ fft_output,
    float* __restrict__ power_spectrum_out,
    const float* __restrict__ cell_total_events,
    const uint32_t* __restrict__ cell_max_sieve_hits,
    SpectralGateConfig cfg,
    float gyro_speed_deg_s,
    CudaDetectionCandidate* __restrict__ candidates,
    uint32_t* __restrict__ num_candidates,
    uint32_t max_candidates) {

    int cell_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell_idx >= 1152) return;

    bool is_pooled = (cell_idx >= 576);
    int base_cell = is_pooled ? (cell_idx - 576) : cell_idx;

    float total_events = cell_total_events[cell_idx];
    uint32_t max_sieve_hits = (cell_max_sieve_hits != nullptr) ? cell_max_sieve_hits[cell_idx] : 0;

    const cufftComplex* cell_fft = &fft_output[cell_idx * kSpectrumBins];

    // Power spectrum in local memory (always exported: ROI diagnostics and CombNet read it).
    float power[kSpectrumBins];
    #pragma unroll 4
    for (int k = 0; k < kSpectrumBins; ++k) {
        float re = cell_fft[k].x;
        float im = cell_fft[k].y;
        float p = (re * re + im * im);
        power[k] = p;
        if (power_spectrum_out != nullptr) {
            power_spectrum_out[cell_idx * kSpectrumBins + k] = p;
        }
    }

    const SpectralGateResult g = evaluate_spectral_gate(power, total_events, max_sieve_hits,
                                                        gyro_speed_deg_s, cfg);
    if (g.verdict != GateVerdict::Pass) return;

    uint32_t slot = atomicAdd(num_candidates, 1);
    if (slot < max_candidates) {
        CudaDetectionCandidate c;
        c.cell_idx = cell_idx;
        c.is_pooled = is_pooled ? 1 : 0;
        c.patch_col = base_cell % 32;
        c.patch_row = base_cell / 32;
        c.fundamental_bpf_hz = g.fundamental_hz;
        c.peak_snr_db = g.snr_db;
        c.spectral_q_factor = g.sharpness;
        c.peak_power = g.peak_power;
        c.noise_floor = g.mean_noise;
        c.confidence = g.confidence;
        c.total_events = total_events;
        c.spectral_flatness = g.flatness;
        c.max_sieve_hits = max_sieve_hits;
        candidates[slot] = c;
    }
}

// Host Implementation of CudaFlickerCore
CudaFlickerCore::CudaFlickerCore(int sensor_width, int sensor_height,
                                 int grid_cols, int grid_rows,
                                 double sample_rate_hz, size_t history_samples)
    : sensor_width_(sensor_width), sensor_height_(sensor_height),
      grid_cols_(grid_cols), grid_rows_(grid_rows),
      sample_rate_hz_(sample_rate_hz), history_samples_(history_samples),
      bin_duration_us_(static_cast<uint64_t>(1000000.0 / sample_rate_hz)),
      sieve_(sensor_width, sensor_height, 75.0, 1200.0, 2) {

    num_base_cells_ = grid_cols_ * grid_rows_;
    num_total_cells_ = num_base_cells_ * 2; // 576 base + 576 pooled = 1152

    // Default CFAR gate (Phase 33.5). Derived before any allocation: it throws
    // std::invalid_argument on an inconsistent geometry, and nothing must leak if it does.
    set_spectral_gate_config(SpectralGateConfig{});

    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUDA_CHECK(cudaEventCreateWithFlags(&upload_done_, cudaEventDisableTiming));

    // Allocate Host Pinned Memory (staging for truly asynchronous H2D event uploads)
    CUDA_CHECK(cudaMallocHost(&h_event_buffer_, max_events_per_batch_ * sizeof(CudaRawEvent)));
    CUDA_CHECK(cudaMallocHost(&h_candidate_buffer_, max_candidates_ * sizeof(CudaDetectionCandidate)));
    CUDA_CHECK(cudaMallocHost(&h_cell_totals_, num_total_cells_ * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_cell_sieve_hits_, num_total_cells_ * sizeof(uint32_t)));

    // Allocate Device GPU Memory
    CUDA_CHECK(cudaMalloc(&d_events_, max_events_per_batch_ * sizeof(CudaRawEvent)));
    CUDA_CHECK(cudaMalloc(&d_homography_matrix_, 9 * sizeof(float)));

    // Allocate Spatial Ring Buffers (1152 cells x 512 samples)
    CUDA_CHECK(cudaMalloc(&d_ring_buffers_, num_total_cells_ * history_samples_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cell_total_events_, num_total_cells_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cell_max_sieve_hits_, num_total_cells_ * sizeof(uint32_t)));

    // Allocate cuFFT Buffers
    CUDA_CHECK(cudaMalloc(&d_fft_input_, num_total_cells_ * history_samples_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fft_output_, num_total_cells_ * 257 * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_power_spectrum_, num_total_cells_ * 257 * sizeof(float)));

    // Allocate Candidate Output Buffers
    CUDA_CHECK(cudaMalloc(&d_candidates_, max_candidates_ * sizeof(CudaDetectionCandidate)));
    CUDA_CHECK(cudaMalloc(&d_num_candidates_, sizeof(uint32_t)));

    // Create Batched 1D Real-to-Complex cuFFT Plan
    int n_samples = static_cast<int>(history_samples_);
    CUFFT_CHECK(cufftPlanMany(&cufft_plan_, 1, &n_samples,
                              NULL, 1, 512,
                              NULL, 1, 257,
                              CUFFT_R2C, num_total_cells_));
    CUFFT_CHECK(cufftSetStream(cufft_plan_, stream_));

    // Initialize Hanning Window in CUDA Constant Memory
    float hanning[512];
    for (size_t i = 0; i < 512; ++i) {
        hanning[i] = 0.5f * (1.0f - std::cos(2.0 * M_PI * i / 511.0));
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_hanning_window, hanning, 512 * sizeof(float)));

    reset();
    initialized_ = true;
    std::cout << "[CUDA] CudaFlickerCore initialized on Jetson Orin Nano ("
              << num_total_cells_ << " cuFFT channels, CPU-ordered 2x2 periodicity sieve "
              << sieve_.tile_width() << "x" << sieve_.tile_height() << " tiles).\n";
}

CudaFlickerCore::~CudaFlickerCore() {
    if (stream_ != nullptr) {
        cudaStreamSynchronize(stream_);
    }
    if (cufft_plan_ != 0) {
        cufftDestroy(cufft_plan_);
    }
    if (upload_done_ != nullptr) {
        cudaEventDestroy(upload_done_);
    }
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }

    if (h_event_buffer_) cudaFreeHost(h_event_buffer_);
    if (h_candidate_buffer_) cudaFreeHost(h_candidate_buffer_);
    if (h_cell_totals_) cudaFreeHost(h_cell_totals_);
    if (h_cell_sieve_hits_) cudaFreeHost(h_cell_sieve_hits_);

    if (d_events_) cudaFree(d_events_);
    if (d_homography_matrix_) cudaFree(d_homography_matrix_);
    if (d_ring_buffers_) cudaFree(d_ring_buffers_);
    if (d_cell_total_events_) cudaFree(d_cell_total_events_);
    if (d_cell_max_sieve_hits_) cudaFree(d_cell_max_sieve_hits_);
    if (d_fft_input_) cudaFree(d_fft_input_);
    if (d_fft_output_) cudaFree(d_fft_output_);
    if (d_power_spectrum_) cudaFree(d_power_spectrum_);
    if (d_candidates_) cudaFree(d_candidates_);
    if (d_num_candidates_) cudaFree(d_num_candidates_);
}

void CudaFlickerCore::reset() {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    sieve_.reset();
    retained_count_.store(0);
    CUDA_CHECK(cudaMemsetAsync(d_ring_buffers_, 0, num_total_cells_ * history_samples_ * sizeof(float), stream_));
    CUDA_CHECK(cudaMemsetAsync(d_cell_total_events_, 0, num_total_cells_ * sizeof(float), stream_));
    CUDA_CHECK(cudaMemsetAsync(d_cell_max_sieve_hits_, 0, num_total_cells_ * sizeof(uint32_t), stream_));
    current_window_start_us_ = 0;
    window_anchored_ = false;
    head_idx_ = 0;
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CudaFlickerCore::reset_sieve_hit_accumulators() {
    // Only the per-cell accumulator is reset here. The CPU tile sieve state is owned by the
    // camera-callback thread and self-resets on period gaps/inconsistency, so touching it from
    // the analysis thread would be a data race.
    CUDA_CHECK(cudaMemsetAsync(d_cell_max_sieve_hits_, 0, num_total_cells_ * sizeof(uint32_t), stream_));
}

uint64_t CudaFlickerCore::get_and_reset_retained_count() {
    return retained_count_.exchange(0, std::memory_order_relaxed);
}

void CudaFlickerCore::advance_temporal_bins(size_t steps) {
    if (steps == 0) return;
    size_t old_head = head_idx_;
    size_t clamped_steps = std::min(steps, history_samples_);

    int threads = 256;
    int blocks = (num_total_cells_ + threads - 1) / threads;

    kernel_advance_temporal_bins<<<blocks, threads, 0, stream_>>>(
        d_ring_buffers_, d_cell_total_events_, d_cell_max_sieve_hits_, old_head, clamped_steps, num_total_cells_);

    head_idx_ = (head_idx_ + steps) % history_samples_;
}

void CudaFlickerCore::ingest_chunk(size_t chunk_size, uint64_t chunk_max_t,
                                   const float* d_suppression_mask, float suppression_threshold) {
    // Window alignment: advance temporal ring buffer bins up to the newest event of this chunk.
    // The head bin start is always snapped to an absolute grid (multiple of bin_duration_us_) so
    // bin boundaries do not depend on which event arrived first or on the stream start offset.
    if (!window_anchored_) {
        uint64_t chunk_min_t = h_event_buffer_[0].t;
        for (size_t i = 1; i < chunk_size; ++i) chunk_min_t = std::min(chunk_min_t, h_event_buffer_[i].t);
        current_window_start_us_ = (chunk_min_t / bin_duration_us_) * bin_duration_us_;
        window_anchored_ = true;
    }
    if (chunk_max_t >= current_window_start_us_ + bin_duration_us_) {
        uint64_t elapsed_us = chunk_max_t - current_window_start_us_;
        uint64_t steps = elapsed_us / bin_duration_us_;
        if (steps >= history_samples_) {
            steps = history_samples_;
            current_window_start_us_ = (chunk_max_t / bin_duration_us_) * bin_duration_us_;
        } else {
            current_window_start_us_ += steps * bin_duration_us_;
        }
        advance_temporal_bins(steps);
    }

    // Asynchronous DMA from pinned staging; upload_done_ guards the staging buffer's reuse.
    CUDA_CHECK(cudaMemcpyAsync(d_events_, h_event_buffer_, chunk_size * sizeof(CudaRawEvent),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaEventRecord(upload_done_, stream_));

    int threads = 256;
    int blocks = (static_cast<int>(chunk_size) + threads - 1) / threads;
    kernel_warp_sieve_ingest<<<blocks, threads, 0, stream_>>>(
        d_events_, chunk_size, d_homography_matrix_,
        d_suppression_mask, suppression_threshold,
        sensor_width_, sensor_height_,
        d_ring_buffers_, d_cell_total_events_, d_cell_max_sieve_hits_,
        head_idx_, current_window_start_us_, static_cast<uint32_t>(bin_duration_us_));
    CUDA_CHECK(cudaGetLastError());
}

void CudaFlickerCore::ingest_event_batch(const void* events, size_t count, const Matrix3x3& H,
                                         uint64_t& out_raw_count, uint64_t& out_retained_count,
                                         const float* d_suppression_mask,
                                         float suppression_threshold,
                                         bool sync) {
    out_raw_count = 0;
    out_retained_count = 0;
    if (count == 0 || events == nullptr) return;

    const auto* ev_arr = static_cast<const Metavision::EventCD*>(events);

    // Homography for this batch (pageable source: copied before cudaMemcpyAsync returns).
    float h_mat[9];
    for (int i = 0; i < 9; ++i) h_mat[i] = static_cast<float>(H.m[i]);
    CUDA_CHECK(cudaMemcpyAsync(d_homography_matrix_, h_mat, 9 * sizeof(float), cudaMemcpyHostToDevice, stream_));

    uint64_t batch_retained = 0;
    size_t offset = 0;
    while (offset < count) {
        const size_t chunk = std::min(count - offset, max_events_per_batch_);

        // The previous async upload must finish before the pinned staging buffer is rewritten.
        CUDA_CHECK(cudaEventSynchronize(upload_done_));

        // CPU stage (timestamp order): periodicity sieve on raw sensor pixels + staging copy.
        // The sieve runs on raw (unwarped) coordinates: blade periodicity is a per-photoreceptor
        // property over <= 2 periods (~27 ms), where stabilization is irrelevant.
        uint64_t chunk_max_t = ev_arr[offset].t;
        for (size_t i = 0; i < chunk; ++i) {
            const Metavision::EventCD& src = ev_arr[offset + i];
            CudaRawEvent& dst = h_event_buffer_[i];
            dst.x = src.x;
            dst.y = src.y;
            dst.p = src.p;
            const uint8_t hits = sieve_.periodic_hits(src.x, src.y, static_cast<uint64_t>(src.t));
            dst.pad = static_cast<int16_t>(hits);
            dst.t = static_cast<uint64_t>(src.t);
            if (hits > 0) ++batch_retained;
            if (dst.t > chunk_max_t) chunk_max_t = dst.t;
        }

        ingest_chunk(chunk, chunk_max_t, d_suppression_mask, suppression_threshold);
        offset += chunk;
    }

    retained_count_.fetch_add(batch_retained, std::memory_order_relaxed);
    out_raw_count = count;
    out_retained_count = batch_retained;

    if (sync) {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
}

void CudaFlickerCore::ingest_device_events(
    const CudaRawEvent* d_events,
    size_t count,
    const Matrix3x3& H,
    uint64_t min_t,
    uint64_t max_t,
    cudaStream_t stream,
    const float* d_suppression_mask,
    float suppression_threshold)
{
    if (count == 0 || d_events == nullptr) return;
    cudaStream_t s = stream ? stream : stream_;

    // 1. Homography matrix (pageable source: copied before cudaMemcpyAsync returns)
    float h_mat[9];
    for (int i = 0; i < 9; ++i) h_mat[i] = static_cast<float>(H.m[i]);
    CUDA_CHECK(cudaMemcpyAsync(d_homography_matrix_, h_mat, 9 * sizeof(float), cudaMemcpyHostToDevice, s));

    // 2. Window alignment: advance temporal ring buffer bins up to max_t
    if (!window_anchored_) {
        current_window_start_us_ = (min_t / bin_duration_us_) * bin_duration_us_;
        window_anchored_ = true;
    }
    if (max_t >= current_window_start_us_ + bin_duration_us_) {
        uint64_t elapsed_us = max_t - current_window_start_us_;
        uint64_t steps = elapsed_us / bin_duration_us_;
        if (steps >= history_samples_) {
            steps = history_samples_;
            current_window_start_us_ = (max_t / bin_duration_us_) * bin_duration_us_;
        } else {
            current_window_start_us_ += steps * bin_duration_us_;
        }
        advance_temporal_bins(steps);
    }

    // 3. Launch accumulation kernel directly on device events
    int threads = 256;
    int blocks = (static_cast<int>(count) + threads - 1) / threads;
    kernel_warp_sieve_ingest<<<blocks, threads, 0, s>>>(
        d_events, static_cast<int>(count), d_homography_matrix_,
        d_suppression_mask, suppression_threshold,
        sensor_width_, sensor_height_,
        d_ring_buffers_, d_cell_total_events_, d_cell_max_sieve_hits_,
        head_idx_, current_window_start_us_, static_cast<uint32_t>(bin_duration_us_));
    CUDA_CHECK(cudaGetLastError());
}

void CudaFlickerCore::set_spectral_gate_config(const SpectralGateConfig& cfg) {
    gate_cfg_ = derive_spectral_gate(cfg, sample_rate_hz_, static_cast<int>(history_samples_), num_total_cells_);
}

void CudaFlickerCore::execute_batched_spectral_analysis(
    double gyro_speed_deg_s,
    std::vector<FlickerDetectionResult>& out_candidates) {

    out_candidates.clear();

    // 1. Prepare Hanning-windowed input for all 1152 cells
    kernel_prepare_fft_window<<<1152, 512, 0, stream_>>>(
        d_ring_buffers_, head_idx_, d_fft_input_);

    // 2. Execute Batched 1D Real-to-Complex cuFFT (1152 channels in parallel in <0.3 ms)
    CUFFT_CHECK(cufftExecR2C(cufft_plan_, d_fft_input_, d_fft_output_));

    // 3. Reset Candidate Counter
    CUDA_CHECK(cudaMemsetAsync(d_num_candidates_, 0, sizeof(uint32_t), stream_));

    // 4. Shared CFAR gate chain (spectral_gate.hpp) across all 1152 cells
    int threads = 256;
    int blocks = (1152 + threads - 1) / threads;

    kernel_analyze_spectral_peaks<<<blocks, threads, 0, stream_>>>(
        d_fft_output_, d_power_spectrum_, d_cell_total_events_, d_cell_max_sieve_hits_,
        gate_cfg_, static_cast<float>(gyro_speed_deg_s),
        d_candidates_, d_num_candidates_, static_cast<uint32_t>(max_candidates_));
    CUDA_CHECK(cudaGetLastError());

    // 5. Read back candidate count and candidate records
    uint32_t num_cands = 0;
    CUDA_CHECK(cudaMemcpyAsync(&num_cands, d_num_candidates_, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    if (num_cands > 0) {
        size_t to_fetch = std::min(static_cast<size_t>(num_cands), max_candidates_);
        CUDA_CHECK(cudaMemcpy(h_candidate_buffer_, d_candidates_, to_fetch * sizeof(CudaDetectionCandidate), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < to_fetch; ++i) {
            const auto& c = h_candidate_buffer_[i];
            FlickerDetectionResult res;
            res.is_drone_detected = true;
            res.fundamental_bpf_hz = c.fundamental_bpf_hz;
            res.estimated_rpm = (c.fundamental_bpf_hz * 60.0) / 2.0; // 2-blade default
            res.confidence = c.confidence;
            res.peak_snr_db = c.peak_snr_db;
            res.spectral_q_factor = c.spectral_q_factor;
            res.spectral_flatness = c.spectral_flatness;
            res.peak_power = c.peak_power;
            res.noise_floor = c.noise_floor;
            res.total_events = c.total_events;
            res.patch_x = c.patch_col;
            res.patch_y = c.patch_row;
            res.max_sieve_hits = c.max_sieve_hits;
            out_candidates.push_back(res);
        }
    }
}

int CudaFlickerCore::get_active_cell_count(double min_threshold) {
    CUDA_CHECK(cudaMemcpy(h_cell_totals_, d_cell_total_events_, num_total_cells_ * sizeof(float), cudaMemcpyDeviceToHost));
    int count = 0;
    for (int i = 0; i < num_base_cells_; ++i) {
        if (h_cell_totals_[i] >= min_threshold) {
            count++;
        }
    }
    return count;
}

RoiDiagnostics CudaFlickerCore::get_roi_diagnostics(int col_min, int col_max, int row_min, int row_max) {
    RoiDiagnostics diag;
    CUDA_CHECK(cudaMemcpy(h_cell_totals_, d_cell_total_events_, num_total_cells_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cell_sieve_hits_, d_cell_max_sieve_hits_, num_total_cells_ * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    col_min = std::max(0, std::min(col_min, grid_cols_ - 1));
    col_max = std::max(0, std::min(col_max, grid_cols_ - 1));
    row_min = std::max(0, std::min(row_min, grid_rows_ - 1));
    row_max = std::max(0, std::min(row_max, grid_rows_ - 1));

    int best_cell_idx = -1;
    for (int r = row_min; r <= row_max; ++r) {
        for (int c = col_min; c <= col_max; ++c) {
            int cell_idx = r * grid_cols_ + c;
            float ev = h_cell_totals_[cell_idx];
            uint32_t hits = h_cell_sieve_hits_[cell_idx];
            diag.total_events += ev;
            if (ev > diag.max_cell_events) {
                diag.max_cell_events = ev;
                best_cell_idx = cell_idx;
            }
            diag.max_sieve_hits = std::max(diag.max_sieve_hits, hits);
            if (ev >= 6.0f) {
                diag.active_cells++;
            }
        }
    }

    if (best_cell_idx >= 0 && diag.max_cell_events >= 20.0f) {
        float power[kSpectrumBins];
        CUDA_CHECK(cudaMemcpy(power, &d_power_spectrum_[best_cell_idx * kSpectrumBins],
                              kSpectrumBins * sizeof(float), cudaMemcpyDeviceToHost));
        const int c = best_cell_idx % grid_cols_;
        const int r = best_cell_idx / grid_cols_;
        const uint32_t hits = h_cell_sieve_hits_[best_cell_idx];
        const float ev = h_cell_totals_[best_cell_idx];

        // Identical gate chain to the GPU detector (gyro texture filter not applied: diagnostic view).
        const SpectralGateResult g = evaluate_spectral_gate(power, ev, hits, 0.0f, gate_cfg_);
        diag.best_snr_db = g.snr_db;
        diag.best_bpf_hz = g.fundamental_hz;
        diag.best_flatness = g.flatness;

        static int diag_throttle = 0;
        if (++diag_throttle % 10 == 0) {
            std::cout << "[ROI-TOP] Cell=(" << c << "," << r << ") Ev=" << ev << " Sieve=" << hits
                      << " | Flatness=" << g.flatness
                      << " | PeakBin=" << g.peak_bin << " FundBin=" << g.fund_bin
                      << " (" << g.fundamental_hz << "Hz) P=" << g.peak_power
                      << " Sharp=" << g.sharpness
                      << " SNR=" << g.snr_db << "dB (CFAR " << gate_cfg_.cfar_threshold_db << "dB) -> "
                      << gate_verdict_name(g.verdict) << "\n";
        }
    }
    return diag;
}

void CudaFlickerCore::get_active_cells_with_spectra(
    std::vector<int>& out_cell_indices,
    std::vector<std::vector<float>>& out_spectra,
    float min_events) {

    out_cell_indices.clear();
    out_spectra.clear();

    CUDA_CHECK(cudaMemcpy(h_cell_totals_, d_cell_total_events_, num_total_cells_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cell_sieve_hits_, d_cell_max_sieve_hits_, num_total_cells_ * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Identify active base and pooled cells with sufficient event density
    // Require micro-sieve periodic lock (hits >= 1 && ev >= min_events) or strong event density (ev >= 15.0f)
    struct CellPriority {
        int index;
        uint32_t hits;
        float events;
    };
    std::vector<CellPriority> cell_cands;
    cell_cands.reserve(num_total_cells_);

    for (int i = 0; i < num_total_cells_; ++i) {
        float ev = h_cell_totals_[i];
        uint32_t hits = h_cell_sieve_hits_[i];
        if ((hits >= 1 && ev >= min_events) || (ev >= 15.0f)) {
            cell_cands.push_back({i, hits, ev});
        }
    }

    if (cell_cands.empty()) return;

    // Prioritize cells with periodic micro-sieve hits first, then highest event counts
    std::sort(cell_cands.begin(), cell_cands.end(), [](const CellPriority& a, const CellPriority& b) {
        if (a.hits != b.hits) {
            return a.hits > b.hits; // Periodic rotor hits first!
        }
        return a.events > b.events; // Higher event count second!
    });

    size_t batch = std::min(cell_cands.size(), static_cast<size_t>(128));
    for (size_t i = 0; i < batch; ++i) {
        out_cell_indices.push_back(cell_cands[i].index);
    }

    out_spectra.resize(batch, std::vector<float>(257, 0.0f));

    for (size_t b = 0; b < batch; ++b) {
        int cell_idx = out_cell_indices[b];
        CUDA_CHECK(cudaMemcpyAsync(out_spectra[b].data(),
                                   &d_power_spectrum_[cell_idx * 257],
                                   257 * sizeof(float),
                                   cudaMemcpyDeviceToHost,
                                   stream_));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // Log10 median normalization matching SpectralCombNet training:
    // log10(1 + P / median_noise)
    for (size_t b = 0; b < batch; ++b) {
        auto& spec = out_spectra[b];
        std::vector<float> noise_slice(spec.begin() + 5, spec.begin() + 128);
        std::sort(noise_slice.begin(), noise_slice.end());
        float median_noise = std::max(1e-4f, noise_slice[noise_slice.size() / 2]);
        for (int k = 0; k < 257; ++k) {
            spec[k] = std::log10(1.0f + spec[k] / median_noise);
        }
    }
}

void CudaFlickerCore::get_cell_total_events(std::vector<float>& out_totals) {
    out_totals.resize(num_total_cells_);
    CUDA_CHECK(cudaMemcpy(out_totals.data(), d_cell_total_events_, num_total_cells_ * sizeof(float), cudaMemcpyDeviceToHost));
}

void CudaFlickerCore::compute_normalized_spectrum(const float* time_series_512, float* out_spectrum_257, float* out_median_noise) {
    if (!time_series_512 || !out_spectrum_257) return;

    // Copy time series into cell 0 ring buffer with chronological alignment (head_idx = 511)
    CUDA_CHECK(cudaMemcpyAsync(&d_ring_buffers_[0], time_series_512, 512 * sizeof(float), cudaMemcpyHostToDevice, stream_));
    head_idx_ = 511;

    // Apply Hanning window
    kernel_prepare_fft_window<<<1, 512, 0, stream_>>>(d_ring_buffers_, head_idx_, d_fft_input_);

    // Execute cuFFT R2C
    CUFFT_CHECK(cufftExecR2C(cufft_plan_, d_fft_input_, d_fft_output_));

    // Read back complex bins for cell 0
    cufftComplex h_cell_fft[257];
    CUDA_CHECK(cudaMemcpyAsync(h_cell_fft, &d_fft_output_[0], 257 * sizeof(cufftComplex), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<float> power(257);
    for (int k = 0; k < 257; ++k) {
        float re = h_cell_fft[k].x;
        float im = h_cell_fft[k].y;
        power[k] = re * re + im * im;
    }

    // Median noise in bins 5..127 (123 elements)
    std::vector<float> noise_slice(power.begin() + 5, power.begin() + 128);
    std::sort(noise_slice.begin(), noise_slice.end());
    float median_noise = std::max(1e-4f, noise_slice[noise_slice.size() / 2]);
    if (out_median_noise) *out_median_noise = median_noise;

    for (int k = 0; k < 257; ++k) {
        out_spectrum_257[k] = std::log10(1.0f + power[k] / median_noise);
    }
}

} // namespace predator
