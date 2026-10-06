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
 * @brief CUDA Kernel: Unwarp events via Homography H, evaluate TensorRT ego-motion suppression mask,
 * evaluate Surface of Active Events (SAE) Periodicity Sieve with atomic timestamp sequencing,
 * and atomically accumulate into spatial temporal ring buffers.
 */
__global__ void kernel_warp_sieve_ingest(
    const CudaRawEvent* __restrict__ events,
    size_t count,
    const float* __restrict__ H,
    const float* __restrict__ suppression_mask,
    float suppression_threshold,
    int width, int height,
    int tile_w, int tile_h,
    uint32_t* __restrict__ sae_timestamp_us,
    uint32_t* __restrict__ sae_last_dt_us,
    uint8_t* __restrict__ sae_hits,
    float* __restrict__ ring_buffers,
    float* __restrict__ cell_total_events,
    uint32_t* __restrict__ cell_max_sieve_hits,
    size_t head_idx,
    uint64_t head_bin_start_us,
    uint32_t bin_duration_us,
    uint32_t min_period_us,
    uint32_t max_period_us,
    uint32_t* __restrict__ retained_counter) {

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

    stab_x = fmaxf(0.0f, fminf(static_cast<float>(width - 1), stab_x));
    stab_y = fmaxf(0.0f, fminf(static_cast<float>(height - 1), stab_y));

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

    int sx = __float2int_rn(stab_x);
    int sy = __float2int_rn(stab_y);

    int tx = sx >> 1;
    int ty = sy >> 1;
    if (tx < 0 || tx >= tile_w || ty < 0 || ty >= tile_h) return;

    int tile_idx = ty * tile_w + tx;
    uint32_t t_cur = static_cast<uint32_t>(ev.t);

    // 3. Atomic timestamp exchange to serialize concurrent micro-tile events
    uint32_t t_last = atomicExch(&sae_timestamp_us[tile_idx], t_cur);

    bool is_periodic = false;
    if (t_last == 0) {
        sae_hits[tile_idx] = 0;
        sae_last_dt_us[tile_idx] = 0;
    } else if (t_cur < t_last) {
        return; // Reject out-of-order bus packets
    } else {
        uint32_t dt = t_cur - t_last;

        // 4. Intra-burst event (< min_period_us): Event belongs to the SAME active blade sweep!
        if (dt < min_period_us) {
            if (sae_hits[tile_idx] >= 2) {
                is_periodic = true;
            }
        }
        // 5. Inter-blade period match [min_period, max_period] (75 Hz to 1200 Hz => 833 us to 13333 us)
        else if (dt <= max_period_us) {
            uint32_t prev_dt = sae_last_dt_us[tile_idx];
            bool period_consistent = (prev_dt > 0) &&
                (((dt > prev_dt) ? (dt - prev_dt) : (prev_dt - dt)) * 100 <= prev_dt * 45); // 45% jitter tolerance

            if (period_consistent) {
                uint8_t h = sae_hits[tile_idx];
                if (h < 255) h++;
                sae_hits[tile_idx] = h;
                if (h >= 2) {
                    is_periodic = true;
                }
            } else {
                sae_hits[tile_idx] = 1;
            }
            sae_last_dt_us[tile_idx] = dt;
        }

        // 6. Cross-tile neighbor boundary check (for small rotors at 30-100ft drifting across 2x2 micro-tiles)
        if (!is_periodic) {
            const int n_offsets[4][2] = { {1, 0}, {-1, 0}, {0, 1}, {0, -1} };
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int nx = tx + n_offsets[k][0];
                int ny = ty + n_offsets[k][1];
                if (nx >= 0 && nx < tile_w && ny >= 0 && ny < tile_h) {
                    int n_idx = ny * tile_w + nx;
                    uint32_t nt_last = sae_timestamp_us[n_idx];
                    if (nt_last > 0 && t_cur > nt_last) {
                        uint32_t ndt = t_cur - nt_last;
                        if (ndt >= min_period_us && ndt <= max_period_us) {
                            uint32_t nprev_dt = sae_last_dt_us[n_idx];
                            bool n_consistent = (nprev_dt > 0) &&
                                (((ndt > nprev_dt) ? (ndt - nprev_dt) : (nprev_dt - ndt)) * 100 <= nprev_dt * 45);
                            if (n_consistent) {
                                uint8_t nh = sae_hits[n_idx];
                                uint8_t h = (nh < 255) ? (nh + 1) : 255;
                                sae_hits[tile_idx] = h;
                                sae_last_dt_us[tile_idx] = ndt;
                                if (h >= 2) {
                                    is_periodic = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        // 7. Time gap > max_period_us: Reset tracking
        if (dt > max_period_us && !is_periodic) {
            sae_hits[tile_idx] = 0;
            sae_last_dt_us[tile_idx] = 0;
        }
    }

    if (is_periodic) {
        atomicAdd(retained_counter, 1);
    }

    // 8. Temporal Ring Buffer Bin Calculation:
    // Map event timestamp ev.t directly to its exact microsecond temporal slot relative to head_bin_start_us
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
            atomicMax(&cell_max_sieve_hits[cell_idx], static_cast<uint32_t>(sae_hits[tile_idx]));
        }

        // 9. Also accumulate into 2x2 pooled cells (576..1151)
        for (int pr = max(0, row - 1); pr <= min(17, row); ++pr) {
            for (int pc = max(0, col - 1); pc <= min(31, col); ++pc) {
                int pooled_idx = 576 + (pr * 32 + pc);
                atomicAdd(&ring_buffers[pooled_idx * 512 + event_slot], 1.0f);
                atomicAdd(&cell_total_events[pooled_idx], 1.0f);
                if (cell_max_sieve_hits != nullptr && is_periodic) {
                    atomicMax(&cell_max_sieve_hits[pooled_idx], static_cast<uint32_t>(sae_hits[tile_idx]));
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
 * @brief CUDA Kernel: Fast Parallel Spectral Peak, SNR, and Harmonic Comb Detector for 1152 cells
 */
__global__ void kernel_analyze_spectral_peaks(
    const cufftComplex* __restrict__ fft_output,
    float* __restrict__ power_spectrum_out,
    const float* __restrict__ cell_total_events,
    const uint32_t* __restrict__ cell_max_sieve_hits,
    float min_freq_hz,
    float max_freq_hz,
    float sample_rate_hz,
    float min_energy,
    float min_snr_db,
    float gyro_speed_deg_s,
    CudaDetectionCandidate* __restrict__ candidates,
    uint32_t* __restrict__ num_candidates,
    uint32_t max_candidates) {

    int cell_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell_idx >= 1152) return;

    bool is_pooled = (cell_idx >= 576);
    int base_cell = is_pooled ? (cell_idx - 576) : cell_idx;
    int patch_col = base_cell % 32;
    int patch_row = base_cell / 32;

    float total_events = cell_total_events[cell_idx];
    uint32_t max_sieve_hits = (cell_max_sieve_hits != nullptr) ? cell_max_sieve_hits[cell_idx] : 0;

    const cufftComplex* cell_fft = &fft_output[cell_idx * 257];

    // Compute power spectrum in registers / local memory
    float power[257];
    #pragma unroll 4
    for (int k = 0; k < 257; ++k) {
        float re = cell_fft[k].x;
        float im = cell_fft[k].y;
        float p = (re * re + im * im);
        power[k] = p;
        if (power_spectrum_out != nullptr) {
            power_spectrum_out[cell_idx * 257 + k] = p;
        }
    }

    // Standoff Dynamic Activity Gate:
    // Allow weak standoff signals (>= 6.0 events) to proceed if periodic micro-sieve locked (max_sieve_hits >= 2).
    // Require higher event density (>= 15 base / 22 pooled) for aperiodic clutter/noise.
    float min_activity_req = (max_sieve_hits >= 2) ? 6.0f : (is_pooled ? 22.0f : 15.0f);
    if (total_events < min_activity_req) return;

    // Peak search using Harmonic Product Spectrum (HPS) in [min_freq_hz, max_freq_hz] (bins 9 to 102 for 70..800 Hz)
    float df = sample_rate_hz / 512.0f; // 7.8125 Hz
    int min_bin = max(1, __float2int_rd(min_freq_hz / df));
    int max_bin = min(255, __float2int_ru(max_freq_hz / df));

    // DDHF Spectral Flatness Calculation (scale-invariant harmonic comb metric):
    // gamma = exp((1/N) * sum(ln(P_k + eps))) / ((1/N) * sum(P_k))
    float sum_log = 0.0f;
    float sum_p = 0.0f;
    int search_band_bins = 0;
    for (int k = min_bin; k <= max_bin; ++k) {
        sum_log += logf(power[k] + 1e-9f);
        sum_p += power[k];
        search_band_bins++;
    }
    float geom_mean = (search_band_bins > 0) ? expf(sum_log / static_cast<float>(search_band_bins)) : 1.0f;
    float arith_mean = (search_band_bins > 0) ? (sum_p / static_cast<float>(search_band_bins)) : 1.0f;
    float spectral_flatness = (arith_mean > 1e-9f) ? (geom_mean / arith_mean) : 1.0f;

    // DDHF Flatness Gate: reject broad diffuse noise / uniform spectrum
    if (spectral_flatness > 0.35f) return;

    int best_bin = -1;
    float max_hps = 0.0f;

    // Minimum energy threshold: scale down if high-purity micro-sieve hit
    float effective_min_energy = (max_sieve_hits >= 2) ? (0.5f * min_energy) : min_energy;

    for (int k = min_bin; k <= max_bin; ++k) {
        float p1 = power[k];
        if (p1 < effective_min_energy || p1 <= power[k - 1] || p1 <= power[k + 1]) continue;

        float p2 = (2 * k < 257) ? power[2 * k] : 0.0f;
        float p3 = (3 * k < 257) ? power[3 * k] : 0.0f;

        float hps_score = p1 + 0.6f * p2 + 0.4f * p3;
        if (hps_score > max_hps) {
            max_hps = hps_score;
            best_bin = k;
        }
    }

    if (best_bin < 0 || power[best_bin] < effective_min_energy) return;

    // Subharmonic Fundamental Disambiguation:
    // If a higher harmonic (e.g. 2x or 3x) was selected, check if subharmonic f0 is also an active peak
    #pragma unroll
    for (int sub = 3; sub >= 2; --sub) {
        int center_sub = __float2int_rn(static_cast<float>(best_bin) / static_cast<float>(sub));
        int found_sub = -1;
        float max_sub_p = 0.0f;
        for (int k = max(min_bin, center_sub - 1); k <= min(max_bin, center_sub + 1); ++k) {
            if (power[k] >= effective_min_energy && power[k] > power[k - 1] && power[k] > power[k + 1]) {
                if (power[k] > max_sub_p) {
                    max_sub_p = power[k];
                    found_sub = k;
                }
            }
        }
        if (found_sub >= 0 && max_sub_p >= 0.35f * power[best_bin]) {
            best_bin = found_sub; // Demote to true fundamental blade passage frequency
        }
    }

    // Wideband noise floor estimation: Median CFAR across bins 5 to 128 (40 Hz to 1000 Hz) excluding signal peaks
    float noise_bins[128];
    int noise_count = 0;
    for (int k = 5; k <= 128; ++k) {
        bool is_signal = (abs(k - best_bin) <= 2) ||
                         (abs(k - 2 * best_bin) <= 2) ||
                         (abs(k - 3 * best_bin) <= 2);
        if (!is_signal && noise_count < 128) {
            noise_bins[noise_count++] = power[k];
        }
    }

    // Fast in-place QuickSelect for median noise computation
    float median_val = 1e-9f;
    if (noise_count > 0) {
        int left = 0, right = noise_count - 1;
        int target_k = noise_count / 2;
        while (left < right) {
            float pivot = noise_bins[target_k];
            int i = left, j = right;
            while (i <= j) {
                while (noise_bins[i] < pivot) i++;
                while (noise_bins[j] > pivot) j--;
                if (i <= j) {
                    float tmp = noise_bins[i];
                    noise_bins[i] = noise_bins[j];
                    noise_bins[j] = tmp;
                    i++;
                    j--;
                }
            }
            if (target_k <= j) {
                right = j;
            } else if (target_k >= i) {
                left = i;
            } else {
                break;
            }
        }
        median_val = noise_bins[target_k];
    }
    // Unbiased exponential noise power estimator scale: 1 / ln(2) = 1.442695
    float mean_noise = median_val * 1.442695f;
    if (mean_noise < 1e-9f) mean_noise = 1e-9f;

    // Parabolic sub-bin interpolation
    float p_left = power[best_bin - 1];
    float p_mid  = power[best_bin];
    float p_right = power[best_bin + 1];
    float delta_bin = 0.0f;
    float denom = 2.0f * (2.0f * p_mid - p_left - p_right);
    if (fabsf(denom) > 1e-9f) {
        delta_bin = (p_right - p_left) / denom;
    }
    float peak_freq_hz = (static_cast<float>(best_bin) + delta_bin) * df;

    // Spectral Sharpness (Q-Factor): blade spike vs broad wind/foliage turbulence
    float neighbor_p = 1e-9f;
    if (best_bin >= 2 && best_bin + 2 < 257) {
        neighbor_p = 0.5f * (power[best_bin - 2] + power[best_bin + 2]);
    }
    float sharpness = (neighbor_p > 1e-9f) ? (p_mid / neighbor_p) : 10.0f;
    if (sharpness < 1.5f) return;

    // Motion-induced background texture scanning frequency filter:
    // v_scan = fx * omega (px/s). Apparent texture frequency band: [0.22 v_scan, 0.70 v_scan]
    // ONLY reject broad foliage texture scan clutter; NEVER reject high-Q, low-flatness mechanical blade harmonics!
    if (gyro_speed_deg_s > 4.0f) {
        float omega_rad_s = gyro_speed_deg_s * (3.14159265f / 180.0f);
        float v_scan = 1646.0f * omega_rad_s;
        float f_texture_min = 0.22f * v_scan;
        float f_texture_max = 0.70f * v_scan;
        if (peak_freq_hz >= f_texture_min && peak_freq_hz <= f_texture_max) {
            if (sharpness < 2.5f && spectral_flatness > 0.20f && max_sieve_hits < 2) {
                // Background moving foliage scan artifact: reject broad diffuse clutter
                return;
            }
        }
    }

    // SNR Calculation
    float snr_linear = p_mid / mean_noise;
    float snr_db = 10.0f * log10f(max(1.0f, snr_linear));

    // Dynamic SNR Threshold: Standoff low-SNR tolerance if confirmed by DDHF flatness and micro-sieve
    float effective_min_snr = (max_sieve_hits >= 2 && spectral_flatness < 0.18f) ? (min_snr_db - 3.0f) : min_snr_db;
    if (snr_db < effective_min_snr) return;

    // Harmonic Bonus (2nd and 3rd harmonics)
    float h2_ratio = (2 * best_bin < 257) ? (power[2 * best_bin] / mean_noise) : 0.0f;
    float h3_ratio = (3 * best_bin < 257) ? (power[3 * best_bin] / mean_noise) : 0.0f;
    float harmonic_bonus = (h2_ratio > 1.5f ? 0.20f : 0.0f) + (h3_ratio > 1.2f ? 0.15f : 0.0f);

    // Standoff DDHF Flatness Bonus & Micro-Sieve Bonus
    float flatness_bonus = (spectral_flatness < 0.18f) ? (0.25f * (0.18f - spectral_flatness) / 0.18f) : 0.0f;
    float sieve_bonus = (max_sieve_hits >= 2) ? 0.15f : 0.0f;

    float fund_score = min(1.0f, max(0.20f, (snr_db - 6.0f) / 10.0f));
    float confidence = min(1.0f, fund_score + harmonic_bonus + flatness_bonus + sieve_bonus);

    if (confidence < 0.30f) return;

    // Atomically write valid candidate
    uint32_t slot = atomicAdd(num_candidates, 1);
    if (slot < max_candidates) {
        CudaDetectionCandidate c;
        c.cell_idx = cell_idx;
        c.is_pooled = is_pooled ? 1 : 0;
        c.patch_col = patch_col;
        c.patch_row = patch_row;
        c.fundamental_bpf_hz = peak_freq_hz;
        c.peak_snr_db = snr_db;
        c.spectral_q_factor = sharpness;
        c.peak_power = p_mid;
        c.noise_floor = mean_noise;
        c.confidence = confidence;
        c.total_events = total_events;
        c.spectral_flatness = spectral_flatness;
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
      bin_duration_us_(static_cast<uint64_t>(1000000.0 / sample_rate_hz)) {

    num_base_cells_ = grid_cols_ * grid_rows_;
    num_total_cells_ = num_base_cells_ * 2; // 576 base + 576 pooled = 1152
    tile_w_ = (sensor_width_ + 1) / 2;
    tile_h_ = (sensor_height_ + 1) / 2;
    num_tiles_ = tile_w_ * tile_h_;

    CUDA_CHECK(cudaStreamCreate(&stream_));

    // Allocate Host Pinned Memory for Zero-Copy DMA transfers
    CUDA_CHECK(cudaMallocHost(&h_event_buffer_, max_events_per_batch_ * sizeof(CudaRawEvent)));
    CUDA_CHECK(cudaMallocHost(&h_candidate_buffer_, max_candidates_ * sizeof(CudaDetectionCandidate)));
    CUDA_CHECK(cudaMallocHost(&h_cell_totals_, num_total_cells_ * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_cell_sieve_hits_, num_total_cells_ * sizeof(uint32_t)));

    // Allocate Device GPU Memory
    CUDA_CHECK(cudaMalloc(&d_events_, max_events_per_batch_ * sizeof(CudaRawEvent)));
    CUDA_CHECK(cudaMalloc(&d_retained_counter_, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_homography_matrix_, 9 * sizeof(float)));

    // Allocate Surface of Active Events (SAE) in GPU memory
    CUDA_CHECK(cudaMalloc(&d_sae_timestamp_us_, num_tiles_ * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sae_last_dt_us_, num_tiles_ * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sae_hits_, num_tiles_ * sizeof(uint8_t)));

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
              << num_total_cells_ << " cuFFT channels, " << num_tiles_ << " SAE micro-tiles).\n";
}

CudaFlickerCore::~CudaFlickerCore() {
    if (cufft_plan_ != 0) {
        cufftDestroy(cufft_plan_);
    }
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }

    if (h_event_buffer_) cudaFreeHost(h_event_buffer_);
    if (h_candidate_buffer_) cudaFreeHost(h_candidate_buffer_);
    if (h_cell_totals_) cudaFreeHost(h_cell_totals_);
    if (h_cell_sieve_hits_) cudaFreeHost(h_cell_sieve_hits_);

    if (d_events_) cudaFree(d_events_);
    if (d_retained_counter_) cudaFree(d_retained_counter_);
    if (d_homography_matrix_) cudaFree(d_homography_matrix_);
    if (d_sae_timestamp_us_) cudaFree(d_sae_timestamp_us_);
    if (d_sae_last_dt_us_) cudaFree(d_sae_last_dt_us_);
    if (d_sae_hits_) cudaFree(d_sae_hits_);
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
    CUDA_CHECK(cudaMemsetAsync(d_sae_timestamp_us_, 0, num_tiles_ * sizeof(uint32_t), stream_));
    CUDA_CHECK(cudaMemsetAsync(d_sae_last_dt_us_, 0, num_tiles_ * sizeof(uint32_t), stream_));
    CUDA_CHECK(cudaMemsetAsync(d_sae_hits_, 0, num_tiles_ * sizeof(uint8_t), stream_));
    CUDA_CHECK(cudaMemsetAsync(d_ring_buffers_, 0, num_total_cells_ * history_samples_ * sizeof(float), stream_));
    CUDA_CHECK(cudaMemsetAsync(d_cell_total_events_, 0, num_total_cells_ * sizeof(float), stream_));
    CUDA_CHECK(cudaMemsetAsync(d_cell_max_sieve_hits_, 0, num_total_cells_ * sizeof(uint32_t), stream_));
    CUDA_CHECK(cudaMemsetAsync(d_retained_counter_, 0, sizeof(uint32_t), stream_));
    current_window_start_us_ = 0;
    head_idx_ = 0;
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

uint64_t CudaFlickerCore::get_and_reset_retained_count() {
    uint32_t h_retained = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_retained, d_retained_counter_, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemsetAsync(d_retained_counter_, 0, sizeof(uint32_t), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    return static_cast<uint64_t>(h_retained);
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

void CudaFlickerCore::ingest_event_batch(const void* events, size_t count, const Matrix3x3& H,
                                         uint64_t& out_raw_count, uint64_t& out_retained_count,
                                         const float* d_suppression_mask,
                                         float suppression_threshold,
                                         bool sync) {
    if (count == 0) {
        out_raw_count = 0;
        out_retained_count = 0;
        return;
    }

    const auto* ev_arr = static_cast<const Metavision::EventCD*>(events);
    size_t batch_size = std::min(count, max_events_per_batch_);

    // Find the latest timestamp in this batch to synchronize temporal advancement
    uint64_t max_t = ev_arr[0].t;
    for (size_t i = 1; i < batch_size; ++i) {
        if (ev_arr[i].t > max_t) {
            max_t = ev_arr[i].t;
        }
    }

    // Window alignment: advance temporal ring buffer bins up to max_t
    if (current_window_start_us_ == 0) {
        current_window_start_us_ = ev_arr[0].t;
    }

    if (max_t >= current_window_start_us_ + bin_duration_us_) {
        uint64_t elapsed_us = max_t - current_window_start_us_;
        uint64_t steps = elapsed_us / bin_duration_us_;
        if (steps >= history_samples_) {
            steps = history_samples_;
            current_window_start_us_ = max_t;
        } else {
            current_window_start_us_ += steps * bin_duration_us_;
        }
        advance_temporal_bins(steps);
    }

    // Direct zero-copy DMA transfer (Metavision::EventCD and CudaRawEvent have identical 16-byte memory layout)
    CUDA_CHECK(cudaMemcpyAsync(d_events_, events, batch_size * sizeof(CudaRawEvent), cudaMemcpyHostToDevice, stream_));

    float h_mat[9];
    for (int i = 0; i < 9; ++i) h_mat[i] = static_cast<float>(H.m[i]);
    CUDA_CHECK(cudaMemcpyAsync(d_homography_matrix_, h_mat, 9 * sizeof(float), cudaMemcpyHostToDevice, stream_));

    if (sync) {
        CUDA_CHECK(cudaMemsetAsync(d_retained_counter_, 0, sizeof(uint32_t), stream_));
    }

    uint32_t min_period_us = 833;   // 1200 Hz (extreme high-speed throttle / FPV sprint)
    uint32_t max_period_us = 13333; // 75 Hz (ground idle / heavy lift)

    int threads = 256;
    int blocks = (static_cast<int>(batch_size) + threads - 1) / threads;

    kernel_warp_sieve_ingest<<<blocks, threads, 0, stream_>>>(
        d_events_, batch_size, d_homography_matrix_,
        d_suppression_mask, suppression_threshold,
        sensor_width_, sensor_height_, tile_w_, tile_h_,
        d_sae_timestamp_us_, d_sae_last_dt_us_, d_sae_hits_,
        d_ring_buffers_, d_cell_total_events_, d_cell_max_sieve_hits_,
        head_idx_, current_window_start_us_, static_cast<uint32_t>(bin_duration_us_),
        min_period_us, max_period_us, d_retained_counter_);

    out_raw_count = batch_size;

    if (sync) {
        uint32_t h_retained = 0;
        CUDA_CHECK(cudaMemcpyAsync(&h_retained, d_retained_counter_, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        out_retained_count = h_retained;
    } else {
        out_retained_count = 0;
    }
}

void CudaFlickerCore::execute_batched_spectral_analysis(
    double min_freq_hz, double max_freq_hz,
    double min_energy, double min_snr_db,
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

    // 4. Parallel Spectral Peak, Q-factor, SNR, and Harmonic Comb Analysis across all 1152 cells
    int threads = 256;
    int blocks = (1152 + threads - 1) / threads;

    kernel_analyze_spectral_peaks<<<blocks, threads, 0, stream_>>>(
        d_fft_output_, d_power_spectrum_, d_cell_total_events_, d_cell_max_sieve_hits_,
        static_cast<float>(min_freq_hz), static_cast<float>(max_freq_hz),
        static_cast<float>(sample_rate_hz_),
        static_cast<float>(min_energy), static_cast<float>(min_snr_db),
        static_cast<float>(gyro_speed_deg_s),
        d_candidates_, d_num_candidates_, static_cast<uint32_t>(max_candidates_));

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

    for (int r = row_min; r <= row_max; ++r) {
        for (int c = col_min; c <= col_max; ++c) {
            int cell_idx = r * grid_cols_ + c;
            float ev = h_cell_totals_[cell_idx];
            uint32_t hits = h_cell_sieve_hits_[cell_idx];
            diag.total_events += ev;
            diag.max_cell_events = std::max(diag.max_cell_events, ev);
            diag.max_sieve_hits = std::max(diag.max_sieve_hits, hits);
            if (ev >= 6.0f) {
                diag.active_cells++;
            }
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
    for (int i = 0; i < num_total_cells_; ++i) {
        float ev = h_cell_totals_[i];
        uint32_t hits = h_cell_sieve_hits_[i];
        if ((hits >= 1 && ev >= min_events) || (ev >= 15.0f)) {
            out_cell_indices.push_back(i);
            if (out_cell_indices.size() >= 128) break; // Capped at max batch size
        }
    }

    if (out_cell_indices.empty()) return;

    size_t batch = out_cell_indices.size();
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
        float median_noise = std::max(0.20f, noise_slice[noise_slice.size() / 2]);
        for (int k = 0; k < 257; ++k) {
            spec[k] = std::log10(1.0f + spec[k] / median_noise);
        }
    }
}

} // namespace predator
