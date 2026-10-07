/**
 * @file gpu_sieve.cu
 * @brief Implementation of Order-Correct GPU Micro-Neighborhood Periodicity Sieve.
 */

#include "gpu_sieve.cuh"

#include <iostream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cub/device/device_radix_sort.cuh>

namespace predator {

namespace {

inline void cuda_check(cudaError_t e, const char* context) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("GpuPeriodicitySieve: ") + context + ": " + cudaGetErrorString(e));
    }
}

constexpr int kThreads = 256;
inline unsigned blocks_for(size_t n) {
    return static_cast<unsigned>((n + kThreads - 1) / kThreads);
}

// 18 bits covers 0..262143 (our 640x360 grid has 230400 tiles)
constexpr int kRadixBits = 18;
constexpr uint32_t kSentinelKey = (1u << kRadixBits) - 1u; // 262143

/**
 * @brief Step 1: Extracts tile keys and initial sequence indices.
 */
__global__ void k_extract_tile_keys(
    const CudaRawEvent* __restrict__ events,
    int n,
    int tile_w,
    int tile_h,
    uint32_t* __restrict__ keys_in,
    uint32_t* __restrict__ values_in)
{
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;

    uint16_t x = events[i].x;
    uint16_t y = events[i].y;
    int tx = static_cast<int>(x >> 1);
    int ty = static_cast<int>(y >> 1);

    uint32_t key;
    if (tx >= 0 && tx < tile_w && ty >= 0 && ty < tile_h) {
        key = static_cast<uint32_t>(ty * tile_w + tx);
    } else {
        key = kSentinelKey;
    }

    keys_in[i] = key;
    values_in[i] = static_cast<uint32_t>(i);
}

/**
 * @brief Evaluates a single event against the micro-tile periodicity state machine.
 * Exactly replicates MicroNeighborhoodPeriodicitySieve::periodic_hits from flicker_dsp.hpp.
 */
__device__ inline uint8_t evaluate_micro_sieve_step(
    MicroTileState& state,
    uint32_t t_cur,
    int tx,
    int ty,
    int tile_w,
    int tile_h,
    uint32_t min_period_us,
    uint32_t max_period_us,
    uint8_t min_consecutive_hits,
    const MicroTileState* all_tile_states)
{
    uint32_t t_last = state.last_timestamp_us;
    if (t_last == 0) {
        state.last_timestamp_us = t_cur;
        state.consecutive_hits = 0;
        state.last_dt_us = 0;
        return 0;
    }

    uint32_t dt = t_cur - t_last;

    // 1. Intra-burst event (< min_period): event belongs to the SAME active blade sweep!
    if (dt < min_period_us) {
        if (state.consecutive_hits >= min_consecutive_hits) {
            return state.consecutive_hits;
        }
        return 0;
    }

    // 2. Inter-sweep recurrence (min_period <= dt <= max_period): a new periodic blade chop has arrived!
    if (dt <= max_period_us) {
        uint16_t prev_dt = state.last_dt_us;
        state.last_dt_us = static_cast<uint16_t>(dt < 65535u ? dt : 65535u);
        state.last_timestamp_us = t_cur;

        bool period_consistent = (prev_dt == 0) ||
            (abs(static_cast<int>(dt) - static_cast<int>(prev_dt)) <= static_cast<int>(prev_dt * 0.45f));

        if (period_consistent) {
            uint8_t h = state.consecutive_hits;
            if (h < 255) h++;
            state.consecutive_hits = h;
            if (h >= min_consecutive_hits) {
                return h;
            }
        } else {
            state.consecutive_hits = 1;
        }
        return 0;
    }

    // 3. Cross-tile boundary check (in case blade tip crossed into adjacent 2x2 tile)
    const int dtx[4] = {1, -1, 0, 0};
    const int dty[4] = {0, 0, 1, -1};

    for (int k = 0; k < 4; ++k) {
        int ntx = tx + dtx[k];
        int nty = ty + dty[k];
        if (ntx >= 0 && ntx < tile_w && nty >= 0 && nty < tile_h) {
            int n_idx = nty * tile_w + ntx;
            uint32_t nt_last = all_tile_states[n_idx].last_timestamp_us;
            if (nt_last > 0) {
                uint32_t ndt = t_cur - nt_last;
                if (ndt >= min_period_us && ndt <= max_period_us) {
                    uint16_t prev_dt = all_tile_states[n_idx].last_dt_us;
                    bool period_consistent = (prev_dt == 0) ||
                        (abs(static_cast<int>(ndt) - static_cast<int>(prev_dt)) <= static_cast<int>(prev_dt * 0.45f));
                    if (period_consistent) {
                        uint8_t h = state.consecutive_hits;
                        if (h < 255) h++;
                        state.consecutive_hits = h;
                        state.last_dt_us = static_cast<uint16_t>(ndt < 65535u ? ndt : 65535u);
                        state.last_timestamp_us = t_cur;
                        if (h >= min_consecutive_hits) {
                            return h;
                        }
                        return 0;
                    }
                }
            }
        }
    }

    // 4. Stale event (> max_period): time gap too large, reset tracking
    state.last_timestamp_us = t_cur;
    state.consecutive_hits = 0;
    state.last_dt_us = 0;
    return 0;
}

/**
 * @brief Step 3: Sequential per-tile scan kernel.
 * One thread per unique tile run processes its events in exact time order.
 */
__global__ void k_sieve_scan(
    CudaRawEvent* __restrict__ events,
    const uint32_t* __restrict__ keys_out,
    const uint32_t* __restrict__ values_out,
    int n,
    int tile_w,
    int tile_h,
    uint32_t num_tiles,
    uint32_t min_period_us,
    uint32_t max_period_us,
    uint8_t min_consecutive_hits,
    MicroTileState* tile_state,
    unsigned long long* __restrict__ retained_counter)
{
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;

    // Check if this thread is the start of a tile run
    if (i > 0 && keys_out[i] == keys_out[i - 1]) {
        return; // Handled by the thread at the start of this run
    }

    uint32_t tile_id = keys_out[i];
    if (tile_id >= num_tiles) {
        // Sentinel / out-of-bounds events: pad = 0
        uint32_t k = static_cast<uint32_t>(i);
        while (k < static_cast<uint32_t>(n) && keys_out[k] == tile_id) {
            events[values_out[k]].pad = 0;
            ++k;
        }
        return;
    }

    int tx = static_cast<int>(tile_id % tile_w);
    int ty = static_cast<int>(tile_id / tile_w);

    MicroTileState state = tile_state[tile_id];
    uint32_t local_retained = 0;

    uint32_t k = static_cast<uint32_t>(i);
    while (k < static_cast<uint32_t>(n) && keys_out[k] == tile_id) {
        uint32_t orig_idx = values_out[k];
        uint32_t t_cur = static_cast<uint32_t>(events[orig_idx].t);

        uint8_t hits = evaluate_micro_sieve_step(
            state, t_cur, tx, ty, tile_w, tile_h,
            min_period_us, max_period_us, min_consecutive_hits,
            tile_state
        );

        events[orig_idx].pad = static_cast<int16_t>(hits);
        if (hits > 0) {
            ++local_retained;
        }
        ++k;
    }

    // Write back persistent state for this tile
    tile_state[tile_id] = state;

    if (local_retained > 0 && retained_counter != nullptr) {
        atomicAdd(retained_counter, static_cast<unsigned long long>(local_retained));
    }
}

} // anonymous namespace

GpuPeriodicitySieve::GpuPeriodicitySieve(const GpuSieveConfig& cfg)
    : cfg_(cfg),
      tile_w_((cfg.sensor_width + 1) / 2),
      tile_h_((cfg.sensor_height + 1) / 2),
      num_tiles_(static_cast<size_t>(tile_w_ * tile_h_)),
      min_period_us_(static_cast<uint32_t>(1000000.0 / cfg.max_freq_hz)),
      max_period_us_(static_cast<uint32_t>(1000000.0 / cfg.min_freq_hz))
{
    // Allocate persistent tile state array in device memory
    cuda_check(cudaMalloc(&d_tile_state_, num_tiles_ * sizeof(MicroTileState)), "cudaMalloc d_tile_state_");
    cuda_check(cudaMemset(d_tile_state_, 0, num_tiles_ * sizeof(MicroTileState)), "cudaMemset d_tile_state_");

    // Allocate retained counters (device + pinned host)
    cuda_check(cudaMalloc(&d_retained_counter_, sizeof(unsigned long long)), "cudaMalloc d_retained_counter_");
    cuda_check(cudaHostAlloc(&h_pinned_retained_, sizeof(unsigned long long), cudaHostAllocMapped), "cudaHostAlloc h_pinned_retained_");

    // Pre-allocate capacity for up to 131,072 events
    ensure_capacity(131072);
}

GpuPeriodicitySieve::~GpuPeriodicitySieve() {
    free_resources();
}

GpuPeriodicitySieve::GpuPeriodicitySieve(GpuPeriodicitySieve&& other) noexcept
    : cfg_(other.cfg_),
      tile_w_(other.tile_w_),
      tile_h_(other.tile_h_),
      num_tiles_(other.num_tiles_),
      min_period_us_(other.min_period_us_),
      max_period_us_(other.max_period_us_),
      d_tile_state_(other.d_tile_state_),
      capacity_(other.capacity_),
      d_keys_in_(other.d_keys_in_),
      d_keys_out_(other.d_keys_out_),
      d_values_in_(other.d_values_in_),
      d_values_out_(other.d_values_out_),
      d_sort_temp_(other.d_sort_temp_),
      sort_temp_bytes_(other.sort_temp_bytes_),
      d_retained_counter_(other.d_retained_counter_),
      h_pinned_retained_(other.h_pinned_retained_)
{
    other.d_tile_state_ = nullptr;
    other.d_keys_in_ = nullptr;
    other.d_keys_out_ = nullptr;
    other.d_values_in_ = nullptr;
    other.d_values_out_ = nullptr;
    other.d_sort_temp_ = nullptr;
    other.sort_temp_bytes_ = 0;
    other.d_retained_counter_ = nullptr;
    other.h_pinned_retained_ = nullptr;
    other.capacity_ = 0;
}

GpuPeriodicitySieve& GpuPeriodicitySieve::operator=(GpuPeriodicitySieve&& other) noexcept {
    if (this != &other) {
        free_resources();

        cfg_ = other.cfg_;
        tile_w_ = other.tile_w_;
        tile_h_ = other.tile_h_;
        num_tiles_ = other.num_tiles_;
        min_period_us_ = other.min_period_us_;
        max_period_us_ = other.max_period_us_;
        d_tile_state_ = other.d_tile_state_;
        capacity_ = other.capacity_;
        d_keys_in_ = other.d_keys_in_;
        d_keys_out_ = other.d_keys_out_;
        d_values_in_ = other.d_values_in_;
        d_values_out_ = other.d_values_out_;
        d_sort_temp_ = other.d_sort_temp_;
        sort_temp_bytes_ = other.sort_temp_bytes_;
        d_retained_counter_ = other.d_retained_counter_;
        h_pinned_retained_ = other.h_pinned_retained_;

        other.d_tile_state_ = nullptr;
        other.d_keys_in_ = nullptr;
        other.d_keys_out_ = nullptr;
        other.d_values_in_ = nullptr;
        other.d_values_out_ = nullptr;
        other.d_sort_temp_ = nullptr;
        other.sort_temp_bytes_ = 0;
        other.d_retained_counter_ = nullptr;
        other.h_pinned_retained_ = nullptr;
        other.capacity_ = 0;
    }
    return *this;
}

void GpuPeriodicitySieve::free_resources() {
    if (d_tile_state_) { cudaFree(d_tile_state_); d_tile_state_ = nullptr; }
    if (d_keys_in_) { cudaFree(d_keys_in_); d_keys_in_ = nullptr; }
    if (d_keys_out_) { cudaFree(d_keys_out_); d_keys_out_ = nullptr; }
    if (d_values_in_) { cudaFree(d_values_in_); d_values_in_ = nullptr; }
    if (d_values_out_) { cudaFree(d_values_out_); d_values_out_ = nullptr; }
    if (d_sort_temp_) { cudaFree(d_sort_temp_); d_sort_temp_ = nullptr; }
    if (d_retained_counter_) { cudaFree(d_retained_counter_); d_retained_counter_ = nullptr; }
    if (h_pinned_retained_) { cudaFreeHost(h_pinned_retained_); h_pinned_retained_ = nullptr; }
    capacity_ = 0;
    sort_temp_bytes_ = 0;
}

void GpuPeriodicitySieve::ensure_capacity(size_t count) {
    if (count <= capacity_) return;

    size_t new_cap = std::max(count, capacity_ == 0 ? size_t{131072} : capacity_ * 2);

    if (d_keys_in_) cudaFree(d_keys_in_);
    if (d_keys_out_) cudaFree(d_keys_out_);
    if (d_values_in_) cudaFree(d_values_in_);
    if (d_values_out_) cudaFree(d_values_out_);
    if (d_sort_temp_) { cudaFree(d_sort_temp_); d_sort_temp_ = nullptr; sort_temp_bytes_ = 0; }

    cuda_check(cudaMalloc(&d_keys_in_, new_cap * sizeof(uint32_t)), "cudaMalloc d_keys_in_");
    cuda_check(cudaMalloc(&d_keys_out_, new_cap * sizeof(uint32_t)), "cudaMalloc d_keys_out_");
    cuda_check(cudaMalloc(&d_values_in_, new_cap * sizeof(uint32_t)), "cudaMalloc d_values_in_");
    cuda_check(cudaMalloc(&d_values_out_, new_cap * sizeof(uint32_t)), "cudaMalloc d_values_out_");

    // Query CUB temp storage requirement for SortPairs at maximum capacity
    size_t req_bytes = 0;
    cuda_check(cub::DeviceRadixSort::SortPairs(
        nullptr, req_bytes,
        d_keys_in_, d_keys_out_,
        d_values_in_, d_values_out_,
        static_cast<int>(new_cap),
        0, kRadixBits, nullptr
    ), "SortPairs query temp bytes");

    cuda_check(cudaMalloc(&d_sort_temp_, req_bytes), "cudaMalloc d_sort_temp_");
    sort_temp_bytes_ = req_bytes;
    capacity_ = new_cap;
}

void GpuPeriodicitySieve::reset(cudaStream_t stream) {
    cuda_check(cudaMemsetAsync(d_tile_state_, 0, num_tiles_ * sizeof(MicroTileState), stream), "cudaMemsetAsync reset");
}

void GpuPeriodicitySieve::process_batch(CudaRawEvent* d_events, size_t count, cudaStream_t stream, uint64_t* out_retained_count) {
    if (count == 0 || d_events == nullptr) {
        if (out_retained_count) *out_retained_count = 0;
        return;
    }

    ensure_capacity(count);

    int n = static_cast<int>(count);
    unsigned blocks = blocks_for(count);

    // 1. Reset retained counter on device
    cuda_check(cudaMemsetAsync(d_retained_counter_, 0, sizeof(unsigned long long), stream), "memset retained counter");

    // 2. Extract 2x2 micro-tile keys and sequence values
    k_extract_tile_keys<<<blocks, kThreads, 0, stream>>>(
        d_events, n, tile_w_, tile_h_, d_keys_in_, d_values_in_
    );
    cuda_check(cudaGetLastError(), "k_extract_tile_keys launch");

    // 3. Stable Radix Sort by tile key (preserves time ordering within each tile)
    size_t temp_bytes = sort_temp_bytes_;
    cuda_check(cub::DeviceRadixSort::SortPairs(
        d_sort_temp_, temp_bytes,
        d_keys_in_, d_keys_out_,
        d_values_in_, d_values_out_,
        n, 0, kRadixBits, stream
    ), "DeviceRadixSort::SortPairs");

    // 4. Sequential scan per tile run: evaluates periodicity and writes hits to d_events[orig_idx].pad
    k_sieve_scan<<<blocks, kThreads, 0, stream>>>(
        d_events, d_keys_out_, d_values_out_, n,
        tile_w_, tile_h_, static_cast<uint32_t>(num_tiles_),
        min_period_us_, max_period_us_, cfg_.min_consecutive_hits,
        d_tile_state_, d_retained_counter_
    );
    cuda_check(cudaGetLastError(), "k_sieve_scan launch");

    // 5. Download retained count if requested
    if (out_retained_count) {
        cuda_check(cudaMemcpyAsync(h_pinned_retained_, d_retained_counter_, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream), "async retained download");
        cuda_check(cudaStreamSynchronize(stream), "sync stream for retained count");
        *out_retained_count = static_cast<uint64_t>(*h_pinned_retained_);
    }
}

} // namespace predator
