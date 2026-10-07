#pragma once
/**
 * @file gpu_sieve.cuh
 * @brief Order-correct GPU Micro-Neighborhood Periodicity Sieve (Phase 33.4b.d).
 *
 * Implements the 2x2 micro-tile Surface of Active Events (SAE) periodicity sieve
 * directly on device memory, achieving exact parity with the CPU MicroNeighborhoodPeriodicitySieve.
 *
 * Algorithmic stages per batch:
 *   1. Tile key extraction: maps each event (x, y) to its 2x2 micro-tile index (tx = x>>1, ty = y>>1).
 *   2. Stable radix sort: CUB DeviceRadixSort::SortPairs sorts keys while preserving exact temporal order
 *      among events mapped to the same micro-tile.
 *   3. Sequential per-tile scan: run-start threads scan contiguous tile runs in time order, evaluating
 *      intra-burst and inter-sweep periodic recurrence and updating persistent per-tile state.
 *   4. Results are written directly into CudaRawEvent::pad in device memory.
 */

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include "gpu_event.hpp"

namespace predator {

/**
 * @brief Persistent state tracked per 2x2 micro-tile in device memory.
 * Exact 8-byte layout allowing single 64-bit memory transactions.
 */
struct alignas(8) MicroTileState {
    uint32_t last_timestamp_us{0}; ///< Microsecond timestamp of last qualifying event
    uint16_t last_dt_us{0};        ///< Last valid inter-sweep period (clamped to 65535 us)
    uint8_t consecutive_hits{0};   ///< Consecutive periodic recurrence counter
    uint8_t pad{0};                ///< Alignment padding
};
static_assert(sizeof(MicroTileState) == 8, "MicroTileState must stay 8 bytes");

/**
 * @brief Configuration parameters for the GPU periodicity sieve.
 */
struct GpuSieveConfig {
    int sensor_width{1280};
    int sensor_height{720};
    double min_freq_hz{70.0};
    double max_freq_hz{800.0};
    uint8_t min_consecutive_hits{2};
};

/**
 * @brief GPU Periodicity Sieve engine maintaining persistent state across batches.
 */
class GpuPeriodicitySieve {
public:
    explicit GpuPeriodicitySieve(const GpuSieveConfig& cfg = GpuSieveConfig{});
    ~GpuPeriodicitySieve();

    // Disable copy
    GpuPeriodicitySieve(const GpuPeriodicitySieve&) = delete;
    GpuPeriodicitySieve& operator=(const GpuPeriodicitySieve&) = delete;

    // Allow move
    GpuPeriodicitySieve(GpuPeriodicitySieve&& other) noexcept;
    GpuPeriodicitySieve& operator=(GpuPeriodicitySieve&& other) noexcept;

    /**
     * @brief In-place periodicity evaluation on device events.
     *
     * @param d_events Pointer to device array of CudaRawEvent.
     * @param count Number of events in the batch.
     * @param stream CUDA stream for asynchronous execution.
     * @param out_retained_count Optional host pointer to receive the count of retained (hits >= min) events.
     *                           If non-null, stream is synchronized before returning.
     */
    void process_batch(CudaRawEvent* d_events, size_t count, cudaStream_t stream = nullptr, uint64_t* out_retained_count = nullptr);

    /**
     * @brief Resets all persistent tile states to zero.
     */
    void reset(cudaStream_t stream = nullptr);

    int tile_width() const { return tile_w_; }
    int tile_height() const { return tile_h_; }
    size_t num_tiles() const { return num_tiles_; }
    uint32_t min_period_us() const { return min_period_us_; }
    uint32_t max_period_us() const { return max_period_us_; }

    const MicroTileState* device_tile_states() const { return d_tile_state_; }

private:
    void ensure_capacity(size_t count);
    void free_resources();

    GpuSieveConfig cfg_;
    int tile_w_{0};
    int tile_h_{0};
    size_t num_tiles_{0};
    uint32_t min_period_us_{0};
    uint32_t max_period_us_{0};

    // Persistent tile state buffer in device memory
    MicroTileState* d_tile_state_{nullptr};

    // Work buffers for CUB radix sort
    size_t capacity_{0};
    uint32_t* d_keys_in_{nullptr};
    uint32_t* d_keys_out_{nullptr};
    uint32_t* d_values_in_{nullptr};
    uint32_t* d_values_out_{nullptr};

    void* d_sort_temp_{nullptr};
    size_t sort_temp_bytes_{0};

    unsigned long long* d_retained_counter_{nullptr};
    unsigned long long* h_pinned_retained_{nullptr};
};

} // namespace predator
