#pragma once
/**
 * @file evt21_gpu_decoder.cuh
 * @brief Deterministic, fully asynchronous EVT 2.1 (IMX636 legacy word order) decoder on the GPU.
 *
 * Phase 33.4b.b. Replaces OpenEB's CPU decode in the hot path: the CPU only hands over a pointer to raw
 * USB words that the GPU can read (device memory or mapped pinned host memory on Jetson); decoding,
 * and later sieving and accumulation, never touch the CPU per event.
 *
 * Exactness: output is bit-identical, in order, to OpenEB 5.2 `EVT21LegacyDecoder` with time shifting
 * disabled (see evt21_format.hpp for the semantics), including dropping words before the first
 * TIME_HIGH of the stream, exact-wrap loop counting and backward TIME_HIGH discrepancies.
 *
 * Algorithm per batch of n words (all on one CUDA stream, no host synchronization):
 *   1. th_key[i] = i if word i is TIME_HIGH else -1           -> inclusive max-scan -> last_th[i]
 *   2. per word: count = decodable CD ? popcount(mask) : 0;     wrap = exact 34-bit wrap at TIME_HIGH i
 *      packed (wrap << 32 | count)                              -> inclusive sum-scan
 *   3. per CD word: t = (loop << 34) | (high << 6) | ts6; write events at the scanned offset, ascending bit
 *   4. one thread: publish the batch event count and advance the carried decoder state
 * Carried state (base-time flag, last TIME_HIGH, loop) lives in device memory, so consecutive batches are
 * decoded exactly as one concatenated stream. EVT 2.1 has no multi-word CD events, so any split is valid.
 */

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "gpu_event.hpp"

namespace predator {

/// Device-resident decoder state carried across batches (one concatenated stream).
struct Evt21DecoderState {
    uint32_t base_time_set;  ///< 1 once a TIME_HIGH has been seen (earlier words are dropped, as in OpenEB)
    uint32_t high;           ///< last TIME_HIGH value, ts[33:6]
    uint64_t loop;           ///< completed 2^34 us counter loops
    uint64_t total_events;   ///< cumulative decoded CD events (telemetry)
};

class GpuEvt21Decoder {
public:
    /**
     * @param max_words_per_batch upper bound on words passed to one decode_async() call
     * @param stream CUDA stream all work is enqueued on (not owned)
     * @throws std::runtime_error on allocation failure
     */
    GpuEvt21Decoder(size_t max_words_per_batch, cudaStream_t stream);
    ~GpuEvt21Decoder();
    GpuEvt21Decoder(const GpuEvt21Decoder&)            = delete;
    GpuEvt21Decoder& operator=(const GpuEvt21Decoder&) = delete;

    /// Worst-case decoded events for a batch of @p words (every word a full 32-pixel CD vector).
    static constexpr size_t max_events_for(size_t words) { return 32 * words; }

    /**
     * @brief Enqueues decoding of @p n words. Returns immediately.
     * @param d_words device-accessible pointer (device memory, or mapped pinned host memory)
     * @param n number of 64-bit words, n <= max_words_per_batch
     * @param d_out device buffer with capacity >= max_events_for(n)
     * @param d_out_count device pointer receiving the number of events written for this batch
     * @throws std::invalid_argument if n exceeds the configured maximum
     */
    void decode_async(const uint64_t* d_words, size_t n, CudaRawEvent* d_out, uint32_t* d_out_count);

    /// Enqueues a state reset (stream restart): the next words before a TIME_HIGH are dropped again.
    void reset_async();

    /// Synchronously reads the carried state (test/telemetry use; synchronizes the stream).
    Evt21DecoderState read_state() const;

    size_t max_words_per_batch() const { return max_words_; }

private:
    size_t max_words_;
    cudaStream_t stream_;
    int32_t* d_th_key_         = nullptr;
    int32_t* d_last_th_        = nullptr;
    uint64_t* d_count_wrap_    = nullptr;
    uint64_t* d_count_wrap_incl_ = nullptr;
    void* d_scan_temp_         = nullptr;
    size_t scan_temp_bytes_    = 0;
    Evt21DecoderState* d_state_ = nullptr;
};

}  // namespace predator
