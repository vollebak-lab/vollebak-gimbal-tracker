#pragma once
/**
 * @file raw_pipeline.cuh
 * @brief Zero-CPU GPU-resident Raw Tap Ingestion Pipeline (Phase 33.4b.c).
 *
 * Replaces the CPU OpenEB CD event callback with direct HAL raw EVT2.1 buffer ingestion:
 *   1. Hardware USB streaming: I_EventsStream::wait_next_buffer() / get_latest_raw_data().
 *   2. Lock-free mapped pinned host memory ring buffer (cudaHostAllocMapped).
 *   3. Dynamic temporal coalescing (~2.0 ms or N words per GPU batch).
 *   4. Single-stream GPU execution: GpuEvt21Decoder -> GpuPeriodicitySieve -> CudaFlickerCore.
 *   5. GPU-resident UI display frame accumulation (Phase 33.4b.e).
 */

#include <cstdint>
#include <cstddef>
#include <memory>
#include <thread>
#include <atomic>
#include <vector>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <functional>

#include <cuda_runtime.h>
#include <metavision/hal/device/device.h>
#include <metavision/hal/facilities/i_events_stream.h>

#include "evt21_gpu_decoder.cuh"
#include "gpu_sieve.cuh"
#include "cuda_flicker_core.cuh"
#include "ego_motion.hpp"

namespace predator {

/**
 * @brief Configuration for the Raw Tap Pipeline.
 */
struct RawPipelineConfig {
    int sensor_width{1280};
    int sensor_height{720};
    size_t ring_slots{16};                 ///< Number of ring buffer slots
    size_t max_words_per_slot{32768};      ///< Maximum 64-bit words per slot (256 KB)
    size_t coalesce_words{16384};          ///< Target word threshold for coalescing
    double coalesce_timeout_ms{2.0};       ///< Maximum wait time before forcing a flush (ms)
    bool enable_ego_warp{false};           ///< Whether continuous gyro warper is active
    bool enable_ui_frame_gen{true};        ///< Whether to generate GPU UI display frames
    double ui_fps{30.0};                   ///< Target UI frame rate
};

/**
 * @brief Telemetry metrics for raw pipeline monitoring.
 */
struct RawPipelineStats {
    uint64_t total_usb_buffers{0};
    uint64_t total_raw_words{0};
    uint64_t total_decoded_events{0};
    uint64_t total_retained_events{0};
    uint64_t dropped_buffers{0};
    uint64_t ring_overruns{0};
    size_t last_batch_words{0};
    uint32_t last_batch_events{0};
    double last_gpu_decode_us{0.0};
    double last_gpu_sieve_us{0.0};
    double last_gpu_ingest_us{0.0};
    double last_batch_interval_us{0.0};
};

/**
 * @brief Mapped pinned memory ring buffer slot for raw 64-bit EVT2.1 words.
 */
struct RawRingSlot {
    uint64_t* h_words{nullptr};            ///< Mapped pinned host pointer
    const uint64_t* d_words{nullptr};      ///< Device pointer directly accessible by GPU
    size_t word_count{0};                  ///< Number of words stored in this slot
    uint64_t min_t{0};                     ///< Earliest timestamp (us)
    uint64_t max_t{0};                     ///< Latest timestamp (us)
    std::chrono::steady_clock::time_point first_buf_time;
};

/**
 * @brief Raw Tap Ingestion Engine managing HAL streaming and GPU execution.
 */
class RawPipeline {
public:
    RawPipeline(const RawPipelineConfig& config,
                CudaFlickerCore& cuda_core,
                ContinuousGyroWarper& gyro_warper);
    ~RawPipeline();

    // Non-copyable
    RawPipeline(const RawPipeline&) = delete;
    RawPipeline& operator=(const RawPipeline&) = delete;

    /**
     * @brief Connects to an opened Metavision HAL device and prepares streaming.
     */
    void connect_device(Metavision::Device* device);

    /**
     * @brief Starts the background HAL reader and GPU pipeline worker threads.
     */
    void start();

    /**
     * @brief Stops all streaming threads gracefully.
     */
    void stop();

    bool is_running() const { return running_.load(std::memory_order_relaxed); }

    /**
     * @brief Callback function type for delivering newly synthesized UI display frames.
     * Arguments: (uint8_t* frame_gray_data, int width, int height, uint64_t timestamp_us)
     */
    using FrameCallback = std::function<void(const uint8_t*, int, int, uint64_t)>;
    void set_frame_callback(FrameCallback cb) { frame_callback_ = cb; }

    /**
     * @brief Synchronously feeds raw words directly into the GPU pipeline (for testing / playback).
     */
    void process_raw_words_sync(const uint64_t* words, size_t word_count);

    /**
     * @brief Retrieves latest telemetry snapshot.
     */
    RawPipelineStats get_stats() const;

    /**
     * @brief Returns the total raw events decoded so far.
     */
    uint64_t total_decoded_events() const { return total_decoded_events_.load(std::memory_order_relaxed); }

    /**
     * @brief Returns the total periodic events retained by the sieve so far.
     */
    uint64_t total_retained_events() const { return total_retained_events_.load(std::memory_order_relaxed); }

    /**
     * @brief Resets decoder and sieve state.
     */
    void reset();

private:
    void allocate_ring();
    void free_ring();

    void hal_reader_thread_func();
    void gpu_worker_thread_func();

    void execute_gpu_batch(const uint64_t* d_words, size_t word_count);

    RawPipelineConfig config_;
    CudaFlickerCore& cuda_core_;
    ContinuousGyroWarper& gyro_warper_;

    Metavision::Device* device_{nullptr};
    Metavision::I_EventsStream* hal_stream_{nullptr};

    std::atomic<bool> running_{false};
    std::thread reader_thread_;
    std::thread worker_thread_;

    // Ring buffer state
    std::vector<RawRingSlot> ring_slots_;
    std::deque<size_t> ready_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // GPU resources
    cudaStream_t cuda_stream_{nullptr};
    std::unique_ptr<GpuEvt21Decoder> decoder_;
    std::unique_ptr<GpuPeriodicitySieve> sieve_;

    // Decode output buffers in device memory
    CudaRawEvent* d_decoded_events_{nullptr};
    size_t decoded_events_capacity_{0};
    uint32_t* d_decoded_count_{nullptr};
    uint32_t* h_pinned_decoded_count_{nullptr};

    // UI Frame Accumulator (Phase 33.4b.e)
    uint8_t* d_ui_frame_{nullptr};
    uint8_t* h_pinned_ui_frame_{nullptr};
    size_t ui_frame_bytes_{0};
    std::chrono::steady_clock::time_point last_ui_frame_time_;
    FrameCallback frame_callback_{nullptr};

    // Diagnostics & telemetry
    mutable std::mutex stats_mutex_;
    RawPipelineStats stats_;
    std::atomic<uint64_t> total_decoded_events_{0};
    std::atomic<uint64_t> total_retained_events_{0};

    // Timing events
    cudaEvent_t ev_start_{nullptr};
    cudaEvent_t ev_decode_end_{nullptr};
    cudaEvent_t ev_sieve_end_{nullptr};
    cudaEvent_t ev_ingest_end_{nullptr};
};

} // namespace predator
