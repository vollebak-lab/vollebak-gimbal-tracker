/**
 * @file raw_pipeline.cu
 * @brief Implementation of the Zero-CPU GPU-Resident Raw Tap Ingestion Pipeline (Phase 33.4b.c).
 */

#include "raw_pipeline.cuh"
#include "evt21_format.hpp"
#include "evt21_encoder.hpp"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <pthread.h>

namespace predator {

namespace {

inline void cuda_check(cudaError_t e, const char* context) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("RawPipeline: ") + context + ": " + cudaGetErrorString(e));
    }
}

constexpr int kThreads = 256;
inline unsigned blocks_for(size_t n) {
    return static_cast<unsigned>((n + kThreads - 1) / kThreads);
}

__global__ void k_accumulate_ui_frame(
    const CudaRawEvent* __restrict__ events,
    int n,
    uint8_t* __restrict__ frame,
    int width, int height)
{
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;

    uint16_t x = events[i].x;
    uint16_t y = events[i].y;
    if (x < width && y < height) {
        frame[y * width + x] = 255;
    }
}

} // anonymous namespace

RawPipeline::RawPipeline(
    const RawPipelineConfig& config,
    CudaFlickerCore& cuda_core,
    ContinuousGyroWarper& gyro_warper)
    : config_(config),
      cuda_core_(cuda_core),
      gyro_warper_(gyro_warper)
{
    // 1. Create dedicated non-blocking CUDA stream
    cuda_check(cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");

    // 2. Timing events
    cuda_check(cudaEventCreate(&ev_start_), "cudaEventCreate ev_start");
    cuda_check(cudaEventCreate(&ev_decode_end_), "cudaEventCreate ev_decode_end");
    cuda_check(cudaEventCreate(&ev_sieve_end_), "cudaEventCreate ev_sieve_end");
    cuda_check(cudaEventCreate(&ev_ingest_end_), "cudaEventCreate ev_ingest_end");

    // 3. Allocate mapped pinned memory ring buffer
    allocate_ring();

    // 4. Instantiate GPU EVT2.1 Decoder & GPU Periodicity Sieve
    decoder_ = std::make_unique<GpuEvt21Decoder>(config_.max_words_per_slot, cuda_stream_);
    GpuSieveConfig sieve_cfg;
    sieve_cfg.sensor_width = config_.sensor_width;
    sieve_cfg.sensor_height = config_.sensor_height;
    sieve_ = std::make_unique<GpuPeriodicitySieve>(sieve_cfg);

    // 5. Pre-allocate decoded events buffer in device memory
    decoded_events_capacity_ = GpuEvt21Decoder::max_events_for(config_.max_words_per_slot);
    cuda_check(cudaMalloc(&d_decoded_events_, decoded_events_capacity_ * sizeof(CudaRawEvent)), "cudaMalloc d_decoded_events");

    // 6. Allocate pinned decoded count pointer
    cuda_check(cudaHostAlloc(&h_pinned_decoded_count_, sizeof(uint32_t), cudaHostAllocMapped), "cudaHostAlloc h_pinned_decoded_count");
    void* d_count_ptr = nullptr;
    cuda_check(cudaHostGetDevicePointer(&d_count_ptr, h_pinned_decoded_count_, 0), "cudaHostGetDevicePointer d_decoded_count");
    d_decoded_count_ = static_cast<uint32_t*>(d_count_ptr);

    // 7. Allocate UI frame buffers (Phase 33.4b.e)
    if (config_.enable_ui_frame_gen) {
        ui_frame_bytes_ = static_cast<size_t>(config_.sensor_width * config_.sensor_height);
        cuda_check(cudaMalloc(&d_ui_frame_, ui_frame_bytes_), "cudaMalloc d_ui_frame");
        cuda_check(cudaMemset(d_ui_frame_, 0, ui_frame_bytes_), "cudaMemset d_ui_frame");
        cuda_check(cudaHostAlloc(&h_pinned_ui_frame_, ui_frame_bytes_, cudaHostAllocDefault), "cudaHostAlloc h_pinned_ui_frame");
        last_ui_frame_time_ = std::chrono::steady_clock::time_point{};
    }
}

RawPipeline::~RawPipeline() {
    stop();

    if (cuda_stream_) {
        cudaStreamSynchronize(cuda_stream_);
    }

    decoder_.reset();
    sieve_.reset();

    if (ev_start_) { cudaEventDestroy(ev_start_); ev_start_ = nullptr; }
    if (ev_decode_end_) { cudaEventDestroy(ev_decode_end_); ev_decode_end_ = nullptr; }
    if (ev_sieve_end_) { cudaEventDestroy(ev_sieve_end_); ev_sieve_end_ = nullptr; }
    if (ev_ingest_end_) { cudaEventDestroy(ev_ingest_end_); ev_ingest_end_ = nullptr; }

    if (d_decoded_events_) { cudaFree(d_decoded_events_); d_decoded_events_ = nullptr; }
    if (h_pinned_decoded_count_) { cudaFreeHost(h_pinned_decoded_count_); h_pinned_decoded_count_ = nullptr; }
    if (d_ui_frame_) { cudaFree(d_ui_frame_); d_ui_frame_ = nullptr; }
    if (h_pinned_ui_frame_) { cudaFreeHost(h_pinned_ui_frame_); h_pinned_ui_frame_ = nullptr; }

    free_ring();

    if (cuda_stream_) {
        cudaStreamDestroy(cuda_stream_);
        cuda_stream_ = nullptr;
    }
}

void RawPipeline::allocate_ring() {
    ring_slots_.resize(config_.ring_slots);
    size_t slot_bytes = config_.max_words_per_slot * sizeof(uint64_t);

    for (size_t i = 0; i < config_.ring_slots; ++i) {
        RawRingSlot& slot = ring_slots_[i];
        cuda_check(cudaHostAlloc(&slot.h_words, slot_bytes, cudaHostAllocMapped), "cudaHostAlloc ring slot");
        void* d_ptr = nullptr;
        cuda_check(cudaHostGetDevicePointer(&d_ptr, slot.h_words, 0), "cudaHostGetDevicePointer ring slot");
        slot.d_words = static_cast<const uint64_t*>(d_ptr);
        slot.word_count = 0;
        slot.min_t = 0;
        slot.max_t = 0;
    }
}

void RawPipeline::free_ring() {
    for (size_t i = 0; i < ring_slots_.size(); ++i) {
        if (ring_slots_[i].h_words) {
            cudaFreeHost(ring_slots_[i].h_words);
            ring_slots_[i].h_words = nullptr;
            ring_slots_[i].d_words = nullptr;
        }
    }
    ring_slots_.clear();
}

void RawPipeline::connect_device(Metavision::Device* device) {
    device_ = device;
    if (device_) {
        hal_stream_ = device_->get_facility<Metavision::I_EventsStream>();
        if (!hal_stream_) {
            throw std::runtime_error("RawPipeline: Device lacks I_EventsStream facility");
        }
    }
}

void RawPipeline::connect_file(const std::string& raw_file_path, bool loop, double playback_rate) {
    is_file_mode_ = true;
    file_path_ = raw_file_path;
    loop_file_ = loop;
    playback_rate_ = playback_rate;
    eof_reached_.store(false);
}

void RawPipeline::start() {
    if (running_.load()) return;
    running_.store(true);
    eof_reached_.store(false);

    if (hal_stream_) {
        hal_stream_->start();
    }

    worker_thread_ = std::thread(&RawPipeline::gpu_worker_thread_func, this);
    if (hal_stream_) {
        reader_thread_ = std::thread(&RawPipeline::hal_reader_thread_func, this);
    } else if (is_file_mode_) {
        file_reader_thread_ = std::thread(&RawPipeline::file_reader_thread_func, this);
    }
}

void RawPipeline::stop() {
    if (!running_.load()) return;
    running_.store(false);

    queue_cv_.notify_all();

    if (hal_stream_) {
        hal_stream_->stop();
    }

    if (reader_thread_.joinable()) {
        reader_thread_.join();
    }

    if (file_reader_thread_.joinable()) {
        file_reader_thread_.join();
    }

    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void RawPipeline::reset() {
    cuda_check(cudaStreamSynchronize(cuda_stream_), "reset sync");
    if (decoder_) decoder_->reset_async();
    if (sieve_) sieve_->reset(cuda_stream_);
    if (d_ui_frame_) cuda_check(cudaMemsetAsync(d_ui_frame_, 0, ui_frame_bytes_, cuda_stream_), "reset ui_frame");
    cuda_check(cudaStreamSynchronize(cuda_stream_), "reset finish");
}

RawPipelineStats RawPipeline::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void RawPipeline::hal_reader_thread_func() {
#if defined(__linux__) && !defined(__ANDROID__)
    pthread_setname_np(pthread_self(), "raw_reader");
#endif

    size_t cur_slot_idx = 0;
    RawRingSlot* cur_slot = &ring_slots_[cur_slot_idx];
    cur_slot->word_count = 0;
    auto last_flush_time = std::chrono::steady_clock::now();

    while (running_.load(std::memory_order_relaxed)) {
        if (hal_stream_->wait_next_buffer() < 0) {
            if (!running_.load(std::memory_order_relaxed)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        Metavision::DataTransfer::BufferPtr buf = hal_stream_->get_latest_raw_data();
        if (!buf || buf.size() == 0) continue;

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.total_usb_buffers++;
        }

        const size_t buf_words = buf.size() / sizeof(uint64_t);
        if (buf_words == 0) continue;

        // Check if appending this buffer would exceed slot capacity
        if (cur_slot->word_count + buf_words > config_.max_words_per_slot) {
            // Flush current slot immediately
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                if (ready_queue_.size() < config_.ring_slots) {
                    ready_queue_.push_back(cur_slot_idx);
                    queue_cv_.notify_one();
                } else {
                    std::lock_guard<std::mutex> slock(stats_mutex_);
                    stats_.ring_overruns++;
                }
            }
            cur_slot_idx = (cur_slot_idx + 1) % config_.ring_slots;
            cur_slot = &ring_slots_[cur_slot_idx];
            cur_slot->word_count = 0;
            last_flush_time = std::chrono::steady_clock::now();
        }

        // Copy raw words into mapped pinned slot
        std::memcpy(cur_slot->h_words + cur_slot->word_count, buf.data(), buf_words * sizeof(uint64_t));
        cur_slot->word_count += buf_words;
        __sync_synchronize();

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.total_raw_words += buf_words;
        }

        // Check coalescing condition
        auto now = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(now - last_flush_time).count();

        if (cur_slot->word_count >= config_.coalesce_words || elapsed_ms >= config_.coalesce_timeout_ms) {
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                if (ready_queue_.size() < config_.ring_slots) {
                    ready_queue_.push_back(cur_slot_idx);
                    queue_cv_.notify_one();
                } else {
                    std::lock_guard<std::mutex> slock(stats_mutex_);
                    stats_.ring_overruns++;
                }
            }
            cur_slot_idx = (cur_slot_idx + 1) % config_.ring_slots;
            cur_slot = &ring_slots_[cur_slot_idx];
            cur_slot->word_count = 0;
            last_flush_time = now;
        }
    }
}

void RawPipeline::file_reader_thread_func() {
#if defined(__linux__) && !defined(__ANDROID__)
    pthread_setname_np(pthread_self(), "raw_file_reader");
#endif

    const bool is_cd = (file_path_.size() >= 3 && file_path_.rfind(".cd") == file_path_.size() - 3);

    std::unique_ptr<predator::evt21::Evt21Encoder> cd_encoder;
    if (is_cd) {
        cd_encoder = std::make_unique<predator::evt21::Evt21Encoder>(config_.sensor_width, config_.sensor_height);
    }

    size_t cur_slot_idx = 0;

    while (running_.load(std::memory_order_relaxed)) {
        std::ifstream file(file_path_, std::ios::binary);
        if (!file.is_open()) {
            std::fprintf(stderr, "[RawPipeline ERROR] Cannot open input file: %s\n", file_path_.c_str());
            eof_reached_.store(true);
            break;
        }

        std::chrono::steady_clock::time_point anchor_wall_time = std::chrono::steady_clock::now();
        uint64_t anchor_ev_time = 0;
        bool have_anchor = false;

        if (is_cd) {
            constexpr size_t kCdChunkSize = 8192;
            std::vector<predator::evt21::CdRecord> cd_records(kCdChunkSize);
            std::vector<uint64_t> encoded_words;
            encoded_words.reserve(kCdChunkSize * 2);

            while (running_.load(std::memory_order_relaxed) && file.good()) {
                file.read(reinterpret_cast<char*>(cd_records.data()), kCdChunkSize * sizeof(predator::evt21::CdRecord));
                size_t records_read = file.gcount() / sizeof(predator::evt21::CdRecord);
                if (records_read == 0) break;

                encoded_words.clear();
                cd_encoder->encode(cd_records.data(), cd_records.data() + records_read, encoded_words);
                if (encoded_words.empty()) continue;

                size_t offset = 0;
                while (offset < encoded_words.size() && running_.load(std::memory_order_relaxed)) {
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        queue_cv_.wait(lock, [&] {
                            return ready_queue_.size() < config_.ring_slots || !running_.load(std::memory_order_relaxed);
                        });
                        if (!running_.load(std::memory_order_relaxed)) break;
                    }

                    RawRingSlot* cur_slot = &ring_slots_[cur_slot_idx];
                    size_t words_to_copy = std::min(encoded_words.size() - offset, config_.max_words_per_slot);
                    std::memcpy(cur_slot->h_words, encoded_words.data() + offset, words_to_copy * sizeof(uint64_t));
                    cur_slot->word_count = words_to_copy;
                    offset += words_to_copy;

                    {
                        std::lock_guard<std::mutex> lock(stats_mutex_);
                        stats_.total_raw_words += words_to_copy;
                        stats_.total_usb_buffers++;
                    }

                    if (playback_rate_ > 0.0) {
                        for (size_t i = 0; i < words_to_copy; ++i) {
                            if (predator::evt21::word_type(cur_slot->h_words[i]) == predator::evt21::kTimeHigh) {
                                uint64_t cur_ev_time = static_cast<uint64_t>(predator::evt21::time_high(cur_slot->h_words[i])) << 6;
                                if (!have_anchor || cur_ev_time < anchor_ev_time) {
                                    anchor_wall_time = std::chrono::steady_clock::now();
                                    anchor_ev_time = cur_ev_time;
                                    have_anchor = true;
                                } else {
                                    double elapsed_ev_s = static_cast<double>(cur_ev_time - anchor_ev_time) * 1e-6;
                                    auto target_wall = anchor_wall_time + std::chrono::duration<double>(elapsed_ev_s / playback_rate_);
                                    if (std::chrono::steady_clock::now() < target_wall) {
                                        std::this_thread::sleep_until(target_wall);
                                    }
                                }
                                break;
                            }
                        }
                    }

                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        ready_queue_.push_back(cur_slot_idx);
                        queue_cv_.notify_one();
                    }
                    cur_slot_idx = (cur_slot_idx + 1) % config_.ring_slots;
                }
            }
        } else {
            const size_t chunk_words = std::min(config_.coalesce_words, config_.max_words_per_slot);

            while (running_.load(std::memory_order_relaxed) && file.good()) {
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    queue_cv_.wait(lock, [&] {
                        return ready_queue_.size() < config_.ring_slots || !running_.load(std::memory_order_relaxed);
                    });
                    if (!running_.load(std::memory_order_relaxed)) break;
                }

                RawRingSlot* cur_slot = &ring_slots_[cur_slot_idx];
                file.read(reinterpret_cast<char*>(cur_slot->h_words), chunk_words * sizeof(uint64_t));
                size_t words_read = file.gcount() / sizeof(uint64_t);
                if (words_read == 0) break;
                cur_slot->word_count = words_read;

                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    stats_.total_raw_words += words_read;
                    stats_.total_usb_buffers++;
                }

                if (playback_rate_ > 0.0) {
                    for (size_t i = 0; i < words_read; ++i) {
                        if (predator::evt21::word_type(cur_slot->h_words[i]) == predator::evt21::kTimeHigh) {
                            uint64_t cur_ev_time = static_cast<uint64_t>(predator::evt21::time_high(cur_slot->h_words[i])) << 6;
                            if (!have_anchor || cur_ev_time < anchor_ev_time) {
                                anchor_wall_time = std::chrono::steady_clock::now();
                                anchor_ev_time = cur_ev_time;
                                have_anchor = true;
                            } else {
                                double elapsed_ev_s = static_cast<double>(cur_ev_time - anchor_ev_time) * 1e-6;
                                auto target_wall = anchor_wall_time + std::chrono::duration<double>(elapsed_ev_s / playback_rate_);
                                if (std::chrono::steady_clock::now() < target_wall) {
                                    std::this_thread::sleep_until(target_wall);
                                }
                            }
                            break;
                        }
                    }
                }

                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    ready_queue_.push_back(cur_slot_idx);
                    queue_cv_.notify_one();
                }
                cur_slot_idx = (cur_slot_idx + 1) % config_.ring_slots;
            }
        }

        if (!loop_file_ || !running_.load(std::memory_order_relaxed)) {
            eof_reached_.store(true);
            std::cout << "[RawPipeline INFO] File replay reached EOF: " << file_path_ << "\n";
            while (running_.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            break;
        }

        // Looping: reset decoder and sieve state for continuous loop
        reset();
    }
}

void RawPipeline::gpu_worker_thread_func() {
#if defined(__linux__) && !defined(__ANDROID__)
    pthread_setname_np(pthread_self(), "gpu_worker");
#endif

    while (running_.load(std::memory_order_relaxed)) {
        size_t slot_idx;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait_for(lock, std::chrono::milliseconds(20), [&] {
                return !ready_queue_.empty() || !running_.load(std::memory_order_relaxed);
            });

            if (!running_.load(std::memory_order_relaxed) && ready_queue_.empty()) break;
            if (ready_queue_.empty()) continue;

            slot_idx = ready_queue_.front();
            ready_queue_.pop_front();
            queue_cv_.notify_one();
        }

        RawRingSlot& slot = ring_slots_[slot_idx];
        if (slot.word_count > 0) {
            execute_gpu_batch(slot.d_words, slot.word_count);
            slot.word_count = 0;
        }
    }
}

void RawPipeline::execute_gpu_batch(const uint64_t* d_words, size_t word_count) {
    if (word_count == 0) return;

    cuda_check(cudaEventRecord(ev_start_, cuda_stream_), "ev_start record");

    // 1. Asynchronous EVT2.1 Decode on GPU
    decoder_->decode_async(d_words, word_count, d_decoded_events_, d_decoded_count_);
    cuda_check(cudaEventRecord(ev_decode_end_, cuda_stream_), "ev_decode_end record");

    // 2. Synchronize decode completion to obtain exact event count
    cuda_check(cudaStreamSynchronize(cuda_stream_), "decode stream sync");
    uint32_t count = *h_pinned_decoded_count_;

    float decode_ms = 0.0f, sieve_ms = 0.0f, ingest_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&decode_ms, ev_start_, ev_decode_end_), "decode elapsed");

    if (count > 0) {
        total_decoded_events_.fetch_add(count, std::memory_order_relaxed);

        // Read earliest and latest event timestamps for time-anchoring and window advance
        CudaRawEvent first_ev, last_ev;
        cuda_check(cudaMemcpyAsync(&first_ev, &d_decoded_events_[0], sizeof(CudaRawEvent), cudaMemcpyDeviceToHost, cuda_stream_), "first_ev copy");
        cuda_check(cudaMemcpyAsync(&last_ev, &d_decoded_events_[count - 1], sizeof(CudaRawEvent), cudaMemcpyDeviceToHost, cuda_stream_), "last_ev copy");
        cuda_check(cudaStreamSynchronize(cuda_stream_), "ev sync");

        uint64_t min_t = first_ev.t;
        uint64_t max_t = last_ev.t;

        // Update gyro warper anchor with camera time
        uint64_t host_now_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        gyro_warper_.update_camera_time_anchor(min_t, host_now_us);

        // Compute batch homography
        Matrix3x3 batch_H = Matrix3x3::identity();
        if (config_.enable_ego_warp) {
            uint64_t mid_t = (min_t + max_t) / 2;
            batch_H = gyro_warper_.compute_homography(min_t, mid_t);
        }

        // 3. In-place Order-Correct GPU Periodicity Sieve (Phase 33.4b.d)
        cuda_check(cudaEventRecord(ev_decode_end_, cuda_stream_), "sieve start record");
        uint64_t retained_count = 0;
        sieve_->process_batch(d_decoded_events_, count, cuda_stream_, &retained_count);
        total_retained_events_.fetch_add(retained_count, std::memory_order_relaxed);
        cuda_check(cudaEventRecord(ev_sieve_end_, cuda_stream_), "sieve end record");

        // 4. Ingest into cuFFT Ring Buffers on GPU (Phase 33.4b.c)
        cuda_core_.ingest_device_events(d_decoded_events_, count, batch_H, min_t, max_t, cuda_stream_);
        cuda_check(cudaEventRecord(ev_ingest_end_, cuda_stream_), "ingest end record");

        // 5. GPU UI Display Frame Accumulation (Phase 33.4b.e)
        if (config_.enable_ui_frame_gen && d_ui_frame_) {
            unsigned blocks = blocks_for(count);
            k_accumulate_ui_frame<<<blocks, kThreads, 0, cuda_stream_>>>(
                d_decoded_events_, static_cast<int>(count), d_ui_frame_,
                config_.sensor_width, config_.sensor_height
            );

            auto now = std::chrono::steady_clock::now();
            double elapsed_ui_ms = std::chrono::duration<double, std::milli>(now - last_ui_frame_time_).count();
            double target_frame_ms = 1000.0 / config_.ui_fps;

            if (elapsed_ui_ms >= target_frame_ms && frame_callback_) {
                // Copy device frame to pinned host memory
                cuda_check(cudaMemcpyAsync(h_pinned_ui_frame_, d_ui_frame_, ui_frame_bytes_, cudaMemcpyDeviceToHost, cuda_stream_), "ui frame d2h");
                // Clear device accumulator for next frame
                cuda_check(cudaMemsetAsync(d_ui_frame_, 0, ui_frame_bytes_, cuda_stream_), "ui frame clear");
                cuda_check(cudaStreamSynchronize(cuda_stream_), "ui frame sync");

                frame_callback_(h_pinned_ui_frame_, config_.sensor_width, config_.sensor_height, max_t);
                last_ui_frame_time_ = now;
            }
        }

        cuda_check(cudaStreamSynchronize(cuda_stream_), "batch completion sync");
        cuda_check(cudaEventElapsedTime(&sieve_ms, ev_decode_end_, ev_sieve_end_), "sieve elapsed");
        cuda_check(cudaEventElapsedTime(&ingest_ms, ev_sieve_end_, ev_ingest_end_), "ingest elapsed");

        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.total_decoded_events += count;
        stats_.total_retained_events += retained_count;
        stats_.last_batch_words = word_count;
        stats_.last_batch_events = count;
        stats_.last_gpu_decode_us = decode_ms * 1000.0;
        stats_.last_gpu_sieve_us = sieve_ms * 1000.0;
        stats_.last_gpu_ingest_us = ingest_ms * 1000.0;
    }
}

void RawPipeline::process_raw_words_sync(const uint64_t* words, size_t word_count) {
    if (word_count == 0 || words == nullptr) return;

    size_t offset = 0;
    while (offset < word_count) {
        size_t chunk = std::min(word_count - offset, config_.max_words_per_slot);
        std::memcpy(ring_slots_[0].h_words, words + offset, chunk * sizeof(uint64_t));
        __sync_synchronize();
        execute_gpu_batch(ring_slots_[0].d_words, chunk);
        offset += chunk;
    }
}

} // namespace predator
