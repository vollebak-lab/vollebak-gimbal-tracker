#ifndef PREDATOR_EVENT_SUPPRESSION_TRT_HPP
#define PREDATOR_EVENT_SUPPRESSION_TRT_HPP

#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <iostream>
#include <mutex>
#include <atomic>
#include <cmath>
#include <algorithm>
#include <cstring>

#include <cuda_runtime_api.h>
#include <NvInfer.h>

namespace predator {

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT " << (severity == Severity::kINTERNAL_ERROR ? "FATAL" :
                                          severity == Severity::kERROR ? "ERROR" : "WARN")
                      << "] " << msg << std::endl;
        }
    }
};

static TrtLogger g_trt_logger;

/**
 * @brief Accumulates raw/stabilized asynchronous events into a 2-bin temporal stack tensor (2 x H x W)
 */
class TemporalEventStackAccumulator {
public:
    TemporalEventStackAccumulator(int width = 640, int height = 360, uint64_t window_duration_us = 40000)
        : width_(width),
          height_(height),
          window_duration_us_(window_duration_us),
          num_pixels_(width * height),
          window_start_us_(0) {
        
        buffer_active_.assign(2 * num_pixels_, 0.0f);
        buffer_ready_.assign(2 * num_pixels_, 0.0f);
    }

    void ingest_event(double x, double y, uint64_t timestamp_us, short polarity) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (window_start_us_ == 0) {
            window_start_us_ = timestamp_us;
        }

        // Check if window period has elapsed (40ms => 25 Hz inference cycle)
        if (timestamp_us >= window_start_us_ + window_duration_us_) {
            std::swap(buffer_active_, buffer_ready_);
            std::fill(buffer_active_.begin(), buffer_active_.end(), 0.0f);
            has_new_frame_ = true;
            window_start_us_ = timestamp_us;
        }

        int px = std::clamp(static_cast<int>(x * (static_cast<double>(width_) / 1280.0)), 0, width_ - 1);
        int py = std::clamp(static_cast<int>(y * (static_cast<double>(height_) / 720.0)), 0, height_ - 1);

        int channel = (polarity > 0) ? 0 : 1; // Channel 0: ON (+), Channel 1: OFF (-)
        size_t idx = channel * num_pixels_ + (py * width_ + px);
        buffer_active_[idx] += 1.0f;
    }

    bool get_latest_stack(std::vector<float>& out_stack, bool force = false) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (has_new_frame_ || force) {
            out_stack = buffer_ready_;
            has_new_frame_ = false;
            return true;
        }
        return false;
    }

    int width() const { return width_; }
    int height() const { return height_; }

private:
    int width_;
    int height_;
    uint64_t window_duration_us_;
    size_t num_pixels_;
    uint64_t window_start_us_;
    bool has_new_frame_{false};

    std::mutex mutex_;
    std::vector<float> buffer_active_;
    std::vector<float> buffer_ready_;
};

/**
 * @brief Anticipatory Dynamic Event Suppression Engine using TensorRT FP16
 * 
 * Implements UZH RSS 2026 Anticipatory Motion Suppression to filter ego-motion background events
 */
class AnticipatorySuppressionEngine {
public:
    AnticipatorySuppressionEngine(int width = 640, int height = 360)
        : width_(width),
          height_(height),
          num_pixels_(width * height) {
        
        dynamic_mask_host_.assign(num_pixels_, 1.0f); // Default 1.0 (pass-through)
    }

    ~AnticipatorySuppressionEngine() {
        cleanup();
    }

    bool load_engine(const std::string& engine_path) {
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good()) {
            std::cout << "[WARN] TensorRT engine not found at: " << engine_path << " (Running pass-through mode)\n";
            is_ready_ = false;
            return false;
        }

        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        file.close();

        runtime_ = nvinfer1::createInferRuntime(g_trt_logger);
        if (!runtime_) {
            std::cerr << "[ERROR] Failed to create TensorRT Runtime.\n";
            return false;
        }

        engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
        if (!engine_) {
            std::cerr << "[ERROR] Failed to deserialize CUDA engine.\n";
            return false;
        }

        context_ = engine_->createExecutionContext();
        if (!context_) {
            std::cerr << "[ERROR] Failed to create TensorRT Execution Context.\n";
            return false;
        }

        cudaStreamCreate(&stream_);

        // Allocate CUDA device buffers
        size_t events_size = 1 * 2 * height_ * width_ * sizeof(float);
        size_t dt_size = 1 * 1 * sizeof(float);
        size_t mask_size = 1 * 1 * height_ * width_ * sizeof(float);
        size_t flow_size = 1 * 2 * height_ * width_ * sizeof(float);

        cudaMalloc(&d_events_, events_size);
        cudaMalloc(&d_dt_, dt_size);
        cudaMalloc(&d_warped_mask_, mask_size);
        cudaMalloc(&d_optical_flow_, flow_size);

        // Bind input/output tensor buffers
        context_->setTensorAddress("events_tensor", d_events_);
        context_->setTensorAddress("forecast_dt_ms", d_dt_);
        context_->setTensorAddress("warped_mask", d_warped_mask_);
        context_->setTensorAddress("optical_flow", d_optical_flow_);

        is_ready_ = true;
        std::cout << "[INFO] TensorRT Anticipatory Motion Suppression Engine successfully loaded!\n";
        return true;
    }

    /**
     * @brief Executes TensorRT inference on the latest 2-bin event stack
     * @param events_stack Vector of 2 * H * W event counts
     * @param forecast_dt_ms Lookahead time in milliseconds (e.g. 40.0 ms)
     */
    bool infer(const std::vector<float>& events_stack, float forecast_dt_ms = 40.0f) {
        if (!is_ready_) {
            return false;
        }

        size_t events_size = 1 * 2 * height_ * width_ * sizeof(float);
        size_t dt_size = 1 * 1 * sizeof(float);
        size_t mask_size = 1 * 1 * height_ * width_ * sizeof(float);

        // 1. Copy inputs to GPU
        cudaMemcpyAsync(d_events_, events_stack.data(), events_size, cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_dt_, &forecast_dt_ms, dt_size, cudaMemcpyHostToDevice, stream_);

        // 2. Enqueue asynchronous inference
        if (!context_->enqueueV3(stream_)) {
            std::cerr << "[ERROR] TensorRT enqueueV3 execution failed.\n";
            return false;
        }

        // 3. Read back warped dynamic mask
        std::vector<float> mask_result(num_pixels_);
        cudaMemcpyAsync(mask_result.data(), d_warped_mask_, mask_size, cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        {
            std::lock_guard<std::mutex> lock(mask_mutex_);
            dynamic_mask_host_ = std::move(mask_result);
        }

        return true;
    }

    /**
     * @brief Evaluates whether an incoming event belongs to an independently moving object (IMO)
     * @param x Event pixel X in 1280x720 coordinates
     * @param y Event pixel Y in 1280x720 coordinates
     * @param threshold Gating threshold (default 0.35)
     * @return true if event should be RETAINED, false if SUPPRESSED (ego-motion)
     */
    bool is_event_retained(double x, double y, float threshold = 0.35f) const {
        if (!is_ready_) {
            return true; // Pass-through when engine is not active
        }

        int px = std::clamp(static_cast<int>(x * (static_cast<double>(width_) / 1280.0)), 0, width_ - 1);
        int py = std::clamp(static_cast<int>(y * (static_cast<double>(height_) / 720.0)), 0, height_ - 1);

        std::lock_guard<std::mutex> lock(mask_mutex_);
        float prob = dynamic_mask_host_[py * width_ + px];
        return prob >= threshold;
    }

    bool is_ready() const { return is_ready_; }

private:
    void cleanup() {
        if (d_events_) cudaFree(d_events_);
        if (d_dt_) cudaFree(d_dt_);
        if (d_warped_mask_) cudaFree(d_warped_mask_);
        if (d_optical_flow_) cudaFree(d_optical_flow_);
        if (stream_) cudaStreamDestroy(stream_);
        if (context_) delete context_;
        if (engine_) delete engine_;
        if (runtime_) delete runtime_;

        d_events_ = nullptr;
        d_dt_ = nullptr;
        d_warped_mask_ = nullptr;
        d_optical_flow_ = nullptr;
        context_ = nullptr;
        engine_ = nullptr;
        runtime_ = nullptr;
        is_ready_ = false;
    }

    int width_;
    int height_;
    size_t num_pixels_;
    std::atomic<bool> is_ready_{false};

    nvinfer1::IRuntime* runtime_{nullptr};
    nvinfer1::ICudaEngine* engine_{nullptr};
    nvinfer1::IExecutionContext* context_{nullptr};
    cudaStream_t stream_{nullptr};

    void* d_events_{nullptr};
    void* d_dt_{nullptr};
    void* d_warped_mask_{nullptr};
    void* d_optical_flow_{nullptr};

    mutable std::mutex mask_mutex_;
    std::vector<float> dynamic_mask_host_;
};

} // namespace predator

#endif // PREDATOR_EVENT_SUPPRESSION_TRT_HPP
