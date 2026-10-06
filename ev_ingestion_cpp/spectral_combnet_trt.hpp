#ifndef PREDATOR_SPECTRAL_COMBNET_TRT_HPP
#define PREDATOR_SPECTRAL_COMBNET_TRT_HPP

#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cuda_runtime_api.h>
#include <NvInfer.h>

namespace predator {

struct SpectralPrediction {
    int cell_idx{0};
    float drone_prob{0.0f};
    float fund_freq_hz{0.0f};
    float harmonic_purity{0.0f};
};

class SpectralCombNetLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR) {
            std::cerr << "[TensorRT SpectralCombNet] " << msg << std::endl;
        }
    }
};

static SpectralCombNetLogger g_spectral_trt_logger;

class SpectralCombNetEngine {
public:
    SpectralCombNetEngine(size_t max_batch = 128)
        : max_batch_(max_batch) {}

    ~SpectralCombNetEngine() {
        cleanup();
    }

    bool load_engine(const std::string& engine_path) {
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[WARN] SpectralCombNet engine not found at: " << engine_path << "\n";
            return false;
        }

        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        file.close();

        runtime_.reset(nvinfer1::createInferRuntime(g_spectral_trt_logger));
        if (!runtime_) {
            std::cerr << "[ERROR] Failed to create TensorRT runtime for SpectralCombNet.\n";
            return false;
        }

        engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
        if (!engine_) {
            std::cerr << "[ERROR] Failed to deserialize TensorRT engine: " << engine_path << "\n";
            return false;
        }

        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            std::cerr << "[ERROR] Failed to create ExecutionContext for SpectralCombNet.\n";
            return false;
        }

        cudaStreamCreate(&stream_);

        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* tname = engine_->getIOTensorName(i);
            auto mode = engine_->getTensorIOMode(tname);
            auto dims = engine_->getTensorShape(tname);
            std::cout << "  [TRT IO] " << i << ": name=" << tname 
                      << " mode=" << (mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT") << " dims=[";
            for (int d = 0; d < dims.nbDims; ++d) std::cout << dims.d[d] << (d + 1 < dims.nbDims ? "x" : "");
            std::cout << "]\n";
        }

        // Allocate device memory for max batch
        size_t input_bytes = max_batch_ * 1 * 257 * sizeof(float);
        size_t prob_bytes = max_batch_ * 1 * sizeof(float);
        size_t freq_bytes = max_batch_ * 1 * sizeof(float);
        size_t purity_bytes = max_batch_ * 1 * sizeof(float);

        cudaMalloc(reinterpret_cast<void**>(&d_input_), input_bytes);
        cudaMalloc(reinterpret_cast<void**>(&d_prob_), prob_bytes);
        cudaMalloc(reinterpret_cast<void**>(&d_freq_), freq_bytes);
        cudaMalloc(reinterpret_cast<void**>(&d_purity_), purity_bytes);

        // Allocate host pinned memory
        cudaMallocHost(reinterpret_cast<void**>(&h_input_), input_bytes);
        cudaMallocHost(reinterpret_cast<void**>(&h_prob_), prob_bytes);
        cudaMallocHost(reinterpret_cast<void**>(&h_freq_), freq_bytes);
        cudaMallocHost(reinterpret_cast<void**>(&h_purity_), purity_bytes);

        is_ready_ = true;
        std::cout << "[INFO] TensorRT SpectralCombNet FP16 Engine active (Max Batch: " 
                  << max_batch_ << ", 257 cuFFT bins, 1D Dilated CombNet).\n";
        return true;
    }

    /**
     * @brief Executes batched neural classification on a set of active candidate spectra
     * @param spectra Host or device array of (batch_size x 257) normalized spectra
     * @param cell_indices Vector of corresponding cell indices (length = batch_size)
     * @param[out] out_predictions Vector of predictions passing neural detection threshold
     * @param min_prob Minimum drone probability threshold (e.g. 0.50)
     */
    bool infer_spectra(const std::vector<const float*>& cell_spectra_ptrs,
                       const std::vector<int>& cell_indices,
                       std::vector<SpectralPrediction>& out_predictions,
                       float min_prob = 0.50f) {
        if (!is_ready_ || cell_spectra_ptrs.empty()) {
            return false;
        }

        size_t batch_size = std::min(cell_spectra_ptrs.size(), max_batch_);

        // Copy spectra into pinned input buffer
        for (size_t b = 0; b < batch_size; ++b) {
            std::memcpy(h_input_ + b * 257, cell_spectra_ptrs[b], 257 * sizeof(float));
        }

        // Copy to GPU
        cudaMemcpyAsync(d_input_, h_input_, batch_size * 257 * sizeof(float), cudaMemcpyHostToDevice, stream_);

        // Set dynamic batch shape in TensorRT
        nvinfer1::Dims3 input_dims{static_cast<int32_t>(batch_size), 1, 257};
        context_->setInputShape("spectrum_in", input_dims);

        context_->setTensorAddress("spectrum_in", d_input_);
        context_->setTensorAddress("drone_prob", d_prob_);
        context_->setTensorAddress("fund_freq_hz", d_freq_);
        context_->setTensorAddress("harmonic_purity", d_purity_);

        if (!context_->enqueueV3(stream_)) {
            std::cerr << "[ERROR] SpectralCombNet enqueueV3 failed.\n";
            return false;
        }

        // Copy outputs back to host
        cudaMemcpyAsync(h_prob_, d_prob_, batch_size * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(h_freq_, d_freq_, batch_size * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(h_purity_, d_purity_, batch_size * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        out_predictions.clear();
        for (size_t b = 0; b < batch_size; ++b) {
            float prob = h_prob_[b];
            if (prob >= min_prob) {
                SpectralPrediction pred;
                pred.cell_idx = cell_indices[b];
                pred.drone_prob = prob;
                pred.harmonic_purity = h_purity_[b];

                // Derive exact physical fundamental frequency from the 257-bin spectrum
                const float* spec = cell_spectra_ptrs[b];
                int best_bin = 0;
                float max_val = 0.0f;
                // Search bins 9 to 102 (70 Hz to 800 Hz)
                for (int k = 9; k <= 102; ++k) {
                    if (spec[k] > max_val) {
                        max_val = spec[k];
                        best_bin = k;
                    }
                }

                // Subharmonic Fundamental Disambiguation (check if best_bin is 2x or 3x harmonic)
                for (int sub = 3; sub >= 2; --sub) {
                    int cand_sub = static_cast<int>(std::round(static_cast<float>(best_bin) / sub));
                    if (cand_sub >= 9 && cand_sub <= 102) {
                        if (spec[cand_sub] >= 0.40f * max_val) {
                            best_bin = cand_sub;
                            break;
                        }
                    }
                }

                float f_hz = best_bin * (4000.0f / 512.0f);
                if (best_bin > 0 && best_bin < 256) {
                    float y1 = spec[best_bin - 1];
                    float y2 = spec[best_bin];
                    float y3 = spec[best_bin + 1];
                    float denom = 2.0f * (2.0f * y2 - y1 - y3);
                    if (std::abs(denom) > 1e-6f) {
                        float delta = (y3 - y1) / denom;
                        delta = std::clamp(delta, -0.5f, 0.5f);
                        f_hz = (best_bin + delta) * (4000.0f / 512.0f);
                    }
                }

                pred.fund_freq_hz = f_hz;
                out_predictions.push_back(pred);
            }
        }

        return true;
    }

    bool is_ready() const { return is_ready_; }

private:
    void cleanup() {
        if (stream_) cudaStreamDestroy(stream_);
        if (d_input_) cudaFree(d_input_);
        if (d_prob_) cudaFree(d_prob_);
        if (d_freq_) cudaFree(d_freq_);
        if (d_purity_) cudaFree(d_purity_);
        if (h_input_) cudaFreeHost(h_input_);
        if (h_prob_) cudaFreeHost(h_prob_);
        if (h_freq_) cudaFreeHost(h_freq_);
        if (h_purity_) cudaFreeHost(h_purity_);
        is_ready_ = false;
    }

    bool is_ready_{false};
    size_t max_batch_{128};

    cudaStream_t stream_{nullptr};
    std::unique_ptr<nvinfer1::IRuntime> runtime_{nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine> engine_{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> context_{nullptr};

    float* d_input_{nullptr};
    float* d_prob_{nullptr};
    float* d_freq_{nullptr};
    float* d_purity_{nullptr};

    float* h_input_{nullptr};
    float* h_prob_{nullptr};
    float* h_freq_{nullptr};
    float* h_purity_{nullptr};
};

} // namespace predator

#endif // PREDATOR_SPECTRAL_COMBNET_TRT_HPP
