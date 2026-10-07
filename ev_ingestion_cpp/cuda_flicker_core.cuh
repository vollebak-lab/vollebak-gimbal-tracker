#ifndef PREDATOR_CUDA_FLICKER_CORE_CUH
#define PREDATOR_CUDA_FLICKER_CORE_CUH

#include <vector>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <cuda_runtime.h>
#include <cufft.h>
#include "flicker_dsp.hpp"
#include "gpu_event.hpp"
#include "spectral_gate.hpp"

namespace predator {

// CudaRawEvent (16 B GPU event record) is defined in gpu_event.hpp, shared with the EVT2.1 GPU decoder.

/**
 * @brief Detection candidate record returned by GPU spectral analysis kernel
 */
struct alignas(16) CudaDetectionCandidate {
    int cell_idx;             ///< 0..575 (base cells), 576..1151 (pooled cells)
    int is_pooled;            ///< 0 = base (40x40 px), 1 = pooled (80x80 px)
    int patch_col;            ///< Grid column (0..31)
    int patch_row;            ///< Grid row (0..17)
    float fundamental_bpf_hz; ///< Refined peak frequency (Hz)
    float peak_snr_db;        ///< Peak-to-Noise Ratio (dB)
    float spectral_q_factor;  ///< Spectral sharpness
    float peak_power;         ///< Absolute peak spectral power
    float noise_floor;        ///< Background noise floor
    float confidence;         ///< Confidence score (0.0 to 1.0)
    float total_events;       ///< Total event count in cell
    float spectral_flatness;  ///< DDHF Spectral Flatness (scale-invariant comb metric: << 1 for harmonic comb)
    uint32_t max_sieve_hits;  ///< Maximum consecutive periodic hits in micro-sieve tiles
};

/**
 * @brief Diagnostic telemetry for a specified spatial Region of Interest (ROI)
 */
struct RoiDiagnostics {
    float total_events{0.0f};
    float max_cell_events{0.0f};
    uint32_t max_sieve_hits{0};
    int active_cells{0};
    int candidates_in_roi{0};
    float best_snr_db{0.0f};
    float best_bpf_hz{0.0f};
    float best_flatness{1.0f};
};

/**
 * @brief High-Performance GPU Acceleration Core for Event Warping, Periodicity Sieving, and Batched cuFFT
 */
class CudaFlickerCore {
public:
    CudaFlickerCore(int sensor_width = 1280, int sensor_height = 720,
                    int grid_cols = 32, int grid_rows = 18,
                    double sample_rate_hz = 4000.0, size_t history_samples = 512);
    ~CudaFlickerCore();

    // Prevent copying
    CudaFlickerCore(const CudaFlickerCore&) = delete;
    CudaFlickerCore& operator=(const CudaFlickerCore&) = delete;

    /**
     * @brief Ingests a batch of events: CPU periodicity sieve (timestamp order), then GPU homography
     *        unwarping and ring buffer accumulation. Batches larger than the pinned staging buffer are
     *        processed in chunks; no event is dropped. Must be called from a single thread (camera callback).
     * @param events Host pointer to Metavision EventCD events (non-decreasing timestamps)
     * @param count Number of events in batch
     * @param H 3x3 Homography matrix mapping current camera frame to stabilized world frame
     * @param[out] out_raw_count Total raw events processed
     * @param[out] out_retained_count Number of events passing the periodicity sieve (this batch)
     */
    void ingest_event_batch(const void* events, size_t count, const Matrix3x3& H,
                            uint64_t& out_raw_count, uint64_t& out_retained_count,
                            const float* d_suppression_mask = nullptr,
                            float suppression_threshold = 0.35f,
                            bool sync = false);

    /**
     * @brief Ingests device-resident events (Phase 33.4b.c).
     * Events have already been decoded and periodic hits evaluated into CudaRawEvent::pad on the GPU.
     * Advances temporal window up to max_t, applies homography H and optional suppression mask,
     * and accumulates into cuFFT ring buffers on device stream.
     *
     * @param d_events Pointer to device array of CudaRawEvent
     * @param count Number of events in batch
     * @param H Homography matrix
     * @param min_t Minimum event timestamp in batch (us)
     * @param max_t Maximum event timestamp in batch (us)
     * @param stream CUDA stream (defaults to internal stream_ if null)
     * @param d_suppression_mask Optional TensorRT suppression mask on device
     * @param suppression_threshold Suppression threshold
     */
    void ingest_device_events(
        const CudaRawEvent* d_events,
        size_t count,
        const Matrix3x3& H,
        uint64_t min_t,
        uint64_t max_t,
        cudaStream_t stream = nullptr,
        const float* d_suppression_mask = nullptr,
        float suppression_threshold = 0.35f);

    cudaStream_t stream() const { return stream_; }

    /**
     * @brief Asynchronously fetches accumulated retained event count from GPU and resets device counter
     */
    uint64_t get_and_reset_retained_count();

    /**
     * @brief Advances temporal ring buffer bins in GPU memory
     * @param steps Number of time bins to advance
     */
    void advance_temporal_bins(size_t steps, cudaStream_t stream = nullptr);

    /**
     * @brief Validates a gate configuration and derives its CFAR threshold (spectral_gate.hpp).
     * @throws std::invalid_argument on an inconsistent configuration (previous config is kept).
     */
    void set_spectral_gate_config(const SpectralGateConfig& cfg);

    /// Active (derived) gate configuration, including the CFAR threshold in use.
    const SpectralGateConfig& spectral_gate_config() const { return gate_cfg_; }

    /**
     * @brief Executes batched 1152-channel cuFFT and the shared CFAR gate chain on GPU
     * @param gyro_speed_deg_s Current angular speed of camera from IMU (texture-scan filter)
     * @param[out] out_candidates Cells passing every gate (each meets the false-alarm budget)
     */
    void execute_batched_spectral_analysis(
        double gyro_speed_deg_s,
        std::vector<FlickerDetectionResult>& out_candidates);

    /**
     * @brief Checks GPU initialization status
     */
    bool is_initialized() const { return initialized_; }

    /**
     * @brief Returns total active cells with event density above threshold
     */
    int get_active_cell_count(double min_threshold);

    /**
     * @brief Computes detailed diagnostic statistics for a spatial ROI (columns [col_min..col_max], rows [row_min..row_max])
     */
    RoiDiagnostics get_roi_diagnostics(int col_min, int col_max, int row_min, int row_max);

    /**
     * @brief Reads back current total event counts across all 1152 cells (for testing & parity verification)
     */
    void get_cell_total_events(std::vector<float>& out_totals);

    /**
     * @brief Fetches active candidate cells and their 257-bin log-normalized power spectra for SpectralCombNet
     * @param out_cell_indices Output vector of cell indices
     * @param out_spectra Output vector of 257-bin normalized spectra
     * @param min_events Minimum event count (default 6.0)
     */
    void get_active_cells_with_spectra(std::vector<int>& out_cell_indices,
                                      std::vector<std::vector<float>>& out_spectra,
                                      float min_events = 6.0f);

    /**
     * @brief Computes normalized 257-bin spectrum directly from a 512-sample time series on GPU (Phase 33.6a).
     * Applies Hanning window, executes 512-point cuFFT R2C, computes power |X_k|^2, evaluates median noise floor
     * from bins 5..127, and log10-normalizes matching SpectralCombNet runtime preprocessing.
     * @param time_series_512 Pointer to 512 chronological temporal float samples
     * @param out_spectrum_257 Pointer to output buffer for 257 normalized float bins
     * @param out_median_noise Optional output pointer for the computed median noise floor
     */
    void compute_normalized_spectrum(const float* time_series_512, float* out_spectrum_257, float* out_median_noise = nullptr);

    /**
     * @brief Resets per-frame periodic micro-sieve hit accumulators in GPU memory
     */
    void reset_sieve_hit_accumulators();

    /**
     * @brief Resets all GPU state, SAE surfaces, and ring buffers
     */
    void reset();

private:
    bool initialized_{false};
    int sensor_width_{1280};
    int sensor_height_{720};
    int grid_cols_{32};
    int grid_rows_{18};
    int num_base_cells_{576};
    int num_total_cells_{1152}; // 576 base + 576 pooled
    double sample_rate_hz_{4000.0};
    size_t history_samples_{512};
    uint64_t bin_duration_us_{250};
    uint64_t current_window_start_us_{0};   ///< Head bin start, snapped to a multiple of bin_duration_us_
    bool window_anchored_{false};           ///< False until the first chunk establishes the bin grid
    size_t head_idx_{0};
    mutable std::recursive_mutex core_mutex_;
    SpectralGateConfig gate_cfg_{};         ///< Derived CFAR gate (set_spectral_gate_config)

    // Order-correct periodicity sieve on 2x2 micro-tiles (Phase 33.4). Runs on the CPU in the
    // camera callback thread, in timestamp order; 75-1200 Hz (833-13333 us periods).
    MicroNeighborhoodPeriodicitySieve sieve_;

    // Events that passed the sieve since the last get_and_reset_retained_count() (telemetry).
    std::atomic<uint64_t> retained_count_{0};

    // CUDA Streams & cuFFT Plan
    cudaStream_t stream_{nullptr};
    cufftHandle cufft_plan_{0};
    cudaEvent_t upload_done_{nullptr}; ///< Signals h_event_buffer_ may be overwritten (async H2D finished)

    // Host Pinned Memory Buffers
    CudaRawEvent* h_event_buffer_{nullptr};
    CudaDetectionCandidate* h_candidate_buffer_{nullptr};
    float* h_cell_totals_{nullptr};
    uint32_t* h_cell_sieve_hits_{nullptr};
    size_t max_events_per_batch_{131072};
    size_t max_candidates_{512};

    // Device GPU Buffers
    CudaRawEvent* d_events_{nullptr};
    float* d_homography_matrix_{nullptr};

    // Spatial Ring Buffers (1152 cells x 512 samples)
    float* d_ring_buffers_{nullptr};
    float* d_cell_total_events_{nullptr};
    uint32_t* d_cell_max_sieve_hits_{nullptr};

    // cuFFT Input/Output and Power Spectrum
    float* d_fft_input_{nullptr};
    cufftComplex* d_fft_output_{nullptr};
    float* d_power_spectrum_{nullptr};

    // Output Candidate Buffers
    CudaDetectionCandidate* d_candidates_{nullptr};
    uint32_t* d_num_candidates_{nullptr};

    /// Uploads one pre-sieved chunk (<= max_events_per_batch_) and launches the ingest kernel.
    void ingest_chunk(size_t chunk_size, uint64_t chunk_max_t, const float* d_suppression_mask,
                      float suppression_threshold);
};

} // namespace predator

#endif // PREDATOR_CUDA_FLICKER_CORE_CUH
