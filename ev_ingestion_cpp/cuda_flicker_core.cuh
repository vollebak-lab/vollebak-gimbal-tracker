#ifndef PREDATOR_CUDA_FLICKER_CORE_CUH
#define PREDATOR_CUDA_FLICKER_CORE_CUH

#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include <cufft.h>
#include "flicker_dsp.hpp"

namespace predator {

/**
 * @brief Raw event struct compatible with CUDA memory layout
 */
struct alignas(16) CudaRawEvent {
    uint16_t x;
    uint16_t y;
    int16_t p;
    int16_t pad;
    uint64_t t;
};

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
     * @brief Ingests a batch of events onto GPU, performs homography unwarping, SAE periodicity sieving, and ring buffer accumulation
     * @param events Host pointer to Metavision EventCD events
     * @param count Number of events in batch
     * @param H 3x3 Homography matrix mapping current camera frame to stabilized world frame
     * @param[out] out_raw_count Total raw events processed
     * @param[out] out_retained_count Number of events passing the periodicity sieve
     */
    void ingest_event_batch(const void* events, size_t count, const Matrix3x3& H,
                            uint64_t& out_raw_count, uint64_t& out_retained_count,
                            const float* d_suppression_mask = nullptr,
                            float suppression_threshold = 0.35f,
                            bool sync = false);

    /**
     * @brief Asynchronously fetches accumulated retained event count from GPU and resets device counter
     */
    uint64_t get_and_reset_retained_count();

    /**
     * @brief Advances temporal ring buffer bins in GPU memory
     * @param steps Number of time bins to advance
     */
    void advance_temporal_bins(size_t steps);

    /**
     * @brief Executes batched 1152-channel cuFFT and parallel peak spectral detection on GPU
     * @param min_freq_hz Minimum search frequency (e.g., 70 Hz)
     * @param max_freq_hz Maximum search frequency (e.g., 800 Hz)
     * @param min_energy Minimum peak power threshold
     * @param min_snr_db Minimum SNR in dB threshold
     * @param gyro_speed_deg_s Current angular speed of camera from IMU
     * @param[out] out_candidates List of detected drone propeller candidates
     */
    void execute_batched_spectral_analysis(
        double min_freq_hz, double max_freq_hz,
        double min_energy, double min_snr_db,
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
     * @brief Fetches active candidate cells and their 257-bin log-normalized power spectra for SpectralCombNet
     * @param out_cell_indices Output vector of cell indices
     * @param out_spectra Output vector of 257-bin normalized spectra
     * @param min_events Minimum event count (default 6.0)
     */
    void get_active_cells_with_spectra(std::vector<int>& out_cell_indices,
                                      std::vector<std::vector<float>>& out_spectra,
                                      float min_events = 6.0f);

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
    uint64_t current_window_start_us_{0};
    size_t head_idx_{0};

    // Micro-tile SAE dimensions (2x2 pixel micro-neighborhood)
    int tile_w_{640};
    int tile_h_{360};
    int num_tiles_{640 * 360};

    // CUDA Streams & cuFFT Plan
    cudaStream_t stream_{nullptr};
    cufftHandle cufft_plan_{0};

    // Host Pinned Memory Buffers
    CudaRawEvent* h_event_buffer_{nullptr};
    CudaDetectionCandidate* h_candidate_buffer_{nullptr};
    float* h_cell_totals_{nullptr};
    uint32_t* h_cell_sieve_hits_{nullptr};
    size_t max_events_per_batch_{131072};
    size_t max_candidates_{512};

    // Device GPU Buffers
    CudaRawEvent* d_events_{nullptr};
    uint32_t* d_retained_counter_{nullptr};
    float* d_homography_matrix_{nullptr};

    // Surface of Active Events (SAE) in GPU memory
    uint32_t* d_sae_timestamp_us_{nullptr};
    uint32_t* d_sae_last_dt_us_{nullptr};
    uint8_t* d_sae_hits_{nullptr};

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
};

} // namespace predator

#endif // PREDATOR_CUDA_FLICKER_CORE_CUH
