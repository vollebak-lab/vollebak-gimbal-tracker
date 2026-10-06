#ifndef PREDATOR_FLICKER_DSP_HPP
#define PREDATOR_FLICKER_DSP_HPP

#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <array>
#include <cstring>

namespace predator {

/**
 * @brief 3D Vector for angular velocity, coordinates, and acceleration
 */
struct Vector3d {
    double x{0.0};
    double y{0.0};
    double z{0.0};

    Vector3d() = default;
    Vector3d(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vector3d operator+(const Vector3d& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vector3d operator-(const Vector3d& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vector3d operator*(double s) const { return {x * s, y * s, z * s}; }
    double norm() const { return std::sqrt(x * x + y * y + z * z); }
};

/**
 * @brief 3x3 Matrix for 3D Rotations and Homographies
 */
struct Matrix3x3 {
    std::array<double, 9> m{};

    Matrix3x3() {
        m.fill(0.0);
    }

    static Matrix3x3 identity() {
        Matrix3x3 mat;
        mat.m[0] = 1.0; mat.m[4] = 1.0; mat.m[8] = 1.0;
        return mat;
    }

    double at(int r, int c) const { return m[r * 3 + c]; }
    double& at(int r, int c) { return m[r * 3 + c]; }

    Matrix3x3 operator*(const Matrix3x3& o) const {
        Matrix3x3 res;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                double sum = 0.0;
                for (int k = 0; k < 3; ++k) {
                    sum += at(r, k) * o.at(k, c);
                }
                res.at(r, c) = sum;
            }
        }
        return res;
    }

    Vector3d operator*(const Vector3d& v) const {
        return {
            at(0, 0) * v.x + at(0, 1) * v.y + at(0, 2) * v.z,
            at(1, 0) * v.x + at(1, 1) * v.y + at(1, 2) * v.z,
            at(2, 0) * v.x + at(2, 1) * v.y + at(2, 2) * v.z
        };
    }

    /**
     * @brief Computes analytical matrix inverse for 3x3 matrix
     */
    Matrix3x3 inverse() const {
        double det = at(0, 0) * (at(1, 1) * at(2, 2) - at(1, 2) * at(2, 1)) -
                     at(0, 1) * (at(1, 0) * at(2, 2) - at(1, 2) * at(2, 0)) +
                     at(0, 2) * (at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0));

        if (std::abs(det) < 1e-12) {
            return identity();
        }

        double invdet = 1.0 / det;
        Matrix3x3 inv;

        inv.at(0, 0) = (at(1, 1) * at(2, 2) - at(1, 2) * at(2, 1)) * invdet;
        inv.at(0, 1) = (at(0, 2) * at(2, 1) - at(0, 1) * at(2, 2)) * invdet;
        inv.at(0, 2) = (at(0, 1) * at(1, 2) - at(0, 2) * at(1, 1)) * invdet;

        inv.at(1, 0) = (at(1, 2) * at(2, 0) - at(1, 0) * at(2, 2)) * invdet;
        inv.at(1, 1) = (at(0, 0) * at(2, 2) - at(0, 2) * at(2, 0)) * invdet;
        inv.at(1, 2) = (at(1, 0) * at(0, 2) - at(0, 0) * at(1, 2)) * invdet;

        inv.at(2, 0) = (at(1, 0) * at(2, 1) - at(2, 0) * at(1, 1)) * invdet;
        inv.at(2, 1) = (at(2, 0) * at(0, 1) - at(0, 0) * at(2, 1)) * invdet;
        inv.at(2, 2) = (at(0, 0) * at(1, 1) - at(1, 0) * at(0, 1)) * invdet;

        return inv;
    }
};

/**
 * @brief Optical and Sensor Intrinsic Parameters
 */
struct LensParameters {
    double focal_length_mm{12.0};    ///< 12mm FL f/2.0 M12 (1/2.5" format, 5MP)
    double pixel_pitch_um{4.86};      ///< Sony IMX636 pixel size (4.86 um)
    int sensor_width{1280};           ///< IMX636 width
    int sensor_height{720};           ///< IMX636 height

    // Focal length in pixel units: f_pix = (f_mm / p_mm)
    double fx_pix() const { return (focal_length_mm * 1000.0) / pixel_pitch_um; }
    double fy_pix() const { return (focal_length_mm * 1000.0) / pixel_pitch_um; }
    double cx_pix() const { return sensor_width / 2.0; }
    double cy_pix() const { return sensor_height / 2.0; }

    /**
     * @brief Computes Azimuth and Elevation angles (in degrees) from pixel coordinates
     * @param px X coordinate in pixels
     * @param py Y coordinate in pixels
     * @param[out] azimuth_deg Azimuth angle relative to optical axis (+ = Right, - = Left)
     * @param[out] elevation_deg Elevation angle relative to optical axis (+ = Up, - = Down)
     */
    void pixel_to_angles(double px, double py, double &azimuth_deg, double &elevation_deg) const {
        double dx = px - cx_pix();
        double dy = cy_pix() - py; // positive Y is up
        double f = fx_pix();
        azimuth_deg = std::atan2(dx, f) * (180.0 / M_PI);
        elevation_deg = std::atan2(dy, f) * (180.0 / M_PI);
    }
};

/**
 * @brief Detection Result from Frequency Analysis
 */
struct FlickerDetectionResult {
    bool is_drone_detected{false};
    double fundamental_bpf_hz{0.0};   ///< Blade Passage Frequency (Hz)
    double estimated_rpm{0.0};        ///< Estimated mechanical RPM (assumes 2-blade default)
    double confidence{0.0};           ///< Normalized confidence score (0.0 to 1.0)
    double peak_snr_db{0.0};          ///< Peak-to-Noise Ratio (dB)
    double harmonic_score{0.0};       ///< Multi-harmonic comb match score
    double spectral_q_factor{0.0};    ///< Sharpness / Q-Factor (blade spike vs wind foliage)
    double spectral_flatness{1.0};    ///< DDHF Spectral Flatness (scale-invariant comb metric: << 1 for harmonic comb)
    double peak_power{0.0};           ///< Absolute peak spectral power
    double noise_floor{0.0};          ///< Poisson noise floor level
    double total_events{0.0};         ///< Event density in patch
    double azimuth_deg{0.0};          ///< Target bearing azimuth
    double elevation_deg{0.0};        ///< Target bearing elevation
    int patch_x{0};                   ///< Patch grid X
    int patch_y{0};                   ///< Patch grid Y
    int centroid_px_x{0};             ///< Centroid pixel X
    int centroid_px_y{0};             ///< Centroid pixel Y
    uint64_t timestamp_us{0};         ///< Timestamp of analysis
    int track_id{0};                  ///< Associated tracking ID
    int track_state{0};               ///< 0: UNTRACKED, 1: TENTATIVE, 2: CONFIRMED, 3: COASTING
    int hit_count{0};                 ///< Consecutive hit count
    int miss_count{0};                ///< Consecutive miss count
    uint32_t max_sieve_hits{0};       ///< Micro-sieve hit count
    bool is_neural_detection{false};  ///< True if detected/confirmed by SpectralCombNet
    float neural_drone_prob{0.0f};    ///< Neural network confidence (0.0 to 1.0)
};

/**
 * @brief Fast Cooley-Tukey Radix-2 Real/Complex 1D FFT Implementation
 */
class FastFourierTransform {
public:
    static void fft(std::vector<std::complex<double>> &a, bool invert = false) {
        size_t n = a.size();
        for (size_t i = 1, j = 0; i < n; i++) {
            size_t bit = n >> 1;
            for (; j & bit; bit >>= 1) {
                j ^= bit;
            }
            j ^= bit;
            if (i < j) {
                std::swap(a[i], a[j]);
            }
        }

        for (size_t len = 2; len <= n; len <<= 1) {
            double ang = 2 * M_PI / len * (invert ? -1 : 1);
            std::complex<double> wlen(std::cos(ang), std::sin(ang));
            for (size_t i = 0; i < n; i += len) {
                std::complex<double> w(1);
                for (size_t j = 0; j < len / 2; j++) {
                    std::complex<double> u = a[i + j];
                    std::complex<double> v = a[i + j + len / 2] * w;
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }

        if (invert) {
            for (auto &x : a) {
                x /= n;
            }
        }
    }
};

/**
 * @brief Micro-Neighborhood Recurrent Periodicity Sieve (HelixTrack / FrequencyCam Neuromorphic Core)
 * 
 * Evaluates the Surface of Active Events (SAE) across a 2x2 micro-tile neighborhood with 4-neighbor boundary cross-checking.
 * Discards single-shot non-repeating ego-motion edge steps, slow windblown foliage sway, and isolated shot noise in O(1) time.
 */
class MicroNeighborhoodPeriodicitySieve {
public:
    MicroNeighborhoodPeriodicitySieve(int width = 1280, int height = 720,
                                      double min_freq_hz = 70.0, double max_freq_hz = 800.0,
                                      uint8_t min_consecutive_hits = 2)
        : width_(width), height_(height),
          tile_w_((width + 1) / 2), tile_h_((height + 1) / 2),
          num_tiles_(tile_w_ * tile_h_),
          min_period_us_(static_cast<uint32_t>(1000000.0 / max_freq_hz)),   // ~1250 us (800 Hz)
          max_period_us_(static_cast<uint32_t>(1000000.0 / min_freq_hz)),   // ~14285 us (70 Hz)
          min_consecutive_hits_(min_consecutive_hits) {
        
        last_timestamp_us_.assign(num_tiles_, 0);
        last_dt_us_.assign(num_tiles_, 0);
        consecutive_hits_.assign(num_tiles_, 0);
    }

    /**
     * @brief Tests if an event belongs to a periodic high-frequency harmonic train
     * @param x Event pixel X (0..width-1)
     * @param y Event pixel Y (0..height-1)
     * @param timestamp_us Event timestamp in microseconds
     * @return true if event is periodic within [min_freq_hz, max_freq_hz], false if aperiodic/clutter
     */
    inline bool is_periodic_event(int x, int y, uint64_t timestamp_us) {
        int tx = x >> 1;
        int ty = y >> 1;
        if (tx < 0 || tx >= tile_w_ || ty < 0 || ty >= tile_h_) return false;

        size_t idx = static_cast<size_t>(ty * tile_w_ + tx);
        uint32_t t_cur = static_cast<uint32_t>(timestamp_us);
        uint32_t t_last = last_timestamp_us_[idx];

        if (t_last == 0) {
            last_timestamp_us_[idx] = t_cur;
            consecutive_hits_[idx] = 0;
            last_dt_us_[idx] = 0;
            return false;
        }

        uint32_t dt = t_cur - t_last;

        // 1. Intra-burst event (< min_period): event belongs to the SAME active blade sweep!
        if (dt < min_period_us_) {
            // If this tile has already achieved periodic lock, pass all events within the blade pass!
            if (consecutive_hits_[idx] >= min_consecutive_hits_) {
                return true;
            }
            return false;
        }

        // 2. Inter-sweep recurrence (min_period <= dt <= max_period): a new periodic blade chop has arrived!
        if (dt <= max_period_us_) {
            uint16_t prev_dt = last_dt_us_[idx];
            last_dt_us_[idx] = static_cast<uint16_t>(std::min(dt, (uint32_t)65535));
            last_timestamp_us_[idx] = t_cur;

            bool period_consistent = (prev_dt == 0) ||
                (std::abs(static_cast<int>(dt) - static_cast<int>(prev_dt)) <= static_cast<int>(prev_dt * 0.45));

            if (period_consistent) {
                uint8_t h = consecutive_hits_[idx];
                if (h < 255) h++;
                consecutive_hits_[idx] = h;
                if (h >= min_consecutive_hits_) {
                    return true;
                }
            } else {
                consecutive_hits_[idx] = 1;
            }
            return false;
        }

        // 3. Cross-tile boundary check (in case blade tip crossed into adjacent 2x2 tile)
        static const int dtx[4] = {1, -1, 0, 0};
        static const int dty[4] = {0, 0, 1, -1};

        for (int k = 0; k < 4; ++k) {
            int ntx = tx + dtx[k];
            int nty = ty + dty[k];
            if (ntx >= 0 && ntx < tile_w_ && nty >= 0 && nty < tile_h_) {
                size_t n_idx = static_cast<size_t>(nty * tile_w_ + ntx);
                uint32_t nt_last = last_timestamp_us_[n_idx];
                if (nt_last > 0) {
                    uint32_t ndt = t_cur - nt_last;
                    if (ndt >= min_period_us_ && ndt <= max_period_us_) {
                        uint16_t prev_dt = last_dt_us_[n_idx];
                        bool period_consistent = (prev_dt == 0) ||
                            (std::abs(static_cast<int>(ndt) - static_cast<int>(prev_dt)) <= static_cast<int>(prev_dt * 0.45));
                        if (period_consistent) {
                            uint8_t h = consecutive_hits_[idx];
                            if (h < 255) h++;
                            consecutive_hits_[idx] = h;
                            last_dt_us_[idx] = static_cast<uint16_t>(std::min(ndt, (uint32_t)65535));
                            last_timestamp_us_[idx] = t_cur;
                            if (h >= min_consecutive_hits_) {
                                return true;
                            }
                            return false;
                        }
                    }
                }
            }
        }

        // 4. Stale event (> max_period): time gap too large, reset tracking
        last_timestamp_us_[idx] = t_cur;
        consecutive_hits_[idx] = 0;
        last_dt_us_[idx] = 0;
        return false;
    }

    void reset() {
        std::fill(last_timestamp_us_.begin(), last_timestamp_us_.end(), 0);
        std::fill(last_dt_us_.begin(), last_dt_us_.end(), 0);
        std::fill(consecutive_hits_.begin(), consecutive_hits_.end(), 0);
    }

    int tile_width() const { return tile_w_; }
    int tile_height() const { return tile_h_; }

private:
    int width_;
    int height_;
    int tile_w_;
    int tile_h_;
    size_t num_tiles_;
    uint32_t min_period_us_;
    uint32_t max_period_us_;
    uint8_t min_consecutive_hits_;

    std::vector<uint32_t> last_timestamp_us_;
    std::vector<uint16_t> last_dt_us_;
    std::vector<uint8_t> consecutive_hits_;
};

/**
 * @brief Frequency-Domain Temporal Event Analyzer (DDHF Core)
 */
/**
 * @brief Frequency-Domain Temporal Event Analyzer with Low-Light Fundamental Dominance
 */
class PropellerFlickerAnalyzer {
public:
    /**
     * @brief Constructor
     * @param sample_rate_hz Sampling rate for temporal binning (default 4000 Hz => 250us bins, 2000Hz Nyquist)
     * @param window_size_samples FFT window size (default 512 samples => 128ms window for coherent integration)
     * @param min_freq_hz Minimum detectable frequency (default 80 Hz)
     * @param max_freq_hz Maximum detectable frequency (default 1200 Hz)
     * @param min_events_threshold Minimum raw events in patch required to compute FFT (default 15)
     */
    PropellerFlickerAnalyzer(double sample_rate_hz = 4000.0,
                             size_t window_size_samples = 512,
                             double min_freq_hz = 80.0,
                             double max_freq_hz = 1200.0,
                             double min_events_threshold = 5.0)
        : sample_rate_hz_(sample_rate_hz),
          window_size_(window_size_samples),
          min_freq_hz_(min_freq_hz),
          max_freq_hz_(max_freq_hz),
          min_events_threshold_(min_events_threshold),
          lens_params_() {
        // Precalculate Hanning window
        window_weights_.resize(window_size_);
        for (size_t i = 0; i < window_size_; ++i) {
            window_weights_[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (window_size_ - 1)));
        }
    }

    /**
     * @brief Extracts multi-candidate spectral peaks to resolve true rotor flicker in the presence of strong building/motion floodlight modulation
     * @param time_series Vector of event counts
     * @param blades Number of blades (default 2)
     * @param max_candidates Max candidates to return (default 2)
     * @return Vector of FlickerDetectionResult for each prominent spectral peak
     */
    std::vector<FlickerDetectionResult> analyze_time_series_candidates(const std::vector<double> &time_series, int blades = 2, int max_candidates = 2) const {
        if (time_series.size() < window_size_) {
            return {};
        }

        size_t offset = time_series.size() - window_size_;

        // 1. Activity Density Check (gating noise before FFT)
        double total_events = 0.0;
        for (size_t i = 0; i < window_size_; ++i) {
            total_events += time_series[offset + i];
        }
        if (total_events < min_events_threshold_) {
            return {};
        }

        // 2. Zero-mean + Hanning window
        std::vector<std::complex<double>> complex_buf(window_size_);
        double mean_val = total_events / window_size_;

        for (size_t i = 0; i < window_size_; ++i) {
            double zero_mean = time_series[offset + i] - mean_val;
            complex_buf[i] = zero_mean * window_weights_[i];
        }

        // 3. Compute 1D FFT
        FastFourierTransform::fft(complex_buf);

        // 4. Compute Power Spectrum: P[k] = |X[k]|^2
        size_t num_bins = window_size_ / 2;
        std::vector<double> power(num_bins, 0.0);
        double freq_resolution = sample_rate_hz_ / window_size_;

        for (size_t k = 0; k < num_bins; ++k) {
            power[k] = std::norm(complex_buf[k]);
        }

        // 5. Compute Power Spectrum Peak & Harmonic Product Spectrum (HPS)
        std::vector<double> hps(num_bins, 0.0);
        size_t min_bin = static_cast<size_t>(std::max(1.0, std::floor(min_freq_hz_ / freq_resolution)));
        size_t max_bin = static_cast<size_t>(std::min(static_cast<double>(num_bins - 1), std::ceil(max_freq_hz_ / freq_resolution)));

        for (size_t k = min_bin; k <= max_bin; ++k) {
            double p1 = power[k];
            double p2 = (2 * k < num_bins) ? power[2 * k] : 0.0;
            double p3 = (3 * k < num_bins) ? power[3 * k] : 0.0;
            // Harmonic enhancement weighted for daylight and shaded rotor reflections
            hps[k] = p1 * (1.0 + 0.8 * std::sqrt(p2)) * (1.0 + 0.4 * std::sqrt(p3));
        }

        // 6. Find Top Distinct Local Maxima in HPS
        std::vector<size_t> peak_bins;
        for (size_t k = min_bin + 1; k < max_bin; ++k) {
            if (hps[k] > hps[k - 1] && hps[k] > hps[k + 1] && hps[k] > 1e-5) {
                peak_bins.push_back(k);
            }
        }

        // Sort peaks descending by HPS power
        std::sort(peak_bins.begin(), peak_bins.end(), [&](size_t a, size_t b) {
            return hps[a] > hps[b];
        });

        // Select top distinct peaks (separated by at least 20 Hz)
        std::vector<size_t> distinct_peaks;
        for (size_t k : peak_bins) {
            bool too_close = false;
            for (size_t d : distinct_peaks) {
                double f_diff = std::abs(static_cast<double>(k) - static_cast<double>(d)) * freq_resolution;
                if (f_diff < 20.0) {
                    too_close = true;
                    break;
                }
            }
            if (!too_close) {
                distinct_peaks.push_back(k);
                if (static_cast<int>(distinct_peaks.size()) >= max_candidates) break;
            }
        }

        if (distinct_peaks.empty()) {
            return {};
        }

        std::vector<FlickerDetectionResult> results;
        for (size_t best_bin : distinct_peaks) {
            // Parabolic Interpolation for Sub-Bin Frequency Refinement
            double refined_bin = best_bin;
            if (best_bin > min_bin && best_bin < max_bin) {
                double alpha = hps[best_bin - 1];
                double beta  = hps[best_bin];
                double gamma = hps[best_bin + 1];
                double denom = (alpha - 2.0 * beta + gamma);
                if (std::abs(denom) > 1e-9) {
                    double delta = 0.5 * (alpha - gamma) / denom;
                    refined_bin += delta;
                }
            }

            double fundamental_bpf = refined_bin * freq_resolution;
            if (fundamental_bpf < min_freq_hz_ || fundamental_bpf > max_freq_hz_) {
                continue;
            }

            // Compute Spectral SNR against Wideband Noise Floor using Median Spectral Estimator (CFAR robust)
            std::vector<double> noise_bins;
            noise_bins.reserve(num_bins);
            size_t noise_start_bin = static_cast<size_t>(std::max(1.0, std::floor(40.0 / freq_resolution)));
            size_t noise_end_bin = static_cast<size_t>(std::min(static_cast<double>(num_bins - 1), std::ceil(1000.0 / freq_resolution)));
            for (size_t k = noise_start_bin; k <= noise_end_bin; ++k) {
                bool is_signal = (std::abs(static_cast<int>(k) - static_cast<int>(best_bin)) <= 2) ||
                                 (std::abs(static_cast<int>(k) - 2 * static_cast<int>(best_bin)) <= 2) ||
                                 (std::abs(static_cast<int>(k) - 3 * static_cast<int>(best_bin)) <= 2);
                if (!is_signal) {
                    noise_bins.push_back(power[k]);
                }
            }

            double median_noise = 1e-9;
            if (!noise_bins.empty()) {
                size_t mid = noise_bins.size() / 2;
                std::nth_element(noise_bins.begin(), noise_bins.begin() + mid, noise_bins.end());
                median_noise = noise_bins[mid] * 1.442695; // Unbiased exponential distribution scale (1/ln 2)
            }

            double peak_power = power[best_bin];
            double snr_linear = (median_noise > 1e-12) ? (peak_power / median_noise) : 1.0;
            double snr_db = 10.0 * std::log10(std::max(1.0, snr_linear));

            // Absolute Peak Energy Gate: Require true periodic sinusoidal energy (scaled for 100ft sparse events)
            if (peak_power < 2.0) {
                continue;
            }

            // Spectral Sharpness (Q-Factor) Gate: Mechanical blade spike vs broad wind/foliage turbulence
            double neighbor_power = 1e-9;
            if (best_bin >= 2 && best_bin + 2 < num_bins) {
                neighbor_power = 0.5 * (power[best_bin - 2] + power[best_bin + 2]);
            }
            double sharpness = (neighbor_power > 1e-12) ? (peak_power / neighbor_power) : 10.0;
            if (sharpness < 2.0) {
                continue; // Reject broad turbulence humps (fluttering leaves, wind sway)
            }

            // Spectral Purity Gate: Reject broadband noise and weak clutter
            if (snr_linear < 6.0 || snr_db < 8.0) {
                continue;
            }

            double h2_ratio = (2 * best_bin < num_bins && median_noise > 1e-12) ? (power[2 * best_bin] / median_noise) : 0.0;
            double h3_ratio = (3 * best_bin < num_bins && median_noise > 1e-12) ? (power[3 * best_bin] / median_noise) : 0.0;

            // DDHF Spectral Flatness across search band (scale-invariant harmonic comb metric)
            double sum_log = 0.0;
            double sum_p = 0.0;
            int count_bins = 0;
            for (size_t k = min_bin; k <= max_bin; ++k) {
                sum_log += std::log(power[k] + 1e-9);
                sum_p += power[k];
                count_bins++;
            }
            double geom_mean = (count_bins > 0) ? std::exp(sum_log / count_bins) : 1.0;
            double arith_mean = (count_bins > 0) ? (sum_p / count_bins) : 1.0;
            double spectral_flatness = (arith_mean > 1e-9) ? (geom_mean / arith_mean) : 1.0;

            double harmonic_bonus = (h2_ratio > 1.8 ? 0.20 : 0.0) + (h3_ratio > 1.3 ? 0.15 : 0.0);
            double flatness_bonus = (spectral_flatness < 0.18) ? (0.20 * (0.18 - spectral_flatness) / 0.18) : 0.0;
            double fund_score = std::min(1.0, std::max(0.0, (snr_db - 8.0) / 6.0));
            double confidence = std::min(1.0, fund_score * 0.70 + harmonic_bonus + flatness_bonus + std::min(0.15, snr_linear / 20.0));

            bool detected = (snr_db >= 10.0 || (snr_db >= 8.5 && harmonic_bonus > 0.10) || (snr_db >= 6.5 && spectral_flatness < 0.15)) &&
                            (fundamental_bpf >= min_freq_hz_ && fundamental_bpf <= max_freq_hz_) &&
                            (confidence >= 0.45);

            FlickerDetectionResult res;
            res.is_drone_detected = detected;
            res.fundamental_bpf_hz = fundamental_bpf;
            res.estimated_rpm = (blades > 0) ? ((fundamental_bpf * 60.0) / blades) : 0.0;
            res.confidence = confidence;
            res.peak_snr_db = snr_db;
            res.spectral_q_factor = sharpness;
            res.spectral_flatness = spectral_flatness;
            res.peak_power = peak_power;
            res.noise_floor = median_noise;
            res.harmonic_score = std::min(1.0, (h2_ratio / 10.0) * 0.6 + (h3_ratio / 5.0) * 0.4);
            res.total_events = total_events;

            if (detected) {
                results.push_back(res);
            }
        }

        return results;
    }

    FlickerDetectionResult analyze_time_series(const std::vector<double> &time_series, int blades = 2) const {
        auto candidates = analyze_time_series_candidates(time_series, blades, 1);
        if (candidates.empty()) return {};
        return candidates[0];
    }

    const LensParameters& lens() const { return lens_params_; }
    void set_lens(const LensParameters &params) { lens_params_ = params; }
    void set_min_freq(double min_freq_hz) { min_freq_hz_ = min_freq_hz; }
    void set_min_events(double min_events) { min_events_threshold_ = min_events; }

private:
    double sample_rate_hz_;
    size_t window_size_;
    double min_freq_hz_;
    double max_freq_hz_;
    double min_events_threshold_;
    LensParameters lens_params_;
    std::vector<double> window_weights_;
};

/**
 * @brief Spatial Patch Grid Manager with Hierarchical Multi-Scale Pooling for IMX636 ($1280 \times 720$)
 */
class SpatialPatchGrid {
public:
    SpatialPatchGrid(int grid_cols = 32, int grid_rows = 18, double sample_rate_hz = 4000.0, size_t history_samples = 512)
        : grid_cols_(grid_cols),
          grid_rows_(grid_rows),
          num_cells_(grid_cols * grid_rows),
          sample_rate_hz_(sample_rate_hz),
          history_samples_(history_samples),
          bin_duration_us_(static_cast<uint64_t>(1000000.0 / sample_rate_hz)),
          current_window_start_us_(0),
          head_idx_(0) {
        
        cell_width_ = 1280.0 / grid_cols_;
        cell_height_ = 720.0 / grid_rows_;

        ring_buffers_.assign(num_cells_ * history_samples_, 0.0);
        cell_total_events_.assign(num_cells_, 0.0);
    }

    void ingest_event(int x, int y, uint64_t timestamp_us) {
        if (current_window_start_us_ == 0) {
            current_window_start_us_ = timestamp_us;
        }

        if (timestamp_us >= current_window_start_us_ + bin_duration_us_) {
            uint64_t elapsed_us = timestamp_us - current_window_start_us_;
            uint64_t steps = elapsed_us / bin_duration_us_;
            if (steps > history_samples_) {
                steps = history_samples_;
                current_window_start_us_ = timestamp_us;
            } else {
                current_window_start_us_ += steps * bin_duration_us_;
            }
            advance_temporal_bins(steps);
        }

        int col = std::clamp(static_cast<int>(x / cell_width_), 0, grid_cols_ - 1);
        int row = std::clamp(static_cast<int>(y / cell_height_), 0, grid_rows_ - 1);
        int cell_idx = row * grid_cols_ + col;

        size_t idx = cell_idx * history_samples_ + head_idx_;
        ring_buffers_[idx] += 1.0;
        cell_total_events_[cell_idx] += 1.0;
    }

    void advance_temporal_bin() {
        advance_temporal_bins(1);
    }

    void advance_temporal_bins(size_t steps) {
        for (size_t s = 0; s < steps; ++s) {
            head_idx_ = (head_idx_ + 1) % history_samples_;
            for (int i = 0; i < num_cells_; ++i) {
                size_t idx = i * history_samples_ + head_idx_;
                cell_total_events_[i] -= ring_buffers_[idx];
                if (cell_total_events_[i] < 0.0) cell_total_events_[i] = 0.0;
                ring_buffers_[idx] = 0.0;
            }
        }
    }

    bool is_cell_active(int col, int row, double min_threshold) const {
        return cell_total_events_[row * grid_cols_ + col] >= min_threshold;
    }

    std::vector<double> get_cell_history(int col, int row) const {
        std::vector<double> hist(history_samples_, 0.0);
        int cell_idx = row * grid_cols_ + col;
        size_t base_idx = cell_idx * history_samples_;
        for (size_t t = 0; t < history_samples_; ++t) {
            size_t ring_k = (head_idx_ + t) % history_samples_;
            hist[t] = ring_buffers_[base_idx + ring_k];
        }
        return hist;
    }

    bool is_pooled_patch_active(int col, int row, double min_threshold) const {
        int cols_to_pool = (col + 1 < grid_cols_) ? 2 : 1;
        int rows_to_pool = (row + 1 < grid_rows_) ? 2 : 1;
        double sum = 0.0;
        for (int r = 0; r < rows_to_pool; ++r) {
            for (int c = 0; c < cols_to_pool; ++c) {
                sum += cell_total_events_[(row + r) * grid_cols_ + (col + c)];
            }
        }
        return sum >= min_threshold;
    }

    std::vector<double> get_pooled_patch_history(int col, int row) const {
        std::vector<double> pooled(history_samples_, 0.0);
        int cols_to_pool = (col + 1 < grid_cols_) ? 2 : 1;
        int rows_to_pool = (row + 1 < grid_rows_) ? 2 : 1;

        for (int r = 0; r < rows_to_pool; ++r) {
            for (int c = 0; c < cols_to_pool; ++c) {
                int cell_idx = (row + r) * grid_cols_ + (col + c);
                size_t base_idx = cell_idx * history_samples_;
                for (size_t t = 0; t < history_samples_; ++t) {
                    size_t ring_k = (head_idx_ + t) % history_samples_;
                    pooled[t] += ring_buffers_[base_idx + ring_k];
                }
            }
        }
        return pooled;
    }

    int grid_cols() const { return grid_cols_; }
    int grid_rows() const { return grid_rows_; }
    double cell_width() const { return cell_width_; }
    double cell_height() const { return cell_height_; }

    void get_patch_center(int col, int row, double &center_x, double &center_y) const {
        center_x = (col + 0.5) * cell_width_;
        center_y = (row + 0.5) * cell_height_;
    }

    void get_pooled_patch_center(int col, int row, double &center_x, double &center_y) const {
        center_x = (col + 1.0) * cell_width_;
        center_y = (row + 1.0) * cell_height_;
    }

    /**
     * @brief Spatially remaps active 512-sample temporal ring buffers using homography H_shift (t_new -> t_old)
     * Preserves continuous harmonic time series across dynamic camera rotations without phase disruption.
     */
    void remap_grid(const Matrix3x3& H_shift) {
        Matrix3x3 H_inv = H_shift.inverse();
        std::vector<double> new_ring_buffers(num_cells_ * history_samples_, 0.0);
        std::vector<double> new_cell_total_events(num_cells_, 0.0);

        for (int r = 0; r < grid_rows_; ++r) {
            for (int c = 0; c < grid_cols_; ++c) {
                double cx = (c + 0.5) * cell_width_;
                double cy = (r + 0.5) * cell_height_;
                Vector3d p_new(cx, cy, 1.0);
                Vector3d p_old = H_inv * p_new;
                if (std::abs(p_old.z) > 1e-6) {
                    double ox = p_old.x / p_old.z;
                    double oy = p_old.y / p_old.z;
                    int oc = static_cast<int>(ox / cell_width_);
                    int orow = static_cast<int>(oy / cell_height_);
                    if (oc >= 0 && oc < grid_cols_ && orow >= 0 && orow < grid_rows_) {
                        int old_cell_idx = orow * grid_cols_ + oc;
                        int new_cell_idx = r * grid_cols_ + c;
                        std::memcpy(&new_ring_buffers[new_cell_idx * history_samples_],
                                    &ring_buffers_[old_cell_idx * history_samples_],
                                    history_samples_ * sizeof(double));
                        new_cell_total_events[new_cell_idx] = cell_total_events_[old_cell_idx];
                    }
                }
            }
        }

        ring_buffers_ = std::move(new_ring_buffers);
        cell_total_events_ = std::move(new_cell_total_events);
    }

private:
    int grid_cols_;
    int grid_rows_;
    int num_cells_;
    double cell_width_;
    double cell_height_;
    double sample_rate_hz_;
    size_t history_samples_;
    uint64_t bin_duration_us_;
    uint64_t current_window_start_us_;
    size_t head_idx_;

    std::vector<double> ring_buffers_;
    std::vector<double> cell_total_events_;
};

/**
 * @brief Global Common-Mode, Spatial NMS, and M-of-N Track State Machine
 */
class SpatialFlickerClusterer {
public:
    enum class TrackState {
        TENTATIVE,
        CONFIRMED,
        DELETED
    };

    struct Track {
        int track_id{0};
        TrackState state{TrackState::TENTATIVE};
        FlickerDetectionResult last_detection;
        int hit_count{1};
        int miss_count{0};
        int total_age{1};
    };

    static std::vector<FlickerDetectionResult> filter_and_cluster(
        const std::vector<FlickerDetectionResult>& raw_candidates,
        int max_common_mode_patches = 6) {
        
        // 1. Group candidates by frequency bin (bin size 6 Hz)
        std::unordered_map<int, std::vector<FlickerDetectionResult>> freq_groups;
        for (const auto& c : raw_candidates) {
            int freq_bin = static_cast<int>(std::round(c.fundamental_bpf_hz / 6.0));
            freq_groups[freq_bin].push_back(c);
        }

        // 2. Spatial Density Clustering & Clutter Filter:
        // Distinguish diffuse ambient flutter across the sensor from localized drone airframe clusters
        std::vector<FlickerDetectionResult> localized_candidates;
        for (const auto& [freq_bin, cands] : freq_groups) {
            if (cands.empty()) continue;

            double freq_hz = cands[0].fundamental_bpf_hz;
            bool is_ac_carrier = (std::abs(freq_hz - 100.0) < 4.0) ||
                                 (std::abs(freq_hz - 120.0) < 4.0) ||
                                 (std::abs(freq_hz - 150.0) < 4.0) ||
                                 (std::abs(freq_hz - 180.0) < 4.0) ||
                                 (std::abs(freq_hz - 200.0) < 4.0) ||
                                 (std::abs(freq_hz - 240.0) < 4.0) ||
                                 (std::abs(freq_hz - 300.0) < 4.0);

            // Group candidates into spatial clusters (neighbors within 160px)
            std::vector<std::vector<FlickerDetectionResult>> clusters;
            std::vector<bool> visited(cands.size(), false);

            for (size_t i = 0; i < cands.size(); ++i) {
                if (visited[i]) continue;
                visited[i] = true;
                std::vector<FlickerDetectionResult> cluster = { cands[i] };

                for (size_t j = i + 1; j < cands.size(); ++j) {
                    if (visited[j]) continue;
                    double dx = (cands[i].patch_x - cands[j].patch_x) * 40.0;
                    double dy = (cands[i].patch_y - cands[j].patch_y) * 40.0;
                    double dist = std::sqrt(dx * dx + dy * dy);
                    if (dist <= 160.0) {
                        visited[j] = true;
                        cluster.push_back(cands[j]);
                    }
                }
                clusters.push_back(cluster);
            }

            // For each spatial cluster:
            // 1. Suppress widespread AC powerline flutter (100/120/150/180/200/240/300 Hz across multiple locations)
            // 2. Suppress isolated single-point AC powerline glints (cl.size() <= 2 on AC harmonic)
            // 3. Suppress isolated single-cell diffuse flutter (cl.size() == 1 with low confidence)
            // 4. Retain real localized multi-rotor drone clusters and high-confidence rotor hits
            bool diffuse_flutter = (clusters.size() >= 5);
            for (const auto& cl : clusters) {
                if (is_ac_carrier && (clusters.size() >= 2 || cl.size() <= 2)) {
                    continue; // Suppress global AC mains lighting and single-point AC powerline glints
                }
                // For extreme-range standoff targets (100m - 300m), all rotors fall into a single 40x40 cell (cl.size() == 1).
                // Retain single-cell candidates if confirmed by low DDHF spectral flatness (gamma <= 0.18) or high confidence.
                if (diffuse_flutter && cl.size() == 1 && cl[0].confidence < 0.70 && cl[0].spectral_flatness > 0.18) {
                    continue; // Suppress isolated single-cell diffuse background noise
                }
                for (const auto& c : cl) {
                    localized_candidates.push_back(c);
                }
            }
        }

        // 3. Drone-Level Airframe Cluster & Multi-Rotor Fusion
        std::vector<FlickerDetectionResult> fused_results;
        if (!localized_candidates.empty()) {
            std::vector<bool> merged(localized_candidates.size(), false);

            for (size_t i = 0; i < localized_candidates.size(); ++i) {
                if (merged[i]) continue;
                auto cluster = localized_candidates[i];
                double weight_sum = cluster.confidence;
                double sum_x = cluster.centroid_px_x * cluster.confidence;
                double sum_y = cluster.centroid_px_y * cluster.confidence;
                double max_snr = cluster.peak_snr_db;
                int rotor_hits = 1;

                for (size_t j = i + 1; j < localized_candidates.size(); ++j) {
                    if (merged[j]) continue;
                    const auto& b = localized_candidates[j];

                    double dx = cluster.centroid_px_x - b.centroid_px_x;
                    double dy = cluster.centroid_px_y - b.centroid_px_y;
                    double dist = std::sqrt(dx * dx + dy * dy);

                    // Airframe cluster radius (140 pixels spans the full quadcopter frame at 15-60ft)
                    if (dist < 140.0) {
                        merged[j] = true;
                        sum_x += b.centroid_px_x * b.confidence;
                        sum_y += b.centroid_px_y * b.confidence;
                        weight_sum += b.confidence;
                        max_snr = std::max(max_snr, b.peak_snr_db);
                        rotor_hits++;
                    }
                }

                if (weight_sum > 0.0) {
                    cluster.centroid_px_x = static_cast<int>(std::round(sum_x / weight_sum));
                    cluster.centroid_px_y = static_cast<int>(std::round(sum_y / weight_sum));
                }
                cluster.peak_snr_db = max_snr;
                // Multi-rotor confidence fusion: combining signals across rotors elevates confidence
                cluster.confidence = std::min(1.0, cluster.confidence + (rotor_hits - 1) * 0.18);
                fused_results.push_back(cluster);
            }
        }

        return update_tracker(fused_results);
    }

    static void reset_tracker() {
        get_tracks().clear();
    }

private:
    static std::vector<Track>& get_tracks() {
        static std::vector<Track> s_tracks;
        return s_tracks;
    }

    static std::vector<FlickerDetectionResult> update_tracker(const std::vector<FlickerDetectionResult>& current_dets) {
        auto& s_tracks = get_tracks();
        static int s_next_id = 1;

        const int M_HITS_FOR_CONFIRM = 3;  // Robust 3 frame confirmation
        const int MAX_COAST_FRAMES   = 5;  // Coast 5 frames (200ms) across sparse blade sweeps
        const int MAX_TENTATIVE_MISS = 3;  // Allow 3 miss frames (120ms) for tentative tracks to bridge standoff intermittency

        std::vector<bool> matched_curr(current_dets.size(), false);
        std::vector<Track> next_tracks;

        // 1. Associate current detections with existing tracks
        for (auto& track : s_tracks) {
            bool found_match = false;
            size_t best_det_idx = 0;
            double min_dist = 1e9;

            for (size_t i = 0; i < current_dets.size(); ++i) {
                if (matched_curr[i]) continue;
                const auto& d = current_dets[i];

                // Invariant Association Gate:
                // 1. World patch space: patch grid cells are 40x40 px in stabilized world coordinates.
                //    Hovering/moving drones stay within 2-3 cells (80-120 px in world frame) regardless of camera pan speed!
                double world_dx = (track.last_detection.patch_x - d.patch_x) * 40.0;
                double world_dy = (track.last_detection.patch_y - d.patch_y) * 40.0;
                double dist_world = std::sqrt(world_dx * world_dx + world_dy * world_dy);

                // 2. Camera pixel space: for static hovering / small displacements
                double cam_dx = track.last_detection.centroid_px_x - d.centroid_px_x;
                double cam_dy = track.last_detection.centroid_px_y - d.centroid_px_y;
                double dist_cam = std::sqrt(cam_dx * cam_dx + cam_dy * cam_dy);

                // 3. Angular Bearing Space: Azimuth and Elevation angular distance (degrees)
                // 140px on 12mm lens (~2469 px/rad) corresponds to ~3.25 degrees; on 8mm was ~4.87 degrees
                double d_az = track.last_detection.azimuth_deg - d.azimuth_deg;
                double d_el = track.last_detection.elevation_deg - d.elevation_deg;
                double dist_bearing_deg = std::sqrt(d_az * d_az + d_el * d_el);
                double dist_bearing_equiv_px = dist_bearing_deg * ((12.0 * 1000.0 / 4.86) * (M_PI / 180.0)); // 43.095 px/deg

                double min_assoc_dist = std::min({dist_world, dist_cam, dist_bearing_equiv_px});

                // Association gate: within 140 pixels (or 220 px during motion / standoff coasting) and +- 8 Hz frequency consistency
                double assoc_gate_px = 140.0;
                if (track.state == TrackState::CONFIRMED || track.miss_count > 0) {
                    assoc_gate_px = 220.0; // Expand to 5.1 degrees to maintain lock during fast camera tracking and hover drift
                }

                if (min_assoc_dist < assoc_gate_px && std::abs(track.last_detection.fundamental_bpf_hz - d.fundamental_bpf_hz) < 8.0) {
                    if (min_assoc_dist < min_dist) {
                        min_dist = min_assoc_dist;
                        best_det_idx = i;
                        found_match = true;
                    }
                }
            }

            if (found_match) {
                matched_curr[best_det_idx] = true;
                const auto& d = current_dets[best_det_idx];

                // Update track state
                track.last_detection = d;
                track.last_detection.track_id = track.track_id;
                track.hit_count++;
                track.miss_count = 0;
                track.total_age++;

                if (track.state == TrackState::TENTATIVE) {
                    if (track.hit_count >= M_HITS_FOR_CONFIRM || 
                        (track.hit_count >= 2 && track.last_detection.is_neural_detection && track.last_detection.confidence >= 0.70)) {
                        track.state = TrackState::CONFIRMED;
                    }
                }
                track.last_detection.track_state = (track.state == TrackState::CONFIRMED) ? 2 : 1;
                track.last_detection.hit_count = track.hit_count;
                track.last_detection.miss_count = 0;
                next_tracks.push_back(track);
            } else {
                track.miss_count++;
                track.total_age++;
                track.last_detection.track_id = track.track_id;
                track.last_detection.hit_count = track.hit_count;
                track.last_detection.miss_count = track.miss_count;
                track.last_detection.track_state = (track.state == TrackState::CONFIRMED) ? 3 : 1;

                if (track.state == TrackState::CONFIRMED) {
                    if (track.miss_count <= MAX_COAST_FRAMES) {
                        next_tracks.push_back(track); // Coast confirmed track
                    }
                } else if (track.state == TrackState::TENTATIVE) {
                    if (track.miss_count <= MAX_TENTATIVE_MISS) {
                        next_tracks.push_back(track);
                    }
                }
            }
        }

        // 2. Instantiate new tentative tracks for unmatched detections
        for (size_t i = 0; i < current_dets.size(); ++i) {
            if (!matched_curr[i]) {
                Track t;
                t.track_id = s_next_id++;
                t.state = TrackState::TENTATIVE;
                t.last_detection = current_dets[i];
                t.last_detection.track_id = t.track_id;
                t.last_detection.track_state = 1;
                t.last_detection.hit_count = 1;
                t.last_detection.miss_count = 0;
                t.hit_count = 1;
                t.miss_count = 0;
                t.total_age = 1;
                next_tracks.push_back(t);
            }
        }

        s_tracks = next_tracks;

        // 3. Return only CONFIRMED tracks sorted by confidence * SNR
        std::vector<Track> confirmed_tracks;
        for (const auto& t : s_tracks) {
            if (t.state == TrackState::CONFIRMED && t.last_detection.confidence >= 0.55) {
                // Filter unreinforced optical border noise (X < 60 or X > 1220, Y < 40 or Y > 680)
                int cx = t.last_detection.centroid_px_x;
                int cy = t.last_detection.centroid_px_y;
                bool is_border = (cx < 60 || cx > 1220 || cy < 40 || cy > 680);
                if (is_border && t.last_detection.confidence < 0.70) {
                    continue; // Suppress extreme edge noise
                }
                confirmed_tracks.push_back(t);
            }
        }

        std::sort(confirmed_tracks.begin(), confirmed_tracks.end(), [](const Track& a, const Track& b) {
            return (a.last_detection.confidence * a.last_detection.peak_snr_db) > 
                   (b.last_detection.confidence * b.last_detection.peak_snr_db);
        });

        std::vector<FlickerDetectionResult> confirmed_results;
        for (const auto& t : confirmed_tracks) {
            confirmed_results.push_back(t.last_detection);
            if (confirmed_results.size() >= 3) break;
        }

        return confirmed_results;
    }

public:
    static std::vector<Track> get_all_tracks() {
        return get_tracks();
    }
};

} // namespace predator

#endif // PREDATOR_FLICKER_DSP_HPP
