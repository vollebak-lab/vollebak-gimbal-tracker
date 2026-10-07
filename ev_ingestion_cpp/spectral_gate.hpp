#ifndef PREDATOR_SPECTRAL_GATE_HPP
#define PREDATOR_SPECTRAL_GATE_HPP

/**
 * @file spectral_gate.hpp
 * @brief Single source of truth for per-cell spectral detection gates (Phase 33.5).
 *
 * The same evaluate_spectral_gate() runs inside the CUDA kernel (all 1152 cells) and on the host
 * (ROI diagnostics, unit tests), so diagnostics can never drift from the detector again.
 *
 * Detection statistic (CFAR):
 *   Under background shot noise, a Hann-windowed event-count periodogram bin normalized by the
 *   mean noise power is ~Exp(1). The detector tests ONE bin per cell (the HPS-selected comb line),
 *   and P(selected bin > eta) <= P(any searched bin > eta) <= N_bins * exp(-eta). Over all cells
 *   (union bound) the field-wide false-alarm probability per analysis window is
 *       Pfa_window <= N_cells * N_bins * exp(-eta)   =>   eta = ln(N_cells * N_bins / Pfa_window).
 *   The bound is conservative: Hann-correlated adjacent bins and overlapping pooled cells make the
 *   effective number of independent tests smaller.
 *
 * Why the budget is per WINDOW, not per frame:
 *   Analysis runs at 25 Hz on a 128 ms window, so one noise excursion persists for ~3 frames, and
 *   the tracker confirms on 2 hits at >= 10 dB. A single noise excursion can therefore confirm a
 *   track; the single-window threshold alone must meet the confirmed-false-alarm budget.
 *   Pfa_window = false_alarms_per_hour * window_s / 3600.
 */

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#if defined(__CUDACC__)
#define PREDATOR_HD __host__ __device__
#else
#define PREDATOR_HD
#endif

namespace predator {

constexpr int kSpectrumBins = 257;   ///< rFFT bins for the 512-sample window
constexpr int kMaxNoiseBins = 128;   ///< Scratch capacity for the median noise estimate

/// Outcome of the gate chain for one cell (first failing gate, or Pass).
enum class GateVerdict : uint8_t {
    Pass = 0,
    LowActivity,   ///< Fewer events than min_events (compute pre-filter, not an FA control)
    NoPeak,        ///< No local maximum in the search band
    BelowCfar,     ///< Peak / mean noise below the CFAR threshold
    Broad,         ///< Peak not narrower than its +-2-bin neighbours (colored / broadband clutter)
    OutOfBand,     ///< Refined fundamental outside [min_freq_hz, max_freq_hz]
    TextureScan    ///< Broad peak inside the gyro-predicted background texture band
};

inline const char* gate_verdict_name(GateVerdict v) {
    switch (v) {
        case GateVerdict::Pass:        return "PASS";
        case GateVerdict::LowActivity: return "FAIL_LOW_ACTIVITY";
        case GateVerdict::NoPeak:      return "FAIL_NO_PEAK";
        case GateVerdict::BelowCfar:   return "FAIL_CFAR";
        case GateVerdict::Broad:       return "FAIL_SHARPNESS";
        case GateVerdict::OutOfBand:   return "FAIL_OUT_OF_BAND";
        case GateVerdict::TextureScan: return "FAIL_TEXTURE_SCAN";
    }
    return "UNKNOWN";
}

/**
 * @brief Detector gate configuration. Set the user fields, then call derive_spectral_gate()
 *        (or CudaFlickerCore::set_spectral_gate_config) to compute the derived fields.
 */
struct SpectralGateConfig {
    // ---- User fields ----
    float min_freq_hz{75.0f};            ///< Blade-pass search band lower edge
    float max_freq_hz{1000.0f};          ///< Blade-pass search band upper edge
    float false_alarms_per_hour{6.0f};   ///< Field-wide single-window FA budget (6/h = 1 per 10 min)
    float min_events{6.0f};              ///< Compute pre-filter only; FA control is the CFAR threshold
    float min_sharpness{1.5f};           ///< Peak / mean(P[k-2], P[k+2]); rejects broad colored-noise bumps
    float subharmonic_power_ratio{0.04f}; ///< Sub-peak power >= ratio * comb peak (HPS octave check: 0.2 amplitude)
    float subharmonic_min_snr{10.0f};    ///< ...AND sub-peak >= this x mean noise (10 dB): noise cannot demote
    int noise_bin_lo{5};                 ///< Median noise band (matches SpectralCombNet normalization)
    int noise_bin_hi{128};

    // ---- Derived fields (derive_spectral_gate) ----
    int min_bin{0};
    int max_bin{0};
    float bin_hz{0.0f};
    float cfar_threshold{0.0f};          ///< Linear threshold on peak / mean noise
    float cfar_threshold_db{0.0f};
    float pfa_per_window{0.0f};
};

/**
 * @brief Validates the user fields and computes the derived CFAR threshold.
 * @param cfg            User configuration (copied).
 * @param sample_rate_hz Ring-buffer sample rate (4000 Hz).
 * @param fft_len        Window length in samples (512).
 * @param num_cells      Number of independently tested cells (1152 = 576 base + 576 pooled).
 * @throws std::invalid_argument if the configuration is inconsistent.
 */
inline SpectralGateConfig derive_spectral_gate(SpectralGateConfig cfg, double sample_rate_hz,
                                               int fft_len, int num_cells) {
    if (sample_rate_hz <= 0.0 || fft_len != 2 * (kSpectrumBins - 1) || num_cells <= 0) {
        throw std::invalid_argument("spectral gate: invalid sample rate / fft length / cell count");
    }
    if (!(cfg.false_alarms_per_hour > 0.0f) || !(cfg.min_sharpness >= 1.0f) ||
        !(cfg.subharmonic_power_ratio > 0.0f && cfg.subharmonic_power_ratio < 1.0f) ||
        !(cfg.subharmonic_min_snr >= 1.0f) || cfg.min_events < 0.0f) {
        throw std::invalid_argument("spectral gate: false_alarms_per_hour must be > 0, min_sharpness >= 1, "
                                    "0 < subharmonic_power_ratio < 1, subharmonic_min_snr >= 1, min_events >= 0");
    }
    const double df = sample_rate_hz / fft_len;
    cfg.bin_hz = static_cast<float>(df);
    cfg.min_bin = static_cast<int>(std::floor(cfg.min_freq_hz / df));
    cfg.max_bin = static_cast<int>(std::ceil(cfg.max_freq_hz / df));
    if (cfg.min_bin < 2) cfg.min_bin = 2;                         // P[k-2] must exist
    if (cfg.max_bin > kSpectrumBins - 3) cfg.max_bin = kSpectrumBins - 3;  // P[k+2] must exist
    if (cfg.min_bin >= cfg.max_bin) {
        throw std::invalid_argument("spectral gate: empty search band");
    }
    if (cfg.noise_bin_lo < 1 || cfg.noise_bin_hi >= kSpectrumBins || cfg.noise_bin_lo >= cfg.noise_bin_hi ||
        cfg.noise_bin_hi - cfg.noise_bin_lo + 1 > kMaxNoiseBins) {
        throw std::invalid_argument("spectral gate: noise band must lie in [1, 256] and span <= " +
                                    std::to_string(kMaxNoiseBins) + " bins");
    }
    const double window_s = fft_len / sample_rate_hz;
    const double pfa_window = static_cast<double>(cfg.false_alarms_per_hour) * window_s / 3600.0;
    const double n_tests = static_cast<double>(num_cells) * (cfg.max_bin - cfg.min_bin + 1);
    if (pfa_window >= n_tests) {
        throw std::invalid_argument("spectral gate: false-alarm budget exceeds the number of tests");
    }
    const double pfa_bin = pfa_window / n_tests;

    // Order-statistic CFAR (Rohling 1983). The noise mean is NOT known: it is 1/ln2 x the median
    // of the reference bins, itself a random variable. For iid Exp reference cells, a test
    // X > T * X_(k) (k-th smallest of N) has Pfa(T) = prod_{i=0}^{k-1} (N - i) / (N - i + T).
    // Reference bins: the noise band minus the worst-case exclusions (5 guard regions x 5 bins).
    // Hann windowing correlates neighbouring bins, so N is replaced by an effective count
    // N_eff = N / (1 + 2 * sum_m |rho_m|^2), rho_m = sum w^2 e^{-j 2 pi m n / L} / sum w^2,
    // computed from the runtime window (symmetric Hann, L-1 denominator).
    double s0 = 0.0, s_re[4] = {0, 0, 0, 0}, s_im[4] = {0, 0, 0, 0};
    for (int n = 0; n < fft_len; ++n) {
        const double w = 0.5 * (1.0 - std::cos(2.0 * M_PI * n / (fft_len - 1)));
        const double w2 = w * w;
        s0 += w2;
        for (int m = 1; m <= 3; ++m) {
            s_re[m] += w2 * std::cos(2.0 * M_PI * m * n / fft_len);
            s_im[m] += w2 * std::sin(2.0 * M_PI * m * n / fft_len);
        }
    }
    double rho_sq_sum = 0.0;
    for (int m = 1; m <= 3; ++m) rho_sq_sum += (s_re[m] * s_re[m] + s_im[m] * s_im[m]) / (s0 * s0);
    const int n_ref_min = (cfg.noise_bin_hi - cfg.noise_bin_lo + 1) - 25;
    if (n_ref_min < 16) {
        throw std::invalid_argument("spectral gate: noise band too narrow for a stable median");
    }
    const int n_eff = static_cast<int>(std::floor(n_ref_min / (1.0 + 2.0 * rho_sq_sum)));
    const int k_eff = n_eff / 2 + 1;  // runtime takes element n/2 (0-based) => (n/2 + 1)-th smallest
    auto os_cfar_log_pfa = [n_eff, k_eff](double T) {
        double lp = 0.0;
        for (int i = 0; i < k_eff; ++i) lp += std::log((n_eff - i) / (n_eff - i + T));
        return lp;
    };
    // Pfa(T) is strictly decreasing in T: bisection on log Pfa.
    const double target = std::log(pfa_bin);
    double lo = 0.0, hi = 1.0;
    while (os_cfar_log_pfa(hi) > target) {
        hi *= 2.0;
        if (hi > 1e9) throw std::invalid_argument("spectral gate: false-alarm budget unreachable");
    }
    for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        (os_cfar_log_pfa(mid) > target ? lo : hi) = mid;
    }
    // Runtime statistic is X / (median / ln2), so threshold on it is T * ln2.
    const double eta = hi * 0.6931471805599453;
    cfg.pfa_per_window = static_cast<float>(pfa_window);
    cfg.cfar_threshold = static_cast<float>(eta);
    cfg.cfar_threshold_db = static_cast<float>(10.0 * std::log10(eta));
    return cfg;
}

/// Per-cell gate output.
struct SpectralGateResult {
    GateVerdict verdict{GateVerdict::NoPeak};
    int peak_bin{-1};             ///< Strongest comb line (detection statistic bin)
    int fund_bin{-1};             ///< Fundamental after subharmonic demotion
    float fundamental_hz{0.0f};   ///< Parabolic-refined fundamental
    float peak_power{0.0f};
    float mean_noise{0.0f};
    float snr_linear{0.0f};
    float snr_db{0.0f};
    float sharpness{0.0f};
    float flatness{1.0f};
    float confidence{0.0f};
};

/// In-place quickselect (Hoare partition): returns the k-th smallest of a[0..n-1]. Device-safe.
PREDATOR_HD inline float quickselect_kth(float* a, int n, int k) {
    int left = 0, right = n - 1;
    while (left < right) {
        const float pivot = a[k];
        int i = left, j = right;
        while (i <= j) {
            while (a[i] < pivot) ++i;
            while (a[j] > pivot) --j;
            if (i <= j) {
                const float tmp = a[i]; a[i] = a[j]; a[j] = tmp;
                ++i; --j;
            }
        }
        if (k <= j) right = j;
        else if (k >= i) left = i;
        else break;
    }
    return a[k];
}

PREDATOR_HD inline bool near_bin(int k, int center, int guard) {
    const int d = k - center;
    return center > 0 && d >= -guard && d <= guard;
}

/**
 * @brief Evaluates the full gate chain on one cell's 257-bin power spectrum.
 * @param power        Power spectrum |X_k|^2, k = 0..256.
 * @param total_events Events accumulated in the cell's window.
 * @param sieve_hits   Max consecutive periodic hits in the cell (confidence feature only).
 * @param gyro_deg_s   Camera angular speed (texture-scan filter; 0 when unknown).
 * @param cfg          Derived configuration (derive_spectral_gate()).
 */
PREDATOR_HD inline SpectralGateResult evaluate_spectral_gate(const float* power, float total_events,
                                                             uint32_t sieve_hits, float gyro_deg_s,
                                                             const SpectralGateConfig& cfg) {
    SpectralGateResult r;
    if (total_events < cfg.min_events) {
        r.verdict = GateVerdict::LowActivity;
        return r;
    }

    // Spectral flatness over the search band (telemetry + confidence feature; not a gate).
    float sum_log = 0.0f, sum_p = 0.0f;
    const int n_band = cfg.max_bin - cfg.min_bin + 1;
    for (int k = cfg.min_bin; k <= cfg.max_bin; ++k) {
        sum_log += logf(power[k] + 1e-9f);
        sum_p += power[k];
    }
    const float arith_mean = sum_p / static_cast<float>(n_band);
    r.flatness = (arith_mean > 1e-9f) ? expf(sum_log / static_cast<float>(n_band)) / arith_mean : 1.0f;

    // Harmonic Product Spectrum over local maxima: selects the strongest harmonic comb line.
    int best = -1;
    float max_hps = 0.0f;
    for (int k = cfg.min_bin; k <= cfg.max_bin; ++k) {
        const float p1 = power[k];
        if (p1 <= power[k - 1] || p1 <= power[k + 1]) continue;
        const float p2 = (2 * k < kSpectrumBins) ? power[2 * k] : 0.0f;
        const float p3 = (3 * k < kSpectrumBins) ? power[3 * k] : 0.0f;
        const float hps = p1 + 0.6f * p2 + 0.4f * p3;
        if (hps > max_hps) { max_hps = hps; best = k; }
    }
    if (best < 0) {
        r.verdict = GateVerdict::NoPeak;
        return r;
    }
    // Detection statistic = strongest in-band comb member of the HPS selection. Harmonics of a
    // non-integer bin land at h*k +- 1, so each is searched over +-1 bin. Only searched bins are
    // eligible, so the union bound over the searched band still covers this choice.
    const int hps_bin = best;
    for (int h = 2; h <= 3; ++h) {
        for (int kh = h * hps_bin - 1; kh <= h * hps_bin + 1; ++kh) {
            if (kh <= cfg.max_bin && power[kh] > power[best] &&
                power[kh] > power[kh - 1] && power[kh] > power[kh + 1]) {
                best = kh;
            }
        }
    }
    r.peak_bin = best;
    r.peak_power = power[best];

    // Median noise floor over the noise band, excluding +-2 bins around the comb line, its
    // 2nd/3rd harmonics and its 1/2, 1/3 subharmonics (candidate fundamentals).
    float noise[kMaxNoiseBins];
    int n_noise = 0;
    const int sub2 = (best + 1) / 2, sub3 = (best + 1) / 3;
    for (int k = cfg.noise_bin_lo; k <= cfg.noise_bin_hi && n_noise < kMaxNoiseBins; ++k) {
        if (near_bin(k, best, 2) || near_bin(k, 2 * best, 2) || near_bin(k, 3 * best, 2) ||
            near_bin(k, sub2, 2) || near_bin(k, sub3, 2)) continue;
        noise[n_noise++] = power[k];
    }
    const float median = (n_noise > 0) ? quickselect_kth(noise, n_noise, n_noise / 2) : 0.0f;
    // Exp(1): mean = median / ln 2
    r.mean_noise = fmaxf(median * 1.442695f, 1e-9f);
    r.snr_linear = r.peak_power / r.mean_noise;
    r.snr_db = 10.0f * log10f(fmaxf(1.0f, r.snr_linear));
    if (r.snr_linear < cfg.cfar_threshold) {
        r.verdict = GateVerdict::BelowCfar;
        return r;
    }

    // Shape test: a tonal line is narrower than its +-2-bin neighbourhood (Hann main lobe = +-2 bins).
    const float neighbor = 0.5f * (power[best - 2] + power[best + 2]);
    r.sharpness = (neighbor > 1e-9f) ? (r.peak_power / neighbor) : 1e9f;
    if (r.sharpness < cfg.min_sharpness) {
        r.verdict = GateVerdict::Broad;
        return r;
    }

    // Iterative subharmonic demotion (octave-error correction, e.g. 800 -> 400 -> 200 Hz). A
    // sub-line demotes the fundamental when it is (a) >= subharmonic_power_ratio of the comb peak
    // (standard HPS check, 0.2 in amplitude) AND (b) itself significant, >= subharmonic_min_snr x
    // mean noise, so noise spikes near f/2 cannot demote weak targets. Demotion only changes the
    // reported frequency, never the detect decision.
    int fund = best;
    const float demote_min = fmaxf(cfg.subharmonic_power_ratio * r.peak_power,
                                   cfg.subharmonic_min_snr * r.mean_noise);
    for (int iter = 0; iter < 3; ++iter) {
        int found = -1;
        for (int sub = 3; sub >= 2 && found < 0; --sub) {
            const int c = (fund + sub / 2) / sub;
            float max_p = 0.0f;
            for (int k = c - 1; k <= c + 1; ++k) {
                if (k < cfg.min_bin || k > cfg.max_bin) continue;
                const float p = power[k];
                if (p >= demote_min && p > power[k - 1] && p > power[k + 1] && p > max_p) {
                    max_p = p;
                    found = k;
                }
            }
        }
        if (found < 0) break;
        fund = found;
    }
    r.fund_bin = fund;

    // Parabolic sub-bin refinement of the fundamental.
    const float pl = power[fund - 1], pm = power[fund], pr = power[fund + 1];
    const float denom = 2.0f * (2.0f * pm - pl - pr);
    const float delta = (fabsf(denom) > 1e-9f) ? (pr - pl) / denom : 0.0f;
    r.fundamental_hz = (static_cast<float>(fund) + delta) * cfg.bin_hz;
    if (r.fundamental_hz < cfg.min_freq_hz || r.fundamental_hz > cfg.max_freq_hz) {
        r.verdict = GateVerdict::OutOfBand;
        return r;
    }

    // Motion-induced background texture filter (unchanged from Phase 32; ego-motion is deferred):
    // v_scan = fx * omega (px/s), texture band [0.22, 0.70] * v_scan. Rejects only broad clutter.
    if (gyro_deg_s > 4.0f) {
        const float v_scan = 1646.0f * gyro_deg_s * (3.14159265f / 180.0f);
        if (r.fundamental_hz >= 0.22f * v_scan && r.fundamental_hz <= 0.70f * v_scan &&
            r.sharpness < 2.5f && r.flatness > 0.20f && sieve_hits < 2) {
            r.verdict = GateVerdict::TextureScan;
            return r;
        }
    }

    // Confidence (ranking / fusion only; every Pass already meets the CFAR false-alarm budget).
    const float h2 = (2 * fund < kSpectrumBins) ? power[2 * fund] / r.mean_noise : 0.0f;
    const float h3 = (3 * fund < kSpectrumBins) ? power[3 * fund] / r.mean_noise : 0.0f;
    const float harmonic_bonus = (h2 > 1.5f ? 0.20f : 0.0f) + (h3 > 1.2f ? 0.15f : 0.0f);
    const float flatness_bonus = (r.flatness < 0.18f) ? 0.25f * (0.18f - r.flatness) / 0.18f : 0.0f;
    const float sieve_bonus = (sieve_hits >= 2) ? 0.15f : 0.0f;
    const float fund_score = fminf(1.0f, fmaxf(0.20f, (r.snr_db - 6.0f) / 10.0f));
    r.confidence = fminf(1.0f, fund_score + harmonic_bonus + flatness_bonus + sieve_bonus);
    r.verdict = GateVerdict::Pass;
    return r;
}

} // namespace predator

#endif // PREDATOR_SPECTRAL_GATE_HPP
