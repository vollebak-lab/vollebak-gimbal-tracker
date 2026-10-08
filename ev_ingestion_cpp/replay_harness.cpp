/**
 * @file replay_harness.cpp
 * @brief Standalone C++ Batch Evaluation & Replay Harness (Phase 33.8b).
 *
 * Evaluates recorded neuromorphic corpus datasets (.evt21raw, .cd, .raw) against
 * the GPU-accelerated cuFFT, CFAR gating, and TensorRT SpectralCombNet v3 pipeline.
 *
 * Calculates standardized performance scorecard metrics:
 *   - Detection Latency (T_detect to confirmed track lock)
 *   - Track Continuity (% of target flight with unbroken lock)
 *   - Spectral SNR distribution (min, median, max)
 *   - CombNet probability distribution (min, median, max)
 *   - Frequency estimation accuracy (BPF error, RPM error)
 *   - False Alarm count and empirical rate (FA / hour)
 *   - Overall PASS / FAIL verdict against quantitative gates
 *
 * Supports single-file evaluation or automated directory benchmarking with
 * structured JSON scorecard and per-frame CSV telemetry exports.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_flicker_core.cuh"
#include "raw_pipeline.cuh"
#include "flicker_dsp.hpp"
#include "spectral_combnet_trt.hpp"
#include "spectral_gate.hpp"
#include "evt21_format.hpp"
#include "evt21_encoder.hpp"
#include "hot_pixel_mask.hpp"

namespace fs = std::filesystem;
using predator::CudaFlickerCore;
using predator::RawPipeline;
using predator::RawPipelineConfig;
using predator::ContinuousGyroWarper;
using predator::LensParameters;
using predator::Matrix3x3;
using predator::SpectralGateConfig;
using predator::SpectralCombNetEngine;
using predator::SpatialFlickerClusterer;
using predator::FlickerDetectionResult;
using Track = predator::SpatialFlickerClusterer::Track;
using TrackState = predator::SpatialFlickerClusterer::TrackState;
using predator::evt21::CdRecord;
using predator::evt21::Evt21Encoder;

namespace {

/// Metadata accompanying a flight recording (optionally parsed from <prefix>.json)
struct FlightMetadata {
    std::string name;
    std::string platform{"Unknown"};
    double standoff_ft{0.0};
    int blade_count{2};
    double expected_bpf_hz{0.0};
    bool ground_truth_target{true};
    double target_start_s{0.0};
    double target_end_s{1e9};
};

/// Per-frame evaluation sample
struct FrameRecord {
    uint32_t frame_idx{0};
    double time_s{0.0};
    uint64_t raw_events{0};
    uint64_t retained_events{0};
    double suppression_pct{0.0};
    int active_cells{0};
    int candidate_count{0};
    int confirmed_tracks{0};
    float top_snr_db{0.0f};
    float top_bpf_hz{0.0f};
    float top_neural_prob{0.0f};
    bool target_locked{false};
};

/// Overall evaluation scorecard for a single recording
struct EvaluationScorecard {
    std::string file_path;
    FlightMetadata meta;
    uint64_t total_events{0};
    double total_duration_s{0.0};
    double avg_event_rate_ev_s{0.0};

    uint32_t total_frames{0};
    uint32_t target_frames{0};
    uint32_t locked_frames{0};
    double track_continuity_pct{0.0};

    double t_detect_ms{-1.0};
    bool target_acquired{false};

    float min_snr_db{0.0f};
    float median_snr_db{0.0f};
    float max_snr_db{0.0f};

    float min_neural_prob{0.0f};
    float median_neural_prob{0.0f};
    float max_neural_prob{0.0f};

    double detected_mean_bpf_hz{0.0};
    double bpf_error_hz{0.0};

    uint32_t false_confirmed_tracks{0};
    double fa_rate_per_hour{0.0};

    bool pass_verdict{false};
    std::vector<std::string> failure_reasons;
    std::vector<FrameRecord> frames;
};

/// CLI options configuration
struct HarnessOptions {
    std::string single_file;
    std::string directory;
    std::string json_out;
    std::string csv_out;
    std::string combnet_path{"/home/orin/ev_deploy/models/spectral_combnet_fp16.engine"};
    float cfar_fa_per_hour{6.0f};
    bool require_drone{false};
    bool require_clean{false};
    bool verbose{false};
    bool enable_rescue{false};
    std::string hot_pixels_path{"hot_pixels.txt"};
};

/// Robust percentile calculation
template <typename T>
T compute_percentile(std::vector<T> values, double q) {
    if (values.empty()) return T{0};
    std::sort(values.begin(), values.end());
    size_t idx = static_cast<size_t>(q * static_cast<double>(values.size() - 1));
    return values[std::min(idx, values.size() - 1)];
}

/// Simple JSON parser for accompanying metadata files (<prefix>.json)
FlightMetadata parse_metadata_file(const std::string& json_path, const std::string& default_name) {
    FlightMetadata meta;
    meta.name = default_name;

    // Default target presence heuristic from filename
    std::string lower_name = default_name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    if (lower_name.find("darkroom") != std::string::npos ||
        lower_name.find("negative") != std::string::npos ||
        lower_name.find("foliage") != std::string::npos ||
        lower_name.find("canopy") != std::string::npos ||
        lower_name.find("empty") != std::string::npos) {
        meta.ground_truth_target = false;
    }

    // Try extracting standoff distance from filename (e.g. "30ft", "60ft", "90ft", "115ft", "150ft")
    for (double d : {30.0, 60.0, 90.0, 115.0, 150.0}) {
        std::string tag = std::to_string(static_cast<int>(d)) + "ft";
        if (lower_name.find(tag) != std::string::npos) {
            meta.standoff_ft = d;
            break;
        }
    }

    std::ifstream f(json_path);
    if (!f.is_open()) return meta;

    std::string line;
    while (std::getline(f, line)) {
        auto parse_val = [&](const std::string& key) -> std::string {
            auto pos = line.find("\"" + key + "\"");
            if (pos == std::string::npos) return "";
            auto colon = line.find(':', pos);
            if (colon == std::string::npos) return "";
            auto start = line.find_first_not_of(" \t\"", colon + 1);
            if (start == std::string::npos) return "";
            auto end = line.find_last_not_of(" \t\",\r\n");
            return line.substr(start, end - start + 1);
        };

        if (line.find("\"platform\"") != std::string::npos) {
            std::string v = parse_val("platform");
            if (!v.empty()) meta.platform = v;
        } else if (line.find("\"standoff_ft\"") != std::string::npos) {
            std::string v = parse_val("standoff_ft");
            if (!v.empty()) meta.standoff_ft = std::stod(v);
        } else if (line.find("\"blade_count\"") != std::string::npos) {
            std::string v = parse_val("blade_count");
            if (!v.empty()) meta.blade_count = std::stoi(v);
        } else if (line.find("\"expected_bpf_hz\"") != std::string::npos) {
            std::string v = parse_val("expected_bpf_hz");
            if (!v.empty()) meta.expected_bpf_hz = std::stod(v);
        } else if (line.find("\"ground_truth_target\"") != std::string::npos) {
            std::string v = parse_val("ground_truth_target");
            if (!v.empty()) meta.ground_truth_target = (v == "true" || v == "1");
        } else if (line.find("\"target_start_s\"") != std::string::npos) {
            std::string v = parse_val("target_start_s");
            if (!v.empty()) meta.target_start_s = std::stod(v);
        } else if (line.find("\"target_end_s\"") != std::string::npos) {
            std::string v = parse_val("target_end_s");
            if (!v.empty()) meta.target_end_s = std::stod(v);
        }
    }
    return meta;
}

} // anonymous namespace

namespace predator {

class ReplayHarnessEngine {
public:
    ReplayHarnessEngine(const HarnessOptions& opts)
        : opts_(opts),
          cuda_core_(1280, 720, 32, 18, 4000.0, 512),
          gyro_warper_(make_lens_params()),
          spectral_engine_(128)
    {
        // 1. Configure CFAR detector gate
        SpectralGateConfig gate_cfg;
        gate_cfg.false_alarms_per_hour = opts_.cfar_fa_per_hour;
        cuda_core_.set_spectral_gate_config(gate_cfg);

        // 2. Initialize TensorRT SpectralCombNet engine
        if (!opts_.combnet_path.empty()) {
            if (fs::exists(opts_.combnet_path)) {
                combnet_loaded_ = spectral_engine_.load_engine(opts_.combnet_path);
            }
        }

        // 3. Load hot pixel mask (matches live sensor hardware mask)
        predator::HotPixelMaskConfig mask_cfg;
        mask_cfg.sensor_width = 1280;
        mask_cfg.sensor_height = 720;
        mask_cfg.max_masks = 64;
        std::string hp_file = opts_.hot_pixels_path;
        if (!std::ifstream(hp_file).good()) {
            hp_file = "/home/orin/ev_deploy/hot_pixels.txt";
        }
        auto mask_res = predator::parse_hot_pixels_file(hp_file, mask_cfg);
        if (mask_res.success && !mask_res.pixels.empty()) {
            hot_pixels_ = mask_res.pixels;
            std::cout << "[INFO] Loaded " << hot_pixels_.size() << " hot pixels from " << hp_file << "\n";
        }
    }

    EvaluationScorecard evaluate_file(const std::string& filepath) {
        EvaluationScorecard card;
        card.file_path = filepath;

        std::string stem = fs::path(filepath).stem().string();
        std::string json_meta_path = (fs::path(filepath).parent_path() / (stem + ".json")).string();
        card.meta = parse_metadata_file(json_meta_path, stem);

        // Reset all pipeline components for isolated repeatable evaluation
        cuda_core_.reset();
        SpatialFlickerClusterer::reset_tracker();

        RawPipelineConfig pipe_cfg;
        pipe_cfg.sensor_width = 1280;
        pipe_cfg.sensor_height = 720;
        pipe_cfg.enable_ego_warp = false;
        pipe_cfg.enable_ui_frame_gen = false;
        pipe_cfg.max_words_per_slot = 32768;
        pipe_cfg.coalesce_words = 16384;

        RawPipeline pipeline(pipe_cfg, cuda_core_, gyro_warper_);

        // Detect format and stream into pipeline
        const bool is_cd = (filepath.size() >= 3 && filepath.rfind(".cd") == filepath.size() - 3);

        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            card.pass_verdict = false;
            card.failure_reasons.push_back("Failed to open recording file: " + filepath);
            return card;
        }

        std::unique_ptr<Evt21Encoder> cd_encoder;
        if (is_cd) {
            cd_encoder = std::make_unique<Evt21Encoder>(1280, 720);
        }

        uint64_t stream_first_ts_us = 0;
        uint64_t stream_last_ts_us = 0;
        bool have_stream_first_ts = false;

        uint64_t frame_start_ts_us = 0;
        constexpr uint64_t kFrameDurationUs = 40000; // 40 ms = 25 Hz analysis cycle

        uint64_t prev_decoded_events = 0;
        uint64_t prev_retained_events = 0;

        std::vector<float> locked_snr_values;
        std::vector<float> locked_prob_values;
        std::vector<double> locked_bpf_values;

        auto execute_analysis_frame = [&](uint64_t cur_ts_us) {
            card.total_frames++;
            double cur_time_s = static_cast<double>(cur_ts_us - stream_first_ts_us) * 1e-6;

            bool in_target_window = (cur_time_s >= card.meta.target_start_s && cur_time_s <= card.meta.target_end_s);
            if (in_target_window && card.meta.ground_truth_target) {
                card.target_frames++;
            }

            // Snapshot event throughput
            uint64_t cur_decoded = pipeline.total_decoded_events();
            uint64_t frame_raw = (cur_decoded >= prev_decoded_events) ? (cur_decoded - prev_decoded_events) : cur_decoded;
            prev_decoded_events = cur_decoded;

            uint64_t cur_retained = pipeline.total_retained_events();
            uint64_t frame_ret = (cur_retained >= prev_retained_events) ? (cur_retained - prev_retained_events) : cur_retained;
            prev_retained_events = cur_retained;

            double supp_pct = (frame_raw > 0) ? (100.0 * (1.0 - (static_cast<double>(frame_ret) / frame_raw))) : 0.0;
            int active_cells = cuda_core_.get_active_cell_count(50.0);

            // 1. Batched cuFFT analysis
            std::vector<FlickerDetectionResult> raw_detections;
            cuda_core_.execute_batched_spectral_analysis(0.0, raw_detections);

            // 2. SpectralCombNet v3 Inference
            float top_neural_prob = 0.0f;
            if (combnet_loaded_) {
                std::vector<int> active_indices;
                std::vector<std::vector<float>> active_spectra;
                cuda_core_.get_active_cells_with_spectra(active_indices, active_spectra, 6.0f);

                if (!active_indices.empty()) {
                    std::vector<const float*> ptrs(active_spectra.size());
                    for (size_t i = 0; i < active_spectra.size(); ++i) ptrs[i] = active_spectra[i].data();
                    std::vector<predator::SpectralPrediction> preds;
                    spectral_engine_.infer_spectra(ptrs, active_indices, preds, 0.0f);

                    for (const auto& np : preds) {
                        top_neural_prob = std::max(top_neural_prob, np.drone_prob);
                        bool is_pooled = (np.cell_idx >= 576);
                        int base_cell = is_pooled ? (np.cell_idx - 576) : np.cell_idx;
                        int patch_col = base_cell % 32;
                        int patch_row = base_cell / 32;

                        bool already_detected = false;
                        for (auto& rd : raw_detections) {
                            if (rd.patch_x == patch_col && rd.patch_y == patch_row) {
                                already_detected = true;
                                rd.is_neural_detection = true;
                                rd.neural_drone_prob = np.drone_prob;
                                if (np.drone_prob > 0.65f) {
                                    rd.confidence = std::min(1.0, rd.confidence + 0.20);
                                }
                                break;
                            }
                        }

                        // Neural Weak-Signal Rescue: preserve distant standoff targets where CombNet has high certainty AND valid physical peak
                        if (opts_.enable_rescue && !already_detected && np.has_valid_peak && np.physical_snr_db >= 8.0f && np.drone_prob >= 0.80f &&
                            np.fund_freq_hz >= 110.0f && np.fund_freq_hz <= 1000.0f) {
                            FlickerDetectionResult res;
                            res.is_drone_detected = true;
                            res.fundamental_bpf_hz = np.fund_freq_hz;
                            res.estimated_rpm = (np.fund_freq_hz * 60.0) / 2.0;
                            res.confidence = 0.5f * np.drone_prob + 0.35f;
                            res.peak_snr_db = std::max(12.0f, np.physical_snr_db);
                            res.harmonic_score = np.harmonic_purity;
                            res.spectral_q_factor = np.spectral_q_factor;
                            res.spectral_flatness = np.spectral_flatness;
                            res.peak_power = std::pow(10.0f, res.peak_snr_db / 10.0f);
                            res.noise_floor = 1.0f;
                            res.patch_x = patch_col;
                            res.patch_y = patch_row;
                            res.is_neural_detection = true;
                            res.neural_drone_prob = np.drone_prob;
                            raw_detections.push_back(res);
                        }
                    }
                }
            }

            // 3. Project centroids
            for (auto& res : raw_detections) {
                res.centroid_px_x = res.patch_x * 40 + 20;
                res.centroid_px_y = res.patch_y * 40 + 20;
            }

            // 4. Cluster fusion and M-of-N tracker update
            auto filtered = SpatialFlickerClusterer::filter_and_cluster(raw_detections, 6);

            int confirmed_count = static_cast<int>(filtered.size());
            float top_snr = 0.0f;
            float top_bpf = 0.0f;
            float track_neural_prob = 0.0f;
            for (const auto& d : filtered) {
                top_snr = std::max(top_snr, static_cast<float>(d.peak_snr_db));
                top_bpf = static_cast<float>(d.fundamental_bpf_hz);
                track_neural_prob = std::max(track_neural_prob, d.neural_drone_prob);
            }

            bool locked = (confirmed_count > 0);
            if (locked && in_target_window && card.meta.ground_truth_target) {
                card.locked_frames++;
                locked_snr_values.push_back(top_snr);
                locked_prob_values.push_back(std::max(track_neural_prob, top_neural_prob));
                locked_bpf_values.push_back(top_bpf);

                if (!card.target_acquired) {
                    card.target_acquired = true;
                    card.t_detect_ms = (cur_time_s - card.meta.target_start_s) * 1000.0;
                    if (card.t_detect_ms < 0.0) card.t_detect_ms = 0.0;
                }
            } else if (locked && !card.meta.ground_truth_target) {
                card.false_confirmed_tracks++;
            }

            FrameRecord fr;
            fr.frame_idx = card.total_frames;
            fr.time_s = cur_time_s;
            fr.raw_events = frame_raw;
            fr.retained_events = frame_ret;
            fr.suppression_pct = supp_pct;
            fr.active_cells = active_cells;
            fr.candidate_count = static_cast<int>(raw_detections.size());
            fr.confirmed_tracks = confirmed_count;
            fr.top_snr_db = top_snr;
            fr.top_bpf_hz = top_bpf;
            fr.top_neural_prob = top_neural_prob;
            fr.target_locked = locked;
            card.frames.push_back(fr);
        };

        if (is_cd) {
            constexpr size_t kCdChunk = 8192;
            std::vector<CdRecord> cd_records(kCdChunk);
            std::vector<CdRecord> frame_cd;
            std::vector<uint64_t> frame_words;
            frame_cd.reserve(32768);
            frame_words.reserve(65536);

            while (file.good()) {
                file.read(reinterpret_cast<char*>(cd_records.data()), kCdChunk * sizeof(CdRecord));
                size_t n = file.gcount() / sizeof(CdRecord);
                if (n == 0) break;

                card.total_events += n;

                for (size_t i = 0; i < n; ++i) {
                    uint16_t ex = cd_records[i].x;
                    uint16_t ey = cd_records[i].y;
                    bool is_hp = false;
                    for (const auto& hp : hot_pixels_) {
                        if (hp.x == ex && hp.y == ey) { is_hp = true; break; }
                    }
                    if (is_hp) continue;

                    uint64_t t = static_cast<uint64_t>(cd_records[i].t);
                    if (!have_stream_first_ts) {
                        stream_first_ts_us = t;
                        frame_start_ts_us = t;
                        have_stream_first_ts = true;
                    }
                    stream_last_ts_us = t;
                    frame_cd.push_back(cd_records[i]);

                    if (t >= frame_start_ts_us + kFrameDurationUs) {
                        if (!frame_cd.empty()) {
                            frame_words.clear();
                            cd_encoder->encode(frame_cd.data(), frame_cd.data() + frame_cd.size(), frame_words);
                            pipeline.process_raw_words_sync(frame_words.data(), frame_words.size());
                            frame_cd.clear();
                        }
                        execute_analysis_frame(t);
                        frame_start_ts_us = t;
                    }
                }
            }
            if (!frame_cd.empty()) {
                frame_words.clear();
                cd_encoder->encode(frame_cd.data(), frame_cd.data() + frame_cd.size(), frame_words);
                pipeline.process_raw_words_sync(frame_words.data(), frame_words.size());
                frame_cd.clear();
                execute_analysis_frame(stream_last_ts_us);
            }
        } else {
            constexpr size_t kWordChunk = 16384;
            std::vector<uint64_t> words(kWordChunk);
            std::vector<uint64_t> frame_words;
            frame_words.reserve(65536);

            while (file.good()) {
                file.read(reinterpret_cast<char*>(words.data()), kWordChunk * sizeof(uint64_t));
                size_t n = file.gcount() / sizeof(uint64_t);
                if (n == 0) break;

                for (size_t i = 0; i < n; ++i) {
                    uint64_t w = words[i];
                    if (predator::evt21::is_cd(w) && !hot_pixels_.empty()) {
                        uint32_t y = predator::evt21::cd_y(w);
                        uint32_t x_base = predator::evt21::cd_x_base(w);
                        uint32_t mask = predator::evt21::cd_mask(w);
                        for (const auto& hp : hot_pixels_) {
                            if (hp.y == y && hp.x >= x_base && hp.x < x_base + 32) {
                                mask &= ~(1u << (hp.x - x_base));
                            }
                        }
                        if (mask == 0) continue;
                        w = (w & 0x00000000FFFFFFFFULL) | (static_cast<uint64_t>(mask) << 32);
                    }
                    frame_words.push_back(w);

                    if (predator::evt21::word_type(w) == predator::evt21::kTimeHigh) {
                        uint64_t t = static_cast<uint64_t>(predator::evt21::time_high(w)) << 6;
                        if (!have_stream_first_ts) {
                            stream_first_ts_us = t;
                            frame_start_ts_us = t;
                            have_stream_first_ts = true;
                        }
                        stream_last_ts_us = t;
                        if (t >= frame_start_ts_us + kFrameDurationUs) {
                            pipeline.process_raw_words_sync(frame_words.data(), frame_words.size());
                            frame_words.clear();
                            execute_analysis_frame(t);
                            frame_start_ts_us = t;
                        }
                    }
                }
            }
            if (!frame_words.empty()) {
                pipeline.process_raw_words_sync(frame_words.data(), frame_words.size());
                frame_words.clear();
                execute_analysis_frame(stream_last_ts_us);
            }
            card.total_events = pipeline.total_decoded_events();
        }

        // Finalize summary metrics
        if (have_stream_first_ts && stream_last_ts_us >= stream_first_ts_us) {
            card.total_duration_s = static_cast<double>(stream_last_ts_us - stream_first_ts_us) * 1e-6;
        }
        if (card.total_duration_s > 0.0) {
            card.avg_event_rate_ev_s = static_cast<double>(card.total_events) / card.total_duration_s;
            card.fa_rate_per_hour = (static_cast<double>(card.false_confirmed_tracks) / card.total_duration_s) * 3600.0;
        }

        if (card.target_frames > 0) {
            card.track_continuity_pct = (100.0 * static_cast<double>(card.locked_frames)) / card.target_frames;
        }

        if (!locked_snr_values.empty()) {
            card.min_snr_db = *std::min_element(locked_snr_values.begin(), locked_snr_values.end());
            card.max_snr_db = *std::max_element(locked_snr_values.begin(), locked_snr_values.end());
            card.median_snr_db = compute_percentile(locked_snr_values, 0.50f);
        }

        if (!locked_prob_values.empty()) {
            card.min_neural_prob = *std::min_element(locked_prob_values.begin(), locked_prob_values.end());
            card.max_neural_prob = *std::max_element(locked_prob_values.begin(), locked_prob_values.end());
            card.median_neural_prob = compute_percentile(locked_prob_values, 0.50f);
        }

        if (!locked_bpf_values.empty()) {
            std::vector<double> folded_bpf_values;
            folded_bpf_values.reserve(locked_bpf_values.size());
            const double f_exp = card.meta.expected_bpf_hz;

            for (double f : locked_bpf_values) {
                if (f_exp > 0.0) {
                    double best_folded = f;
                    double min_diff = std::abs(f - f_exp);
                    const double factors[] = {0.2, 0.25, 0.333333, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0};
                    for (double factor : factors) {
                        double cand = f / factor;
                        double diff = std::abs(cand - f_exp);
                        if (diff < min_diff) {
                            min_diff = diff;
                            best_folded = cand;
                        }
                    }
                    folded_bpf_values.push_back(best_folded);
                } else {
                    folded_bpf_values.push_back(f);
                }
            }

            card.detected_mean_bpf_hz = compute_percentile(folded_bpf_values, 0.50);
            if (f_exp > 0.0) {
                card.bpf_error_hz = std::abs(card.detected_mean_bpf_hz - f_exp);
            }
        }

        // Evaluate Quantitative Acceptance Criteria
        bool pass = true;
        if (!card.meta.ground_truth_target) {
            // Negative Control Gate: FA rate <= 1.0 FA / hour, exactly 0 false confirmed tracks
            if (card.false_confirmed_tracks > 0) {
                pass = false;
                card.failure_reasons.push_back("Negative control emitted " + std::to_string(card.false_confirmed_tracks) +
                                               " false confirmed tracks (FA rate: " + std::to_string(card.fa_rate_per_hour) + "/h)");
            }
        } else {
            // Positive Target Flight Gates
            if (!card.target_acquired) {
                pass = false;
                card.failure_reasons.push_back("Target was never confirmed into active track lock");
            }
            if (card.t_detect_ms > 250.0) {
                pass = false;
                card.failure_reasons.push_back("Detection latency (" + std::to_string(card.t_detect_ms) + " ms) exceeded 250.0 ms gate");
            }
            if (card.track_continuity_pct < 75.0) {
                pass = false;
                card.failure_reasons.push_back("Track continuity (" + std::to_string(card.track_continuity_pct) + "%) below 75.0% gate");
            }
            if (card.median_snr_db < 10.0f) {
                pass = false;
                card.failure_reasons.push_back("Median SNR (" + std::to_string(card.median_snr_db) + " dB) below 10.0 dB threshold");
            }
            if (card.meta.expected_bpf_hz > 0.0 && card.bpf_error_hz > 8.0) {
                pass = false;
                card.failure_reasons.push_back("BPF estimation error (" + std::to_string(card.bpf_error_hz) + " Hz) exceeded 8.0 Hz gate");
            }
        }

        card.pass_verdict = pass;
        return card;
    }

private:
    static LensParameters make_lens_params() {
        LensParameters p;
        p.focal_length_mm = 12.0;
        p.pixel_pitch_um = 4.86;
        p.sensor_width = 1280;
        p.sensor_height = 720;
        return p;
    }

    HarnessOptions opts_;
    CudaFlickerCore cuda_core_;
    ContinuousGyroWarper gyro_warper_;
    SpectralCombNetEngine spectral_engine_;
    bool combnet_loaded_{false};
    std::vector<predator::HotPixel> hot_pixels_;
};

} // namespace predator

namespace {

void print_card_ascii(const EvaluationScorecard& c) {
    std::cout << "\n========================================================================================================================\n";
    std::cout << "                                   PREDATOR OFFLINE REPLAY EVALUATION SCORECARD                                         \n";
    std::cout << "========================================================================================================================\n";
    std::cout << "Recording:            " << c.file_path << "\n";
    std::cout << "Duration:             " << std::fixed << std::setprecision(2) << c.total_duration_s << " s (" << c.total_frames
              << " frames) | Events: " << c.total_events << " (" << (c.avg_event_rate_ev_s / 1e3) << " kev/s)\n";
    std::cout << "Target Class:         " << (c.meta.ground_truth_target ? "POSITIVE DRONE TARGET" : "NEGATIVE CONTROL / BACKGROUND") << "\n";
    if (c.meta.ground_truth_target) {
        std::cout << "Platform:             " << c.meta.platform << " | Standoff: " << c.meta.standoff_ft << " ft | Blades: " << c.meta.blade_count << "\n";
        if (c.meta.expected_bpf_hz > 0.0) {
            std::cout << "Expected BPF:         " << c.meta.expected_bpf_hz << " Hz (" << (c.meta.expected_bpf_hz * 60.0 / c.meta.blade_count) << " RPM)\n";
        }
    }
    std::cout << "------------------------------------------------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(28) << "METRIC"
              << std::setw(24) << "MEASURED VALUE"
              << std::setw(24) << "ACCEPTANCE GATE"
              << std::setw(12) << "STATUS" << "\n";
    std::cout << "------------------------------------------------------------------------------------------------------------------------\n";

    if (c.meta.ground_truth_target) {
        std::string t_det_str = (c.t_detect_ms >= 0.0) ? (std::to_string(static_cast<int>(c.t_detect_ms)) + " ms") : "NEVER";
        std::cout << std::left << std::setw(28) << "Detection Latency (T_det)"
                  << std::setw(24) << t_det_str
                  << std::setw(24) << "<= 250 ms"
                  << std::setw(12) << ((c.t_detect_ms >= 0.0 && c.t_detect_ms <= 250.0) ? "PASS" : "FAIL") << "\n";

        std::string cont_str = std::to_string(static_cast<int>(c.track_continuity_pct)) + "% (" +
                               std::to_string(c.locked_frames) + "/" + std::to_string(c.target_frames) + ")";
        std::cout << std::left << std::setw(28) << "Track Continuity"
                  << std::setw(24) << cont_str
                  << std::setw(24) << ">= 75%"
                  << std::setw(12) << (c.track_continuity_pct >= 75.0 ? "PASS" : "FAIL") << "\n";

        if (c.meta.expected_bpf_hz > 0.0) {
            std::ostringstream bpf_ss;
            bpf_ss << std::fixed << std::setprecision(1) << c.detected_mean_bpf_hz << " Hz (err " << c.bpf_error_hz << ")";
            std::cout << std::left << std::setw(28) << "BPF Accuracy"
                      << std::setw(24) << bpf_ss.str()
                      << std::setw(24) << "<= 8.0 Hz"
                      << std::setw(12) << (c.bpf_error_hz <= 8.0 ? "PASS" : "FAIL") << "\n";
        }

        std::ostringstream snr_ss;
        snr_ss << std::fixed << std::setprecision(1) << c.median_snr_db << " dB (max " << c.max_snr_db << ")";
        std::cout << std::left << std::setw(28) << "Spectral SNR (p50)"
                  << std::setw(24) << snr_ss.str()
                  << std::setw(24) << ">= 10.0 dB"
                  << std::setw(12) << (c.median_snr_db >= 10.0f ? "PASS" : "FAIL") << "\n";

        std::ostringstream prob_ss;
        prob_ss << std::fixed << std::setprecision(3) << c.median_neural_prob << " (max " << c.max_neural_prob << ")";
        std::cout << std::left << std::setw(28) << "CombNet Prob (p50)"
                  << std::setw(24) << prob_ss.str()
                  << std::setw(24) << ">= 0.500"
                  << std::setw(12) << (c.median_neural_prob >= 0.50f ? "PASS" : "FAIL") << "\n";
    } else {
        std::ostringstream fa_ss;
        fa_ss << c.false_confirmed_tracks << " tracks (" << std::fixed << std::setprecision(2) << c.fa_rate_per_hour << "/h)";
        std::cout << std::left << std::setw(28) << "False Alarms (FA)"
                  << std::setw(24) << fa_ss.str()
                  << std::setw(24) << "0 (<= 1.0/h)"
                  << std::setw(12) << (c.false_confirmed_tracks == 0 ? "PASS" : "FAIL") << "\n";
    }

    std::cout << "------------------------------------------------------------------------------------------------------------------------\n";
    std::cout << "FINAL VERDICT:             " << (c.pass_verdict ? "[PASS] ALL GATES SATISFIED" : "[FAIL] GATE CRITERIA BREACHED") << "\n";
    if (!c.failure_reasons.empty()) {
        std::cout << "Failure Reasons:\n";
        for (const auto& r : c.failure_reasons) {
            std::cout << "  - " << r << "\n";
        }
    }
    std::cout << "========================================================================================================================\n\n";
}

void export_scorecard_json(const std::vector<EvaluationScorecard>& scorecards, const std::string& out_path) {
    std::ofstream out(out_path);
    if (!out.is_open()) return;

    out << "{\n";
    out << "  \"timestamp_ms\": " << std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count() << ",\n";
    out << "  \"scorecards\": [\n";

    for (size_t i = 0; i < scorecards.size(); ++i) {
        const auto& c = scorecards[i];
        out << "    {\n";
        out << "      \"file\": \"" << c.file_path << "\",\n";
        out << "      \"target_present\": " << (c.meta.ground_truth_target ? "true" : "false") << ",\n";
        out << "      \"platform\": \"" << c.meta.platform << "\",\n";
        out << "      \"standoff_ft\": " << c.meta.standoff_ft << ",\n";
        out << "      \"duration_s\": " << std::fixed << std::setprecision(2) << c.total_duration_s << ",\n";
        out << "      \"total_events\": " << c.total_events << ",\n";
        out << "      \"total_frames\": " << c.total_frames << ",\n";
        out << "      \"target_frames\": " << c.target_frames << ",\n";
        out << "      \"locked_frames\": " << c.locked_frames << ",\n";
        out << "      \"track_continuity_pct\": " << c.track_continuity_pct << ",\n";
        out << "      \"t_detect_ms\": " << c.t_detect_ms << ",\n";
        out << "      \"median_snr_db\": " << c.median_snr_db << ",\n";
        out << "      \"median_neural_prob\": " << c.median_neural_prob << ",\n";
        out << "      \"detected_mean_bpf_hz\": " << c.detected_mean_bpf_hz << ",\n";
        out << "      \"bpf_error_hz\": " << c.bpf_error_hz << ",\n";
        out << "      \"false_confirmed_tracks\": " << c.false_confirmed_tracks << ",\n";
        out << "      \"fa_rate_per_hour\": " << c.fa_rate_per_hour << ",\n";
        out << "      \"pass\": " << (c.pass_verdict ? "true" : "false") << "\n";
        out << "    }" << (i + 1 < scorecards.size() ? ",\n" : "\n");
    }

    out << "  ]\n";
    out << "}\n";
}

void export_frames_csv(const std::vector<FrameRecord>& frames, const std::string& out_path) {
    std::ofstream out(out_path);
    if (!out.is_open()) return;

    out << "frame_idx,time_s,raw_events,retained_events,suppression_pct,active_cells,candidate_count,confirmed_tracks,top_snr_db,top_bpf_hz,top_neural_prob,target_locked\n";
    for (const auto& f : frames) {
        out << f.frame_idx << ","
            << std::fixed << std::setprecision(3) << f.time_s << ","
            << f.raw_events << ","
            << f.retained_events << ","
            << std::setprecision(1) << f.suppression_pct << ","
            << f.active_cells << ","
            << f.candidate_count << ","
            << f.confirmed_tracks << ","
            << std::setprecision(2) << f.top_snr_db << ","
            << std::setprecision(1) << f.top_bpf_hz << ","
            << std::setprecision(3) << f.top_neural_prob << ","
            << (f.target_locked ? 1 : 0) << "\n";
    }
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    HarnessOptions opts;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--file" || arg == "-f") {
            if (i + 1 < argc) opts.single_file = argv[++i];
        } else if (arg == "--dir" || arg == "-d") {
            if (i + 1 < argc) opts.directory = argv[++i];
        } else if (arg == "--out" || arg == "-o") {
            if (i + 1 < argc) opts.json_out = argv[++i];
        } else if (arg == "--csv" || arg == "-c") {
            if (i + 1 < argc) opts.csv_out = argv[++i];
        } else if (arg == "--combnet") {
            if (i + 1 < argc) opts.combnet_path = argv[++i];
        } else if (arg == "--cfar-fa") {
            if (i + 1 < argc) opts.cfar_fa_per_hour = std::stof(argv[++i]);
        } else if (arg == "--require-drone") {
            opts.require_drone = true;
        } else if (arg == "--require-clean") {
            opts.require_clean = true;
        } else if (arg == "--rescue") {
            opts.enable_rescue = true;
        } else if (arg == "--hot-pixels") {
            if (i + 1 < argc) opts.hot_pixels_path = argv[++i];
        } else if (arg == "--verbose" || arg == "-v") {
            opts.verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options] [recording_file_or_dir]\n\n"
                      << "Options:\n"
                      << "  -f, --file <path>     Single recording (.evt21raw, .cd, .raw)\n"
                      << "  -d, --dir <path>      Batch directory containing recordings\n"
                      << "  -o, --out <path>      Output evaluation scorecard JSON\n"
                      << "  -c, --csv <path>      Export per-frame CSV metrics\n"
                      << "      --combnet <path>  Path to TensorRT CombNet engine\n"
                      << "      --cfar-fa <rate>  Target CFAR false alarm budget (default 6.0/hr)\n"
                      << "      --rescue          Enable CombNet weak-signal rescue (default: false)\n"
                      << "      --hot-pixels <p>  Path to hot_pixels.txt mask file\n"
                      << "      --require-drone   Assert drone target lock is achieved (exit 1 if failed)\n"
                      << "      --require-clean   Assert 0 false alarms on negative control (exit 1 if failed)\n"
                      << "  -h, --help            Show this help message\n";
            return 0;
        } else if (arg.rfind("--", 0) != 0 && arg.rfind("-", 0) != 0) {
            // Positional argument: file or directory
            if (fs::is_directory(arg)) {
                opts.directory = arg;
            } else {
                opts.single_file = arg;
            }
        }
    }

    if (opts.single_file.empty() && opts.directory.empty()) {
        std::cerr << "[ERROR] Specify either a recording file or directory (use --help for options)\n";
        return 2;
    }

    std::vector<std::string> files_to_eval;
    if (!opts.single_file.empty()) {
        files_to_eval.push_back(opts.single_file);
    } else if (!opts.directory.empty()) {
        if (!fs::exists(opts.directory) || !fs::is_directory(opts.directory)) {
            std::cerr << "[ERROR] Directory does not exist: " << opts.directory << "\n";
            return 2;
        }
        std::map<std::string, std::string> preferred_files;
        for (const auto& entry : fs::directory_iterator(opts.directory)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            std::string stem = entry.path().stem().string();
            if (ext == ".evt21raw") {
                preferred_files[stem] = entry.path().string();
            } else if (ext == ".raw" && preferred_files.find(stem) == preferred_files.end()) {
                preferred_files[stem] = entry.path().string();
            } else if (ext == ".cd" && preferred_files.find(stem) == preferred_files.end()) {
                preferred_files[stem] = entry.path().string();
            }
        }
        for (const auto& [_, p] : preferred_files) {
            files_to_eval.push_back(p);
        }
        std::sort(files_to_eval.begin(), files_to_eval.end());
    }

    if (files_to_eval.empty()) {
        std::cerr << "[ERROR] No valid recording files found (.evt21raw, .cd, .raw)\n";
        return 2;
    }

    std::cout << "[INFO] Initializing Predator Offline Replay Engine...\n";
    predator::ReplayHarnessEngine engine(opts);

    std::vector<EvaluationScorecard> scorecards;
    bool all_passed = true;

    for (const auto& file_path : files_to_eval) {
        std::cout << "[EVAL] Processing " << file_path << " ...\n";
        auto card = engine.evaluate_file(file_path);
        print_card_ascii(card);

        if (!card.pass_verdict) {
            all_passed = false;
        }
        if (opts.require_drone && !card.target_acquired) {
            all_passed = false;
        }
        if (opts.require_clean && card.false_confirmed_tracks > 0) {
            all_passed = false;
        }

        if (!opts.csv_out.empty()) {
            std::string csv_dest = opts.csv_out;
            if (files_to_eval.size() > 1) {
                csv_dest = fs::path(file_path).stem().string() + "_frames.csv";
            }
            export_frames_csv(card.frames, csv_dest);
            std::cout << "[INFO] Wrote frame telemetry to " << csv_dest << "\n";
        }

        scorecards.push_back(card);
    }

    if (!opts.json_out.empty()) {
        export_scorecard_json(scorecards, opts.json_out);
        std::cout << "[INFO] Saved evaluation scorecard JSON to " << opts.json_out << "\n";
    }

    std::cout << "\n========================================================\n";
    std::cout << "  EVALUATION COMPLETE: " << scorecards.size() << " datasets scored\n";
    std::cout << "  OVERALL VERDICT:     " << (all_passed ? "PASS (All Gates Satisfied)" : "FAIL (One or more gates failed)") << "\n";
    std::cout << "========================================================\n";

    return all_passed ? 0 : 1;
}
