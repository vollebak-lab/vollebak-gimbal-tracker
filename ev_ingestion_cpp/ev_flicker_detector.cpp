#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <string>
#include <sstream>
#include <cstring>
#include <csignal>
#include <condition_variable>
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <cmath>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>

#include <metavision/sdk/stream/camera.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/hal/device/device.h>
#include <metavision/hal/device/device_discovery.h>
#include <metavision/hal/utils/device_config.h>
#include <metavision/hal/facilities/i_hw_identification.h>
#include <metavision/hal/facilities/i_events_stream.h>
#include <metavision/hal/facilities/i_ll_biases.h>
#include <opencv2/opencv.hpp>

#include "flicker_dsp.hpp"
#include "ego_motion.hpp"
#include "spectral_combnet_trt.hpp"
#include "cuda_flicker_core.cuh"
#include "hot_pixel_mask.hpp"
#include "raw_pipeline.cuh"
#include <omp.h>

// Global shutdown flag
static std::atomic<bool> g_running{true};

// Demand-gated UI JPEG compression activity timestamp (Phase 33.4b.e)
static std::atomic<uint64_t> g_last_client_request_ms{0};

// Hardware pixel masking telemetry state (Phase 33.7a)
static bool g_hardware_mask_applied = false;
static std::string g_hardware_mask_facility = "none";
static std::vector<predator::HotPixel> g_masked_hot_pixels;

// Build provenance stamp injected by CMake (-DPREDATOR_BUILD_ID=...). Lets the
// running binary be matched to the exact source tree (prevents deploy drift).
#ifndef PREDATOR_BUILD_ID
#define PREDATOR_BUILD_ID "unstamped"
#endif

/// Reads a boolean feature flag from the environment.
/// Accepts "1"/"0"; any other value or an unset variable yields `default_value`.
static bool env_flag(const char* name, bool default_value) {
    const char* v = std::getenv(name);
    if (!v) return default_value;
    const std::string s(v);
    if (s == "1") return true;
    if (s == "0") return false;
    std::cerr << "[WARN] Ignoring invalid value for " << name << "='" << s << "' (expected 0 or 1)\n";
    return default_value;
}

/// Reads a strictly positive finite float from the environment; invalid values are rejected
/// with a warning and `default_value` is used.
static float env_positive_float(const char* name, float default_value) {
    const char* v = std::getenv(name);
    if (!v) return default_value;
    char* end = nullptr;
    const float f = std::strtof(v, &end);
    if (end == v || *end != '\0' || !std::isfinite(f) || f <= 0.0f) {
        std::cerr << "[WARN] Ignoring invalid value for " << name << "='" << v << "' (expected a positive number)\n";
        return default_value;
    }
    return f;
}

// Runtime feature flags (resolved once in main(), read by telemetry).
static bool g_ego_warp_enabled = false;     // PREDATOR_ENABLE_EGO_WARP (default OFF, Phase 33)
static bool g_combnet_prune_enabled = false; // PREDATOR_COMBNET_PRUNE (default OFF until CombNet v3 gate)
static bool g_combnet_rescue_enabled = false; // PREDATOR_COMBNET_RESCUE (default OFF, Phase 33.7)
static bool g_focus_mode_enabled = false;    // PREDATOR_FOCUS_MODE (default OFF, toggle via /toggle_focus)
static std::atomic<double> g_live_focus_score{0.0};
static std::atomic<double> g_peak_focus_score{0.0};
static float g_cfar_fa_per_hour = 0.0f;      // PREDATOR_CFAR_FA_PER_HOUR (Phase 33.5 false-alarm budget)
static float g_cfar_threshold_db = 0.0f;     // Derived CFAR threshold in use

struct DiagnosticsLogRecord {
    uint64_t timestamp_us{0};
    uint64_t host_ms{0};
    int num_targets{0};
    double target_bpf{0.0};
    double target_rpm{0.0};
    double target_snr{0.0};
    double target_q{0.0};
    double target_peak_power{0.0};
    double target_noise_floor{0.0};
    int cam_x{0};
    int cam_y{0};
    int world_x{0};
    int world_y{0};
    double azimuth_deg{0.0};
    double elevation_deg{0.0};
    int track_id{0};
    int hit_count{0};
    int miss_count{0};
    double gyro_wx{0.0};
    double gyro_wy{0.0};
    double gyro_wz{0.0};
    double gyro_speed_deg_s{0.0};
    int active_cells{0};
    double foliage_dispersion_pct{0.0};
    uint64_t raw_events{0};
    uint64_t retained_events{0};
    double suppressed_pct{0.0};
    int tentative_count{0};
    float roi_events{0.0f};
    float roi_max_cell{0.0f};
    uint32_t roi_max_sieve{0};
    int roi_active_cells{0};
    int neural_eval_cells{0};
    int neural_detections{0};
    float top_neural_prob{0.0f};
};

class DiagnosticsLogger {
public:
    DiagnosticsLogger(const std::string& log_path = "/home/orin/ev_deploy/logs/flicker_diagnostics.csv",
                      const std::string& debug_path = "/home/orin/ev_deploy/logs/pipeline_debug.log")
        : log_path_(log_path), debug_path_(debug_path) {}

    ~DiagnosticsLogger() {
        stop();
    }

    void start() {
        mkdir("/home/orin/ev_deploy/logs", 0777);
        log_file_.open(log_path_, std::ios::out | std::ios::trunc);
        if (log_file_.is_open()) {
            log_file_ << "timestamp_us,host_ms,num_targets,target_bpf_hz,target_rpm,target_snr_db,"
                      << "target_q_factor,target_peak_power,target_noise_floor,target_cam_x,target_cam_y,"
                      << "target_world_x,target_world_y,azimuth_deg,elevation_deg,track_id,hit_count,miss_count,"
                      << "gyro_wx_rad_s,gyro_wy_rad_s,gyro_wz_rad_s,gyro_speed_deg_s,active_cells,"
                      << "foliage_dispersion_pct,raw_events_40ms,retained_events_40ms,suppressed_pct,"
                      << "tentative_count,roi_events,roi_max_cell,roi_max_sieve,roi_active_cells,"
                      << "neural_eval_cells,neural_detections,top_neural_prob\n";
            log_file_.flush();
        }
        debug_file_.open(debug_path_, std::ios::out | std::ios::trunc);
        if (debug_file_.is_open()) {
            debug_file_ << "=== Predator Pipeline Diagnostics Log Started ===\n";
            debug_file_.flush();
        }
        running_ = true;
        worker_ = std::thread(&DiagnosticsLogger::worker_thread, this);
    }

    void stop() {
        if (running_) {
            running_ = false;
            cv_.notify_all();
            if (worker_.joinable()) {
                worker_.join();
            }
            if (log_file_.is_open()) {
                log_file_.flush();
                log_file_.close();
            }
            if (debug_file_.is_open()) {
                debug_file_.flush();
                debug_file_.close();
            }
        }
    }

    void log_frame(const DiagnosticsLogRecord& rec) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (queue_.size() < 10000) {
            queue_.push_back(rec);
            cv_.notify_one();
        }
    }

    void log_debug_line(const std::string& line) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (debug_queue_.size() < 10000) {
            debug_queue_.push_back(line);
            cv_.notify_one();
        }
    }

    const std::string& log_path() const { return log_path_; }
    const std::string& debug_path() const { return debug_path_; }

private:
    void worker_thread() {
#if defined(__linux__) && !defined(__ANDROID__)
        pthread_setname_np(pthread_self(), "diag_logger");
#endif
        std::vector<DiagnosticsLogRecord> local_batch;
        std::vector<std::string> local_debug;
        local_batch.reserve(200);
        local_debug.reserve(200);

        while (running_ || !queue_.empty() || !debug_queue_.empty()) {
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                cv_.wait_for(lock, std::chrono::milliseconds(100), [&] {
                    return !queue_.empty() || !debug_queue_.empty() || !running_;
                });
                while (!queue_.empty()) {
                    local_batch.push_back(queue_.front());
                    queue_.pop_front();
                }
                while (!debug_queue_.empty()) {
                    local_debug.push_back(debug_queue_.front());
                    debug_queue_.pop_front();
                }
            }

            if (!local_batch.empty() && log_file_.is_open()) {
                for (const auto& r : local_batch) {
                    log_file_ << r.timestamp_us << ","
                              << r.host_ms << ","
                              << r.num_targets << ","
                              << std::fixed << std::setprecision(2)
                              << r.target_bpf << ","
                              << r.target_rpm << ","
                              << std::setprecision(2) << r.target_snr << ","
                              << r.target_q << ","
                              << r.target_peak_power << ","
                              << r.target_noise_floor << ","
                              << r.cam_x << ","
                              << r.cam_y << ","
                              << r.world_x << ","
                              << r.world_y << ","
                              << std::setprecision(3) << r.azimuth_deg << ","
                              << r.elevation_deg << ","
                              << r.track_id << ","
                              << r.hit_count << ","
                              << r.miss_count << ","
                              << std::setprecision(4) << r.gyro_wx << ","
                              << r.gyro_wy << ","
                              << r.gyro_wz << ","
                              << std::setprecision(2) << r.gyro_speed_deg_s << ","
                              << r.active_cells << ","
                              << r.foliage_dispersion_pct << ","
                              << r.raw_events << ","
                              << r.retained_events << ","
                              << r.suppressed_pct << ","
                              << r.tentative_count << ","
                              << r.roi_events << ","
                              << r.roi_max_cell << ","
                              << r.roi_max_sieve << ","
                              << r.roi_active_cells << ","
                              << r.neural_eval_cells << ","
                              << r.neural_detections << ","
                              << std::setprecision(3) << r.top_neural_prob << "\n";
                }
                log_file_.flush();
                local_batch.clear();
            }

            if (!local_debug.empty() && debug_file_.is_open()) {
                for (const auto& line : local_debug) {
                    debug_file_ << line << "\n";
                }
                debug_file_.flush();
                local_debug.clear();
            }
        }
    }

    std::string log_path_;
    std::string debug_path_;
    std::ofstream log_file_;
    std::ofstream debug_file_;
    std::deque<DiagnosticsLogRecord> queue_;
    std::deque<std::string> debug_queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::atomic<bool> running_{false};
    std::thread worker_;
};

static DiagnosticsLogger g_diag_logger;

class StreamBroadcaster;
class FrameBufferManager;

// Broadcaster for video stream + HUD
class StreamBroadcaster {
public:
    void update_frame(const std::vector<uchar>& jpeg_data) {
        std::lock_guard<std::mutex> lock(mutex_);
        current_jpeg_ = jpeg_data;
        frame_id_++;
        cv_.notify_all();
    }

    void notify_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        cv_.notify_all();
    }

    bool get_latest_frame(std::vector<uchar>& out_jpeg) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (current_jpeg_.empty()) return false;
        out_jpeg = current_jpeg_;
        return true;
    }

    bool get_frame(uint64_t last_frame_id, std::vector<uchar>& out_jpeg, uint64_t& out_frame_id, int timeout_ms = 50) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return frame_id_ > last_frame_id || !g_running; })) {
            if (!g_running || current_jpeg_.empty()) return false;
            out_jpeg = current_jpeg_;
            out_frame_id = frame_id_;
            return true;
        }
        return false;
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<uchar> current_jpeg_;
    uint64_t frame_id_{0};
};

static StreamBroadcaster g_stream_broadcaster;

// Frame buffer manager for decoupled background JPEG encoding
class FrameBufferManager {
public:
    void push_frame(const cv::Mat& frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        frame.copyTo(latest_frame_);
        has_new_frame_ = true;
        cv_.notify_one();
    }

    void notify_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        cv_.notify_all();
    }

    bool wait_for_frame(cv::Mat& out_frame, int timeout_ms = 50) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return has_new_frame_ || !g_running; })) {
            if (!g_running) return false;
            latest_frame_.copyTo(out_frame);
            has_new_frame_ = false;
            return true;
        }
        return false;
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    cv::Mat latest_frame_;
    bool has_new_frame_{false};
};

static FrameBufferManager g_frame_mgr;

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        g_running = false;
        g_frame_mgr.notify_all();
        g_stream_broadcaster.notify_all();
    }
}

// Global detection and ego-motion stats manager
class DetectionManager {
public:
    struct EgoMotionStats {
        bool trt_suppression_active{false};
        bool spectral_combnet_active{false};
        int spectral_eval_cells{0};
        int spectral_detections{0};
        bool imu_connected{false};
        uint64_t imu_packets{0};
        double gyro_wx{0.0};
        double gyro_wy{0.0};
        double gyro_wz{0.0};
        double gyro_speed_deg_s{0.0};
        int active_cells{0};
        double foliage_dispersion_pct{0.0};
        double suppressed_pct{0.0};
        uint64_t total_events{0};
        uint64_t retained_events{0};
    };

    struct UiEncoderStats {
        uint64_t total_frames{0};
        double last_draw_us{0.0};
        double avg_draw_us{0.0};
        double last_encode_us{0.0};
        double avg_encode_us{0.0};
    };

    void update_ui_stats(double draw_us, double encode_us) {
        std::lock_guard<std::mutex> lock(mutex_);
        ui_stats_.total_frames++;
        ui_stats_.last_draw_us = draw_us;
        ui_stats_.last_encode_us = encode_us;
        if (ui_stats_.total_frames == 1) {
            ui_stats_.avg_draw_us = draw_us;
            ui_stats_.avg_encode_us = encode_us;
        } else {
            ui_stats_.avg_draw_us = 0.95 * ui_stats_.avg_draw_us + 0.05 * draw_us;
            ui_stats_.avg_encode_us = 0.95 * ui_stats_.avg_encode_us + 0.05 * encode_us;
        }
    }

    void update_detections(const std::vector<predator::FlickerDetectionResult>& detections,
                           const std::vector<predator::SpatialFlickerClusterer::Track>& all_tracks,
                           const predator::RoiDiagnostics& roi_diag) {
        std::lock_guard<std::mutex> lock(mutex_);
        active_detections_ = detections;
        all_tracks_ = all_tracks;
        roi_diag_ = roi_diag;
    }

    void update_ego_stats(const EgoMotionStats& stats) {
        std::lock_guard<std::mutex> lock(mutex_);
        ego_stats_ = stats;
    }

    void update_raw_stats(const predator::RawPipelineStats& stats) {
        std::lock_guard<std::mutex> lock(mutex_);
        raw_stats_ = stats;
    }

    std::vector<predator::FlickerDetectionResult> get_detections() {
        std::lock_guard<std::mutex> lock(mutex_);
        return active_detections_;
    }

    std::vector<predator::SpatialFlickerClusterer::Track> get_all_tracks() {
        std::lock_guard<std::mutex> lock(mutex_);
        return all_tracks_;
    }

    predator::RoiDiagnostics get_roi_stats() {
        std::lock_guard<std::mutex> lock(mutex_);
        return roi_diag_;
    }

    EgoMotionStats get_ego_stats() {
        std::lock_guard<std::mutex> lock(mutex_);
        return ego_stats_;
    }

    std::string get_telemetry_json() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << "{\n"
           << "  \"timestamp_ms\": " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() << ",\n"
           << "  \"build_id\": \"" << PREDATOR_BUILD_ID << "\",\n"
           << "  \"flags\": {\"ego_warp\": " << (g_ego_warp_enabled ? "true" : "false")
           << ", \"combnet_prune\": " << (g_combnet_prune_enabled ? "true" : "false")
           << ", \"combnet_rescue\": " << (g_combnet_rescue_enabled ? "true" : "false")
           << ", \"focus_mode\": " << (g_focus_mode_enabled ? "true" : "false") << "},\n"
           << "  \"focus\": {\"score\": " << g_live_focus_score.load(std::memory_order_relaxed)
           << ", \"peak\": " << g_peak_focus_score.load(std::memory_order_relaxed)
           << ", \"mode\": " << (g_focus_mode_enabled ? "true" : "false") << "},\n"
           << "  \"cfar\": {\"fa_per_hour\": " << g_cfar_fa_per_hour << ", \"threshold_db\": " << g_cfar_threshold_db << "},\n"
           << "  \"lens\": {\"model\": \"12mm f/2.5 M12 (1/2.5\\\" format)\", \"fl_mm\": 12.0, \"hfov_deg\": 29.1, \"vfov_deg\": 16.6},\n"
           << "  \"hot_pixel_mask\": {\n"
           << "    \"applied\": " << (g_hardware_mask_applied ? "true" : "false") << ",\n"
           << "    \"facility\": \"" << g_hardware_mask_facility << "\",\n"
           << "    \"count\": " << g_masked_hot_pixels.size() << ",\n"
           << "    \"pixels\": [\n";
        for (size_t i = 0; i < g_masked_hot_pixels.size(); ++i) {
            const auto& p = g_masked_hot_pixels[i];
            ss << "      {\"x\": " << p.x << ", \"y\": " << p.y << ", \"rate_ev_s\": " << p.rate << "}"
               << (i + 1 < g_masked_hot_pixels.size() ? ",\n" : "\n");
        }
        ss << "    ]\n"
           << "  },\n"
           << "  \"ego_motion\": {\n"
           << "    \"imu_connected\": " << (ego_stats_.imu_connected ? "true" : "false") << ",\n"
           << "    \"imu_packets\": " << ego_stats_.imu_packets << ",\n"
           << "    \"trt_suppression_active\": " << (ego_stats_.trt_suppression_active ? "true" : "false") << ",\n"
           << "    \"spectral_combnet_active\": " << (ego_stats_.spectral_combnet_active ? "true" : "false") << ",\n"
           << "    \"spectral_eval_cells\": " << ego_stats_.spectral_eval_cells << ",\n"
           << "    \"spectral_detections\": " << ego_stats_.spectral_detections << ",\n"
           << "    \"gyro_rad_s\": [" << ego_stats_.gyro_wx << ", " << ego_stats_.gyro_wy << ", " << ego_stats_.gyro_wz << "],\n"
           << "    \"gyro_speed_deg_s\": " << ego_stats_.gyro_speed_deg_s << ",\n"
           << "    \"active_cells\": " << ego_stats_.active_cells << ",\n"
           << "    \"foliage_dispersion_pct\": " << ego_stats_.foliage_dispersion_pct << ",\n"
           << "    \"suppressed_events_pct\": " << ego_stats_.suppressed_pct << ",\n"
           << "    \"total_raw_events\": " << ego_stats_.total_events << ",\n"
           << "    \"retained_imo_events\": " << ego_stats_.retained_events << "\n"
           << "  },\n"
           << "  \"roi_diagnostics\": {\n"
           << "    \"total_events\": " << roi_diag_.total_events << ",\n"
           << "    \"max_cell_events\": " << roi_diag_.max_cell_events << ",\n"
           << "    \"max_sieve_hits\": " << roi_diag_.max_sieve_hits << ",\n"
           << "    \"active_cells\": " << roi_diag_.active_cells << "\n"
           << "  },\n"
           << "  \"ui_encoder\": {\n"
           << "    \"total_frames\": " << ui_stats_.total_frames << ",\n"
           << "    \"last_draw_us\": " << ui_stats_.last_draw_us << ",\n"
           << "    \"avg_draw_us\": " << ui_stats_.avg_draw_us << ",\n"
           << "    \"last_encode_us\": " << ui_stats_.last_encode_us << ",\n"
           << "    \"avg_encode_us\": " << ui_stats_.avg_encode_us << "\n"
           << "  },\n"
           << "  \"raw_pipeline\": {\n"
           << "    \"total_usb_buffers\": " << raw_stats_.total_usb_buffers << ",\n"
           << "    \"total_raw_words\": " << raw_stats_.total_raw_words << ",\n"
           << "    \"total_decoded_events\": " << raw_stats_.total_decoded_events << ",\n"
           << "    \"total_retained_events\": " << raw_stats_.total_retained_events << ",\n"
           << "    \"dropped_buffers\": " << raw_stats_.dropped_buffers << ",\n"
           << "    \"ring_overruns\": " << raw_stats_.ring_overruns << ",\n"
           << "    \"last_batch_words\": " << raw_stats_.last_batch_words << ",\n"
           << "    \"last_batch_events\": " << raw_stats_.last_batch_events << ",\n"
           << "    \"last_gpu_decode_us\": " << raw_stats_.last_gpu_decode_us << ",\n"
           << "    \"last_gpu_sieve_us\": " << raw_stats_.last_gpu_sieve_us << ",\n"
           << "    \"last_gpu_ingest_us\": " << raw_stats_.last_gpu_ingest_us << "\n"
           << "  },\n"
           << "  \"num_targets\": " << active_detections_.size() << ",\n"
           << "  \"num_tracks\": " << all_tracks_.size() << ",\n"
           << "  \"tracks\": [\n";

        for (size_t i = 0; i < all_tracks_.size(); ++i) {
            const auto& trk = all_tracks_[i];
            const auto& t = trk.last_detection;
            std::string state_str = (trk.state == predator::SpatialFlickerClusterer::TrackState::CONFIRMED) ? (trk.miss_count > 0 ? "COASTING" : "CONFIRMED") : "TENTATIVE";
            ss << "    {\n"
               << "      \"track_id\": " << trk.track_id << ",\n"
               << "      \"state\": \"" << state_str << "\",\n"
               << "      \"hit_count\": " << trk.hit_count << ",\n"
               << "      \"miss_count\": " << trk.miss_count << ",\n"
               << "      \"total_age\": " << trk.total_age << ",\n"
               << "      \"bpf_hz\": " << t.fundamental_bpf_hz << ",\n"
               << "      \"estimated_rpm\": " << t.estimated_rpm << ",\n"
               << "      \"confidence\": " << t.confidence << ",\n"
               << "      \"snr_db\": " << t.peak_snr_db << ",\n"
               << "      \"spectral_flatness\": " << t.spectral_flatness << ",\n"
               << "      \"bearing\": {\"azimuth_deg\": " << t.azimuth_deg << ", \"elevation_deg\": " << t.elevation_deg << "},\n"
               << "      \"centroid_px\": {\"x\": " << t.centroid_px_x << ", \"y\": " << t.centroid_px_y << "}\n"
               << "    }" << (i + 1 < all_tracks_.size() ? "," : "") << "\n";
        }
        ss << "  ],\n"
           << "  \"targets\": [\n";
        
        for (size_t i = 0; i < active_detections_.size(); ++i) {
            const auto& t = active_detections_[i];
            ss << "    {\n"
               << "      \"target_id\": " << t.track_id << ",\n"
               << "      \"bpf_hz\": " << t.fundamental_bpf_hz << ",\n"
               << "      \"estimated_rpm\": " << t.estimated_rpm << ",\n"
               << "      \"confidence\": " << t.confidence << ",\n"
               << "      \"snr_db\": " << t.peak_snr_db << ",\n"
               << "      \"bearing\": {\"azimuth_deg\": " << t.azimuth_deg << ", \"elevation_deg\": " << t.elevation_deg << "},\n"
               << "      \"centroid_px\": {\"x\": " << t.centroid_px_x << ", \"y\": " << t.centroid_px_y << "}\n"
               << "    }" << (i + 1 < active_detections_.size() ? "," : "") << "\n";
        }
        ss << "  ]\n}";
        return ss.str();
    }

private:
    std::mutex mutex_;
    std::vector<predator::FlickerDetectionResult> active_detections_;
    std::vector<predator::SpatialFlickerClusterer::Track> all_tracks_;
    predator::RoiDiagnostics roi_diag_;
    EgoMotionStats ego_stats_;
    UiEncoderStats ui_stats_;
    predator::RawPipelineStats raw_stats_;
};

static DetectionManager g_detection_mgr;

void display_encoder_thread_func(int width, int height) {
#if defined(__linux__) && !defined(__ANDROID__)
    pthread_setname_np(pthread_self(), "disp_encoder");
#endif
    // Fast Turbo JPEG encoding parameters (sub-8ms latency)
    std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, 60, cv::IMWRITE_JPEG_OPTIMIZE, 0};
    cv::Mat frame;

    while (g_running) {
        if (g_frame_mgr.wait_for_frame(frame, 30)) {
            // Demand-gated UI JPEG compression (Phase 33.4b.e): only draw HUD and compress JPEG when active clients are connected
            uint64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            bool client_active = (now_ms - g_last_client_request_ms.load(std::memory_order_relaxed)) < 2500;
            if (!client_active) {
                continue;
            }

            auto t0 = std::chrono::steady_clock::now();
            auto all_tracks = g_detection_mgr.get_all_tracks();
            auto active_dets = g_detection_mgr.get_detections();
            auto ego_stats = g_detection_mgr.get_ego_stats();

            // Compute Live Focus Sharpness (Laplacian variance on central 640x360 ROI)
            int roi_w = std::min(640, width);
            int roi_h = std::min(360, height);
            int roi_x = (width - roi_w) / 2;
            int roi_y = (height - roi_h) / 2;
            cv::Rect center_roi(roi_x, roi_y, roi_w, roi_h);

            cv::Mat roi = frame(center_roi);
            cv::Mat gray_roi;
            cv::cvtColor(roi, gray_roi, cv::COLOR_BGR2GRAY);
            cv::Mat lap;
            cv::Laplacian(gray_roi, lap, CV_16S);
            cv::Scalar mean, stddev;
            cv::meanStdDev(lap, mean, stddev);
            double focus_score = stddev.val[0] * stddev.val[0];
            g_live_focus_score.store(focus_score, std::memory_order_relaxed);
            double cur_peak = g_peak_focus_score.load(std::memory_order_relaxed);
            if (focus_score > cur_peak) {
                cur_peak = focus_score;
                g_peak_focus_score.store(cur_peak, std::memory_order_relaxed);
            }

            if (g_focus_mode_enabled) {
                // Focus Assist Mode: suppress target boxes, highlight central focus region and sharpness bar
                cv::Scalar yellow(0, 255, 255);
                cv::rectangle(frame, center_roi, yellow, 2);
                int cx = width / 2;
                int cy = height / 2;
                cv::line(frame, cv::Point(cx - 30, cy), cv::Point(cx + 30, cy), yellow, 2);
                cv::line(frame, cv::Point(cx, cy - 30), cv::Point(cx, cy + 30), yellow, 2);

                int bar_w = 440;
                int bar_h = 24;
                int bar_x = (width - bar_w) / 2;
                int bar_y = height - 55;
                cv::rectangle(frame, cv::Rect(bar_x - 2, bar_y - 2, bar_w + 4, bar_h + 4), cv::Scalar(30, 30, 30), -1);
                cv::rectangle(frame, cv::Rect(bar_x - 2, bar_y - 2, bar_w + 4, bar_h + 4), cv::Scalar(180, 180, 180), 1);

                float fill_ratio = (cur_peak > 1e-3) ? static_cast<float>(std::min(1.0, focus_score / cur_peak)) : 0.0f;
                int fill_w = static_cast<int>(bar_w * fill_ratio);
                if (fill_w > 0) {
                    cv::Scalar fill_color = (fill_ratio >= 0.95f) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 215, 255);
                    cv::rectangle(frame, cv::Rect(bar_x, bar_y, fill_w, bar_h), fill_color, -1);
                }

                char score_text[128];
                snprintf(score_text, sizeof(score_text), "FOCUS SHARPNESS: %.1f / PEAK: %.1f (%d%%)",
                         focus_score, cur_peak, static_cast<int>(fill_ratio * 100.0f));
                cv::putText(frame, score_text, cv::Point(bar_x, bar_y - 8),
                            cv::FONT_HERSHEY_SIMPLEX, 0.52, yellow, 2, cv::LINE_AA);
            } else {
                // Draw detection bounding boxes and HUD on frame
                // Only render CONFIRMED drone targets (Green for active lock, Amber for coasting)
                for (const auto& trk : all_tracks) {
                    bool is_confirmed = (trk.state == predator::SpatialFlickerClusterer::TrackState::CONFIRMED);
                    if (!is_confirmed) {
                        continue; // Suppress unconfirmed tentative tracks from HUD
                    }
                    const auto& d = trk.last_detection;
                    int bx = d.centroid_px_x - 40;
                    int by = d.centroid_px_y - 40;
                    cv::Rect target_rect(std::max(0, bx), std::max(0, by), 
                                         std::min(80, width - std::max(0, bx)), 
                                         std::min(80, height - std::max(0, by)));
                    
                    bool is_coasting = (trk.miss_count > 0);
                    cv::Scalar box_color = is_coasting ? cv::Scalar(0, 215, 255) : cv::Scalar(0, 255, 128); // Amber coasting vs Green locked
                    cv::rectangle(frame, target_rect, box_color, 2);
                    
                    char label[128];
                    const char* net_tag = d.is_neural_detection ? " [NET]" : "";
                    if (is_coasting) {
                        snprintf(label, sizeof(label), "DRONE #%d [COAST %d] %.0fHz [%.1fdB]%s", 
                                 trk.track_id, trk.miss_count, d.fundamental_bpf_hz, d.peak_snr_db, net_tag);
                    } else {
                        snprintf(label, sizeof(label), "DRONE #%d %.0fHz (%.0f RPM) [%.1fdB]%s", 
                                 trk.track_id, d.fundamental_bpf_hz, d.estimated_rpm, d.peak_snr_db, net_tag);
                    }
                    cv::putText(frame, label, cv::Point(target_rect.x, std::max(16, target_rect.y - 6)),
                                cv::FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv::LINE_AA);
                }
            }

            // Top HUD
            char hud_top[256];
            snprintf(hud_top, sizeof(hud_top), "PREDATOR-01 | 12mm f/2.5 M12 | FOCUS: %.1f [PEAK: %.1f]%s",
                     focus_score, cur_peak, g_focus_mode_enabled ? " [FOCUS MODE]" : "");
            cv::putText(frame, hud_top, cv::Point(16, 28), cv::FONT_HERSHEY_SIMPLEX, 0.60,
                        g_focus_mode_enabled ? cv::Scalar(0, 255, 255) : cv::Scalar(220, 220, 220), 2, cv::LINE_AA);

            // Ego-Motion & Frequency-Domain Core HUD Badge
            char ego_badge[256];
            snprintf(ego_badge, sizeof(ego_badge), "NICLA: %s | GYRO: [%+.2f, %+.2f, %+.2f] (%.1f deg/s) | WARP: %s | COMBNET: %s (%d DET)",
                     ego_stats.imu_connected ? "200Hz" : "OFFLINE",
                     ego_stats.gyro_wx, ego_stats.gyro_wy, ego_stats.gyro_wz, ego_stats.gyro_speed_deg_s,
                     ego_stats.trt_suppression_active ? "BYPASS" : "ACTIVE",
                     ego_stats.spectral_combnet_active ? "FP16" : "OFF",
                     ego_stats.spectral_detections);
            cv::putText(frame, ego_badge, cv::Point(16, 56), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                        ego_stats.imu_connected ? cv::Scalar(0, 255, 200) : cv::Scalar(180, 180, 180), 1, cv::LINE_AA);

            auto t1 = std::chrono::steady_clock::now();

            std::vector<uchar> jpeg_buf;
            cv::imencode(".jpg", frame, jpeg_buf, encode_params);
            auto t2 = std::chrono::steady_clock::now();

            double draw_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            double encode_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
            g_detection_mgr.update_ui_stats(draw_us, encode_us);

            g_stream_broadcaster.update_frame(jpeg_buf);
        }
    }
}

// HTML Dashboard with Live HUD and Target Tracks
static const char* HTML_DASHBOARD = R"html(
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predator — Frequency-Domain Propeller Flicker Detector</title>
    <style>
        :root {
            --bg-primary: #0a0d12;
            --bg-card: #131822;
            --border-color: #232b3b;
            --accent-cyan: #00d2ff;
            --accent-green: #00ff88;
            --accent-red: #ff3366;
            --accent-gold: #ffaa00;
            --text-primary: #e1e7f0;
            --text-muted: #7a889b;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            background-color: var(--bg-card);
            border-bottom: 1px solid var(--border-color);
            padding: 16px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .brand { display: flex; align-items: center; gap: 12px; }
        .logo {
            background: linear-gradient(135deg, #0077ff, #00d2ff);
            color: #000;
            font-weight: 900;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 13px;
            letter-spacing: 1px;
        }
        .title { font-size: 16px; font-weight: 700; letter-spacing: 0.5px; }
        .tag {
            background: rgba(0, 255, 136, 0.15);
            border: 1px solid var(--accent-green);
            color: var(--accent-green);
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }
        main {
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 20px;
            padding: 20px;
            flex: 1;
        }
        .stream-panel {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .panel-header {
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.02);
            border-bottom: 1px solid var(--border-color);
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .video-container {
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            background: #000;
            position: relative;
        }
        .video-container canvas {
            width: 100%;
            height: auto;
            max-height: calc(100vh - 180px);
            object-fit: contain;
            display: block;
        }
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
        }
        .card-title {
            font-size: 12px;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }
        .metric-box {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 6px;
            padding: 10px;
        }
        .metric-label { font-size: 11px; color: var(--text-muted); margin-bottom: 4px; }
        .metric-value { font-size: 18px; font-weight: 700; color: var(--accent-cyan); font-family: monospace; }
        .target-list { display: flex; flex-direction: column; gap: 8px; max-height: 280px; overflow-y: auto; }
        .target-item {
            background: rgba(0, 255, 136, 0.05);
            border-left: 3px solid var(--accent-green);
            padding: 10px;
            border-radius: 0 6px 6px 0;
            font-size: 12px;
        }
        .target-item.no-target {
            background: rgba(255, 255, 255, 0.02);
            border-left: 3px solid var(--text-muted);
            color: var(--text-muted);
            text-align: center;
            padding: 20px;
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <div class="logo">PREDATOR</div>
            <div class="title">Neuromorphic Counter-UAS Engine</div>
        </div>
        <div style="display:flex; gap:10px; align-items:center;">
            <a href="/download_csv" target="_blank" class="tag" style="text-decoration:none; background:rgba(0,210,255,0.15); border-color:var(--accent-cyan); color:var(--accent-cyan); font-weight:700;">⬇ Download CSV Log</a>
            <div class="tag">Active Defense</div>
        </div>
    </header>
    <main>
        <div class="stream-panel">
            <div class="panel-header">Stabilized Neuromorphic Focal Stream (1280x720) — Zero-Buffer Canvas Engine</div>
            <div class="video-container" style="background:#000; display:flex; justify-content:center; align-items:center;">
                <canvas id="stream-canvas" width="1280" height="720" style="width:100%; height:auto; max-height:calc(100vh - 180px); object-fit:contain; display:block;"></canvas>
            </div>
        </div>
        <div class="sidebar">
            <div class="card">
                <div class="card-title">Ego-Motion & Clutter Gating</div>
                <div class="metric-grid">
                    <div class="metric-box">
                        <div class="metric-label">Nicla IMU (BHI260)</div>
                        <div class="metric-value" id="val-imu" style="font-size:13px; color:var(--accent-green);">LOCKED</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Angular Speed</div>
                        <div class="metric-value" id="val-speed" style="font-size:14px; color:var(--accent-cyan);">0.0°/s</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Active GPU Cells</div>
                        <div class="metric-value" id="val-foliage" style="font-size:14px;">0</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Suppressed Events</div>
                        <div class="metric-value" id="val-suppressed">0%</div>
                    </div>
                </div>
            </div>
            <div class="card">
                <div class="card-title">Focus Assist Engine <span id="val-focus-mode" style="color:var(--text-muted); font-size:11px;">STANDBY</span></div>
                <div class="metric-grid">
                    <div class="metric-box">
                        <div class="metric-label">Laplacian Sharpness</div>
                        <div class="metric-value" id="val-focus-score" style="font-size:16px; color:var(--accent-gold);">0.0</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Peak Sharpness</div>
                        <div class="metric-value" id="val-focus-peak" style="font-size:16px; color:var(--accent-green);">0.0</div>
                    </div>
                </div>
                <div style="display:flex; gap:8px; margin-top:10px;">
                    <button onclick="fetch('/toggle_focus')" style="flex:1; background:rgba(255,170,0,0.15); border:1px solid var(--accent-gold); color:var(--accent-gold); padding:6px; border-radius:4px; font-weight:700; cursor:pointer;">Toggle Focus Reticle</button>
                    <button onclick="fetch('/reset_focus')" style="background:rgba(255,255,255,0.05); border:1px solid var(--border-color); color:var(--text-muted); padding:6px 12px; border-radius:4px; font-weight:600; cursor:pointer;">Reset Peak</button>
                </div>
            </div>
            <div class="card">
                <div class="card-title">Optical & CUDA Core</div>
                <div class="metric-grid">
                    <div class="metric-box">
                        <div class="metric-label">Optics / GPU</div>
                        <div class="metric-value" style="font-size:13px;">12mm f/2.5 | cuFFT</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Sample Rate</div>
                        <div class="metric-value" style="font-size:14px;">4000 Hz</div>
                    </div>
                </div>
            </div>
            <div class="card" style="flex: 1;">
                <div class="card-title">Target Tracks <span id="target-count" style="color:var(--accent-green);">0</span></div>
                <div class="target-list" id="targets-container">
                    <div class="target-item no-target">Scanning airspace for propeller harmonics...</div>
                </div>
            </div>
        </div>
    </main>
    <script>
        const canvas = document.getElementById('stream-canvas');
        const ctx = canvas.getContext('2d');
        let isFetchingFrame = false;

        async function updateVideoFrame() {
            if (isFetchingFrame) {
                requestAnimationFrame(updateVideoFrame);
                return;
            }
            isFetchingFrame = true;
            try {
                const response = await fetch('/frame.jpg?t=' + performance.now(), { cache: 'no-store' });
                if (response.ok) {
                    const blob = await response.blob();
                    const bitmap = await createImageBitmap(blob);
                    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
                    bitmap.close();
                }
            } catch (e) {}
            isFetchingFrame = false;
            requestAnimationFrame(updateVideoFrame);
        }
        requestAnimationFrame(updateVideoFrame);

        async function fetchStats() {
            try {
                const res = await fetch('/stats');
                if (res.ok) {
                    const data = await res.json();
                    document.getElementById('val-imu').textContent = data.ego_motion.imu_connected ? ("200Hz (" + (data.ego_motion.imu_packets || 0) + ")") : "OFFLINE";
                    document.getElementById('val-imu').style.color = data.ego_motion.imu_connected ? "var(--accent-green)" : "var(--accent-red)";
                    const speed = data.ego_motion.gyro_speed_deg_s || 0.0;
                    document.getElementById('val-speed').textContent = speed.toFixed(1) + "°/s";
                    document.getElementById('val-foliage').textContent = (data.ego_motion.active_cells || 0);
                    document.getElementById('val-suppressed').textContent = (data.ego_motion.suppressed_events_pct || 0).toFixed(0) + "%";
                    document.getElementById('target-count').textContent = data.num_targets;
                    if (data.focus) {
                        const scoreElem = document.getElementById('val-focus-score');
                        const peakElem = document.getElementById('val-focus-peak');
                        const modeElem = document.getElementById('val-focus-mode');
                        if (scoreElem) scoreElem.textContent = (data.focus.score || 0).toFixed(1);
                        if (peakElem) peakElem.textContent = (data.focus.peak || 0).toFixed(1);
                        if (modeElem) {
                            modeElem.textContent = data.focus.mode ? "ACTIVE" : "STANDBY";
                            modeElem.style.color = data.focus.mode ? "var(--accent-gold)" : "var(--text-muted)";
                        }
                    }
                    
                    const container = document.getElementById('targets-container');
                    if (data.num_targets === 0) {
                        container.innerHTML = '<div class="target-item no-target">Scanning airspace for propeller harmonics...</div>';
                    } else {
                        container.innerHTML = data.targets.map(t => `
                            <div class="target-item">
                                <div style="display:flex; justify-content:space-between; font-weight:700; color:var(--accent-green); margin-bottom:4px;">
                                    <span>TARGET #${t.target_id} (DRONE)</span>
                                    <span>${t.bpf_hz.toFixed(1)} Hz</span>
                                </div>
                                <div style="color:var(--text-muted);">
                                    Est. RPM: <b>${t.estimated_rpm.toFixed(0)}</b> | SNR: <b>${t.snr_db.toFixed(1)} dB</b><br>
                                    Bearing: Az <b>${t.bearing.azimuth_deg.toFixed(1)}°</b>, El <b>${t.bearing.elevation_deg.toFixed(1)}°</b>
                                </div>
                            </div>
                        `).join('');
                    }
                }
            } catch (err) {}
            setTimeout(fetchStats, 150);
        }
        fetchStats();
    </script>
</body>
</html>
)html";

void handle_http_client(int client_fd) {
    int nodelay = 1;
    setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

    char buffer[4096];
    ssize_t bytes_read = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
    if (bytes_read <= 0) {
        close(client_fd);
        return;
    }
    buffer[bytes_read] = '\0';
    std::string request(buffer);

    // Record visual streaming request activity for demand-gated encoding (Phase 33.4b.e)
    if (request.find("/frame.jpg") != std::string::npos ||
        request.find("/stream.mjpg") != std::string::npos ||
        request.find("GET / ") != std::string::npos ||
        request.find("GET /index.html") != std::string::npos) {
        uint64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        g_last_client_request_ms.store(now_ms, std::memory_order_relaxed);
    }

    if (request.find("/frame.jpg") != std::string::npos) {
        std::vector<uchar> jpeg_data;
        if (g_stream_broadcaster.get_latest_frame(jpeg_data)) {
            std::string header = "HTTP/1.1 200 OK\r\n"
                                 "Content-Type: image/jpeg\r\n"
                                 "Cache-Control: no-cache, no-store, must-revalidate\r\n"
                                 "Pragma: no-cache\r\n"
                                 "Expires: 0\r\n"
                                 "Access-Control-Allow-Origin: *\r\n"
                                 "Content-Length: " + std::to_string(jpeg_data.size()) + "\r\n"
                                 "Connection: close\r\n\r\n";
            send(client_fd, header.c_str(), header.length(), MSG_NOSIGNAL);
            send(client_fd, jpeg_data.data(), jpeg_data.size(), MSG_NOSIGNAL);
        } else {
            std::string header = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            send(client_fd, header.c_str(), header.length(), MSG_NOSIGNAL);
        }
    } else if (request.find("/download_csv") != std::string::npos || request.find("/download_logs") != std::string::npos || request.find("/flicker_diagnostics.csv") != std::string::npos) {
        std::ifstream csv_in(g_diag_logger.log_path(), std::ios::in | std::ios::binary);
        if (csv_in.is_open()) {
            std::stringstream ss;
            ss << csv_in.rdbuf();
            std::string content = ss.str();
            std::string header = "HTTP/1.1 200 OK\r\n"
                                 "Content-Type: text/csv\r\n"
                                 "Content-Disposition: attachment; filename=\"flicker_diagnostics.csv\"\r\n"
                                 "Cache-Control: no-cache, no-store\r\n"
                                 "Access-Control-Allow-Origin: *\r\n"
                                 "Content-Length: " + std::to_string(content.length()) + "\r\n"
                                 "Connection: close\r\n\r\n";
            send(client_fd, header.c_str(), header.length(), MSG_NOSIGNAL);
            send(client_fd, content.data(), content.length(), MSG_NOSIGNAL);
        } else {
            std::string header = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            send(client_fd, header.c_str(), header.length(), MSG_NOSIGNAL);
        }
    } else if (request.find("/pipeline_debug.log") != std::string::npos || request.find("/download_debug") != std::string::npos) {
        std::ifstream dbg_in(g_diag_logger.debug_path(), std::ios::in | std::ios::binary);
        if (dbg_in.is_open()) {
            std::stringstream ss;
            ss << dbg_in.rdbuf();
            std::string content = ss.str();
            std::string header = "HTTP/1.1 200 OK\r\n"
                                 "Content-Type: text/plain\r\n"
                                 "Content-Disposition: attachment; filename=\"pipeline_debug.log\"\r\n"
                                 "Cache-Control: no-cache, no-store\r\n"
                                 "Access-Control-Allow-Origin: *\r\n"
                                 "Content-Length: " + std::to_string(content.length()) + "\r\n"
                                 "Connection: close\r\n\r\n";
            send(client_fd, header.c_str(), header.length(), MSG_NOSIGNAL);
            send(client_fd, content.data(), content.length(), MSG_NOSIGNAL);
        } else {
            std::string header = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            send(client_fd, header.c_str(), header.length(), MSG_NOSIGNAL);
        }
    } else if (request.find("GET /stream.mjpg") != std::string::npos) {
        int sndbuf = 32768; // 32KB minimal buffer: never buffer stale frames
        setsockopt(client_fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));

        std::string header = "HTTP/1.1 200 OK\r\n"
                             "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                             "Cache-Control: no-cache, no-store, must-revalidate\r\n"
                             "Pragma: no-cache\r\n"
                             "Expires: 0\r\n"
                             "Access-Control-Allow-Origin: *\r\n"
                             "Connection: close\r\n\r\n";
        send(client_fd, header.c_str(), header.length(), 0);

        uint64_t last_frame_id = 0;
        while (g_running) {
            std::vector<uchar> jpeg_data;
            uint64_t frame_id = 0;
            if (g_stream_broadcaster.get_frame(last_frame_id, jpeg_data, frame_id, 100)) {
                last_frame_id = frame_id;
                std::ostringstream ss;
                ss << "--frame\r\n"
                   << "Content-Type: image/jpeg\r\n"
                   << "Content-Length: " << jpeg_data.size() << "\r\n\r\n";
                std::string frame_header = ss.str();
                if (send(client_fd, frame_header.c_str(), frame_header.length(), MSG_NOSIGNAL) <= 0) break;
                if (send(client_fd, jpeg_data.data(), jpeg_data.size(), MSG_NOSIGNAL) <= 0) break;
                if (send(client_fd, "\r\n", 2, MSG_NOSIGNAL) <= 0) break;
            }
        }
    } else if (request.find("/toggle_focus") != std::string::npos) {
        g_focus_mode_enabled = !g_focus_mode_enabled;
        std::string res = std::string("{\"focus_mode\": ") + (g_focus_mode_enabled ? "true" : "false") + "}\n";
        std::string header = "HTTP/1.1 200 OK\r\n"
                             "Content-Type: application/json\r\n"
                             "Access-Control-Allow-Origin: *\r\n"
                             "Content-Length: " + std::to_string(res.length()) + "\r\n"
                             "Connection: close\r\n\r\n";
        send(client_fd, header.c_str(), header.length(), MSG_NOSIGNAL);
        send(client_fd, res.c_str(), res.length(), MSG_NOSIGNAL);
    } else if (request.find("/reset_focus") != std::string::npos) {
        g_peak_focus_score.store(0.0, std::memory_order_relaxed);
        std::string res = "{\"reset\": true}\n";
        std::string header = "HTTP/1.1 200 OK\r\n"
                             "Content-Type: application/json\r\n"
                             "Access-Control-Allow-Origin: *\r\n"
                             "Content-Length: " + std::to_string(res.length()) + "\r\n"
                             "Connection: close\r\n\r\n";
        send(client_fd, header.c_str(), header.length(), MSG_NOSIGNAL);
        send(client_fd, res.c_str(), res.length(), MSG_NOSIGNAL);
    } else if (request.find("GET /stats") != std::string::npos || request.find("GET /flicker_stats") != std::string::npos || request.find("GET /pipeline_stats") != std::string::npos) {
        std::string json = g_detection_mgr.get_telemetry_json();
        std::string header = "HTTP/1.1 200 OK\r\n"
                             "Content-Type: application/json\r\n"
                             "Cache-Control: no-cache, no-store\r\n"
                             "Access-Control-Allow-Origin: *\r\n"
                             "Content-Length: " + std::to_string(json.length()) + "\r\n"
                             "Connection: close\r\n\r\n";
        send(client_fd, header.c_str(), header.length(), 0);
        send(client_fd, json.c_str(), json.length(), 0);
    } else {
        std::string html = HTML_DASHBOARD;
        std::string header = "HTTP/1.1 200 OK\r\n"
                             "Content-Type: text/html\r\n"
                             "Connection: close\r\n\r\n";
        send(client_fd, header.c_str(), header.length(), 0);
        send(client_fd, html.c_str(), html.length(), 0);
    }

    close(client_fd);
}

void http_server_thread_func(int port) {
#if defined(__linux__) && !defined(__ANDROID__)
    pthread_setname_np(pthread_self(), "http_server");
#endif
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) return;

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        close(server_fd);
        return;
    }

    if (listen(server_fd, 10) < 0) {
        close(server_fd);
        return;
    }

    while (g_running) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd >= 0) {
            std::thread(handle_http_client, client_fd).detach();
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    close(server_fd);
}

int main(int argc, char* argv[]) {
#if defined(__linux__) && !defined(__ANDROID__)
    pthread_setname_np(pthread_self(), "analysis_main");
#endif
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Configure high throughput decoding, 8MB transfer buffer pool, and eliminate grammar validator log storms
    setenv("MV_FLAGS_EVT3_UNSAFE_DECODER", "1", 1);
    setenv("MV_LOG_LEVEL", "ERROR", 1);
    setenv("MV_PSEE_PLUGIN_DATA_TRANSFER_BUFFER_POOL_BYTE_SIZE", "8388608", 1);

    int port = 8080;
    if (argc > 1) {
        port = std::stoi(argv[1]);
    }

    // Resolve runtime feature flags once (Phase 33 defaults: both OFF).
    g_ego_warp_enabled = env_flag("PREDATOR_ENABLE_EGO_WARP", false);
    g_combnet_prune_enabled = env_flag("PREDATOR_COMBNET_PRUNE", false);
    g_combnet_rescue_enabled = env_flag("PREDATOR_COMBNET_RESCUE", false);
    g_focus_mode_enabled = env_flag("PREDATOR_FOCUS_MODE", false);

    std::cout << "========================================================\n";
    std::cout << "  Predator — Real-Time Propeller Flicker Detector Engine\n";
    std::cout << "  Ego-Motion Compensation + TensorRT Suppression Core   \n";
    std::cout << "  Lens: 12mm f/2.5 M12 1/2.5\" (5MP)                      \n";
    std::cout << "  Build: " << PREDATOR_BUILD_ID << "\n";
    std::cout << "========================================================\n";

    try {
        std::cout << "[INFO] Opening Metavision Event Camera with EVT21 format...\n";
        Metavision::DeviceConfig dev_cfg;
        dev_cfg.set_format("EVT21");
        std::unique_ptr<Metavision::Device> device = Metavision::DeviceDiscovery::open("", dev_cfg);
        if (!device) {
            throw std::runtime_error("No Metavision event camera found");
        }

        auto* hwid = device->get_facility<Metavision::I_HW_Identification>();
        std::string cam_serial = hwid ? hwid->get_serial() : "unknown";
        std::string cam_format = hwid ? hwid->get_current_data_encoding_format() : "unknown";

        const int width = 1280;
        const int height = 720;
        std::cout << "[INFO] Camera initialized! Serial: " << cam_serial << ", Format: " << cam_format 
                  << ", Resolution: " << width << " x " << height << "\n";

        // Configure IMX636 Sensor Biases for 12mm f/2.5 Optics (Adaptive Shade & Solar Flux Tuning)
        try {
            auto *biases = device->get_facility<Metavision::I_LL_Biases>();
            if (biases) {
                // Support environment overrides for shade or high-flux tuning (defaults: diff_on=6, diff_off=6 for 80-115ft sensitivity)
                int diff_on = 6;
                int diff_off = 6;
                const char* env_on = std::getenv("PREDATOR_BIAS_DIFF_ON");
                const char* env_off = std::getenv("PREDATOR_BIAS_DIFF_OFF");
                if (env_on) diff_on = std::stoi(env_on);
                if (env_off) diff_off = std::stoi(env_off);
                biases->set("bias_diff_on", diff_on);
                biases->set("bias_diff_off", diff_off);
                biases->set("bias_refr", 20);
                biases->set("bias_fo", -8);
                std::cout << "[INFO] Adaptive IMX636 biases active: diff_on=" << diff_on 
                          << ", diff_off=" << diff_off << ", refr=+20, fo=-8.\n";
            }
        } catch (const std::exception &e) {
            std::cout << "[WARN] Could not set analog biases: " << e.what() << "\n";
        }

        // Phase 33.7a: Configure IMX636 Hardware Pixel Mask (I_DigitalEventMask)
        try {
            const char* mask_file_env = std::getenv("PREDATOR_HOT_PIXELS_FILE");
            std::string mask_filepath = mask_file_env ? mask_file_env : "hot_pixels.txt";

            // If default file doesn't exist in current working directory, check /home/orin/ev_deploy/hot_pixels.txt
            if (!mask_file_env && !std::ifstream(mask_filepath).good()) {
                if (std::ifstream("/home/orin/ev_deploy/hot_pixels.txt").good()) {
                    mask_filepath = "/home/orin/ev_deploy/hot_pixels.txt";
                }
            }

            predator::HotPixelMaskConfig mask_cfg;
            mask_cfg.sensor_width = static_cast<uint16_t>(width);
            mask_cfg.sensor_height = static_cast<uint16_t>(height);

            auto parse_res = predator::parse_hot_pixels_file(mask_filepath, mask_cfg);
            if (parse_res.success) {
                for (const auto& w : parse_res.warnings) {
                    std::cout << "[WARN] Hot pixel mask: " << w << "\n";
                }
                auto mask_status = predator::apply_hardware_pixel_mask(*device, parse_res.pixels);
                std::cout << "[INFO] " << mask_status.message << "\n";
                if (mask_status.applied) {
                    g_hardware_mask_applied = true;
                    g_hardware_mask_facility = mask_status.facility_name;
                    g_masked_hot_pixels = parse_res.pixels;
                    for (const auto& p : g_masked_hot_pixels) {
                        std::cout << "  -> Masked hot pixel (" << p.x << ", " << p.y
                                  << ") with baseline rate " << p.rate << " ev/s\n";
                    }
                }
            } else if (mask_file_env) {
                std::cout << "[WARN] Hot pixel mask file requested but could not be parsed: " << parse_res.error << "\n";
            } else {
                std::cout << "[INFO] No hot_pixels.txt found; running with all hardware pixel masks clear.\n";
            }
        } catch (const std::exception &e) {
            std::cout << "[WARN] Could not apply hardware pixel mask: " << e.what() << "\n";
        }

        predator::LensParameters lens_params;
        lens_params.focal_length_mm = 12.0;
        lens_params.pixel_pitch_um = 4.86;
        lens_params.sensor_width = width;
        lens_params.sensor_height = height;

        // High-Performance GPU Flicker & cuFFT Core (1152 parallel channels on Jetson Orin Nano)
        predator::CudaFlickerCore cuda_core(width, height, 32, 18, 4000.0, 512);

        // Phase 33.5 CFAR gate: the only detector knob is the field-wide false-alarm budget.
        {
            predator::SpectralGateConfig gate_cfg;
            gate_cfg.false_alarms_per_hour = env_positive_float("PREDATOR_CFAR_FA_PER_HOUR", gate_cfg.false_alarms_per_hour);
            cuda_core.set_spectral_gate_config(gate_cfg);
            const auto& g = cuda_core.spectral_gate_config();
            g_cfar_fa_per_hour = g.false_alarms_per_hour;
            g_cfar_threshold_db = g.cfar_threshold_db;
            std::cout << "[INFO] CFAR gate: band " << g.min_freq_hz << "-" << g.max_freq_hz << " Hz (bins "
                      << g.min_bin << "-" << g.max_bin << "), FA budget " << g.false_alarms_per_hour
                      << "/h -> Pfa/window " << g.pfa_per_window << ", threshold " << g.cfar_threshold
                      << " (" << g.cfar_threshold_db << " dB), min_sharpness " << g.min_sharpness << "\n";
        }

        // Continuous Gyroscope Warper & Stabilization Engine
        predator::ContinuousGyroWarper gyro_warper(lens_params);

        // Connect to Arduino Nicla Sense ME IMU reader on /dev/ttyACM0
        predator::NiclaSerialReader imu_reader(gyro_warper, "/dev/ttyACM0");
        imu_reader.start();

        // Start High-Rate Diagnostic Logger
        g_diag_logger.start();
        std::cout << "[INFO] High-rate CSV diagnostics logger active at " << g_diag_logger.log_path() << "\n";

        // TensorRT FP16 SpectralCombNet Engine (Frequency-Domain Propeller Classifier)
        predator::SpectralCombNetEngine spectral_engine(128);
        std::string spectral_engine_path = "/home/orin/ev_deploy/models/spectral_combnet_fp16.engine";
        bool spectral_engine_loaded = spectral_engine.load_engine(spectral_engine_path);
        if (spectral_engine_loaded) {
            std::cout << "[INFO] SpectralCombNet TRT FP16 Engine active for frequency-domain propeller detection.\n";
        } else {
            std::cout << "[WARN] SpectralCombNet TRT Engine not loaded, continuing with cuFFT peak detector.\n";
        }

        // Configurable Ego-Motion Stabilization (Phase 33: default OFF, opt in with PREDATOR_ENABLE_EGO_WARP=1).
        // Rationale: re-anchoring has no GPU ring-buffer remap, so any warp jump corrupts per-cell
        // FFT histories. Ego-motion is out of scope until that remap exists (see Phase 34 plan).
        const bool enable_ego_warp = g_ego_warp_enabled;
        std::cout << "[INFO] Pipeline Ingestion Mode: Ego-Warp=" << (enable_ego_warp ? "ACTIVE (Nicla 200Hz)" : "BYPASS (Identity)")
                  << ", CombNet-Prune=" << (g_combnet_prune_enabled ? "ON" : "OFF")
                  << ", CombNet-Rescue=" << (g_combnet_rescue_enabled ? "ON" : "OFF")
                  << ", Focus-Mode=" << (g_focus_mode_enabled ? "ON" : "OFF")
                  << ", Pure Frequency-Domain Harmonic Pipeline Active.\n";

        // Zero-CPU GPU-Resident Raw Tap Ingestion Pipeline (Phase 33.4b.c)
        predator::RawPipelineConfig pipe_cfg;
        pipe_cfg.sensor_width = width;
        pipe_cfg.sensor_height = height;
        pipe_cfg.enable_ego_warp = enable_ego_warp;
        pipe_cfg.enable_ui_frame_gen = true;
        pipe_cfg.ui_fps = 30.0;

        predator::RawPipeline raw_pipeline(pipe_cfg, cuda_core, gyro_warper);
        raw_pipeline.connect_device(device.get());

        // Connect synthesized GPU UI frame callback (Phase 33.4b.e)
        raw_pipeline.set_frame_callback([&](const uint8_t* gray_data, int w, int h, uint64_t ts) {
            cv::Mat gray(h, w, CV_8UC1, const_cast<uint8_t*>(gray_data));
            cv::Mat bgr;
            cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
            g_frame_mgr.push_frame(bgr);
        });

        // Start background display & JPEG encoder thread
        std::thread encoder_thread(display_encoder_thread_func, width, height);

        // Launch HTTP Server
        std::thread server_thread(http_server_thread_func, port);

        raw_pipeline.start();
        std::cout << "[INFO] Real-time propeller flicker detector active with Zero-CPU Raw Tap & GPU pipeline.\n";

        // Main analysis loop: 25 Hz deterministic analysis cycle with precision monotonic cadence timer
        auto next_cycle_epoch = std::chrono::steady_clock::now();
        const auto cycle_interval = std::chrono::milliseconds(40);

        uint64_t last_total_decoded = 0;
        uint64_t last_total_retained = 0;
        std::atomic<uint64_t> current_epoch_ref_us{0};
        while (g_running && raw_pipeline.is_running()) {
            next_cycle_epoch += cycle_interval;

            // 2. Snapshot metrics
            uint64_t current_decoded = raw_pipeline.total_decoded_events();
            uint64_t total_events = (current_decoded >= last_total_decoded) ? (current_decoded - last_total_decoded) : current_decoded;
            last_total_decoded = current_decoded;
            uint64_t current_retained = raw_pipeline.total_retained_events();
            uint64_t retained_events = (current_retained >= last_total_retained) ? (current_retained - last_total_retained) : current_retained;
            last_total_retained = current_retained;
            double suppressed_pct = (total_events > 0) ? (100.0 * (1.0 - (static_cast<double>(retained_events) / total_events))) : 0.0;

            g_detection_mgr.update_raw_stats(raw_pipeline.get_stats());

            predator::Vector3d omega = gyro_warper.get_latest_angular_velocity();
            double gyro_speed_deg_s = std::sqrt(omega.x * omega.x + omega.y * omega.y + omega.z * omega.z) * (180.0 / M_PI);

            int active_cells = cuda_core.get_active_cell_count(50.0);
            double foliage_dispersion_pct = (100.0 * static_cast<double>(active_cells)) / 576.0;

            DetectionManager::EgoMotionStats ego_stats;
            ego_stats.trt_suppression_active = false;
            ego_stats.imu_connected = imu_reader.is_connected();
            ego_stats.imu_packets = imu_reader.packet_count();
            ego_stats.gyro_wx = omega.x;
            ego_stats.gyro_wy = omega.y;
            ego_stats.gyro_wz = omega.z;
            ego_stats.gyro_speed_deg_s = gyro_speed_deg_s;
            ego_stats.active_cells = active_cells;
            ego_stats.foliage_dispersion_pct = foliage_dispersion_pct;
            ego_stats.total_events = total_events;
            ego_stats.retained_events = retained_events;
            ego_stats.suppressed_pct = suppressed_pct;
            g_detection_mgr.update_ego_stats(ego_stats);

            // Compute projection matrix from continuous world anchor to current camera view
            uint64_t t_anchor = current_epoch_ref_us.load();
            uint64_t t_now_host = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            uint64_t t_now_cam = gyro_warper.host_to_camera_time(t_now_host);
            if (t_anchor == 0 && t_now_cam > 0) {
                current_epoch_ref_us.store(t_now_cam);
                t_anchor = t_now_cam;
            }

            predator::Matrix3x3 H_world_to_cam = (t_now_cam > 0 && t_anchor > 0) ?
                gyro_warper.compute_forward_homography(t_anchor, t_now_cam) : predator::Matrix3x3::identity();

            // Sliding Epoch Dynamic Re-anchoring: Remap spatial grid history if angular displacement > 5.7 deg or anchor > 800ms
            if (t_anchor > 0 && t_now_cam > 0) {
                predator::Matrix3x3 R_anchor_to_now = gyro_warper.compute_rotation_matrix(t_anchor, t_now_cam);
                double cos_angle = (R_anchor_to_now.at(0, 0) + R_anchor_to_now.at(1, 1) + R_anchor_to_now.at(2, 2) - 1.0) * 0.5;
                cos_angle = std::clamp(cos_angle, -1.0, 1.0);
                double angle_rad = std::acos(cos_angle);

                if (angle_rad > 0.10 || (t_now_cam > t_anchor + 800000)) {
                    current_epoch_ref_us.store(t_now_cam);
                }
            }

            // 3. Batched cuFFT and GPU Spectral Harmonic Analysis across all 1152 cells in parallel (<0.4 ms)
            std::vector<predator::FlickerDetectionResult> raw_detections;
            cuda_core.execute_batched_spectral_analysis(gyro_speed_deg_s, raw_detections);

            // 3b. TensorRT SpectralCombNet Neural Classification on Active Cells (Shade & Weak Signal Boost)
            int spectral_eval_cells = 0;
            int spectral_detections = 0;
            float top_neural_prob = 0.0f;

            if (spectral_engine.is_ready()) {
                std::vector<int> active_cell_indices;
                std::vector<std::vector<float>> active_spectra;
                // Query active cells with event density >= 6.0 events (retains shaded/weak blade sweeps)
                cuda_core.get_active_cells_with_spectra(active_cell_indices, active_spectra, 6.0f);
                spectral_eval_cells = static_cast<int>(active_cell_indices.size());

                if (!active_cell_indices.empty()) {
                    std::vector<const float*> ptrs(active_spectra.size());
                    for (size_t i = 0; i < active_spectra.size(); ++i) {
                        ptrs[i] = active_spectra[i].data();
                    }
                    std::vector<predator::SpectralPrediction> neural_preds;
                    spectral_engine.infer_spectra(ptrs, active_cell_indices, neural_preds, 0.0f);

                    for (const auto& np : neural_preds) {
                        top_neural_prob = std::max(top_neural_prob, np.drone_prob);
                        bool is_pooled = (np.cell_idx >= 576);
                        int base_cell = is_pooled ? (np.cell_idx - 576) : np.cell_idx;
                        int patch_col = base_cell % 32;
                        int patch_row = base_cell / 32;

                        if (np.drone_prob < 0.35f) {
                            // Neural Clutter Rejection (Phase 33: opt-in via PREDATOR_COMBNET_PRUNE=1).
                            // Disabled by default: the deployed v2 engine outputs ~0 on every real
                            // spectrum (training/runtime preprocessing mismatch), so pruning would
                            // erase weak distant candidates. Re-enable only after CombNet v3 passes its gate.
                            // Prune ONLY weak/ambiguous physical candidates (SNR < 10 dB AND no micro-sieve periodic lock AND not a sharp physical blade spike).
                            // NEVER prune high-SNR, micro-sieve-locked, or sharp high-Q physical blade harmonics!
                            if (g_combnet_prune_enabled) {
                                raw_detections.erase(
                                    std::remove_if(raw_detections.begin(), raw_detections.end(),
                                        [&](const predator::FlickerDetectionResult& rd) {
                                            bool is_sharp_blade = (rd.spectral_q_factor >= 4.0 && rd.peak_power >= 60.0);
                                            return rd.patch_x == patch_col && rd.patch_y == patch_row &&
                                                   rd.peak_snr_db < 10.0f && rd.max_sieve_hits < 2 && !is_sharp_blade;
                                        }),
                                    raw_detections.end());
                            }
                            continue;
                        }

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

                        // Neural Weak-Signal Rescue: ONLY rescue if cell has true physical peak validity AND rescue is enabled!
                        if (g_combnet_rescue_enabled && !already_detected && np.has_valid_peak && np.physical_snr_db >= 8.0f && np.drone_prob >= 0.80f &&
                            np.fund_freq_hz >= 75.0f && np.fund_freq_hz <= 1000.0f) {
                            predator::FlickerDetectionResult res;
                            res.is_drone_detected = true;
                            res.fundamental_bpf_hz = np.fund_freq_hz;
                            res.estimated_rpm = (np.fund_freq_hz * 60.0) / 2.0;
                            res.confidence = 0.5f * np.drone_prob + 0.5f * std::min(1.0f, np.physical_snr_db / 15.0f);
                            res.peak_snr_db = np.physical_snr_db;
                            res.harmonic_score = np.harmonic_purity;
                            res.spectral_q_factor = np.spectral_q_factor;
                            res.spectral_flatness = np.spectral_flatness;
                            res.peak_power = std::pow(10.0f, np.physical_snr_db / 10.0f);
                            res.noise_floor = 1.0f;
                            res.patch_x = patch_col;
                            res.patch_y = patch_row;
                            res.is_neural_detection = true;
                            res.neural_drone_prob = np.drone_prob;
                            raw_detections.push_back(res);
                            spectral_detections++;
                        }
                    }
                }
            }

            ego_stats.spectral_combnet_active = spectral_engine.is_ready();
            ego_stats.spectral_eval_cells = spectral_eval_cells;
            ego_stats.spectral_detections = spectral_detections;
            g_detection_mgr.update_ego_stats(ego_stats);

            for (auto& res : raw_detections) {
                double world_x = res.patch_x * 40.0 + 20.0;
                double world_y = res.patch_y * 40.0 + 20.0;

                // Project world coordinate to current camera viewpoint (only if ego warping active)
                double cam_x = world_x;
                double cam_y = world_y;
                if (enable_ego_warp && t_now_cam > 0 && t_anchor > 0) {
                    predator::Vector3d p_world(world_x, world_y, 1.0);
                    predator::Vector3d p_cam = H_world_to_cam * p_world;
                    if (std::abs(p_cam.z) > 1e-6) {
                        cam_x = p_cam.x / p_cam.z;
                        cam_y = p_cam.y / p_cam.z;
                    }
                }

                res.centroid_px_x = static_cast<int>(std::round(cam_x));
                res.centroid_px_y = static_cast<int>(std::round(cam_y));
                lens_params.pixel_to_angles(cam_x, cam_y, res.azimuth_deg, res.elevation_deg);
            }

            // 4. Apply Drone-Level Airframe Cluster Fusion & M-of-N Tracker
            auto filtered_detections = predator::SpatialFlickerClusterer::filter_and_cluster(raw_detections, 6);
            auto all_tracks = predator::SpatialFlickerClusterer::get_all_tracks();
            auto roi_diag = cuda_core.get_roi_diagnostics(17, 24, 5, 12); // Shaded drone ROI (cols 17..24, rows 5..12)

            g_detection_mgr.update_detections(filtered_detections, all_tracks, roi_diag);

            // Precision Cadence Sleeping: Maintain steady 25 Hz without cycle drift
            auto now = std::chrono::steady_clock::now();
            if (now < next_cycle_epoch) {
                std::this_thread::sleep_until(next_cycle_epoch);
            } else {
                next_cycle_epoch = now;
            }

            // Log diagnostic record
            DiagnosticsLogRecord log_rec;
            log_rec.timestamp_us = (t_now_cam > 0) ? t_now_cam : t_anchor;
            log_rec.host_ms = t_now_host / 1000;
            log_rec.num_targets = static_cast<int>(filtered_detections.size());
            log_rec.gyro_wx = omega.x;
            log_rec.gyro_wy = omega.y;
            log_rec.gyro_wz = omega.z;
            log_rec.gyro_speed_deg_s = gyro_speed_deg_s;
            log_rec.active_cells = active_cells;
            log_rec.foliage_dispersion_pct = foliage_dispersion_pct;
            log_rec.raw_events = total_events;
            log_rec.retained_events = retained_events;
            log_rec.suppressed_pct = suppressed_pct;
            log_rec.neural_eval_cells = spectral_eval_cells;
            log_rec.neural_detections = spectral_detections;
            log_rec.top_neural_prob = top_neural_prob;

            int tentative_cnt = 0;
            for (const auto& trk : all_tracks) {
                if (trk.state == predator::SpatialFlickerClusterer::TrackState::TENTATIVE) {
                    tentative_cnt++;
                }
            }
            log_rec.tentative_count = tentative_cnt;
            log_rec.roi_events = roi_diag.total_events;
            log_rec.roi_max_cell = roi_diag.max_cell_events;
            log_rec.roi_max_sieve = roi_diag.max_sieve_hits;
            log_rec.roi_active_cells = roi_diag.active_cells;

            if (!filtered_detections.empty()) {
                const auto& top_t = filtered_detections[0];
                log_rec.target_bpf = top_t.fundamental_bpf_hz;
                log_rec.target_rpm = top_t.estimated_rpm;
                log_rec.target_snr = top_t.peak_snr_db;
                log_rec.target_q = top_t.spectral_q_factor;
                log_rec.target_peak_power = top_t.peak_power;
                log_rec.target_noise_floor = top_t.noise_floor;
                log_rec.cam_x = top_t.centroid_px_x;
                log_rec.cam_y = top_t.centroid_px_y;
                log_rec.world_x = top_t.patch_x * 40 + 20;
                log_rec.world_y = top_t.patch_y * 40 + 20;
                log_rec.azimuth_deg = top_t.azimuth_deg;
                log_rec.elevation_deg = top_t.elevation_deg;
                log_rec.track_id = top_t.track_id;
                log_rec.hit_count = top_t.hit_count;
                log_rec.miss_count = top_t.miss_count;
            } else if (!all_tracks.empty()) {
                const auto& top_t = all_tracks[0].last_detection;
                log_rec.target_bpf = top_t.fundamental_bpf_hz;
                log_rec.target_rpm = top_t.estimated_rpm;
                log_rec.target_snr = top_t.peak_snr_db;
                log_rec.target_q = top_t.spectral_q_factor;
                log_rec.target_peak_power = top_t.peak_power;
                log_rec.target_noise_floor = top_t.noise_floor;
                log_rec.cam_x = top_t.centroid_px_x;
                log_rec.cam_y = top_t.centroid_px_y;
                log_rec.world_x = top_t.patch_x * 40 + 20;
                log_rec.world_y = top_t.patch_y * 40 + 20;
                log_rec.azimuth_deg = top_t.azimuth_deg;
                log_rec.elevation_deg = top_t.elevation_deg;
                log_rec.track_id = all_tracks[0].track_id;
                log_rec.hit_count = all_tracks[0].hit_count;
                log_rec.miss_count = all_tracks[0].miss_count;
            }
            g_diag_logger.log_frame(log_rec);

            // Structured diagnostic file & console logging
            static uint64_t s_frame_idx = 0;
            s_frame_idx++;
            if (roi_diag.total_events > 0 || !raw_detections.empty() || !all_tracks.empty() || (s_frame_idx % 25 == 0)) {
                char debug_hdr[384];
                snprintf(debug_hdr, sizeof(debug_hdr),
                         "[PIPELINE F#%lu] Raw: %lu ev | Retained: %lu ev (%.1f%% supp) | CombNet: eval=%d det=%d (p_max=%.2f) | ROI(cols 17-24, rows 5-12): Ev=%.0f MaxCell=%.0f MaxSieve=%u ActCells=%d | Cands: %zu | Confirmed: %zu | Tracks: %zu",
                         (unsigned long)s_frame_idx,
                         (unsigned long)total_events,
                         (unsigned long)retained_events,
                         suppressed_pct,
                         spectral_eval_cells, spectral_detections, top_neural_prob,
                         roi_diag.total_events, roi_diag.max_cell_events, roi_diag.max_sieve_hits, roi_diag.active_cells,
                         raw_detections.size(), filtered_detections.size(), all_tracks.size());
                g_diag_logger.log_debug_line(debug_hdr);

                for (const auto& trk : all_tracks) {
                    char trk_line[384];
                    const char* state_str = (trk.state == predator::SpatialFlickerClusterer::TrackState::CONFIRMED) ? (trk.miss_count > 0 ? "COAST" : "CONF") : "TENT";
                    const char* net_tag = trk.last_detection.is_neural_detection ? "+NET" : "";
                    snprintf(trk_line, sizeof(trk_line),
                             "  -> Trk#%d [%s%s] hit=%d miss=%d age=%d | BPF=%.1f Hz (%.0f RPM) | SNR=%.1f dB | Flat=%.3f | Pos=(%d,%d) Az=%+.1f El=%+.1f",
                             trk.track_id, state_str, net_tag, trk.hit_count, trk.miss_count, trk.total_age,
                             trk.last_detection.fundamental_bpf_hz, trk.last_detection.estimated_rpm,
                             trk.last_detection.peak_snr_db, trk.last_detection.spectral_flatness,
                             trk.last_detection.centroid_px_x, trk.last_detection.centroid_px_y,
                             trk.last_detection.azimuth_deg, trk.last_detection.elevation_deg);
                    g_diag_logger.log_debug_line(trk_line);
                }

                if (!all_tracks.empty() || !filtered_detections.empty() || roi_diag.total_events >= 10.0f) {
                    std::cout << debug_hdr << "\n";
                    for (const auto& trk : all_tracks) {
                        const char* state_str = (trk.state == predator::SpatialFlickerClusterer::TrackState::CONFIRMED) ? (trk.miss_count > 0 ? "COAST" : "CONF") : "TENT";
                        const char* net_tag = trk.last_detection.is_neural_detection ? "+NET" : "";
                        std::cout << "  [TRACK #" << trk.track_id << " " << state_str << net_tag << "] hit=" << trk.hit_count 
                                  << " miss=" << trk.miss_count << " BPF=" << trk.last_detection.fundamental_bpf_hz 
                                  << "Hz SNR=" << trk.last_detection.peak_snr_db << "dB pos=(" 
                                  << trk.last_detection.centroid_px_x << "," << trk.last_detection.centroid_px_y << ")\n";
                    }
                }
            }

            // Reset sieve hit accumulators every 3 analysis cycles (~120ms) so slow hover blade chops (75-120 Hz)
            // have sufficient window to accumulate periodic micro-sieve hits
            if (s_frame_idx % 3 == 0) {
                cuda_core.reset_sieve_hit_accumulators();
            }
        }

        std::cout << "[INFO] Shutting down camera, IMU reader, and diagnostics logger...\n";
        g_running = false;
        g_diag_logger.stop();
        imu_reader.stop();
        raw_pipeline.stop();
        g_frame_mgr.notify_all();
        g_stream_broadcaster.notify_all();

        if (encoder_thread.joinable()) {
            encoder_thread.join();
        }
        if (server_thread.joinable()) {
            server_thread.join();
        }

        std::cout << "[INFO] Engine terminated cleanly.\n";
        return 0;

    } catch (const Metavision::CameraException& e) {
        std::cerr << "[ERROR] CameraException: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception: " << e.what() << "\n";
        return 2;
    }
}
