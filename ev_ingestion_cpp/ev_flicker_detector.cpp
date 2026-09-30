#include <iostream>
#include <vector>
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

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>

#include <metavision/sdk/stream/camera.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/core/utils/colors.h>
#include <metavision/hal/facilities/i_ll_biases.h>
#include <opencv2/opencv.hpp>

#include "flicker_dsp.hpp"
#include "ego_motion.hpp"
#include "event_suppression_trt.hpp"

// Global shutdown flag
static std::atomic<bool> g_running{true};

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

    bool get_frame(uint64_t last_frame_id, std::vector<uchar>& out_jpeg, uint64_t& out_frame_id, int timeout_ms = 100) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return frame_id_ > last_frame_id || !g_running; })) {
            if (!g_running) return false;
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
        bool imu_connected{false};
        uint64_t imu_packets{0};
        double gyro_wx{0.0};
        double gyro_wy{0.0};
        double gyro_wz{0.0};
        double suppressed_pct{0.0};
        uint64_t total_events{0};
        uint64_t retained_events{0};
    };

    void update_detections(const std::vector<predator::FlickerDetectionResult>& detections) {
        std::lock_guard<std::mutex> lock(mutex_);
        active_detections_ = detections;
    }

    void update_ego_stats(const EgoMotionStats& stats) {
        std::lock_guard<std::mutex> lock(mutex_);
        ego_stats_ = stats;
    }

    std::vector<predator::FlickerDetectionResult> get_detections() {
        std::lock_guard<std::mutex> lock(mutex_);
        return active_detections_;
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
           << "  \"lens\": {\"model\": \"Edmund Optics 8mm f/8 M12\", \"fl_mm\": 8.0, \"hfov_deg\": 44.5, \"vfov_deg\": 25.1},\n"
           << "  \"ego_motion\": {\n"
           << "    \"imu_connected\": " << (ego_stats_.imu_connected ? "true" : "false") << ",\n"
           << "    \"imu_packets\": " << ego_stats_.imu_packets << ",\n"
           << "    \"trt_suppression_active\": " << (ego_stats_.trt_suppression_active ? "true" : "false") << ",\n"
           << "    \"gyro_rad_s\": [" << ego_stats_.gyro_wx << ", " << ego_stats_.gyro_wy << ", " << ego_stats_.gyro_wz << "],\n"
           << "    \"suppressed_events_pct\": " << ego_stats_.suppressed_pct << ",\n"
           << "    \"total_raw_events\": " << ego_stats_.total_events << ",\n"
           << "    \"retained_imo_events\": " << ego_stats_.retained_events << "\n"
           << "  },\n"
           << "  \"num_targets\": " << active_detections_.size() << ",\n"
           << "  \"targets\": [\n";
        
        for (size_t i = 0; i < active_detections_.size(); ++i) {
            const auto& t = active_detections_[i];
            ss << "    {\n"
               << "      \"target_id\": " << (i + 1) << ",\n"
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
    EgoMotionStats ego_stats_;
};

static DetectionManager g_detection_mgr;

void display_encoder_thread_func(int width, int height) {
    std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, 75};
    cv::Mat frame;

    while (g_running) {
        if (g_frame_mgr.wait_for_frame(frame, 50)) {
            auto active_dets = g_detection_mgr.get_detections();
            auto ego_stats = g_detection_mgr.get_ego_stats();

            // Draw detection bounding boxes and HUD on frame
            for (size_t i = 0; i < active_dets.size(); ++i) {
                const auto& d = active_dets[i];
                int bx = d.centroid_px_x - 40;
                int by = d.centroid_px_y - 40;
                cv::Rect target_rect(std::max(0, bx), std::max(0, by), 
                                     std::min(80, width - std::max(0, bx)), 
                                     std::min(80, height - std::max(0, by)));
                
                // Neon Cyan / Green target box
                cv::rectangle(frame, target_rect, cv::Scalar(0, 255, 128), 2);
                
                // Text label
                char label[128];
                snprintf(label, sizeof(label), "DRONE %.0fHz (%.0f RPM) [%.1fdB]", d.fundamental_bpf_hz, d.estimated_rpm, d.peak_snr_db);
                cv::putText(frame, label, cv::Point(target_rect.x, std::max(16, target_rect.y - 6)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 128), 1, cv::LINE_AA);
            }

            // Top HUD
            std::string hud_top = "PREDATOR-01 | EO 8mm f/8 | DDHF + Ego-Motion Core";
            cv::putText(frame, hud_top, cv::Point(16, 28), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(220, 220, 220), 2, cv::LINE_AA);

            // Ego-Motion & Suppression HUD Badge
            char ego_badge[160];
            snprintf(ego_badge, sizeof(ego_badge), "NICLA IMU: %s (%lu PKTS) | TRT SUPPRESS: %s (%.0f%% REJECTED)",
                     ego_stats.imu_connected ? "LOCKED 200Hz" : "OFFLINE",
                     (unsigned long)ego_stats.imu_packets,
                     ego_stats.trt_suppression_active ? "ACTIVE" : "PASS-THRU",
                     ego_stats.suppressed_pct);
            cv::putText(frame, ego_badge, cv::Point(16, 56), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                        ego_stats.imu_connected ? cv::Scalar(0, 255, 200) : cv::Scalar(180, 180, 180), 1, cv::LINE_AA);

            std::vector<uchar> jpeg_buf;
            cv::imencode(".jpg", frame, jpeg_buf, encode_params);
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
        .video-container img {
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
        <div class="tag">Active Defense</div>
    </header>
    <main>
        <div class="stream-panel">
            <div class="panel-header">Stabilized Neuromorphic Focal Stream (1280x720)</div>
            <div class="video-container">
                <img src="/stream.mjpg" alt="Live Event Stream">
            </div>
        </div>
        <div class="sidebar">
            <div class="card">
                <div class="card-title">Ego-Motion & IMU Gyro Gating</div>
                <div class="metric-grid">
                    <div class="metric-box">
                        <div class="metric-label">Nicla IMU (BHI260)</div>
                        <div class="metric-value" id="val-imu" style="font-size:13px; color:var(--accent-green);">LOCKED</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">TRT Suppression</div>
                        <div class="metric-value" id="val-trt" style="font-size:13px;">ACTIVE</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Gyro Rates (rad/s)</div>
                        <div class="metric-value" id="val-gyro" style="font-size:12px; color:var(--accent-cyan);">[0.0, 0.0, 0.0]</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Suppressed Events</div>
                        <div class="metric-value" id="val-suppressed">0%</div>
                    </div>
                </div>
            </div>
            <div class="card">
                <div class="card-title">Optical & DSP Geometry</div>
                <div class="metric-grid">
                    <div class="metric-box">
                        <div class="metric-label">Optics</div>
                        <div class="metric-value" style="font-size:14px;">8mm f/8 M12</div>
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
        async function fetchStats() {
            try {
                const res = await fetch('/stats');
                if (res.ok) {
                    const data = await res.json();
                    document.getElementById('val-imu').textContent = data.ego_motion.imu_connected ? ("200Hz (" + (data.ego_motion.imu_packets || 0) + ")") : "OFFLINE";
                    document.getElementById('val-imu').style.color = data.ego_motion.imu_connected ? "var(--accent-green)" : "var(--accent-red)";
                    document.getElementById('val-trt').textContent = data.ego_motion.trt_suppression_active ? "ACTIVE" : "PASS-THRU";
                    const g = data.ego_motion.gyro_rad_s || [0,0,0];
                    document.getElementById('val-gyro').textContent = `[${g[0].toFixed(2)}, ${g[1].toFixed(2)}, ${g[2].toFixed(2)}]`;
                    document.getElementById('val-suppressed').textContent = (data.ego_motion.suppressed_events_pct || 0).toFixed(0) + "%";
                    document.getElementById('target-count').textContent = data.num_targets;
                    
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
            setTimeout(fetchStats, 200);
        }
        fetchStats();
    </script>
</body>
</html>
)html";

void handle_http_client(int client_fd) {
    char buffer[4096];
    ssize_t bytes_read = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
    if (bytes_read <= 0) {
        close(client_fd);
        return;
    }
    buffer[bytes_read] = '\0';
    std::string request(buffer);

    if (request.find("GET /stream.mjpg") != std::string::npos) {
        std::string header = "HTTP/1.1 200 OK\r\n"
                             "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
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
    } else if (request.find("GET /stats") != std::string::npos || request.find("GET /flicker_stats") != std::string::npos) {
        std::string json = g_detection_mgr.get_telemetry_json();
        std::string header = "HTTP/1.1 200 OK\r\n"
                             "Content-Type: application/json\r\n"
                             "Access-Control-Allow-Origin: *\r\n"
                             "Content-Length: " + std::to_string(json.length()) + "\r\n"
                             "Connection: close\r\n\r\n";
        send(client_fd, header.c_str(), header.length(), 0);
        send(client_fd, json.c_str(), json.length(), 0);
    } else {
        std::string html = HTML_DASHBOARD;
        std::string header = "HTTP/1.1 200 OK\r\n"
                             "Content-Type: text/html\r\n"
                             "Content-Length: " + std::to_string(html.length()) + "\r\n"
                             "Connection: close\r\n\r\n";
        send(client_fd, header.c_str(), header.length(), 0);
        send(client_fd, html.c_str(), html.length(), 0);
    }

    close(client_fd);
}

void http_server_thread_func(int port) {
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
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Configure high throughput decoding and eliminate grammar validator log storms
    setenv("MV_FLAGS_EVT3_UNSAFE_DECODER", "1", 1);
    setenv("MV_LOG_LEVEL", "ERROR", 1);
    setenv("MV_PSEE_PLUGIN_DATA_TRANSFER_BUFFER_POOL_BYTE_SIZE", "67108864", 1);

    int port = 8080;
    if (argc > 1) {
        port = std::stoi(argv[1]);
    }

    std::cout << "========================================================\n";
    std::cout << "  Predator — Real-Time Propeller Flicker Detector Engine\n";
    std::cout << "  Ego-Motion Compensation + TensorRT Suppression Core   \n";
    std::cout << "  Lens: Edmund Optics 8mm f/8 M12 (#27052)              \n";
    std::cout << "========================================================\n";

    try {
        std::cout << "[INFO] Opening Metavision Event Camera...\n";
        Metavision::Camera camera = Metavision::Camera::from_first_available();

        int width = camera.geometry().get_width();
        int height = camera.geometry().get_height();
        std::cout << "[INFO] Camera initialized! Resolution: " << width << " x " << height << "\n";

        // Attempt bias configuration for IMX636 analog front-end (Outdoor nominal profile)
        try {
            auto *biases = camera.get_device().get_facility<Metavision::I_LL_Biases>();
            if (biases) {
                std::cout << "[INFO] Engaging IMX636 outdoor nominal profile (factory contrast, clean background)...\n";
                biases->set("bias_diff_on", 0);
                biases->set("bias_diff_off", 0);
                biases->set("bias_fo", 0);
                biases->set("bias_hpf", 0);
                biases->set("bias_refr", 0);
                std::cout << "[INFO] Outdoor nominal bias profile successfully engaged.\n";
            }
        } catch (const std::exception &e) {
            std::cout << "[WARN] Could not set analog biases: " << e.what() << "\n";
        }

        predator::LensParameters lens_params;
        lens_params.focal_length_mm = 8.0;
        lens_params.pixel_pitch_um = 4.86;
        lens_params.sensor_width = width;
        lens_params.sensor_height = height;

        // Base spatial grid: 32 cols x 18 rows (each cell 40x40 pixels)
        predator::SpatialPatchGrid patch_grid(32, 18, 4000.0, 512);
        predator::PropellerFlickerAnalyzer analyzer(4000.0, 512, 140.0, 285.0, 18.0);
        analyzer.set_lens(lens_params);

        // Continuous Gyroscope Warper & Stabilization Engine
        predator::ContinuousGyroWarper gyro_warper(lens_params);

        // Connect to Arduino Nicla Sense ME IMU reader on /dev/ttyACM0
        predator::NiclaSerialReader imu_reader(gyro_warper, "/dev/ttyACM0");
        imu_reader.start();

        // 2-Bin Temporal Event Stack Accumulator for TensorRT Dynamic Suppression
        predator::TemporalEventStackAccumulator event_stack_acc(640, 360, 40000); // 40ms frames

        // TensorRT FP16 Dynamic Motion Suppression Engine
        predator::AnticipatorySuppressionEngine suppression_engine(640, 360);
        std::string engine_path = "/home/orin/ev_deploy/models/event_suppression_fp16.engine";
        suppression_engine.load_engine(engine_path);

        std::mutex grid_mutex;
        std::atomic<uint64_t> total_raw_counter{0};
        std::atomic<uint64_t> retained_counter{0};
        std::atomic<uint64_t> current_epoch_ref_us{0};

        // Visual frame generation for UI (30 FPS)
        Metavision::PeriodicFrameGenerationAlgorithm frame_gen(width, height, 25000, 30.0, Metavision::ColorPalette::Dark);

        frame_gen.set_output_callback([&](Metavision::timestamp ts, cv::Mat& frame) {
            g_frame_mgr.push_frame(frame);
        });

        // Start background display & JPEG encoder thread
        std::thread encoder_thread(display_encoder_thread_func, width, height);

        // Fast CD callback: zero-copy ring buffer event ingestion with Ego-Motion Stabilization
        camera.cd().add_callback([&](const Metavision::EventCD* begin, const Metavision::EventCD* end) {
            frame_gen.process_events(begin, end);

            uint64_t t_ref = current_epoch_ref_us.load();

            std::lock_guard<std::mutex> lock(grid_mutex);
            for (auto it = begin; it != end; ++it) {
                total_raw_counter++;

                if (t_ref == 0) {
                    t_ref = it->t;
                    current_epoch_ref_us.store(t_ref);
                }

                // Ingest into 2-bin temporal stack accumulator
                event_stack_acc.ingest_event(it->x, it->y, it->t, it->p);

                // Tier 1: Continuous Gyroscope Point-Wise Coordinate Stabilization
                double stab_x = 0.0, stab_y = 0.0;
                bool valid = gyro_warper.unwarp_event(it->x, it->y, it->t, t_ref, stab_x, stab_y);

                // Tier 2: Anticipatory Motion Suppression Gate (UZH RSS 2026 TensorRT)
                bool retain = suppression_engine.is_event_retained(stab_x, stab_y, 0.30f);

                if (valid && retain) {
                    retained_counter++;
                    patch_grid.ingest_event(static_cast<int>(std::round(stab_x)), static_cast<int>(std::round(stab_y)), it->t);
                }
            }
        });

        // Launch HTTP Server
        std::thread server_thread(http_server_thread_func, port);

        camera.start();
        std::cout << "[INFO] Real-time propeller flicker detector active with ego-motion compensation.\n";

        struct CellSnapshot {
            int col;
            int row;
            bool is_pooled;
            std::vector<double> history;
        };

        // Main analysis loop: 25 Hz analysis cycle with lock-free snapshot processing
        while (g_running && camera.is_running()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(40));

            // 1. Run TensorRT Anticipatory Suppression Inference on Latest 2-Bin Event Stack
            std::vector<float> event_stack;
            if (event_stack_acc.get_latest_stack(event_stack)) {
                suppression_engine.infer(event_stack, 40.0f); // 40ms lookahead
            }

            // 2. Snapshot metrics
            uint64_t total_events = total_raw_counter.exchange(0);
            uint64_t retained_events = retained_counter.exchange(0);
            double suppressed_pct = (total_events > 0) ? (100.0 * (1.0 - (static_cast<double>(retained_events) / total_events))) : 0.0;

            DetectionManager::EgoMotionStats ego_stats;
            ego_stats.trt_suppression_active = suppression_engine.is_ready();
            ego_stats.imu_connected = imu_reader.is_connected();
            ego_stats.imu_packets = imu_reader.packet_count();
            ego_stats.total_events = total_events;
            ego_stats.retained_events = retained_events;
            ego_stats.suppressed_pct = suppressed_pct;

            predator::Vector3d omega;
            gyro_warper.get_angular_velocity(current_epoch_ref_us.load(), omega);
            ego_stats.gyro_wx = omega.x;
            ego_stats.gyro_wy = omega.y;
            ego_stats.gyro_wz = omega.z;
            g_detection_mgr.update_ego_stats(ego_stats);

            // Advance reference epoch for continuous stabilization
            current_epoch_ref_us.store(0);

            std::vector<CellSnapshot> active_snapshots;

            {
                std::lock_guard<std::mutex> lock(grid_mutex);

                // Scale 1: Single cell (40x40 px)
                for (int r = 0; r < patch_grid.grid_rows(); ++r) {
                    for (int c = 0; c < patch_grid.grid_cols(); ++c) {
                        if (patch_grid.is_cell_active(c, r, 25.0)) {
                            active_snapshots.push_back({c, r, false, patch_grid.get_cell_history(c, r)});
                        }
                    }
                }

                // Scale 2: 2x2 pooled cells (80x80 px)
                for (int r = 0; r < patch_grid.grid_rows(); ++r) {
                    for (int c = 0; c < patch_grid.grid_cols(); ++c) {
                        if (patch_grid.is_pooled_patch_active(c, r, 40.0)) {
                            active_snapshots.push_back({c, r, true, patch_grid.get_pooled_patch_history(c, r)});
                        }
                    }
                }
            } // Lock released immediately (< 20 us)

            // Compute FFTs completely lock-free outside camera thread
            std::vector<predator::FlickerDetectionResult> raw_detections;
            for (const auto& snap : active_snapshots) {
                auto candidates = analyzer.analyze_time_series_candidates(snap.history, 2, 2);
                for (auto& res : candidates) {
                    if (res.is_drone_detected) {
                        double cx, cy;
                        if (snap.is_pooled) {
                            patch_grid.get_pooled_patch_center(snap.col, snap.row, cx, cy);
                        } else {
                            patch_grid.get_patch_center(snap.col, snap.row, cx, cy);
                        }
                        res.patch_x = snap.col;
                        res.patch_y = snap.row;
                        res.centroid_px_x = static_cast<int>(cx);
                        res.centroid_px_y = static_cast<int>(cy);
                        lens_params.pixel_to_angles(cx, cy, res.azimuth_deg, res.elevation_deg);
                        raw_detections.push_back(res);
                    }
                }
            }

            // Apply Drone-Level Airframe Cluster Fusion & M-of-N Tracker
            auto filtered_detections = predator::SpatialFlickerClusterer::filter_and_cluster(raw_detections, 6);
            g_detection_mgr.update_detections(filtered_detections);
        }

        std::cout << "[INFO] Shutting down camera & IMU reader...\n";
        g_running = false;
        imu_reader.stop();
        g_frame_mgr.notify_all();
        g_stream_broadcaster.notify_all();

        camera.stop();

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
