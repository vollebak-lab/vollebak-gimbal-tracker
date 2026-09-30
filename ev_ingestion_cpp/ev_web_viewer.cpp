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
#include <opencv2/opencv.hpp>

// Global shutdown flag
static std::atomic<bool> g_running{true};

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        g_running = false;
    }
}

// Thread-safe Frame Broadcaster for HTTP MJPEG streaming
class FrameBroadcaster {
public:
    void update_frame(const std::vector<uchar>& jpeg_data, uint64_t ev_rate, double fps, int width, int height) {
        std::lock_guard<std::mutex> lock(mutex_);
        current_jpeg_ = jpeg_data;
        ev_rate_ = ev_rate;
        fps_ = fps;
        width_ = width;
        height_ = height;
        frame_id_++;
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

    std::string get_stats_json() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::ostringstream ss;
        ss << "{"
           << "\"status\": \"ONLINE\","
           << "\"sensor\": \"Sony IMX636 HD\","
           << "\"integrator\": \"IDS Imaging Development Systems\","
           << "\"resolution\": \"" << width_ << "x" << height_ << "\","
           << "\"event_rate_ev_s\": " << ev_rate_ << ","
           << "\"event_rate_mev_s\": " << (ev_rate_ / 1000000.0) << ","
           << "\"stream_fps\": " << fps_ << ","
           << "\"frame_id\": " << frame_id_
           << "}";
        return ss.str();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<uchar> current_jpeg_;
    uint64_t frame_id_{0};
    uint64_t ev_rate_{0};
    double fps_{0.0};
    int width_{1280};
    int height_{720};
};

static FrameBroadcaster g_broadcaster;

// HTML5 Dashboard Page
static const char* HTML_DASHBOARD = R"html(
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predator — IDS IMX636 Live Neuromorphic Stream</title>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-card: #161b22;
            --border-color: #30363d;
            --accent-cyan: #58a6ff;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-amber: #d29922;
            --text-primary: #c9d1d9;
            --text-muted: #8b949e;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            overflow-x: hidden;
        }
        header {
            background-color: var(--bg-card);
            border-bottom: 1px solid var(--border-color);
            padding: 16px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .logo-badge {
            background: linear-gradient(135deg, #1f6feb, #238636);
            color: #fff;
            font-weight: 800;
            font-size: 14px;
            padding: 4px 10px;
            border-radius: 6px;
            letter-spacing: 1px;
        }
        .title {
            font-size: 18px;
            font-weight: 600;
            color: #fff;
        }
        .status-pill {
            display: flex;
            align-items: center;
            gap: 8px;
            background-color: rgba(63, 185, 80, 0.15);
            border: 1px solid var(--accent-green);
            color: var(--accent-green);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
        }
        .pulse-dot {
            width: 8px;
            height: 8px;
            background-color: var(--accent-green);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--accent-green);
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { opacity: 0.4; transform: scale(0.9); }
            50% { opacity: 1; transform: scale(1.1); }
            100% { opacity: 0.4; transform: scale(0.9); }
        }
        main {
            flex: 1;
            padding: 24px;
            max-width: 1440px;
            margin: 0 auto;
            width: 100%;
            display: grid;
            grid-template-columns: 1fr 340px;
            gap: 24px;
        }
        @media (max-width: 1024px) {
            main { grid-template-columns: 1fr; }
        }
        .viewport-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        }
        .viewport-header {
            padding: 12px 16px;
            background-color: rgba(255,255,255,0.02);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
            color: var(--text-muted);
        }
        .legend {
            display: flex;
            gap: 16px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .legend-color {
            width: 10px;
            height: 10px;
            border-radius: 2px;
        }
        .viewport-body {
            position: relative;
            background-color: #05070a;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 480px;
        }
        .stream-img {
            width: 100%;
            height: auto;
            max-height: 720px;
            object-fit: contain;
            display: block;
        }
        .telemetry-sidebar {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .telemetry-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 18px;
        }
        .card-title {
            font-size: 14px;
            font-weight: 600;
            color: #fff;
            margin-bottom: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        .stat-box {
            background-color: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 12px;
        }
        .stat-box.full {
            grid-column: span 2;
        }
        .stat-label {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .stat-value {
            font-size: 18px;
            font-weight: 700;
            color: #fff;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }
        .stat-value.highlight {
            color: var(--accent-cyan);
        }
        .stat-value.success {
            color: var(--accent-green);
        }
        .meta-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
            font-size: 13px;
        }
        .meta-row {
            display: flex;
            justify-content: space-between;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .meta-row:last-child { border-bottom: none; }
        .meta-k { color: var(--text-muted); }
        .meta-v { font-weight: 500; color: #fff; font-family: monospace; }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <div class="logo-badge">PREDATOR-01</div>
            <div class="title">Neuromorphic Live Stream Viewer</div>
        </div>
        <div class="status-pill">
            <div class="pulse-dot"></div>
            <span>LIVE HARDWARE STREAMING</span>
        </div>
    </header>
    <main>
        <div class="viewport-card">
            <div class="viewport-header">
                <div>LIVE VIEWPORT &bull; 1280 &times; 720 @ 30 FPS</div>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #3fb950;"></div>
                        <span>ON (Polarity +1)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #f85149;"></div>
                        <span>OFF (Polarity 0)</span>
                    </div>
                </div>
            </div>
            <div class="viewport-body">
                <img class="stream-img" src="/stream.mjpg" alt="IDS IMX636 Live Stream" />
            </div>
        </div>
        <div class="telemetry-sidebar">
            <div class="telemetry-card">
                <div class="card-title">Live Pipeline Metrics</div>
                <div class="stat-grid">
                    <div class="stat-box full">
                        <div class="stat-label">Event Throughput</div>
                        <div class="stat-value highlight" id="ev-rate">0.00 MEv/s</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Video FPS</div>
                        <div class="stat-value success" id="fps-val">30.0</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Total Events/s</div>
                        <div class="stat-value" id="ev-total">0</div>
                    </div>
                </div>
            </div>
            <div class="telemetry-card">
                <div class="card-title">Hardware Topology</div>
                <div class="meta-list">
                    <div class="meta-row">
                        <span class="meta-k">Camera Model</span>
                        <span class="meta-v">IDS UE-39B0XCP-E</span>
                    </div>
                    <div class="meta-row">
                        <span class="meta-k">Event Sensor</span>
                        <span class="meta-v">Sony IMX636 HD</span>
                    </div>
                    <div class="meta-row">
                        <span class="meta-k">Host Platform</span>
                        <span class="meta-v">Jetson Orin Nano 8GB</span>
                    </div>
                    <div class="meta-row">
                        <span class="meta-k">Interface Link</span>
                        <span class="meta-v">USB 3.0 (5000 Mbps)</span>
                    </div>
                    <div class="meta-row">
                        <span class="meta-k">Driver Layer</span>
                        <span class="meta-v">OpenEB 5.2.0 Treuzell</span>
                    </div>
                </div>
            </div>
        </div>
    </main>
    <script>
        setInterval(async () => {
            try {
                const res = await fetch('/stats');
                if (res.ok) {
                    const data = await res.json();
                    document.getElementById('ev-rate').innerText = data.event_rate_mev_s.toFixed(2) + ' MEv/s';
                    document.getElementById('fps-val').innerText = data.stream_fps.toFixed(1);
                    document.getElementById('ev-total').innerText = Number(data.event_rate_ev_s).toLocaleString();
                }
            } catch (e) {}
        }, 1000);
    </script>
</body>
</html>
)html";

// Handle individual HTTP client connections
void handle_client(int client_fd) {
    char buffer[2048];
    ssize_t bytes_read = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
    if (bytes_read <= 0) {
        close(client_fd);
        return;
    }
    buffer[bytes_read] = '\0';
    std::string request(buffer);

    if (request.find("GET /stats") == 0 || request.find("GET /stats ") != std::string::npos) {
        std::string json = g_broadcaster.get_stats_json();
        std::ostringstream response;
        response << "HTTP/1.1 200 OK\r\n"
                 << "Content-Type: application/json\r\n"
                 << "Access-Control-Allow-Origin: *\r\n"
                 << "Content-Length: " << json.size() << "\r\n"
                 << "Connection: close\r\n\r\n"
                 << json;
        std::string resp_str = response.str();
        send(client_fd, resp_str.c_str(), resp_str.size(), 0);
        close(client_fd);
        return;
    }

    if (request.find("GET /stream.mjpg") != std::string::npos) {
        std::ostringstream header;
        header << "HTTP/1.1 200 OK\r\n"
               << "Server: Predator-Live-Streamer\r\n"
               << "Connection: close\r\n"
               << "Max-Age: 0\r\n"
               << "Expires: 0\r\n"
               << "Cache-Control: no-cache, private\r\n"
               << "Pragma: no-cache\r\n"
               << "Content-Type: multipart/x-mixed-replace; boundary=--frame\r\n\r\n";
        std::string h_str = header.str();
        if (send(client_fd, h_str.c_str(), h_str.size(), MSG_NOSIGNAL) <= 0) {
            close(client_fd);
            return;
        }

        uint64_t last_frame_id = 0;
        while (g_running) {
            std::vector<uchar> jpeg;
            uint64_t frame_id = 0;
            if (g_broadcaster.get_frame(last_frame_id, jpeg, frame_id, 100)) {
                last_frame_id = frame_id;
                std::ostringstream frame_header;
                frame_header << "--frame\r\n"
                             << "Content-Type: image/jpeg\r\n"
                             << "Content-Length: " << jpeg.size() << "\r\n\r\n";
                std::string fh_str = frame_header.str();
                if (send(client_fd, fh_str.c_str(), fh_str.size(), MSG_NOSIGNAL) <= 0) break;
                if (send(client_fd, jpeg.data(), jpeg.size(), MSG_NOSIGNAL) <= 0) break;
                if (send(client_fd, "\r\n", 2, MSG_NOSIGNAL) <= 0) break;
            }
        }
        close(client_fd);
        return;
    }

    // Default: Serve HTML Dashboard
    std::string html = HTML_DASHBOARD;
    std::ostringstream response;
    response << "HTTP/1.1 200 OK\r\n"
             << "Content-Type: text/html; charset=utf-8\r\n"
             << "Content-Length: " << html.size() << "\r\n"
             << "Connection: close\r\n\r\n"
             << html;
    std::string resp_str = response.str();
    send(client_fd, resp_str.c_str(), resp_str.size(), 0);
    close(client_fd);
}

// HTTP Server Thread
void http_server_loop(int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "[HTTP ERROR] Socket creation failed\n";
        return;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "[HTTP ERROR] Bind failed on port " << port << "\n";
        close(server_fd);
        return;
    }

    if (listen(server_fd, 10) < 0) {
        std::cerr << "[HTTP ERROR] Listen failed\n";
        close(server_fd);
        return;
    }

    fcntl(server_fd, F_SETFL, O_NONBLOCK);
    std::cout << "[INFO] Web UI Server active at http://0.0.0.0:" << port << "\n";

    while (g_running) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd >= 0) {
            std::thread(handle_client, client_fd).detach();
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    close(server_fd);
}

int main(int argc, char* argv[]) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Set OpenEB HAL stability environment variables
    setenv("MV_FLAGS_EVT3_ROBUST_DECODER", "1", 1);
    setenv("MV_PSEE_PLUGIN_DATA_TRANSFER_BUFFER_POOL_BYTE_SIZE", "67108864", 1); // 64 MB USB transfer buffer pool

    int port = 8080;
    if (argc > 1) {
        port = std::stoi(argv[1]);
    }

    std::cout << "========================================================\n";
    std::cout << "  Predator — IDS IMX636 Robust Neuromorphic Streamer     \n";
    std::cout << "========================================================\n";

    try {
        std::cout << "[INFO] Opening Metavision Camera...\n";
        Metavision::Camera camera = Metavision::Camera::from_first_available();

        int width = camera.geometry().get_width();
        int height = camera.geometry().get_height();
        std::cout << "[INFO] Camera initialized! Geometry: " << width << " x " << height << "\n";

        // Register runtime error callback for resilient continuous streaming
        camera.add_runtime_error_callback([](const Metavision::CameraException& e) {
            std::cerr << "[WARNING] Camera runtime error caught: " << e.what() << "\n";
        });

        // Lockless Frame Generator from Metavision SDK (25ms accumulation, 30 FPS, Dark Palette)
        Metavision::PeriodicFrameGenerationAlgorithm frame_gen(width, height, 25000, 30.0, Metavision::ColorPalette::Dark);

        std::atomic<uint64_t> event_counter{0};
        std::atomic<uint64_t> total_frames{0};
        std::atomic<uint64_t> current_ev_rate{0};
        std::atomic<double> current_stream_fps{30.0};

        std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, 75};

        // Output callback: triggered strictly when a frame is generated by the SIMD time surface
        frame_gen.set_output_callback([&](Metavision::timestamp ts, cv::Mat& frame) {
            total_frames++;

            // Overlay telemetry HUD
            std::string hud_left = "IDS UE-39B0XCP (IMX636) | 1280x720";
            char hud_right[64];
            snprintf(hud_right, sizeof(hud_right), "Rate: %.2f MEv/s | FPS: %.1f", 
                     current_ev_rate.load() / 1000000.0, current_stream_fps.load());

            cv::putText(frame, hud_left, cv::Point(16, 32), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(220, 220, 220), 2, cv::LINE_AA);
            cv::putText(frame, hud_right, cv::Point(16, 68), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(80, 220, 80), 2, cv::LINE_AA);

            // Compress to JPEG and broadcast
            std::vector<uchar> jpeg_buffer;
            cv::imencode(".jpg", frame, jpeg_buffer, encode_params);
            g_broadcaster.update_frame(jpeg_buffer, current_ev_rate.load(), current_stream_fps.load(), width, height);
        });

        // Fast CD event callback - purely feeds the lockless SIMD reslicer
        camera.cd().add_callback([&](const Metavision::EventCD* begin, const Metavision::EventCD* end) {
            uint64_t count = std::distance(begin, end);
            event_counter += count;
            frame_gen.process_events(begin, end);
        });

        // Launch HTTP Server Thread
        std::thread server_thread(http_server_loop, port);

        camera.start();
        std::cout << "[INFO] Camera streaming active. Serving live visualizer...\n";

        auto last_sec = std::chrono::steady_clock::now();
        uint64_t last_frame_count = 0;

        while (g_running && camera.is_running()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            auto now = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_sec).count();
            last_sec = now;

            uint64_t ev = event_counter.exchange(0);
            current_ev_rate = ev * (1000.0 / elapsed_ms);

            uint64_t frames = total_frames.load();
            current_stream_fps = (frames - last_frame_count) * (1000.0 / elapsed_ms);
            last_frame_count = frames;
        }

        std::cout << "[INFO] Stopping camera...\n";
        camera.stop();

        if (server_thread.joinable()) {
            server_thread.join();
        }

        std::cout << "[INFO] Web visualizer terminated cleanly.\n";
        return 0;

    } catch (const Metavision::CameraException& e) {
        std::cerr << "[ERROR] CameraException: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception: " << e.what() << "\n";
        return 2;
    }
}
