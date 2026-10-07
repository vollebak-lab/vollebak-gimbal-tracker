/**
 * @file bias_sweep.cpp
 * @brief Phase 33.7b: Automated IMX636 Sensor Bias Optimization & Sweep Tool.
 *
 * Systematically sweeps Sony IMX636 analog biases (bias_diff_on, bias_diff_off,
 * bias_fo, bias_refr) against the live running detector pipeline via HTTP endpoints
 * (/set_bias and /pipeline_stats).
 *
 * Evaluates across each operating point:
 *   1. Raw array event throughput (ev/s) under current optical illumination.
 *   2. Retained event rate after ego-motion / spatial clutter filter.
 *   3. Suppression ratio (%).
 *   4. Active 40x40 cuFFT analysis cells.
 *   5. Max micro-sieve periodic lock hits.
 *   6. Downstream CFAR/DSP candidates & confirmed track count (verifying false alarm immunity).
 *   7. Optical sharpness focus score.
 *
 * Automatically restores original baseline sensor biases upon completion or interruption.
 */

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <map>
#include <cstring>
#include <cstdint>
#include <csignal>
#include <algorithm>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>

namespace {

volatile std::sig_atomic_t g_running = 1;

void signal_handler(int) {
    g_running = 0;
}

// Lightweight zero-dependency HTTP client
std::string http_get(const std::string& host, int port, const std::string& path) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return "";

    struct timeval tv;
    tv.tv_sec = 3;
    tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host.c_str(), &server_addr.sin_addr) <= 0) {
        close(sock);
        return "";
    }

    if (connect(sock, reinterpret_cast<sockaddr*>(&server_addr), sizeof(server_addr)) < 0) {
        close(sock);
        return "";
    }

    std::ostringstream req;
    req << "GET " << path << " HTTP/1.1\r\n"
        << "Host: " << host << ":" << port << "\r\n"
        << "Connection: close\r\n\r\n";
    std::string req_str = req.str();

    if (send(sock, req_str.data(), req_str.size(), 0) <= 0) {
        close(sock);
        return "";
    }

    std::string response;
    char buf[4096];
    ssize_t bytes = 0;
    while ((bytes = recv(sock, buf, sizeof(buf), 0)) > 0) {
        response.append(buf, static_cast<size_t>(bytes));
    }
    close(sock);

    size_t header_end = response.find("\r\n\r\n");
    if (header_end != std::string::npos) {
        return response.substr(header_end + 4);
    }
    return response;
}

// Minimal robust JSON field extraction helpers
double extract_json_double(const std::string& json, const std::string& key, double default_val = 0.0) {
    size_t pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return default_val;
    size_t colon = json.find(':', pos);
    if (colon == std::string::npos) return default_val;
    size_t start = json.find_first_not_of(" \t\r\n", colon + 1);
    if (start == std::string::npos) return default_val;
    try {
        return std::stod(json.substr(start));
    } catch (...) {
        return default_val;
    }
}

int extract_json_int(const std::string& json, const std::string& key, int default_val = 0) {
    size_t pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return default_val;
    size_t colon = json.find(':', pos);
    if (colon == std::string::npos) return default_val;
    size_t start = json.find_first_not_of(" \t\r\n", colon + 1);
    if (start == std::string::npos) return default_val;
    try {
        return std::stoi(json.substr(start));
    } catch (...) {
        return default_val;
    }
}

struct PipelineMetrics {
    uint64_t raw_events = 0;
    uint64_t retained_events = 0;
    double suppressed_pct = 0.0;
    int active_cells = 0;
    int max_sieve_hits = 0;
    int num_targets = 0;
    int num_tracks = 0;
    double focus_score = 0.0;
    uint64_t timestamp_ms = 0;
};

PipelineMetrics parse_metrics(const std::string& json) {
    PipelineMetrics m;
    m.timestamp_ms = static_cast<uint64_t>(extract_json_double(json, "timestamp_ms", 0.0));
    m.num_targets = extract_json_int(json, "num_targets", 0);
    m.num_tracks = extract_json_int(json, "num_tracks", 0);

    // Look inside ego_motion
    size_t em_pos = json.find("\"ego_motion\"");
    if (em_pos != std::string::npos) {
        std::string em_sub = json.substr(em_pos);
        m.active_cells = extract_json_int(em_sub, "active_cells", 0);
        m.suppressed_pct = extract_json_double(em_sub, "suppressed_events_pct", 0.0);
        m.raw_events = static_cast<uint64_t>(extract_json_double(em_sub, "total_raw_events", 0.0));
        m.retained_events = static_cast<uint64_t>(extract_json_double(em_sub, "retained_imo_events", 0.0));
    }

    // Look inside roi_diagnostics
    size_t roi_pos = json.find("\"roi_diagnostics\"");
    if (roi_pos != std::string::npos) {
        std::string roi_sub = json.substr(roi_pos);
        m.max_sieve_hits = extract_json_int(roi_sub, "max_sieve_hits", 0);
    }

    // Look inside focus
    size_t focus_pos = json.find("\"focus\"");
    if (focus_pos != std::string::npos) {
        std::string focus_sub = json.substr(focus_pos);
        m.focus_score = extract_json_double(focus_sub, "score", 0.0);
    }

    return m;
}

struct SweepConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    int settle_ms = 600;
    int sample_ms = 2000;
    std::string output_csv = "bias_sweep_results.csv";

    // Sweep dimensions
    std::vector<int> diff_on_vals = {4, 6, 8, 10};
    std::vector<int> diff_off_vals = {4, 6, 8, 10};
    std::vector<int> fo_vals = {-8};
    std::vector<int> refr_vals = {20};
};

struct SweepResultRow {
    int diff_on;
    int diff_off;
    int fo;
    int refr;
    double raw_ev_rate;
    double retained_ev_rate;
    double supp_pct;
    double avg_active_cells;
    int peak_sieve_hits;
    int max_tracks;
    double avg_focus;
};

} // anonymous namespace

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    SweepConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--host" && i + 1 < argc) {
            cfg.host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            cfg.port = std::stoi(argv[++i]);
        } else if (arg == "--sample-ms" && i + 1 < argc) {
            cfg.sample_ms = std::stoi(argv[++i]);
        } else if (arg == "--settle-ms" && i + 1 < argc) {
            cfg.settle_ms = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            cfg.output_csv = argv[++i];
        } else if (arg == "--sweep-fo") {
            cfg.diff_on_vals = {6};
            cfg.diff_off_vals = {6};
            cfg.fo_vals = {-16, -12, -8, -4, 0, 4};
        } else if (arg == "--sweep-refr") {
            cfg.diff_on_vals = {6};
            cfg.diff_off_vals = {6};
            cfg.fo_vals = {-8};
            cfg.refr_vals = {10, 15, 20, 30, 40};
        } else if (arg == "--diff-on" && i + 1 < argc) {
            cfg.diff_on_vals = {std::stoi(argv[++i])};
        } else if (arg == "--diff-off" && i + 1 < argc) {
            cfg.diff_off_vals = {std::stoi(argv[++i])};
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: bias_sweep [options]\n"
                      << "  --host <ip>          Detector host (default: 127.0.0.1)\n"
                      << "  --port <port>        Detector port (default: 8080)\n"
                      << "  --sample-ms <ms>     Measurement window per point (default: 2000)\n"
                      << "  --settle-ms <ms>     Settlement delay after bias change (default: 600)\n"
                      << "  --output <file.csv>  CSV output filename (default: bias_sweep_results.csv)\n"
                      << "  --sweep-fo           Sweep fo from -16 to +4 at nominal diff thresholds\n"
                      << "  --sweep-refr         Sweep refr from 10 to 40 at nominal diff thresholds\n"
                      << "  --diff-on <val>      Pin diff_on to single value\n"
                      << "  --diff-off <val>     Pin diff_off to single value\n";
            return 0;
        }
    }

    std::cout << "========================================================================\n"
              << "  Predator — Phase 33.7b IMX636 Sensor Bias Optimization Sweep Engine  \n"
              << "========================================================================\n"
              << "[INFO] Target Host: " << cfg.host << ":" << cfg.port << "\n"
              << "[INFO] Settle Time: " << cfg.settle_ms << " ms | Sample Window: " << cfg.sample_ms << " ms\n"
              << "[INFO] Output CSV: " << cfg.output_csv << "\n";

    // 1. Verify connection and capture initial baseline biases for automatic restore
    std::string initial_stats = http_get(cfg.host, cfg.port, "/pipeline_stats");
    if (initial_stats.empty()) {
        std::cerr << "[ERROR] Could not connect to predator detector at " << cfg.host << ":" << cfg.port << "!\n"
                  << "Ensure predator-camera.service is running.\n";
        return 1;
    }

    int base_diff_on = extract_json_int(initial_stats, "bias_diff_on", 6);
    int base_diff_off = extract_json_int(initial_stats, "bias_diff_off", 6);
    int base_fo = extract_json_int(initial_stats, "bias_fo", -8);
    int base_refr = extract_json_int(initial_stats, "bias_refr", 20);

    std::cout << "[INFO] Initial Baseline Biases Locked: diff_on=" << base_diff_on
              << ", diff_off=" << base_diff_off << ", fo=" << base_fo << ", refr=" << base_refr << "\n\n";

    std::vector<SweepResultRow> results;

    // Header print
    std::cout << std::left
              << std::setw(9)  << "diff_on"
              << std::setw(10) << "diff_off"
              << std::setw(6)  << "fo"
              << std::setw(6)  << "refr"
              << std::setw(13) << "Raw (ev/s)"
              << std::setw(15) << "Retained(ev/s)"
              << std::setw(10) << "Supp %"
              << std::setw(10) << "ActCells"
              << std::setw(8)  << "Sieve"
              << std::setw(8)  << "Tracks"
              << std::setw(10) << "Focus"
              << "\n";
    std::cout << std::string(95, '-') << "\n";

    for (int don : cfg.diff_on_vals) {
        for (int doff : cfg.diff_off_vals) {
            for (int fo : cfg.fo_vals) {
                for (int refr : cfg.refr_vals) {
                    if (!g_running) break;

                    // 1. Set biases via HTTP
                    std::ostringstream path;
                    path << "/set_bias?diff_on=" << don << "&diff_off=" << doff << "&fo=" << fo << "&refr=" << refr;
                    http_get(cfg.host, cfg.port, path.str());

                    // 2. Settlement sleep
                    std::this_thread::sleep_for(std::chrono::milliseconds(cfg.settle_ms));

                    // 3. Sample over measurement window
                    auto start_tp = std::chrono::steady_clock::now();
                    std::string m0_json = http_get(cfg.host, cfg.port, "/pipeline_stats");
                    PipelineMetrics m0 = parse_metrics(m0_json);

                    double sum_active_cells = 0.0;
                    double sum_supp = 0.0;
                    double sum_focus = 0.0;
                    int max_sieve = 0;
                    int max_tracks = 0;
                    int sample_count = 0;

                    while (g_running) {
                        auto cur_tp = std::chrono::steady_clock::now();
                        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cur_tp - start_tp).count();
                        if (elapsed_ms >= cfg.sample_ms) break;

                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        std::string cur_json = http_get(cfg.host, cfg.port, "/pipeline_stats");
                        PipelineMetrics cur_m = parse_metrics(cur_json);

                        sum_active_cells += cur_m.active_cells;
                        sum_supp += cur_m.suppressed_pct;
                        sum_focus += cur_m.focus_score;
                        max_sieve = std::max(max_sieve, cur_m.max_sieve_hits);
                        max_tracks = std::max(max_tracks, cur_m.num_tracks);
                        sample_count++;
                    }

                    std::string m1_json = http_get(cfg.host, cfg.port, "/pipeline_stats");
                    PipelineMetrics m1 = parse_metrics(m1_json);

                    double elapsed_s = (m1.timestamp_ms > m0.timestamp_ms) 
                        ? (m1.timestamp_ms - m0.timestamp_ms) / 1000.0 
                        : (cfg.sample_ms / 1000.0);

                    double raw_rate = (m1.raw_events >= m0.raw_events && elapsed_s > 0.01) 
                        ? (m1.raw_events - m0.raw_events) / elapsed_s 
                        : 0.0;
                    double retained_rate = (m1.retained_events >= m0.retained_events && elapsed_s > 0.01) 
                        ? (m1.retained_events - m0.retained_events) / elapsed_s 
                        : 0.0;

                    SweepResultRow row;
                    row.diff_on = don;
                    row.diff_off = doff;
                    row.fo = fo;
                    row.refr = refr;
                    row.raw_ev_rate = raw_rate;
                    row.retained_ev_rate = retained_rate;
                    row.supp_pct = (sample_count > 0) ? (sum_supp / sample_count) : 0.0;
                    row.avg_active_cells = (sample_count > 0) ? (sum_active_cells / sample_count) : 0.0;
                    row.peak_sieve_hits = max_sieve;
                    row.max_tracks = max_tracks;
                    row.avg_focus = (sample_count > 0) ? (sum_focus / sample_count) : 0.0;

                    results.push_back(row);

                    std::cout << std::left
                              << std::setw(9)  << row.diff_on
                              << std::setw(10) << row.diff_off
                              << std::setw(6)  << row.fo
                              << std::setw(6)  << row.refr
                              << std::setw(13) << std::fixed << std::setprecision(1) << row.raw_ev_rate
                              << std::setw(15) << std::fixed << std::setprecision(1) << row.retained_ev_rate
                              << std::setw(10) << std::fixed << std::setprecision(1) << row.supp_pct
                              << std::setw(10) << std::fixed << std::setprecision(1) << row.avg_active_cells
                              << std::setw(8)  << row.peak_sieve_hits
                              << std::setw(8)  << row.max_tracks
                              << std::setw(10) << std::fixed << std::setprecision(1) << row.avg_focus
                              << "\n" << std::flush;
                }
            }
        }
    }

    // 4. Always restore baseline biases
    std::cout << "\n[INFO] Restoring baseline sensor biases ("
              << "diff_on=" << base_diff_on << ", diff_off=" << base_diff_off
              << ", fo=" << base_fo << ", refr=" << base_refr << ")...\n";
    std::ostringstream restore_path;
    restore_path << "/set_bias?diff_on=" << base_diff_on << "&diff_off=" << base_diff_off
                 << "&fo=" << base_fo << "&refr=" << base_refr;
    http_get(cfg.host, cfg.port, restore_path.str());

    // 5. Write CSV file
    std::ofstream csv(cfg.output_csv);
    if (csv.is_open()) {
        csv << "diff_on,diff_off,fo,refr,raw_ev_rate,retained_ev_rate,supp_pct,avg_active_cells,peak_sieve_hits,max_tracks,avg_focus\n";
        for (const auto& r : results) {
            csv << r.diff_on << "," << r.diff_off << "," << r.fo << "," << r.refr << ","
                << std::fixed << std::setprecision(2)
                << r.raw_ev_rate << "," << r.retained_ev_rate << "," << r.supp_pct << ","
                << r.avg_active_cells << "," << r.peak_sieve_hits << "," << r.max_tracks << ","
                << r.avg_focus << "\n";
        }
        std::cout << "[INFO] Successfully wrote " << results.size() << " sweep data points to " << cfg.output_csv << "\n";
    }

    std::cout << "[INFO] Bias sweep complete.\n";
    return 0;
}
