#ifndef PREDATOR_EGO_MOTION_HPP
#define PREDATOR_EGO_MOTION_HPP

#include <vector>
#include <cmath>
#include <array>
#include <deque>
#include <mutex>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#ifdef __linux__
#include <linux/serial.h>
#include <sys/ioctl.h>
#endif

#include "flicker_dsp.hpp"

namespace predator {

/**
 * @brief High-rate 6-DoF IMU sample
 */
struct ImuSample {
    uint64_t timestamp_us{0};      ///< Microsecond timestamp
    Vector3d gyro_rad_s{0, 0, 0};  ///< Angular velocity [wx, wy, wz] in rad/s
    Vector3d accel_m_s2{0, 0, 0};  ///< Linear acceleration [ax, ay, az] in m/s^2
};

/**
 * @brief Stabilized 2D Event representation
 */
struct StabilizedEvent {
    int orig_x{0};
    int orig_y{0};
    double warped_x{0.0};
    double warped_y{0.0};
    uint64_t timestamp_us{0};
    short polarity{0};
    bool is_valid{true};
};

/**
 * @brief High-Rate Continuous Gyroscope Event Warper & Spherical Homography Stabilizer
 * 
 * Implements microsecond point-by-point coordinate unwarping using the camera rotation matrix R(t_ref, t_i):
 *   x'_hom = K * R(t_ref, t_i) * K^-1 * x_hom
 */
class ContinuousGyroWarper {
public:
    explicit ContinuousGyroWarper(const LensParameters& lens = LensParameters(), size_t max_imu_history = 5000)
        : lens_(lens), max_imu_history_(max_imu_history) {
        update_intrinsic_matrices();
    }

    void set_lens(const LensParameters& lens) {
        lens_ = lens;
        update_intrinsic_matrices();
    }

    const LensParameters& lens() const { return lens_; }

    /**
     * @brief Anchor camera microsecond clock with host clock
     */
    void set_camera_time_anchor(uint64_t cam_t0_us, uint64_t host_t0_us) {
        int64_t offset = static_cast<int64_t>(cam_t0_us) - static_cast<int64_t>(host_t0_us);
        clock_offset_us_.store(offset, std::memory_order_release);
        time_anchored_.store(true, std::memory_order_release);
    }

    /**
     * @brief Continuously tracks clock drift between camera microsecond clock and host clock
     */
    void update_camera_time_anchor(uint64_t cam_t_us, uint64_t host_t_us) {
        int64_t measured_offset = static_cast<int64_t>(cam_t_us) - static_cast<int64_t>(host_t_us);
        if (!time_anchored_.load(std::memory_order_acquire)) {
            clock_offset_us_.store(measured_offset, std::memory_order_release);
            time_anchored_.store(true, std::memory_order_release);
            return;
        }
        int64_t current_offset = clock_offset_us_.load(std::memory_order_relaxed);
        // Exponential Moving Average filter for clock offset (alpha = 0.05) with ZERO lag on time advancement
        int64_t filtered_offset = static_cast<int64_t>(current_offset * 0.95 + measured_offset * 0.05);
        clock_offset_us_.store(filtered_offset, std::memory_order_release);
    }

    bool is_time_anchored() const {
        return time_anchored_.load(std::memory_order_acquire);
    }

    uint64_t host_to_camera_time(uint64_t host_t_us) const {
        if (!time_anchored_.load(std::memory_order_acquire)) {
            return host_t_us;
        }
        int64_t offset = clock_offset_us_.load(std::memory_order_relaxed);
        int64_t cam_t = static_cast<int64_t>(host_t_us) + offset;
        return (cam_t > 0) ? static_cast<uint64_t>(cam_t) : 0;
    }

    /**
     * @brief Ingests an IMU sample into the circular buffer
     */
    void ingest_imu(const ImuSample& sample) {
        // Physical sanity gate: reject corrupted/glitched samples
        if (!std::isfinite(sample.gyro_rad_s.x) || !std::isfinite(sample.gyro_rad_s.y) || !std::isfinite(sample.gyro_rad_s.z) ||
            std::abs(sample.gyro_rad_s.x) > 35.0 || std::abs(sample.gyro_rad_s.y) > 35.0 || std::abs(sample.gyro_rad_s.z) > 35.0) {
            return;
        }

        double raw_wx = sample.gyro_rad_s.x;
        double raw_wy = sample.gyro_rad_s.y;
        double raw_wz = sample.gyro_rad_s.z;
        double raw_norm = std::sqrt(raw_wx * raw_wx + raw_wy * raw_wy + raw_wz * raw_wz);

        // Zero-velocity bias calibration: if raw angular speed < 0.12 rad/s (~6.8 deg/s), camera is resting
        if (raw_norm < 0.12) {
            double b_x = bias_wx_.load(std::memory_order_relaxed);
            double b_y = bias_wy_.load(std::memory_order_relaxed);
            double b_z = bias_wz_.load(std::memory_order_relaxed);
            bias_wx_.store(b_x * 0.995 + raw_wx * 0.005, std::memory_order_relaxed);
            bias_wy_.store(b_y * 0.995 + raw_wy * 0.005, std::memory_order_relaxed);
            bias_wz_.store(b_z * 0.995 + raw_wz * 0.005, std::memory_order_relaxed);
        }

        double corr_wx = raw_wx - bias_wx_.load(std::memory_order_relaxed);
        double corr_wy = raw_wy - bias_wy_.load(std::memory_order_relaxed);
        double corr_wz = raw_wz - bias_wz_.load(std::memory_order_relaxed);

        // Gyro deadband: if residual rotation < 0.02 rad/s (~1.1 deg/s), snap to 0 to prevent phantom smearing
        if (std::abs(corr_wx) < 0.02) corr_wx = 0.0;
        if (std::abs(corr_wy) < 0.02) corr_wy = 0.0;
        if (std::abs(corr_wz) < 0.02) corr_wz = 0.0;

        latest_wx_.store(corr_wx, std::memory_order_relaxed);
        latest_wy_.store(corr_wy, std::memory_order_relaxed);
        latest_wz_.store(corr_wz, std::memory_order_relaxed);
        latest_ax_.store(sample.accel_m_s2.x, std::memory_order_relaxed);
        latest_ay_.store(sample.accel_m_s2.y, std::memory_order_relaxed);
        latest_az_.store(sample.accel_m_s2.z, std::memory_order_relaxed);

        ImuSample corrected_sample = sample;
        corrected_sample.gyro_rad_s = Vector3d(corr_wx, corr_wy, corr_wz);

        std::lock_guard<std::mutex> lock(imu_mutex_);
        if (!imu_buffer_.empty() && corrected_sample.timestamp_us <= imu_buffer_.back().timestamp_us) {
            return; // Ignore non-monotonic samples
        }
        imu_buffer_.push_back(corrected_sample);
        while (imu_buffer_.size() > max_imu_history_) {
            imu_buffer_.pop_front();
        }
    }

    /**
     * @brief Ingests raw IMU readings with explicit microsecond timestamp
     */
    void ingest_imu_raw(uint64_t timestamp_us, double wx, double wy, double wz, double ax = 0.0, double ay = 0.0, double az = 0.0) {
        ImuSample s;
        s.timestamp_us = timestamp_us;
        s.gyro_rad_s = Vector3d(wx, wy, wz);
        s.accel_m_s2 = Vector3d(ax, ay, az);
        ingest_imu(s);
    }

    /**
     * @brief Instantly returns latest raw angular velocity without searching buffer
     */
    Vector3d get_latest_angular_velocity() const {
        return Vector3d(latest_wx_.load(std::memory_order_relaxed),
                        latest_wy_.load(std::memory_order_relaxed),
                        latest_wz_.load(std::memory_order_relaxed));
    }

    /**
     * @brief Ultra-fast direct homography unwarping for batch event processing (zero mutex / matrix ops)
     */
    static inline bool apply_homography_fast(const Matrix3x3& H, int x, int y, int max_w, int max_h, double& wx, double& wy) {
        double pz = H.at(2, 0) * x + H.at(2, 1) * y + H.at(2, 2);
        if (std::abs(pz) < 1e-6) {
            wx = static_cast<double>(x);
            wy = static_cast<double>(y);
            return false;
        }
        double inv_z = 1.0 / pz;
        wx = (H.at(0, 0) * x + H.at(0, 1) * y + H.at(0, 2)) * inv_z;
        wy = (H.at(1, 0) * x + H.at(1, 1) * y + H.at(1, 2)) * inv_z;
        return (wx >= 0.0 && wx < max_w && wy >= 0.0 && wy < max_h);
    }

    /**
     * @brief Interpolates angular velocity at timestamp t_us
     */
    bool get_angular_velocity(uint64_t t_us, Vector3d& out_omega) const {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        if (imu_buffer_.empty()) {
            out_omega = get_latest_angular_velocity();
            return false;
        }

        if (t_us <= imu_buffer_.front().timestamp_us) {
            out_omega = imu_buffer_.front().gyro_rad_s;
            return true;
        }
        if (t_us >= imu_buffer_.back().timestamp_us) {
            out_omega = imu_buffer_.back().gyro_rad_s;
            return true;
        }

        // Binary search for interval
        auto it = std::lower_bound(imu_buffer_.begin(), imu_buffer_.end(), t_us,
            [](const ImuSample& a, uint64_t t) { return a.timestamp_us < t; });

        if (it == imu_buffer_.begin()) {
            out_omega = it->gyro_rad_s;
            return true;
        }

        auto prev = it - 1;
        double dt = static_cast<double>(it->timestamp_us - prev->timestamp_us);
        if (dt <= 0.0) {
            out_omega = prev->gyro_rad_s;
            return true;
        }

        double alpha = static_cast<double>(t_us - prev->timestamp_us) / dt;
        out_omega = prev->gyro_rad_s * (1.0 - alpha) + it->gyro_rad_s * alpha;
        return true;
    }

    /**
     * @brief Computes 3D rotation matrix R(t_ref, t_target) by integrating gyro angular velocity
     * Uses Rodrigues' formula on integrated rotation vector theta = \int_t_ref^t_target omega(tau) dtau
     */
    Matrix3x3 compute_rotation_matrix(uint64_t t_ref_us, uint64_t t_target_us) const {
        if (t_ref_us == t_target_us) {
            return Matrix3x3::identity();
        }

        Vector3d theta(0, 0, 0);
        bool reverse = (t_target_us < t_ref_us);
        uint64_t t_start = reverse ? t_target_us : t_ref_us;
        uint64_t t_end   = reverse ? t_ref_us : t_target_us;

        std::lock_guard<std::mutex> lock(imu_mutex_);
        if (imu_buffer_.empty()) {
            return Matrix3x3::identity();
        }

        // Find relevant IMU span
        auto it_start = std::lower_bound(imu_buffer_.begin(), imu_buffer_.end(), t_start,
            [](const ImuSample& a, uint64_t t) { return a.timestamp_us < t; });
        if (it_start != imu_buffer_.begin() && (it_start == imu_buffer_.end() || it_start->timestamp_us > t_start)) {
            --it_start;
        }

        uint64_t prev_t = t_start;
        Vector3d prev_w;
        {
            // Sample at t_start
            if (it_start == imu_buffer_.end()) {
                prev_w = imu_buffer_.back().gyro_rad_s;
            } else {
                prev_w = it_start->gyro_rad_s;
            }
        }

        for (auto it = it_start; it != imu_buffer_.end(); ++it) {
            if (it->timestamp_us <= t_start) {
                prev_t = t_start;
                prev_w = it->gyro_rad_s;
                continue;
            }
            if (it->timestamp_us >= t_end) {
                double dt_s = static_cast<double>(t_end - prev_t) * 1e-6;
                Vector3d curr_w = it->gyro_rad_s;
                theta = theta + (prev_w + curr_w) * 0.5 * dt_s;
                prev_t = t_end;
                break;
            }

            double dt_s = static_cast<double>(it->timestamp_us - prev_t) * 1e-6;
            theta = theta + (prev_w + it->gyro_rad_s) * 0.5 * dt_s;
            prev_t = it->timestamp_us;
            prev_w = it->gyro_rad_s;
        }

        if (prev_t < t_end) {
            double dt_s = static_cast<double>(t_end - prev_t) * 1e-6;
            theta = theta + prev_w * dt_s;
        }

        if (reverse) {
            theta = theta * -1.0;
        }

        return rodrigues(theta);
    }

    /**
     * @brief Computes Rodrigues rotation matrix from rotation vector theta
     */
    static Matrix3x3 rodrigues(const Vector3d& theta) {
        double angle = theta.norm();
        if (angle < 1e-9) {
            return Matrix3x3::identity();
        }

        Vector3d k = theta * (1.0 / angle);
        double c = std::cos(angle);
        double s = std::sin(angle);
        double v = 1.0 - c;

        Matrix3x3 R;
        R.at(0, 0) = k.x * k.x * v + c;
        R.at(0, 1) = k.x * k.y * v - k.z * s;
        R.at(0, 2) = k.x * k.z * v + k.y * s;

        R.at(1, 0) = k.x * k.y * v + k.z * s;
        R.at(1, 1) = k.y * k.y * v + c;
        R.at(1, 2) = k.y * k.z * v - k.x * s;

        R.at(2, 0) = k.x * k.z * v - k.y * s;
        R.at(2, 1) = k.y * k.z * v + k.x * s;
        R.at(2, 2) = k.z * k.z * v + c;

        return R;
    }

    /**
     * @brief Computes 3x3 spherical unwarping homography mapping current camera coordinates at t_target to stabilized world anchor at t_ref:
     *   H_cam_to_world = K * R(t_ref, t_target) * K^-1
     */
    Matrix3x3 compute_homography(uint64_t t_ref_us, uint64_t t_target_us) const {
        Matrix3x3 R = compute_rotation_matrix(t_ref_us, t_target_us);
        return K_ * R * K_inv_;
    }

    /**
     * @brief Computes forward projection homography mapping stabilized world anchor at t_ref to current camera view at t_cam:
     *   H_world_to_cam = K * R(t_ref, t_cam)^-1 * K^-1
     */
    Matrix3x3 compute_forward_homography(uint64_t t_ref_us, uint64_t t_cam_us) const {
        Matrix3x3 R = compute_rotation_matrix(t_ref_us, t_cam_us);
        return K_ * R.inverse() * K_inv_;
    }

    /**
     * @brief Point-wise unwarps an event from timestamp t_i to reference time t_ref
     * 
     * @param x Original pixel X
     * @param y Original pixel Y
     * @param t_us Event timestamp in microseconds
     * @param t_ref_us Reference timestamp in microseconds
     * @param[out] warped_x Stabilized pixel X
     * @param[out] warped_y Stabilized pixel Y
     * @return true if warped pixel is within sensor FOV
     */
    bool unwarp_event(int x, int y, uint64_t t_us, uint64_t t_ref_us, double& warped_x, double& warped_y) const {
        if (t_us == t_ref_us) {
            warped_x = static_cast<double>(x);
            warped_y = static_cast<double>(y);
            return true;
        }

        Matrix3x3 H = compute_homography(t_ref_us, t_us);
        Vector3d p_hom(static_cast<double>(x), static_cast<double>(y), 1.0);
        Vector3d p_warped = H * p_hom;

        if (std::abs(p_warped.z) < 1e-6) {
            warped_x = static_cast<double>(x);
            warped_y = static_cast<double>(y);
            return false;
        }

        warped_x = p_warped.x / p_warped.z;
        warped_y = p_warped.y / p_warped.z;

        return (warped_x >= 0.0 && warped_x < lens_.sensor_width &&
                warped_y >= 0.0 && warped_y < lens_.sensor_height);
    }

    /**
     * @brief Computes instantaneous expected background velocity at pixel (x, y) due to rotation
     */
    Vector3d compute_background_flow(double x, double y, const Vector3d& omega_rad_s) const {
        double fx = lens_.fx_pix();
        double fy = lens_.fy_pix();
        double cx = lens_.cx_pix();
        double cy = lens_.cy_pix();

        double dx = x - cx;
        double dy = cy - y; // positive Y up in coordinate standard

        // Optical flow equation for pure rotational camera motion
        double vx = -fx * omega_rad_s.y + (dx * dy / fy) * omega_rad_s.x + dy * omega_rad_s.z;
        double vy =  fy * omega_rad_s.x - (dx * dy / fx) * omega_rad_s.y - dx * omega_rad_s.z;

        return {vx, vy, 0.0};
    }

    /**
     * @brief Batch unwarps a stream of events into stabilized coordinates
     */
    template<typename EventIterator>
    std::vector<StabilizedEvent> unwarp_event_stream(EventIterator begin, EventIterator end, uint64_t t_ref_us) const {
        std::vector<StabilizedEvent> out;
        out.reserve(std::distance(begin, end));

        for (auto it = begin; it != end; ++it) {
            StabilizedEvent se;
            se.orig_x = it->x;
            se.orig_y = it->y;
            se.timestamp_us = it->t;
            se.polarity = it->p;
            se.is_valid = unwarp_event(it->x, it->y, it->t, t_ref_us, se.warped_x, se.warped_y);
            out.push_back(se);
        }

        return out;
    }

private:
    void update_intrinsic_matrices() {
        double fx = lens_.fx_pix();
        double fy = lens_.fy_pix();
        double cx = lens_.cx_pix();
        double cy = lens_.cy_pix();

        K_ = Matrix3x3::identity();
        K_.at(0, 0) = fx;
        K_.at(0, 2) = cx;
        K_.at(1, 1) = fy;
        K_.at(1, 2) = cy;

        K_inv_ = K_.inverse();
    }

    LensParameters lens_;
    size_t max_imu_history_;
    Matrix3x3 K_;
    Matrix3x3 K_inv_;

    std::atomic<bool> time_anchored_{false};
    std::atomic<int64_t> clock_offset_us_{0};

    std::atomic<double> latest_wx_{0.0};
    std::atomic<double> latest_wy_{0.0};
    std::atomic<double> latest_wz_{0.0};
    std::atomic<double> latest_ax_{0.0};
    std::atomic<double> latest_ay_{0.0};
    std::atomic<double> latest_az_{0.0};

    std::atomic<double> bias_wx_{0.0};
    std::atomic<double> bias_wy_{0.0};
    std::atomic<double> bias_wz_{0.0};

    mutable std::mutex imu_mutex_;
    std::deque<ImuSample> imu_buffer_;
};

/**
 * @brief High-Rate USB CDC Serial Reader for Arduino Nicla Sense ME
 * Reads 200 Hz binary IMU packets from /dev/ttyACM0 and automatically feeds ContinuousGyroWarper
 */
class NiclaSerialReader {
public:
    explicit NiclaSerialReader(ContinuousGyroWarper& warper, std::string port = "/dev/ttyACM0", int baud = 115200)
        : warper_(warper), port_(std::move(port)), baud_(baud) {}

    ~NiclaSerialReader() {
        stop();
    }

    bool start() {
        if (running_.load()) return true;
        running_.store(true);
        thread_ = std::thread(&NiclaSerialReader::read_loop, this);
        return true;
    }

    void stop() {
        if (!running_.load()) return;
        running_.store(false);
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    bool is_connected() const { return connected_.load(); }
    uint64_t packet_count() const { return packets_received_.load(); }

private:
    void read_loop() {
        while (running_.load()) {
            int fd = open(port_.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
            if (fd < 0) {
                connected_.store(false);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }

            struct termios tty{};
            if (tcgetattr(fd, &tty) != 0) {
                close(fd);
                connected_.store(false);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }

            cfsetospeed(&tty, B115200);
            cfsetispeed(&tty, B115200);

            tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
            tty.c_iflag &= ~IGNBRK;
            tty.c_lflag = 0;
            tty.c_oflag = 0;
            tty.c_cc[VMIN]  = 1;
            tty.c_cc[VTIME] = 1;

            tty.c_iflag &= ~(IXON | IXOFF | IXANY);
            tty.c_cflag |= (CLOCAL | CREAD);
            tty.c_cflag &= ~(PARENB | PARODD);
            tty.c_cflag &= ~CSTOPB;
            tty.c_cflag &= ~CRTSCTS;

            if (tcsetattr(fd, TCSANOW, &tty) != 0) {
                close(fd);
                connected_.store(false);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }

#ifdef __linux__
            struct serial_struct ser_info;
            if (ioctl(fd, TIOCGSERIAL, &ser_info) == 0) {
                ser_info.flags |= ASYNC_LOW_LATENCY;
                ioctl(fd, TIOCSSERIAL, &ser_info);
            }
#endif

            connected_.store(true);
            std::cout << "[INFO] Nicla Sense ME IMU connected on " << port_ << " (Low Latency Mode Active)\n";

            uint64_t sync_offset_us = 0;
            bool first_sync = true;

            std::vector<uint8_t> rx_buf;
            rx_buf.reserve(1024);
            uint8_t chunk[256];

            auto compute_chk = [](const uint8_t* data, size_t length) -> uint16_t {
                uint16_t chk = 0;
                for (size_t i = 0; i < length; i += 2) {
                    if (i + 1 < length) {
                        chk ^= static_cast<uint16_t>(data[i]) | (static_cast<uint16_t>(data[i + 1]) << 8);
                    } else {
                        chk ^= static_cast<uint16_t>(data[i]);
                    }
                }
                return chk;
            };

            while (running_.load()) {
                ssize_t n = read(fd, chunk, sizeof(chunk));
                if (n <= 0) {
                    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                        std::this_thread::sleep_for(std::chrono::microseconds(500));
                        continue;
                    }
                    break; // Error or disconnect
                }

                rx_buf.insert(rx_buf.end(), chunk, chunk + n);

                while (rx_buf.size() >= 32) {
                    // Search for 0xAA 0x55 preamble
                    size_t sync_pos = 0;
                    bool found = false;
                    for (size_t i = 0; i + 1 < rx_buf.size(); ++i) {
                        if (rx_buf[i] == 0xAA && rx_buf[i + 1] == 0x55) {
                            sync_pos = i;
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        uint8_t last = rx_buf.back();
                        rx_buf.clear();
                        if (last == 0xAA) rx_buf.push_back(last);
                        break;
                    }

                    if (sync_pos > 0) {
                        rx_buf.erase(rx_buf.begin(), rx_buf.begin() + sync_pos);
                    }

                    if (rx_buf.size() < 32) {
                        break; // Need more bytes
                    }

                    // Verify XOR Checksum over 28 bytes of payload (indices 2..29)
                    uint16_t expected_chk = compute_chk(rx_buf.data() + 2, 28);
                    uint16_t packet_chk = 0;
                    std::memcpy(&packet_chk, rx_buf.data() + 30, 2);

                    if (expected_chk != packet_chk) {
                        // Corrupted packet or false preamble: slide by 1 byte
                        rx_buf.erase(rx_buf.begin(), rx_buf.begin() + 1);
                        continue;
                    }

                    // Valid verified packet! Unpack
                    uint32_t t_nicla_us = 0;
                    float wx = 0, wy = 0, wz = 0;
                    float ax = 0, ay = 0, az = 0;

                    std::memcpy(&t_nicla_us, rx_buf.data() + 2, 4);
                    std::memcpy(&wx, rx_buf.data() + 6, 4);
                    std::memcpy(&wy, rx_buf.data() + 10, 4);
                    std::memcpy(&wz, rx_buf.data() + 14, 4);
                    std::memcpy(&ax, rx_buf.data() + 18, 4);
                    std::memcpy(&ay, rx_buf.data() + 22, 4);
                    std::memcpy(&az, rx_buf.data() + 26, 4);

                    rx_buf.erase(rx_buf.begin(), rx_buf.begin() + 32);

                    if (!std::isfinite(wx) || !std::isfinite(wy) || !std::isfinite(wz) ||
                        !std::isfinite(ax) || !std::isfinite(ay) || !std::isfinite(az)) {
                        continue;
                    }

                    // Physical sanity check (BHI260AP range is +/- 2000 dps = +/- 34.9 rad/s)
                    if (std::abs(wx) > 35.0f || std::abs(wy) > 35.0f || std::abs(wz) > 35.0f ||
                        std::abs(ax) > 100.0f || std::abs(ay) > 100.0f || std::abs(az) > 100.0f) {
                        continue;
                    }

                    // Physical camera optical axis alignment from Arduino Nicla Sense ME:
                    //   nicla_predator_imu.ino already mapped:
                    //     g_packet.gyro_x = nicla_wy (Camera Pitch rate in rad/s)
                    //     g_packet.gyro_y = nicla_wx (Camera Yaw rate in rad/s)
                    //     g_packet.gyro_z = -nicla_wz (Camera Roll rate in rad/s)
                    // Packet payload is already in SI units of rad/s and m/s^2.
                    float cam_wx = wx; // Pitch rate around X-axis (rad/s)
                    float cam_wy = wy; // Yaw rate around Y-axis (rad/s)
                    float cam_wz = wz; // Roll rate around Z-axis (rad/s)

                    float cam_ax = ax;
                    float cam_ay = ay;
                    float cam_az = az;

                    uint64_t host_now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();

                    if (first_sync) {
                        sync_offset_us = host_now_us - static_cast<uint64_t>(t_nicla_us);
                        first_sync = false;
                    } else {
                        // Gentle drift tracking (0.1% EMA)
                        int64_t current_drift = static_cast<int64_t>(host_now_us) - static_cast<int64_t>(static_cast<uint64_t>(t_nicla_us) + sync_offset_us);
                        if (std::abs(current_drift) > 50000) { // If jump > 50ms (e.g. board reset), re-anchor
                            sync_offset_us = host_now_us - static_cast<uint64_t>(t_nicla_us);
                        } else {
                            sync_offset_us = static_cast<uint64_t>(static_cast<int64_t>(sync_offset_us) + static_cast<int64_t>(current_drift * 0.001));
                        }
                    }

                    uint64_t sample_host_us = static_cast<uint64_t>(t_nicla_us) + sync_offset_us;
                    uint64_t cam_time_us = warper_.host_to_camera_time(sample_host_us);
                    warper_.ingest_imu_raw(cam_time_us, cam_wx, cam_wy, cam_wz, cam_ax, cam_ay, cam_az);
                    packets_received_++;
                }
            }

            close(fd);
            connected_.store(false);
            std::cout << "[WARN] Nicla Sense ME IMU disconnected. Retrying in 500ms...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    ContinuousGyroWarper& warper_;
    std::string port_;
    int baud_;
    std::atomic<bool> running_{false};
    std::atomic<bool> connected_{false};
    std::atomic<uint64_t> packets_received_{0};
    std::thread thread_;
};

} // namespace predator

#endif // PREDATOR_EGO_MOTION_HPP
