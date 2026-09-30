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

#include "flicker_dsp.hpp"

namespace predator {

/**
 * @brief 3D Vector for angular velocity and acceleration
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
     * @brief Ingests an IMU sample into the circular buffer
     */
    void ingest_imu(const ImuSample& sample) {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        if (!imu_buffer_.empty() && sample.timestamp_us <= imu_buffer_.back().timestamp_us) {
            return; // Ignore non-monotonic samples
        }
        imu_buffer_.push_back(sample);
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
     * @brief Interpolates angular velocity at timestamp t_us
     */
    bool get_angular_velocity(uint64_t t_us, Vector3d& out_omega) const {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        if (imu_buffer_.empty()) {
            out_omega = Vector3d(0, 0, 0);
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
     * @brief Computes 3x3 spherical homography H = K * R * K_inv
     */
    Matrix3x3 compute_homography(uint64_t t_ref_us, uint64_t t_target_us) const {
        Matrix3x3 R = compute_rotation_matrix(t_ref_us, t_target_us);
        return K_ * R * K_inv_;
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

            connected_.store(true);
            std::cout << "[INFO] Nicla Sense ME IMU connected on " << port_ << "\n";

            uint64_t sync_offset_us = 0;
            bool first_sync = true;

            while (running_.load()) {
                uint8_t b1 = 0;
                if (read(fd, &b1, 1) <= 0) {
                    break; // Error or disconnect
                }

                if (b1 == 0xAA) {
                    uint8_t b2 = 0;
                    if (read(fd, &b2, 1) <= 0) break;
                    if (b2 == 0x55) {
                        uint8_t payload[30];
                        size_t bytes_read = 0;
                        while (bytes_read < 30 && running_.load()) {
                            ssize_t n = read(fd, payload + bytes_read, 30 - bytes_read);
                            if (n <= 0) break;
                            bytes_read += n;
                        }

                        if (bytes_read == 30) {
                            uint32_t t_nicla_us = 0;
                            float wx = 0, wy = 0, wz = 0;
                            float ax = 0, ay = 0, az = 0;
                            uint16_t chk = 0;

                            std::memcpy(&t_nicla_us, payload + 0, 4);
                            std::memcpy(&wx, payload + 4, 4);
                            std::memcpy(&wy, payload + 8, 4);
                            std::memcpy(&wz, payload + 12, 4);
                            std::memcpy(&ax, payload + 16, 4);
                            std::memcpy(&ay, payload + 20, 4);
                            std::memcpy(&az, payload + 24, 4);
                            std::memcpy(&chk, payload + 28, 2);

                            uint64_t host_now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count();

                            if (first_sync) {
                                sync_offset_us = host_now_us - t_nicla_us;
                                first_sync = false;
                            }

                            uint64_t syncd_timestamp_us = t_nicla_us + sync_offset_us;
                            warper_.ingest_imu_raw(syncd_timestamp_us, wx, wy, wz, ax, ay, az);
                            packets_received_++;
                        }
                    }
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
