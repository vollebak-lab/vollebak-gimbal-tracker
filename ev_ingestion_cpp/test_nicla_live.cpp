#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <atomic>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <cstring>

#include "ego_motion.hpp"

int open_serial_port(const char* port_name, int baud = B115200) {
    int fd = open(port_name, O_RDWR | O_NOCTTY | O_SYNC);
    if (fd < 0) {
        std::cerr << "[ERROR] Could not open serial port: " << port_name << " (" << strerror(errno) << ")\n";
        return -1;
    }

    struct termios tty{};
    if (tcgetattr(fd, &tty) != 0) {
        close(fd);
        return -1;
    }

    cfsetospeed(&tty, baud);
    cfsetispeed(&tty, baud);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8; // 8-bit chars
    tty.c_iflag &= ~IGNBRK;                     // disable break processing
    tty.c_lflag = 0;                            // no signaling chars, no echo, no canonical processing
    tty.c_oflag = 0;                            // no remapping, no delays
    tty.c_cc[VMIN]  = 1;                        // read at least 1 byte
    tty.c_cc[VTIME] = 1;                        // 0.1 seconds read timeout

    tty.c_iflag &= ~(IXON | IXOFF | IXANY);     // shut off xon/xoff ctrl
    tty.c_cflag |= (CLOCAL | CREAD);            // ignore modem controls, enable reading
    tty.c_cflag &= ~(PARENB | PARODD);          // shut off parity
    tty.c_cflag &= ~CSTOPB;                     // 1 stop bit
    tty.c_cflag &= ~CRTSCTS;                    // no hardware flow control

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        close(fd);
        return -1;
    }

    return fd;
}

static uint16_t compute_checksum(const uint8_t* data, size_t length) {
    uint16_t chk = 0;
    for (size_t i = 0; i < length; i += 2) {
        if (i + 1 < length) {
            chk ^= static_cast<uint16_t>(data[i]) | (static_cast<uint16_t>(data[i + 1]) << 8);
        } else {
            chk ^= static_cast<uint16_t>(data[i]);
        }
    }
    return chk;
}

int main(int argc, char* argv[]) {
    const char* port = "/dev/ttyACM0";
    if (argc > 1) port = argv[1];

    std::cout << "======================================================================\n";
    std::cout << "  Predator — Arduino Nicla Sense ME Live IMU Stream Verifier          \n";
    std::cout << "  Port: " << port << "                                                \n";
    std::cout << "======================================================================\n\n";

    int fd = open_serial_port(port);
    if (fd < 0) {
        return 1;
    }

    std::cout << "[INFO] Serial port opened. Listening for 200 Hz binary IMU packets...\n\n";

    int packet_count = 0;
    std::vector<uint8_t> rx_buf;
    rx_buf.reserve(1024);
    uint8_t chunk[256];

    while (packet_count < 20) {
        ssize_t n = read(fd, chunk, sizeof(chunk));
        if (n <= 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        rx_buf.insert(rx_buf.end(), chunk, chunk + n);

        while (rx_buf.size() >= 32) {
            // Find preamble 0xAA 0x55
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
                // Keep last byte in case 0xAA was at the very end
                uint8_t last = rx_buf.back();
                rx_buf.clear();
                if (last == 0xAA) rx_buf.push_back(last);
                break;
            }

            if (sync_pos > 0) {
                rx_buf.erase(rx_buf.begin(), rx_buf.begin() + sync_pos);
            }

            if (rx_buf.size() < 32) {
                break; // Need more bytes for a full packet
            }

            // Verify XOR Checksum over 28 bytes of payload (indices 2..29)
            uint16_t expected_chk = compute_checksum(rx_buf.data() + 2, 28);
            uint16_t packet_chk = 0;
            std::memcpy(&packet_chk, rx_buf.data() + 30, 2);

            if (expected_chk != packet_chk) {
                // Corrupted packet or false preamble: slide by 1 byte
                rx_buf.erase(rx_buf.begin(), rx_buf.begin() + 1);
                continue;
            }

            // Valid packet! Unpack
            uint32_t t_us = 0;
            float wx = 0, wy = 0, wz = 0;
            float ax = 0, ay = 0, az = 0;

            std::memcpy(&t_us, rx_buf.data() + 2, 4);
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

            if (std::abs(wx) > 35.0f || std::abs(wy) > 35.0f || std::abs(wz) > 35.0f ||
                std::abs(ax) > 100.0f || std::abs(ay) > 100.0f || std::abs(az) > 100.0f) {
                continue;
            }

            packet_count++;
            std::cout << "[" << std::setw(2) << packet_count << "] "
                      << "Timestamp: " << std::setw(10) << t_us << " us | "
                      << std::fixed << std::setprecision(4)
                      << "Gyro (Cam Frame): [wx=" << std::setw(7) << wx 
                      << ", wy=" << std::setw(7) << wy 
                      << ", wz=" << std::setw(7) << wz << "] rad/s | "
                      << std::setprecision(2)
                      << "Accel: [" << std::setw(6) << ax 
                      << ", " << std::setw(6) << ay 
                      << ", " << std::setw(6) << az << "] m/s^2\n";
        }
    }

    close(fd);
    std::cout << "\n[INFO] Successfully received and verified 20 live IMU packets from Nicla Sense ME!\n";
    return 0;
}
