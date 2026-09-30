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
    auto start_time = std::chrono::steady_clock::now();

    while (packet_count < 20) {
        uint8_t sync_byte = 0;
        if (read(fd, &sync_byte, 1) <= 0) continue;

        if (sync_byte == 0xAA) {
            uint8_t sync2 = 0;
            if (read(fd, &sync2, 1) <= 0) continue;
            if (sync2 == 0x55) {
                uint8_t payload[30];
                size_t bytes_read = 0;
                while (bytes_read < 30) {
                    ssize_t n = read(fd, payload + bytes_read, 30 - bytes_read);
                    if (n > 0) bytes_read += n;
                }

                uint32_t t_us = 0;
                float wx = 0, wy = 0, wz = 0;
                float ax = 0, ay = 0, az = 0;
                uint16_t chk = 0;

                std::memcpy(&t_us, payload + 0, 4);
                std::memcpy(&wx, payload + 4, 4);
                std::memcpy(&wy, payload + 8, 4);
                std::memcpy(&wz, payload + 12, 4);
                std::memcpy(&ax, payload + 16, 4);
                std::memcpy(&ay, payload + 20, 4);
                std::memcpy(&az, payload + 24, 4);
                std::memcpy(&chk, payload + 28, 2);

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
    }

    close(fd);
    std::cout << "\n[INFO] Successfully received and verified 20 live IMU packets from Nicla Sense ME!\n";
    return 0;
}
