/**
 * @file nicla_predator_imu.ino
 * @brief High-Rate 6-DoF IMU Streamer for Predator Ego-Motion Compensation Pipeline
 * @details Streams Bosch BHI260AP calibrated angular velocity and linear acceleration
 *          over USB CDC Serial to NVIDIA Jetson Orin Nano at 200-400 Hz.
 * 
 * Hardware: Arduino Nicla Sense ME (Nordic nRF52840 + Bosch BHI260AP)
 * Mounting: Rear plate of IDS IMX636 event camera (Rotated 90 deg clockwise)
 * Optical Coordinate Frame Mapping:
 *   w_x_cam = +w_y_nicla (Pitch / Tilt)
 *   w_y_cam = +w_x_nicla (Yaw / Pan)
 *   w_z_cam = -w_z_nicla (Roll)
 */

#include "Arduino.h"
#include "Arduino_BHY2.h"

// Bosch BHI260AP Sensor Handles
SensorXYZ accel(SENSOR_ID_ACC);
SensorXYZ gyro(SENSOR_ID_GYRO);

// Constant conversion factors
// BHI260AP Gyro default range is +/- 2000 dps => 1 LSB = (2000 * pi / 180) / 32768 = 0.001065264436 rad/s
static const float GYRO_SCALE_RAD_S = (2000.0f * 3.141592653589793f / 180.0f) / 32768.0f;
// BHI260AP Accel default range is +/- 8g => 1 LSB = (8 * 9.80665) / 32768 = 0.00239420166 m/s^2
static const float ACCEL_SCALE_M_S2 = (8.0f * 9.80665f) / 32768.0f;

// Binary Packet Structure (Packed, 32 bytes)
struct __attribute__((packed)) ImuBinaryPacket {
    uint8_t sync1{0xAA};         // Sync byte 1
    uint8_t sync2{0x55};         // Sync byte 2
    uint32_t timestamp_us{0};    // Microsecond timestamp from board boot
    float gyro_x{0.0f};          // Angular rate X in Camera Frame (rad/s)
    float gyro_y{0.0f};          // Angular rate Y in Camera Frame (rad/s)
    float gyro_z{0.0f};          // Angular rate Z in Camera Frame (rad/s)
    float accel_x{0.0f};         // Linear accel X in Camera Frame (m/s^2)
    float accel_y{0.0f};         // Linear accel Y in Camera Frame (m/s^2)
    float accel_z{0.0f};         // Linear accel Z in Camera Frame (m/s^2)
    uint16_t checksum{0};        // XOR Checksum of payload
};

static ImuBinaryPacket g_packet;

// Computes 16-bit XOR checksum over packet payload bytes
uint16_t compute_checksum(const uint8_t* data, size_t length) {
    uint16_t chk = 0;
    for (size_t i = 0; i < length; i += 2) {
        if (i + 1 < length) {
            chk ^= (uint16_t)data[i] | ((uint16_t)data[i + 1] << 8);
        } else {
            chk ^= (uint16_t)data[i];
        }
    }
    return chk;
}

void setup() {
    // USB CDC Serial
    Serial.begin(115200);

    // Initialize BHY2 sensor hub
    BHY2.begin();

    // Configure sensors at 200 Hz ODR with 0 latency for minimum phase delay
    accel.begin(200, 0);
    gyro.begin(200, 0);
}

void loop() {
    // Continuously service BHY2 sensor FIFO interrupt buffer
    BHY2.update();

    static uint32_t last_sample_us = 0;
    uint32_t now_us = micros();

    // 200 Hz stream interval (5000 microseconds)
    if (now_us - last_sample_us >= 5000) {
        last_sample_us = now_us;

        // Read raw integer values from Bosch BHI260AP
        int16_t raw_gx = gyro.x();
        int16_t raw_gy = gyro.y();
        int16_t raw_gz = gyro.z();

        int16_t raw_ax = accel.x();
        int16_t raw_ay = accel.y();
        int16_t raw_az = accel.z();

        // Convert to SI units in Nicla sensor body frame
        float nicla_wx = (float)raw_gx * GYRO_SCALE_RAD_S;
        float nicla_wy = (float)raw_gy * GYRO_SCALE_RAD_S;
        float nicla_wz = (float)raw_gz * GYRO_SCALE_RAD_S;

        float nicla_ax = (float)raw_ax * ACCEL_SCALE_M_S2;
        float nicla_ay = (float)raw_ay * ACCEL_SCALE_M_S2;
        float nicla_az = (float)raw_az * ACCEL_SCALE_M_S2;

        // Apply 90 deg clockwise physical rear-mount optical frame transformation:
        // w_cam_x = +w_nicla_y
        // w_cam_y = +w_nicla_x
        // w_cam_z = -w_nicla_z
        g_packet.timestamp_us = now_us;
        g_packet.gyro_x = nicla_wy;
        g_packet.gyro_y = nicla_wx;
        g_packet.gyro_z = -nicla_wz;

        g_packet.accel_x = nicla_ay;
        g_packet.accel_y = nicla_ax;
        g_packet.accel_z = -nicla_az;

        // Compute checksum over payload (bytes 2 to 29)
        g_packet.checksum = compute_checksum((const uint8_t*)&g_packet + 2, sizeof(ImuBinaryPacket) - 4);

        // Write binary packet over USB CDC
        Serial.write((const uint8_t*)&g_packet, sizeof(ImuBinaryPacket));
    }
}
