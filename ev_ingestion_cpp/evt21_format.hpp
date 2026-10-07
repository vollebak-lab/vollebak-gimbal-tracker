#pragma once
/**
 * @file evt21_format.hpp
 * @brief Prophesee EVT 2.1 word layout as emitted by the IMX636 (OpenEB "endianness=legacy"),
 *        shared by host tools, tests and CUDA kernels (Phase 33.4b).
 *
 * Source of truth: OpenEB 5.2 `hal/decoders/evt21/evt21_event_types.h` (Evt21LegacyRaw) and
 * `hal_psee_plugins/src/devices/imx636/imx636_tz_device.cpp`, whose get_output_format() reports
 * "EVT21;height=720;width=1280;endianness=legacy" for this sensor. Each 64-bit little-endian word:
 *
 *   bits  0..10  y        (CD)          | bits 0..27 ts[33:6]  (TIME_HIGH)
 *   bits 11..21  x base   (CD)          |
 *   bits 22..27  ts[5:0]  (CD/OTHERS/EXT_TRIGGER)
 *   bits 28..31  type
 *   bits 32..63  validity mask (CD): event k at (x_base + k, y) for every set bit k, ascending k
 *
 * Decode semantics reproduced by the GPU decoder (OpenEB EVT21GenericDecoder, time shifting off):
 *   - words before the first TIME_HIGH of the stream are dropped;
 *   - TIME_HIGH sets ts[33:6]; the 34-bit counter loop (2^34 us = 4.77 h) increments only on an exact
 *     wrap (previous high == 2^28 - 1 and new high == 0); any other change just replaces the high part;
 *   - a CD word's timestamp is (loop << 34) | (high << 6) | ts[5:0]; type 0x1 = ON (p = 1), 0x0 = OFF.
 */

#include <cstdint>

#if defined(__CUDACC__)
#define EVT21_HD __host__ __device__ __forceinline__
#else
#define EVT21_HD inline
#endif

namespace predator::evt21 {

/// EVT 2.1 4-bit word types.
enum WordType : uint32_t {
    kCdOff      = 0x0,
    kCdOn       = 0x1,
    kTimeHigh   = 0x8,
    kExtTrigger = 0xA,
    kOthers     = 0xE,
    kContinued  = 0xF,
};

inline constexpr int kTimeHighBits      = 28;
inline constexpr int kTimeLowBits       = 6;
inline constexpr int kLoopShift         = kTimeHighBits + kTimeLowBits;  // 34
inline constexpr uint32_t kTimeHighMax  = (1u << kTimeHighBits) - 1u;
inline constexpr int kMaxEventsPerWord  = 32;

EVT21_HD uint32_t word_type(uint64_t w) { return static_cast<uint32_t>(w >> 28) & 0xFu; }
EVT21_HD bool is_cd(uint64_t w) { return word_type(w) <= kCdOn; }
EVT21_HD uint32_t cd_y(uint64_t w) { return static_cast<uint32_t>(w) & 0x7FFu; }
EVT21_HD uint32_t cd_x_base(uint64_t w) { return static_cast<uint32_t>(w >> 11) & 0x7FFu; }
EVT21_HD uint32_t ts_low(uint64_t w) { return static_cast<uint32_t>(w >> 22) & 0x3Fu; }
EVT21_HD uint32_t cd_mask(uint64_t w) { return static_cast<uint32_t>(w >> 32); }
EVT21_HD uint32_t cd_polarity(uint64_t w) { return word_type(w) == kCdOn ? 1u : 0u; }
EVT21_HD uint32_t time_high(uint64_t w) { return static_cast<uint32_t>(w) & kTimeHighMax; }

/// Builds a CD word (legacy layout).
EVT21_HD uint64_t make_cd_word(uint32_t polarity, uint32_t ts6, uint32_t x_base, uint32_t y, uint32_t mask) {
    const uint32_t lo = (y & 0x7FFu) | ((x_base & 0x7FFu) << 11) | ((ts6 & 0x3Fu) << 22) |
                        ((polarity ? kCdOn : kCdOff) << 28);
    return (static_cast<uint64_t>(mask) << 32) | lo;
}

/// Builds a TIME_HIGH word (legacy layout) carrying ts[33:6].
EVT21_HD uint64_t make_time_high_word(uint32_t high28) {
    return static_cast<uint64_t>((high28 & kTimeHighMax) | (static_cast<uint32_t>(kTimeHigh) << 28));
}

}  // namespace predator::evt21
