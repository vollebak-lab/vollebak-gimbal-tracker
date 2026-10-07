#pragma once
/**
 * @file evt3_encoder.hpp
 * @brief Reference EVT3 (Prophesee IMX636 "EVT 3.0") encoder for benchmarks and decoder parity tests.
 *
 * Phase 33.4b. The live camera streams EVT3 (verified with metavision_platform_info: "Current Data
 * Encoding Format EVT3", alternatives EVT21). To move decoding onto the GPU we need (a) synthetic raw
 * streams at controlled event rates to benchmark the OpenEB CPU decoder, and (b) a ground truth for
 * bit-exact parity tests of the GPU decoder. This encoder provides both.
 *
 * Contract: for any time-ordered event sequence, decoding the produced words with OpenEB 5.2
 * `Metavision::EVT3Decoder` (time shifting disabled) yields exactly the input sequence, in order,
 * including across 24-bit timestamp loops (16.777216 s) and arbitrary decode-buffer splits.
 *
 * Word grammar emitted (OpenEB 5.2 semantics, hal/decoders/evt3/evt3_decoder.h):
 *   EVT_TIME_HIGH (0x8)  t[23:12]; emitted for EVERY 4096 us step, as the sensor does, so the
 *                        decoder's loop counter (high wraps 4095 -> 0) stays correct across gaps.
 *   EVT_TIME_LOW  (0x6)  t[11:0]; emitted after any TIME_HIGH change or when t[11:0] changes.
 *   EVT_ADDR_Y    (0x0)  y, orig=0 (CD source); emitted when y changes (or after a TIME_HIGH-free start).
 *   EVT_ADDR_X    (0x2)  x | pol<<11 : one event.
 *   VECT_BASE_X   (0x3)  x | pol<<11, followed by VECT_12 (0x4), VECT_12 (0x4), VECT_8 (0x5):
 *                        a 32-bit validity mask; event k at x_base + k. OpenEB's BasicCheckValidator
 *                        REJECTS vectors with x_base + 32 > width, so vectors are only used when they fit.
 *
 * Ordering rule (defines the canonical order the decoder reproduces): a vector covers a maximal run of
 * CONSECUTIVE input events with identical (t, y, p) and strictly increasing x inside [x0, x0 + 32).
 * Events are never reordered, so decode(encode(seq)) == seq element-wise.
 */

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace predator::evt3 {

/// EVT3 4-bit word types (subset used by CD streams).
enum WordType : uint16_t {
    kAddrY     = 0x0,
    kAddrX     = 0x2,
    kVectBaseX = 0x3,
    kVect12    = 0x4,
    kVect8     = 0x5,
    kTimeLow   = 0x6,
    kTimeHigh  = 0x8,
};

/// Packs a 4-bit type and 12-bit payload into one little-endian EVT3 word.
constexpr uint16_t make_word(uint16_t type, uint16_t payload) {
    return static_cast<uint16_t>((type << 12) | (payload & 0x0FFFu));
}

/// Runs of fewer events than this are emitted as individual EVT_ADDR_X words. A vector costs 4 words
/// (VECT_BASE_X + 3), so it only saves bandwidth from 4 events up; 3 matches break-even + 1 word.
inline constexpr size_t kMinVectorRun = 3;

/**
 * @brief Stateful encoder; successive encode() calls continue one stream (as consecutive USB buffers).
 */
class Evt3Encoder {
public:
    Evt3Encoder(int width, int height) : width_(width), height_(height) {
        if (width <= 0 || width > 2048 || height <= 0 || height > 2048) {
            throw std::invalid_argument("Evt3Encoder: geometry must fit 11-bit EVT3 addresses");
        }
    }

    /**
     * @brief Appends the EVT3 encoding of [begin, end) to @p out.
     * @tparam Ev any type with integral members x, y, p (0/1) and t (microseconds, >= 0).
     * @throws std::invalid_argument on out-of-range coordinates, bad polarity, or decreasing time.
     */
    template <class Ev>
    void encode(const Ev* begin, const Ev* end, std::vector<uint16_t>& out) {
        for (const Ev* it = begin; it != end;) {
            validate(*it);
            emit_time(static_cast<int64_t>(it->t), out);
            const uint16_t y = static_cast<uint16_t>(it->y);
            if (!y_valid_ || y != last_y_) {
                out.push_back(make_word(kAddrY, y));  // orig bit (11) = 0 -> CD event source
                last_y_  = y;
                y_valid_ = true;
            }

            // Measure the vectorizable run starting at `it` (consecutive, same t/y/p, increasing x).
            const int x0     = static_cast<int>(it->x);
            const uint16_t p = static_cast<uint16_t>(it->p);
            size_t run       = 1;
            uint32_t mask    = 1u;
            if (x0 + 32 <= width_) {
                int last_x = x0;
                for (const Ev* nx = it + 1; nx != end; ++nx) {
                    if (static_cast<int64_t>(nx->t) != static_cast<int64_t>(it->t) || nx->y != it->y ||
                        static_cast<uint16_t>(nx->p) != p) {
                        break;
                    }
                    const int x = static_cast<int>(nx->x);
                    if (x <= last_x || x >= x0 + 32) break;
                    validate(*nx);
                    mask |= 1u << (x - x0);
                    last_x = x;
                    ++run;
                }
            }

            if (run >= kMinVectorRun) {
                out.push_back(make_word(kVectBaseX, static_cast<uint16_t>(x0 | (p << 11))));
                out.push_back(make_word(kVect12, static_cast<uint16_t>(mask & 0xFFFu)));
                out.push_back(make_word(kVect12, static_cast<uint16_t>((mask >> 12) & 0xFFFu)));
                out.push_back(make_word(kVect8, static_cast<uint16_t>((mask >> 24) & 0xFFu)));
                it += run;
            } else {
                out.push_back(make_word(kAddrX, static_cast<uint16_t>(x0 | (p << 11))));
                ++it;
            }
        }
    }

private:
    template <class Ev>
    void validate(const Ev& e) const {
        if (static_cast<int64_t>(e.x) < 0 || static_cast<int64_t>(e.x) >= width_ ||
            static_cast<int64_t>(e.y) < 0 || static_cast<int64_t>(e.y) >= height_) {
            throw std::invalid_argument("Evt3Encoder: event outside sensor geometry");
        }
        if (e.p != 0 && e.p != 1) throw std::invalid_argument("Evt3Encoder: polarity must be 0 or 1");
        if (static_cast<int64_t>(e.t) < 0) throw std::invalid_argument("Evt3Encoder: negative timestamp");
        if (time_valid_ && static_cast<int64_t>(e.t) < last_t_) {
            throw std::invalid_argument("Evt3Encoder: timestamps must be non-decreasing (got " +
                                        std::to_string(static_cast<int64_t>(e.t)) + " after " +
                                        std::to_string(last_t_) + ")");
        }
    }

    /// Emits TIME_HIGH for every 4096 us step up to t (sensor behaviour), then TIME_LOW if needed.
    void emit_time(int64_t t, std::vector<uint16_t>& out) {
        const int64_t high = t >> 12;
        const uint16_t low = static_cast<uint16_t>(t & 0xFFF);
        bool high_changed  = false;
        if (!time_valid_) {
            // EVT3 has no loop field: the decoder starts its loop counter at 0, exactly like the sensor,
            // whose counter starts at 0 when streaming begins. A first timestamp beyond one 24-bit period
            // cannot round-trip.
            if (t >= (int64_t{1} << 24)) {
                throw std::invalid_argument("Evt3Encoder: first timestamp must be < 2^24 us");
            }
            out.push_back(make_word(kTimeHigh, static_cast<uint16_t>(high & 0xFFF)));
            high_changed = true;
        } else {
            for (int64_t h = last_high_ + 1; h <= high; ++h) {
                out.push_back(make_word(kTimeHigh, static_cast<uint16_t>(h & 0xFFF)));
                high_changed = true;
            }
        }
        // The decoder zeroes time-low whenever time-high changes, so re-send it unconditionally then.
        if (high_changed || low != last_low_) {
            out.push_back(make_word(kTimeLow, low));
        }
        last_high_  = high;
        last_low_   = low;
        last_t_     = t;
        time_valid_ = true;
    }

    int width_;
    int height_;
    int64_t last_high_ = 0;
    uint16_t last_low_ = 0;
    int64_t last_t_    = 0;
    bool time_valid_   = false;
    uint16_t last_y_   = 0;
    bool y_valid_      = false;
};

}  // namespace predator::evt3
