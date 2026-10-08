#pragma once
/**
 * @file evt21_encoder.hpp
 * @brief Reference EVT 2.1 (IMX636 legacy word order) encoder for GPU-decoder parity tests (Phase 33.4b).
 *
 * Contract: for any time-ordered CD sequence whose first timestamp is < 2^34 us, decoding the produced
 * words with OpenEB 5.2 `EVT21LegacyDecoder` (time shifting off) yields exactly the input sequence.
 *
 * Emission policy (mirrors the sensor):
 *   - TIME_HIGH (ts[33:6]) for EVERY 64 us step between consecutive events, so the decoder's exact-wrap
 *     loop rule (high 2^28-1 -> 0) sees the wrap even across idle gaps;
 *   - CD words group a maximal run of CONSECUTIVE input events with identical (t, y, p) and strictly
 *     increasing x inside one 32-pixel-aligned column group [x & ~31, (x & ~31) + 32), which is how the
 *     IMX636 vectorizes. Events are never reordered, so decode(encode(seq)) == seq element-wise.
 * `inject_*` helpers add non-CD words (OTHERS, EXT_TRIGGER, CONTINUED) that a correct decoder must skip.
 */

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "evt21_format.hpp"

namespace predator::evt21 {

class Evt21Encoder {
public:
    Evt21Encoder(int width, int height) : width_(width), height_(height) {
        if (width <= 0 || width > 2048 || height <= 0 || height > 2048) {
            throw std::invalid_argument("Evt21Encoder: geometry must fit 11-bit addresses");
        }
    }

    /// Appends the encoding of [begin, end). Ev needs integral x, y, p (0/1), t (us).
    template <class Ev>
    void encode(const Ev* begin, const Ev* end, std::vector<uint64_t>& out) {
        for (const Ev* it = begin; it != end;) {
            validate(*it);
            const int64_t t = static_cast<int64_t>(it->t);
            emit_time_high(t, out);

            const uint32_t y  = static_cast<uint32_t>(it->y);
            const uint32_t p  = static_cast<uint32_t>(it->p);
            const uint32_t x0 = static_cast<uint32_t>(it->x) & ~31u;
            uint32_t mask     = 1u << (static_cast<uint32_t>(it->x) - x0);
            uint32_t last_x   = static_cast<uint32_t>(it->x);
            const Ev* nx      = it + 1;
            for (; nx != end; ++nx) {
                if (static_cast<int64_t>(nx->t) != t || static_cast<uint32_t>(nx->y) != y ||
                    static_cast<uint32_t>(nx->p) != p) {
                    break;
                }
                const uint32_t x = static_cast<uint32_t>(nx->x);
                if (x <= last_x || x >= x0 + 32u) break;
                validate(*nx);
                mask |= 1u << (x - x0);
                last_x = x;
            }
            out.push_back(make_cd_word(p, static_cast<uint32_t>(t) & 0x3Fu, x0, y, mask));
            it = nx;
        }
    }

    /// Appends an OTHERS word (e.g. ERC counter) at the current time-low; decoders must not emit CD for it.
    static void inject_others(std::vector<uint64_t>& out, uint32_t subtype, uint32_t payload, uint32_t ts6) {
        const uint32_t lo = (subtype & 0xFFFFu) | ((ts6 & 0x3Fu) << 22) | (static_cast<uint32_t>(kOthers) << 28);
        out.push_back((static_cast<uint64_t>(payload) << 32) | lo);
    }
    /// Appends an EXT_TRIGGER word.
    static void inject_ext_trigger(std::vector<uint64_t>& out, uint32_t polarity, uint32_t id, uint32_t ts6) {
        const uint32_t lo = (polarity & 1u) | ((id & 0x1Fu) << 8) | ((ts6 & 0x3Fu) << 22) |
                            (static_cast<uint32_t>(kExtTrigger) << 28);
        out.push_back(lo);
    }
    /// Appends a CONTINUED word.
    static void inject_continued(std::vector<uint64_t>& out, uint64_t payload60) {
        out.push_back((payload60 & 0x0FFFFFFFull) | (static_cast<uint64_t>(kContinued) << 28) |
                      ((payload60 >> 28) << 32));
    }

private:
    template <class Ev>
    void validate(const Ev& e) const {
        if (static_cast<int64_t>(e.x) < 0 || static_cast<int64_t>(e.x) >= width_ ||
            static_cast<int64_t>(e.y) < 0 || static_cast<int64_t>(e.y) >= height_) {
            throw std::invalid_argument("Evt21Encoder: event outside sensor geometry");
        }
        if (e.p != 0 && e.p != 1) throw std::invalid_argument("Evt21Encoder: polarity must be 0 or 1");
        const int64_t t = static_cast<int64_t>(e.t);
        if (t < 0) throw std::invalid_argument("Evt21Encoder: negative timestamp");
        if (started_ && t < last_t_) throw std::invalid_argument("Evt21Encoder: timestamps must be non-decreasing");
    }

    void emit_time_high(int64_t t, std::vector<uint64_t>& out) {
        const int64_t high = t >> kTimeLowBits;  // includes loop bits above 28
        if (!started_) {
            if (t >= (int64_t{1} << kLoopShift)) {
                throw std::invalid_argument("Evt21Encoder: first timestamp must be < 2^34 us");
            }
            out.push_back(make_time_high_word(static_cast<uint32_t>(high)));
            started_ = true;
        } else {
            for (int64_t h = last_high_ + 1; h <= high; ++h) {
                out.push_back(make_time_high_word(static_cast<uint32_t>(h & kTimeHighMax)));
            }
        }
        last_high_ = high;
        last_t_    = t;
    }

    int width_;
    int height_;
    bool started_      = false;
    int64_t last_high_ = 0;
    int64_t last_t_    = 0;
};

}  // namespace predator::evt21
