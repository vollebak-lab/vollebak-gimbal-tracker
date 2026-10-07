/**
 * @file evt21_capture.cpp
 * @brief Phase 33.4b: records raw EVT 2.1 from the live IMX636 through the HAL (no SDK decode in the
 *        loop), verifies the word layout, and writes a parity fixture for the GPU decoder.
 *
 * Outputs (prefix given on the command line):
 *   <prefix>.evt21raw  concatenated raw USB buffers, exactly as received (8-byte words)
 *   <prefix>.cd        reference decode by OpenEB EVT21LegacyDecoder (time shifting OFF), records of
 *                      struct CdRecord {uint16 x, uint16 y, int16 p, int16 reserved, int64 t} (16 B)
 *
 * Layout verification: every word is classified under the legacy layout (type = bits 28..31) and the
 * little-endian layout (type = bits 60..63). For the correct layout, all words must be CD/TIME_HIGH/
 * EXT_TRIGGER/OTHERS/CONTINUED types, TIME_HIGH must be non-decreasing, and all decoded pixels must lie
 * inside the 1280x720 array. The tool exits non-zero if the legacy layout fails these checks.
 * It also reports raw-buffer size/interval statistics, which size the 33.4b.c mapped ring.
 *
 * Usage: evt21_capture <prefix> [seconds=5]     (camera must not be in use: stop the service first)
 * Exit: 0 ok, 1 layout check failed, 2 usage, 3 camera/HAL error.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <metavision/hal/decoders/evt21/evt21_decoder.h>
#include <metavision/hal/device/device.h>
#include <metavision/hal/device/device_discovery.h>
#include <metavision/hal/facilities/i_event_decoder.h>
#include <metavision/hal/facilities/i_events_stream.h>
#include <metavision/hal/facilities/i_hw_identification.h>
#include <metavision/hal/utils/device_config.h>
#include <metavision/sdk/base/events/event_cd.h>

#include "evt21_format.hpp"

namespace {

constexpr uint32_t kWidth  = 1280;
constexpr uint32_t kHeight = 720;

/// Fixture record (explicit layout, independent of Metavision::EventCD padding).
struct CdRecord {
    uint16_t x;
    uint16_t y;
    int16_t p;
    int16_t reserved;
    int64_t t;
};
static_assert(sizeof(CdRecord) == 16, "fixture record must be 16 bytes");

/// Per-layout word classification counters.
struct LayoutStats {
    std::array<uint64_t, 16> type_hist{};
    uint64_t time_high_regressions = 0;
    uint64_t pixels_outside        = 0;
    uint32_t last_high             = 0;
    bool have_high                 = false;
};

bool is_known_type(uint32_t t) {
    using namespace predator::evt21;
    return t == kCdOff || t == kCdOn || t == kTimeHigh || t == kExtTrigger || t == kOthers || t == kContinued;
}

void classify_legacy(uint64_t w, LayoutStats& s) {
    using namespace predator::evt21;
    const uint32_t t = word_type(w);
    ++s.type_hist[t];
    if (t == kTimeHigh) {
        const uint32_t h = time_high(w);
        if (s.have_high && h < s.last_high) ++s.time_high_regressions;
        s.last_high = h;
        s.have_high = true;
    } else if (t <= kCdOn) {
        const uint32_t mask = cd_mask(w);
        const uint32_t top  = mask ? 31u - static_cast<uint32_t>(__builtin_clz(mask)) : 0u;
        if (cd_y(w) >= kHeight || cd_x_base(w) + top >= kWidth) ++s.pixels_outside;
    }
}

void classify_little(uint64_t w, LayoutStats& s) {
    // Non-legacy EVT21: type in bits 60..63, TIME_HIGH ts in bits 32..59, CD y bits 32..42, x 43..53.
    const uint32_t t = static_cast<uint32_t>(w >> 60) & 0xFu;
    ++s.type_hist[t];
    if (t == predator::evt21::kTimeHigh) {
        const uint32_t h = static_cast<uint32_t>(w >> 32) & predator::evt21::kTimeHighMax;
        if (s.have_high && h < s.last_high) ++s.time_high_regressions;
        s.last_high = h;
        s.have_high = true;
    } else if (t <= 1u) {
        const uint32_t y = static_cast<uint32_t>(w >> 32) & 0x7FFu, x = static_cast<uint32_t>(w >> 43) & 0x7FFu;
        if (y >= kHeight || x >= kWidth) ++s.pixels_outside;
    }
}

void print_stats(const char* name, const LayoutStats& s, uint64_t words) {
    uint64_t unknown = 0;
    for (uint32_t t = 0; t < 16; ++t) {
        if (!is_known_type(t)) unknown += s.type_hist[t];
    }
    std::printf("  %-7s CD_OFF=%llu CD_ON=%llu TIME_HIGH=%llu EXT_TRIG=%llu OTHERS=%llu CONT=%llu UNKNOWN=%llu (%.2f%%) "
                "TH_regressions=%llu pixels_outside=%llu\n",
                name, (unsigned long long)s.type_hist[0], (unsigned long long)s.type_hist[1],
                (unsigned long long)s.type_hist[8], (unsigned long long)s.type_hist[0xA],
                (unsigned long long)s.type_hist[0xE], (unsigned long long)s.type_hist[0xF],
                (unsigned long long)unknown, words ? 100.0 * unknown / words : 0.0,
                (unsigned long long)s.time_high_regressions, (unsigned long long)s.pixels_outside);
}

double percentile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[static_cast<size_t>(q * static_cast<double>(v.size() - 1))];
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);  // keep diagnostics if the process aborts
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: %s <output_prefix> [seconds=5]\n", argv[0]);
        return 2;
    }
    const std::string prefix = argv[1];
    const double seconds     = argc == 3 ? std::atof(argv[2]) : 5.0;
    if (!(seconds > 0.0 && seconds <= 600.0)) {
        std::fprintf(stderr, "seconds must be in (0, 600]\n");
        return 2;
    }

    try {
        Metavision::DeviceConfig cfg;
        cfg.set_format("EVT21");
        std::unique_ptr<Metavision::Device> device = Metavision::DeviceDiscovery::open("", cfg);
        if (!device) {
            std::fprintf(stderr, "[ERROR] no camera found\n");
            return 3;
        }
        auto* hwid   = device->get_facility<Metavision::I_HW_Identification>();
        auto* stream = device->get_facility<Metavision::I_EventsStream>();
        if (!hwid || !stream) {
            std::fprintf(stderr, "[ERROR] camera lacks I_HW_Identification / I_EventsStream\n");
            return 3;
        }
        const std::string fmt = hwid->get_current_data_encoding_format();
        std::printf("[INFO] camera %s, current format: %s\n", hwid->get_serial().c_str(), fmt.c_str());
        if (fmt.rfind("EVT21", 0) != 0) {
            std::fprintf(stderr, "[ERROR] camera did not switch to EVT21\n");
            return 3;
        }

        // Reference decoder: OpenEB's legacy EVT21 decoder, absolute timestamps. All sinks are supplied
        // exactly as hal_psee_plugins/src/utils/make_decoder.cpp does: EVT21GenericDecoder dereferences its
        // monitoring forwarder unconditionally, so omitting the EventMonitoring sink crashes on the first
        // OTHERS word (live streams carry them; synthetic ones did not).
        auto cd_sink      = std::make_shared<Metavision::I_EventDecoder<Metavision::EventCD>>();
        auto trig_sink    = std::make_shared<Metavision::I_EventDecoder<Metavision::EventExtTrigger>>();
        auto erc_sink     = std::make_shared<Metavision::I_EventDecoder<Metavision::EventERCCounter>>();
        auto monitor_sink = std::make_shared<Metavision::I_EventDecoder<Metavision::EventMonitoring>>();
        Metavision::EVT21LegacyDecoder ref_decoder(false, cd_sink, trig_sink, erc_sink, monitor_sink);
        std::vector<CdRecord> decoded;
        cd_sink->add_event_buffer_callback([&](const Metavision::EventCD* b, const Metavision::EventCD* e) {
            for (; b != e; ++b) decoded.push_back(CdRecord{b->x, b->y, b->p, 0, b->t});
        });

        std::ofstream raw_out(prefix + ".evt21raw", std::ios::binary);
        if (!raw_out) {
            std::fprintf(stderr, "[ERROR] cannot write %s.evt21raw\n", prefix.c_str());
            return 3;
        }

        LayoutStats legacy, little;
        std::vector<double> buf_bytes, buf_gap_us;
        uint64_t words = 0, partial_word_bytes = 0;
        auto last_buf = std::chrono::steady_clock::now();
        const auto t_end = last_buf + std::chrono::duration<double>(seconds);

        stream->start();
        while (std::chrono::steady_clock::now() < t_end) {
            if (stream->wait_next_buffer() < 0) {
                std::fprintf(stderr, "[ERROR] wait_next_buffer failed\n");
                return 3;
            }
            Metavision::DataTransfer::BufferPtr buf = stream->get_latest_raw_data();
            if (!buf || buf.size() == 0) continue;
            const auto now = std::chrono::steady_clock::now();
            buf_gap_us.push_back(std::chrono::duration<double, std::micro>(now - last_buf).count());
            last_buf = now;
            buf_bytes.push_back(static_cast<double>(buf.size()));
            partial_word_bytes += buf.size() % 8;

            raw_out.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
            ref_decoder.decode(buf.data(), buf.data() + buf.size());

            const size_t n = buf.size() / 8;
            for (size_t i = 0; i < n; ++i) {
                uint64_t w;
                std::memcpy(&w, buf.data() + 8 * i, sizeof(w));
                classify_legacy(w, legacy);
                classify_little(w, little);
            }
            words += n;
        }
        stream->stop();
        raw_out.close();

        std::ofstream cd_out(prefix + ".cd", std::ios::binary);
        cd_out.write(reinterpret_cast<const char*>(decoded.data()),
                     static_cast<std::streamsize>(decoded.size() * sizeof(CdRecord)));

        const double span_s = decoded.size() > 1 ? 1e-6 * static_cast<double>(decoded.back().t - decoded.front().t) : 0.0;
        std::printf("[INFO] %zu buffers, %llu words (%.1f MB), %zu decoded events over %.2f s event time (%.0f ev/s), "
                    "%.2f B/event\n",
                    buf_bytes.size(), (unsigned long long)words, 8.0 * words / 1e6, decoded.size(), span_s,
                    span_s > 0 ? decoded.size() / span_s : 0.0, decoded.empty() ? 0.0 : 8.0 * words / decoded.size());
        std::printf("[INFO] buffer bytes p0/p50/p90/p100 = %.0f / %.0f / %.0f / %.0f ; gap us p50/p90/p99 = %.0f / %.0f / %.0f ; "
                    "non-multiple-of-8 bytes: %llu\n",
                    percentile(buf_bytes, 0.0), percentile(buf_bytes, 0.5), percentile(buf_bytes, 0.9),
                    percentile(buf_bytes, 1.0), percentile(buf_gap_us, 0.5), percentile(buf_gap_us, 0.9),
                    percentile(buf_gap_us, 0.99), (unsigned long long)partial_word_bytes);
        std::printf("[INFO] word classification:\n");
        print_stats("legacy", legacy, words);
        print_stats("little", little, words);

        uint64_t legacy_unknown = 0;
        for (uint32_t t = 0; t < 16; ++t) {
            if (!is_known_type(t)) legacy_unknown += legacy.type_hist[t];
        }
        const bool ok = words > 0 && legacy.type_hist[predator::evt21::kTimeHigh] > 0 && legacy_unknown == 0 &&
                        legacy.time_high_regressions == 0 && legacy.pixels_outside == 0 && partial_word_bytes == 0;
        std::printf(ok ? "[OK] legacy EVT2.1 layout verified on live data\n"
                       : "[FAIL] legacy EVT2.1 layout check failed\n");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ERROR] %s\n", e.what());
        return 3;
    }
}
