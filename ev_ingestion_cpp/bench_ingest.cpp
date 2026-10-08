/**
 * @file bench_ingest.cpp
 * @brief Phase 33.4b.a: measures the per-event CPU cost of every stage of the current ingest path.
 *
 * Stages measured, in the order the live camera thread executes them:
 *   1. OpenEB EVT3 decode   (Metavision::EVT3Decoder, the SDK's CPU decoder used by Camera::cd())
 *   2. frame_gen            (PeriodicFrameGenerationAlgorithm::process_events, UI preview, per event)
 *   3. ingest_event_batch   (Phase 33.4 CPU pass: copy + 2x2 sieve + max_t into pinned memory, then
 *                            async upload + GPU accumulation launch)
 *
 * Input: a synthetic but structurally realistic scene (isolated noise + simultaneous row segments
 * from edges + 4 flickering rotor disks), encoded to EVT3 with the reference encoder. Before any
 * timing, the stream is decoded by OpenEB and compared element-wise against the input: this both
 * validates evt3_encoder.hpp (needed for 33.4b.b parity tests) and guarantees we time real decoding.
 * Each run starts 0.5 s before a 24-bit timestamp loop, so loop handling is exercised every time.
 *
 * Output: ns/event (thread CPU time and wall time) and the fraction of one Cortex-A78AE core each
 * stage needs at the given event rate. Run with the service stopped (exclusive GPU, quiet CPU).
 *
 * Usage: bench_ingest [rate_mev_s ...]   (default: 1 3 10)
 * Exit codes: 0 ok, 1 encoder/decoder parity failure, 2 bad arguments, 3 CUDA failure.
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <metavision/hal/decoders/evt3/evt3_decoder.h>
#include <metavision/hal/facilities/i_event_decoder.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/core/utils/colors.h>

#include "cuda_flicker_core.cuh"
#include "evt3_encoder.hpp"
#include "synthetic_scene.hpp"

namespace {

constexpr int kWidth  = 1280;
constexpr int kHeight = 720;
// Bytes handed to the decoder per call. Matches the order of magnitude of a libusb bulk transfer;
// the decoder carries split multiword events across calls, which this also exercises.
constexpr size_t kDecodeChunkWords = 16 * 1024;
constexpr double kDurationS        = 1.0;
constexpr int kRepeats             = 3;  // decode/frame_gen are re-run; the median is reported

/// Wall and thread-CPU time of one measured region.
struct Timing {
    double wall_s = 0.0;
    double cpu_s  = 0.0;
};

double thread_cpu_seconds() {
    timespec ts{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return static_cast<double>(ts.tv_sec) + 1e-9 * static_cast<double>(ts.tv_nsec);
}

template <class F>
Timing measure(F&& fn) {
    const auto w0  = std::chrono::steady_clock::now();
    const double c0 = thread_cpu_seconds();
    fn();
    Timing t;
    t.cpu_s  = thread_cpu_seconds() - c0;
    t.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
    return t;
}

Timing median_of(std::vector<Timing> v) {
    std::sort(v.begin(), v.end(), [](const Timing& a, const Timing& b) { return a.cpu_s < b.cpu_s; });
    return v[v.size() / 2];
}

/// Synthetic scene (see synthetic_scene.hpp), sorted in canonical EVT order.
std::vector<Metavision::EventCD> make_scene(double rate_ev_s, int64_t t0_us, uint64_t seed) {
    return predator::synth::make_scene<Metavision::EventCD>(rate_ev_s, kDurationS, t0_us, seed, kWidth, kHeight);
}

/// Decoder plus the CD sink it dispatches into. time shifting OFF -> absolute timestamps.
struct SdkDecoder {
    std::shared_ptr<Metavision::I_EventDecoder<Metavision::EventCD>> cd;
    std::unique_ptr<Metavision::I_EventsStreamDecoder> stream;
    SdkDecoder() : cd(std::make_shared<Metavision::I_EventDecoder<Metavision::EventCD>>()) {
        stream = Metavision::make_evt3_decoder(false, kHeight, kWidth, cd);
    }
    void decode_all(const std::vector<uint16_t>& words) {
        const auto* base = reinterpret_cast<const uint8_t*>(words.data());
        const size_t total_bytes = words.size() * sizeof(uint16_t);
        const size_t chunk_bytes = kDecodeChunkWords * sizeof(uint16_t);
        for (size_t off = 0; off < total_bytes; off += chunk_bytes) {
            const size_t n = std::min(chunk_bytes, total_bytes - off);
            stream->decode(base + off, base + off + n);
        }
    }
};

/// Element-wise equality of decoded vs. input; prints the first mismatch.
bool check_parity(const std::vector<Metavision::EventCD>& in, const std::vector<Metavision::EventCD>& out) {
    if (in.size() != out.size()) {
        std::fprintf(stderr, "[PARITY] count mismatch: encoded %zu, decoded %zu\n", in.size(), out.size());
    }
    const size_t n = std::min(in.size(), out.size());
    for (size_t i = 0; i < n; ++i) {
        const auto& a = in[i];
        const auto& b = out[i];
        if (a.x != b.x || a.y != b.y || a.p != b.p || a.t != b.t) {
            std::fprintf(stderr, "[PARITY] first mismatch at %zu: in(x=%u y=%u p=%d t=%lld) out(x=%u y=%u p=%d t=%lld)\n",
                         i, a.x, a.y, a.p, static_cast<long long>(a.t), b.x, b.y, b.p, static_cast<long long>(b.t));
            return false;
        }
    }
    return in.size() == out.size();
}

void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[CUDA] %s: %s\n", what, cudaGetErrorString(e));
        std::exit(3);
    }
}

/// Runs all stages for one event rate and prints one result row. Returns false on parity failure.
bool run_rate(double rate_mev_s, uint64_t seed) {
    const int64_t t0_us = (int64_t{1} << 24) - 500000;  // cross a 24-bit loop mid-run
    const auto events   = make_scene(rate_mev_s * 1e6, t0_us, seed);
    const double span_s = 1e-6 * static_cast<double>(events.back().t - events.front().t);

    std::vector<uint16_t> words;
    words.reserve(events.size() * 3);
    predator::evt3::Evt3Encoder enc(kWidth, kHeight);
    enc.encode(events.data(), events.data() + events.size(), words);
    const double bytes_per_ev = 2.0 * static_cast<double>(words.size()) / static_cast<double>(events.size());

    // Parity pass (untimed): capture decoded events and the callback batch boundaries.
    std::vector<Metavision::EventCD> decoded;
    decoded.reserve(events.size());
    std::vector<size_t> batch_sizes;
    {
        SdkDecoder dec;
        dec.cd->add_event_buffer_callback([&](const Metavision::EventCD* b, const Metavision::EventCD* e) {
            decoded.insert(decoded.end(), b, e);
            batch_sizes.push_back(static_cast<size_t>(e - b));
        });
        dec.decode_all(words);
    }
    if (!check_parity(events, decoded)) return false;

    // 1. SDK decode (sink only counts, like a cheap callback).
    std::vector<Timing> dec_t;
    for (int r = 0; r < kRepeats; ++r) {
        SdkDecoder dec;
        size_t sink = 0;
        dec.cd->add_event_buffer_callback(
            [&](const Metavision::EventCD* b, const Metavision::EventCD* e) { sink += static_cast<size_t>(e - b); });
        dec_t.push_back(measure([&] { dec.decode_all(words); }));
        if (sink != events.size()) throw std::runtime_error("decode sink count mismatch");
    }

    // 2. frame_gen (same batching as the live callback).
    std::vector<Timing> fg_t;
    for (int r = 0; r < kRepeats; ++r) {
        Metavision::PeriodicFrameGenerationAlgorithm fg(kWidth, kHeight, 25000, 30.0,
                                                        Metavision::ColorPalette::Dark);
        size_t frames = 0;
        fg.set_output_callback([&](Metavision::timestamp, cv::Mat&) { ++frames; });
        fg_t.push_back(measure([&] {
            size_t off = 0;
            for (size_t n : batch_sizes) {
                fg.process_events(decoded.data() + off, decoded.data() + off + n);
                off += n;
            }
        }));
    }

    // 3. ingest_event_batch (fresh core per run so the absolute bin grid anchors on this stream).
    //    Run with the SDK's native callback batching and with coalesced batches: the difference
    //    isolates per-call CUDA API overhead from per-event CPU work (copy + sieve + max_t).
    auto time_ingest = [&](const std::vector<size_t>& batches, Timing& sync_out) {
        predator::CudaFlickerCore core(kWidth, kHeight, 32, 18, 4000.0, 512);
        const predator::Matrix3x3 H = predator::Matrix3x3::identity();
        uint64_t raw = 0, kept = 0, raw_total = 0;
        const Timing t = measure([&] {
            size_t off = 0;
            for (size_t n : batches) {
                core.ingest_event_batch(decoded.data() + off, n, H, raw, kept, nullptr, 0.35f, false);
                raw_total += raw;
                off += n;
            }
        });
        sync_out = measure([&] { cuda_check(cudaDeviceSynchronize(), "ingest sync"); });
        if (raw_total != decoded.size()) throw std::runtime_error("ingest raw count mismatch");
        return t;
    };
    auto coalesce = [&](size_t size) {
        std::vector<size_t> b;
        for (size_t off = 0; off < decoded.size(); off += size) b.push_back(std::min(size, decoded.size() - off));
        return b;
    };
    Timing sync_t, sync_unused;
    const Timing ing_t   = time_ingest(batch_sizes, sync_t);
    const Timing ing_4k  = time_ingest(coalesce(4096), sync_unused);
    const Timing ing_64k = time_ingest(coalesce(65536), sync_unused);

    const Timing d = median_of(dec_t), f = median_of(fg_t);
    const double n = static_cast<double>(events.size());
    auto ns = [n](double s) { return 1e9 * s / n; };
    auto core_pct = [span_s](double s) { return 100.0 * s / span_s; };
    const double total_cpu = d.cpu_s + f.cpu_s + ing_t.cpu_s;
    std::printf("%6.1f | %9zu | %5.2f | %6.1f %6.1f | %6.1f %6.1f | %6.1f %6.1f | %5.1f%% %5.1f%% %5.1f%% | %6.1f%% | %5zu | %7.2f\n",
                rate_mev_s, events.size(), bytes_per_ev, ns(d.cpu_s), ns(d.wall_s), ns(f.cpu_s), ns(f.wall_s),
                ns(ing_t.cpu_s), ns(ing_t.wall_s), core_pct(d.cpu_s), core_pct(f.cpu_s), core_pct(ing_t.cpu_s),
                core_pct(total_cpu), batch_sizes.size(), 1e3 * sync_t.wall_s);
    std::printf("       `- ingest cpu ns/ev by batch size: native(avg %.0f ev) %.1f | 4096 %.1f | 65536 %.1f\n",
                n / static_cast<double>(batch_sizes.size()), ns(ing_t.cpu_s), ns(ing_4k.cpu_s), ns(ing_64k.cpu_s));
    std::fflush(stdout);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<double> rates;
    for (int i = 1; i < argc; ++i) {
        char* end = nullptr;
        const double r = std::strtod(argv[i], &end);
        if (end == argv[i] || *end != '\0' || !(r > 0.0) || r > 50.0) {
            std::fprintf(stderr, "usage: %s [rate_mev_s ...]  (0 < rate <= 50)\n", argv[0]);
            return 2;
        }
        rates.push_back(r);
    }
    if (rates.empty()) rates = {1.0, 3.0, 10.0};

    // CUDA warm-up: context creation and module load must not be charged to the first rate.
    {
        predator::CudaFlickerCore warm(kWidth, kHeight, 32, 18, 4000.0, 512);
        std::vector<Metavision::EventCD> w(1024, Metavision::EventCD(10, 10, 1, 1000));
        uint64_t a = 0, b = 0;
        warm.ingest_event_batch(w.data(), w.size(), predator::Matrix3x3::identity(), a, b, nullptr, 0.35f, true);
        cuda_check(cudaDeviceSynchronize(), "warm-up");
    }

    std::printf("Per-event CPU cost of the live ingest path (span %.1f s per rate, decode chunk %zu B, median of %d)\n",
                kDurationS, kDecodeChunkWords * 2, kRepeats);
    std::printf("  Mev/s |    events |  B/ev | decode ns/ev  | frame_gen ns/ev | ingest ns/ev  | %% of one core (dec fg ing) | total  | batch | sync ms\n");
    std::printf("        |           |       |   cpu   wall  |   cpu   wall    |   cpu   wall  |                            |        |       |\n");
    uint64_t seed = 0x5EED33B4ull;
    for (double r : rates) {
        if (!run_rate(r, seed++)) {
            std::fprintf(stderr, "[FAIL] EVT3 encoder/OpenEB decoder parity failed at %.1f Mev/s\n", r);
            return 1;
        }
    }
    std::printf("[OK] EVT3 encoder round-trips bit-exactly through OpenEB EVT3Decoder at all rates (incl. 24-bit loop)\n");
    return 0;
}
