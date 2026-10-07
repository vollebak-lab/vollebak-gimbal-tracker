/**
 * @file test_evt21_decoder.cpp
 * @brief Phase 33.4b.b parity tests: GpuEvt21Decoder must reproduce OpenEB 5.2 `EVT21LegacyDecoder`
 *        (time shifting off) bit-exactly and in order, for any split of the word stream into batches.
 *
 * Synthetic suite (default; run by the deploy script):
 *   1. encode -> OpenEB == input and encode -> GPU == input at 0.1 / 1 / 10 Mev/s, with random batch splits
 *      (1-word batches, batches without TIME_HIGH, batches up to the configured maximum) and max-size batches;
 *   2. words before the first TIME_HIGH are dropped, including a whole TIME_HIGH-free first batch;
 *   3. OTHERS / EXT_TRIGGER / CONTINUED words are ignored;
 *   4. 2^34 us counter wrap;
 *   5. backward and near-wrap TIME_HIGH discrepancies match OpenEB for every 2-way split, plus carried state;
 *   6. reset_async() restores stream-start semantics, no-reset continuation equals one concatenated stream,
 *      empty batch, rejected oversize batch and invalid construction;
 *   7. determinism across runs.
 * Fixture mode: `test_evt21_decoder --fixture <prefix> [--batch N]` streams <prefix>.evt21raw (live capture
 *   from evt21_capture) through the GPU decoder in N-word batches (default 16384) from mapped pinned memory,
 *   compares every event against <prefix>.cd (OpenEB reference) and reports host enqueue time and GPU time
 *   per batch.
 * Exit: 0 all passed, 1 a check failed, 2 usage / IO / CUDA error.
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <metavision/hal/decoders/evt21/evt21_decoder.h>
#include <metavision/hal/facilities/i_event_decoder.h>
#include <metavision/sdk/base/events/event_cd.h>

#include "evt21_encoder.hpp"
#include "evt21_format.hpp"
#include "evt21_gpu_decoder.cuh"
#include "synthetic_scene.hpp"

namespace {

using predator::CudaRawEvent;
using predator::GpuEvt21Decoder;
namespace evt21 = predator::evt21;

constexpr int kWidth  = 1280;
constexpr int kHeight = 720;

/// Same 16-byte record as the evt21_capture fixture (.cd files).
struct CdRecord {
    uint16_t x;
    uint16_t y;
    int16_t p;
    int16_t reserved;
    int64_t t;
};
static_assert(sizeof(CdRecord) == 16, "fixture record must be 16 bytes");

bool same_record(const CdRecord& a, const CdRecord& b) { return a.x == b.x && a.y == b.y && a.p == b.p && a.t == b.t; }

/// Event type for synthetic_scene.hpp and Evt21Encoder.
struct SceneEvent {
    uint16_t x;
    uint16_t y;
    int16_t p;
    int64_t t;
    SceneEvent(unsigned short x_, unsigned short y_, short p_, int64_t t_) : x(x_), y(y_), p(p_), t(t_) {}
};

int g_failures = 0;

void check(bool ok, const std::string& name) {
    std::printf("[TEST] %-78s: %s\n", name.c_str(), ok ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");
    if (!ok) ++g_failures;
}

void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

/**
 * @brief Element-wise comparison; prints the first divergence.
 * @param base global index of got[0] / want[0] (streaming comparisons)
 */
bool same_events(const CdRecord* got, size_t n_got, const CdRecord* want, size_t n_want, const std::string& label,
                 uint64_t base = 0) {
    const size_t n = std::min(n_got, n_want);
    for (size_t i = 0; i < n; ++i) {
        if (!same_record(got[i], want[i])) {
            std::printf("  [%s] first mismatch at event %llu: got (x=%u y=%u p=%d t=%lld) want (x=%u y=%u p=%d t=%lld)\n",
                        label.c_str(), (unsigned long long)(base + i), got[i].x, got[i].y, got[i].p,
                        (long long)got[i].t, want[i].x, want[i].y, want[i].p, (long long)want[i].t);
            return false;
        }
    }
    if (n_got != n_want) {
        std::printf("  [%s] event count %zu != expected %zu\n", label.c_str(), n_got, n_want);
        return false;
    }
    return true;
}

bool same_events(const std::vector<CdRecord>& got, const std::vector<CdRecord>& want, const std::string& label) {
    return same_events(got.data(), got.size(), want.data(), want.size(), label);
}

std::vector<CdRecord> to_records(const std::vector<SceneEvent>& ev) {
    std::vector<CdRecord> r;
    r.reserve(ev.size());
    for (const auto& e : ev) r.push_back(CdRecord{e.x, e.y, e.p, 0, e.t});
    return r;
}

std::vector<uint64_t> encode(const std::vector<SceneEvent>& ev) {
    evt21::Evt21Encoder enc(kWidth, kHeight);
    std::vector<uint64_t> words;
    enc.encode(ev.data(), ev.data() + ev.size(), words);
    return words;
}

/// Batch sizes partitioning a word stream, in order.
using Splits = std::vector<size_t>;

Splits chunk_splits(size_t n, size_t chunk) {
    Splits s;
    for (size_t pos = 0; pos < n; pos += chunk) s.push_back(std::min(chunk, n - pos));
    return s;
}

/// First 4096 words in 1..8-word batches (mostly without TIME_HIGH), then 10% single words,
/// 30% 2..64 words, 60% 65..max_batch words.
Splits random_splits(size_t n, size_t max_batch, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    Splits s;
    for (size_t pos = 0; pos < n;) {
        size_t len;
        if (pos < 4096) {
            len = std::uniform_int_distribution<size_t>(1, 8)(rng);
        } else {
            const double u = u01(rng);
            if (u < 0.1)      len = 1;
            else if (u < 0.4) len = std::uniform_int_distribution<size_t>(2, 64)(rng);
            else              len = std::uniform_int_distribution<size_t>(65, max_batch)(rng);
        }
        len = std::min(len, n - pos);
        s.push_back(len);
        pos += len;
    }
    return s;
}

Splits concat_splits(Splits a, const Splits& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

/**
 * @brief Reference decode with a fresh OpenEB EVT21LegacyDecoder, fed batch by batch.
 * All sinks are supplied (EVT21GenericDecoder dereferences the monitoring forwarder unconditionally).
 */
std::vector<CdRecord> openeb_decode(const std::vector<uint64_t>& words, const Splits& splits) {
    auto cd_sink      = std::make_shared<Metavision::I_EventDecoder<Metavision::EventCD>>();
    auto trig_sink    = std::make_shared<Metavision::I_EventDecoder<Metavision::EventExtTrigger>>();
    auto erc_sink     = std::make_shared<Metavision::I_EventDecoder<Metavision::EventERCCounter>>();
    auto monitor_sink = std::make_shared<Metavision::I_EventDecoder<Metavision::EventMonitoring>>();
    Metavision::EVT21LegacyDecoder decoder(false, cd_sink, trig_sink, erc_sink, monitor_sink);
    std::vector<CdRecord> out;
    out.reserve(words.size() * 2);
    cd_sink->add_event_buffer_callback([&out](const Metavision::EventCD* b, const Metavision::EventCD* e) {
        for (; b != e; ++b) out.push_back(CdRecord{b->x, b->y, b->p, 0, b->t});
    });
    const auto* bytes = reinterpret_cast<const uint8_t*>(words.data());
    size_t pos        = 0;
    for (size_t len : splits) {
        decoder.decode(bytes + 8 * pos, bytes + 8 * (pos + len));
        pos += len;
    }
    if (pos != words.size()) throw std::logic_error("openeb_decode: splits do not cover the stream");
    return out;
}

// ---------------------------------------------------------------------------------------------------------------
// GPU harness: mapped pinned input ring slot (as in production), device output, synchronous per-batch readback.
// ---------------------------------------------------------------------------------------------------------------

struct DeviceFree {
    void operator()(void* p) const { cudaFree(p); }
};
struct PinnedFree {
    void operator()(void* p) const { cudaFreeHost(p); }
};
template <class T>
using DevPtr = std::unique_ptr<T, DeviceFree>;
template <class T>
using PinnedPtr = std::unique_ptr<T, PinnedFree>;

template <class T>
DevPtr<T> device_alloc(size_t n) {
    void* p = nullptr;
    cuda_check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    return DevPtr<T>(static_cast<T*>(p));
}
template <class T>
PinnedPtr<T> pinned_alloc(size_t n, unsigned flags) {
    void* p = nullptr;
    cuda_check(cudaHostAlloc(&p, n * sizeof(T), flags), "cudaHostAlloc");
    return PinnedPtr<T>(static_cast<T*>(p));
}

struct StreamHolder {
    cudaStream_t s = nullptr;
    StreamHolder() { cuda_check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreate"); }
    ~StreamHolder() { cudaStreamDestroy(s); }
    StreamHolder(const StreamHolder&)            = delete;
    StreamHolder& operator=(const StreamHolder&) = delete;
};

struct EventHolder {
    cudaEvent_t e = nullptr;
    EventHolder() { cuda_check(cudaEventCreate(&e), "cudaEventCreate"); }
    ~EventHolder() { cudaEventDestroy(e); }
    EventHolder(const EventHolder&)            = delete;
    EventHolder& operator=(const EventHolder&) = delete;
};

struct BatchTiming {
    double host_enqueue_us = 0.0;  ///< CPU time spent inside decode_async (all launches + CUB dispatch)
    double gpu_us          = 0.0;  ///< GPU execution time of the batch (CUDA events)
};

class GpuHarness {
public:
    explicit GpuHarness(size_t max_words)
        : max_words_(max_words),
          h_words_(pinned_alloc<uint64_t>(max_words, cudaHostAllocMapped)),
          d_out_(device_alloc<CudaRawEvent>(GpuEvt21Decoder::max_events_for(max_words))),
          d_count_(device_alloc<uint32_t>(1)),
          h_count_(pinned_alloc<uint32_t>(1, cudaHostAllocDefault)),
          decoder_(std::make_unique<GpuEvt21Decoder>(max_words, stream_.s)) {
        void* d = nullptr;
        cuda_check(cudaHostGetDevicePointer(&d, h_words_.get(), 0), "cudaHostGetDevicePointer");
        d_words_ = static_cast<const uint64_t*>(d);
    }

    size_t max_words() const { return max_words_; }
    GpuEvt21Decoder& decoder() { return *decoder_; }
    cudaStream_t stream() const { return stream_.s; }
    /// Host view of the mapped input slot (fixture mode reads files straight into it).
    uint64_t* mapped_words() { return h_words_.get(); }
    /// Events seen with a non-zero `pad` (the decoder contract says 0).
    uint64_t nonzero_pad() const { return nonzero_pad_; }

    /// Decodes the n words currently in mapped_words(), appends the events, returns the batch event count.
    uint32_t decode_mapped(size_t n, std::vector<CdRecord>& out, BatchTiming* timing = nullptr) {
        cuda_check(cudaEventRecord(ev_start_.e, stream_.s), "event record");
        const auto c0 = std::chrono::steady_clock::now();
        decoder_->decode_async(d_words_, n, d_out_.get(), d_count_.get());
        const auto c1 = std::chrono::steady_clock::now();
        cuda_check(cudaEventRecord(ev_stop_.e, stream_.s), "event record");
        cuda_check(cudaMemcpyAsync(h_count_.get(), d_count_.get(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream_.s),
                   "count readback");
        cuda_check(cudaStreamSynchronize(stream_.s), "batch sync");
        const uint32_t count = *h_count_;
        if (count > GpuEvt21Decoder::max_events_for(n)) throw std::runtime_error("decoder reported impossible count");
        if (count > 0) {
            host_events_.resize(count);
            cuda_check(cudaMemcpyAsync(host_events_.data(), d_out_.get(), count * sizeof(CudaRawEvent),
                                       cudaMemcpyDeviceToHost, stream_.s),
                       "event readback");
            cuda_check(cudaStreamSynchronize(stream_.s), "readback sync");
            for (uint32_t i = 0; i < count; ++i) {
                const CudaRawEvent& e = host_events_[i];
                if (e.pad != 0) ++nonzero_pad_;
                out.push_back(CdRecord{e.x, e.y, e.p, 0, static_cast<int64_t>(e.t)});
            }
        }
        if (timing) {
            float ms = 0.0f;
            cuda_check(cudaEventElapsedTime(&ms, ev_start_.e, ev_stop_.e), "event elapsed");
            timing->gpu_us          = 1000.0 * ms;
            timing->host_enqueue_us = std::chrono::duration<double, std::micro>(c1 - c0).count();
        }
        return count;
    }

    uint32_t decode_batch(const uint64_t* words, size_t n, std::vector<CdRecord>& out) {
        if (n > max_words_) throw std::invalid_argument("decode_batch: batch larger than harness slot");
        if (n > 0) std::memcpy(h_words_.get(), words, n * sizeof(uint64_t));
        return decode_mapped(n, out);
    }

    /**
     * @brief Decodes a whole stream batch by batch.
     * @param reset enqueue reset_async() first (new stream); false continues the carried state
     * @param batches_without_th receives the number of batches containing no TIME_HIGH word
     */
    std::vector<CdRecord> decode(const std::vector<uint64_t>& words, const Splits& splits, bool reset = true,
                                 size_t* batches_without_th = nullptr) {
        if (reset) decoder_->reset_async();
        std::vector<CdRecord> out;
        out.reserve(words.size() * 2);
        size_t pos = 0, no_th = 0;
        for (size_t len : splits) {
            const uint64_t* b = words.data() + pos;
            if (std::none_of(b, b + len, [](uint64_t w) { return evt21::word_type(w) == evt21::kTimeHigh; })) ++no_th;
            decode_batch(b, len, out);
            pos += len;
        }
        if (pos != words.size()) throw std::logic_error("GpuHarness::decode: splits do not cover the stream");
        if (batches_without_th) *batches_without_th = no_th;
        return out;
    }

private:
    size_t max_words_;
    StreamHolder stream_;  // declared first: destroyed last
    PinnedPtr<uint64_t> h_words_;
    const uint64_t* d_words_ = nullptr;
    DevPtr<CudaRawEvent> d_out_;
    DevPtr<uint32_t> d_count_;
    PinnedPtr<uint32_t> h_count_;
    EventHolder ev_start_, ev_stop_;
    std::vector<CudaRawEvent> host_events_;
    uint64_t nonzero_pad_ = 0;
    std::unique_ptr<GpuEvt21Decoder> decoder_;  // declared last: destroyed first (syncs the stream)
};

// ---------------------------------------------------------------------------------------------------------------
// Synthetic suite
// ---------------------------------------------------------------------------------------------------------------

/// Test 1 (+7): round trip at one event rate.
void test_roundtrip(GpuHarness& gpu, double rate, double duration_s, int64_t t0, uint64_t seed) {
    const auto scene = predator::synth::make_scene<SceneEvent>(rate, duration_s, t0, seed, kWidth, kHeight);
    const auto words = encode(scene);
    const auto want  = to_records(scene);
    char tag[64];
    std::snprintf(tag, sizeof(tag), "%.1f Mev/s (%zu ev, %zu words)", rate / 1e6, want.size(), words.size());
    const std::string t(tag);

    check(same_events(openeb_decode(words, {words.size()}), want, t + " openeb"), t + ": OpenEB single buffer == input");
    const Splits sp = random_splits(words.size(), gpu.max_words(), seed ^ 0x9E3779B97F4A7C15ull);
    check(same_events(openeb_decode(words, sp), want, t + " openeb split"), t + ": OpenEB random splits == input");

    size_t no_th = 0;
    const auto got = gpu.decode(words, sp, true, &no_th);
    check(same_events(got, want, t + " gpu split") && no_th > 0,
          t + ": GPU " + std::to_string(sp.size()) + " random batches (" + std::to_string(no_th) +
              " w/o TIME_HIGH) == input");
    check(same_events(gpu.decode(words, chunk_splits(words.size(), gpu.max_words())), want, t + " gpu max"),
          t + ": GPU max-size batches == input");
    check(same_events(gpu.decode(words, sp), got, t + " gpu rerun"), t + ": GPU deterministic across runs");
}

/// Random CD / non-CD words that a decoder must drop because no TIME_HIGH precedes them.
std::vector<uint64_t> make_garbage(size_t n_cd, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<uint64_t> g;
    for (size_t i = 0; i < n_cd; ++i) {
        const uint32_t x0 = 32u * std::uniform_int_distribution<uint32_t>(0, kWidth / 32 - 1)(rng);
        g.push_back(evt21::make_cd_word(static_cast<uint32_t>(rng() & 1u), static_cast<uint32_t>(rng() & 63u), x0,
                                        std::uniform_int_distribution<uint32_t>(0, kHeight - 1)(rng),
                                        static_cast<uint32_t>(rng()) | 1u));
    }
    evt21::Evt21Encoder::inject_others(g, 0x0014, 1234, 7);
    evt21::Evt21Encoder::inject_ext_trigger(g, 1, 3, 9);
    evt21::Evt21Encoder::inject_continued(g, 0x0123456789ABCDEull);
    return g;
}

/// Test 2: words before the first TIME_HIGH are dropped.
void test_leading_garbage(GpuHarness& gpu) {
    const auto scene   = predator::synth::make_scene<SceneEvent>(1e6, 0.2, 5'000'000, 11, kWidth, kHeight);
    const auto want    = to_records(scene);
    const auto garbage = make_garbage(40, 12);
    auto words         = garbage;
    const auto body    = encode(scene);
    words.insert(words.end(), body.begin(), body.end());

    // A: the first batch is exactly the garbage (no TIME_HIGH at all). B: garbage and stream start share a batch.
    const Splits a = concat_splits({garbage.size()}, random_splits(body.size(), gpu.max_words(), 13));
    const Splits b = concat_splits({garbage.size() + 3}, random_splits(body.size() - 3, gpu.max_words(), 14));
    check(same_events(openeb_decode(words, a), want, "garbage openeb A"), "leading garbage: OpenEB, TIME_HIGH-free first batch");
    check(same_events(gpu.decode(words, a), want, "garbage gpu A"), "leading garbage: GPU, TIME_HIGH-free first batch dropped");
    check(same_events(openeb_decode(words, b), want, "garbage openeb B"), "leading garbage: OpenEB, garbage + start in one batch");
    check(same_events(gpu.decode(words, b), want, "garbage gpu B"), "leading garbage: GPU, garbage + start in one batch");
    check(same_events(gpu.decode(words, {words.size()}), want, "garbage gpu single"), "leading garbage: GPU, single batch");
}

/// Test 3: non-CD words are ignored.
void test_injected_words(GpuHarness& gpu) {
    const auto scene = predator::synth::make_scene<SceneEvent>(1e6, 0.3, 9'000'000, 21, kWidth, kHeight);
    const auto want  = to_records(scene);
    const auto body  = encode(scene);
    std::mt19937_64 rng(22);
    std::vector<uint64_t> words;
    size_t injected = 0;
    for (size_t i = 0; i < body.size(); ++i) {
        words.push_back(body[i]);
        if (rng() % 20 != 0) continue;  // ~5% of positions, always after the first TIME_HIGH
        const uint32_t ts6 = static_cast<uint32_t>(rng() & 63u);
        switch (rng() % 3) {
            case 0: evt21::Evt21Encoder::inject_others(words, static_cast<uint32_t>(rng()), static_cast<uint32_t>(rng()), ts6); break;
            case 1: evt21::Evt21Encoder::inject_ext_trigger(words, static_cast<uint32_t>(rng()), static_cast<uint32_t>(rng()), ts6); break;
            default: evt21::Evt21Encoder::inject_continued(words, rng()); break;
        }
        ++injected;
    }
    const Splits sp = random_splits(words.size(), gpu.max_words(), 23);
    const std::string t = "injected words (" + std::to_string(injected) + " OTHERS/EXT_TRIGGER/CONTINUED)";
    check(same_events(openeb_decode(words, sp), want, "inject openeb"), t + ": OpenEB == input");
    check(same_events(gpu.decode(words, sp), want, "inject gpu"), t + ": GPU == input");
}

/// Test 4: the 34-bit counter wraps at 2^34 us (4.77 h of uptime).
void test_counter_wrap(GpuHarness& gpu) {
    const int64_t t0 = (int64_t{1} << evt21::kLoopShift) - 150'000;
    const auto scene = predator::synth::make_scene<SceneEvent>(1e6, 0.3, t0, 31, kWidth, kHeight);
    const auto want  = to_records(scene);
    const auto words = encode(scene);
    const Splits sp  = random_splits(words.size(), gpu.max_words(), 32);
    const bool spans = !want.empty() && want.front().t < (int64_t{1} << 34) && want.back().t >= (int64_t{1} << 34);
    check(spans, "2^34 wrap: scene spans the wrap");
    check(same_events(openeb_decode(words, sp), want, "wrap openeb"), "2^34 wrap: OpenEB == input");
    check(same_events(gpu.decode(words, sp), want, "wrap gpu"), "2^34 wrap: GPU == input");
    const auto st = gpu.decoder().read_state();
    check(st.loop == 1 && st.base_time_set == 1 && st.total_events == want.size(), "2^34 wrap: carried loop == 1, totals");
}

/// Test 5: hand-built TIME_HIGH discrepancies (backward jumps, near-wrap, exact wrap, repeats).
void test_time_high_discrepancies(GpuHarness& gpu) {
    using evt21::make_cd_word;
    using evt21::make_time_high_word;
    const uint32_t M = evt21::kTimeHighMax;
    const std::vector<uint64_t> words = {
        make_cd_word(1, 1, 0, 0, 1u),                  // dropped: before the first TIME_HIGH
        make_time_high_word(1000),
        make_cd_word(1, 5, 32, 10, 0b101u),            // t = 1000<<6 | 5, x = 32, 34
        make_time_high_word(500),                      // backward: discrepancy, no loop
        make_cd_word(0, 7, 64, 11, 1u << 31),          // x = 95
        make_time_high_word(500),                      // repeat
        make_cd_word(1, 9, 0, 12, 0xFFFFFFFFu),        // 32 events
        make_cd_word(1, 2, 96, 5, 0u),                 // empty mask: no event
        make_time_high_word(M - 1),
        make_cd_word(0, 3, 128, 13, 0x00010001u),
        make_time_high_word(0),                        // jump of M-1 < M: discrepancy, no loop
        make_cd_word(1, 4, 160, 14, 0x2u),
        make_time_high_word(M),
        make_cd_word(1, 63, 192, 15, 0x4u),
        make_time_high_word(0),                        // exact wrap: loop 1
        make_cd_word(0, 0, 224, 16, 0x8u),
        make_time_high_word(3),
        make_cd_word(0, 63, 1248, 719, 0x80000001u),   // x = 1248, 1279
    };
    const size_t n = words.size();

    const auto ref = openeb_decode(words, {n});
    // Spot-check the reference itself so the test does not only compare two decoders.
    const int64_t wrap_t = (int64_t{1} << 34) | 0;
    const bool ref_ok    = ref.size() == 2 + 1 + 32 + 2 + 1 + 1 + 1 + 2 && ref[0].x == 32 && ref[0].t == ((1000 << 6) | 5) &&
                        ref[1].x == 34 && ref[2].x == 95 && ref[2].t == ((500 << 6) | 7) && ref[2].p == 0 &&
                        ref[37].t == ((int64_t{0} << 6) | 4) && ref[39].t == wrap_t && ref.back().x == 1279 &&
                        ref.back().t == (wrap_t | (3 << 6) | 63);
    check(ref_ok, "TIME_HIGH discrepancies: OpenEB reference has the expected values");

    bool all = same_events(gpu.decode(words, {n}), ref, "th single");
    const auto st = gpu.decoder().read_state();
    check(all && st.loop == 1 && st.high == 3 && st.base_time_set == 1, "TIME_HIGH discrepancies: GPU single batch + state");

    all = same_events(gpu.decode(words, Splits(n, 1)), ref, "th 1-word");
    for (size_t cut = 1; cut < n && all; ++cut) {
        all = same_events(gpu.decode(words, {cut, n - cut}), ref, "th cut " + std::to_string(cut));
    }
    check(all, "TIME_HIGH discrepancies: GPU 1-word batches and every 2-way split == OpenEB");
}

/// Test 6: stream control and API contracts.
void test_stream_control(GpuHarness& gpu) {
    const auto scene_a = predator::synth::make_scene<SceneEvent>(1e6, 0.1, 40'000'000, 41, kWidth, kHeight);
    const auto scene_b = predator::synth::make_scene<SceneEvent>(1e6, 0.1, 20'000'000, 42, kWidth, kHeight);
    const auto words_a = encode(scene_a);
    auto words_b       = make_garbage(25, 43);
    const size_t n_garbage = words_b.size();
    const auto body_b  = encode(scene_b);
    words_b.insert(words_b.end(), body_b.begin(), body_b.end());
    const Splits sp_a = random_splits(words_a.size(), gpu.max_words(), 44);
    const Splits sp_b = concat_splits({n_garbage}, random_splits(body_b.size(), gpu.max_words(), 45));

    // Reset: B decoded after A equals a fresh decode of B (garbage dropped again).
    gpu.decode(words_a, sp_a);
    check(same_events(gpu.decode(words_b, sp_b, true), to_records(scene_b), "reset"),
          "reset_async: stream restart drops pre-TIME_HIGH words again");

    // No reset: A then B equals OpenEB decoding the concatenation (garbage now takes A's last TIME_HIGH).
    std::vector<uint64_t> ab = words_a;
    ab.insert(ab.end(), words_b.begin(), words_b.end());
    const auto ref_ab = openeb_decode(ab, concat_splits(sp_a, sp_b));
    std::vector<CdRecord> got_ab = gpu.decode(words_a, sp_a, true);
    const auto got_b             = gpu.decode(words_b, sp_b, false);
    got_ab.insert(got_ab.end(), got_b.begin(), got_b.end());
    check(same_events(got_ab, ref_ab, "continue") && ref_ab.size() > scene_a.size() + scene_b.size(),
          "no reset: continuation == OpenEB on the concatenated stream");

    // Empty batch: zero events, state untouched.
    const auto before = gpu.decoder().read_state();
    std::vector<CdRecord> sink;
    const uint32_t c  = gpu.decode_batch(nullptr, 0, sink);
    const auto after  = gpu.decoder().read_state();
    check(c == 0 && sink.empty() && before.high == after.high && before.loop == after.loop &&
              before.total_events == after.total_events,
          "empty batch: 0 events, state unchanged");

    bool threw = false;
    try {
        gpu.decoder().decode_async(nullptr, gpu.max_words() + 1, nullptr, nullptr);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "oversize batch rejected with std::invalid_argument");

    int bad_ctor = 0;
    for (size_t bad : {size_t{0}, (size_t{1} << 26) + 1}) {
        try {
            GpuEvt21Decoder d(bad, gpu.stream());
        } catch (const std::invalid_argument&) {
            ++bad_ctor;
        }
    }
    check(bad_ctor == 2, "construction rejects max_words 0 and > 2^26");
    check(gpu.nonzero_pad() == 0, "decoder output pad field is always 0");
}

int run_synthetic() {
    std::printf("========================================================\n");
    std::printf("  Predator EVT2.1 GPU decoder parity suite (33.4b.b)\n");
    std::printf("========================================================\n");
    GpuHarness gpu(size_t{1} << 17);
    test_roundtrip(gpu, 1e5, 2.0, 0, 1);
    test_roundtrip(gpu, 1e6, 1.0, 7'654'321, 2);
    test_roundtrip(gpu, 1e7, 0.3, 123'456'789, 3);
    test_leading_garbage(gpu);
    test_injected_words(gpu);
    test_counter_wrap(gpu);
    test_time_high_discrepancies(gpu);
    test_stream_control(gpu);
    return g_failures;
}

// ---------------------------------------------------------------------------------------------------------------
// Fixture mode
// ---------------------------------------------------------------------------------------------------------------

double percentile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[static_cast<size_t>(q * static_cast<double>(v.size() - 1))];
}

int run_fixture(const std::string& prefix, size_t batch) {
    std::ifstream raw(prefix + ".evt21raw", std::ios::binary | std::ios::ate);
    std::ifstream ref(prefix + ".cd", std::ios::binary);
    if (!raw || !ref) {
        std::fprintf(stderr, "[ERROR] cannot open %s.evt21raw / %s.cd\n", prefix.c_str(), prefix.c_str());
        return 2;
    }
    const uint64_t raw_bytes = static_cast<uint64_t>(raw.tellg());
    raw.seekg(0);
    if (raw_bytes == 0 || raw_bytes % 8 != 0) {
        std::fprintf(stderr, "[ERROR] %s.evt21raw size %llu is not a positive multiple of 8\n", prefix.c_str(),
                     (unsigned long long)raw_bytes);
        return 2;
    }
    const uint64_t total_words = raw_bytes / 8;
    std::printf("[INFO] fixture %s: %llu words, batch %zu words\n", prefix.c_str(), (unsigned long long)total_words, batch);

    GpuHarness gpu(batch);
    std::vector<CdRecord> got, want;
    std::vector<double> host_us, gpu_us;
    uint64_t compared = 0;
    bool ok           = true;
    for (uint64_t done = 0; done < total_words && ok;) {
        const size_t n = static_cast<size_t>(std::min<uint64_t>(batch, total_words - done));
        raw.read(reinterpret_cast<char*>(gpu.mapped_words()), static_cast<std::streamsize>(n * 8));
        if (static_cast<size_t>(raw.gcount()) != n * 8) throw std::runtime_error("short read on .evt21raw");
        got.clear();
        BatchTiming tm;
        gpu.decode_mapped(n, got, &tm);
        host_us.push_back(tm.host_enqueue_us);
        gpu_us.push_back(tm.gpu_us);

        want.resize(got.size());
        ref.read(reinterpret_cast<char*>(want.data()), static_cast<std::streamsize>(want.size() * sizeof(CdRecord)));
        const size_t n_want = static_cast<size_t>(ref.gcount()) / sizeof(CdRecord);
        ok = same_events(got.data(), got.size(), want.data(), n_want, "fixture", compared);
        compared += std::min(got.size(), n_want);
        done += n;
    }
    if (ok && ref.peek() != std::char_traits<char>::eof()) {
        std::printf("  [fixture] reference has events beyond the GPU output (%llu compared)\n", (unsigned long long)compared);
        ok = false;
    }

    double gpu_total = 0.0, host_total = 0.0;
    for (double v : gpu_us) gpu_total += v;
    for (double v : host_us) host_total += v;
    std::printf("[INFO] %zu batches, %llu events compared\n", gpu_us.size(), (unsigned long long)compared);
    std::printf("[INFO] host enqueue us/batch p50/p99/max = %.1f / %.1f / %.1f  (total %.1f ms)\n",
                percentile(host_us, 0.5), percentile(host_us, 0.99), percentile(host_us, 1.0), host_total / 1000.0);
    std::printf("[INFO] GPU us/batch p50/p99/max = %.1f / %.1f / %.1f  (total %.1f ms, %.2f ns/event, %.0f Mev/s)\n",
                percentile(gpu_us, 0.5), percentile(gpu_us, 0.99), percentile(gpu_us, 1.0), gpu_total / 1000.0,
                compared ? 1000.0 * gpu_total / compared : 0.0, gpu_total > 0 ? compared / gpu_total : 0.0);
    check(ok && compared > 0 && gpu.nonzero_pad() == 0,
          "fixture: GPU decode == OpenEB reference (" + std::to_string(compared) + " events)");
    return g_failures;
}

void usage(const char* argv0) {
    std::fprintf(stderr, "usage: %s                                (synthetic parity suite)\n"
                         "       %s --fixture <prefix> [--batch N]  (live capture parity + timing)\n",
                 argv0, argv0);
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        if (argc == 1) return run_synthetic() == 0 ? 0 : 1;
        if (std::string(argv[1]) == "--fixture" && (argc == 3 || argc == 5)) {
            size_t batch = 16384;
            if (argc == 5) {
                if (std::string(argv[3]) != "--batch") {
                    usage(argv[0]);
                    return 2;
                }
                const unsigned long long b = std::strtoull(argv[4], nullptr, 10);
                if (b == 0 || b > (1ull << 26)) {
                    std::fprintf(stderr, "--batch must be in [1, 2^26]\n");
                    return 2;
                }
                batch = static_cast<size_t>(b);
            }
            return run_fixture(argv[2], batch) == 0 ? 0 : 1;
        }
        usage(argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ERROR] %s\n", e.what());
        return 2;
    }
}
