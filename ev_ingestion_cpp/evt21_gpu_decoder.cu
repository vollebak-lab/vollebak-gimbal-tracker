/**
 * @file evt21_gpu_decoder.cu
 * @brief Implementation of the scan-based EVT 2.1 GPU decoder (see evt21_gpu_decoder.cuh).
 */

#include "evt21_gpu_decoder.cuh"

#include <stdexcept>
#include <string>

#include <cub/device/device_scan.cuh>

#include "evt21_format.hpp"

namespace predator {

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("GpuEvt21Decoder: ") + what + ": " + cudaGetErrorString(e));
    }
}

constexpr int kThreads = 256;

/// Largest batch: int32 word indices and packed 32-bit event counts (<= 32 events/word) must not overflow.
constexpr size_t kMaxWordsLimit = size_t{1} << 26;

inline unsigned blocks_for(size_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }

/// Step 1: index of every TIME_HIGH word, -1 elsewhere (max-scanned into "last TIME_HIGH at or before i").
__global__ void k_time_high_key(const uint64_t* __restrict__ words, int32_t n, int32_t* __restrict__ key) {
    const int32_t i = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    key[i] = evt21::word_type(words[i]) == evt21::kTimeHigh ? i : -1;
}

/// A CD word is decoded only once the stream has a base time (OpenEB drops earlier words).
__device__ __forceinline__ bool cd_decodable(const Evt21DecoderState& st, int32_t last_th) {
    return st.base_time_set != 0u || last_th >= 0;
}

/// Step 2: per-word event count (low 32 bits) and exact-wrap flag (high 32 bits), packed for one sum-scan.
__global__ void k_count_wrap(const uint64_t* __restrict__ words, int32_t n, const int32_t* __restrict__ last_th,
                             const Evt21DecoderState* __restrict__ state, uint64_t* __restrict__ count_wrap) {
    const int32_t i = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const Evt21DecoderState st = *state;
    const uint64_t w           = words[i];
    const uint32_t type        = evt21::word_type(w);
    uint64_t v                 = 0;
    if (type <= evt21::kCdOn) {
        if (cd_decodable(st, last_th[i])) v = static_cast<uint64_t>(__popc(evt21::cd_mask(w)));
    } else if (type == evt21::kTimeHigh) {
        // Previous high: last TIME_HIGH strictly before i in this batch, else the carried value. OpenEB counts
        // a loop only when the jump back is >= 2^28 - 1, i.e. exactly max -> 0; other regressions just replace it.
        const int32_t prev_idx = i > 0 ? last_th[i - 1] : -1;
        const uint32_t prev    = prev_idx >= 0 ? evt21::time_high(words[prev_idx]) : st.high;
        const uint32_t h       = evt21::time_high(w);
        if (h < prev && prev - h >= evt21::kTimeHighMax) v = uint64_t{1} << 32;
    }
    count_wrap[i] = v;
}

/// Step 3: expand every decodable CD word into its events at the scanned output offset.
__global__ void k_emit(const uint64_t* __restrict__ words, int32_t n, const int32_t* __restrict__ last_th,
                       const uint64_t* __restrict__ incl, const Evt21DecoderState* __restrict__ state,
                       CudaRawEvent* __restrict__ out) {
    const int32_t i = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const uint64_t w = words[i];
    if (evt21::word_type(w) > evt21::kCdOn) return;
    const Evt21DecoderState st = *state;
    const int32_t th           = last_th[i];
    if (!cd_decodable(st, th)) return;
    uint32_t mask = evt21::cd_mask(w);
    if (mask == 0u) return;

    const uint64_t inc   = incl[i];
    uint32_t offset      = static_cast<uint32_t>(inc) - static_cast<uint32_t>(__popc(mask));
    const uint64_t loop  = st.loop + (inc >> 32);  // CD words carry no wrap flag of their own
    const uint32_t high  = th >= 0 ? evt21::time_high(words[th]) : st.high;
    const uint64_t t     = (loop << evt21::kLoopShift) | (static_cast<uint64_t>(high) << evt21::kTimeLowBits) |
                       evt21::ts_low(w);
    const uint32_t x0    = evt21::cd_x_base(w);
    const uint16_t y     = static_cast<uint16_t>(evt21::cd_y(w));
    const int16_t p      = static_cast<int16_t>(evt21::cd_polarity(w));
    while (mask) {
        const uint32_t bit = static_cast<uint32_t>(__ffs(mask)) - 1u;  // ascending pixel order, as OpenEB
        mask &= mask - 1u;
        CudaRawEvent e;
        e.x   = static_cast<uint16_t>(x0 + bit);
        e.y   = y;
        e.p   = p;
        e.pad = 0;
        e.t   = t;
        out[offset++] = e;
    }
}

/// Step 4: publish the batch count and advance the carried state (single thread, stream-ordered).
__global__ void k_finalize(const uint64_t* __restrict__ words, int32_t n, const int32_t* __restrict__ last_th,
                           const uint64_t* __restrict__ incl, Evt21DecoderState* __restrict__ state,
                           uint32_t* __restrict__ out_count) {
    const uint64_t total = incl[n - 1];
    const uint32_t count = static_cast<uint32_t>(total);
    *out_count           = count;
    state->loop += total >> 32;
    state->total_events += count;
    const int32_t th = last_th[n - 1];
    if (th >= 0) {
        state->high          = evt21::time_high(words[th]);
        state->base_time_set = 1u;
    }
}

}  // namespace

GpuEvt21Decoder::GpuEvt21Decoder(size_t max_words_per_batch, cudaStream_t stream)
    : max_words_(max_words_per_batch), stream_(stream) {
    if (max_words_ == 0 || max_words_ > kMaxWordsLimit) {
        throw std::invalid_argument("GpuEvt21Decoder: max_words_per_batch must be in [1, 2^26]");
    }
    try {
        check(cudaMalloc(&d_th_key_, max_words_ * sizeof(int32_t)), "alloc th_key");
        check(cudaMalloc(&d_last_th_, max_words_ * sizeof(int32_t)), "alloc last_th");
        check(cudaMalloc(&d_count_wrap_, max_words_ * sizeof(uint64_t)), "alloc count_wrap");
        check(cudaMalloc(&d_count_wrap_incl_, max_words_ * sizeof(uint64_t)), "alloc count_wrap_incl");
        check(cudaMalloc(&d_state_, sizeof(Evt21DecoderState)), "alloc state");

        // One temp buffer sized for the larger of the two scans at the maximum batch.
        size_t max_bytes = 0, sum_bytes = 0;
        const int n = static_cast<int>(max_words_);
        check(cub::DeviceScan::InclusiveScan(nullptr, max_bytes, d_th_key_, d_last_th_, cub::Max(), n, stream_),
              "size max-scan");
        check(cub::DeviceScan::InclusiveSum(nullptr, sum_bytes, d_count_wrap_, d_count_wrap_incl_, n, stream_),
              "size sum-scan");
        scan_temp_bytes_ = max_bytes > sum_bytes ? max_bytes : sum_bytes;
        check(cudaMalloc(&d_scan_temp_, scan_temp_bytes_), "alloc scan temp");
        check(cudaMemsetAsync(d_state_, 0, sizeof(Evt21DecoderState), stream_), "init state");
        check(cudaStreamSynchronize(stream_), "init sync");
    } catch (...) {
        cudaFree(d_th_key_);
        cudaFree(d_last_th_);
        cudaFree(d_count_wrap_);
        cudaFree(d_count_wrap_incl_);
        cudaFree(d_state_);
        cudaFree(d_scan_temp_);
        throw;
    }
}

GpuEvt21Decoder::~GpuEvt21Decoder() {
    // Pending work may still reference these buffers.
    cudaStreamSynchronize(stream_);
    cudaFree(d_th_key_);
    cudaFree(d_last_th_);
    cudaFree(d_count_wrap_);
    cudaFree(d_count_wrap_incl_);
    cudaFree(d_state_);
    cudaFree(d_scan_temp_);
}

void GpuEvt21Decoder::decode_async(const uint64_t* d_words, size_t n, CudaRawEvent* d_out, uint32_t* d_out_count) {
    if (n > max_words_) throw std::invalid_argument("GpuEvt21Decoder: batch exceeds max_words_per_batch");
    if (n == 0) {
        check(cudaMemsetAsync(d_out_count, 0, sizeof(uint32_t), stream_), "zero count");
        return;
    }
    const int32_t ni = static_cast<int32_t>(n);
    const unsigned blocks = blocks_for(n);
    size_t temp = scan_temp_bytes_;

    k_time_high_key<<<blocks, kThreads, 0, stream_>>>(d_words, ni, d_th_key_);
    check(cudaGetLastError(), "k_time_high_key");
    check(cub::DeviceScan::InclusiveScan(d_scan_temp_, temp, d_th_key_, d_last_th_, cub::Max(), ni, stream_),
          "max-scan");
    k_count_wrap<<<blocks, kThreads, 0, stream_>>>(d_words, ni, d_last_th_, d_state_, d_count_wrap_);
    check(cudaGetLastError(), "k_count_wrap");
    temp = scan_temp_bytes_;
    check(cub::DeviceScan::InclusiveSum(d_scan_temp_, temp, d_count_wrap_, d_count_wrap_incl_, ni, stream_),
          "sum-scan");
    k_emit<<<blocks, kThreads, 0, stream_>>>(d_words, ni, d_last_th_, d_count_wrap_incl_, d_state_, d_out);
    check(cudaGetLastError(), "k_emit");
    k_finalize<<<1, 1, 0, stream_>>>(d_words, ni, d_last_th_, d_count_wrap_incl_, d_state_, d_out_count);
    check(cudaGetLastError(), "k_finalize");
}

void GpuEvt21Decoder::reset_async() {
    check(cudaMemsetAsync(d_state_, 0, sizeof(Evt21DecoderState), stream_), "reset state");
}

Evt21DecoderState GpuEvt21Decoder::read_state() const {
    Evt21DecoderState s{};
    check(cudaMemcpyAsync(&s, d_state_, sizeof(s), cudaMemcpyDeviceToHost, stream_), "read state");
    check(cudaStreamSynchronize(stream_), "read state sync");
    return s;
}

}  // namespace predator
