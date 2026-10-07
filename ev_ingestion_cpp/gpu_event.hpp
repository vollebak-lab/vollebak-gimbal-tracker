#pragma once
/**
 * @file gpu_event.hpp
 * @brief GPU-resident CD event record shared by the EVT2.1 GPU decoder and CudaFlickerCore.
 *
 * Layout is fixed (16 B, 16-byte aligned) so decoded events can be consumed in place by the sieve and
 * accumulation kernels without repacking. `pad` carries the 2x2 periodicity-sieve hit count once the
 * sieve has run (0 straight out of the decoder).
 */

#include <cstdint>

namespace predator {

/**
 * @brief Raw event struct compatible with CUDA memory layout
 */
struct alignas(16) CudaRawEvent {
    uint16_t x;
    uint16_t y;
    int16_t p;
    int16_t pad;
    uint64_t t;
};
static_assert(sizeof(CudaRawEvent) == 16, "CudaRawEvent must stay 16 bytes");

}  // namespace predator
