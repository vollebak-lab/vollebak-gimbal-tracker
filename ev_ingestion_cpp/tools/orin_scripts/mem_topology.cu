// Probe: reports whether the CUDA device has dedicated memory or shares system DRAM.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) { std::fprintf(stderr, "no device\n"); return 1; }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::printf("name=%s\nintegrated=%d\ncanMapHostMemory=%d\nunifiedAddressing=%d\n"
                "pageableMemoryAccess=%d\nconcurrentManagedAccess=%d\n"
                "totalGlobalMem_MB=%zu\ncudaMemGetInfo_total_MB=%zu\ncudaMemGetInfo_free_MB=%zu\n"
                "memoryBusWidth_bits=%d\n",
                p.name, p.integrated, p.canMapHostMemory, p.unifiedAddressing,
                p.pageableMemoryAccess, p.concurrentManagedAccess,
                p.totalGlobalMem >> 20, total_b >> 20, free_b >> 20, p.memoryBusWidth);
    return 0;
}
