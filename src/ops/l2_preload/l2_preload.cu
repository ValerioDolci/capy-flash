#include "ninfer/ops/l2_preload.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {

namespace {

constexpr int kThreads = 256;
constexpr int kUnroll  = 8; // 16-byte loads in flight per thread
static_assert(std::size_t(kThreads) * kUnroll * 16 == kL2PreloadBlockInFlightBytes);

struct Ranges {
    const uint4* data[kL2PreloadMaxRanges]     = {};
    std::uint64_t vectors[kL2PreloadMaxRanges] = {};
    int count                                  = 0;
};

// Worker w reads vectors w, w + workers, ... of every range, kUnroll at a time; the XOR of what
// it read keeps the loads alive and makes the thread wait for them. The launch passes a null
// `sink`, which the compiler cannot know: nothing is written.
__global__ void __launch_bounds__(kThreads) l2_preload_kernel(Ranges ranges, std::uint32_t* sink) {
    const std::uint64_t worker  = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t workers = std::uint64_t(gridDim.x) * blockDim.x;
    std::uint32_t folded        = 0;
    for (int r = 0; r < ranges.count; ++r) {
        const uint4* base       = ranges.data[r];
        const std::uint64_t end = ranges.vectors[r];
        for (std::uint64_t v = worker; v < end; v += workers * kUnroll) {
            uint4 loaded[kUnroll];
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) {
                const std::uint64_t i = v + std::uint64_t(u) * workers;
                loaded[u]             = i < end ? __ldcg(base + i) : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) {
                folded ^= loaded[u].x ^ loaded[u].y ^ loaded[u].z ^ loaded[u].w;
            }
        }
    }
    if (sink != nullptr) { sink[worker] = folded; }
}

} // namespace

void l2_preload(std::span<const L2PreloadRange> ranges, int blocks, cudaStream_t stream) {
    if (ranges.size() > static_cast<std::size_t>(kL2PreloadMaxRanges) || blocks <= 0) {
        throw std::invalid_argument("l2_preload: too many ranges or no blocks");
    }
    Ranges r;
    for (const L2PreloadRange& range : ranges) {
        const auto begin           = reinterpret_cast<std::uintptr_t>(range.data);
        const std::uintptr_t first = (begin + 15u) & ~std::uintptr_t{15};
        const std::uintptr_t last  = (begin + range.bytes) & ~std::uintptr_t{15};
        if (range.data == nullptr || last <= first) { continue; }
        r.data[r.count]    = reinterpret_cast<const uint4*>(first);
        r.vectors[r.count] = (last - first) / 16u;
        ++r.count;
    }
    if (r.count == 0) { return; }
    l2_preload_kernel<<<blocks, kThreads, 0, stream>>>(r, nullptr);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
