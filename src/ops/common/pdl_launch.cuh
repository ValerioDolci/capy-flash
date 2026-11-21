#pragma once

// Programmatic dependent launch (core/pdl.cuh) for kernels at decode and verification widths: the
// kernel may start while the kernel before it on the stream finishes. A kernel launched so calls
// pdl_enter() on every path before its first access to data an earlier kernel of the stream wrote
// (or will read): only constant data (weights) may be read before it.

#include "core/device.h"
#include "core/pdl.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace ninfer::ops::detail {

// Widest call launched as a dependent (decode and verification widths).
inline constexpr std::int64_t kDependentLaunchMaxTokens = 16;

// Waits for the stream's previous kernel (complete, its writes visible), then lets the next
// kernel of the stream launch.
__device__ __forceinline__ void pdl_enter() {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
}

// A dependent launch when `dependent`, else a plain one.
template <class... KernelArgs, class... CallArgs>
void launch_maybe_dependent(bool dependent, dim3 grid, dim3 block, std::size_t smem,
                            cudaStream_t stream, void (*kernel)(KernelArgs...),
                            CallArgs&&... args) {
    if (dependent) {
        CUDA_CHECK(pdl::launch_dependent(pdl::LaunchConfig{grid, block, smem, stream}, kernel,
                                         std::forward<CallArgs>(args)...));
        return;
    }
    kernel<<<grid, block, smem, stream>>>(std::forward<CallArgs>(args)...);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
