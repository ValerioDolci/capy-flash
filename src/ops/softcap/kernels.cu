// Implements: include/ninfer/ops/softcap.h
// Elementwise grid-stride kernel with IEEE division and the accurate tanhf.
#include "ops/softcap/launch.h"

#include "core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;

__global__ void softcap_kernel(const __nv_bfloat16* __restrict__ x, float cap,
                               float* __restrict__ out, std::int64_t n) {
    const std::int64_t step = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += step) {
        out[i] = cap * tanhf(__fdiv_rn(__bfloat162float(x[i]), cap));
    }
}

} // namespace

void softcap_launch(const Tensor& x, float cap, Tensor& out, cudaStream_t stream) {
    const std::int64_t n = x.numel();
    const int grid =
        static_cast<int>(std::clamp<std::int64_t>((n + kBlock - 1) / kBlock, 1, 1 << 20));
    softcap_kernel<<<grid, kBlock, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data), cap,
                                                static_cast<float*>(out.data), n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
