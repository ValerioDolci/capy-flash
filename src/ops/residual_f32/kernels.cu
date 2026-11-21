// Implements: include/ninfer/ops/residual_f32.h
// Elementwise grid-stride kernels; __fadd_rn pins the one FP32 addition of the contract.
#include "ops/residual_f32/launch.h"

#include "core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;

int grid_for(std::int64_t work) {
    return static_cast<int>(std::clamp<std::int64_t>((work + kBlock - 1) / kBlock, 1, 1 << 20));
}

__global__ void residual_add_f32_kernel(const __nv_bfloat16* __restrict__ x,
                                        const __nv_bfloat16* __restrict__ y,
                                        float* __restrict__ sum, std::int64_t n) {
    const std::int64_t step = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += step) {
        const float yv = __bfloat162float(y[i]);
        sum[i]         = x == nullptr ? yv : __fadd_rn(__bfloat162float(x[i]), yv);
    }
}

// `z` and `out` may be the same storage: each element is read before it is written.
__global__ void residual_add_round_bf16_kernel(const float* __restrict__ sum,
                                               const __nv_bfloat16* z, __nv_bfloat16* out,
                                               std::int64_t n) {
    const std::int64_t step = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += step) {
        out[i] = __float2bfloat16_rn(__fadd_rn(sum[i], __bfloat162float(z[i])));
    }
}

} // namespace

void residual_add_f32_launch(const Tensor* x, const Tensor& y, Tensor& sum, cudaStream_t stream) {
    const std::int64_t n = y.numel();
    residual_add_f32_kernel<<<grid_for(n), kBlock, 0, stream>>>(
        x == nullptr ? nullptr : static_cast<const __nv_bfloat16*>(x->data),
        static_cast<const __nv_bfloat16*>(y.data), static_cast<float*>(sum.data), n);
    CUDA_CHECK(cudaGetLastError());
}

void residual_add_round_bf16_launch(const Tensor& sum, const Tensor& z, Tensor& out,
                                    cudaStream_t stream) {
    const std::int64_t n = z.numel();
    residual_add_round_bf16_kernel<<<grid_for(n), kBlock, 0, stream>>>(
        static_cast<const float*>(sum.data), static_cast<const __nv_bfloat16*>(z.data),
        static_cast<__nv_bfloat16*>(out.data), n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
