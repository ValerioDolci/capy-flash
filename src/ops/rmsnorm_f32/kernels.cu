// Implements: include/ninfer/ops/rmsnorm_f32.h
// Naive SIMT route: one CTA per row, a fixed-order FP32 block sum, one BF16 rounding per element.
#include "ops/rmsnorm_f32/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/pdl_launch.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;
constexpr int kWarps = kBlock / 32;

__global__ void __launch_bounds__(kBlock)
    rmsnorm_f32_kernel(const float* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
                       float eps, __nv_bfloat16* __restrict__ out, int width) {
    __shared__ float scratch[kWarps];
    pdl_enter();
    const std::int64_t base = static_cast<std::int64_t>(blockIdx.x) * width;
    float squares           = 0.0F;
    for (int i = static_cast<int>(threadIdx.x); i < width; i += kBlock) {
        squares += x[base + i] * x[base + i];
    }
    squares        = warp_sum(squares);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) { scratch[warp] = squares; }
    __syncthreads();
    float total = 0.0F;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) { total += scratch[w]; }
    const float inv = rsqrtf(total / static_cast<float>(width) + eps);
    for (int i = static_cast<int>(threadIdx.x); i < width; i += kBlock) {
        out[base + i] = __float2bfloat16_rn(x[base + i] * inv * __bfloat162float(weight[i]));
    }
}

} // namespace

void rmsnorm_f32_launch(const Tensor& x, const Tensor& weight, float eps, Tensor& out,
                        cudaStream_t stream) {
    const int width = x.ne[0];
    const auto rows = static_cast<unsigned>(x.numel() / width);
    launch_maybe_dependent(rows <= kDependentLaunchMaxTokens, dim3(rows), dim3(kBlock), 0, stream,
                           rmsnorm_f32_kernel, static_cast<const float*>(x.data),
                           static_cast<const __nv_bfloat16*>(weight.data), eps,
                           static_cast<__nv_bfloat16*>(out.data), width);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
