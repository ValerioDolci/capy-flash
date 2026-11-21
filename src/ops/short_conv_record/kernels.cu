// Implements: include/ninfer/ops/short_conv_record.h
// The output arithmetic is that of src/ops/short_conv/kernels.cu, term for term (one thread per
// (channel, token), FP32 accumulation in tap order, one BF16 rounding), so out has its bits; the
// same thread copies its input into the record, and the token-0 thread the three state columns.
// The fold copies, per entry, the three contiguous record columns n .. n + 2 over the tail.
#include "ops/short_conv_record/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/pdl_launch.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock       = 256;
constexpr int kHistory     = kShortConvTaps - 1;
constexpr int kFoldBlocksX = 8;

int grid_for(std::int64_t work) {
    return static_cast<int>(std::clamp<std::int64_t>((work + kBlock - 1) / kBlock, 1, 1 << 20));
}

// u[c, t - back] of the contract: x when the token is in this call, else a state column.
__device__ __forceinline__ float input_back(const __nv_bfloat16* __restrict__ x,
                                            const __nv_bfloat16* __restrict__ state,
                                            std::int64_t channels, std::int64_t c, std::int64_t t,
                                            int back) {
    const std::int64_t s = t - back;
    if (s >= 0) { return __bfloat162float(x[s * channels + c]); }
    return __bfloat162float(state[(kHistory + s) * channels + c]);
}

// record[c, 3 + t] = x[c, t]; at t = 0 also record[c, k] = state[c, k].
__device__ __forceinline__ void record_input(const __nv_bfloat16* __restrict__ x,
                                             const __nv_bfloat16* __restrict__ state,
                                             __nv_bfloat16* __restrict__ record,
                                             std::int64_t channels, std::int64_t c, std::int64_t t,
                                             std::int64_t i) {
    record[(kHistory + t) * channels + c] = x[i];
    if (t == 0) {
#pragma unroll
        for (int k = 0; k < kHistory; ++k) { record[k * channels + c] = state[k * channels + c]; }
    }
}

__global__ void short_conv_static_record_kernel(const __nv_bfloat16* __restrict__ x,
                                                const float* __restrict__ weight, float scale,
                                                const __nv_bfloat16* __restrict__ state,
                                                __nv_bfloat16* __restrict__ record,
                                                __nv_bfloat16* __restrict__ out,
                                                std::int64_t channels, std::int64_t tokens) {
    pdl_enter();
    const std::int64_t total = channels * tokens;
    const std::int64_t step  = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += step) {
        const std::int64_t t = i / channels;
        const std::int64_t c = i - t * channels;
        float acc            = 0.0F;
#pragma unroll
        for (int j = 0; j < kShortConvTaps; ++j) {
            acc += weight[c * kShortConvTaps + j] *
                   input_back(x, state, channels, c, t, kShortConvTaps - 1 - j);
        }
        out[i] = __float2bfloat16_rn((acc + __bfloat162float(x[i])) * scale);
        record_input(x, state, record, channels, c, t, i);
    }
}

__global__ void short_conv_dynamic_record_kernel(
    const __nv_bfloat16* __restrict__ x, const float* __restrict__ static_weight,
    const __nv_bfloat16* __restrict__ dyn, float scale, const __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ record, __nv_bfloat16* __restrict__ out, std::int64_t channels,
    std::int64_t tokens, std::int64_t group_width, std::int64_t dyn_rows) {
    pdl_enter();
    const std::int64_t total = channels * tokens;
    const std::int64_t step  = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += step) {
        const std::int64_t t      = i / channels;
        const std::int64_t c      = i - t * channels;
        const __nv_bfloat16* taps = dyn + t * dyn_rows + (c / group_width) * kShortConvTaps;
        float acc                 = 0.0F;
#pragma unroll
        for (int lag = 0; lag < kShortConvTaps; ++lag) {
            const float tap = static_weight[lag * channels + c] + __bfloat162float(taps[lag]);
            acc += tap * input_back(x, state, channels, c, t, lag);
        }
        out[i] = __float2bfloat16_rn((acc + __bfloat162float(x[i])) * scale);
        record_input(x, state, record, channels, c, t, i);
    }
}

// Block (bx, e): entry e, a grid-stride share of its 3 C tail elements; 16-byte words when both
// ends are 16-byte aligned (C % 8 == 0 keeps the record offset n C aligned), else BF16 elements.
__global__ void conv_tail_fold_kernel(const ConvTailFoldEntry* __restrict__ entries,
                                      std::int32_t width,
                                      const std::int32_t* __restrict__ committed) {
    pdl_enter();
    const ConvTailFoldEntry entry = entries[blockIdx.y];
    const std::int64_t n          = min(max(committed[0], 0), width);
    const std::int64_t C          = entry.channels;
    const auto* source            = static_cast<const __nv_bfloat16*>(entry.record) + n * C;
    auto* target                  = static_cast<__nv_bfloat16*>(entry.tail);
    const std::int64_t first = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t step  = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const auto ends =
        reinterpret_cast<std::uintptr_t>(source) | reinterpret_cast<std::uintptr_t>(target);
    if (C % 8 == 0 && ends % 16 == 0) {
        const auto* from         = reinterpret_cast<const uint4*>(source);
        auto* to                 = reinterpret_cast<uint4*>(target);
        const std::int64_t words = kHistory * C / 8;
        for (std::int64_t w = first; w < words; w += step) { to[w] = from[w]; }
    } else {
        const std::int64_t elements = kHistory * C;
        for (std::int64_t e = first; e < elements; e += step) { target[e] = source[e]; }
    }
}

} // namespace

void short_conv_record_launch(const Tensor& x, const Tensor& weight, const Tensor* dyn, float scale,
                              const Tensor& state, Tensor& record, Tensor& out,
                              cudaStream_t stream) {
    const std::int64_t channels = x.ne[0];
    const std::int64_t tokens   = x.ne[1];
    const auto* xp              = static_cast<const __nv_bfloat16*>(x.data);
    const auto* sp              = static_cast<const __nv_bfloat16*>(state.data);
    auto* rp                    = static_cast<__nv_bfloat16*>(record.data);
    auto* op                    = static_cast<__nv_bfloat16*>(out.data);
    const int grid              = grid_for(channels * tokens);
    if (dyn == nullptr) {
        launch_maybe_dependent(tokens <= kDependentLaunchMaxTokens, dim3(grid), dim3(kBlock), 0,
                               stream, short_conv_static_record_kernel, xp,
                               static_cast<const float*>(weight.data), scale, sp, rp, op, channels,
                               tokens);
    } else {
        const std::int64_t dyn_rows    = dyn->ne[0];
        const std::int64_t group_width = channels / (dyn_rows / kShortConvTaps);
        launch_maybe_dependent(tokens <= kDependentLaunchMaxTokens, dim3(grid), dim3(kBlock), 0,
                               stream, short_conv_dynamic_record_kernel, xp,
                               static_cast<const float*>(weight.data),
                               static_cast<const __nv_bfloat16*>(dyn->data), scale, sp, rp, op,
                               channels, tokens, group_width, dyn_rows);
    }
    CUDA_CHECK(cudaGetLastError());
}

void conv_tail_fold_launch(const ConvTailFoldEntry* entries, std::int32_t count, std::int32_t width,
                           const std::int32_t* committed, cudaStream_t stream) {
    const dim3 grid(kFoldBlocksX, static_cast<unsigned>(count));
    launch_maybe_dependent(true, grid, dim3(kBlock), 0, stream, conv_tail_fold_kernel, entries,
                           width, committed);
}

} // namespace ninfer::ops::detail
