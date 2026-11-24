// Implements: include/ninfer/ops/short_conv.h
// Two SIMT routes with one arithmetic (FP32 accumulation in tap order, one BF16 rounding):
//   T <= kChannelRouteTokens: one launch, one thread per channel walking the tokens;
//   longer: one thread per (channel, token) output, then one thread per channel for the state
//   transition. The state kernel runs after the output kernel on the same stream, so an in-place
//   state update never feeds the outputs.
#include "ops/short_conv/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ninfer/ops/short_conv.h"
#include "ops/common/pdl_launch.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock   = 256;
constexpr int kHistory = kShortConvTaps - 1;
// Extents up to this many tokens take the one-launch channel route (a single decode token: longer
// extents measured no faster).
constexpr std::int64_t kChannelRouteTokens = 1;

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

__global__ void short_conv_static_kernel(const __nv_bfloat16* __restrict__ x,
                                         const float* __restrict__ weight, float scale,
                                         const __nv_bfloat16* __restrict__ state,
                                         __nv_bfloat16* __restrict__ out, std::int64_t channels,
                                         std::int64_t tokens) {
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
    }
}

__global__ void short_conv_dynamic_kernel(const __nv_bfloat16* __restrict__ x,
                                          const float* __restrict__ static_weight,
                                          const __nv_bfloat16* __restrict__ dyn, float scale,
                                          const __nv_bfloat16* __restrict__ state,
                                          __nv_bfloat16* __restrict__ out, std::int64_t channels,
                                          std::int64_t tokens, std::int64_t group_width,
                                          std::int64_t dyn_rows) {
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
    }
}

// state_out column k = u[c, T - (3 - k)]; every column is read before any is written.
__global__ void short_conv_state_kernel(const __nv_bfloat16* __restrict__ x,
                                        const __nv_bfloat16* state_in, __nv_bfloat16* state_out,
                                        std::int64_t channels, std::int64_t tokens) {
    const std::int64_t step = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t c = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         c < channels; c += step) {
        __nv_bfloat16 column[kHistory];
#pragma unroll
        for (int k = 0; k < kHistory; ++k) {
            const std::int64_t s = tokens - (kHistory - k);
            column[k] = s >= 0 ? x[s * channels + c] : state_in[(kHistory + s) * channels + c];
        }
#pragma unroll
        for (int k = 0; k < kHistory; ++k) { state_out[k * channels + c] = column[k]; }
    }
}

// Short extents: one thread per channel walks the tokens with the window in registers and writes
// the state at the end, in one launch. Same expressions and order as the two kernels above, so the
// same bits; the state is read and written by the thread that owns the channel, so in place is
// safe.
// The operands of one call on the channel route.
struct ChannelCall {
    const __nv_bfloat16* x;
    const float* weight;
    const __nv_bfloat16* dyn;
    float scale;
    const __nv_bfloat16* state_in;
    __nv_bfloat16* state_out;
    __nv_bfloat16* out;
    std::int64_t channels, group_width, dyn_rows;
};

// Up to two calls (short_conv_pair): call blockIdx.y.
struct ChannelCalls {
    ChannelCall call[2];
};

template <bool kDynamic>
__global__ void short_conv_channel_kernel(ChannelCalls calls, std::int64_t tokens) {
    const ChannelCall& a = calls.call[blockIdx.y];
    const std::int64_t c = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= a.channels) { return; }
    pdl_enter();
    const std::int64_t channels = a.channels;
    // window[k] = the input 3 - k tokens back.
    __nv_bfloat16 window[kHistory];
#pragma unroll
    for (int k = 0; k < kHistory; ++k) { window[k] = a.state_in[k * channels + c]; }
    for (std::int64_t t = 0; t < tokens; ++t) {
        const __nv_bfloat16 current = a.x[t * channels + c];
        float acc                   = 0.0F;
        if constexpr (kDynamic) {
            const __nv_bfloat16* taps =
                a.dyn + t * a.dyn_rows + (c / a.group_width) * kShortConvTaps;
#pragma unroll
            for (int lag = 0; lag < kShortConvTaps; ++lag) {
                const float tap = a.weight[lag * channels + c] + __bfloat162float(taps[lag]);
                acc += tap * __bfloat162float(lag == 0 ? current : window[kHistory - lag]);
            }
        } else {
#pragma unroll
            for (int j = 0; j < kShortConvTaps; ++j) {
                acc += a.weight[c * kShortConvTaps + j] *
                       __bfloat162float(j == kHistory ? current : window[j]);
            }
        }
        a.out[t * channels + c] = __float2bfloat16_rn((acc + __bfloat162float(current)) * a.scale);
#pragma unroll
        for (int k = 0; k + 1 < kHistory; ++k) { window[k] = window[k + 1]; }
        window[kHistory - 1] = current;
    }
#pragma unroll
    for (int k = 0; k < kHistory; ++k) { a.state_out[k * channels + c] = window[k]; }
}

ChannelCall channel_call(const Tensor& x, const Tensor& weight, const Tensor* dyn, float scale,
                         const Tensor& state_in, Tensor& state_out, Tensor& out) {
    const std::int64_t channels = x.ne[0];
    const auto rows             = dyn == nullptr ? std::int64_t{0} : std::int64_t{dyn->ne[0]};
    return {static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const float*>(weight.data),
            dyn == nullptr ? nullptr : static_cast<const __nv_bfloat16*>(dyn->data),
            scale,
            static_cast<const __nv_bfloat16*>(state_in.data),
            static_cast<__nv_bfloat16*>(state_out.data),
            static_cast<__nv_bfloat16*>(out.data),
            channels,
            dyn == nullptr ? channels : channels / (rows / kShortConvTaps),
            rows};
}

void launch_channel(const ChannelCalls& calls, int count, bool dynamic, std::int64_t tokens,
                    cudaStream_t stream) {
    const std::int64_t widest =
        std::max(calls.call[0].channels, count > 1 ? calls.call[1].channels : std::int64_t{0});
    const dim3 grid(static_cast<unsigned>(grid_for(widest)), static_cast<unsigned>(count));
    if (dynamic) {
        launch_maybe_dependent(true, grid, dim3(kBlock), 0, stream, short_conv_channel_kernel<true>,
                               calls, tokens);
    } else {
        launch_maybe_dependent(true, grid, dim3(kBlock), 0, stream,
                               short_conv_channel_kernel<false>, calls, tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void short_conv_launch(const Tensor& x, const Tensor& weight, const Tensor* dyn, float scale,
                       const Tensor& state_in, Tensor& state_out, Tensor& out,
                       cudaStream_t stream) {
    const std::int64_t channels = x.ne[0];
    const std::int64_t tokens   = x.ne[1];
    const auto* xp              = static_cast<const __nv_bfloat16*>(x.data);
    const auto* sp              = static_cast<const __nv_bfloat16*>(state_in.data);
    auto* op                    = static_cast<__nv_bfloat16*>(out.data);
    const int grid              = grid_for(channels * tokens);
    if (tokens <= kChannelRouteTokens) {
        ChannelCalls calls{};
        calls.call[0] = channel_call(x, weight, dyn, scale, state_in, state_out, out);
        launch_channel(calls, 1, dyn != nullptr, tokens, stream);
        return;
    }
    if (dyn == nullptr) {
        short_conv_static_kernel<<<grid, kBlock, 0, stream>>>(
            xp, static_cast<const float*>(weight.data), scale, sp, op, channels, tokens);
    } else {
        const std::int64_t dyn_rows    = dyn->ne[0];
        const std::int64_t group_width = channels / (dyn_rows / kShortConvTaps);
        short_conv_dynamic_kernel<<<grid, kBlock, 0, stream>>>(
            xp, static_cast<const float*>(weight.data),
            static_cast<const __nv_bfloat16*>(dyn->data), scale, sp, op, channels, tokens,
            group_width, dyn_rows);
    }
    CUDA_CHECK(cudaGetLastError());
    short_conv_state_kernel<<<grid_for(channels), kBlock, 0, stream>>>(
        xp, sp, static_cast<__nv_bfloat16*>(state_out.data), channels, tokens);
    CUDA_CHECK(cudaGetLastError());
}

bool short_conv_pair_launch(const Tensor& x0, const Tensor& w0, const Tensor* dyn0, float scale0,
                            const Tensor& in0, Tensor& state0, Tensor& out0, const Tensor& x1,
                            const Tensor& w1, const Tensor* dyn1, float scale1, const Tensor& in1,
                            Tensor& state1, Tensor& out1, cudaStream_t stream) {
    const std::int64_t tokens = x0.ne[1];
    if (tokens > kChannelRouteTokens) { return false; }
    ChannelCalls calls{};
    calls.call[0] = channel_call(x0, w0, dyn0, scale0, in0, state0, out0);
    calls.call[1] = channel_call(x1, w1, dyn1, scale1, in1, state1, out1);
    launch_channel(calls, 2, dyn0 != nullptr, tokens, stream);
    return true;
}

} // namespace ninfer::ops::detail
