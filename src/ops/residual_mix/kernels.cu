// Implements: include/ninfer/ops/residual_mix.h
// One CTA of 256 threads per token, element h on thread h mod 256 (the mapping of depth_attention,
// rmsnorm_f32 and of every elementwise Op it replaces): the short convolution and the residual
// addition are elementwise; the depth attention and the RMS norms copy the block sums of
// depth_attention/kernels.cu and rmsnorm_f32/kernels.cu statement for statement, so every value is
// bit-identical to the separate Ops. The current state the mix attends to is kept in registers.
#include "ops/residual_mix/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ninfer/ops/depth_attention.h"
#include "ninfer/ops/residual_mix.h"
#include "ninfer/ops/short_conv.h"
#include "ops/common/pdl_launch.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock      = 256;
constexpr int kWarps      = kBlock / 32;
constexpr int kPer        = kResidualMixMaxHidden / kBlock;
constexpr int kHistory    = kShortConvTaps - 1;
constexpr int kMaxSources = kDepthAttentionMaxStates + 1;
constexpr int kMaxQueries = kDepthAttentionMaxQueries;

// The inputs of channel c, token t of the convolution: in[j] = u[c, t - (3 - j)] (short_conv.h: x
// in this call, else a state column) and the taps. Loaded before any store of the call so that
// every load of a thread is in flight at once.
struct ConvLoads {
    __nv_bfloat16 in[kShortConvTaps];
    float weight[kShortConvTaps];
};

__device__ __forceinline__ void conv_load(const ResidualConvArgs& a, int hidden, int c, int t,
                                          ConvLoads& l) {
#pragma unroll
    for (int j = 0; j < kShortConvTaps; ++j) {
        const int s = t - (kShortConvTaps - 1 - j);
        l.in[j]     = s >= 0 ? a.x[static_cast<std::int64_t>(s) * hidden + c]
                             : a.state[static_cast<std::int64_t>(kHistory + s) * hidden + c];
        l.weight[j] = a.weight[c * kShortConvTaps + j];
    }
}

// The BF16 convolution output (short_conv_static_kernel's expressions) as FP32.
__device__ __forceinline__ float conv_value(const ResidualConvArgs& a, const ConvLoads& l) {
    float acc = 0.0F;
#pragma unroll
    for (int j = 0; j < kShortConvTaps; ++j) { acc += l.weight[j] * __bfloat162float(l.in[j]); }
    const __nv_bfloat16 out =
        __float2bfloat16_rn((acc + __bfloat162float(l.in[kShortConvTaps - 1])) * a.scale);
    return __bfloat162float(out);
}

// The channel's share of the record (its input; at t = 0 also the state columns, which are then
// in[0..2]) or of the in-place state transition (one token: the state is in[0..2], the new one
// in[1..3]).
__device__ __forceinline__ void conv_store(const ResidualConvArgs& a, int hidden, int c, int t,
                                           const ConvLoads& l) {
    if (a.record != nullptr) {
        a.record[static_cast<std::int64_t>(kHistory + t) * hidden + c] = l.in[kShortConvTaps - 1];
        if (t == 0) {
#pragma unroll
            for (int k = 0; k < kHistory; ++k) {
                a.record[static_cast<std::int64_t>(k) * hidden + c] = l.in[k];
            }
        }
    } else {
#pragma unroll
        for (int k = 0; k < kHistory; ++k) {
            a.state[static_cast<std::int64_t>(k) * hidden + c] = l.in[k + 1];
        }
    }
}

// depth_attention/kernels.cu block_sums.
template <int kMax>
__device__ __forceinline__ void block_sums(const float (&values)[kMax], int count,
                                           float (*partial)[kWarps], float* totals) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
    for (int j = 0; j < kMax; ++j) {
        if (j < count) {
            const float v = warp_sum(values[j]);
            if (lane == 0) { partial[j][warp] = v; }
        }
    }
    __syncthreads();
    for (int j = warp; j < count; j += kWarps) {
        float total = lane < kWarps ? partial[j][lane] : 0.0F;
        total       = warp_sum(total);
        if (lane == 0) { totals[j] = total; }
    }
    __syncthreads();
}

// v[k][s] = source s at h = tid + kBlock k (s < kSources - 1: a stored state; the last: current,
// from `cur` when kCurrentInRegisters, else from the BF16 current), 0 out of range.
template <int kSources, bool kCurrentInRegisters>
__device__ __forceinline__ void load_sources(const MixArgs& m, std::int64_t base, int hidden,
                                             const float (&cur)[kPer], float (&v)[kPer][kSources]) {
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        const int h = static_cast<int>(threadIdx.x) + kBlock * k;
#pragma unroll
        for (int s = 0; s < kSources - 1; ++s) {
            v[k][s] = h < hidden ? __bfloat162float(m.state[s][base + h]) : 0.0F;
        }
        if constexpr (kCurrentInRegisters) {
            v[k][kSources - 1] = h < hidden ? cur[k] : 0.0F;
        } else {
            v[k][kSources - 1] = h < hidden ? __bfloat162float(m.current[base + h]) : 0.0F;
        }
    }
}

// The depth attention of token t over kSources sources (depth_attention_kernel's three passes),
// mixed[q][k] its output at h = tid + kBlock k; then the RMS norms of the first m.norms queries.
template <int kSources, bool kCurrentInRegisters>
__device__ __forceinline__ void mix_and_norm(const MixArgs& m, int t, int hidden, int tokens,
                                             const float (&cur)[kPer]) {
    constexpr int kMaxDots = kMaxQueries * kSources;
    __shared__ float partial[kMaxDots][kWarps];
    __shared__ float totals[kMaxDots];
    __shared__ float inv[kMaxSources];
    __shared__ float weight[kMaxQueries][kMaxSources];
    __shared__ float norm_partial[kMaxQueries][kWarps];
    const std::int64_t base = static_cast<std::int64_t>(t) * hidden;
    const int tid           = static_cast<int>(threadIdx.x);
    const int queries       = m.queries;
    float v[kPer][kSources];

    float squares[kSources];
#pragma unroll
    for (int s = 0; s < kSources; ++s) { squares[s] = 0.0F; }
    load_sources<kSources, kCurrentInRegisters>(m, base, hidden, cur, v);
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        if (tid + kBlock * k < hidden) {
#pragma unroll
            for (int s = 0; s < kSources; ++s) { squares[s] += v[k][s] * v[k][s]; }
        }
    }
    block_sums(squares, kSources, partial, totals);
    if (tid < kSources) { inv[tid] = rsqrtf(totals[tid] / static_cast<float>(hidden) + m.eps); }
    __syncthreads();

    // Shared values in registers: the global stores below could alias shared memory otherwise.
    float inv_r[kSources];
#pragma unroll
    for (int s = 0; s < kSources; ++s) { inv_r[s] = inv[s]; }
    float dots[kMaxDots];
#pragma unroll
    for (int j = 0; j < kMaxDots; ++j) { dots[j] = 0.0F; }
    load_sources<kSources, kCurrentInRegisters>(m, base, hidden, cur, v);
    {
        float qv[kPer][kMaxQueries];
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            const int h = tid + kBlock * k;
#pragma unroll
            for (int q = 0; q < kMaxQueries; ++q) {
                qv[k][q] = q < queries && h < hidden
                               ? m.query[static_cast<std::int64_t>(q) * hidden + h]
                               : 0.0F;
            }
        }
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            if (tid + kBlock * k < hidden) {
#pragma unroll
                for (int s = 0; s < kSources; ++s) {
                    const float nv = v[k][s] * inv_r[s];
#pragma unroll
                    for (int q = 0; q < kMaxQueries; ++q) {
                        if (q < queries) { dots[q * kSources + s] += qv[k][q] * nv; }
                    }
                }
            }
        }
    }
    block_sums(dots, queries * kSources, partial, totals);
    if (tid < queries * kSources) {
        weight[tid / kSources][tid % kSources] = totals[tid] * m.scale;
    }
    __syncthreads();
    if (tid < queries) {
        float* logits = weight[tid];
        float maximum = logits[0];
        for (int s = 1; s < kSources; ++s) { maximum = fmaxf(maximum, logits[s]); }
        float total = 0.0F;
        for (int s = 0; s < kSources; ++s) {
            logits[s] = expf(logits[s] - maximum);
            total += logits[s];
        }
        for (int s = 0; s < kSources; ++s) { logits[s] /= total; }
    }
    __syncthreads();

    float w_r[kMaxQueries][kSources];
#pragma unroll
    for (int q = 0; q < kMaxQueries; ++q) {
#pragma unroll
        for (int s = 0; s < kSources; ++s) { w_r[q][s] = q < queries ? weight[q][s] : 0.0F; }
    }
    float mixed[kMaxQueries][kPer];
    load_sources<kSources, kCurrentInRegisters>(m, base, hidden, cur, v);
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        const int h = tid + kBlock * k;
        float normalized[kSources];
#pragma unroll
        for (int s = 0; s < kSources; ++s) { normalized[s] = v[k][s] * inv_r[s]; }
#pragma unroll
        for (int q = 0; q < kMaxQueries; ++q) {
            mixed[q][k] = 0.0F;
            if (q < queries && h < hidden) {
                float sum = 0.0F;
#pragma unroll
                for (int s = 0; s < kSources; ++s) { sum += w_r[q][s] * normalized[s]; }
                mixed[q][k] = sum;
                if (m.mix != nullptr) {
                    m.mix[(static_cast<std::int64_t>(q) * tokens + t) * hidden + h] = sum;
                }
            }
        }
    }
    if (m.norms == 0) { return; }

    // rmsnorm_f32_kernel per normed query; the gains loaded before the first store.
    const int lane = tid & 31;
    const int warp = tid >> 5;
    __nv_bfloat16 gains[kMaxQueries][kPer];
#pragma unroll
    for (int q = 0; q < kMaxQueries; ++q) {
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            const int h = tid + kBlock * k;
            if (q < m.norms && h < hidden) { gains[q][k] = m.norm[q][h]; }
        }
    }
#pragma unroll
    for (int q = 0; q < kMaxQueries; ++q) {
        if (q < m.norms) {
            float sq = 0.0F;
#pragma unroll
            for (int k = 0; k < kPer; ++k) {
                if (tid + kBlock * k < hidden) { sq += mixed[q][k] * mixed[q][k]; }
            }
            sq = warp_sum(sq);
            if (lane == 0) { norm_partial[q][warp] = sq; }
        }
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < kMaxQueries; ++q) {
        if (q < m.norms) {
            float total = 0.0F;
#pragma unroll
            for (int w = 0; w < kWarps; ++w) { total += norm_partial[q][w]; }
            const float n = rsqrtf(total / static_cast<float>(hidden) + m.norm_eps);
#pragma unroll
            for (int k = 0; k < kPer; ++k) {
                const int h = tid + kBlock * k;
                if (h < hidden) {
                    m.normed[q][base + h] =
                        __float2bfloat16_rn(mixed[q][k] * n * __bfloat162float(gains[q][k]));
                }
            }
        }
    }
}

// residual_conv_mix: the residual update of token blockIdx.x in registers, then the mix.
template <int kSources>
__global__ void __launch_bounds__(kBlock)
    residual_conv_mix_kernel(ResidualConvArgs c, MixArgs m, int hidden, int tokens) {
    pdl_enter();
    const int t = static_cast<int>(blockIdx.x);
    ConvLoads loads[kPer];
    __nv_bfloat16 base[kPer];
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        const int h = static_cast<int>(threadIdx.x) + kBlock * k;
        if (h < hidden) {
            conv_load(c, hidden, h, t, loads[k]);
            if (c.base != nullptr) base[k] = c.base[static_cast<std::int64_t>(t) * hidden + h];
        }
    }
    float cur[kPer];
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        const int h = static_cast<int>(threadIdx.x) + kBlock * k;
        cur[k]      = 0.0F;
        if (h < hidden) {
            const float yv = conv_value(c, loads[k]);
            cur[k]         = c.base == nullptr ? yv : __fadd_rn(__bfloat162float(base[k]), yv);
        }
    }
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        const int h = static_cast<int>(threadIdx.x) + kBlock * k;
        if (h < hidden) {
            c.residual[static_cast<std::int64_t>(t) * hidden + h] = cur[k];
            conv_store(c, hidden, h, t, loads[k]);
        }
    }
    mix_and_norm<kSources, true>(m, t, hidden, tokens, cur);
}

template <int kSources>
__global__ void __launch_bounds__(kBlock)
    depth_attention_rmsnorm_kernel(MixArgs m, int hidden, int tokens) {
    pdl_enter();
    const float cur[kPer] = {};
    mix_and_norm<kSources, false>(m, static_cast<int>(blockIdx.x), hidden, tokens, cur);
}

// residual_conv_round: one thread per (channel, token).
__global__ void __launch_bounds__(kBlock)
    residual_conv_round_kernel(ResidualConvArgs c, const float* __restrict__ residual,
                               __nv_bfloat16* __restrict__ out, int hidden, int tokens) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * kBlock + threadIdx.x;
    if (i >= static_cast<std::int64_t>(hidden) * tokens) { return; }
    pdl_enter();
    const int t = static_cast<int>(i / hidden);
    const int h = static_cast<int>(i - static_cast<std::int64_t>(t) * hidden);
    ConvLoads l;
    conv_load(c, hidden, h, t, l);
    const float r = residual[i];
    out[i]        = __float2bfloat16_rn(__fadd_rn(r, conv_value(c, l)));
    conv_store(c, hidden, h, t, l);
}

template <class Launch>
void by_sources(int sources, Launch&& launch) {
    [&]<int... kS>(std::integer_sequence<int, kS...>) {
        ((sources == kS + 1 ? (launch.template operator()<kS + 1>(), void()) : void()), ...);
    }(std::make_integer_sequence<int, kMaxSources>{});
}

} // namespace

// Every entry serves decode and verification widths only: always a dependent launch.
void residual_conv_mix_launch(const ResidualConvArgs& c, const MixArgs& m, int sources, int hidden,
                              int tokens, cudaStream_t stream) {
    by_sources(sources, [&]<int kS>() {
        launch_maybe_dependent(true, dim3(static_cast<unsigned>(tokens)), dim3(kBlock), 0, stream,
                               residual_conv_mix_kernel<kS>, c, m, hidden, tokens);
    });
}

void depth_attention_rmsnorm_launch(const MixArgs& m, int sources, int hidden, int tokens,
                                    cudaStream_t stream) {
    by_sources(sources, [&]<int kS>() {
        launch_maybe_dependent(true, dim3(static_cast<unsigned>(tokens)), dim3(kBlock), 0, stream,
                               depth_attention_rmsnorm_kernel<kS>, m, hidden, tokens);
    });
}

void residual_conv_round_launch(const ResidualConvArgs& c, const float* residual,
                                __nv_bfloat16* out, int hidden, int tokens, cudaStream_t stream) {
    const std::int64_t n = static_cast<std::int64_t>(hidden) * tokens;
    launch_maybe_dependent(true, dim3(static_cast<unsigned>((n + kBlock - 1) / kBlock)),
                           dim3(kBlock), 0, stream, residual_conv_round_kernel, c, residual, out,
                           hidden, tokens);
}

} // namespace ninfer::ops::detail
