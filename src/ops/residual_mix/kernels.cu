// Implements: include/ninfer/ops/residual_mix.h
// residual_conv_mix and depth_attention_rmsnorm: one CTA per token, ceil(H / 8) threads rounded up
// to whole warps; thread j owns elements [8j, 8j + 8) (16-byte loads and stores). Before the wait
// for the previous kernel the CTA holds the constant operands in registers (queries, conv taps,
// norm gains). After it: the convolution and residual (elementwise: the expressions of
// short_conv_static and residual_add_f32), then every source's squares and its dot products with
// the queries in one pass and one block reduction (a thread's 8 terms in order, a warp xor tree,
// the warps in order), so a logit is dot(q, s) * inv(s) * scale; the softmax per query; the mix
// sum_s (w_qs inv_s) s; the RMS norm of each normed query (same reductions). The order of every
// sum is fixed per column: a column's bits depend neither on T nor on the call.
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

constexpr int kBlock      = 256; // residual_conv_round
constexpr int kVec        = 8;   // elements per thread of the mix kernels
constexpr int kMixThreads = kResidualMixMaxHidden / kVec;
constexpr int kMixWarps   = kMixThreads / 32;
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
__device__ __forceinline__ float conv_output(const float (&weight)[kShortConvTaps],
                                             const float (&in)[kShortConvTaps], float scale) {
    float acc = 0.0F;
#pragma unroll
    for (int j = 0; j < kShortConvTaps; ++j) { acc += weight[j] * in[j]; }
    return __bfloat162float(__float2bfloat16_rn((acc + in[kShortConvTaps - 1]) * scale));
}

__device__ __forceinline__ float conv_value(const ResidualConvArgs& a, const ConvLoads& l) {
    float in[kShortConvTaps];
#pragma unroll
    for (int j = 0; j < kShortConvTaps; ++j) { in[j] = __bfloat162float(l.in[j]); }
    return conv_output(l.weight, in, a.scale);
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

__device__ __forceinline__ void unpack(const uint4& v, float (&out)[kVec]) {
    const auto* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int k = 0; k < kVec / 2; ++k) {
        const float2 f = __bfloat1622float2(p[k]);
        out[2 * k]     = f.x;
        out[2 * k + 1] = f.y;
    }
}

__device__ __forceinline__ uint4 pack(const float (&v)[kVec]) {
    uint4 out;
    auto* p = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
    for (int k = 0; k < kVec / 2; ++k) { p[k] = __floats2bfloat162_rn(v[2 * k], v[2 * k + 1]); }
    return out;
}

__device__ __forceinline__ void load_f32(const float* p, float (&out)[kVec]) {
    const float4 a = *reinterpret_cast<const float4*>(p);
    const float4 b = *reinterpret_cast<const float4*>(p + 4);
    out[0] = a.x, out[1] = a.y, out[2] = a.z, out[3] = a.w;
    out[4] = b.x, out[5] = b.y, out[6] = b.z, out[7] = b.w;
}

__device__ __forceinline__ void store_f32(float* p, const float (&v)[kVec]) {
    *reinterpret_cast<float4*>(p)     = make_float4(v[0], v[1], v[2], v[3]);
    *reinterpret_cast<float4*>(p + 4) = make_float4(v[4], v[5], v[6], v[7]);
}

// The convolution and residual of the thread's 8 channels of token t into cur; writes the
// residual and the state or record.
__device__ __forceinline__ void conv_residual(const ResidualConvArgs& c,
                                              const float (&taps)[kVec][kShortConvTaps], int hidden,
                                              int t, int h0, float (&cur)[kVec]) {
    uint4 in[kShortConvTaps];
#pragma unroll
    for (int j = 0; j < kShortConvTaps; ++j) {
        const int s   = t - (kShortConvTaps - 1 - j);
        const auto* p = s >= 0 ? c.x + static_cast<std::int64_t>(s) * hidden + h0
                               : c.state + static_cast<std::int64_t>(kHistory + s) * hidden + h0;
        in[j]         = *reinterpret_cast<const uint4*>(p);
    }
    uint4 base = make_uint4(0U, 0U, 0U, 0U);
    if (c.base != nullptr) {
        base = *reinterpret_cast<const uint4*>(c.base + static_cast<std::int64_t>(t) * hidden + h0);
    }
    float x[kShortConvTaps][kVec];
#pragma unroll
    for (int j = 0; j < kShortConvTaps; ++j) { unpack(in[j], x[j]); }
    float b[kVec];
    unpack(base, b);
#pragma unroll
    for (int k = 0; k < kVec; ++k) {
        float window[kShortConvTaps];
#pragma unroll
        for (int j = 0; j < kShortConvTaps; ++j) { window[j] = x[j][k]; }
        const float yv = conv_output(taps[k], window, c.scale);
        cur[k]         = c.base == nullptr ? yv : __fadd_rn(b[k], yv);
    }
    store_f32(c.residual + static_cast<std::int64_t>(t) * hidden + h0, cur);
    if (c.record != nullptr) {
        *reinterpret_cast<uint4*>(c.record + static_cast<std::int64_t>(kHistory + t) * hidden +
                                  h0) = in[kShortConvTaps - 1];
        if (t == 0) {
#pragma unroll
            for (int k = 0; k < kHistory; ++k) {
                *reinterpret_cast<uint4*>(c.record + static_cast<std::int64_t>(k) * hidden + h0) =
                    in[k];
            }
        }
    } else {
#pragma unroll
        for (int k = 0; k < kHistory; ++k) {
            *reinterpret_cast<uint4*>(c.state + static_cast<std::int64_t>(k) * hidden + h0) =
                in[k + 1];
        }
    }
}

// One CTA per token. kConv: residual_conv_mix (the current source is the residual, computed
// here); else depth_attention_rmsnorm (BF16 or FP32 current).
template <int kSources, bool kConv>
__global__ void __launch_bounds__(kMixThreads)
    mix_kernel(ResidualConvArgs c, MixArgs m, int hidden, int tokens) {
    constexpr int kDots = kSources * (1 + kMaxQueries); // squares, then [q][s] dot products
    __shared__ float partial[kDots][kMixWarps];
    __shared__ float coefficient[kMaxQueries][kMaxSources];
    __shared__ float norm_partial[kMaxQueries][kMixWarps];
    const int t            = static_cast<int>(blockIdx.x);
    const int j            = static_cast<int>(threadIdx.x);
    const int lane         = j & 31;
    const int warp         = j >> 5;
    const int warps        = static_cast<int>(blockDim.x) >> 5;
    const int h0           = kVec * j;
    const bool live        = h0 < hidden;
    const int queries      = m.queries;
    const std::int64_t row = static_cast<std::int64_t>(t) * hidden + h0;

    // Constant operands, before the wait.
    float q[kMaxQueries][kVec] = {};
    float taps[kVec][kShortConvTaps];
    uint4 gain[kMaxQueries];
    if (live) {
#pragma unroll
        for (int i = 0; i < kMaxQueries; ++i) {
            if (i < queries) {
                load_f32(m.query + static_cast<std::int64_t>(i) * hidden + h0, q[i]);
            }
            if (i < m.norms) { gain[i] = *reinterpret_cast<const uint4*>(m.norm[i] + h0); }
        }
        if constexpr (kConv) {
#pragma unroll
            for (int k = 0; k < kVec; ++k) {
                const float4 w =
                    *reinterpret_cast<const float4*>(c.weight + (h0 + k) * kShortConvTaps);
                taps[k][0] = w.x, taps[k][1] = w.y, taps[k][2] = w.z, taps[k][3] = w.w;
            }
        }
    }
    pdl_enter();

    // The sources at the thread's elements: stored states, then the current one.
    float v[kSources][kVec] = {};
    if (live) {
        uint4 raw[kSources];
#pragma unroll
        for (int s = 0; s < kSources - 1; ++s) {
            raw[s] = *reinterpret_cast<const uint4*>(m.state[s] + row);
        }
        if constexpr (kConv) {
            conv_residual(c, taps, hidden, t, h0, v[kSources - 1]);
        } else if (m.current_f32 != nullptr) {
            // An FP32 current (the residual written by the unfused steps): the values the fused
            // residual_conv_mix holds in registers.
            load_f32(m.current_f32 + row, v[kSources - 1]);
        } else {
            raw[kSources - 1] = *reinterpret_cast<const uint4*>(m.current + row);
            unpack(raw[kSources - 1], v[kSources - 1]);
        }
#pragma unroll
        for (int s = 0; s < kSources - 1; ++s) { unpack(raw[s], v[s]); }
    }

    // Squares and dot products, one block reduction for all of them.
#pragma unroll
    for (int s = 0; s < kSources; ++s) {
        float sq = 0.0F;
#pragma unroll
        for (int k = 0; k < kVec; ++k) { sq = fmaf(v[s][k], v[s][k], sq); }
        sq = warp_sum(sq);
        if (lane == 0) { partial[s][warp] = sq; }
#pragma unroll
        for (int i = 0; i < kMaxQueries; ++i) {
            if (i < queries) {
                float dot = 0.0F;
#pragma unroll
                for (int k = 0; k < kVec; ++k) { dot = fmaf(q[i][k], v[s][k], dot); }
                dot = warp_sum(dot);
                if (lane == 0) { partial[kSources + i * kSources + s][warp] = dot; }
            }
        }
    }
    __syncthreads();
    // Thread i < Q: the softmax of query i over the sources, as coefficients w_is * inv_s.
    if (j < queries) {
        float inv[kSources], logit[kSources];
        float maximum = 0.0F;
#pragma unroll
        for (int s = 0; s < kSources; ++s) {
            float sq = 0.0F, dot = 0.0F;
            for (int w = 0; w < warps; ++w) {
                sq += partial[s][w];
                dot += partial[kSources + j * kSources + s][w];
            }
            inv[s]   = rsqrtf(sq / static_cast<float>(hidden) + m.eps);
            logit[s] = dot * inv[s] * m.scale;
            maximum  = s == 0 ? logit[0] : fmaxf(maximum, logit[s]);
        }
        float total = 0.0F;
#pragma unroll
        for (int s = 0; s < kSources; ++s) {
            logit[s] = expf(logit[s] - maximum);
            total += logit[s];
        }
#pragma unroll
        for (int s = 0; s < kSources; ++s) { coefficient[j][s] = logit[s] / total * inv[s]; }
    }
    __syncthreads();

    // The mix of every query, its store, and the squares of the normed ones.
    float mixed[kMaxQueries][kVec];
#pragma unroll
    for (int i = 0; i < kMaxQueries; ++i) {
        if (i < queries) {
            float w[kSources];
#pragma unroll
            for (int s = 0; s < kSources; ++s) { w[s] = coefficient[i][s]; }
#pragma unroll
            for (int k = 0; k < kVec; ++k) {
                float sum = 0.0F;
#pragma unroll
                for (int s = 0; s < kSources; ++s) { sum = fmaf(w[s], v[s][k], sum); }
                mixed[i][k] = sum;
            }
            if (live && m.mix != nullptr) {
                store_f32(m.mix + (static_cast<std::int64_t>(i) * tokens + t) * hidden + h0,
                          mixed[i]);
            }
            if (i < m.norms) {
                float sq = 0.0F;
#pragma unroll
                for (int k = 0; k < kVec; ++k) { sq = fmaf(mixed[i][k], mixed[i][k], sq); }
                sq = warp_sum(sq);
                if (lane == 0) { norm_partial[i][warp] = sq; }
            }
        }
    }
    if (m.norms == 0) { return; }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kMaxQueries; ++i) {
        if (i < m.norms) {
            float total = 0.0F;
            for (int w = 0; w < warps; ++w) { total += norm_partial[i][w]; }
            const float n = rsqrtf(total / static_cast<float>(hidden) + m.norm_eps);
            if (live) {
                float g[kVec], out[kVec];
                unpack(gain[i], g);
#pragma unroll
                for (int k = 0; k < kVec; ++k) { out[k] = mixed[i][k] * n * g[k]; }
                *reinterpret_cast<uint4*>(m.normed[i] + row) = pack(out);
            }
        }
    }
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

int mix_threads(int hidden) { return (hidden / kVec + 31) / 32 * 32; }

// Every entry serves decode and verification widths only: always a dependent launch.
void residual_conv_mix_launch(const ResidualConvArgs& c, const MixArgs& m, int sources, int hidden,
                              int tokens, cudaStream_t stream) {
    by_sources(sources, [&]<int kS>() {
        launch_maybe_dependent(true, dim3(static_cast<unsigned>(tokens)),
                               dim3(static_cast<unsigned>(mix_threads(hidden))), 0, stream,
                               mix_kernel<kS, true>, c, m, hidden, tokens);
    });
}

void depth_attention_rmsnorm_launch(const MixArgs& m, int sources, int hidden, int tokens,
                                    cudaStream_t stream) {
    by_sources(sources, [&]<int kS>() {
        launch_maybe_dependent(true, dim3(static_cast<unsigned>(tokens)),
                               dim3(static_cast<unsigned>(mix_threads(hidden))), 0, stream,
                               mix_kernel<kS, false>, ResidualConvArgs{}, m, hidden, tokens);
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
