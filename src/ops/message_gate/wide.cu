// Implements: the wide route of include/ninfer/ops/message_gate.h (prefill widths).
// The arithmetic of the cluster and token routes (kernels.cu), reduction by reduction, hence their
// bits: each norm is the fixed 256-thread block sum (thread tid sums the elements tid + 256 k in
// increasing k, then a warp tree, then the eight warp totals through a warp tree), each bottleneck
// value is lane l's sum of the elements i = l mod 32 in increasing i followed by a warp tree, each
// gate argument sums r = 0..R-1 in order, with the same expressions (the same products and the
// same contractions). What changes is the work around it: one CTA owns a tile of kTokens tokens
// and computes all of the tile's bottleneck values and gates, so every weight is read once per
// tile and every SM runs a tile (no cluster, no per-token barrier).
//   bottleneck: warp w owns the rows r = 8 w .. 8 w + 7 for every token of the tile; z is formed
//     in registers from hidden, message and the token's inverse norms (the bits of the cluster
//     route's z in shared memory).
//   gate: the tile's bottleneck in shared memory; thread tid owns rows tid + 256 n for every token
//     of the tile and reads each row of forget and gain once.
#include "ops/message_gate/wide.h"

#include "core/device.h" // CUDA_CHECK
#include "ninfer/ops/message_gate.h"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock        = 256;
constexpr int kWarps        = kBlock / 32;
constexpr int kMaxPerThread = 12; // the cluster route's elements per thread (H <= 3072)
constexpr int kRank         = 64; // R: eight bottleneck rows per warp
constexpr int kRowsPerWarp  = kRank / kWarps;
constexpr int kChunk        = 16; // values of r a gate thread holds at a time

// The cluster route's block_sum for kTokens values at once: values[t] is this thread's term of
// token t; returns the block totals in total[t] (every thread).
template <int kTokens>
__device__ __forceinline__ void block_sums(float (&values)[kTokens], float (*scratch)[kWarps],
                                           float* totals) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
    for (int t = 0; t < kTokens; ++t) {
        const float v = warp_sum(values[t]);
        if (lane == 0) { scratch[t][warp] = v; }
    }
    __syncthreads();
    if (warp < kTokens) {
        float total = lane < kWarps ? scratch[warp][lane] : 0.0F;
        total       = warp_sum(total);
        if (lane == 0) { totals[warp] = total; }
    }
    __syncthreads();
}

template <int kTokens>
__global__ void __launch_bounds__(kBlock, 2)
    message_gate_wide_kernel(const float* hidden, const float* message,
                             const __nv_bfloat16* __restrict__ down,
                             const float* __restrict__ forget, const float* __restrict__ gain,
                             float eps, float* out, int width, int tokens) {
    static_assert(kTokens <= kWarps && kRank % kChunk == 0);
    __shared__ float scratch[kTokens][kWarps];
    __shared__ float sum_h[kTokens], sum_m[kTokens];
    __shared__ __align__(16) float d_tile[kTokens][kRank];
    const int tid   = static_cast<int>(threadIdx.x);
    const int lane  = tid & 31;
    const int warp  = tid >> 5;
    const int t0    = static_cast<int>(blockIdx.x) * kTokens;
    const int count = min(kTokens, tokens - t0);

    // Norms: the cluster route's per-thread terms (element tid + kBlock k, zero past H).
    {
        float hh[kTokens], mm[kTokens];
#pragma unroll
        for (int t = 0; t < kTokens; ++t) {
            float hv[kMaxPerThread];
            float mv[kMaxPerThread];
            const float* h = hidden + static_cast<std::int64_t>(t0 + t) * width;
            const float* m = message + static_cast<std::int64_t>(t0 + t) * width;
#pragma unroll
            for (int k = 0; k < kMaxPerThread; ++k) {
                const int i = tid + kBlock * k;
                hv[k]       = i < width && t < count ? h[i] : 0.0F;
                mv[k]       = i < width && t < count ? m[i] : 0.0F;
            }
            hh[t] = 0.0F;
            mm[t] = 0.0F;
#pragma unroll
            for (int k = 0; k < kMaxPerThread; ++k) {
                hh[t] += hv[k] * hv[k];
                mm[t] += mv[k] * mv[k];
            }
        }
        block_sums<kTokens>(hh, scratch, sum_h);
        block_sums<kTokens>(mm, scratch, sum_m);
    }
    float inv_h[kTokens], inv_m[kTokens];
#pragma unroll
    for (int t = 0; t < kTokens; ++t) {
        inv_h[t] = rsqrtf(sum_h[t] / static_cast<float>(width) + eps);
        inv_m[t] = rsqrtf(sum_m[t] / static_cast<float>(width) + eps);
    }

    // Bottleneck: lane l of warp w, rows 8 w + j, every token, elements i = l + 32 n.
    {
        float dot[kRowsPerWarp][kTokens];
#pragma unroll
        for (int j = 0; j < kRowsPerWarp; ++j)
#pragma unroll
            for (int t = 0; t < kTokens; ++t) dot[j][t] = 0.0F;
        const __nv_bfloat16* rows =
            down + static_cast<std::int64_t>(warp) * kRowsPerWarp * 2 * width;
#pragma unroll 2
        for (int i = lane; i < width; i += 32) {
            float zh[kTokens], zm[kTokens];
#pragma unroll
            for (int t = 0; t < kTokens; ++t) {
                const std::int64_t index = static_cast<std::int64_t>(t0 + t) * width + i;
                const float hv           = t < count ? hidden[index] : 0.0F;
                const float mv           = t < count ? message[index] : 0.0F;
                zh[t]                    = __bfloat162float(__float2bfloat16_rn(hv * inv_h[t]));
                zm[t]                    = __bfloat162float(__float2bfloat16_rn(mv * inv_m[t]));
            }
#pragma unroll
            for (int j = 0; j < kRowsPerWarp; ++j) {
                const __nv_bfloat16* row = rows + static_cast<std::int64_t>(j) * 2 * width;
                const float wh           = __bfloat162float(row[i]);
                const float wm           = __bfloat162float(row[width + i]);
#pragma unroll
                for (int t = 0; t < kTokens; ++t) dot[j][t] += wh * zh[t] + wm * zm[t];
            }
        }
#pragma unroll
        for (int j = 0; j < kRowsPerWarp; ++j) {
#pragma unroll
            for (int t = 0; t < kTokens; ++t) {
                const float v = warp_sum(dot[j][t]);
                if (lane == 0) {
                    d_tile[t][warp * kRowsPerWarp + j] = __bfloat162float(__float2bfloat16_rn(v));
                }
            }
        }
    }

    // Gate: thread tid owns the rows i = tid + kBlock n for every token of the tile and reads each
    // row of forget and gain once, kChunk values of r at a time (r in order).
    __syncthreads(); // the whole bottleneck is visible
    for (int i = tid; i < width; i += kBlock) {
        const auto* f_row =
            reinterpret_cast<const float4*>(forget + static_cast<std::int64_t>(i) * kRank);
        const auto* g_row =
            reinterpret_cast<const float4*>(gain + static_cast<std::int64_t>(i) * kRank);
        float f[kTokens], g[kTokens];
#pragma unroll
        for (int t = 0; t < kTokens; ++t) {
            f[t] = 0.0F;
            g[t] = 0.0F;
        }
#pragma unroll
        for (int r0 = 0; r0 < kRank; r0 += kChunk) {
            float fr[kChunk], gr[kChunk];
#pragma unroll
            for (int q = 0; q < kChunk / 4; ++q) {
                const float4 fv = __ldg(f_row + r0 / 4 + q);
                const float4 gv = __ldg(g_row + r0 / 4 + q);
                fr[4 * q] = fv.x, fr[4 * q + 1] = fv.y, fr[4 * q + 2] = fv.z, fr[4 * q + 3] = fv.w;
                gr[4 * q] = gv.x, gr[4 * q + 1] = gv.y, gr[4 * q + 2] = gv.z, gr[4 * q + 3] = gv.w;
            }
#pragma unroll
            for (int k = 0; k < kChunk; k += 4) {
#pragma unroll
                for (int t = 0; t < kTokens; ++t) {
                    const float4 dv   = *reinterpret_cast<const float4*>(&d_tile[t][r0 + k]);
                    const float d4[4] = {dv.x, dv.y, dv.z, dv.w};
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        f[t] += fr[k + e] * d4[e];
                        g[t] += gr[k + e] * d4[e];
                    }
                }
            }
        }
#pragma unroll
        for (int t = 0; t < kTokens; ++t) {
            if (t < count) {
                const std::int64_t index = static_cast<std::int64_t>(t0 + t) * width + i;
                const float hv           = hidden[index];
                const float mv           = message[index];
                out[index]               = (1.0F - tanhf(f[t])) * hv + tanhf(g[t]) * mv;
            }
        }
    }
}

bool aligned16(const Tensor& t) { return (reinterpret_cast<std::uintptr_t>(t.data) & 15U) == 0; }

} // namespace

bool message_gate_wide(const Tensor& hidden, const Tensor& message, const Tensor& down,
                       const Tensor& forget, const Tensor& gain, float eps, Tensor& out,
                       cudaStream_t stream) {
    const int width  = hidden.ne[0];
    const int rank   = down.ne[1];
    const int tokens = hidden.ne[1];
    if (rank != kRank || width % 32 != 0 || width > kBlock * kMaxPerThread || tokens < 1 ||
        !aligned16(forget) || !aligned16(gain)) {
        return false;
    }
    // Eight tokens per tile from 1024 tokens (one wave of two tiles per SM), four below.
    const auto launch = [&](auto kernel, int tile) {
        kernel<<<static_cast<unsigned>((tokens + tile - 1) / tile), kBlock, 0, stream>>>(
            static_cast<const float*>(hidden.data), static_cast<const float*>(message.data),
            static_cast<const __nv_bfloat16*>(down.data), static_cast<const float*>(forget.data),
            static_cast<const float*>(gain.data), eps, static_cast<float*>(out.data), width,
            tokens);
    };
    if (tokens >= 1024) {
        launch(message_gate_wide_kernel<8>, 8);
    } else {
        launch(message_gate_wide_kernel<4>, 4);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

} // namespace ninfer::ops::detail
