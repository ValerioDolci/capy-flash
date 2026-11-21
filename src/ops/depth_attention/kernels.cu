// Implements: include/ninfer/ops/depth_attention.h
// One CTA per token. Every reduction is a fixed-order block sum in FP32, so the result depends only
// on the inputs (replicated tensor-parallel calls agree bitwise); the norms and then the dot
// products are reduced in two batches.
#include "ops/depth_attention/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ninfer/ops/depth_attention.h"
#include "ops/common/pdl_launch.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock      = 256;
constexpr int kWarps      = kBlock / 32;
constexpr int kMaxSources = kDepthAttentionMaxStates + 1;

struct BlockStates {
    const __nv_bfloat16* state[kDepthAttentionMaxStates];
};

__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ float to_float(float v) { return v; }

template <class Current>
__device__ __forceinline__ float source(const BlockStates& states, int count,
                                        const Current* __restrict__ current, int s,
                                        std::int64_t index) {
    return s < count ? __bfloat162float(states.state[s][index]) : to_float(current[index]);
}

// Reduces values[j] for j < count over the CTA, each with the tree of a single block sum (a warp
// xor tree, then warp 0's xor tree over the kWarps warp totals), so a batch gives the bits of
// count separate block sums with two barriers instead of 3 count. totals[j] receives sum j.
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

// Elements per thread loaded at once from every source: one batch of independent loads per pass
// instead of a chain of dependent loop iterations.
constexpr int kPer = 12;

// v[k][s] = source s at h = h0 + tid + kBlock k, or 0 where h is out of range (never added).
template <class Current, int kSources>
__device__ __forceinline__ void load_chunk(const BlockStates& states,
                                           const Current* __restrict__ current, std::int64_t base,
                                           int hidden, int h0, float (&v)[kPer][kSources]) {
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        const int h = h0 + static_cast<int>(threadIdx.x) + kBlock * k;
#pragma unroll
        for (int s = 0; s < kSources; ++s) {
            v[k][s] = h < hidden ? source(states, kSources - 1, current, s, base + h) : 0.0F;
        }
    }
}

// kSources = stored states + 1 (the current state), a template parameter so that every loop over
// the sources is exact.
template <class Current, int kSources>
__global__ void __launch_bounds__(kBlock)
    depth_attention_kernel(BlockStates states, const Current* __restrict__ current,
                           const float* __restrict__ query, int queries, int hidden,
                           std::int64_t tokens, float eps, float scale, float* __restrict__ out) {
    pdl_enter();
    constexpr int kMaxDots = kDepthAttentionMaxQueries * kSources;
    constexpr int kChunk   = kBlock * kPer;
    __shared__ float partial[kMaxDots][kWarps];
    __shared__ float totals[kMaxDots];
    __shared__ float inv[kMaxSources];
    __shared__ float weight[kDepthAttentionMaxQueries][kMaxSources];
    const std::int64_t t    = blockIdx.x;
    const std::int64_t base = t * hidden;
    constexpr int sources   = kSources;
    const int tid           = static_cast<int>(threadIdx.x);
    float v[kPer][kSources];

    // Every accumulator sums its h = tid + kBlock k terms in increasing h, as one block sum per
    // value would; the batch only shares the loads and the barriers.
    float squares[kSources];
#pragma unroll
    for (int s = 0; s < kSources; ++s) { squares[s] = 0.0F; }
    for (int h0 = 0; h0 < hidden; h0 += kChunk) {
        load_chunk(states, current, base, hidden, h0, v);
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            if (h0 + tid + kBlock * k < hidden) {
#pragma unroll
                for (int s = 0; s < kSources; ++s) { squares[s] += v[k][s] * v[k][s]; }
            }
        }
    }
    block_sums(squares, sources, partial, totals);
    if (tid < sources) { inv[tid] = rsqrtf(totals[tid] / static_cast<float>(hidden) + eps); }
    __syncthreads();

    float dots[kMaxDots]; // [q][s], s < kSources
#pragma unroll
    for (int j = 0; j < kMaxDots; ++j) { dots[j] = 0.0F; }
    for (int h0 = 0; h0 < hidden; h0 += kChunk) {
        load_chunk(states, current, base, hidden, h0, v);
        float qv[kPer][kDepthAttentionMaxQueries];
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            const int h = h0 + tid + kBlock * k;
#pragma unroll
            for (int q = 0; q < kDepthAttentionMaxQueries; ++q) {
                qv[k][q] = q < queries && h < hidden
                               ? query[static_cast<std::int64_t>(q) * hidden + h]
                               : 0.0F;
            }
        }
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            if (h0 + tid + kBlock * k < hidden) {
#pragma unroll
                for (int s = 0; s < kSources; ++s) {
                    const float nv = v[k][s] * inv[s];
#pragma unroll
                    for (int q = 0; q < kDepthAttentionMaxQueries; ++q) {
                        if (q < queries) { dots[q * kSources + s] += qv[k][q] * nv; }
                    }
                }
            }
        }
    }
    block_sums(dots, queries * kSources, partial, totals);
    if (tid < queries * kSources) { weight[tid / kSources][tid % kSources] = totals[tid] * scale; }
    __syncthreads();
    if (tid < queries) {
        float* logits = weight[tid];
        float maximum = logits[0];
        for (int s = 1; s < sources; ++s) { maximum = fmaxf(maximum, logits[s]); }
        float total = 0.0F;
        for (int s = 0; s < sources; ++s) {
            logits[s] = expf(logits[s] - maximum);
            total += logits[s];
        }
        for (int s = 0; s < sources; ++s) { logits[s] /= total; }
    }
    __syncthreads();

    for (int h0 = 0; h0 < hidden; h0 += kChunk) {
        load_chunk(states, current, base, hidden, h0, v);
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            const int h = h0 + tid + kBlock * k;
            if (h < hidden) {
                float normalized[kSources];
#pragma unroll
                for (int s = 0; s < kSources; ++s) { normalized[s] = v[k][s] * inv[s]; }
#pragma unroll
                for (int q = 0; q < kDepthAttentionMaxQueries; ++q) {
                    if (q < queries) {
                        float sum = 0.0F;
#pragma unroll
                        for (int s = 0; s < kSources; ++s) { sum += weight[q][s] * normalized[s]; }
                        out[(static_cast<std::int64_t>(q) * tokens + t) * hidden + h] = sum;
                    }
                }
            }
        }
    }
}

} // namespace

void depth_attention_launch(std::span<const Tensor> states, const Tensor& current,
                            const Tensor& query, float eps, float scale, Tensor& out,
                            cudaStream_t stream) {
    BlockStates block{};
    const int count = static_cast<int>(states.size());
    for (int s = 0; s < count; ++s) {
        block.state[s] =
            static_cast<const __nv_bfloat16*>(states[static_cast<std::size_t>(s)].data);
    }
    const int hidden          = current.ne[0];
    const std::int64_t tokens = current.ne[1];
    const int queries         = query.ne[1];
    const auto* q             = static_cast<const float*>(query.data);
    auto* o                   = static_cast<float*>(out.data);
    const auto grid           = static_cast<unsigned>(tokens);
    // One instantiation per source count 1..kMaxSources.
    const auto dispatch = [&]<class Current, int... kS>(const Current* cur,
                                                        std::integer_sequence<int, kS...>) {
        (((count + 1 == kS + 1) ? launch_maybe_dependent(
                                      tokens <= kDependentLaunchMaxTokens, dim3(grid), dim3(kBlock),
                                      0, stream, depth_attention_kernel<Current, kS + 1>, block,
                                      cur, q, queries, hidden, tokens, eps, scale, o)
                                : void()),
         ...);
    };
    if (current.dtype == DType::FP32) {
        dispatch(static_cast<const float*>(current.data),
                 std::make_integer_sequence<int, kMaxSources>{});
    } else {
        dispatch(static_cast<const __nv_bfloat16*>(current.data),
                 std::make_integer_sequence<int, kMaxSources>{});
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
