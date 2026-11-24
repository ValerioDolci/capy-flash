#pragma once

#include "ninfer/ops/attention_geometry.h"

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/** Semantic constants of one relative-sink attention call; every value comes from the caller. */
struct RelativeSinkAttentionParams {
    std::int32_t relative_dim    = 0; // dr: features of one relative state
    std::int32_t relative_extent = 0; // R: distances 0..R-1 carry a bias
    float qk_clip                = 0; // L2 threshold of q (after its scale) and of k
    float relative_clip          = 0; // L2 threshold of r (after its scale)
    float clip_epsilon           = 0; // added to the L2 norm in both clips
    float value_epsilon          = 0; // added to the mean square of v
    float scale                  = 0; // logit scale applied to q.k
    float log_scale_alpha        = 0; // 0 disables the positional scale of q and r
    float log_scale_floor        = 1; // positional scale grows once position + 1 > floor
};

/**
 * Op: causal grouped-query attention with clipped q/k, RMS-normalized v, a token-conditioned
 * relative-distance bias, and one sink logit per query and head; appends the block's K/V.
 *
 * Math / indexing:
 *   Per batch row b and live column t, the absolute position is p = positions[t,b] = L_b + t,
 *   where L_b = positions[0,b] is the number of keys committed before this call. Query head h
 *   reads KV head g = floor(h / (Hq/Hkv)). bf16(x) is round-to-nearest-even to BF16, f(x) is
 *   the exact FP32 value of a represented input, and |x| is the Euclidean norm over one head
 *   vector. With c = clip_epsilon, a = log_scale_alpha, F = log_scale_floor:
 *
 *     clip(x, thr)  = x * min(1, thr / (|x| + c))
 *     tau(p)        = 1 + a * ln(max((p + 1) / F, 1))
 *     q1[:,h,t]     = bf16(clip(f(q[:,h,t]) * f(q_scale[:,h]), qk_clip))
 *     q'[:,h,t]     = bf16(q1[:,h,t] * tau(p))
 *     r1[:,h,t]     = bf16(clip(f(r[:,h,t]) * f(r_scale[:,h]), relative_clip))
 *     r'[:,h,t]     = bf16(r1[:,h,t] * tau(p))
 *     k'[:,g,t]     = bf16(clip(f(k[:,g,t]), qk_clip))
 *     v'[:,g,t]     = FP16_RNE(bf16(f(v[:,g,t]) / sqrt(mean_d(f(v[d,g,t])^2) + value_epsilon)))
 *
 *   With a = 0, tau = 1 and q' = q1, r' = r1 exactly. Committed cache rows hold earlier k'
 *   (BF16) and v' (FP16) at their absolute positions. For query position p and a visible key
 *   at position s with distance delta = p - s:
 *
 *     bias(h,p,s)   = delta < R ? bf16(sum_j r'[j,h,t] * f(rel_proj[delta,j])) : 0
 *     logit(h,p,s)  = bf16(bf16(bf16(q'[:,h,t] . k'[:,g,s]) * scale) + bias(h,p,s))
 *     m             = max(max_s logit(h,p,s), f(sink[h,t]))
 *     Z             = exp(f(sink[h,t]) - m) + sum_s exp(logit(h,p,s) - m)
 *     prob(h,p,s)   = bf16(exp(logit(h,p,s) - m) / Z)
 *     o[:,h,t,b]    = bf16(sum_s prob(h,p,s) * v'[:,g,s])
 *     out[:,h,t,b]  = o[:,h,t,b]                                          without gate
 *     out[:,h,t,b]  = bf16(o[:,h,t,b] * sigmoid(f(gate[:,h,t,b])))         with gate
 *
 *   sigmoid(x) = 1 / (1 + exp(-x)) in FP32: the gate is the arithmetic of ops::sigmoid_mul applied
 *   to o, so a gated call equals an ungated call followed by sigmoid_mul(gate, out) bit for bit.
 *
 *   The sink enters the normalization unscaled and without bias; its own probability is
 *   discarded. Visible keys: relative_sink_attention_windowed sees p - window < s <= p;
 *   relative_sink_attention_paged sees every 0 <= s <= p (the bias is zero for delta >= R).
 *   Both read keys s < L_b from the cache and keys L_b <= s <= p from this call's k'/v', so a
 *   prefill split into blocks of any size and token-by-token decode define the same values
 *   (see Numeric for which of them also share bits).
 *
 * Logical shapes (ne order, innermost first):
 *   q, out     BF16 [D,Hq,T,B]     k, v   BF16 [D,Hkv,T,B]     r  BF16 [dr,Hq,T,B]
 *   gate       nullptr, or BF16 [D,Hq,T,B]
 *   sink       BF16 [Hq,T,B]       positions  device I32 [T,B], sequential within each row
 *   q_scale    BF16 [D,Hq]         r_scale    BF16 [dr,Hq]
 *   rel_proj   BF16 [R,dr]: element (delta, j) is P[j][delta] of a row-major [dr,R] matrix
 *   lanes / kv_table_rows   device I32 [B]: cyclic-cache lane / paged table row of batch row b
 *   T >= 1 and B >= 1. A tensor-parallel rank passes its contiguous head shard (Hq/n, Hkv/n and
 *   the matching rows of q_scale/r_scale); the formula is per head, so no new math is involved.
 *
 * Supported domain:
 *   D % 32 == 0 and 32 <= D <= 256; Hq % Hkv == 0; 1 <= dr <= 32; R >= 1; scale, thresholds and
 *   F > 0; epsilons >= 0. Windowed: window >= 1, cache.capacity >= window - 1, cache geometry
 *   [D, padded_capacity, Hkv, lane_capacity] with BF16 K and FP16 V, and committed position x
 *   stored at slot x mod capacity. Paged: the BFloat16 storage profile (BF16 K, FP16 V planes)
 *   in page-major order [D, 64, Hkv, pages], block_tables I32 [logical pages, rows].
 *
 * Numeric:
 *   Every bf16(...) above is a semantic rounding point (it reproduces the reference model).
 *   Sums inside a bf16(...) are evaluated with at least FP32 accumulation, the exponentials,
 *   max, Z and the division in at least FP32. A value whose higher-precision result lies within
 *   FP32 accumulation error of a BF16 rounding midpoint may round either way; the qualification
 *   criterion accounts for that. v' crosses its FP16 persistent boundary before attention
 *   observes it, for cached and for block rows alike (exact for |v'| >= 2^-17; |v'| <= sqrt(D)).
 *   Non-finite inputs are outside the domain.
 *   relative_sink_attention_paged with at most 16 tokens per row and D <= 128 evaluates a query
 *   that sees more than 128 keys in one pass over the keys: it keeps exp(logit - m) unrounded
 *   against running maxima and divides the combined value sum by Z, so prob is not rounded to
 *   BF16 there (one rounding point fewer than the formula above; the qualification criterion
 *   covers it). Every other query rounds prob as written.
 *   Bits by call extent: calls with at most 16 tokens per row, and every call with D > 128,
 *   give each query the same bits whatever the call's T or block boundaries (a split prefill
 *   equals token-by-token decode bit for bit). Longer calls with D <= 128 use tensor-core
 *   products in one pass over the keys: exp(logit - m) against running maxima is rounded to FP16
 *   (finer than BF16) as the operand of the value product, which sums each run of 64 keys in FP16
 *   before the FP32 total, and the sum is divided by Z at the end, so prob is not rounded to BF16
 *   there; they agree with the shorter calls within the
 *   qualification criterion, not bit for bit; equal
 *   calls (same block boundaries) always give equal bits. The cache contents written by any
 *   call are the same bits in every case.
 *
 * Effects:
 *   out is completely overwritten. Windowed: for every b, cache lane lanes[b] receives k'/v' of
 *   positions max(L_b, L_b + T - capacity) .. L_b + T - 1 at slot x mod capacity; the attention
 *   of this call reads the cache before that write, so committed rows it overwrites are still
 *   observed with their old value. Paged: every position L_b .. L_b + T - 1 is written through
 *   block table row kv_table_rows[b]. No other cache byte is read beyond the visible committed
 *   rows, and no other cache byte is written. Inputs are unchanged. Inputs, out, cache planes,
 *   tables and workspace are pairwise non-overlapping. The caller guarantees that the committed
 *   rows L_b - window + 1 .. L_b - 1 (windowed) or 0 .. L_b - 1 (paged) are populated, that
 *   the pages of the written positions are mapped, and that batch rows use distinct lanes or
 *   table rows. The Op owns no frontier, lane or page allocation.
 *
 * Workspace:
 *   Caller-owned arena of at least relative_sink_attention_workspace_capacity_bytes for the
 *   call's geometry, dr, B and T; used only for the duration of the call on `stream`.
 *
 * Execution:
 *   Positions, lanes and table rows are read on the device; there is no host envelope. With
 *   fixed tensor views the call is CUDA Graph capturable and replays for any valid positions.
 */
void relative_sink_attention_windowed(const Tensor& q, const Tensor& k, const Tensor& v,
                                      const Tensor& r, const Tensor& sink, const Tensor& positions,
                                      const Tensor& lanes, const Tensor& q_scale,
                                      const Tensor& r_scale, const Tensor& rel_proj,
                                      AttentionHeadGeometry geometry,
                                      const RelativeSinkAttentionParams& params,
                                      std::int32_t window, const CyclicKVCacheLayerView& cache,
                                      const Tensor* gate, WorkspaceArena& workspace, Tensor& out,
                                      cudaStream_t stream);

/** Same contract as relative_sink_attention_windowed with every key 0 <= s <= p visible. */
void relative_sink_attention_paged(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& r, const Tensor& sink, const Tensor& positions,
                                   const Tensor& kv_table_rows, const Tensor& q_scale,
                                   const Tensor& r_scale, const Tensor& rel_proj,
                                   AttentionHeadGeometry geometry,
                                   const RelativeSinkAttentionParams& params,
                                   const PagedKVBatchLayerView& cache, const Tensor* gate,
                                   WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

/**
 * Transient capacity sufficient for every T in [min_tokens, max_tokens] at the exact batch size,
 * for both entries. Invalid geometry, dr or interval throws.
 */
[[nodiscard]] std::size_t
relative_sink_attention_workspace_capacity_bytes(AttentionHeadGeometry geometry,
                                                 std::int32_t relative_dim, std::int32_t batch_size,
                                                 std::int32_t min_tokens, std::int32_t max_tokens);

} // namespace ninfer::ops
