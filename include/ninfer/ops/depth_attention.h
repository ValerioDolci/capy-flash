#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

namespace ninfer::ops {

// The largest number of stored block states one call attends over (the current state excluded).
inline constexpr std::int32_t kDepthAttentionMaxStates = 8;
// The largest number of learned queries one call evaluates.
inline constexpr std::int32_t kDepthAttentionMaxQueries = 4;

/**
 * Op: depth attention of learned queries over normalized block states
 *
 * Math / indexing (token t; source s = 0..n, where s < n is states[s] and s = n is current):
 *   v[s,h]    = src_s[h,t] / sqrt((1/H) sum_{h'} src_s[h',t]^2 + eps)     RMS without gain
 *   l[q,s]    = scale * sum_h query[h,q] * v[s,h]
 *   a[q,s]    = exp(l[q,s]) / sum_{s'=0..n} exp(l[q,s'])
 *   out[h,t,q] = sum_{s=0..n} a[q,s] * v[s,h]
 *   Keys and values are the same normalized states. Tokens are independent: there is no state
 *   across tokens. With n = 0 the output is the normalized current state for every query.
 *
 * Logical shapes:
 *   states[s]  [T, H]  ne = {H, T}, 0 <= n = states.size() <= kDepthAttentionMaxStates
 *   current    [T, H]  ne = {H, T}
 *   query      [Q, H]  ne = {H, Q}, 1 <= Q <= kDepthAttentionMaxQueries
 *   out        [Q, T, H]  ne = {H, T, Q}: query q's [T, H] block is contiguous, so
 *              out.slice(2, q, 1) is a dense [H, T] FP32 tensor
 *   H >= 1, T >= 1.
 *
 * Supported domain:
 *   states contiguous BF16; current contiguous BF16 or FP32; query and out contiguous FP32;
 *   eps finite and positive; scale finite (callers pass 1/sqrt(H) for the scaled dot product).
 *
 * Numeric:
 *   The oracle evaluates every line above in FP64 from the represented inputs; out is compared
 *   with it directly. The implementation's accumulation precision and order are private. One
 *   implementation gives the same bits for the same inputs on every device of one architecture
 *   (no atomics, no data-dependent order), so replicated tensor-parallel calls agree bitwise.
 *
 * Effects:
 *   Writes all of out; out overlaps no input. Inputs are unchanged.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Stream ordered on `stream`, CUDA Graph capturable.
 */
void depth_attention(std::span<const Tensor> states, const Tensor& current, const Tensor& query,
                     float eps, float scale, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
