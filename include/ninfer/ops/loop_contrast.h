#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: extrapolation of a BF16 state away from a reference state (loop contrast, Hidden)
 *
 * Math / indexing (every element i), FP32 arithmetic with one rounding to BF16:
 *   out[i] = bf16( fma(w, f32(last[i]) - f32(reference[i]), f32(last[i])) )
 *
 * Logical shapes:
 *   last, reference and out have the same shape (any ne).
 *
 * Supported domain:
 *   last, reference and out contiguous BF16; w finite.
 *
 * Numeric:
 *   One FP32 subtraction and one fused multiply-add, rounded to nearest even: equal bit for bit
 *   to the same two FP32 operations on the host.
 *
 * Effects:
 *   Writes all of out; out may be `last` itself, and overlaps `reference` nowhere.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Stream ordered on `stream`, CUDA Graph capturable.
 */
void loop_extrapolate(const Tensor& last, const Tensor& reference, float w, Tensor& out,
                      cudaStream_t stream);

/**
 * Op: contrastive combination of soft-capped logits with a reference's (loop contrast, Logits)
 *
 * Math / indexing (column j of n, rows v < V), FP32:
 *   w_j          = w                               without `adaptive`
 *                = w * (1 - (p1 - p2))             with it, where p1 >= p2 are the two largest
 *                                                   values of softmax(capped[:, j])
 *   out[v, j]    = fma(w_j, capped[v, j] - reference[v, j], capped[v, j])
 *   logits[v, j] = bf16(out[v, j])
 *
 * Logical shapes:
 *   capped, reference, out FP32 [V, n]; logits BF16 [V, n].
 *
 * Supported domain:
 *   All contiguous; capped and reference finite; w finite. out may be `capped` itself.
 *
 * Numeric:
 *   p1 = 1 / S and p2 = exp(m2 - m1) / S with m1 >= m2 the two largest values of the column (m2 =
 *   m1 when the largest occurs twice) and S = sum_v exp(capped[v] - m1), accumulated in FP32 in
 *   a fixed order per column: the same column gives the same w_j whatever n or its position
 *   among the columns. The combination is one FP32 subtraction and one fused multiply-add.
 *
 * Effects:
 *   Writes all of out and logits; no overlap but out == capped.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   One thread block per column. Stream ordered on `stream`, CUDA Graph capturable.
 */
void loop_contrast_logits(const Tensor& capped, const Tensor& reference, float w, bool adaptive,
                          Tensor& out, Tensor& logits, cudaStream_t stream);

} // namespace ninfer::ops
