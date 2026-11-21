#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: residual sum widened to FP32, and its single rounding back to BF16
 *
 * Math / indexing (every element i):
 *   residual_add_f32:        sum[i] = fl32( f32(x[i]) + f32(y[i]) )     (x given)
 *                            sum[i] = f32(y[i])                          (x null)
 *   residual_add_round_bf16: out[i] = bf16( fl32( sum[i] + f32(z[i]) ) )
 *   fl32 is one IEEE FP32 addition rounded to nearest even; bf16 rounds to nearest even.
 *
 * Logical shapes:
 *   All operands of one call have the same shape (any ne).
 *
 * Supported domain:
 *   x, y, z, out contiguous BF16; sum contiguous FP32.
 *
 * Numeric:
 *   Exact: the result is the stated FP32 addition (and, for the round, the stated BF16 rounding
 *   of the FP32 sum), bit for bit.
 *
 * Effects:
 *   residual_add_f32 writes all of sum, which overlaps neither x nor y.
 *   residual_add_round_bf16 writes all of out, which is disjoint from sum and z or exactly the
 *   storage of z (in-place). Inputs are otherwise unchanged.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Stream ordered on `stream`, CUDA Graph capturable.
 */
void residual_add_f32(const Tensor* x, const Tensor& y, Tensor& sum, cudaStream_t stream);
void residual_add_round_bf16(const Tensor& sum, const Tensor& z, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
