#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: RMSNorm of an FP32 input with a BF16 output
 *
 * Math / indexing (row r = every index of ne[1..3]; d < D = ne[0]):
 *   inv_r    = 1 / sqrt((1/D) sum_d x[d,r]^2 + eps)
 *   out[d,r] = bf16( x[d,r] * inv_r * weight[d] )
 *
 * Logical shapes:
 *   x, out  same shape, D = ne[0] >= 1, any number of rows >= 1; weight [D].
 *
 * Supported domain:
 *   x contiguous FP32; weight and out contiguous BF16; eps finite and positive. (ops::rmsnorm is
 *   the same normalization for a BF16 input.)
 *
 * Numeric:
 *   The oracle evaluates the formula in FP64 from the represented inputs; out is that value
 *   rounded once to BF16. Reduction order and accumulator precision are private.
 *
 * Effects:
 *   Writes all of out; out overlaps no input.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Stream ordered on `stream`, CUDA Graph capturable.
 */
void rmsnorm_f32(const Tensor& x, const Tensor& weight, float eps, Tensor& out,
                 cudaStream_t stream);

} // namespace ninfer::ops
