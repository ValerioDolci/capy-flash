#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: tanh soft cap of BF16 values into FP32
 *
 * Math / indexing (every element i):
 *   out[i] = cap * tanh( f32(x[i]) / cap )
 *
 * Logical shapes:
 *   x and out have the same shape (any ne).
 *
 * Supported domain:
 *   x contiguous BF16; out contiguous FP32; cap finite and positive.
 *
 * Numeric:
 *   The oracle evaluates the formula in FP64 from the represented input; out is compared with it
 *   pointwise. +-inf maps to +-cap, NaN stays NaN, and the sign of zero is kept. The map is
 *   monotone non-decreasing, so it never changes an argmax.
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
void softcap(const Tensor& x, float cap, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
