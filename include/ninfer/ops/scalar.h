#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: finite typed scalar state transitions
 *
 * Math / indexing:
 *   set_i32_scalar:       destination' = value
 *   increment_i32_scalar: scalar' = scalar + 1
 *
 * Logical shapes:
 *   Every tensor argument is one contiguous scalar element of the named dtype.
 *
 * Numeric:
 *   Increment callers keep the exact result within the signed integer range.
 *
 * Effects:
 *   Each call writes only its destination scalar, in place.
 *
 * Workspace:
 *   None. There is no state side effect beyond the stated destination transition.
 */
void set_i32_scalar(Tensor& destination, std::int32_t value, cudaStream_t stream);
void increment_i32_scalar(Tensor& scalar, cudaStream_t stream);

} // namespace ninfer::ops
