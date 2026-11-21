#pragma once

// ninfer::ops::detail - private launch prototype for rmsnorm_f32.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void rmsnorm_f32_launch(const Tensor& x, const Tensor& weight, float eps, Tensor& out,
                        cudaStream_t stream);

} // namespace ninfer::ops::detail
