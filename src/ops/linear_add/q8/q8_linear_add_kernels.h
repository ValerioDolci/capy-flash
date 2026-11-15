#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q8_linear_add_splitk_capacity_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                          cudaStream_t stream);

void q8_linear_add_grouped_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                  cudaStream_t stream);

} // namespace ninfer::ops::detail
