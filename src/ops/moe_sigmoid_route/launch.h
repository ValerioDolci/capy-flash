#pragma once

// ninfer::ops::detail - private launch of moe_sigmoid_route.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void moe_sigmoid_route_launch(const Tensor& x, const Tensor& router, const Tensor& bias,
                              std::int32_t top_k, float route_scale, Tensor& ids, Tensor& weights,
                              const Tensor* valid_columns, cudaStream_t stream);

} // namespace ninfer::ops::detail
