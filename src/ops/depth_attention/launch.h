#pragma once

// ninfer::ops::detail - private launch prototype for depth_attention.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <span>

namespace ninfer::ops::detail {

void depth_attention_launch(std::span<const Tensor> states, const Tensor& current,
                            const Tensor& query, float eps, float scale, Tensor& out,
                            cudaStream_t stream);

} // namespace ninfer::ops::detail
