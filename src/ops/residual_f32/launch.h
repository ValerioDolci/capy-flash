#pragma once

// ninfer::ops::detail - private launch prototypes for the FP32 residual family.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void residual_add_f32_launch(const Tensor* x, const Tensor& y, Tensor& sum, cudaStream_t stream);
void residual_add_round_bf16_launch(const Tensor& sum, const Tensor& z, Tensor& out,
                                    cudaStream_t stream);

} // namespace ninfer::ops::detail
