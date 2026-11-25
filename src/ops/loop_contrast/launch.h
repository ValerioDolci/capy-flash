#pragma once

// ninfer::ops::detail - private launch prototypes for loop_contrast.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void loop_extrapolate_launch(const Tensor& last, const Tensor& reference, float w, Tensor& out,
                             cudaStream_t stream);
void loop_contrast_logits_launch(const Tensor& capped, const Tensor& reference, float w,
                                 bool adaptive, Tensor& out, Tensor& logits, cudaStream_t stream);

} // namespace ninfer::ops::detail
