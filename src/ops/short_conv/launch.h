#pragma once

// ninfer::ops::detail - private launch prototypes for the short convolution family.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// `dyn` null selects the static tap order (weight ne = {4, C}); otherwise the dynamic one
// (weight ne = {C, 4} plus dyn ne = {4 G, T}).
void short_conv_launch(const Tensor& x, const Tensor& weight, const Tensor* dyn, float scale,
                       const Tensor& state_in, Tensor& state_out, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
