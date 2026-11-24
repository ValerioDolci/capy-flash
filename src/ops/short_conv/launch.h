#pragma once

// ninfer::ops::detail - private launch prototypes for the short convolution family.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// `dyn` null selects the static tap order (weight ne = {4, C}); otherwise the dynamic one
// (weight ne = {C, 4} plus dyn ne = {4 G, T}).
void short_conv_launch(const Tensor& x, const Tensor& weight, const Tensor* dyn, float scale,
                       const Tensor& state_in, Tensor& state_out, Tensor& out, cudaStream_t stream);

// Both calls in one launch when they take the one-launch channel route (false: nothing
// launched). Same kind (dyn both null or both given) and the same T.
bool short_conv_pair_launch(const Tensor& x0, const Tensor& w0, const Tensor* dyn0, float scale0,
                            const Tensor& in0, Tensor& state0, Tensor& out0, const Tensor& x1,
                            const Tensor& w1, const Tensor* dyn1, float scale1, const Tensor& in1,
                            Tensor& state1, Tensor& out1, cudaStream_t stream);

} // namespace ninfer::ops::detail
