#pragma once

// ninfer::ops::detail - private launch prototype for softcap.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void softcap_launch(const Tensor& x, float cap, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
