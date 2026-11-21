#pragma once

// ninfer::ops::detail - private launch prototypes for the recording short convolutions and the
// state fold.

#include "core/tensor.h"
#include "ninfer/ops/short_conv_record.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// `dyn` null selects the static tap order (weight ne = {4, C}); otherwise the dynamic one
// (weight ne = {C, 4} plus dyn ne = {4 G, T}).
void short_conv_record_launch(const Tensor& x, const Tensor& weight, const Tensor* dyn, float scale,
                              const Tensor& state, Tensor& record, Tensor& out,
                              cudaStream_t stream);

void conv_tail_fold_launch(const ConvTailFoldEntry* entries, std::int32_t count, std::int32_t width,
                           const std::int32_t* committed, cudaStream_t stream);

} // namespace ninfer::ops::detail
