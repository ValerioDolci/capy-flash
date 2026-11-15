#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q8_linear_swiglu_mma_r64_c128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream);
void q8_dflash2_linear_swiglu_small_t_launch(const Tensor& x, const Weight& w, Tensor& out,
                                             cudaStream_t stream);

void q8_dflash2_linear_swiglu_mma_r32_c64_k128_launch(const Tensor&, const Weight&, Tensor&,
                                                      cudaStream_t);

void q8_dflash2_linear_swiglu_mma_r64_c64_k128_launch(const Tensor&, const Weight&, Tensor&,
                                                      cudaStream_t);

void q8_dflash2_linear_swiglu_mma_r64_c80_k128_launch(const Tensor&, const Weight&, Tensor&,
                                                      cudaStream_t);

void q8_dflash2_linear_swiglu_mma_r64_c96_k128_launch(const Tensor&, const Weight&, Tensor&,
                                                      cudaStream_t);

} // namespace ninfer::ops::detail
