#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q8_pair_simt_r8_c4_launch(const Tensor& x, const Weight& first_weight,
                               const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                               cudaStream_t stream);
void q8_pair_gemm_mma_r32_c64_launch(bool full, const Tensor& x, const Weight& first_weight,
                                     const Weight& second_weight, Tensor& first_out,
                                     Tensor& second_out, cudaStream_t stream);
void q8_pair_gemm_mma_r32_c128_launch(bool full, const Tensor& x, const Weight& first_weight,
                                      const Weight& second_weight, Tensor& first_out,
                                      Tensor& second_out, cudaStream_t stream);

} // namespace ninfer::ops::detail
