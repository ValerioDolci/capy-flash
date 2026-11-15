#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q8_dflash2_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                          Tensor& k, Tensor& v, cudaStream_t stream);
void q8_dflash2_attn_input_mma_r32_c64_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                              Tensor& k, Tensor& v, cudaStream_t stream);
void q8_dflash2_attn_input_mma_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                               Tensor& k, Tensor& v, cudaStream_t stream);

void q8_dflash2_attn_input_mma_r16_c64_k128_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                                   Tensor&, cudaStream_t);

void q8_dflash2_attn_input_mma_r32_c32_k128_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                                   Tensor&, cudaStream_t);

void q8_dflash2_attn_input_mma_r32_c64_k128_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                                   Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail
