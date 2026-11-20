#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using Q8Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_q8_a16_simt_r8_t4(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_simt_r8_t8(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r32_t64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r32_t128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r64_t96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r64_t128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r128_t64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r64x16_t48_k128_a1(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r64x32_t64_k128_a1(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_q8_a16_warp_row_r4_p10(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_warp_row_r4_p3(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_warp_row_r2_w2_p5(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_warp_row_r4_p10_t8(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_warp_row_r4_p3_t8(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_q8_a16_simt_r4_t4_w2_g16_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_sliced_r16_t16_w8_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_sliced_r16_t16_w4_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_sliced_r16_t32_w4_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
} // namespace ninfer::ops::detail
