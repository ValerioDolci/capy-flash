#pragma once

// The two-device output-row half [17408,5120] of the FP8 [34816,5120] gate/up problem: one
// device's block of the gate rows over its matching block of the up rows, out [8704,T].
//
// The parent's decode and small-T kernels take the output rows as a template argument, so the
// half has its own instances, defined next to the parent's (fp8/fp8_linear_swiglu_decode.cu,
// fp8/fp8_linear_swiglu_a16.cu). The parent's plan, matrix and A8 launchers take the row count
// from the weight and serve the half unchanged: the wrapper (src/ops/wrapper/linear_swiglu.cpp)
// runs fp8_linear_swiglu_dispatch() with the half's A16 launchers below. Both share K, so every
// route, schedule and token cutoff is the parent's by construction. The NVFP4 half needs no copy:
// every NVFP4 launcher reads the rows from the weight (nvfp4/nvfp4_linear_swiglu_plan.h).

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kLinearSwiGluHalfRows = 17408;

void fp8_linear_swiglu_half_decode_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                          cudaStream_t stream);
void fp8_linear_swiglu_half_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                           cudaStream_t stream);

inline constexpr Fp8LinearSwiGluA16Launchers kFp8LinearSwiGluHalfA16Launchers{
    fp8_linear_swiglu_half_decode_launch, fp8_linear_swiglu_half_small_t_launch,
    fp8_linear_swiglu_matrix_launch};

} // namespace ninfer::ops::detail
