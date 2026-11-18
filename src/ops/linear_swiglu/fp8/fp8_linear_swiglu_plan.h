#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_linear_swiglu_partial_capacity_bytes(std::int32_t max_tokens);

[[nodiscard]] std::size_t fp8_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                                     std::int32_t min_tokens,
                                                                     std::int32_t max_tokens);

void fp8_linear_swiglu_decode_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream);
void fp8_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream);
void fp8_linear_swiglu_matrix_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void fp8_linear_swiglu_a8_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                 WorkspaceArena& workspace, cudaStream_t stream);

using Fp8LinearSwiGluA16Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

// The A16 launchers of one gate|up problem. The decode and small-T kernels take the output rows as
// a template argument, so each registered problem has its own (the two-device half's are in
// tp2/linear_swiglu_half.h); the plan, the A8 launcher and the workspace are shared by all.
struct Fp8LinearSwiGluA16Launchers {
    Fp8LinearSwiGluA16Launch decode;
    Fp8LinearSwiGluA16Launch small_t;
    Fp8LinearSwiGluA16Launch matrix;
};

inline constexpr Fp8LinearSwiGluA16Launchers kFp8LinearSwiGluA16Launchers{
    fp8_linear_swiglu_decode_launch, fp8_linear_swiglu_small_t_launch,
    fp8_linear_swiglu_matrix_launch};

// `workspace` may be null when the resolved route needs none; the A8 route then throws.
void fp8_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                LinearPolicy policy, const Fp8LinearSwiGluA16Launchers& a16,
                                WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
