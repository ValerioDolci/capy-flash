#pragma once

#include "ops/linear/bf16/bf16_launch.h"

namespace ninfer::ops::detail {

[[nodiscard]] Bf16Launch select_bf16_n14336_k5120(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n5120_k6144(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n256_k5120(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n7168_k5120(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n5120_k3072(std::int32_t tokens);

// Qwen3.8-Flash-Next BF16 projections (shapes/qwen4_exp.cu): hyper-connection Down(+inject) and
// Up, and the dense [Rows, K] family (MTP dense layer, PLE key|value parent).
[[nodiscard]] Bf16Launch select_bf16_qwen4_hc_down(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_qwen4_hc_up(std::int32_t tokens);
template <int Rows, int K>
[[nodiscard]] Bf16Launch select_bf16_qwen4_dense(std::int32_t tokens);

// Generic BF16 A16 linear (bf16_generic.cu) for every shape the registry does not name.
void launch_bf16_generic(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
