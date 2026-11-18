#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                   std::int32_t min_tokens,
                                                                   std::int32_t max_tokens);

void nvfp4_gdn_input_a16_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream);

void nvfp4_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   cudaStream_t stream);

void nvfp4_gdn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    cudaStream_t stream);

using Nvfp4GdnInputA4Launch = void (*)(const Weight&, Tensor&, Tensor&, Nvfp4A4Workspace,
                                       std::int32_t, cudaStream_t);

// The A4 route's GEMM at a width and the activation-scale layout it reads; the quantizer writes
// that layout before the GEMM runs, so the two are selected together.
struct Nvfp4GdnInputA4Route {
    Nvfp4GdnInputA4Launch launch;
    Nvfp4ScaleLayout scales;
};

// The selections of nvfp4_gdn_input_a4_launch() and nvfp4_gdn_input_shard_a4_launch().
[[nodiscard]] Nvfp4GdnInputA4Route nvfp4_gdn_input_select_a4(std::int32_t tokens);
[[nodiscard]] Nvfp4GdnInputA4Route nvfp4_gdn_input_shard_select_a4(std::int32_t tokens);

void nvfp4_gdn_input_a4_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                               Nvfp4A4Workspace workspace, cudaStream_t stream);

void nvfp4_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                              LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
