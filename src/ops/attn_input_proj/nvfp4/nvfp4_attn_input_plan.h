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

// The route of the parent [14336,5120] and its two-device shard [7168,5120] (one plan).
enum class Nvfp4AttnInputRoute : std::uint8_t {
    A16,
    A4,
};

[[nodiscard]] Nvfp4AttnInputRoute nvfp4_attn_input_route(LinearPolicy policy, std::int32_t tokens);

[[nodiscard]] std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                    std::int32_t min_tokens,
                                                                    std::int32_t max_tokens);

void nvfp4_attn_input_a16_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                 Tensor& k, Tensor& v, cudaStream_t stream);

void nvfp4_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream);

void nvfp4_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream);

using Nvfp4AttnInputA4Launch = void (*)(const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                                        Nvfp4A4Workspace, std::int32_t, cudaStream_t);

// The A4 route's GEMM at a width and the activation-scale layout it reads; the quantizer writes
// that layout before the GEMM runs, so the two are selected together.
struct Nvfp4AttnInputA4Route {
    Nvfp4AttnInputA4Launch launch;
    Nvfp4ScaleLayout scales;
};

// The selections of nvfp4_attn_input_a4_launch() and of the shard's parent-family launcher
// (nvfp4_attn_input_shard_a4_launch_upstream()); the shard's own per-GPU TMA band
// (nvfp4_attn_input_shard_a4_launch()) is outside them.
[[nodiscard]] Nvfp4AttnInputA4Route nvfp4_attn_input_select_a4(std::int32_t tokens);
[[nodiscard]] Nvfp4AttnInputA4Route nvfp4_attn_input_shard_select_a4(std::int32_t tokens);

void nvfp4_attn_input_a4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                Tensor& k, Tensor& v, Nvfp4A4Workspace workspace,
                                cudaStream_t stream);

void nvfp4_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                               cudaStream_t stream);

} // namespace ninfer::ops::detail
