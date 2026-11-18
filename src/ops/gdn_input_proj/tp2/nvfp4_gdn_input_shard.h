#pragma once

// The NVFP4 [8192,5120] two-device shard of gdn_input_proj: one device's 8 of 16 key heads and 24
// of 48 value heads of the [16384,5120] parent, in the parent's Q|K|V|Z order, so qkv takes
// 1024 + 1024 + 3072 rows and z 3072.
//
// Both share K, so every route, schedule, token cutoff and the workspace capacity are the
// parent's by construction. The parent's plan (nvfp4/nvfp4_gdn_input_plan.cpp) and its launcher
// sources (decode, small-T, A4, TMA) serve the shard too, with the shard's section output as a
// template argument (Nvfp4GdnInputShardOutput, nvfp4/nvfp4_gdn_input_output.cuh); the shard's
// entry points below are defined next to the parent's, except the A16 one, which instantiates the
// parent's template (nvfp4/nvfp4_gdn_input_a16.cuh) in its own unit (nvfp4_gdn_input_shard_a16.cu).

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear/nvfp4/nvfp4_operands.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_gdn_input_shard_workspace_capacity_bytes(LinearPolicy policy,
                                                                         std::int32_t min_tokens,
                                                                         std::int32_t max_tokens);

void nvfp4_gdn_input_shard_a16_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                      cudaStream_t stream);
void nvfp4_gdn_input_shard_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                         Tensor& z, cudaStream_t stream);
void nvfp4_gdn_input_shard_small_t_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                          Tensor& z, cudaStream_t stream);
void nvfp4_gdn_input_shard_a4_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                     Nvfp4A4Workspace workspace, cudaStream_t stream);
// In the non-RDC archive, next to the parent's launch_nvfp4_a4_tma_gdn().
void launch_nvfp4_a4_tma_gdn_shard(const Nvfp4A4Operands& p, __nv_bfloat16* qkv, __nv_bfloat16* z,
                                   cudaStream_t stream);

// nvfp4_gdn_input_dispatch() at the shard shape, with the shard's launchers.
void nvfp4_gdn_input_shard_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    LinearPolicy policy, WorkspaceArena* workspace,
                                    cudaStream_t stream);

// Whether the snapshot and record forms quantize the activation for this policy and B/W block:
// both take A4 exactly on the materialized schedule under an A4 policy. Their frontier differs
// from the bare projection's (every T under AllowA4); the two-device shard follows it too
// (nvfp4/nvfp4_gdn_snapshot_plan.cpp).
[[nodiscard]] bool nvfp4_gdn_conv_uses_a4(LinearPolicy policy, std::int32_t width,
                                          std::int32_t batch_size);

} // namespace ninfer::ops::detail
