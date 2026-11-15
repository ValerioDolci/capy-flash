#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

void bf16_gdn_norm_gating_proj_27_launch(const Tensor& x, const Tensor& norm_weight, float eps,
                                         Tensor& h, const Weight& a_weight, const Weight& b_weight,
                                         const Tensor& alog, const Tensor& bias, Tensor& g,
                                         Tensor& beta, cudaStream_t stream);

enum class Bf16GdnGatingTokenVariant {
    None,
    Full,
    Predicated,
};

void bf16_gdn_gating_proj_gemv_launch(const Tensor& x, const Weight& a_weight,
                                      const Weight& b_weight, const Tensor& A_log,
                                      const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                      cudaStream_t stream);
void bf16_gdn_gating_proj_small_t_split10_launch(const Tensor& x, const Weight& a_weight,
                                                 const Weight& b_weight, const Tensor& A_log,
                                                 const Tensor& dt_bias, void* workspace,
                                                 std::size_t workspace_bytes, Tensor& g,
                                                 Tensor& beta, cudaStream_t stream);

// Cooperative launchers return false without submitting work only when the selected device cannot
// make one complete token tile resident. The Op wrapper owns the non-cooperative fallback.
[[nodiscard]] bool bf16_gdn_gating_proj_mma_split8_launch(
    Bf16GdnGatingTokenVariant variant, const Tensor& x, const Weight& a_weight,
    const Weight& b_weight, const Tensor& A_log, const Tensor& dt_bias, void* workspace, Tensor& g,
    Tensor& beta, std::int32_t multiprocessor_count, cudaStream_t stream);
[[nodiscard]] bool bf16_gdn_gating_proj_mma_split4_launch(
    Bf16GdnGatingTokenVariant variant, const Tensor& x, const Weight& a_weight,
    const Weight& b_weight, const Tensor& A_log, const Tensor& dt_bias, void* workspace, Tensor& g,
    Tensor& beta, std::int32_t multiprocessor_count, cudaStream_t stream);
[[nodiscard]] bool bf16_gdn_gating_proj_mma_split2_launch(
    Bf16GdnGatingTokenVariant variant, const Tensor& x, const Weight& a_weight,
    const Weight& b_weight, const Tensor& A_log, const Tensor& dt_bias, void* workspace, Tensor& g,
    Tensor& beta, std::int32_t multiprocessor_count, cudaStream_t stream);
void bf16_gdn_gating_proj_mma_unsplit_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                             const Weight& a_weight, const Weight& b_weight,
                                             const Tensor& A_log, const Tensor& dt_bias, Tensor& g,
                                             Tensor& beta, cudaStream_t stream);

// Two-device column shard [24,5120] per weight; see kShardN in bf16_gdn_gating_proj_kernels.cu.
void bf16_gdn_gating_proj_gemv_shard_launch(const Tensor& x, const Weight& a_weight,
                                            const Weight& b_weight, const Tensor& A_log,
                                            const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                            cudaStream_t stream);
void bf16_gdn_gating_proj_small_t_split10_shard_launch(const Tensor& x, const Weight& a_weight,
                                                       const Weight& b_weight, const Tensor& A_log,
                                                       const Tensor& dt_bias, void* workspace,
                                                       std::size_t workspace_bytes, Tensor& g,
                                                       Tensor& beta, cudaStream_t stream);
// T=2..32 in one launch: the split-10 partial sums of every slice reduced in split order inside
// the CTA, bit-identical to bf16_gdn_gating_proj_small_t_split10_shard_launch at the same T.
void bf16_gdn_gating_proj_small_t_fused_shard_launch(const Tensor& x, const Weight& a_weight,
                                                     const Weight& b_weight, const Tensor& A_log,
                                                     const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                                     cudaStream_t stream);
// GEMV at T=1 and the fused split-10 kernel through T=32 (workspace unused), the two-kernel
// split-10 form above that.
void bf16_gdn_gating_dispatch_shard(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                                    const Tensor& A_log, const Tensor& dt_bias, void* workspace,
                                    std::size_t workspace_bytes, Tensor& g, Tensor& beta,
                                    cudaStream_t stream);
[[nodiscard]] std::size_t bf16_gdn_gating_shard_workspace_bytes(std::int32_t tokens);

} // namespace ninfer::ops::detail
