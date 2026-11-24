#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

namespace ninfer::ops {

// The widest call of the fused residual Ops (decode and verification widths).
inline constexpr std::int32_t kResidualMixMaxTokens = 16;
// The widest row: one CTA of 256 threads holds a row in registers, 12 elements per thread.
inline constexpr std::int32_t kResidualMixMaxHidden = 3072;

/**
 * The static short convolution of a residual branch (short_conv.h, short_conv_record.h):
 *   x       BF16 ne = {H, T}, the branch output
 *   weight  FP32 ne = {4, H}
 *   scale   finite
 *   state   BF16 ne = {H, 3}: advanced in place when `record` is empty (then T must be 1),
 *           read only otherwise
 *   record  BF16 ne = {H, T + 3}, or empty (data == nullptr): the call's input history
 *           (short_conv_record.h)
 */
struct ResidualConv {
    Tensor x;
    Tensor weight;
    float scale = 1.0F;
    Tensor state;
    Tensor record;
};

/**
 * Op: residual_conv_mix - residual update, depth attention and RMS norms in one pass
 *
 * Math / indexing (token t; c = conv.x after short_conv_static, BF16 as short_conv.h writes it):
 *   residual[h,t] = base[h,t] + c[h,t]        (FP32 addition; c alone when base is null)
 *   mix           = depth_attention(states, residual, query, mix_eps, mix_scale)   ne {H, T, Q}
 *   normed[q]     = rmsnorm_f32(mix[:, :, q], norms[q], norm_eps)                  q < norms.size()
 *   The conv state advances (or the record is written) as short_conv_static(_record) does.
 *
 * Numeric:
 *   residual and the conv state (or record) are bit-identical to short_conv_static(_record) then
 *   residual_add_f32. mix and normed have depth_attention's and rmsnorm_f32's math with the sums
 *   in another order (a logit is dot(q, s) * inv(s) * scale; results within those Ops'
 *   tolerances, not bit-identical); every sum's order is fixed per column, so a column's bits
 *   depend neither on T nor on the other columns, and depth_attention_rmsnorm with the same
 *   sources gives the same mix bits.
 *
 * Shapes and domain:
 *   1 <= T <= kResidualMixMaxTokens, H a multiple of 8 in 8..kResidualMixMaxHidden; every tensor
 *   16-byte aligned; base BF16 [H, T] or null; states, query, eps and scale as depth_attention.h;
 *   norms BF16 ne {H} (0 or Q of them; with 0, `mix` must be given); residual FP32 [H, T]; mix
 *   FP32 ne {H, T, Q} or null; normed BF16 [H, T], one per norm. Outputs overlap no input and no
 *   other output.
 *
 * Execution: one kernel, stream ordered on `stream`, CUDA Graph capturable. Workspace: none.
 */
void residual_conv_mix(const ResidualConv& conv, const Tensor* base, std::span<const Tensor> states,
                       const Tensor& query, float mix_eps, float mix_scale,
                       std::span<const Tensor> norms, float norm_eps, Tensor& residual, Tensor* mix,
                       std::span<Tensor> normed, cudaStream_t stream);

/**
 * Op: depth_attention_rmsnorm - depth attention of a BF16 or FP32 current state, then RMS norms
 *
 * Math: mix = depth_attention(states, current, query, mix_eps, mix_scale);
 *       normed[q] = rmsnorm_f32(mix[:, :, q], norms[q], norm_eps) for q < norms.size().
 * Numeric: as residual_conv_mix's mix and normed.
 * Shapes and domain: current BF16 or FP32 [H, T] (with the FP32 residual of residual_conv_mix as
 *   current it gives residual_conv_mix's mix and normed bits); 0 or Q norms, one output each; mix
 *   FP32 ne {H, T, Q}
 *   or null (with 0 norms it must be given); T, H, alignment and overlaps as residual_conv_mix.
 * Execution: one kernel, stream ordered, CUDA Graph capturable. Workspace: none.
 */
void depth_attention_rmsnorm(std::span<const Tensor> states, const Tensor& current,
                             const Tensor& query, float mix_eps, float mix_scale,
                             std::span<const Tensor> norms, float norm_eps,
                             std::span<Tensor> normed, Tensor* mix, cudaStream_t stream);

/**
 * Op: residual_conv_round - out = bf16(residual + short_conv_static(conv.x))
 *
 * Numeric: bit-identical to short_conv_static(_record) then residual_add_round_bf16.
 * Shapes and domain: residual FP32 [H, T], out BF16 [H, T]; conv and T as residual_conv_mix;
 * out overlaps no input.
 * Execution: one kernel, stream ordered, CUDA Graph capturable. Workspace: none.
 */
void residual_conv_round(const ResidualConv& conv, const Tensor& residual, Tensor& out,
                         cudaStream_t stream);

} // namespace ninfer::ops
