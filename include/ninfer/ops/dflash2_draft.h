#pragma once

// Ops of a DFlash2 block drafter at any hidden width, beside the DSpark drafter's (dspark_draft.h:
// per-head norm and RoPE, context K/V store, block attention over a sliding window) and the
// candidate selector (candidate_selector.h): the grouped dynamic convolution around a sublayer of
// the block, and the vocabulary's top candidates of the block's columns over one or two ranks'
// vocabulary blocks.
//
// Convolution notation: H channels in G = H / 16 groups of 16 (channel h in group h / 16), two
// taps; base BF16 [H, 2, 2] (ne {channel, tap, side}: the checkpoint's [side, tap, channel]); c
// BF16 [4G, W], the kernel projection of the sublayer's normed input, row (side * 2 + tap) * G +
// group. Side 0 convolves the sublayer's input, side 1 its output. Column 0 has no previous tap: a
// block never reads another block.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Op: grouped dynamic convolution of a sublayer's input (side 0)
 *
 * Math, FP32 from the stored values, one BF16 rounding:
 *   out[h, i] = bf16((f32(base[h, 0, 0]) + f32(c[g, i])) x[h, i]
 *                    + I(i > 0) (f32(base[h, 1, 0]) + f32(c[G + g, i])) x[h, i - 1])
 * x, out BF16 [H, W] with H % 16 == 0, W >= 1; c BF16 [4G, W]; base BF16 [H, 2, 2]. All contiguous;
 * out overlaps no input. Capturable; no workspace.
 */
void dynamic_conv_input(const Tensor& x, const Tensor& coefficients, const Tensor& base,
                        Tensor& out, cudaStream_t stream);

/**
 * Op: grouped dynamic convolution of a sublayer's output (side 1), added to the residual
 *
 * Math, FP32 from the stored values, one BF16 rounding:
 *   residual[h, i] = bf16(f32(residual[h, i]) + (f32(base[h, 0, 1]) + f32(c[2G + g, i])) z[h, i]
 *                         + I(i > 0) (f32(base[h, 1, 1]) + f32(c[3G + g, i])) z[h, i - 1])
 * z, residual BF16 [H, W]; c and base as dynamic_conv_input. residual overlaps no input.
 * Capturable; no workspace.
 */
void dynamic_conv_output_add(const Tensor& z, const Tensor& coefficients, const Tensor& base,
                             Tensor& residual, cudaStream_t stream);

/**
 * Op: the top candidates of one rank's vocabulary block, as exchange slots
 *
 * Math: logits BF16 [V_r, C] are rows [first, first + V_r) of the vocabulary. For every column c
 * the 16 largest values, in descending order of the value with ties to the lower id, are written
 * into slot `slot` of pairs FP32 [2, 16, C, slots] as (value, global id) (ids below 2^24 are exact
 * in FP32); every other slot of the column is zeroed, so that the FP32 sum of the slots of every
 * rank (allreduce_sum_f32) leaves each rank's slot intact on both ranks.
 * Supported: V_r >= 16, 1 <= C <= 16, slots 1 or 2, first + V_r <= 2^24.
 * Workspace: vocab_top16_workspace_bytes(C). Capturable.
 */
void vocab_top16_slots(const Tensor& logits, std::int32_t first, std::int32_t slot, Tensor& pairs,
                       WorkspaceArena& workspace, cudaStream_t stream);
[[nodiscard]] std::size_t vocab_top16_workspace_bytes(std::int32_t columns);

/**
 * Op: the top candidates over every slot
 *
 * Math: from pairs FP32 [2, 16, C, slots] (vocab_top16_slots after the exchange), the 16 largest
 * values of each column over all its slots, in descending order with ties to the lower id: ids I32
 * [16, C] and values FP32 [16, C] (candidate rank fastest; the candidate selector's candidate_ids
 * and unary_scores). Capturable; no workspace.
 */
void vocab_top16_merge(const Tensor& pairs, Tensor& ids, Tensor& values, cudaStream_t stream);

} // namespace ninfer::ops
