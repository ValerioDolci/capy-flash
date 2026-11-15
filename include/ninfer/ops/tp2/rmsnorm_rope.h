#pragma once

// Two-device (tp 2) form of rmsnorm_rope.

#include "ninfer/ops/rmsnorm_rope.h"

namespace ninfer::ops {

/**
 * The pair form of rmsnorm_rope over one tensor-parallel rank's contiguous block of heads: q BF16
 * [128,16,W,B] and k BF16 [128,4,W,B], every other contract as the [128,32]/[128,8] pair. Each
 * head is normalized and rotated by the same per-head code as the complete profile, so rank r's
 * heads equal the complete profile's heads [16 r, 16 r + 16) and [4 r, 4 r + 4) bit for bit.
 */
void rmsnorm_rope_head_block(const Tensor& positions, const Tensor& q_norm_weight,
                             const Tensor& k_norm_weight, Tensor& q, Tensor& k,
                             cudaStream_t stream);

/**
 * Query/key normalization and RoPE of one rank of the Qwen3.8-27B two-device attention shard, in
 * one launch: q BF16 [256,12,T], k BF16 [256,2,T], q_norm_weight and k_norm_weight BF16 [256],
 * positions I32 [T] or axis-major [T,3] (interleaved MRoPE: rotary pair i takes axis i % 3),
 * rotary_dim 64. For every head and token:
 *
 *   inv      = 1 / sqrt(sum_d x[d]^2 / 256 + eps)
 *   n[d]     = bf16(x[d] * inv * (1 + norm_weight[d]))                  (materialized BF16)
 *   angle(i) = position(i) * theta^(-2*i/64), 0<=i<32
 *   out[i]    = bf16(n[i]    * cos(angle(i)) - n[i+32] * sin(angle(i)))
 *   out[i+32] = bf16(n[i+32] * cos(angle(i)) + n[i]    * sin(angle(i)))
 *   out[d]    = n[d], 64<=d<256.
 *
 * Unlike rmsnorm_rope, the BF16 rounding of n is part of the contract: the outputs equal
 * ops::rmsnorm(q, q_norm_weight, eps, true, q_out) and the same for k followed by
 * ops::rope(positions, 64, theta, q_out, k_out), bit for bit, on every input (the Op runs the
 * same arithmetic in the same order; the test compares the two exactly), also with the YaRN table
 * of rope_install_text_yarn, whose angle and cos/sin factor replace the ones above as in ops::rope.
 * q_out and k_out are completely overwritten, must not overlap each other or any input; inputs are
 * unchanged. All tensors are contiguous and 4-byte aligned. No workspace; the YaRN table is the
 * only persistent state.
 */
void rmsnorm_rope_text_shard(const Tensor& positions, int rotary_dim, float theta,
                             const Tensor& q_norm_weight, const Tensor& k_norm_weight, float eps,
                             const Tensor& q, const Tensor& k, Tensor& q_out, Tensor& k_out,
                             cudaStream_t stream);

} // namespace ninfer::ops
