#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

namespace ninfer::ops {

// The largest bottleneck rank R of message_gate.
inline constexpr std::int32_t kMessageGateMaxRank = 256;

/**
 * Op: low-rank tanh gate mixing a hidden state with a message
 *
 * Math / indexing (token t; i < H; r < R):
 *   nh[i]   = hidden[i,t]  / sqrt((1/H) sum_i hidden[i,t]^2  + eps)        RMS without gain
 *   nm[i]   = message[i,t] / sqrt((1/H) sum_i message[i,t]^2 + eps)
 *   z[i]    = bf16(nh[i]) for i < H,  bf16(nm[i-H]) for H <= i < 2H
 *   d[r]    = bf16( sum_{i<2H} down[i,r] * z[i] )
 *   f[i]    = tanh( sum_r forget[r,i] * d[r] )
 *   g[i]    = tanh( sum_r gain[r,i]   * d[r] )
 *   out[i,t] = (1 - f[i]) * hidden[i,t] + g[i] * message[i,t]
 *
 * Logical shapes:
 *   hidden, message, out  [T, H]   ne = {H, T}; H >= 1, T >= 1
 *   down     [R, 2H]  ne = {2H, R}  (the linear layer's [out, in] matrix)
 *   forget   [H, R]   ne = {R, H}
 *   gain     [H, R]   ne = {R, H}
 *   1 <= R <= kMessageGateMaxRank.
 *
 * Supported domain:
 *   hidden, message, out, forget, gain contiguous FP32; down contiguous BF16; eps finite and
 *   positive.
 *
 * Numeric:
 *   The two BF16 roundings z and d are semantic boundaries (the bottleneck is a BF16 linear layer
 *   with a BF16 result); the oracle applies them to its FP64 values and evaluates everything else
 *   in FP64 from the represented inputs. A value of z or d that lies within the implementation's
 *   rounding error of a BF16 rounding boundary may round to the neighbouring BF16 value; the
 *   suite's criterion covers that. out is FP32.
 *
 * Effects:
 *   Writes all of out. out is disjoint from every input or exactly the storage of hidden or of
 *   message (in-place update). Other inputs are unchanged.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Stream ordered on `stream`, CUDA Graph capturable.
 */
void message_gate(const Tensor& hidden, const Tensor& message, const Tensor& down,
                  const Tensor& forget, const Tensor& gain, float eps, Tensor& out,
                  cudaStream_t stream);

// The most gates one message_gates_rmsnorm call takes.
inline constexpr std::int32_t kMessageGateMaxBatch = 4;

// One gate of message_gates_rmsnorm: message_gate's operands, then the RMS norm of its output.
struct MessageGateNormed {
    Tensor hidden, message, down, forget, gain, out;
    Tensor norm;   // BF16 ne {H}
    Tensor normed; // BF16 [H, T]
};

/**
 * Op: message_gates_rmsnorm - independent message gates, each followed by an RMS norm
 *
 * Math (gate g):
 *   out_g    = message_gate(hidden_g, message_g, down_g, forget_g, gain_g, eps)
 *   normed_g = rmsnorm_f32(out_g, norm_g, norm_eps)
 *
 * Numeric:
 *   Bit-identical to message_gate then rmsnorm_f32 for every gate.
 *
 * Shapes and domain:
 *   1 <= gates.size() <= kMessageGateMaxBatch; every gate as message_gate.h with the same H, R and
 *   T; norm BF16 ne {H}, normed contiguous BF16 [H, T]; norm_eps finite and positive. normed
 *   overlaps no operand; the gates' outputs overlap no other gate's operands.
 *
 * Execution:
 *   One kernel on message_gate's cluster route (the gates' clusters run concurrently), else the two
 *   Ops per gate. Stream ordered on `stream`, CUDA Graph capturable. Workspace: none.
 */
void message_gates_rmsnorm(std::span<const MessageGateNormed> gates, float eps, float norm_eps,
                           cudaStream_t stream);

} // namespace ninfer::ops
