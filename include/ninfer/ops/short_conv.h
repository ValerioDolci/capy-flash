#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// Taps of the short convolutions below; their state holds the kShortConvTaps - 1 previous inputs.
inline constexpr std::int32_t kShortConvTaps = 4;

/**
 * Op: residual depthwise causal convolution with static taps
 *
 * Math / indexing (channel c, token t; u = [state_in ; x], so u[c,-3..-1] are the three state
 * columns, oldest first, and u[c,t] = x[c,t] for t >= 0):
 *   acc[c,t]   = sum_{j=0..3} weight[j,c] * u[c,t-3+j]       tap j = 3 weighs the CURRENT token,
 *                                                            tap j = 0 the token t-3
 *   out[c,t]   = bf16( (acc[c,t] + x[c,t]) * scale )
 *   state_out  = the last three columns of [state_in ; x], oldest first
 *
 * Logical shapes:
 *   x, out     [C, T]  ne = {C, T}; C >= 1, T >= 1
 *   weight     [C, 4]  ne = {4, C}: the four taps of channel c are contiguous (the depthwise
 *                      [C, 1, 4] kernel layout)
 *   state_in, state_out  [3, C]  ne = {C, 3}: column k holds the input three - k tokens back
 *
 * Supported domain:
 *   x, out, state_in, state_out contiguous BF16; weight contiguous FP32; scale finite and positive.
 *   A zero state_in is the start of a sequence. Splitting a sequence into consecutive calls that
 *   carry state_out into the next state_in gives the outputs and final state of one call.
 *
 * Numeric:
 *   The oracle evaluates acc + x and the product with scale in FP64 from the represented inputs;
 *   out is that value rounded once to BF16 (one rounding: the implementation's private
 *   accumulator precision is not observable beyond the criterion). state_out is an exact copy.
 *
 * Effects:
 *   Writes all of out and state_out. state_out and state_in are disjoint or exactly the same
 *   storage (in-place state update). out overlaps no other operand; x and weight are unchanged.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Stream ordered on `stream`, CUDA Graph capturable.
 */
void short_conv_static(const Tensor& x, const Tensor& weight, float scale, const Tensor& state_in,
                       Tensor& state_out, Tensor& out, cudaStream_t stream);

/**
 * Op: residual depthwise causal convolution with per-token, per-group dynamic taps
 *
 * Math / indexing (channel c in group g = c / D, D = C / G; token t; lag l = 0..3; u as above):
 *   tap[l,c,t] = static_weight[l,c] + dyn[4 g + l, t]
 *   acc[c,t]   = sum_{l=0..3} tap[l,c,t] * u[c,t-l]          lag l = 0 weighs the CURRENT token
 *   out[c,t]   = bf16( (acc[c,t] + x[c,t]) * scale )
 *   state_out  = the last three columns of [state_in ; x], oldest first
 *
 *   NOTE the tap order is the opposite of short_conv_static: here index 0 is the current token.
 *
 * Logical shapes:
 *   x, out          [C, T]   ne = {C, T}; C >= 1, T >= 1
 *   static_weight   [4, C]   ne = {C, 4}: row l (lag l) is contiguous over the channels
 *   dyn             [T, 4G]  ne = {4 G, T}: element 4 g + l of token t is the dynamic tap of
 *                            group g at lag l (the output of a linear projection, group-major)
 *   state_in, state_out      ne = {C, 3} as for short_conv_static (the same state)
 *   G = dyn.ne[0] / 4 >= 1 groups and C % G == 0.
 *
 * Supported domain:
 *   x, out, dyn, state_in, state_out contiguous BF16; static_weight contiguous FP32; scale finite
 *   and positive. Sequence splitting and the zero start state as for short_conv_static.
 *
 * Numeric:
 *   The oracle evaluates tap, acc + x and the product with scale in FP64 from the represented
 *   inputs; out is that value rounded once to BF16. state_out is an exact copy.
 *
 * Effects:
 *   As short_conv_static; dyn and static_weight are unchanged and out overlaps neither.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Stream ordered on `stream`, CUDA Graph capturable.
 */
void short_conv_dynamic(const Tensor& x, const Tensor& static_weight, const Tensor& dyn,
                        float scale, const Tensor& state_in, Tensor& state_out, Tensor& out,
                        cudaStream_t stream);

} // namespace ninfer::ops
