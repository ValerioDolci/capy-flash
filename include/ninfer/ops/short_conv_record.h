#pragma once

#include "core/tensor.h"
#include "ninfer/ops/short_conv.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: residual depthwise causal convolution that records its inputs instead of advancing its state
 *
 * The convolutions of short_conv.h for a call whose tokens may be committed only in part (a
 * speculative verification): the same out, and in place of the state transition the call's whole
 * input history, from which conv_tail_fold later writes the state after any committed prefix.
 *
 * Math / indexing (channel c, token t, u = [state ; x] as in short_conv.h):
 *   out[c,t]      = short_conv_static / short_conv_dynamic of (x, weights, scale, state)
 *   record[c,j]   = u[c, j - 3]       j = 0 .. T + 2: the three state columns, oldest first, then x
 *
 * Logical shapes:
 *   x, out       ne = {C, T}; C >= 1, T >= 1
 *   weight       static: ne = {4, C}; dynamic: static_weight ne = {C, 4} and dyn ne = {4 G, T}
 *   state        ne = {C, 3} (short_conv.h)
 *   record       ne = {C, T + 3}
 *
 * Supported domain:
 *   That of short_conv_static / short_conv_dynamic; record contiguous BF16.
 *
 * Numeric:
 *   out is bit-identical to short_conv_static / short_conv_dynamic on the same operands (the same
 *   arithmetic, one BF16 rounding). record is an exact copy.
 *
 * Effects:
 *   Writes all of out and record. state is read only. out and record overlap no other operand.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   One kernel, stream ordered on `stream`, CUDA Graph capturable.
 */
void short_conv_static_record(const Tensor& x, const Tensor& weight, float scale,
                              const Tensor& state, Tensor& record, Tensor& out,
                              cudaStream_t stream);

void short_conv_dynamic_record(const Tensor& x, const Tensor& static_weight, const Tensor& dyn,
                               float scale, const Tensor& state, Tensor& record, Tensor& out,
                               cudaStream_t stream);

// One state of a conv_tail_fold call: `record` BF16 ne = {channels, width + 3} written by a
// *_record convolution, `tail` its state BF16 ne = {channels, 3}.
struct ConvTailFoldEntry {
    const void* record    = nullptr;
    void* tail            = nullptr;
    std::int64_t channels = 0;
};

/**
 * Op: conv_tail_fold - the convolution states after the first n tokens of recorded calls
 *
 * Math / indexing (entry e, channel c, n = committed[0] clamped to [0, width]):
 *   tail_e[c,k] = record_e[c, n + k]      k = 0, 1, 2
 *
 *   That is the state short_conv_static / short_conv_dynamic leave after the first n tokens of
 *   the recorded call (n = width: after all of them; n = 0: the state the call started from).
 *
 * Logical shapes:
 *   entries     device array of `count` ConvTailFoldEntry; each record ne = {channels, width + 3},
 *               each tail ne = {channels, 3}, both contiguous BF16
 *   committed   device I32 [1]
 *
 * Supported domain:
 *   count >= 1, width >= 1, every channels >= 1. Records and tails pairwise disjoint.
 *
 * Effects:
 *   Writes every tail; records, entries and committed are unchanged.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   One kernel; n is read on the device, so the call is CUDA Graph capturable and replays for
 *   any committed count.
 */
void conv_tail_fold(const ConvTailFoldEntry* entries, std::int32_t count, std::int32_t width,
                    const Tensor& committed, cudaStream_t stream);

} // namespace ninfer::ops
