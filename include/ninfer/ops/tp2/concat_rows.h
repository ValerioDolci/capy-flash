#pragma once

// Two-device (tp 2) gather of a row-split BF16 tensor onto rank 0.

#include "ninfer/ops/allreduce.h"

#include "core/device.h"
#include "core/tensor.h"

#include <array>

namespace ninfer::ops {

/**
 * Op: concatenation of two ranks' row blocks on rank 0
 *
 * Math / indexing (column t; R0, R1 = the ranks' row counts):
 *   out[i, t] = part[0][i, t]        for 0 <= i < R0
 *   out[i, t] = part[1][i - R0, t]   for R0 <= i < R0 + R1
 *   This is how rank 0 assembles vocabulary-split logits: rank r computed rows [r R0, r R0 + R_r)
 *   of the complete [R0 + R1, T] matrix with a column-parallel projection.
 *
 * Logical shapes:
 *   part[r]   [T, R_r]      ne = {R_r, T} on ec.dev[r]; R0, R1 >= 1, T >= 1
 *   staging   [T, R1]       ne = {R1, T} on ec.dev[0]
 *   out       [T, R0 + R1]  ne = {R0 + R1, T} on ec.dev[0]
 *
 * Supported domain:
 *   Every operand contiguous BF16. Requires ec.tp == 2 and a live `events`.
 *
 * Numeric:
 *   Exact copy, bit for bit.
 *
 * Effects:
 *   Writes all of out and staging (whose contents afterwards are unspecified) on rank 0; rank 1
 *   writes nothing. out and staging overlap neither each other nor part[0]. Ordering follows the
 *   pull discipline of allreduce.h with the same events: rank 0's stream reads part[1] after
 *   rank 1's stream completed it (inputs_ready(1)), and rank 1's stream is ordered after that
 *   read (pull_done(0)), so rank 1 may overwrite part[1] afterwards. On return out is ordered on
 *   rank 0's stream. The caller's current device is preserved.
 *
 * Workspace:
 *   staging (caller owned, rank 0).
 *
 * Execution:
 *   Each rank's DeviceContext::stream. Capturable inside one two-device capture; an eager call
 *   with a HostExchange attached to `events` moves part[1] through it, with the same bits.
 */
void concat_rows_rank0(const std::array<Tensor, 2>& part, const Tensor& staging, Tensor& out,
                       const ExecutionContext& ec, const PeerEvents& events);

} // namespace ninfer::ops
