#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/moe_slots.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace ninfer::ops {

// Routed-expert MoE over an expert slot pool for prefill-width calls (any T), on NVFP4 tensor
// cores: the wide counterpart of moe_slots_nvfp4 (same geometry, slot table, divisors and W4A4
// activation recipe), grouped by expert instead of by token.

// Columns of one piece. A wider call quantizes and groups all its columns and runs gate/up once
// over the whole call (each expert's gate/up weights read once per call), then the down
// projection and the merge per piece of this width (their FP32 per-assignment outputs are what
// the workspace holds per piece). Bit-identical to calling the pieces separately.
inline constexpr std::int32_t kMoeGroupedChunkColumns = 2048;

/**
 * Workspace bytes of moe_grouped_nvfp4 for every T in [1, max_tokens].
 */
[[nodiscard]] std::size_t moe_grouped_nvfp4_workspace_bytes(std::int32_t max_tokens);

/**
 * Op: NVFP4 (W4A4) routed-expert MoE partial over the experts of a slot pool, grouped by expert
 *
 * Math / indexing: as moe_slots_nvfp4 with MoeSlotsOutput::Store, per token t < T and position
 * k < 10, with e = routed_ids[k,t]:
 *   s(k,t) = slot_table[e] if 0 <= e < E and 0 <= slot_table[e] < pool.slots, else none
 *   partial[:,t] = sum over k with s(k,t) of routed_weights[k,t] *
 *                  W_down(s) . ( silu(W_gate(s) x[:,t]) * (W_up(s) x[:,t]) )
 *   out[:,t] = partial[:,t]   (0 for a token without any resident expert)
 * Divisors by expert id: gate_up_divisors[2e] (gate), [2e+1] (up), down_divisors[e].
 *
 * Logical shapes: x BF16 [2560,T]; routed_ids I32 [10,T]; routed_weights FP32 [10,T];
 *   out FP32 [2560,T].
 * Supported domain: T >= 1; E == 512; 1 <= pool.slots and pool.slots * 2560 < 2^31; every plane's
 *   stride at least its per-expert bytes and a multiple of 16 (planar: one array per plane, the
 *   stride = the plane's bytes; slot-major: one contiguous slot of all four planes, every stride =
 *   the slot's bytes);
 *   finite positive input divisors. Slot contents must be valid NVFP4 (finite scales).
 * Numeric (W4A4, the recipe of moe_slots_nvfp4): x and h quantized to E2M1 with one E4M3FN scale
 *   per 16 values against the layer's input divisors; block products on the NVFP4 tensor cores
 *   with FP32 accumulation; gate/up outputs and h rounded to BF16 before quantization; each
 *   expert's weighted down output stays FP32 and the partial is the FP32 sum over k in top-k
 *   order (deterministic; independent of the slot that holds an expert and of the grouping
 *   order). Same recipe as moe_slots_nvfp4 (the two differ only in FP32 accumulation order and
 *   in ties of the h quantization: the reciprocal here, a division there).
 *   FP64 oracle from represented inputs with unquantized activations.
 * Effects: writes `out` only; must not overlap any input or the workspace.
 * Workspace: moe_grouped_nvfp4_workspace_bytes(T) bytes of `workspace`, no state between calls.
 * Execution: stream ordered, no host synchronization, CUDA Graph capturable (the grid is sized
 *   from T; the slot table is read at execution).
 */
void moe_grouped_nvfp4(const Tensor& x, const Tensor& routed_ids, const Tensor& routed_weights,
                       const MoeSlotsLayer& layer, const MoeSlotPool& pool, Tensor& out,
                       WorkspaceArena& workspace, cudaStream_t stream);

// Columns [first, first + count) of `out` are final once the work enqueued before this call runs.
using MoeGroupedPieceDone = std::function<void(std::int32_t first, std::int32_t count)>;

/**
 * moe_grouped_nvfp4 that reports each piece: a call wider than kMoeGroupedChunkColumns calls
 * `piece_done` on the host, in column order, right after enqueuing the down projection and the
 * merge of each piece (gate/up has run once over the whole call before the first piece), so the
 * caller can enqueue work on the final columns while later pieces run; a narrower call reports
 * [0, T) once. Same results as moe_grouped_nvfp4.
 */
void moe_grouped_nvfp4(const Tensor& x, const Tensor& routed_ids, const Tensor& routed_weights,
                       const MoeSlotsLayer& layer, const MoeSlotPool& pool, Tensor& out,
                       WorkspaceArena& workspace, cudaStream_t stream,
                       const MoeGroupedPieceDone& piece_done);

} // namespace ninfer::ops
