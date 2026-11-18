#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/moe_slots.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Routed-expert MoE over ggml block banks (the GSQ-RCO formats), at the Qwen3.8-Flash-Next
// geometry (hidden 2560, expert intermediate 640, top-10): the counterparts of moe_slots_nvfp4
// (decode windows, T <= 8) and moe_grouped_nvfp4 (prefill widths) for experts stored as verbatim
// GGUF blocks (artifact layout gguf_blocks_v1).

// ggml type ids of the supported block formats.
enum class MoeGgufFormat : std::int32_t {
    IQ3_XXS = 18,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    Q2_0    = 42,
};

// Bytes of one row of `k` values (k a multiple of the format's block), or 0 for an unknown format.
[[nodiscard]] std::int64_t moe_gguf_row_bytes(MoeGgufFormat format, std::int32_t k) noexcept;
// gate/up: IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL (K = 2560); down: IQ4_NL, Q2_0 (K = 640).
[[nodiscard]] bool moe_gguf_gate_up_supported(MoeGgufFormat format) noexcept;
[[nodiscard]] bool moe_gguf_down_supported(MoeGgufFormat format) noexcept;
// Bytes of one expert: 1280 gate|up rows of 2560 and 2560 down rows of 640.
[[nodiscard]] std::int64_t moe_gguf_expert_bytes(MoeGgufFormat gate_up,
                                                 MoeGgufFormat down) noexcept;

// One layer's experts. An expert is resident in slot s = slot_table[e] (-1, or s outside
// [0, slots): not here). Its rows are found in one of two ways:
//   blob     (slot_offset != nullptr): the expert's blob begins at base + slot_offset[s] (device
//            int64 table over the GPU's whole slot pool, 16-byte aligned values): 640 gate rows,
//            640 up rows, then 2560 down rows, each row its verbatim ggml blocks;
//   planar   (slot_offset == nullptr): the gate|up rows at gate_up_base + s * 1280 * gu_row and the
//            down rows at down_base + s * 2560 * down_row (a layer's host-mapped bank read in
//            place, slot_table = identity: the Mapped route and GPU reads of missing experts).
struct MoeGgufLayer {
    Tensor slot_table; // I32 [E]
    std::int32_t slots              = 0;
    const std::int64_t* slot_offset = nullptr;
    const std::byte* base           = nullptr; // blob form
    const std::byte* gate_up_base   = nullptr; // planar form
    const std::byte* down_base      = nullptr; // planar form
    MoeGgufFormat gate_up_format    = MoeGgufFormat::IQ3_XXS;
    MoeGgufFormat down_format       = MoeGgufFormat::IQ4_NL;
    // moe_grouped_gguf only, blob form (prefill streaming): an expert without a
    // slot whose host_table[e] >= 0 (device I32 [E]) is read in place from the planar bases at row
    // block host_table[e] (the layer's host-mapped bank, zero-copy over PCIe); -1 = not here.
    // Null = blob experts only.
    const std::int32_t* host_table = nullptr;
};

/**
 * Workspace bytes of moe_slots_gguf for every T in [1, max_tokens].
 */
[[nodiscard]] std::size_t moe_slots_gguf_workspace_bytes(std::int32_t max_tokens);

/**
 * Op: GGUF routed-expert MoE partial over the resident experts of one layer (decode windows)
 *
 * Math / indexing, per token t < T and position k < 10, with e = routed_ids[k,t]:
 *   s(k,t) = slot_table[e] if 0 <= e < E and 0 <= slot_table[e] < layer.slots, else none
 *   W[n,c]  = the ggml dequantization of the stored blocks (dequantize_row_* of ggml-quants.c)
 *   partial[:,t] = sum over k with s(k,t) of routed_weights[k,t] *
 *                  W_down(s) . ( silu(W_gate(s) x[:,t]) * (W_up(s) x[:,t]) )
 *   Store: out[:,t] = partial[:,t]; Accumulate: out[:,t] += partial[:,t].
 * Positions without a slot contribute nothing (the caller computes missing experts elsewhere);
 * a token without any resident expert gets 0 (Store) or keeps `out` (Accumulate).
 *
 * Logical shapes: x BF16 [2560,T]; routed_ids I32 [10,T]; routed_weights FP32 [10,T];
 *   out FP32 [2560,T] (or BF16 with Store).
 * Supported domain: 1 <= T <= 8; E >= 1; slots >= 0; supported formats; blob offsets 16-byte
 *   aligned; x and out 16-byte aligned.
 * Numeric (fngine's decode recipe, ggml vec_dot_*_q8_1 arithmetic): x and h quantized per 32
 *   values to int8 with an FP32 scale, d = amax / 127, q = roundf(v / d) (0 for an all-zero
 *   block); integer sub-block dot products of the stored codes (dp4a) with the IQ2_S/IQ3_XXS
 *   half-scale terms exact; FP32 accumulation; h = silu(g) * u in FP32; the partial is the FP32
 *   sum over k in top-k order (deterministic, independent of which slot holds an expert).
 *   The host path (ops/host_moe/host_gguf.h, quantize_pack_*_q8f32) quantizes x and h with
 *   the same rule, so CPU and GPU see the same integer codes; their FP32 sums differ in order.
 *   Oracle: FP64 over ggml-dequantized weights and the op's own quantized activations.
 * Effects: writes `out` only; must not overlap any input or the workspace.
 * Workspace: moe_slots_gguf_workspace_bytes(T) bytes of `workspace`, no state between calls.
 * Execution: stream ordered, CUDA Graph capturable (table, offsets and slots read at replay),
 *   kernels launched as programmatic dependents (PDL) of the preceding kernel.
 */
void moe_slots_gguf(const Tensor& x, const Tensor& routed_ids, const Tensor& routed_weights,
                    const MoeGgufLayer& layer, MoeSlotsOutput mode, Tensor& out,
                    WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Workspace bytes of moe_grouped_gguf for every T in [1, max_tokens].
 */
[[nodiscard]] std::size_t moe_grouped_gguf_workspace_bytes(std::int32_t max_tokens);

/**
 * Op: GGUF routed-expert MoE partial for prefill-width calls (any T), grouped by expert
 *
 * Math / indexing: as moe_slots_gguf with MoeSlotsOutput::Store.
 * Logical shapes: x BF16 [2560,T]; routed_ids I32 [10,T]; routed_weights FP32 [10,T];
 *   out FP32 [2560,T].
 * Supported domain: T >= 1; 1 <= E <= 1024 (experts are grouped by id); supported formats; blob
 *   offsets 16-byte aligned; x and out 16-byte aligned. Calls wider than 2048 columns run as
 *   pieces of 2048 (same result as separate calls).
 * Numeric (fngine's prefill recipe): weights dequantized exactly and rounded to BF16, BF16
 *   activations, BF16 tensor-core products with FP32 accumulation; h rounded to BF16; the
 *   partial is deterministic (independent of grouping and slot placement): each assignment's down
 *   output stays FP32 and the partial is the FP32 sum over k in top-k order.
 * Effects: writes `out` only; must not overlap any input or the workspace.
 * Workspace: moe_grouped_gguf_workspace_bytes(T) bytes of `workspace`, no state between calls.
 * Execution: stream ordered, no host synchronization, CUDA Graph capturable.
 */
void moe_grouped_gguf(const Tensor& x, const Tensor& routed_ids, const Tensor& routed_weights,
                      const MoeGgufLayer& layer, Tensor& out, WorkspaceArena& workspace,
                      cudaStream_t stream);

} // namespace ninfer::ops
