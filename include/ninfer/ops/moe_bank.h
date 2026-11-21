#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight_view.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Routed experts of one expert bank: E' experts with contiguous global ids
// [first_expert, first_expert + E'), stacked in two weight planes, each expert a capped-SwiGLU
// MLP with A16 activations (BF16, never quantized). The bank's routed contribution is written
// or added as an FP32 partial; the block output is rounded to BF16 once, after every bank's
// partial is summed (moe_partial_to_bf16 on one device, allreduce_sum_f32 across two).

// The stacked weights of one bank. Both views are rank-2 matrices covering one contiguous
// range of whole rows of one resident parent (a complete parent is the usual case):
//   gate_up [E'*2F, H]: rows [e'*2F, e'*2F + F) are expert e''s gate rows, the next F its up
//                       rows (gate and up already de-interleaved)
//   down    [E'*H, F]:  rows [e'*H, (e'+1)*H) are expert e''s down rows
// Both parents have the same format, one of:
//   nvfp4 / block_scale_k16_m128x4_v1 (any divisor count the layout admits; parent row r uses
//         divisor word r / (N / divisors), i.e. one word per expert, or per gate and up, when
//         the parent stacks the experts as its sources)
//   mxfp8_e4m3fn_g32_ue8m0 / block_scale_k32_v1 (E4M3FN codes, one UE8M0 exponent per row and
//         32 columns: W[n,c] = e4m3(code[n,c]) * 2^(s[n,c/32] - 127); docs tensor-formats)
//   bf16  / contiguous
// H, F and E' are read from the view shapes: H = gate_up.shape[1], F = down.shape[1],
// E' = down.shape[0] / H, and gate_up.shape[0] must equal E' * 2F.
struct MoeBank {
    const WeightView& gate_up;
    const WeightView& down;
    std::int32_t first_expert = 0;
};

// Caps of the capped SwiGLU: h = softcap(g, gate_cap) * sigmoid(g) * softcap(a, up_cap) with
// softcap(v, c) = c * tanh(v / c).
struct CappedSwiglu {
    float gate_cap = 0.0F;
    float up_cap   = 0.0F;
};

enum class MoeBankOutput : std::uint8_t {
    Store,      // partial  = bank sum
    Accumulate, // partial += bank sum (one FP32 addition per element)
};

/**
 * Workspace bytes of moe_bank_capped_swiglu for every T in [1, max_tokens] with `top_k`
 * positions per token and expert intermediate width F = `intermediate`.
 */
[[nodiscard]] std::size_t moe_bank_capped_swiglu_workspace_bytes(std::int32_t top_k,
                                                                 std::int32_t intermediate,
                                                                 std::int32_t max_tokens);

/**
 * Workspace bytes of moe_bank_capped_swiglu for every T in [1, max_tokens] with hidden width
 * H = `hidden` that let a call wider than 47 tokens run in one pass (its down projection grouped
 * by expert keeps every BF16 output o until each token's positions are summed). With the
 * three-argument size such a call runs in several passes over the tokens: same result, the
 * weights read once per pass. One pass is sized for banks of up to 4096 experts.
 */
[[nodiscard]] std::size_t moe_bank_capped_swiglu_workspace_bytes(std::int32_t top_k,
                                                                 std::int32_t intermediate,
                                                                 std::int32_t hidden,
                                                                 std::int32_t max_tokens);

/**
 * Op: capped-SwiGLU routed experts of one bank, FP32 partial
 *
 * Math / indexing, per token t < T and position j < k, with e = ids[j,t] and
 * b = e - bank.first_expert:
 *   (t,j) is in the bank iff 0 <= b < E' (ids outside, including -1, contribute nothing)
 *   W_g = gate_up rows [b*2F, b*2F+F), W_u = gate_up rows [b*2F+F, (b+1)*2F), W_d = down rows
 *         [b*H, (b+1)*H), each the exact represented value of its format:
 *           nvfp4: W[n,c] = decode_e2m1(code[n,c]) * decode_e4m3fn(scale[n,c/16]) / d_w[n']
 *                  (n' = n's parent row / (parent rows / divisors); docs tensor-formats §3.3)
 *           mxfp8: W[n,c] = e4m3fn(code[n,c]) * 2^(scale[n,c/32] - 127)
 *           bf16:  W[n,c] = the BF16 value
 *   g   = BF16(W_g . x[:,t])                [F]
 *   a   = BF16(W_u . x[:,t])                [F]
 *   h   = BF16(softcap(g, gate_cap) * sigmoid(g) * softcap(a, up_cap))   [F]
 *   o   = BF16(W_d . h)                     [H]
 *   S[:,t] = sum over j with (t,j) in the bank of weights[j,t] * o_j     (FP32; +0 when no
 *            (t,j) of the token is in the bank)
 *   Store: partial[:,t] = S[:,t]
 *   Accumulate: partial[:,t] = fl32(partial[:,t] + S[:,t])
 * An expert repeated in a token row counts with each of its weights.
 *
 * Logical shapes: x BF16 [H,T]; ids I32 [k,T] (global expert ids, -1 = none, as written by
 *   moe_sigmoid_route); weights FP32 [k,T]; partial FP32 [H,T]; bank as MoeBank.
 * Supported domain: T >= 1; 1 <= k <= 64; H % 64 == 0 and F % 64 == 0 (and H <= 23 F for
 *   calls wider than 47 tokens); E' >= 1;
 *   first_expert >= 0; finite gate_cap > 0 and up_cap > 0; contiguous tensors, x 16-byte
 *   aligned; nvfp4 scale words valid (sign-zero finite) and divisors finite positive; mxfp8
 *   codes not NaN, exponents below 255, values representable in BF16; both
 *   parents resident on the stream's device.
 * Numeric: the three BF16 roundings g/a, h and o are semantic boundaries (round to nearest
 *   even); the dot products before them are FP32 accumulations over the exact represented
 *   weights and the represented BF16 activations (the NVFP4 divisor is applied in FP32 to the
 *   accumulated dot, not to each weight); the activation is evaluated in FP32 from the rounded
 *   g and a; S is the FP32 sum of FP32 products weights[j,t] * o_j. Deterministic and
 *   bit-reproducible for given inputs; Accumulate adds exactly the S that Store would write, so
 *   Store(bank 0) then Accumulate(bank 1) on one device gives the same bits as
 *   allreduce_sum_f32 of the two Store partials across two devices. FP64 oracle from the
 *   exact-decoded weights and the represented input, with the three semantic BF16 roundings.
 *   Routes by token count: calls of up to 47 tokens accumulate each dot in one fixed per-lane
 *   order, and the results of a token do not depend on T within that range; wider calls
 *   accumulate on tensor cores in another order, so a token's bits in a call wider than 47
 *   tokens differ from its bits in a narrower call (same numeric bound, both against the same
 *   oracle). Within one route the result does not depend on the workspace size. An mxfp8 bank
 *   gives the same bits as a bf16 bank holding the same (exactly representable) values.
 * Effects: writes `partial` only (every element in Store mode); `partial` must not overlap the
 *   inputs, the weights or the workspace.
 * Workspace: at least moe_bank_capped_swiglu_workspace_bytes(k, F, T) bytes of `workspace` (h of
 *   every (t,j)); moe_bank_capped_swiglu_workspace_bytes(k, F, H, T) bytes let calls wider than
 *   47 tokens run in one pass (faster, same result); no state between calls.
 * Execution: stream ordered, no host synchronization, CUDA Graph capturable (ids and weights
 *   are read at execution; the launch is sized from T and the view shapes).
 */
void moe_bank_capped_swiglu(const Tensor& x, const Tensor& ids, const Tensor& weights,
                            const MoeBank& bank, CappedSwiglu activation, MoeBankOutput mode,
                            Tensor& partial, WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Op: FP32 MoE partial rounded to BF16
 *
 *   y[i] = BF16_RNE(partial[i])
 *
 * The one-device closing step of a block whose banks were summed into one FP32 partial (the
 * two-device form is allreduce_sum_f32 with its BF16 output, which gives the same bits).
 * Logical shapes: partial FP32 [H,T]; y BF16 [H,T]; contiguous, same shape; y must not overlap
 *   partial. Numeric: exact (one RNE rounding). Workspace: none. Execution: stream ordered,
 *   CUDA Graph capturable.
 */
void moe_partial_to_bf16(const Tensor& partial, Tensor& y, cudaStream_t stream);

} // namespace ninfer::ops
