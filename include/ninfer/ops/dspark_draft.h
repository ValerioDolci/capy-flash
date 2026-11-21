#pragma once

// Ops of a DSpark block drafter: the mixing of the target's states into context K/V, the per-head
// norm and RoPE of the block, the block attention over the context and the block, the residual add
// with its norm, and the recurrent head chained over the proposals (rnn_hidden with its output
// projection folded into the vocabulary rows).
//
// Common notation: bf16(x) rounds to nearest even, f32(x) is the exact FP32 value of a stored
// input, rms(x, w) = x * w / sqrt(mean(x^2) + eps) evaluated from the FP32 values. The rotation
// angle of RoPE at position p and pair i < D/2 is the FP32 product
//   angle(p, i) = fl32(f32(p) * inv(i)),   inv(i) = fl32(1 / fl32(theta^(2i/D)))
// (the FP32 cos/sin table of the reference implementations), rotated split-half (neox):
//   out[i] = n[i] cos - n[i + D/2] sin,   out[i + D/2] = n[i + D/2] cos + n[i] sin.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Op: feature mixing with RMSNorm (DFlare)
 *
 * Math: features F BF16 [H, W, S] (block s = state s of W columns, a contiguous [H, W]; blocks at
 * any stride of at least H W), logits BF16 [S, R] (ne {S, R}:
 * row r mixes for output r), norms[r] BF16 [H]:
 *   alpha[r, s] = bf16(softmax_s(f32(logits[r, :])))
 *   m_r(t)      = sum_s f32(F[:, t, s]) * f32(alpha[r, s])                        (FP32)
 *   out[:, t, r] = bf16(rms(m_r(t), norms[r]))        for r < R - 1, every column t
 *   last        = bf16(rms(m_{R-1}(c - 1), norms[R - 1]))  with c = *count, written only if c > 0
 * Supported domain: S = 4, 2 <= R <= 8, H % 8 == 0, W >= 1; out BF16 [H, W, R - 1], last BF16 [H];
 * count device I32 [1] (null: c = W). Outputs contiguous; they overlap no input.
 * Workspace: none. Execution: stream ordered, capturable (count read on the device).
 */
void feature_mix_rmsnorm(const Tensor& features, const Tensor& logits,
                         const std::array<const Tensor*, 8>& norms, std::int32_t rows, float eps,
                         const Tensor* count, Tensor& out, Tensor& last, cudaStream_t stream);

/**
 * Op: consecutive positions
 *
 * Math: out[t] = *start - (minus ? *minus : 0) + t for t < T. out I32 [T], start/minus device
 * I32 [1]. Capturable.
 */
void consecutive_positions(const Tensor& start, const Tensor* minus, Tensor& out,
                           cudaStream_t stream);

/**
 * Op: per-head RMSNorm and RoPE in place (query and key heads of a packed projection)
 *
 * Math: x BF16 [rows, T]; heads [0, nq) of D rows are query heads, [nq, nq + nk) key heads (rows
 * past (nq + nk) D are untouched). For head h of column t at position p = positions[t]:
 *   n = bf16(rms(x_h, h < nq ? q_norm : k_norm));  x_h = bf16(rotate(n, p))
 * Supported: D = 128, rows >= (nq + nk) D, theta > 0. Capturable; no workspace.
 */
void head_rmsnorm_rope(Tensor& x, std::int32_t query_heads, std::int32_t key_heads,
                       const Tensor& q_norm, const Tensor& k_norm, const Tensor& positions,
                       float theta, float eps, cudaStream_t stream);

/** One layer's context cache: K and V BF16, position p in slot p mod capacity. */
struct DraftKVCacheView {
    Tensor k; // BF16 [D, Hkv, capacity]
    Tensor v; // BF16 [D, Hkv, capacity]
    std::int32_t capacity = 0;
};

/**
 * Op: context K/V store (norm, RoPE, cache write) for up to eight layers
 *
 * Math: for layer l, raw[l] BF16 [2 Hkv D, W] (K rows then V rows, as projected), and every
 * column j < c = *count at position p = positions[j]:
 *   K(p) = bf16(rotate(bf16(rms(k_h, k_norm[l])), p)) per KV head h;  V(p) = v_h
 * written into slot p mod capacity of caches[l]; a column whose slot a later live column of the
 * same call also addresses (p < positions[c - 1] + 1 - capacity) is skipped. Other slots keep their
 * values. Supported: D = 128, W >= 1, count device I32 [1]. Capturable; no workspace.
 */
void context_kv_store(const std::array<const Tensor*, 8>& raw,
                      const std::array<const Tensor*, 8>& k_norms,
                      const std::array<DraftKVCacheView, 8>& caches, std::int32_t layers,
                      const Tensor& positions, const Tensor& count, float theta, float eps,
                      cudaStream_t stream);

/**
 * Op: non-causal block attention over a context cache and the block
 *
 * Math: qkv BF16 [rows, B] holds per column the query heads [0, Hq D), the block keys
 * [Hq D, (Hq + Hkv) D) and values [(Hq + Hkv) D, (Hq + 2 Hkv) D) (already normed and rotated). With
 * P = *frontier, keys are the cached positions [lo, P) (lo = max(0, P - window), or 0 when
 * window = 0) followed by the B block keys; every query row sees all of them:
 *   out[:, h, t] = bf16(sum_j softmax_j(scale * f32(q_{h,t}) . f32(k_j)) f32(v_j)), head h reading
 *   KV head h / (Hq / Hkv). FP32 scores and softmax.
 * Supported: D = 128, B <= 8, Hq % Hkv == 0, Hq / Hkv <= 4, capacity >= P - lo (no live slot
 * reused). out BF16 [Hq D, B].
 * Workspace: block_attention_workspace_bytes(Hq, B). Capturable (P read on the device).
 */
void block_attention(const Tensor& qkv, std::int32_t query_heads, std::int32_t kv_heads,
                     const DraftKVCacheView& cache, const Tensor& frontier, std::int32_t window,
                     float scale, WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);
[[nodiscard]] std::size_t block_attention_workspace_bytes(std::int32_t query_heads,
                                                          std::int32_t block);

/**
 * Op: residual add with RMSNorm of the unrounded sum
 *
 * Math: s = f32(residual) + f32(y) (FP32); residual = bf16(s); out = bf16(rms(s, weight)).
 * BF16 [H, T] all; out overlaps neither input. Capturable; no workspace.
 */
void add_rmsnorm(Tensor& residual, const Tensor& y, const Tensor& weight, float eps, Tensor& out,
                 cudaStream_t stream);

/**
 * Op: recurrent head context (the step-independent part of its projection)
 *
 * Math: W BF16 [3r, 2r + 2H] (ne {2r + 2H, 3r}), b BF16 [3r], hidden BF16 [H, G], f BF16 [H]:
 *   out[o, s] = f32(b[o]) + sum_k W[o, 2r + k] hidden[k, s] + sum_k W[o, 2r + H + k] f[k]   (FP32)
 * out FP32 [3r, G]. Capturable; no workspace.
 */
void recurrent_head_context(const Tensor& weight, const Tensor& bias, const Tensor& hidden,
                            const Tensor& context, Tensor& out, cudaStream_t stream);

/**
 * Op: one step of the recurrent head (rnn_hidden) with its folded vocabulary scores, top-1
 *
 * Math, step s over a vocabulary block [first, first + V_r) of a vocabulary split over `ranks`:
 *   prev  = s == 0 ? *anchor : the winner of choice (two slots (value, id) as FP32: slot 1 wins
 * when its value is larger, ties to slot 0); when s > 0 drafts[s - 1] = prev raw   = context[:, s]
 * + W[:, 0:r] state_in + W[:, r:2r] embedding[prev]          (FP32 [3r]) g =
 * bf16(sigmoid(bf16(raw_g))); c = bf16(tanh(bf16(raw_c))) state_out = bf16(bf16(g state_in) +
 * bf16(bf16(1 - g) c));  lat = bf16(tanh(bf16(raw_o))) bias[v] = bf16(f32(folded_bias[v]) + sum_k
 * lat[k] folded[v, k]) logit[v] = bf16(cap tanh(bf16(f32(base[v, s]) + bias[v]) / cap)) choice  =
 * slot `rank`: (max_v logit[v], first + lowest argmax v); the other slot 0 when ranks = 2, (-inf,
 * 0) when ranks = 1 state_in/out BF16 [r] (distinct), embedding BF16 [V, r], folded BF16 [V_r, r],
 * folded_bias BF16 [V_r], base BF16 [V_r, G], context FP32 [3r, G], drafts I32 [G], choice FP32 [4]
 * (in/out), anchor I32 [1]. r = 256. Workspace: recurrent_head_step_workspace_bytes(). Capturable.
 * recurrent_head_select writes drafts[G - 1] from choice.
 */
struct RecurrentHeadOperands {
    Tensor weight, embedding, folded, folded_bias, base, context;
    float cap          = 0;
    std::int32_t first = 0, rank = 0, ranks = 1;
};

void recurrent_head_step(const RecurrentHeadOperands& head, std::int32_t step, const Tensor& anchor,
                         const Tensor& state_in, Tensor& state_out, Tensor& drafts, Tensor& choice,
                         WorkspaceArena& workspace, cudaStream_t stream);
[[nodiscard]] std::size_t recurrent_head_step_workspace_bytes();
void recurrent_head_select(const Tensor& choice, Tensor& drafts, cudaStream_t stream);

} // namespace ninfer::ops
