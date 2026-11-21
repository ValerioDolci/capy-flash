#pragma once

// Block attention residuals (depth attention over the completed blocks), the cross-pass messages
// and their gates, the attention-input norms, and the static short convolution of the residual
// stream. All replicated: every rank issues the same Ops on its own copy.
//
// Member definitions of Forward (forward.h), included only by forward.cpp.

#include "models/inkling2/execution/forward.h"

#include "ninfer/ops/depth_attention.h"
#include "ninfer/ops/message_gate.h"
#include "ninfer/ops/residual_mix.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rmsnorm_f32.h"
#include "ninfer/ops/short_conv.h"
#include "ninfer/ops/short_conv_record.h"

#include <array>
#include <cmath>
#include <span>
#include <stdexcept>

namespace ninfer::models::inkling2::execution {

// The block residuals' and messages' RMS (no gain) epsilon: a constant of the reference's
// unweighted norm, not the config's rms_norm_eps.
inline constexpr float kDepthAttentionEpsilon = 1e-6F;

// Up to this many columns (decode and verification) a visit's gates and their norms run as one
// launch whose clusters run concurrently; wider calls keep one gate per launch.
inline constexpr std::int32_t kBatchedGateTokens = 16;

template <class Arena>
void Forward<Arena>::depth(const std::vector<const RankTensors*>& states,
                           const RankTensors& current, std::array<const Tensor*, 2> queries,
                           RankTensors& out) {
    if (states.size() > static_cast<std::size_t>(ops::kDepthAttentionMaxStates)) {
        throw std::logic_error("Inkling 2 forward: more block states than depth attention takes");
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>(hidden()));
    each([&](int r) {
        if constexpr (kExecute) {
            std::array<Tensor, ops::kDepthAttentionMaxStates> rank_states;
            for (std::size_t i = 0; i < states.size(); ++i) {
                rank_states[i] = (*states[i])[at(r)];
            }
            ops::depth_attention(std::span<const Tensor>(rank_states.data(), states.size()),
                                 current[at(r)], *queries[at(r)], kDepthAttentionEpsilon, scale,
                                 out[at(r)], s(r));
        } else {
            (void)current, (void)queries, (void)out, (void)scale;
        }
    });
}

template <class Arena>
bool Forward<Arena>::fused_residual(std::uint32_t slot) const {
    return kExecute && tap_ == nullptr && T_ <= ops::kResidualMixMaxTokens &&
           hidden() <= ops::kResidualMixMaxHidden && (T_ == 1 || records(0, slot).has_value());
}

template <class Arena>
void Forward<Arena>::attention_inputs(std::uint32_t index, Block& b, const RankTensors& x,
                                      std::array<RankTensors, 3>& xin) {
    const Visit& v       = config_.visits[index];
    const std::int32_t H = hidden();
    const auto vi        = static_cast<std::int32_t>(index);
    const float eps      = config_.rms_norm_eps;
    const bool global    = v.kind == LayerKind::Global;
    for (auto& t : xin) { t = alloc(DType::BF16, {H, T_}); }
    if (v.layer == 0) {
        each([&](int r) {
            if constexpr (kExecute) {
                ops::rmsnorm(x[at(r)], layer(r, 0).input_norms.at(0), eps, false, xin[0][at(r)],
                             s(r));
            }
        });
        xin[1] = xin[0];
        xin[2] = xin[0];
        return;
    }
    const auto Q = static_cast<std::int32_t>(config_.attention_queries(v.layer));
    auto a       = alloc(DType::FP32, {H, T_, Q});
    // Outside the message passes the mix and the input norms run as one Op.
    if (!v.messages && fused_residual(v.slot)) {
        const bool shared = !global;
        each([&](int r) {
            std::array<Tensor, ops::kDepthAttentionMaxStates> rank_states;
            for (std::size_t i = 0; i < b.states.size(); ++i) {
                rank_states[i] = (*b.states[i])[at(r)];
            }
            const auto& norms = layer(r, v.layer).input_norms;
            std::array<Tensor, 3> weights, outs;
            for (int j = 0; j < Q; ++j) {
                weights[at(j)] = norms.at(global ? at(j) : 0);
                outs[at(j)]    = xin[at(j)][at(r)];
            }
            ops::depth_attention_rmsnorm(
                std::span<const Tensor>(rank_states.data(), b.states.size()), x[at(r)],
                *layer(r, v.layer).attention_queries, kDepthAttentionEpsilon,
                1.0F / std::sqrt(static_cast<float>(H)),
                std::span<const Tensor>(weights.data(), at(Q)), eps,
                std::span<Tensor>(outs.data(), at(Q)), s(r));
        });
        if (shared) {
            xin[1] = xin[0];
            xin[2] = xin[0];
        }
        return;
    }
    depth(b.states, x,
          {&*layer(0, v.layer).attention_queries,
           width_ == 2 ? &*layer(1, v.layer).attention_queries : nullptr},
          a);
    tap(vi, "ain", a);
    // In the later passes each attention input is gated by its message first.
    std::array<RankTensors, 3> gated;
    if (v.messages) {
        if (b.previous == nullptr) {
            throw std::logic_error("Inkling 2 forward: a message visit without a previous pass");
        }
        b.messages_storage = alloc(DType::FP32, {H, T_, 4});
        b.messages         = &b.messages_storage;
        depth(b.bank, *b.previous,
              {&layer(0, v.layer).core->message_queries,
               width_ == 2 ? &layer(1, v.layer).core->message_queries : nullptr},
              b.messages_storage);
        tap(vi, "msg", b.messages_storage);
        for (auto& t : gated) { t = alloc(DType::FP32, {H, T_}); }
        if (T_ <= kBatchedGateTokens) {
            each([&](int r) {
                if constexpr (kExecute) {
                    const auto& norms = layer(r, v.layer).input_norms;
                    std::array<ops::MessageGateNormed, 3> batch;
                    for (int j = 0; j < 3; ++j) {
                        const auto& gate = layer(r, v.layer).core->gates[at(j)];
                        batch[at(j)]     = {a[at(r)].slice(2, global ? j : 0, 1).view({H, T_}),
                                            b.messages_storage[at(r)].slice(2, j, 1).view({H, T_}),
                                            gate.down,
                                            gate.forget,
                                            gate.gain,
                                            gated[at(j)][at(r)],
                                            norms.at(global ? at(j) : 0),
                                            xin[at(j)][at(r)]};
                    }
                    ops::message_gates_rmsnorm(batch, eps, eps, s(r));
                }
            });
            return;
        }
        each([&](int r) {
            if constexpr (kExecute) {
                for (int j = 0; j < 3; ++j) {
                    const auto& gate = layer(r, v.layer).core->gates[static_cast<std::size_t>(j)];
                    const Tensor hidden_in = a[at(r)].slice(2, global ? j : 0, 1).view({H, T_});
                    const Tensor message   = b.messages_storage[at(r)].slice(2, j, 1).view({H, T_});
                    ops::message_gate(hidden_in, message, gate.down, gate.forget, gate.gain, eps,
                                      gated[at(j)][at(r)], s(r));
                }
            }
        });
    }
    const bool shared = !global && !v.messages;
    each([&](int r) {
        if constexpr (kExecute) {
            const auto& norms = layer(r, v.layer).input_norms;
            for (int j = 0; j < (shared ? 1 : 3); ++j) {
                const Tensor in = v.messages ? gated[at(j)][at(r)]
                                             : a[at(r)].slice(2, global ? j : 0, 1).view({H, T_});
                ops::rmsnorm_f32(in, norms.at(global ? at(j) : 0), eps, xin[at(j)][at(r)], s(r));
            }
        }
    });
    if (shared) {
        xin[1] = xin[0];
        xin[2] = xin[0];
    }
}

template <class Arena>
void Forward<Arena>::short_conv(int r, const Tensor& x, const Tensor& weight, const Tensor& state,
                                const Tensor* record, Tensor& out) {
    const float scale = 1.0F / std::sqrt(static_cast<float>(config_.conv_kernel + 1));
    if (record != nullptr) {
        Tensor history = *record;
        ops::short_conv_static_record(x, weight, scale, state, history, out, s(r));
        return;
    }
    Tensor tail = state;
    ops::short_conv_static(x, weight, scale, tail, tail, out, s(r));
}

} // namespace ninfer::models::inkling2::execution
