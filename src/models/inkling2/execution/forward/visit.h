#pragma once

// The schedule: the normed embedding, the 48 visits in execution order with the block states and
// the core's passes, then the head. One visit:
//
//   a      = x (layer 0) or depth attention of the block states and x      [replicated]
//   xq,k,v = norms of a (in the later passes: of a gated by the messages)  [replicated]
//   y      = attention, output projection, all-reduce                      [heads per rank]
//   y'     = post-attention convolution (the slot's tail)                  [replicated]
//   r      = f32(x) + f32(y'), or f32(y') at a block boundary (B += [x])   [replicated]
//   u      = norm(depth attention of B and r, gated by the FFN message)    [replicated]
//   z      = MoE: router, the rank's bank, FP32 all-reduce rounded once    [bank per rank]
//   x'     = bf16(r + f32(post-FFN convolution of z))                      [replicated]
//
// Member definitions of Forward (forward.h), included only by forward.cpp.

#include "models/inkling2/execution/forward.h"

#include "ninfer/ops/embedding.h"
#include "ninfer/ops/loop_contrast.h"
#include "ninfer/ops/residual_f32.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rmsnorm_f32.h"
#include "ninfer/ops/message_gate.h"
#include "ninfer/ops/residual_mix.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::inkling2::execution {

// The call's block states. `keep` holds the outputs of the visits that end a block (layer + 1 a
// multiple of block_layers): they are the block sources of later visits, the pre-core state, the
// message bank and the previous pass's output, so they live for the whole call; every other
// visit output alternates between two temporaries. With the loop contrast, `guided` holds the
// Hidden extrapolations for the whole call, and the Logits reference coda leaves its block states
// and output for the reference head.
template <class Arena>
struct Forward<Arena>::Block {
    RankTensors embedding, temporary[2];
    std::vector<RankTensors> keep;
    std::vector<const RankTensors*> states; // B: completed blocks
    std::vector<const RankTensors*> bank;   // messages: the previous pass's later blocks
    const RankTensors* previous = nullptr;  // messages: the previous pass's output
    const RankTensors* messages = nullptr;  // this visit's messages, FP32 {H, T, 4}
    RankTensors messages_storage;
    std::vector<RankTensors> guided;
    std::vector<const RankTensors*> reference_states;
    const RankTensors* reference_output = nullptr;
};

template <class Arena>
void Forward<Arena>::run(const RankTensors& ids, const ForwardColumns& columns,
                         const ForwardLogits& out, ForwardTap* taps) {
    if constexpr (kExecute) {
        if (columns.width <= 0 || columns.width > max_tokens_ || columns.first_position < 0 ||
            columns.logit_columns < 0 || columns.logit_columns > columns.width ||
            columns.logit_columns > max_logit_columns_) {
            throw std::invalid_argument("Inkling 2 forward columns are out of the planned bounds");
        }
        for (int r = 0; r < width_; ++r) {
            const Rank& rank = r_[at(r)];
            if (rank.state == nullptr || rank.positions == nullptr ||
                rank.positions->ne[0] != columns.width ||
                std::uint64_t(columns.first_position) + std::uint64_t(columns.width) >
                    rank.state->max_context()) {
                throw std::invalid_argument("Inkling 2 forward rank " + std::to_string(r) +
                                            " has no state or positions for the columns");
            }
            const FeatureSink* sink = rank.features;
            if ((sink == nullptr) != (r_[0].features == nullptr) ||
                (sink != nullptr &&
                 (sink->features.dtype != DType::BF16 || sink->features.nb[0] != 2 ||
                  sink->features.nb[1] != 2 * std::int64_t(hidden()) ||
                  sink->features.nb[2] < 2 * std::int64_t(hidden()) * columns.width ||
                  sink->features.ne[0] != hidden() || sink->features.ne[1] != columns.width ||
                  sink->features.ne[2] != 4 || sink->features.ne[3] != 1 ||
                  sink->visits != r_[0].features->visits ||
                  std::any_of(
                      sink->visits.begin(), sink->visits.end(),
                      [&](std::uint32_t v) { return v >= config_.visits.size(); })))) {
                throw std::invalid_argument("Inkling 2 forward rank " + std::to_string(r) +
                                            ": feature sinks must be BF16 [H, T, 4] on every "
                                            "rank, for the same visits");
            }
        }
        T_ = columns.width;
    } else {
        T_ = max_tokens_;
    }
    first_position_ = columns.first_position;
    tap_            = taps;

    const auto held      = scopes();
    const std::int32_t H = hidden();
    Block b;
    b.embedding      = alloc(DType::BF16, {H, T_});
    b.temporary[0]   = alloc(DType::BF16, {H, T_});
    b.temporary[1]   = alloc(DType::BF16, {H, T_});
    std::size_t kept = 0;
    for (const Visit& v : config_.visits) { kept += (v.layer + 1) % config_.block_layers == 0; }
    b.keep.reserve(kept);
    for (std::size_t i = 0; i < kept; ++i) { b.keep.push_back(alloc(DType::BF16, {H, T_})); }
    if (config_.contrast.mode == LoopContrast::Hidden) {
        // The last pass's output and the block states it produced after its input (which every
        // pass shares): one per block of the core.
        const std::uint32_t guided = config_.core_layers() / config_.block_layers;
        b.guided.reserve(guided);
        for (std::uint32_t i = 0; i < guided; ++i) {
            b.guided.push_back(alloc(DType::BF16, {H, T_}));
        }
    }

    embed(ids, b.embedding);
    tap(ForwardTap::kHeadTap, "embed", b.embedding);

    const std::int32_t split = overlap_split(T_);
    if constexpr (!kExecute) {
        // The workspace holds either schedule of a call that may overlap.
        if (split > 0) { (void)visits(b, 0); }
    }
    const RankTensors* x = visits(b, split);
    head(b, *x, columns, out);
    join_prefetch();
}

template <class Arena>
auto Forward<Arena>::visits(Block& b, std::int32_t split) -> const RankTensors* {
    b.states.clear();
    b.bank.clear();
    b.previous      = nullptr;
    b.messages      = nullptr;
    const auto held = scopes();
    std::array<Half, 2> parts;
    if (split > 0) { parts = halves(split); }
    const RankTensors* x = &b.embedding;
    // Saved at the first core visit of pass 1, restored at the start of every later pass.
    const RankTensors* core_input = nullptr;
    std::size_t core_blocks       = 0;
    // The block states and output at the end of the latest pass.
    std::vector<const RankTensors*> pass_states;
    const RankTensors* pass_output = nullptr;
    std::size_t next_keep          = 0;
    // The feature sinks: the input of a captured visit is computed into its block when it is a
    // temporary, copied there otherwise.
    std::array<std::optional<RankTensors>, 3> sinks{sink(0), sink(1), sink(2)};
    const auto sink_of = [&](std::uint32_t index) -> RankTensors* {
        if (!sinks[0] || index >= config_.visits.size()) { return nullptr; }
        for (std::size_t j = 0; j < sinks.size(); ++j) {
            if (r_[0].features->visits[j] == index) { return &*sinks[j]; }
        }
        return nullptr;
    };
    const auto restarts = [&](std::uint32_t index) {
        return index < config_.visits.size() && config_.visits[index].layer == config_.core_start &&
               config_.visits[index].pass > 1;
    };
    const auto close_halves = [&] {
        try {
            for (Half& half : parts) {
                on_half(half, [&] { close_half(half); });
            }
        } catch (...) {
            abandon_halves(parts);
            throw;
        }
    };
    // The loop contrast: the block states and output at the end of the reference pass, and the
    // main coda's while the Logits reference coda runs.
    const LoopContrastOptions& contrast = config_.contrast;
    std::vector<const RankTensors*> reference_states, main_states;
    const RankTensors* reference_output = nullptr;
    const RankTensors* main_output      = nullptr;
    for (std::uint32_t index = 0; index < config_.visits.size(); ++index) {
        const Visit& v = config_.visits[index];
        if (contrast.mode == LoopContrast::Hidden && v.layer == config_.core_end) {
            // The extrapolation reads the last pass's output whole.
            if (split > 0) { close_halves(); }
            extrapolate(b, x, reference_states, *reference_output);
        }
        if (v.reference && main_output == nullptr) {
            main_output = x;
            main_states = b.states;
            x           = reference_output;
            b.states    = reference_states;
        }
        if (v.layer == config_.core_start) {
            if (v.pass == 1) {
                core_input  = x;
                core_blocks = b.states.size();
            } else {
                b.bank.assign(pass_states.begin() + static_cast<std::ptrdiff_t>(core_blocks) + 1,
                              pass_states.end());
                b.previous = pass_output;
                x          = core_input;
                b.states.resize(core_blocks);
            }
        }
        RankTensors* captured = sink_of(index);
        if (captured == x) { captured = nullptr; }
        if (captured != nullptr && split == 0) {
            each([&](int r) {
                if constexpr (kExecute) {
                    CUDA_CHECK(::ninfer::memcpy_device_async(
                        (*captured)[at(r)].data, (*x)[at(r)].data, (*x)[at(r)].bytes(), s(r)));
                }
            });
        }
        RankTensors* target = nullptr;
        if ((v.layer + 1) % config_.block_layers == 0) {
            target = &b.keep.at(next_keep++);
        } else if (RankTensors* next = restarts(index + 1) ? nullptr : sink_of(index + 1)) {
            target = next;
        } else {
            target = x == &b.temporary[0] ? &b.temporary[1] : &b.temporary[0];
        }
        if (split == 0) {
            visit(index, b, *x, *target);
        } else {
            try {
                visit_halves(index, b, *x, *target, captured, parts);
            } catch (...) {
                abandon_halves(parts);
                throw;
            }
        }
        x = target;
        if (config_.core(v.layer) && v.layer + 1 == config_.core_end) {
            pass_states = b.states;
            pass_output = x;
            if (contrast.mode != LoopContrast::Off && v.pass == contrast.reference_pass) {
                reference_states = b.states;
                reference_output = x;
            }
        }
    }
    // The last visit's exchanges close before the head reads its output.
    if (split > 0) { close_halves(); }
    if (main_output != nullptr) {
        b.reference_states = b.states;
        b.reference_output = x;
        b.states           = main_states;
        x                  = main_output;
    }
    return x;
}

template <class Arena>
void Forward<Arena>::extrapolate(Block& b, const RankTensors*& x,
                                 const std::vector<const RankTensors*>& reference_states,
                                 const RankTensors& reference_output) {
    if (reference_states.size() != b.states.size()) {
        throw std::logic_error("Inkling 2 forward: the reference pass has other block states");
    }
    const float w     = config_.contrast.strength;
    std::size_t next  = 0;
    const auto guided = [&](const RankTensors& last, const RankTensors& reference) {
        RankTensors& out = b.guided.at(next++);
        each([&](int r) {
            if constexpr (kExecute) {
                ops::loop_extrapolate(last[at(r)], reference[at(r)], w, out[at(r)], s(r));
            } else {
                (void)last, (void)reference, (void)w;
            }
        });
        return &out;
    };
    for (std::size_t i = 0; i < b.states.size(); ++i) {
        if (b.states[i] != reference_states[i]) {
            b.states[i] = guided(*b.states[i], *reference_states[i]);
        }
    }
    x = guided(*x, reference_output);
}

template <class Arena>
void Forward<Arena>::embed(const RankTensors& ids, RankTensors& x) {
    const auto held = scopes();
    auto e          = alloc(DType::BF16, {hidden(), T_});
    each([&](int r) {
        if constexpr (kExecute) {
            ops::embedding(ids[at(r)], embedding_[at(r)], e[at(r)], s(r));
            ops::rmsnorm(e[at(r)], text(r).embedding_norm, config_.rms_norm_eps, false, x[at(r)],
                         s(r));
        }
    });
}

template <class Arena>
void Forward<Arena>::visit(std::uint32_t index, Block& b, const RankTensors& x, RankTensors& out) {
    const Visit& v       = config_.visits[index];
    const std::int32_t H = hidden();
    const auto held      = scopes();
    const auto vi        = static_cast<std::int32_t>(index);

    std::array<RankTensors, 3> xin;
    for (auto& t : xin) { t = alloc(DType::BF16, {H, T_}); }
    attention_inputs(index, b, x, xin);

    auto y = alloc(DType::BF16, {H, T_});
    attention(index, xin, y);
    tap(vi, "attn", y);

    // Fused: conv + residual + mix (+ the norm outside the message passes) in one Op per rank.
    const bool fused       = fused_residual(index);
    const float conv_scale = 1.0F / std::sqrt(static_cast<float>(config_.conv_kernel + 1));
    const auto conv        = [&](const Tensor& in, const Tensor& weight, const Tensor& tail,
                          const std::optional<Tensor>& record) {
        return ops::ResidualConv{in, weight, conv_scale, tail, record ? *record : Tensor{}};
    };
    auto yc = alloc(DType::BF16, {H, T_});
    if (!fused) {
        each([&](int r) {
            if constexpr (kExecute) {
                const auto record = records(r, v.slot);
                short_conv(r, y[at(r)], layer(r, v.layer).post_attention_conv,
                           state(r).tails(v.slot).attention, record ? &record->attention : nullptr,
                           yc[at(r)]);
            }
        });
    }
    tap(vi, "attn_conv", yc);

    auto residual = alloc(DType::FP32, {H, T_});
    if (!fused) {
        each([&](int r) {
            if constexpr (kExecute) {
                ops::residual_add_f32(v.block_boundary ? nullptr : &x[at(r)], yc[at(r)],
                                      residual[at(r)], s(r));
            }
        });
    }
    const bool boundary = v.block_boundary;
    if (v.block_boundary) { b.states.push_back(&x); }

    auto ffn_in      = alloc(DType::FP32, {H, T_, 1});
    auto u           = alloc(DType::BF16, {H, T_});
    bool normed_here = false; // u written with the mix
    if (fused) {
        each([&](int r) {
            if constexpr (kExecute) {
                std::array<Tensor, ops::kDepthAttentionMaxStates> rank_states;
                for (std::size_t i = 0; i < b.states.size(); ++i) {
                    rank_states[i] = (*b.states[i])[at(r)];
                }
                const auto record = records(r, v.slot);
                const Tensor norm = layer(r, v.layer).ffn_norm;
                Tensor normed     = u[at(r)];
                ops::residual_conv_mix(
                    conv(y[at(r)], layer(r, v.layer).post_attention_conv,
                         state(r).tails(v.slot).attention,
                         record ? std::optional<Tensor>(record->attention) : std::nullopt),
                    boundary ? nullptr : &x[at(r)],
                    std::span<const Tensor>(rank_states.data(), b.states.size()),
                    layer(r, v.layer).ffn_queries, kDepthAttentionEpsilon,
                    1.0F / std::sqrt(static_cast<float>(H)),
                    v.messages ? std::span<const Tensor>() : std::span<const Tensor>(&norm, 1),
                    config_.rms_norm_eps, residual[at(r)], v.messages ? &ffn_in[at(r)] : nullptr,
                    v.messages ? std::span<Tensor>() : std::span<Tensor>(&normed, 1), s(r));
            }
        });
    } else if (!v.messages && kExecute && T_ <= ops::kResidualMixMaxTokens &&
               H <= ops::kResidualMixMaxHidden) {
        // Unfused at decode and verification widths: the mix and its norm in one Op over the
        // FP32 residual, the bits of the fused residual_conv_mix.
        each([&](int r) {
            std::array<Tensor, ops::kDepthAttentionMaxStates> rank_states;
            for (std::size_t i = 0; i < b.states.size(); ++i) {
                rank_states[i] = (*b.states[i])[at(r)];
            }
            const Tensor norm = layer(r, v.layer).ffn_norm;
            Tensor normed     = u[at(r)];
            Tensor mix        = ffn_in[at(r)];
            ops::depth_attention_rmsnorm(
                std::span<const Tensor>(rank_states.data(), b.states.size()), residual[at(r)],
                layer(r, v.layer).ffn_queries, kDepthAttentionEpsilon,
                1.0F / std::sqrt(static_cast<float>(H)), std::span<const Tensor>(&norm, 1),
                config_.rms_norm_eps, std::span<Tensor>(&normed, 1), &mix, s(r));
        });
        normed_here = true;
    } else {
        depth(b.states, residual,
              {&layer(0, v.layer).ffn_queries,
               width_ == 2 ? &layer(1, v.layer).ffn_queries : nullptr},
              ffn_in);
    }
    tap(vi, "min", ffn_in);
    // At decode widths the MLP gate and its norm are one launch.
    const bool gate_norm = v.messages && T_ <= kBatchedGateTokens;
    if (gate_norm) {
        each([&](int r) {
            if constexpr (kExecute) {
                const auto& gate = layer(r, v.layer).core->gates[3];
                const Tensor in  = ffn_in[at(r)].view({H, T_});
                const ops::MessageGateNormed one{in,
                                                 (*b.messages)[at(r)].slice(2, 3, 1).view({H, T_}),
                                                 gate.down,
                                                 gate.forget_bf16,
                                                 gate.gain_bf16,
                                                 in,
                                                 layer(r, v.layer).ffn_norm,
                                                 u[at(r)]};
                ops::message_gates_rmsnorm(std::span<const ops::MessageGateNormed>(&one, 1),
                                           config_.rms_norm_eps, config_.rms_norm_eps, s(r));
            }
        });
    } else if (v.messages) {
        each([&](int r) {
            if constexpr (kExecute) {
                const auto& gate = layer(r, v.layer).core->gates[3];
                Tensor u         = ffn_in[at(r)].view({H, T_});
                ops::message_gate(u, (*b.messages)[at(r)].slice(2, 3, 1).view({H, T_}), gate.down,
                                  gate.forget, gate.gain, config_.rms_norm_eps, u, s(r));
            }
        });
    }
    if (!gate_norm && (!fused || v.messages) && !normed_here) {
        each([&](int r) {
            if constexpr (kExecute) {
                ops::rmsnorm_f32(ffn_in[at(r)].view({H, T_}), layer(r, v.layer).ffn_norm,
                                 config_.rms_norm_eps, u[at(r)], s(r));
            }
        });
    }
    tap(vi, "moe_in", u);

    auto z = alloc(DType::BF16, {H, T_});
    moe(index, u, z);
    tap(vi, "moe", z);

    auto zc = alloc(DType::BF16, {H, T_});
    each([&](int r) {
        if constexpr (kExecute) {
            const auto record = records(r, v.slot);
            if (fused) {
                ops::residual_conv_round(
                    conv(z[at(r)], layer(r, v.layer).post_ffn_conv, state(r).tails(v.slot).ffn,
                         record ? std::optional<Tensor>(record->ffn) : std::nullopt),
                    residual[at(r)], out[at(r)], s(r));
                return;
            }
            short_conv(r, z[at(r)], layer(r, v.layer).post_ffn_conv, state(r).tails(v.slot).ffn,
                       record ? &record->ffn : nullptr, zc[at(r)]);
            ops::residual_add_round_bf16(residual[at(r)], zc[at(r)], out[at(r)], s(r));
        }
    });
    tap(vi, "", out);
    b.messages = nullptr;
}

} // namespace ninfer::models::inkling2::execution
