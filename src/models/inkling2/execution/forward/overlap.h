#pragma once

// The half-chunk overlap of a wide tp 2 call: one half's exchanges move while the other half
// computes, with the bits of the call in one piece.
//
// The call's T columns are two halves, A = [0, split) and B = [split, T) (Half: column windows of
// the call's tensors). Outside the attention every value is per column but the post-attention and
// post-FFN convolution tails of each visit's slot, which B reads as A advanced them, as in two
// consecutive calls. Each visit runs as steps on its halves' lanes of the split exchange
// (allreduce_sum_begin / allreduce_sum_finish):
//   inputs A, inputs B   each closes its half's previous visit (close_half below), then the
//                        attention inputs of its columns;
//   attention            the attention of the whole call (projections, K/V convolutions, the
//                        attention itself: the Op of the call in one piece, whose tiles and key
//                        passes a half-width call would use less well);
//   output A, output B   each rank's output projection of the half's columns, then the BF16
//                        all-reduce of y begun on the half's lane;
//   FFN A, FFN B         each finishes its lane (y complete), then the post-attention
//                        convolution, the FP32 residual, the FFN input mix, gate and norm, the
//                        router and the bank(s) into the rank's FP32 partial, its BF16 rounding,
//                        and begins the BF16 all-reduce of the MoE output on the lane (moe()'s
//                        prefill-width exchange);
//   close h              finishes the lane, then the post-FFN convolution and the BF16 visit
//                        output.
// So y of B moves during FFN A, the MoE output of A during FFN B, that of B during the next
// visit's inputs A; y of A only during output B. Each step issues the Ops visit() issues at the
// half's width (a wide call has no fused steps; the overlap runs without records), and a tap
// receives each point of a half once it is complete, as the half's columns.
//
// Member definitions of Forward (forward.h), included only by forward.cpp.

#include "models/inkling2/execution/forward.h"

#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/message_gate.h"
#include "ninfer/ops/moe_bank.h"
#include "ninfer/ops/residual_f32.h"
#include "ninfer/ops/rmsnorm_f32.h"

#include <array>
#include <vector>

namespace ninfer::models::inkling2::execution {

// Halves are whole multiples of this many columns (the token tiles of the Ops that run per half
// then start on the same columns as in the call in one piece) and at least kOverlapMinHalf wide, so
// a call of fewer than 2 * kOverlapMinHalf columns runs in one piece.
inline constexpr std::int32_t kOverlapAlign   = 128;
inline constexpr std::int32_t kOverlapMinHalf = 384;

template <class Arena>
struct Forward<Arena>::Half {
    enum class Open { None, Attention, Moe };
    std::int32_t first = 0;
    std::int32_t count = 0;
    int lane           = 0;
    // Column windows of the call's buffers, [H, count]: the BF16 operand both of the half's
    // exchanges sum in place (y, then the MoE output z) and its staging; the FP32 residual r of
    // the visit whose MoE is open.
    RankTensors sum, staging, residual;
    Open open           = Open::None;
    std::uint32_t visit = 0; // the visit whose MoE is open, and its output's columns
    RankTensors out;
};

template <class Arena>
std::int32_t Forward<Arena>::overlap_split(std::int32_t T) const {
    const std::int32_t split = T / 2 / kOverlapAlign * kOverlapAlign;
    if (width_ != 2 || split < kOverlapMinHalf || T - split < kOverlapMinHalf) { return 0; }
    if constexpr (kExecute) {
        if (r_[0].record != nullptr ||
            !ops::allreduce_sum_split_available(*tp_.execution, *tp_.events)) {
            return 0;
        }
    }
    return split;
}

template <class Arena>
auto Forward<Arena>::window(const RankTensors& t, const Half& half) const -> RankTensors {
    RankTensors out;
    for (int r = 0; r < width_; ++r) { out[at(r)] = t[at(r)].slice(1, half.first, half.count); }
    return out;
}

template <class Arena>
auto Forward<Arena>::halves(std::int32_t split) -> std::array<Half, 2> {
    const std::int32_t H = hidden();
    const auto sum       = alloc(DType::BF16, {H, T_});
    const auto staging   = alloc(DType::BF16, {H, T_});
    const auto residual  = alloc(DType::FP32, {H, T_});
    std::array<Half, 2> out;
    for (int h = 0; h < 2; ++h) {
        Half& half    = out[at(h)];
        half.first    = h == 0 ? 0 : split;
        half.count    = h == 0 ? split : T_ - split;
        half.lane     = h;
        half.sum      = window(sum, half);
        half.staging  = window(staging, half);
        half.residual = window(residual, half);
    }
    return out;
}

template <class Arena>
template <class Body>
void Forward<Arena>::on_half(Half& half, Body&& body) {
    const std::int32_t T     = T_;
    const std::int32_t first = first_position_;
    const auto restore       = [&] {
        T_              = T;
        first_position_ = first;
    };
    T_              = half.count;
    first_position_ = first + half.first;
    try {
        body();
    } catch (...) {
        restore();
        throw;
    }
    restore();
}

template <class Arena>
void Forward<Arena>::visit_halves(std::uint32_t index, Block& b, const RankTensors& x,
                                  RankTensors& out, RankTensors* captured,
                                  std::array<Half, 2>& halves) {
    const Visit& v  = config_.visits[index];
    const auto held = scopes();
    // The halves' messages live from their inputs step to their FFN step.
    std::array<RankTensors, 2> messages;
    if (v.messages) {
        for (int h = 0; h < 2; ++h) {
            messages[at(h)] = alloc(DType::FP32, {hidden(), halves[at(h)].count, 4});
        }
    }
    // The call's block (states, message bank, previous pass) at one half's columns.
    const auto columns = [&](int h, Block& view, std::vector<RankTensors>& windows) {
        const Half& half = halves[at(h)];
        windows.reserve(b.states.size() + b.bank.size() + 1);
        for (const RankTensors* state : b.states) {
            windows.push_back(window(*state, half));
            view.states.push_back(&windows.back());
        }
        for (const RankTensors* state : b.bank) {
            windows.push_back(window(*state, half));
            view.bank.push_back(&windows.back());
        }
        if (b.previous != nullptr) {
            windows.push_back(window(*b.previous, half));
            view.previous = &windows.back();
        }
        if (v.messages) {
            view.messages_storage = messages[at(h)];
            view.messages         = &view.messages_storage;
        }
    };
    {
        const auto attention_scope = scopes();
        const std::int32_t H       = hidden();
        std::array<RankTensors, 3> xin;
        for (auto& t : xin) { t = alloc(DType::BF16, {H, T_}); }
        for (int h = 0; h < 2; ++h) {
            Block view;
            std::vector<RankTensors> windows;
            columns(h, view, windows);
            Half& half = halves[at(h)];
            on_half(half, [&] { half_inputs(index, view, x, captured, xin, half); });
        }
        auto o = alloc(DType::BF16, {static_cast<std::int32_t>(rc().query_width()), T_});
        each([&](int r) { attention_rank(index, r, xin, o[at(r)]); });
        // linear_row_parallel by halves, each all-reduce begun on its half's lane.
        for (Half& half : halves) {
            on_half(half, [&] {
                const auto oh = window(o, half);
                each([&](int r) {
                    linear(r, oh[at(r)], layer(r, v.layer).attention.output, half.sum[at(r)]);
                });
                if constexpr (kExecute) {
                    ops::allreduce_sum_begin(half.sum, half.staging, *tp_.execution, *tp_.events,
                                             half.lane);
                }
                half.open = Half::Open::Attention;
            });
        }
    }
    if (v.block_boundary) { b.states.push_back(&x); }
    for (int h = 0; h < 2; ++h) {
        Block view;
        std::vector<RankTensors> windows;
        columns(h, view, windows);
        Half& half = halves[at(h)];
        on_half(half, [&] { half_ffn(index, view, x, out, half); });
    }
}

template <class Arena>
void Forward<Arena>::half_inputs(std::uint32_t index, Block& b, const RankTensors& x,
                                 RankTensors* captured, std::array<RankTensors, 3>& xin,
                                 Half& half) {
    close_half(half);
    const auto held = scopes();
    const auto in   = window(x, half);
    if (captured != nullptr) {
        const auto sink = window(*captured, half);
        each([&](int r) {
            if constexpr (kExecute) {
                CUDA_CHECK(::ninfer::memcpy_device_async(sink[at(r)].data, in[at(r)].data,
                                                         in[at(r)].bytes(), s(r)));
            }
        });
    }
    std::array<RankTensors, 3> columns{window(xin[0], half), window(xin[1], half),
                                       window(xin[2], half)};
    attention_inputs(index, b, in, columns);
    // One input for all three: the whole call's inputs share it too.
    if (columns[1][0].data == columns[0][0].data) {
        xin[1] = xin[0];
        xin[2] = xin[0];
    }
}

template <class Arena>
void Forward<Arena>::half_ffn(std::uint32_t index, Block& b, const RankTensors& x, RankTensors& out,
                              Half& half) {
    const Visit& v       = config_.visits[index];
    const std::int32_t H = hidden();
    const auto vi        = static_cast<std::int32_t>(index);
    close_half(half);
    tap(vi, "attn", half.sum);
    const auto held = scopes();
    const auto in   = window(x, half);
    auto yc         = alloc(DType::BF16, {H, T_});
    each([&](int r) {
        if constexpr (kExecute) {
            short_conv(r, half.sum[at(r)], layer(r, v.layer).post_attention_conv,
                       state(r).tails(v.slot).attention, nullptr, yc[at(r)]);
            ops::residual_add_f32(v.block_boundary ? nullptr : &in[at(r)], yc[at(r)],
                                  half.residual[at(r)], s(r));
        }
    });
    tap(vi, "attn_conv", yc);
    auto ffn_in = alloc(DType::FP32, {H, T_, 1});
    depth(b.states, half.residual,
          {&layer(0, v.layer).ffn_queries, width_ == 2 ? &layer(1, v.layer).ffn_queries : nullptr},
          ffn_in);
    tap(vi, "min", ffn_in);
    auto u = alloc(DType::BF16, {H, T_});
    each([&](int r) {
        if constexpr (kExecute) {
            Tensor mix = ffn_in[at(r)].view({H, T_});
            if (v.messages) {
                const auto& gate = layer(r, v.layer).core->gates[3];
                ops::message_gate(mix, (*b.messages)[at(r)].slice(2, 3, 1).view({H, T_}), gate.down,
                                  gate.forget, gate.gain, config_.rms_norm_eps, mix, s(r));
            }
            ops::rmsnorm_f32(mix, layer(r, v.layer).ffn_norm, config_.rms_norm_eps, u[at(r)], s(r));
        }
    });
    tap(vi, "moe_in", u);
    auto partial = alloc(DType::FP32, {H, T_});
    moe_partial(index, u, partial);
    // moe()'s prefill-width exchange, begun on the lane; y is spent (yc holds its convolution).
    if constexpr (kExecute) {
        each([&](int r) { ops::moe_partial_to_bf16(partial[at(r)], half.sum[at(r)], s(r)); });
        ops::allreduce_sum_begin(half.sum, half.staging, *tp_.execution, *tp_.events, half.lane);
    }
    half.open  = Half::Open::Moe;
    half.visit = index;
    half.out   = window(out, half);
}

template <class Arena>
void Forward<Arena>::close_half(Half& half) {
    const typename Half::Open open = half.open;
    if (open == Half::Open::None) { return; }
    half.open = Half::Open::None;
    if constexpr (kExecute) { ops::allreduce_sum_finish(*tp_.execution, *tp_.events, half.lane); }
    if (open != Half::Open::Moe) { return; }
    const Visit& v       = config_.visits[half.visit];
    const std::int32_t H = hidden();
    const auto held      = scopes();
    auto zc              = alloc(DType::BF16, {H, half.count});
    const auto vi        = static_cast<std::int32_t>(half.visit);
    tap(vi, "moe", half.sum);
    each([&](int r) {
        if constexpr (kExecute) {
            short_conv(r, half.sum[at(r)], layer(r, v.layer).post_ffn_conv,
                       state(r).tails(v.slot).ffn, nullptr, zc[at(r)]);
            ops::residual_add_round_bf16(half.residual[at(r)], zc[at(r)], half.out[at(r)], s(r));
        }
    });
    tap(vi, "", half.out);
}

template <class Arena>
void Forward<Arena>::abandon_halves(std::array<Half, 2>& halves) noexcept {
    for (Half& half : halves) {
        if (half.open == Half::Open::None) { continue; }
        half.open = Half::Open::None;
        if constexpr (kExecute) {
            try {
                ops::allreduce_sum_finish(*tp_.execution, *tp_.events, half.lane);
            } catch (...) {}
        }
    }
}

} // namespace ninfer::models::inkling2::execution
