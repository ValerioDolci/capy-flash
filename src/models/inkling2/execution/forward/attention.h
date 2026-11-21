#pragma once

// The attention of a visit over each rank's heads (12/4 of 24/8 at tp 2):
//
//   q, gate, r, sink = column-parallel projections of xq; k, v of xk, xv: one Linear per distinct
//           input over the stacked rows (Parameters: queries, all, keys, values), each block then
//           a view (one column) or a row copy into its own buffer
//   k, v  = short convolution with the slot's tails: static taps (local) or static + per-token
//           dynamic taps (global)
//   o     = relative_sink_attention: clipped q/k, RMS-normed v, relative bias, sink, the window
//           (local, the slot's ring) or every position (global, the slot's pages, with tau)
//   o     = o * sigmoid(gate)
//   y     = Wo o: one Linear, or at tp 2 row-parallel with one BF16 all-reduce
//
// Member definitions of Forward (forward.h), included only by forward.cpp.

#include "models/inkling2/execution/forward.h"

#include "ninfer/ops/copy_rows.h"
#include "ninfer/ops/relative_sink_attention.h"
#include "ninfer/ops/short_conv.h"
#include "ninfer/ops/short_conv_record.h"
#include "ninfer/ops/tp2/linear.h"

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <variant>

namespace ninfer::models::inkling2::execution {

// The L2 clips' epsilon (added to the norm): a constant of the reference, not a config field.
inline constexpr float kClipEpsilon = 1e-6F;

template <class Arena>
void Forward<Arena>::attention(std::uint32_t index, const std::array<RankTensors, 3>& xin,
                               RankTensors& y) {
    const Visit& v  = config_.visits[index];
    const auto held = scopes();
    const auto qw   = static_cast<std::int32_t>(rc().query_width());
    auto o          = alloc(DType::BF16, {qw, T_});
    each([&](int r) { attention_rank(index, r, xin, o[at(r)]); });
    if (width_ == 1) {
        linear(0, o[0], layer(0, v.layer).attention.output, y[0]);
        return;
    }
    auto staging = alloc(DType::BF16, {hidden(), T_});
    if constexpr (kExecute) {
        const auto& w0 = layer(0, v.layer).attention.output;
        const auto& w1 = layer(1, v.layer).attention.output;
        ops::linear_row_parallel(o, {w0.weight, w1.weight}, y, staging, w0.policy,
                                 {&work(0), &work(1)}, *tp_.execution, *tp_.events);
    } else {
        for (int r = 0; r < width_; ++r) {
            linear(r, o[at(r)], layer(r, v.layer).attention.output, y[at(r)]);
        }
    }
}

template <class Arena>
void Forward<Arena>::project(int r, const Tensor& x, const LinearParameters& stacked,
                             std::initializer_list<std::pair<Tensor*, std::int32_t>> blocks) {
    Tensor out = work(r).alloc(DType::BF16, {stacked.weight.n, T_});
    linear(r, x, stacked, out);
    std::int32_t row = 0;
    for (const auto& [block, rows] : blocks) {
        if constexpr (kExecute) {
            const auto* first = static_cast<const std::byte*>(out.data) + std::size_t(row) * 2;
            if (T_ == 1) {
                *block = Tensor(const_cast<std::byte*>(first), DType::BF16, {rows, 1});
            } else {
                ops::copy_rows(block->data, first, std::size_t(stacked.weight.n) * 2,
                               std::size_t(rows) * 2, std::size_t(T_), s(r));
            }
        }
        row += rows;
    }
    if (row != stacked.weight.n) {
        throw std::logic_error("Inkling 2 forward: stacked projection rows differ from its blocks");
    }
}

template <class Arena>
void Forward<Arena>::attention_rank(std::uint32_t index, int r,
                                    const std::array<RankTensors, 3>& xin, Tensor& o) {
    const Visit& v       = config_.visits[index];
    const auto& p        = layer(r, v.layer).attention;
    const TextConfig& c  = rc();
    const bool global    = v.kind == LayerKind::Global;
    const auto D         = static_cast<std::int32_t>(c.head_dim);
    const auto Hq        = static_cast<std::int32_t>(c.num_attention_heads);
    const auto Hkv       = static_cast<std::int32_t>(c.num_key_value_heads);
    const auto dr        = static_cast<std::int32_t>(c.relative_dim);
    const auto C         = Hkv * D;
    const std::int32_t T = T_;
    auto scope           = work(r).scope();
    Tensor q             = work(r).alloc(DType::BF16, {Hq * D, T});
    Tensor gate          = work(r).alloc(DType::BF16, {Hq * D, T});
    Tensor rel           = work(r).alloc(DType::BF16, {Hq * dr, T});
    Tensor sink          = work(r).alloc(DType::BF16, {Hq, T});
    Tensor k             = work(r).alloc(DType::BF16, {C, T});
    Tensor vv            = work(r).alloc(DType::BF16, {C, T});
    Tensor kc            = work(r).alloc(DType::BF16, {C, T});
    Tensor vc            = work(r).alloc(DType::BF16, {C, T});
    const auto taps      = static_cast<std::int32_t>(config_.conv_kernel) * Hkv;
    Tensor dk, dv;
    if (global) {
        dk = work(r).alloc(DType::BF16, {taps, T});
        dv = work(r).alloc(DType::BF16, {taps, T});
    }
    const Tensor& xq = xin[0][at(r)];
    const Tensor& xk = xin[1][at(r)];
    const Tensor& xv = xin[2][at(r)];
    // One Linear per input over the stacked rows, then each block in its own buffer.
    const bool shared = !global && !v.messages;
    if (shared) {
        project(r, xq, *p.all,
                {{&q, Hq * D}, {&gate, Hq * D}, {&rel, Hq * dr}, {&sink, Hq}, {&k, C}, {&vv, C}});
    } else {
        project(r, xq, p.queries, {{&q, Hq * D}, {&gate, Hq * D}, {&rel, Hq * dr}, {&sink, Hq}});
        if (global) {
            project(r, xk, *p.keys, {{&k, C}, {&dk, taps}});
            project(r, xv, *p.values, {{&vv, C}, {&dv, taps}});
        } else {
            linear(r, xk, p.key, k);
            linear(r, xv, p.value, vv);
        }
    }

    const auto K = static_cast<float>(config_.conv_kernel);
    if (const auto* conv = std::get_if<StaticKvConvParameters>(&p.kv_conv)) {
        if constexpr (kExecute) {
            const auto tails  = state(r).tails(v.slot);
            const float scale = 1.0F / std::sqrt(K + 1.0F);
            Tensor tk = tails.key, tv = tails.value;
            if (auto record = records(r, v.slot)) {
                ops::short_conv_static_record(k, conv->key, scale, tk, record->key, kc, s(r));
                ops::short_conv_static_record(vv, conv->value, scale, tv, record->value, vc, s(r));
            } else {
                ops::short_conv_static(k, conv->key, scale, tk, tk, kc, s(r));
                ops::short_conv_static(vv, conv->value, scale, tv, tv, vc, s(r));
            }
        }
    } else {
        const auto& dyn = std::get<DynamicKvConvParameters>(p.kv_conv);
        if constexpr (kExecute) {
            const auto tails  = state(r).tails(v.slot);
            const float scale = 1.0F / std::sqrt(2.0F * K + 1.0F);
            Tensor tk = tails.key, tv = tails.value;
            if (auto record = records(r, v.slot)) {
                ops::short_conv_dynamic_record(k, dyn.key_static, dk, scale, tk, record->key, kc,
                                               s(r));
                ops::short_conv_dynamic_record(vv, dyn.value_static, dv, scale, tv, record->value,
                                               vc, s(r));
            } else {
                ops::short_conv_dynamic(k, dyn.key_static, dk, scale, tk, tk, kc, s(r));
                ops::short_conv_dynamic(vv, dyn.value_static, dv, scale, tv, tv, vc, s(r));
            }
        } else {
            (void)dyn;
        }
    }

    const ops::AttentionHeadGeometry geometry{D, Hq, Hkv};
    ops::RelativeSinkAttentionParams params;
    params.relative_dim    = dr;
    params.relative_extent = static_cast<std::int32_t>(config_.relative_bias_extent(v.layer));
    params.qk_clip         = config_.qk_clip_threshold;
    params.relative_clip   = config_.relative_clip_threshold;
    params.clip_epsilon    = kClipEpsilon;
    params.value_epsilon   = config_.rms_norm_eps;
    params.scale           = 1.0F / std::sqrt(static_cast<float>(D));
    params.log_scale_alpha = global ? config_.log_scaling_alpha : 0.0F;
    params.log_scale_floor = static_cast<float>(config_.log_scaling_floor);
    if constexpr (kExecute) {
        const Tensor q4      = q.view({D, Hq, T, 1});
        const Tensor k4      = kc.view({D, Hkv, T, 1});
        const Tensor v4      = vc.view({D, Hkv, T, 1});
        const Tensor r4      = rel.view({dr, Hq, T, 1});
        const Tensor s3      = sink.view({Hq, T, 1});
        const Tensor pos     = r_[at(r)].positions->view({T, 1});
        Tensor o4            = o.view({D, Hq, T, 1});
        const Tensor g4      = gate.view({D, Hq, T, 1});
        const VisitState& st = state(r);
        if (global) {
            ops::relative_sink_attention_paged(q4, k4, v4, r4, s3, pos, st.row(), p.query_scale,
                                               p.relative_scale, p.relative_profile, geometry,
                                               params, st.pages(v.slot), &g4, work(r), o4, s(r));
        } else {
            ops::relative_sink_attention_windowed(
                q4, k4, v4, r4, s3, pos, st.row(), p.query_scale, p.relative_scale,
                p.relative_profile, geometry, params, static_cast<std::int32_t>(c.sliding_window),
                st.ring(v.slot), &g4, work(r), o4, s(r));
        }
    } else {
        reserve(
            ops::relative_sink_attention_workspace_capacity_bytes(geometry, dr, 1, 1, max_tokens_),
            r);
    }
}

} // namespace ninfer::models::inkling2::execution
