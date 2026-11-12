#pragma once

// The sequence-parallel tp 2 prefill of the plain residual (NINFER_TP_PREFILL_SP=1, opt-in). Under
// the half-chunk overlap each half's residual columns are split between the
// ranks: rank 0 owns the half's first (n + 1) / 2 columns, rank 1 its last (n + 1) / 2 (an odd half
// shares its middle column; both ranks write identical bytes there). A block runs in two stages
// per half:
//
//   stage 1: finish the half's reduce-scatter and add both partials into the owned columns (rank
//            0's first); norm the owned columns; run
//            the block's column-parallel input projection over them under an NVFP4 capture (and,
//            in a GDN layer, the gating of both ranks' heads, the peer's with this rank's copy of
//            its weights); begin the all-gather of the captured plane (and the peer heads' gating)
//            on the half's lane.
//   stage 2: finish the all-gather; the input projection over the peer's columns from the
//            received plane; the mixer core or the FFN activation over the whole half (bound as
//            half_block binds it); the row-parallel output projection as a plain BF16 partial per
//            rank, the peer's columns first, whose reduce-scatter (rank r sends the peer's columns
//            of its partial) begins before the owned columns are projected.
//
// Blocks issue A1 B1 A2 B2, so half A's all-gather moves under half B's stage 1 and its
// reduce-scatter under half B's stage 2: 0.64x of the overlap's all-reduce bytes (0.5 + 0.14). A
// layer's whole residual is all-gathered in BF16 (lane 2 + h) only for the layers a feature tap
// keeps and at the chunk's end.
//
// Buffers that outlive a stage (the norm output, the planes, the block intermediates and the
// partial) are one region per half and rank, allocated once per call; a
// move only ever names them, x or staging. Stage temporaries live in stage scopes.
//
// Member definitions of Qwen4Forward (qwen4_forward.h), included only by qwen4_forward.cpp.

#include "models/qwen3_5/execution/qwen4_forward.h"
#include "models/qwen3_5/execution/tp2/attention_split.h"
#include "models/qwen3_5/execution/tp2/gdn_split.h"
#include "models/qwen3_5/execution/tp2/linear_split.h"
#include "models/qwen3_5/execution/tp2/prefill_slices.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/tp2/gdn_gating_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/nvfp4_activation.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/tp2/linear.h"

#include <algorithm>
#include <cstdio>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

namespace ninfer::models::qwen3_5::execution {

namespace sp_detail {

constexpr std::size_t kAlign = 256;

[[nodiscard]] constexpr std::size_t align_up(std::size_t bytes) {
    return (bytes + kAlign - 1) / kAlign * kAlign;
}

[[nodiscard]] inline std::byte* column_pointer(const Tensor& t, std::int32_t column) {
    return static_cast<std::byte*>(t.data) + static_cast<std::size_t>(column) * t.nb[1];
}

// The widths one half's region holds, in bytes, in carve order.
struct HalfLayout {
    std::size_t h = 0, own_codes = 0, own_scales = 0, peer_codes = 0, peer_scales = 0, block = 0,
                partial = 0;

    [[nodiscard]] std::size_t total() const {
        return h + own_codes + own_scales + peer_codes + peer_scales + block + partial;
    }
};

} // namespace sp_detail

template <class Arena>
bool Qwen4Forward<Arena>::sp_possible() const {
    if (width_ != 2 || has_hc() || moe_ffn() || !qwen4_prefill_sp_env() ||
        config_.layer_types.empty()) {
        return false;
    }
    const auto nvfp4 = [](const ops::ProjectionWeights& w) {
        const auto* one = std::get_if<LinearParameters>(&w);
        return one != nullptr && one->weight.qtype == QType::NVFP4;
    };
    for (int r = 0; r < width_; ++r) {
        for (std::size_t layer = 0; layer < config_.layer_types.size(); ++layer) {
            const Qwen4BlockParameters& b = qwen4(r).layers[layer];
            const auto* dense             = std::get_if<DenseParameters>(&b.ffn);
            if (dense == nullptr || !b.fused.swiglu ||
                dense->gate_up.weight.qtype != QType::NVFP4) {
                return false;
            }
            if (const auto* a = std::get_if<AttentionParameters>(&b.mixer)) {
                if (!nvfp4(a->projection)) { return false; }
            } else if (const auto* g = std::get_if<GdnParameters>(&b.mixer)) {
                if (!b.fused.gdn_gating || !b.fused.gdn_input || !nvfp4(g->projection)) {
                    return false;
                }
            } else {
                return false; // QSA
            }
        }
    }
    return true;
}

template <class Arena>
bool Qwen4Forward<Arena>::sp_eligible(std::int32_t split) const {
    if constexpr (kExecute) {
        if (split <= 0 || !sp_possible() || tp_.events == nullptr ||
            tp_.events->host_exchange() == nullptr || tp_.events->peer_lanes() != nullptr) {
            return false;
        }
        const std::size_t layers = config_.layer_types.size();
        for (int r = 0; r < width_; ++r) {
            const Qwen4Runtime* runtime = state(r).runtime;
            if (runtime == nullptr || runtime->sp_peer_control.size() != layers) { return false; }
            for (std::size_t layer = 0; layer < layers; ++layer) {
                if (config_.layer_types[layer] != MixerKind::FullAttention &&
                    !runtime->sp_peer_control[layer].present) {
                    return false;
                }
            }
        }
        return true;
    } else {
        (void)split;
        return false;
    }
}

template <class Arena>
std::size_t Qwen4Forward<Arena>::sp_half_bytes(std::int32_t count) const {
    using sp_detail::align_up;
    const std::int32_t H   = hidden();
    const std::int32_t own = (count + 1) / 2;
    const auto& a          = *rc().attention;
    const auto& g          = *rc().gdn;
    std::size_t ffn_rows   = 0;
    for (std::size_t layer = 0; layer < config_.layer_types.size(); ++layer) {
        const auto& dense = std::get<DenseParameters>(qwen4(0).layers[layer].ffn);
        ffn_rows = std::max(ffn_rows, static_cast<std::size_t>(dense.gate_up.weight.n / 2));
    }
    const auto W                = static_cast<std::size_t>(count);
    const auto n                = static_cast<std::size_t>(own);
    const auto bf16             = std::size_t{2};
    const auto fp32             = std::size_t{4};
    const std::size_t attention = align_up(dimension(a.query_width()) * W * bf16) * 2 +
                                  align_up(dimension(a.key_width()) * W * bf16) * 2;
    const auto heads      = static_cast<std::size_t>(dimension(g.linear_num_value_heads));
    const std::size_t gdn = align_up(dimension(g.conv_channels()) * W * bf16) +
                            align_up(dimension(g.value_width()) * W * bf16) +
                            align_up(heads * W * fp32) * 2 + align_up(heads * n * fp32) * 2;
    const std::size_t ffn = align_up(ffn_rows * W * bf16);
    sp_detail::HalfLayout l;
    l.h           = align_up(static_cast<std::size_t>(H) * n * bf16);
    l.own_codes   = align_up(ops::nvfp4_activation_code_capacity(own, H));
    l.own_scales  = align_up(ops::nvfp4_activation_scale_capacity(own, H));
    l.peer_codes  = l.own_codes;
    l.peer_scales = l.own_scales;
    l.block       = std::max({attention, gdn, ffn});
    l.partial     = align_up(static_cast<std::size_t>(H) * W * bf16);
    return l.total();
}

template <class Arena>
void Qwen4Forward<Arena>::sp_plan() {
    if constexpr (!kExecute) {
        // The widest half any chunk of [768, last_] columns can have: the split rounds down to
        // 128-column steps, so the second half is widest just below a multiple of 256.
        std::int32_t widest = 0;
        for (std::int32_t T = 1; T <= last_; ++T) {
            const std::int32_t split = prefill_overlap_split(T);
            if (split > 0) { widest = std::max({widest, split, T - split}); }
        }
        if (widest == 0) { return; }
        auto sp_scope = scopes();
        for (int r = 0; r < width_; ++r) {
            (void)work(r).alloc_bytes(sp_half_bytes(widest), sp_detail::kAlign);
            (void)work(r).alloc_bytes(sp_half_bytes(widest), sp_detail::kAlign);
        }
        const auto& a       = *rc().attention;
        const auto& g       = *rc().gdn;
        const auto head_dim = dimension(a.head_dim);
        const auto heads    = dimension(a.num_attention_heads);
        const auto kv_heads = dimension(a.num_key_value_heads);
        const auto gheads   = dimension(g.linear_num_value_heads);
        const auto kheads   = dimension(g.linear_num_key_heads);
        const auto dv       = dimension(g.linear_value_head_dim);
        const ops::CausalAttentionExecutionEnvelope envelope{1, std::max(c(0).max_visible, 1U)};
        for (std::size_t layer = 0; layer < config_.layer_types.size(); ++layer) {
            const auto p = [&](int r) -> const Qwen4BlockParameters& {
                return qwen4(r).layers[layer];
            };
            const auto& dense0 = std::get<DenseParameters>(p(0).ffn);
            {
                auto stage = scopes();
                if (const auto* attn = std::get_if<AttentionParameters>(&p(0).mixer)) {
                    (void)alloc(DType::BF16, {head_dim, heads, widest});
                    (void)alloc(DType::BF16, {head_dim, kv_heads, widest});
                    (void)alloc(DType::BF16, {head_dim, heads, widest});
                    for (int r = 0; r < width_; ++r) {
                        const auto& ar = std::get<AttentionParameters>(p(r).mixer);
                        reserve(attention_projection_split_workspace_bytes(ar, first_, last_), r);
                        reserve(ops::linear_workspace_capacity_bytes(
                                    ar.output.weight.qtype, ar.output.weight.n, ar.output.weight.k,
                                    ar.output.policy, first_, last_),
                                r);
                    }
                    reserve(ops::causal_softmax_attention_workspace_capacity_bytes(
                        {head_dim, heads, kv_heads}, c(0).kv_storage, envelope, 1, first_, last_));
                    (void)attn;
                } else {
                    (void)alloc(DType::BF16, {dimension(g.key_width()), widest});
                    (void)alloc(DType::BF16, {dimension(g.key_width()), widest});
                    (void)alloc(DType::BF16, {dimension(g.value_width()), widest});
                    (void)alloc(DType::BF16, {dv, gheads, widest});
                    (void)alloc(DType::BF16, {dv, gheads, widest});
                    for (int r = 0; r < width_; ++r) {
                        const auto& gr = std::get<GdnParameters>(p(r).mixer);
                        reserve(gdn_control_split_workspace_bytes(g, hidden(), first_, last_), r);
                        reserve(gdn_projection_split_workspace_bytes(gr, first_, last_), r);
                        reserve(ops::linear_workspace_capacity_bytes(
                                    gr.output.weight.qtype, gr.output.weight.n, gr.output.weight.k,
                                    gr.output.policy, first_, last_),
                                r);
                    }
                    reserve(ops::gated_delta_net_workspace_capacity_bytes(kheads, gheads, first_,
                                                                          last_));
                }
            }
            {
                auto stage = scopes();
                for (int r = 0; r < width_; ++r) {
                    const auto& d = std::get<DenseParameters>(p(r).ffn);
                    reserve(ops::linear_swiglu_workspace_capacity_bytes(
                                d.gate_up.weight.qtype, d.gate_up.weight.n, d.gate_up.weight.k,
                                d.gate_up.policy, first_, last_),
                            r);
                    reserve(ops::linear_workspace_capacity_bytes(d.down.weight.qtype,
                                                                 d.down.weight.n, d.down.weight.k,
                                                                 d.down.policy, first_, last_),
                            r);
                }
                (void)dense0;
            }
        }
    }
}

template <class Arena>
template <class Body>
void Qwen4Forward<Arena>::bind_half(const PlainHalf& half, Body&& body) {
    const std::array<Qwen4Columns, 2> chunk{r_[0].columns, r_[1].columns};
    const auto restore = [&] {
        for (int r = 0; r < width_; ++r) { r_[at(r)].columns = chunk[at(r)]; }
    };
    for (int r = 0; r < width_; ++r) {
        Qwen4Columns& col   = r_[at(r)].columns;
        col.width           = half.x[at(r)].ne[1];
        col.cache_positions = &half.positions[at(r)];
        col.rope_positions  = &half.rope[at(r)];
        col.max_visible     = half.envelope.max_visible_keys;
        col.envelope        = &half.envelope;
        col.source_slot =
            half.first == 0 ? chunk[at(r)].source_slot : chunk[at(r)].destination_slot;
        col.first_position = chunk[at(r)].first_position + half.first;
    }
    try {
        body();
    } catch (...) {
        restore();
        throw;
    }
    restore();
}

template <class Arena>
bool Qwen4Forward<Arena>::sp_tapped(std::size_t layer) const {
    return feature_tap_ != nullptr && feature_tap_->captures_layer(static_cast<int>(layer));
}

template <class Arena>
auto Qwen4Forward<Arena>::sp_begin(std::array<PlainHalf, 2>& halves) -> SpCall {
    using sp_detail::align_up;
    SpCall call;
    const std::int32_t H  = hidden();
    const auto& a         = *rc().attention;
    const auto& g         = *rc().gdn;
    const auto heads      = dimension(g.linear_num_value_heads);
    std::int32_t ffn_rows = 0;
    for (std::size_t layer = 0; layer < config_.layer_types.size(); ++layer) {
        ffn_rows = std::max(
            ffn_rows, std::get<DenseParameters>(qwen4(0).layers[layer].ffn).gate_up.weight.n / 2);
    }
    for (int h = 0; h < 2; ++h) {
        SpHalf& sh              = call.halves[at(h)];
        sh.half                 = &halves[at(h)];
        sh.count                = halves[at(h)].x[0].ne[1];
        sh.own                  = (sh.count + 1) / 2;
        sh.own_first            = {0, sh.count - sh.own};
        const auto W            = sh.count;
        const auto n            = sh.own;
        const std::size_t bytes = sp_half_bytes(W);
        for (int r = 0; r < width_; ++r) {
            auto* p = static_cast<std::byte*>(work(r).alloc_bytes(bytes, sp_detail::kAlign).data);
            const auto take = [&](std::size_t size) {
                std::byte* out = p;
                p += align_up(size);
                return out;
            };
            sh.h[at(r)] = Tensor(take(static_cast<std::size_t>(H) * n * 2), DType::BF16, {H, n});
            for (auto* plane : {&sh.own_plane[at(r)], &sh.peer_plane[at(r)]}) {
                plane->code_capacity  = ops::nvfp4_activation_code_capacity(n, H);
                plane->scale_capacity = ops::nvfp4_activation_scale_capacity(n, H);
                plane->codes          = take(plane->code_capacity);
                plane->scales         = take(plane->scale_capacity);
            }
            // The block intermediates share one span (a block is attention, GDN or FFN).
            std::byte* block = p;
            const auto carve = [&](std::byte*& at_, DType dtype, std::int32_t rows,
                                   std::int32_t columns) {
                Tensor t(at_, dtype, {rows, columns});
                at_ += align_up(t.bytes());
                return t;
            };
            std::byte* cursor    = block;
            sh.q[at(r)]          = carve(cursor, DType::BF16, dimension(a.query_width()), W);
            sh.gate[at(r)]       = carve(cursor, DType::BF16, dimension(a.query_width()), W);
            sh.k[at(r)]          = carve(cursor, DType::BF16, dimension(a.key_width()), W);
            sh.v[at(r)]          = carve(cursor, DType::BF16, dimension(a.key_width()), W);
            std::byte* end       = cursor;
            cursor               = block;
            sh.qkv[at(r)]        = carve(cursor, DType::BF16, dimension(g.conv_channels()), W);
            sh.z[at(r)]          = carve(cursor, DType::BF16, dimension(g.value_width()), W);
            sh.g[at(r)]          = carve(cursor, DType::FP32, heads, W);
            sh.beta[at(r)]       = carve(cursor, DType::FP32, heads, W);
            sh.g_peer[at(r)]     = carve(cursor, DType::FP32, heads, n);
            sh.beta_peer[at(r)]  = carve(cursor, DType::FP32, heads, n);
            end                  = std::max(end, cursor);
            cursor               = block;
            sh.activation[at(r)] = carve(cursor, DType::BF16, ffn_rows, W);
            end                  = std::max(end, cursor);
            sh.block[at(r)]      = block;
            p                    = end;
            sh.partial[at(r)] =
                Tensor(take(static_cast<std::size_t>(H) * W * 2), DType::BF16, {H, W});
        }
    }
    return call;
}

namespace sp_detail {

// Per-rank column windows of both ranks' operands (they may differ per rank).
inline std::array<Tensor, 2> windows(const std::array<Tensor, 2>& t,
                                     const std::array<std::int32_t, 2>& first, std::int32_t count) {
    return {t[0].slice(1, first[0], count), t[1].slice(1, first[1], count)};
}

struct Handoff {
    std::optional<ops::ScopedNvfp4Activation> rank[2];

    void open(ops::ScopedNvfp4Activation::Mode mode, const std::array<Tensor, 2>& input,
              std::array<ops::Nvfp4ActivationPlane, 2>& planes) {
        for (std::size_t r = 0; r < 2; ++r) {
            rank[r].emplace(mode, input[r].data, input[r].ne[1], planes[r]);
        }
    }

    void require_used(const char* what) const {
        for (const auto& scope : rank) {
            if (scope && !scope->used()) {
                throw std::logic_error(std::string("SP prefill: the ") + what +
                                       " projection did not take the NVFP4 hand-off");
            }
        }
    }
};

} // namespace sp_detail

template <class Arena>
void Qwen4Forward<Arena>::sp_close_reduce(SpHalf& sh, int h) {
    if constexpr (kExecute) {
        if (!sh.reduce_open) { return; }
        ops::HostExchange* exchange = tp_.events->host_exchange();
        exchange->move_finish(h, *tp_.execution);
        sh.reduce_open  = false;
        PlainHalf& half = *sh.half;
        each([&](int r) {
            const std::size_t i      = at(r);
            const std::int32_t first = sh.own_first[i], n = sh.own;
            Tensor residual = half.x[i].slice(1, first, n);
            Tensor local    = sh.partial[i].slice(1, first, n);
            Tensor received = half.staging[i].slice(1, first, n);
            ops::residual_add(r == 0 ? local : received, residual, s(r));
            ops::residual_add(r == 0 ? received : local, residual, s(r));
        });
    } else {
        (void)sh, (void)h;
    }
}

template <class Arena>
void Qwen4Forward<Arena>::sp_begin_x_gather(SpHalf& sh, int h, int tapped_layer) {
    if constexpr (kExecute) {
        ops::HostExchange::MoveSegments segments{};
        const std::size_t column_bytes = static_cast<std::size_t>(hidden()) * 2;
        PlainHalf& half                = *sh.half;
        for (std::size_t r = 0; r < 2; ++r) {
            const std::int32_t first = sh.own_first[r];
            segments[r][0]           = {sp_detail::column_pointer(half.x[r], first),
                                        sp_detail::column_pointer(half.x[1 - r], first),
                                        static_cast<std::size_t>(sh.own) * column_bytes};
        }
        tp_.events->host_exchange()->move_begin(2 + h, segments, *tp_.execution);
        sh.x_open       = true;
        sh.tapped_layer = tapped_layer;
    } else {
        (void)sh, (void)h, (void)tapped_layer;
    }
}

template <class Arena>
void Qwen4Forward<Arena>::sp_close_x_gather(SpHalf& sh, int h) {
    if constexpr (kExecute) {
        if (!sh.x_open) { return; }
        tp_.events->host_exchange()->move_finish(2 + h, *tp_.execution);
        sh.x_open = false;
        if (feature_tap_ != nullptr && sh.tapped_layer >= 0) {
            feature_tap_->capture_layer_columns(sh.tapped_layer, sh.half->x, sh.half->first);
        }
        sh.tapped_layer = -1;
    } else {
        (void)sh, (void)h;
    }
}

template <class Arena>
void Qwen4Forward<Arena>::sp_stage1(SpHalf& sh, int h, SpKind kind, std::size_t layer) {
    if constexpr (kExecute) {
        sp_close_x_gather(sh, h);
        sp_close_reduce(sh, h);
        if (kind != SpKind::Ffn && layer > 0 && sp_tapped(layer - 1)) {
            sp_begin_x_gather(sh, h, static_cast<int>(layer - 1));
        }
        PlainHalf& half      = *sh.half;
        const std::int32_t n = sh.own;
        const auto own       = [&](const RankTensors& t) {
            return sp_detail::windows(t, sh.own_first, n);
        };
        const std::array<WorkspaceArena*, 2> ws{&work(0), &work(1)};
        const ExecutionContext& execution = *tp_.execution;
        const RankTensors x_own           = own(half.x);
        each([&](int r) {
            const Qwen4BlockParameters& b = block(r, layer);
            const Tensor& norm = kind == SpKind::Ffn ? b.post_attention_norm : b.input_norm;
            Tensor out         = sh.h[at(r)];
            ops::rmsnorm(x_own[at(r)], norm, config_.rms_norm_eps, true, out, s(r));
        });
        sp_detail::Handoff handoff;
        handoff.open(ops::ScopedNvfp4Activation::Mode::Capture, sh.h, sh.own_plane);
        if (kind == SpKind::Attention) {
            const std::array<const AttentionParameters*, 2> p{
                &std::get<AttentionParameters>(block(0, layer).mixer),
                &std::get<AttentionParameters>(block(1, layer).mixer)};
            attention_projection_split(sh.h, p, own(sh.q), own(sh.gate), own(sh.k), own(sh.v), ws,
                                       execution);
        } else if (kind == SpKind::Gdn) {
            const std::array<const GdnParameters*, 2> p{
                &std::get<GdnParameters>(block(0, layer).mixer),
                &std::get<GdnParameters>(block(1, layer).mixer)};
            gdn_control_split(sh.h, p, own(sh.g), own(sh.beta), ws, execution);
            // The peer's heads over the owned columns: the split gating Op with each rank's copy
            // of the other rank's weights (two calls).
            const Qwen4PeerGdnControl& peer0 = state(0).runtime->sp_peer_control[layer];
            const Qwen4PeerGdnControl& peer1 = state(1).runtime->sp_peer_control[layer];
            {
                auto scope0 = work(0).scope();
                auto scope1 = work(1).scope();
                const std::array<Tensor, 2> a_log{peer0.a_log, peer1.a_log};
                const std::array<Tensor, 2> dt_bias{peer0.dt_bias, peer1.dt_bias};
                if (!peer0.paired) {
                    ops::gdn_gating_proj_column_parallel(sh.h, {peer0.first, peer1.first}, a_log,
                                                         dt_bias, ws, sh.g_peer, sh.beta_peer,
                                                         execution);
                } else {
                    ops::gdn_gating_proj_column_parallel(
                        sh.h, {peer0.first, peer1.first}, {peer0.second, peer1.second}, a_log,
                        dt_bias, ws, sh.g_peer, sh.beta_peer, execution);
                }
            }
            gdn_projection_split(sh.h, p, own(sh.qkv), own(sh.z), ws, execution);
        } else {
            project_swiglu_column_parallel(
                sh.h,
                {&std::get<DenseParameters>(block(0, layer).ffn).gate_up,
                 &std::get<DenseParameters>(block(1, layer).ffn).gate_up},
                own(sh.activation), ws, execution);
        }
        handoff.require_used("block input");
        // All-gather: the captured plane (and the peer heads' gating) to the peer.
        ops::HostExchange::MoveSegments segments{};
        for (std::size_t r = 0; r < 2; ++r) {
            const std::size_t peer                = 1 - r;
            const ops::Nvfp4ActivationPlane& mine = sh.own_plane[r];
            ops::Nvfp4ActivationPlane& theirs     = sh.peer_plane[peer];
            theirs.layout                         = mine.layout;
            theirs.k                              = mine.k;
            theirs.code_bytes                     = mine.code_bytes;
            theirs.scale_bytes                    = mine.scale_bytes;
            segments[r][0]                        = {mine.codes, theirs.codes, mine.code_bytes};
            segments[r][1]                        = {mine.scales, theirs.scales, mine.scale_bytes};
            if (kind == SpKind::Gdn) {
                const std::int32_t first = sh.own_first[r];
                segments[r][2] = {sh.g_peer[r].data, sp_detail::column_pointer(sh.g[peer], first),
                                  sh.g_peer[r].bytes()};
                segments[r][3] = {sh.beta_peer[r].data,
                                  sp_detail::column_pointer(sh.beta[peer], first),
                                  sh.beta_peer[r].bytes()};
            }
        }
        tp_.events->host_exchange()->move_begin(h, segments, execution);
        sh.gather_open = true;
    } else {
        (void)sh, (void)h, (void)kind, (void)layer;
    }
}

template <class Arena>
void Qwen4Forward<Arena>::sp_stage2(SpHalf& sh, int h, SpKind kind, std::size_t layer) {
    if constexpr (kExecute) {
        ops::HostExchange* exchange       = tp_.events->host_exchange();
        const ExecutionContext& execution = *tp_.execution;
        exchange->move_finish(h, execution);
        sh.gather_open       = false;
        PlainHalf& half      = *sh.half;
        const std::int32_t n = sh.own;
        const std::int32_t W = sh.count;
        const std::int32_t H = hidden();
        const std::array<std::int32_t, 2> peer_first{sh.own_first[1], sh.own_first[0]};
        const auto own = [&](const RankTensors& t) {
            return sp_detail::windows(t, sh.own_first, n);
        };
        const auto peer = [&](const RankTensors& t) {
            return sp_detail::windows(t, peer_first, n);
        };
        const std::array<WorkspaceArena*, 2> ws{&work(0), &work(1)};
        bind_half(half, [&] {
            auto stage = scopes();
            // The consuming projections key on the input pointer and never read it: the owned
            // columns' buffer, of the same width, stands in for the peer's.
            sp_detail::Handoff handoff;
            handoff.open(ops::ScopedNvfp4Activation::Mode::Consume, sh.h, sh.peer_plane);
            RankTensors mixed;
            std::array<const LinearParameters*, 2> output{};
            if (kind == SpKind::Attention) {
                const std::array<const AttentionParameters*, 2> p{
                    &std::get<AttentionParameters>(block(0, layer).mixer),
                    &std::get<AttentionParameters>(block(1, layer).mixer)};
                attention_projection_split(sh.h, p, peer(sh.q), peer(sh.gate), peer(sh.k),
                                           peer(sh.v), ws, execution);
                handoff.require_used("attention input");
                const auto& a        = *rc().attention;
                const auto head_dim  = dimension(a.head_dim);
                const auto heads     = dimension(a.num_attention_heads);
                const auto kv_heads  = dimension(a.num_key_value_heads);
                RankTensors qn       = alloc(DType::BF16, {head_dim, heads, W});
                RankTensors kn       = alloc(DType::BF16, {head_dim, kv_heads, W});
                RankTensors attended = alloc(DType::BF16, {head_dim, heads, W});
                const auto fidx      = dimension(config_.compact_layer_indices[layer]);
                each([&](int r) {
                    dense_attention_rank(layer, r, fidx, sh.q, sh.gate, sh.k, sh.v, qn, kn,
                                         attended);
                });
                const auto qw = dimension(a.query_width());
                mixed         = {attended[0].view({qw, W}), attended[1].view({qw, W})};
                output        = {&p[0]->output, &p[1]->output};
            } else if (kind == SpKind::Gdn) {
                const std::array<const GdnParameters*, 2> p{
                    &std::get<GdnParameters>(block(0, layer).mixer),
                    &std::get<GdnParameters>(block(1, layer).mixer)};
                gdn_projection_split(sh.h, p, peer(sh.qkv), peer(sh.z), ws, execution);
                handoff.require_used("GDN input");
                const auto& g    = *rc().gdn;
                const auto heads = dimension(g.linear_num_value_heads);
                const auto dv    = dimension(g.linear_value_head_dim);
                const auto kw    = dimension(g.key_width());
                const auto vw    = dimension(g.value_width());
                const auto gidx  = dimension(config_.compact_layer_indices[layer]);
                const auto gl    = static_cast<std::uint32_t>(gidx);
                RankTensors qc   = alloc(DType::BF16, {kw, W});
                RankTensors kc   = alloc(DType::BF16, {kw, W});
                RankTensors vc   = alloc(DType::BF16, {vw, W});
                RankTensors o    = alloc(DType::BF16, {dv, heads, W});
                RankTensors on   = alloc(DType::BF16, {dv, heads, W});
                each([&](int r) {
                    LinearAttentionStatePool& pool = *state(r).linear;
                    Tensor conv_in                 = pool.conv_slot(gl, c(r).source_slot);
                    Tensor conv_out                = pool.conv_slot(gl, c(r).destination_slot);
                    ops::causal_conv1d_silu_split(sh.qkv[at(r)], p[at(r)]->convolution, conv_in,
                                                  conv_out, qc[at(r)], kc[at(r)], vc[at(r)], s(r));
                });
                gdn_recurrence(layer, gl, gidx, qc, kc, vc, sh.g, sh.beta, sh.z, o, on);
                mixed  = {on[0].view({vw, W}), on[1].view({vw, W})};
                output = {&p[0]->output, &p[1]->output};
            } else {
                const std::array<const LinearParameters*, 2> gu{
                    &std::get<DenseParameters>(block(0, layer).ffn).gate_up,
                    &std::get<DenseParameters>(block(1, layer).ffn).gate_up};
                project_swiglu_column_parallel(sh.h, gu, peer(sh.activation), ws, execution);
                handoff.require_used("FFN gate|up");
                mixed  = sh.activation;
                output = {&std::get<DenseParameters>(block(0, layer).ffn).down,
                          &std::get<DenseParameters>(block(1, layer).ffn).down};
            }
            // Row-parallel output projection as a plain partial per rank; the peer's columns
            // first, so their reduce-scatter moves while the owned columns are projected.
            const auto policy  = split_policy(output, "SP prefill output projection");
            const auto weights = split_weights(output);
            {
                auto scope0 = work(0).scope();
                auto scope1 = work(1).scope();
                ops::linear_column_parallel(peer(mixed), weights, peer(sh.partial), policy, ws,
                                            execution);
            }
            ops::HostExchange::MoveSegments segments{};
            const std::size_t column_bytes = static_cast<std::size_t>(H) * 2;
            for (std::size_t r = 0; r < 2; ++r) {
                segments[r][0] = {sp_detail::column_pointer(sh.partial[r], peer_first[r]),
                                  sp_detail::column_pointer(half.staging[1 - r], peer_first[r]),
                                  static_cast<std::size_t>(n) * column_bytes};
            }
            exchange->move_begin(h, segments, execution);
            sh.reduce_open = true;
            {
                auto scope0 = work(0).scope();
                auto scope1 = work(1).scope();
                ops::linear_column_parallel(own(mixed), weights, own(sh.partial), policy, ws,
                                            execution);
            }
        });
    } else {
        (void)sh, (void)h, (void)kind, (void)layer;
    }
}

template <class Arena>
void Qwen4Forward<Arena>::sp_layer(std::size_t layer, SpCall& call) {
    const SpKind mixer =
        config_.layer_types[layer] == MixerKind::FullAttention ? SpKind::Attention : SpKind::Gdn;
    for (const SpKind kind : {mixer, SpKind::Ffn}) {
        sp_stage1(call.halves[0], 0, kind, layer);
        sp_stage1(call.halves[1], 1, kind, layer);
        sp_stage2(call.halves[0], 0, kind, layer);
        sp_stage2(call.halves[1], 1, kind, layer);
    }
}

template <class Arena>
void Qwen4Forward<Arena>::sp_end(SpCall& call) {
    const std::size_t last = config_.layer_types.size() - 1;
    for (int h = 0; h < 2; ++h) {
        SpHalf& sh = call.halves[at(h)];
        sp_close_x_gather(sh, h);
        sp_close_reduce(sh, h);
        sp_begin_x_gather(sh, h, sp_tapped(last) ? static_cast<int>(last) : -1);
    }
    for (int h = 0; h < 2; ++h) { sp_close_x_gather(call.halves[at(h)], h); }
}

template <class Arena>
void Qwen4Forward<Arena>::sp_abandon(SpCall& call) noexcept {
    if constexpr (kExecute) {
        if (tp_.events == nullptr || tp_.events->host_exchange() == nullptr) { return; }
        ops::HostExchange* exchange = tp_.events->host_exchange();
        for (int h = 0; h < 2; ++h) {
            SpHalf& sh = call.halves[at(h)];
            if (sh.reduce_open || sh.gather_open) { exchange->abandon_move(h, *tp_.execution); }
            if (sh.x_open) { exchange->abandon_move(2 + h, *tp_.execution); }
            sh.reduce_open = sh.gather_open = sh.x_open = false;
            sh.tapped_layer                             = -1;
        }
    } else {
        (void)call;
    }
}

} // namespace ninfer::models::qwen3_5::execution
