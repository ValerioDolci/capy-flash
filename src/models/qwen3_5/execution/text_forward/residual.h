#pragma once

// The hyper-connection residual (docs/maintainer/qwen4-exp-model.md#hyper-connection-residual):
// the embedding widened to S streams, one mixer per block input (the pending block output
// committed in the same pass), the PLE layer, and the output mixer that feeds the head. A model
// without hyper-connections takes the plain residual (plain.h) from run().
//
// Member definitions of TextForward (text_forward.h), included only by text_forward.cpp, which
// instantiates them for execution and for planning.

#include "models/qwen3_5/execution/text_forward.h"
#include "core/device.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/ple.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/scatter.h"
#include "ops/host_moe/host_moe_mailbox.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <exception>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::execution {

template <class Arena>
void TextForward<Arena>::run(const Tensor& ids, const Tensor* ple_embedding, Tensor& wide_out,
                             const VisualColumns* visual) {
    const RankTensors rank_ids{ids, Tensor{}};
    RankTensors rank_ple;
    if (ple_embedding != nullptr) { rank_ple = {*ple_embedding, Tensor{}}; }
    RankTensors rank_out{wide_out, Tensor{}};
    run(rank_ids, ple_embedding != nullptr ? &rank_ple : nullptr, rank_out, visual);
}

// The residual state of a hyper-connection forward: the committed wide residual, the next block
// input (`mixed`), the two block outputs and their stream gates (a block writes one pair while the
// next mixer still reads the other), the pending output that the next mixer or a commit folds
// into the residual, and the tp 2 all-reduce staging of the two exchanges per block (FP32 for the
// MoE).
template <class Arena>
struct TextForward<Arena>::HcStreams {
    RankTensors wide, mixed;
    std::array<RankTensors, 2> y, inject;
    int pending = -1;
    int next    = 0;
    RankTensors staging_bf16, staging_fp32;
};

template <class Arena>
void TextForward<Arena>::run(const RankTensors& ids, const RankTensors* ple_embedding,
                             RankTensors& wide_out, const VisualColumns* visual) {
    if (!has_hc()) {
        if (ple_embedding != nullptr || config_.ple) {
            throw std::invalid_argument("Qwen4 PLE needs the hyper-connection residual");
        }
        run_plain(ids, wide_out, visual);
        return;
    }
    if (feature_tap_ != nullptr) {
        throw std::invalid_argument("drafter feature taps need the plain residual");
    }
    const std::int32_t T = columns();
    auto scope           = scopes();
    HcStreams st;
    st.wide = wide_buffers(wide_out, T);
    embed_streams(ids, visual, st.wide);
    if (width_ == 2) {
        st.staging_bf16 = alloc(DType::BF16, {hidden(), T});
        if (moe_ffn()) { st.staging_fp32 = alloc(DType::FP32, {hidden(), T}); }
    }
    st.mixed     = alloc(DType::BF16, {hidden(), T});
    st.y[0]      = alloc(DType::BF16, {hidden(), T});
    st.y[1]      = alloc(DType::BF16, {hidden(), T});
    st.inject[0] = alloc(DType::FP32, {streams(), T});
    st.inject[1] = alloc(DType::FP32, {streams(), T});
    for (std::size_t layer = 0; layer < config_.layer_types.size(); ++layer) {
        try {
            hc_block(layer, ple_embedding, st);
        } catch (const std::exception& error) {
            throw std::runtime_error("text/layers/" + std::to_string(layer) + " " + phase() +
                                     (width_ == 2 ? " tp2" : "") + " columns=" + std::to_string(T) +
                                     ": " + error.what());
        }
    }
    hc_commit(st);
    if constexpr (kExecute) {
        if (Qwen4Taps* taps = prompt_taps(-1)) {
            auto tap_scope = scopes();
            RankTensors x  = alloc(DType::BF16, {hidden(), T});
            head(st.wide, x);
            capture_taps(*taps, -1, x);
        }
    }
}

template <class Arena>
void TextForward<Arena>::embed_streams(const RankTensors& ids, const VisualColumns* visual,
                                       RankTensors& wide) {
    const std::int32_t T = columns();
    auto embed_scope     = scopes();
    RankTensors e        = alloc(DType::BF16, {hidden(), T});
    each([&](int r) {
        if constexpr (kExecute) {
            ops::embedding(ids[at(r)], parameters(r).text.token_embedding, e[at(r)], s(r));
            if (visual != nullptr) {
                ops::scatter(visual->embeddings[at(r)], visual->indices[at(r)], e[at(r)], s(r));
            }
            ops::hyper_connection_expand(e[at(r)], streams(), wide[at(r)], s(r));
        }
    });
}

template <class Arena>
void TextForward<Arena>::hc_block(std::size_t layer, const RankTensors* ple_embedding,
                                  HcStreams& st) {
    const auto compact   = dimension(config_.compact_layer_indices[layer]);
    const bool attention = config_.layer_types[layer] == MixerKind::FullAttention;
    if (block(0, layer).ple) {
        if (ple_embedding == nullptr) {
            throw std::logic_error("Qwen4Exp PLE block has no n-gram embedding");
        }
        hc_commit(st);
        ple(layer, *ple_embedding, st.wide);
    }
    hc_mix(st, &Qwen4BlockParameters::attention_hc, layer);
    const std::size_t hc_prefetch = hc_exchange_prefetch();
    if (hc_prefetch != 0) { hc_stage_prefetch(hc_ffn_hint(layer, hc_prefetch)); }
    // The mixer's output projection is its last work; a sliced one (tp 2 prompt chunk) runs the
    // FFN mixer slice by slice inside its exchange (MixerTail).
    MixerTail tail{&st, &Qwen4BlockParameters::ffn_hc, layer};
    {
        struct Offer {
            MixerTail*& slot;

            ~Offer() { slot = nullptr; }
        } offer{mixer_tail_};

        mixer_tail_ = &tail;
        // A hyper-connection model's attention is QSA (qwen4_extensions binds both).
        if (!attention) {
            gdn(layer, st.mixed, st.y[at(st.next)], compact, st.staging_bf16);
        } else {
            qsa(layer, st.mixed, &st.y[at(st.next)], static_cast<std::uint32_t>(compact),
                st.staging_bf16);
        }
    }
    st.pending = st.next;
    st.next ^= 1;
    if (tail.ran) {
        st.pending = -1; // committed and mixed by the output projection
    } else {
        hc_mix(st, &Qwen4BlockParameters::ffn_hc, layer); // commits the mixer output too
    }
    if (hc_prefetch != 0) {
        hc_stage_prefetch(hc_next_hint(layer));
        stage_wait_prefetch(layer);
    }
    ffn(layer, st.mixed, st.y[at(st.next)], st.staging_bf16, st.staging_fp32);
    st.pending = st.next;
    st.next ^= 1;
    if constexpr (kExecute) {
        if (Qwen4Taps* taps = prompt_taps(static_cast<std::int32_t>(layer))) {
            hc_commit(st);
            capture_taps(*taps, static_cast<std::int32_t>(layer), st.wide);
        }
    }
}

// The hyper-connection block's decode prefetch (graph decode at tp 2 through the mailbox; eager
// rounds, staged exchanges and NINFER_TP_EXCHANGE_PREFETCH_KIB=0 leave all of it off). Rank 1
// spins in the MoE exchange and rank 0 waits there for the host's experts; each prefetches the
// next layer's HC mixer and mixer input meanwhile, up to half its L2 (rank 0's join waits for
// the rest of the prefetch only when the host is quicker).
inline std::size_t hc_half_l2(int device) {
    static std::array<std::atomic<std::size_t>, 16> cached{};
    if (device < 0 || device >= static_cast<int>(cached.size())) { return 0; }
    std::size_t bytes = cached[static_cast<std::size_t>(device)].load(std::memory_order_relaxed);
    if (bytes == 0) {
        int l2 = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device));
        bytes = static_cast<std::size_t>(std::max(l2, 0)) / 2;
        cached[static_cast<std::size_t>(device)].store(bytes, std::memory_order_relaxed);
    }
    return bytes;
}

inline std::size_t hc_append_tensor(const Tensor& t, std::size_t bytes, ExchangePrefetch& out) {
    const std::size_t n = std::min<std::size_t>(t.bytes(), bytes);
    if (t.data != nullptr && n != 0) { out.push_back({t.data, n}); }
    return n;
}

// The leading whole rows of `w` within `bytes` (codes, then scales); returns their bytes.
inline std::size_t hc_append_weight(const Weight& w, std::size_t bytes, ExchangePrefetch& out) {
    const std::size_t row = prefetch_row_bytes(w, PrefetchFormats::HyperConnection);
    if (row == 0) { return 0; }
    const auto rows = std::min<std::int64_t>(w.n, static_cast<std::int64_t>(bytes / row));
    append_prefetch_rows(w, 0, rows, out, PrefetchFormats::HyperConnection);
    return static_cast<std::size_t>(rows) * row;
}

template <class Arena>
std::size_t TextForward<Arena>::hc_exchange_bytes() const {
    return static_cast<std::size_t>(hidden()) * static_cast<std::size_t>(columns()) * sizeof(float);
}

// The mailbox budget (NINFER_TP_EXCHANGE_PREFETCH_KIB) of the block's FP32 exchanges, 0 when off.
template <class Arena>
std::size_t TextForward<Arena>::hc_exchange_prefetch() const {
    if constexpr (kExecute) {
        if (width_ == 2 && !c(0).prefill) {
            return exchange_prefetch_budget(*tp_.events, *tp_.execution, hc_exchange_bytes());
        }
    }
    return 0;
}

template <class Arena>
void TextForward<Arena>::hc_stage_prefetch(const std::array<ExchangePrefetch, 2>& hint) {
    if constexpr (kExecute) {
        stage_exchange_prefetch(*tp_.events, *tp_.execution, hc_exchange_bytes(), hint);
    } else {
        (void)hint;
    }
}

template <class Arena>
std::array<ExchangePrefetch, 2> TextForward<Arena>::hc_ffn_hint(std::size_t layer,
                                                                std::size_t bytes) const {
    std::array<ExchangePrefetch, 2> hint;
    for (int r = 0; r < width_; ++r) {
        ExchangePrefetch& out = hint[at(r)];
        if (const auto* moe = std::get_if<Qwen4MoeParameters>(&block(r, layer).ffn)) {
            std::size_t left = bytes;
            left -= hc_append_tensor(moe->router, left, out);
            if (moe->shared) { (void)hc_append_weight(moe->shared->gate_up.weight, left, out); }
        }
    }
    return hint;
}

// Rank r's next HC mixer weights, then the leading rows of that layer's mixer input projection.
template <class Arena>
void TextForward<Arena>::append_hc_next(int r, std::size_t layer, std::size_t bytes,
                                        ExchangePrefetch& out) const {
    const Qwen4BlockParameters& next = block(r, layer);
    std::size_t left                 = bytes;
    left -= hc_append_weight(next.attention_hc.weights.down, left, out);
    left -= hc_append_weight(next.attention_hc.weights.up, left, out);
    if (const auto* g = std::get_if<GdnParameters>(&next.mixer)) {
        append_mixer_input_prefetch(*g, left, out, PrefetchFormats::HyperConnection);
    } else if (const auto* q = std::get_if<QsaParameters>(&next.mixer)) {
        append_prefetch_prefix(q->projection.weight, left, out, PrefetchFormats::HyperConnection);
    }
}

template <class Arena>
std::array<ExchangePrefetch, 2> TextForward<Arena>::hc_next_hint(std::size_t layer) const {
    std::array<ExchangePrefetch, 2> hint;
    if (layer + 1 >= config_.layer_types.size()) { return hint; }
    for (int r = 0; r < width_; ++r) {
        // Rank 0's own wait covers its part (stage_wait_prefetch).
        append_hc_next(r, layer + 1, r == 0 ? 0 : hc_half_l2(r_[at(r)].device), hint[at(r)]);
    }
    return hint;
}

template <class Arena>
void TextForward<Arena>::stage_wait_prefetch(std::size_t layer) {
    if constexpr (kExecute) {
        const std::size_t bytes = hc_half_l2(r_[at(0)].device);
        if (bytes == 0 || !moe_ffn() || layer + 1 >= config_.layer_types.size()) { return; }
        ExchangePrefetch out;
        append_hc_next(0, layer + 1, bytes, out);
        routed_experts(0, layer)->set_wait_prefetch(out);
    } else {
        (void)layer;
    }
}

template <class Arena>
void TextForward<Arena>::hc_commit(HcStreams& st) {
    if (st.pending < 0) { return; }
    each([&](int r) {
        if constexpr (kExecute) {
            ops::hyper_connection_combine(st.y[at(st.pending)][at(r)],
                                          st.inject[at(st.pending)][at(r)], st.wide[at(r)], s(r));
        }
    });
    st.pending = -1;
}

template <class Arena>
void TextForward<Arena>::hc_mix(HcStreams& st,
                                HyperConnectionParameters Qwen4BlockParameters::* member,
                                std::size_t layer) {
    reserve(ops::hyper_connection_mix_workspace_bytes(first_, last_, true, hidden(), hc_rank()));
    each([&](int r) {
        if constexpr (kExecute) {
            const HyperConnectionParameters& hc = block(r, layer).*member;
            if (st.pending >= 0) {
                ops::hyper_connection_combine_mix(st.wide[at(r)], st.y[at(st.pending)][at(r)],
                                                  st.inject[at(st.pending)][at(r)], hc.weights,
                                                  config_.rms_norm_eps, st.mixed[at(r)],
                                                  &st.inject[at(st.next)][at(r)], work(r), s(r));
            } else {
                ops::hyper_connection_mix(st.wide[at(r)], hc.weights, config_.rms_norm_eps,
                                          st.mixed[at(r)], &st.inject[at(st.next)][at(r)], work(r),
                                          s(r));
            }
        }
    });
    st.pending = -1;
}

template <class Arena>
void TextForward<Arena>::hc_mix_columns(const MixerTail& tail, std::int32_t first, std::int32_t n) {
    HcStreams& st = *tail.st;
    each([&](int r) {
        if constexpr (kExecute) {
            const HyperConnectionParameters& hc = block(r, tail.layer).*tail.member;
            Tensor wide                         = st.wide[at(r)].slice(1, first, n);
            Tensor mixed                        = st.mixed[at(r)].slice(1, first, n);
            Tensor inject        = st.inject[at(st.next ^ 1)][at(r)].slice(1, first, n);
            const Tensor y       = st.y[at(st.next)][at(r)].slice(1, first, n);
            const Tensor pending = st.inject[at(st.next)][at(r)].slice(1, first, n);
            ops::hyper_connection_combine_mix(wide, y, pending, hc.weights, config_.rms_norm_eps,
                                              mixed, &inject, work(r), s(r));
        }
    });
}

template <class Arena>
void TextForward<Arena>::head(const RankTensors& wide, RankTensors& x) {
    head({&qwen4(0).head, width_ == 2 ? &qwen4(1).head : nullptr}, wide, x);
}

template <class Arena>
void TextForward<Arena>::head(std::array<const HyperConnectionParameters*, 2> hc,
                              const RankTensors& wide, RankTensors& x) {
    if (!has_hc()) {
        if constexpr (kExecute) {
            for (int r = 0; r < width_; ++r) { x[at(r)] = wide[at(r)]; }
        }
        return;
    }
    reserve(ops::hyper_connection_mix_workspace_bytes(first_, last_, false, hidden(), hc_rank()));
    each([&](int r) {
        if constexpr (kExecute) {
            ops::hyper_connection_mix(wide[at(r)], hc[at(r)]->weights, config_.rms_norm_eps,
                                      x[at(r)], nullptr, work(r), s(r));
        }
    });
}

template <class Arena>
void TextForward<Arena>::head(const HyperConnectionParameters& hc, const Tensor& wide, Tensor& x) {
    const RankTensors rank_wide{wide, Tensor{}};
    RankTensors rank_x{x, Tensor{}};
    head({&hc, nullptr}, rank_wide, rank_x);
    x = rank_x[0];
}

template <class Arena>
void TextForward<Arena>::ple(std::size_t layer, const RankTensors& embedding, RankTensors& wide) {
    const std::int32_t T  = columns();
    const auto& ple       = *config_.ple;
    const std::int32_t S  = streams();
    auto scope            = scopes();
    RankTensors key_value = alloc(DType::BF16, {wide_rows() + hidden(), T});
    RankTensors gated     = alloc(DType::BF16, {wide_rows(), T});
    RankTensors normed    = alloc(DType::BF16, {wide_rows(), T});
    RankTensors slots     = alloc(DType::I32, {2});
    each([&](int r) {
        const PleParameters& p = *block(r, layer).ple;
        if constexpr (kExecute) {
            // A decode round's deferred rows: wait for the host's gather, then upload.
            const ForwardRuntime* runtime = state(r).runtime;
            const Tensor& rows            = embedding[at(r)];
            if (runtime != nullptr && runtime->decode_ple_staged != nullptr &&
                rows.data == runtime->decode_ple_rows.data) {
                ops::host::consume_host_flag(s(r), runtime->decode_ple_staged);
                CUDA_CHECK(cudaMemcpyAsync(rows.data, runtime->decode_ple_host, rows.bytes(),
                                           cudaMemcpyHostToDevice, s(r)));
            }
            // A prompt chunk's deferred rows: the host gather ends, then this rank uploads.
            if (pending_ple_ != nullptr && rows.data == pending_ple_->rows[at(r)].data) {
                pending_ple_->wait();
                CUDA_CHECK(cudaMemcpyAsync(rows.data, pending_ple_->host, rows.bytes(),
                                           cudaMemcpyHostToDevice, s(r)));
            }
        }
        linear(r, embedding[at(r)], p.key_value, key_value[at(r)]);
        if constexpr (kExecute) {
            const ForwardColumns& col = c(r);
            const auto W = col.width, B = col.batch;
            ops::ple_gate(wide[at(r)], key_value[at(r)], p.query_norm, p.key_norm, S,
                          config_.rms_norm_eps, gated[at(r)], s(r));
            ops::grouped_offset_rmsnorm(gated[at(r)], p.conv_norm, S, config_.rms_norm_eps,
                                        normed[at(r)], s(r));
            Tensor states = state(r).runtime->ple_states;
            Tensor source, destination;
            if (col.prefill) {
                source      = slots[at(r)].slice(0, 0, 1);
                destination = slots[at(r)].slice(0, 1, 1);
                ops::set_i32_scalar(source, col.source_slot, s(r));
                ops::set_i32_scalar(destination, col.destination_slot, s(r));
            } else {
                source      = *col.source_slots;
                destination = *col.destination_slots;
            }
            Tensor normed3 = normed[at(r)].view({wide_rows(), W, B});
            Tensor gated3  = gated[at(r)].view({wide_rows(), W, B});
            Tensor wide3   = wide[at(r)].view({wide_rows(), W, B});
            ops::ple_dilated_conv(normed3, gated3, p.convolution,
                                  static_cast<std::int32_t>(ple.conv_dilation()), states, source,
                                  wide3, s(r));
            if (col.replay_records != nullptr) {
                // Verification: the history advances at commit by the accepted prefix.
                const Tensor& record = state(r).runtime->ple_record;
                if (record.data == nullptr || record.ne[0] != wide_rows() || record.ne[1] != W ||
                    record.ne[2] < B) {
                    throw std::logic_error("Qwen4Exp PLE speculative record is unavailable");
                }
                CUDA_CHECK(::ninfer::memcpy_device_async(record.data, normed[at(r)].data,
                                                         normed[at(r)].bytes(), s(r)));
            } else {
                ops::ple_conv_advance(normed3, col.prefill ? nullptr : col.valid_columns,
                                      static_cast<std::int32_t>(ple.conv_history()), states, source,
                                      destination, s(r));
            }
        }
    });
}

} // namespace ninfer::models::qwen3_5::execution
