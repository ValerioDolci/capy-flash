#pragma once

// The closing projection of a block - into the hyper-connection block output or added into the
// plain residual - and its tp 2 row-parallel exchange: the BF16 all-reduce, the FP32 partials of
// the mixer output and the teacher-forcing taps of the exchanged partials.
//
// Member definitions of TextForward (text_forward.h), included only by text_forward.cpp, which
// instantiates them for execution and for planning.

#include "models/qwen3_5/execution/text_forward.h"
#include "models/qwen3_5/execution/tp2/linear_split.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/tp2/linear_add.h"

#include <algorithm>

namespace ninfer::models::qwen3_5::execution {

// Abandons the HostExchange sum a forward opened (null: none) when the work before its
// finish_sum_f32 throws, so a failed prompt chunk leaves the exchange free for the next one.
// finish_sum_f32 closes the call, and abandoning a closed call does nothing.
class OpenExchangeSum {
public:
    explicit OpenExchangeSum(ops::HostExchange* pipe) noexcept : pipe_(pipe) {}

    ~OpenExchangeSum() {
        if (pipe_ != nullptr) { pipe_->abandon_sum_f32(); }
    }

    OpenExchangeSum(const OpenExchangeSum&)            = delete;
    OpenExchangeSum& operator=(const OpenExchangeSum&) = delete;

private:
    ops::HostExchange* pipe_;
};

template <class Arena>
Qwen4Taps* TextForward<Arena>::exchange_taps() const {
    if constexpr (kExecute) {
        const ForwardRuntime* runtime = state(0).runtime;
        if (width_ == 2 && c(0).prefill && runtime != nullptr && runtime->taps != nullptr &&
            runtime->taps->wants_exchange()) {
            return runtime->taps;
        }
    }
    return nullptr;
}

template <class Arena>
void TextForward<Arena>::mixer_exchanged(std::int32_t layer) {
    // The routed-expert backend of a Text layer may start the next layer's expert copies.
    if constexpr (kExecute) {
        if (moe_ffn() && layer >= 0 && static_cast<std::size_t>(layer) < mtp_layer()) {
            each([&](int r) {
                routed_experts(r, static_cast<std::size_t>(layer))->mixer_exchanged(layer, s(r));
            });
        }
    } else {
        (void)layer;
    }
}

template <class Arena>
void TextForward<Arena>::output_projection(const RankTensors& x,
                                           std::array<const LinearParameters*, 2> p, RankTensors& y,
                                           const RankTensors& staging, std::int32_t layer,
                                           int site) {
    if (width_ == 1) {
        linear(0, x[0], *p[0], y[0]);
        return;
    }
    // The mixer output exchange carries the ranks' FP32 partials (an FP16 payload in prompt
    // chunks, FP32 in decode) and rounds the sum to BF16 once, as one device does.
    if (p[0]->weight.qtype == QType::Q8_G32_FP16 ||
        p[0]->weight.qtype == QType::MXFP8_E4M3FN_G32_UE8M0) {
        const std::int32_t T = columns();
        auto scope           = scopes();
        RankTensors part     = alloc(DType::FP32, {hidden(), T});
        RankTensors stage    = alloc(DType::FP32, {hidden(), T});
        if (mixer_tail_ != nullptr && last_ > kPrefillOutputSliceColumns) {
            // The FFN mixer of one slice (prefill_output_slice: under 1.5 slices) runs above the
            // partials.
            const std::int32_t widest =
                std::min(last_, kPrefillOutputSliceColumns + kPrefillOutputSliceColumns / 2 - 1);
            reserve(ops::hyper_connection_mix_workspace_bytes(std::min(first_, widest), widest,
                                                              true, hidden(), hc_rank()));
        }
        if constexpr (kExecute) {
            // A prompt chunk wider than one slice projects and sends by column slices, so the
            // exchange of a slice travels while the next ones are projected (a Linear's columns
            // are independent, and narrowing and combining are elementwise: the bits of one
            // call). The teacher-forcing taps read whole partials and keep the one call.
            ops::HostExchange* sliced =
                c(0).prefill && T > kPrefillOutputSliceColumns && exchange_taps() == nullptr
                    ? tp_.events->host_exchange()
                    : nullptr;
            if (sliced != nullptr &&
                sliced->begin_sum_f32(part, stage, &y, prefill_moe_exchange().payload,
                                      *tp_.execution)) {
                const OpenExchangeSum open_sum{sliced};
                each([&](int r) {
                    for (std::int32_t first = 0, n = 0; first < T; first += n) {
                        n                = prefill_output_slice(first, T);
                        auto slice_scope = scopes();
                        Tensor out       = part[at(r)].slice(1, first, n);
                        linear(r, x[at(r)].slice(1, first, n), *p[at(r)], out);
                        sliced->columns_ready(r, first, n);
                    }
                });
                // A hyper-connection block's mixer output: the work that follows it up to the
                // FFN (combine into the residual, FFN mixer) is per column, so each slice goes on
                // as soon as it has arrived instead of the ranks idling until the last one has.
                if (MixerTail* tail = mixer_tail_) {
                    for (std::int32_t first = 0, n = 0; first < T; first += n) {
                        n = prefill_output_slice(first, T);
                        if (!sliced->combine_columns(n)) { break; } // INT8 payload: one combine
                        hc_mix_columns(*tail, first, n);
                        tail->ran = true;
                    }
                }
                sliced->finish_sum_f32();
                mixer_exchanged(layer);
                return;
            }
        }
        each([&](int r) { linear(r, x[at(r)], *p[at(r)], part[at(r)]); });
        if constexpr (kExecute) {
            (void)staging;
            if (Qwen4Taps* taps = exchange_taps()) { // FP32 partials of the mixer
                taps->capture_exchange(layer, site, part[0], part[1], c(0).first_position, s(0),
                                       s(1));
            }
            ops::HostExchange* pipe = c(0).prefill ? tp_.events->host_exchange() : nullptr;
            if (pipe != nullptr &&
                pipe->begin_sum_f32(
                    part, stage, &y,
                    prefill_moe_exchange().payload == ops::HostExchange::Payload::INT8
                        ? ops::HostExchange::Payload::INT8 // same payload as the MoE exchange
                        : ops::HostExchange::Payload::FP16,
                    *tp_.execution)) {
                const OpenExchangeSum open_sum{pipe};
                pipe->columns_ready(0, 0, T);
                pipe->columns_ready(1, 0, T);
                pipe->finish_sum_f32();
                mixer_exchanged(layer);
            } else {
                ops::allreduce_sum_f32(part, stage, &y, *tp_.execution, *tp_.events);
            }
        }
        return;
    }
    if constexpr (kExecute) {
        if (Qwen4Taps* taps = exchange_taps()) {
            // linear_row_parallel unrolled: each rank's partial, the tap, the BF16 all-reduce.
            each([&](int r) { linear(r, x[at(r)], *p[at(r)], y[at(r)]); });
            taps->capture_exchange(layer, site, y[0], y[1], c(0).first_position, s(0), s(1));
            ops::allreduce_sum(y, staging, *tp_.execution, *tp_.events);
            return;
        }
        const std::array<WorkspaceArena*, 2> ws{&work(0), &work(1)};
        project_row_parallel(x, p, y, staging, ws, *tp_.execution, *tp_.events);
    } else {
        for (int r = 0; r < width_; ++r) {
            const LinearParameters& rp = *p[at(r)];
            reserve(ops::linear_workspace_capacity_bytes(rp.weight.qtype, rp.weight.n, rp.weight.k,
                                                         rp.policy, first_, last_),
                    r);
        }
    }
}

template <class Arena>
void TextForward<Arena>::close(const RankTensors& in, std::array<const LinearParameters*, 2> p,
                               RankTensors& out, const RankTensors& staging, std::int32_t layer,
                               int site) {
    if (has_hc()) {
        output_projection(in, p, out, staging, layer, site);
        return;
    }
    if constexpr (kExecute) {
        if (width_ == 1) {
            ops::linear_add(in[0], p[0]->weight, out[0], p[0]->policy, work(0), s(0));
            return;
        }
        const std::array<WorkspaceArena*, 2> ws{&work(0), &work(1)};
        if (slices_.usable()) {
            // Slice s's projection begins its all-reduce on lane s (plain_ffn finishes it).
            const std::int32_t T = columns();
            for (std::int32_t slice = 0; slice < slices_.count; ++slice) {
                const std::int32_t first = slices_.first(slice);
                const std::int32_t count = slices_.columns(slice, T);
                RankTensors residual     = window(out, first, count);
                project_add_row_parallel(window(in, first, count), p, residual,
                                         window(staging, first, count), ws, *tp_.execution,
                                         *tp_.events, static_cast<int>(slice));
            }
            return;
        }
        // Under the half-chunk overlap the all-reduce is only begun, on the half's lane.
        project_add_row_parallel(in, p, out, staging, ws, *tp_.execution, *tp_.events, lane_);
    } else {
        (void)staging, (void)layer, (void)site, (void)out, (void)in;
        for (int r = 0; r < width_; ++r) {
            const Weight& w = p[at(r)]->weight;
            reserve(width_ == 1 ? ops::linear_add_workspace_capacity_bytes(
                                      w.qtype, w.n, w.k, p[at(r)]->policy, first_, last_)
                                : ops::linear_add_row_parallel_workspace_capacity_bytes(
                                      w.qtype, w.n, w.k, p[at(r)]->policy, first_, last_),
                    r);
        }
    }
}

} // namespace ninfer::models::qwen3_5::execution
