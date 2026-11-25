#pragma once

// The output head: h = bf16(rmsnorm(depth attention of every block state and the last visit's
// output, final_norm)); logits = the output head over the last `logit_columns` columns, each rank
// its vocabulary half at tp 2 gathered on rank 0 (concat_rows_rank0), BF16 as the projection
// rounds them; capped = softcap(logits) in FP32 on rank 0. With the Logits loop contrast a second
// head reads the reference coda's states, and on rank 0 the capped logits become the guided ones
// (ops::loop_contrast_logits), `logits` their BF16 rounding.
//
// Member definitions of Forward (forward.h), included only by forward.cpp.

#include "models/inkling2/execution/forward.h"

#include "ninfer/ops/loop_contrast.h"
#include "ninfer/ops/rmsnorm_f32.h"
#include "ninfer/ops/softcap.h"
#include "ninfer/ops/tp2/concat_rows.h"

#include <stdexcept>

namespace ninfer::models::inkling2::execution {

// The widest stacked projection of the Logits loop contrast's two heads: the BF16 Linear's
// columns keep the route of one column up to 16.
inline constexpr std::int32_t kStackedHeadColumns = 16;

template <class Arena>
void Forward<Arena>::head_input(const std::vector<const RankTensors*>& states, const RankTensors& x,
                                RankTensors& hn) {
    const std::int32_t H = hidden();
    const auto held      = scopes();
    auto h               = alloc(DType::FP32, {H, T_, 1});
    depth(states, x, {&text(0).final_queries, width_ == 2 ? &text(1).final_queries : nullptr}, h);
    each([&](int r) {
        if constexpr (kExecute) {
            ops::rmsnorm_f32(h[at(r)].view({H, T_}), text(r).final_norm, config_.rms_norm_eps,
                             hn[at(r)], s(r));
        }
    });
}

template <class Arena>
void Forward<Arena>::head_logits(const RankTensors& last, Tensor& logits) {
    const std::int32_t n = last[0].ne[1];
    if (width_ == 1) {
        linear(0, last[0], text(0).output_head, logits);
        return;
    }
    RankTensors part;
    for (int r = 0; r < width_; ++r) {
        part[at(r)] = work(r).alloc(DType::BF16, {text(r).output_head.weight.n, n});
    }
    Tensor staging = work(0).alloc(DType::BF16, {text(1).output_head.weight.n, n});
    each([&](int r) { linear(r, last[at(r)], text(r).output_head, part[at(r)]); });
    if constexpr (kExecute) {
        ops::concat_rows_rank0(part, staging, logits, *tp_.execution, *tp_.events);
    } else {
        (void)logits;
    }
}

template <class Arena>
void Forward<Arena>::head(Block& b, const RankTensors& x, const ForwardColumns& columns,
                          const ForwardLogits& out) {
    const std::int32_t H = hidden();
    const auto held      = scopes();
    // The head input lands in the drafter's feature block 3 when the call has sinks.
    auto hn = alloc(DType::BF16, {H, T_});
    if (const auto features = sink(3)) { hn = *features; }
    head_input(b.states, x, hn);
    tap(ForwardTap::kHeadTap, "final", hn);

    const std::int32_t n = kExecute ? columns.logit_columns : max_logit_columns_;
    if (n == 0) { return; }
    const auto vocabulary = static_cast<std::int32_t>(config_.unpadded_vocab_size);
    const bool contrast   = config_.contrast.mode == LoopContrast::Logits;
    if constexpr (kExecute) {
        const auto matches = [&](const Tensor* t, DType dtype) {
            return t->dtype == dtype && t->ne[0] == vocabulary && t->ne[1] == n &&
                   t->is_contiguous();
        };
        if (out.logits == nullptr || !matches(out.logits, DType::BF16) ||
            (out.capped != nullptr && !matches(out.capped, DType::FP32)) ||
            (out.reference != nullptr && (!contrast || !matches(out.reference, DType::FP32)))) {
            throw std::invalid_argument("Inkling 2 forward: logits must be BF16 [Vu, n] (capped "
                                        "and the contrast's reference FP32 [Vu, n]) on rank 0");
        }
    }
    const auto tail = [&](const RankTensors& t) {
        RankTensors last;
        for (int r = 0; r < width_; ++r) { last[at(r)] = t[at(r)].slice(1, T_ - n, n); }
        return last;
    };
    Tensor dummy;
    Tensor& logits = kExecute ? *out.logits : dummy;
    if (!contrast) {
        head_logits(tail(hn), logits);
        if constexpr (kExecute) {
            if (out.capped != nullptr) {
                std::optional<ScopedCurrentDevice> rank0;
                if (width_ == 2) { rank0.emplace(r_[0].device); }
                ops::softcap(*out.logits, config_.final_logit_softcap, *out.capped, s(0));
            }
        }
        return;
    }
    // The Logits loop contrast: the reference coda's head, then on rank 0 both soft caps and the
    // guided logits written over the capped ones and into `logits`.
    auto hr = alloc(DType::BF16, {H, T_});
    head_input(b.reference_states, *b.reference_output, hr);
    tap(ForwardTap::kHeadTap, "reference", hr);
    // Up to kStackedHeadColumns columns both heads' columns go through one projection: the output
    // head's weights are read once, and every column keeps the route (and the bits) of one column.
    Tensor main = logits, reference;
    if (2 * n <= kStackedHeadColumns) {
        auto both = alloc(DType::BF16, {H, 2 * n});
        each([&](int r) {
            if constexpr (kExecute) {
                const Tensor a = tail(hn)[at(r)], c = tail(hr)[at(r)];
                CUDA_CHECK(
                    ::ninfer::memcpy_device_async(both[at(r)].data, a.data, a.bytes(), s(r)));
                CUDA_CHECK(::ninfer::memcpy_device_async(both[at(r)].slice(1, n, n).data, c.data,
                                                         c.bytes(), s(r)));
            }
        });
        Tensor pair = work(0).alloc(DType::BF16, {vocabulary, 2 * n});
        head_logits(both, pair);
        main      = pair.slice(1, 0, n);
        reference = pair.slice(1, n, n);
    } else {
        head_logits(tail(hn), logits);
        reference = work(0).alloc(DType::BF16, {vocabulary, n});
        head_logits(tail(hr), reference);
    }
    Tensor capped =
        out.capped != nullptr ? *out.capped : work(0).alloc(DType::FP32, {vocabulary, n});
    Tensor reference_capped =
        out.reference != nullptr ? *out.reference : work(0).alloc(DType::FP32, {vocabulary, n});
    if constexpr (kExecute) {
        std::optional<ScopedCurrentDevice> rank0;
        if (width_ == 2) { rank0.emplace(r_[0].device); }
        const float cap = config_.final_logit_softcap;
        ops::softcap(main, cap, capped, s(0));
        ops::softcap(reference, cap, reference_capped, s(0));
        ops::loop_contrast_logits(capped, reference_capped, config_.contrast.strength,
                                  config_.contrast.adaptive, capped, *out.logits, s(0));
    }
}

} // namespace ninfer::models::inkling2::execution
