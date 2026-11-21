#pragma once

// The output head: h = bf16(rmsnorm(depth attention of every block state and the last visit's
// output, final_norm)); logits = the output head over the last `logit_columns` columns, each rank
// its vocabulary half at tp 2 gathered on rank 0 (concat_rows_rank0), BF16 as the projection
// rounds them; capped = softcap(logits) in FP32 on rank 0.
//
// Member definitions of Forward (forward.h), included only by forward.cpp.

#include "models/inkling2/execution/forward.h"

#include "ninfer/ops/rmsnorm_f32.h"
#include "ninfer/ops/softcap.h"
#include "ninfer/ops/tp2/concat_rows.h"

#include <stdexcept>

namespace ninfer::models::inkling2::execution {

template <class Arena>
void Forward<Arena>::head(Block& b, const RankTensors& x, const ForwardColumns& columns,
                          const ForwardLogits& out) {
    const std::int32_t H = hidden();
    const auto held      = scopes();
    auto h               = alloc(DType::FP32, {H, T_, 1});
    depth(b.states, x, {&text(0).final_queries, width_ == 2 ? &text(1).final_queries : nullptr}, h);
    // The head input lands in the drafter's feature block 3 when the call has sinks.
    auto hn = alloc(DType::BF16, {H, T_});
    if (const auto features = sink(3)) { hn = *features; }
    each([&](int r) {
        if constexpr (kExecute) {
            ops::rmsnorm_f32(h[at(r)].view({H, T_}), text(r).final_norm, config_.rms_norm_eps,
                             hn[at(r)], s(r));
        }
    });
    tap(ForwardTap::kHeadTap, "final", hn);

    const std::int32_t n = kExecute ? columns.logit_columns : max_logit_columns_;
    if (n == 0) { return; }
    const auto vocabulary = static_cast<std::int32_t>(config_.unpadded_vocab_size);
    if constexpr (kExecute) {
        if (out.logits == nullptr || out.logits->dtype != DType::BF16 ||
            out.logits->ne[0] != vocabulary || out.logits->ne[1] != n ||
            !out.logits->is_contiguous() ||
            (out.capped != nullptr &&
             (out.capped->dtype != DType::FP32 || out.capped->ne[0] != vocabulary ||
              out.capped->ne[1] != n || !out.capped->is_contiguous()))) {
            throw std::invalid_argument("Inkling 2 forward: logits must be BF16 [Vu, n] (capped "
                                        "FP32 [Vu, n]) on rank 0");
        }
    }
    RankTensors last;
    for (int r = 0; r < width_; ++r) { last[at(r)] = hn[at(r)].slice(1, T_ - n, n); }
    if (width_ == 1) {
        Tensor dummy;
        linear(0, last[0], text(0).output_head, kExecute ? *out.logits : dummy);
    } else {
        RankTensors part;
        for (int r = 0; r < width_; ++r) {
            part[at(r)] = work(r).alloc(DType::BF16, {text(r).output_head.weight.n, n});
        }
        Tensor staging = work(0).alloc(DType::BF16, {text(1).output_head.weight.n, n});
        each([&](int r) { linear(r, last[at(r)], text(r).output_head, part[at(r)]); });
        if constexpr (kExecute) {
            ops::concat_rows_rank0(part, staging, *out.logits, *tp_.execution, *tp_.events);
        }
    }
    if constexpr (kExecute) {
        if (out.capped != nullptr) {
            std::optional<ScopedCurrentDevice> rank0;
            if (width_ == 2) { rank0.emplace(r_[0].device); }
            ops::softcap(*out.logits, config_.final_logit_softcap, *out.capped, s(0));
        }
    }
}

} // namespace ninfer::models::inkling2::execution
