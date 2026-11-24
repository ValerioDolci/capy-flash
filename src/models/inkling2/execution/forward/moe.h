#pragma once

// The MoE of a visit: the sigmoid router (replicated: both ranks route every token to the same
// experts), each resident bank's capped-SwiGLU experts into one FP32 partial (Store for the
// first bank, Accumulate for the next), and one BF16 rounding of the sum: moe_partial_to_bf16 at
// one device, allreduce_sum_f32 across two at decode widths (both give the same bits for the same
// partials); at prefill widths two devices round each partial to BF16 before a BF16 all-reduce.
//
// Member definitions of Forward (forward.h), included only by forward.cpp.

#include "models/inkling2/execution/forward.h"

#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/moe_bank.h"
#include "ninfer/ops/moe_sigmoid_route.h"

#include <stdexcept>

namespace ninfer::models::inkling2::execution {

// Widest call whose two MoE partials are summed in FP32 with one BF16 rounding (decode and
// verification widths, so blocks of up to this many tokens keep the decode's bits); wider calls
// round each partial to BF16 before the sum.
inline constexpr std::int32_t kExactMoeExchangeMaxTokens = 16;

template <class Arena>
void Forward<Arena>::moe(std::uint32_t index, const RankTensors& u, RankTensors& z) {
    const std::int32_t H = hidden();
    const auto held      = scopes();
    auto partial         = alloc(DType::FP32, {H, T_});
    moe_partial(index, u, partial);
    if (width_ == 1) {
        if constexpr (kExecute) { ops::moe_partial_to_bf16(partial[0], z[0], s(0)); }
    } else if (T_ > kExactMoeExchangeMaxTokens) {
        // Prefill widths: each rank rounds its partial to BF16 and the pair sums in BF16 with FP32
        // accumulation (the attention output's exchange), half the bytes of the FP32 exchange.
        auto staging = alloc(DType::BF16, {H, T_});
        if constexpr (kExecute) {
            each([&](int r) { ops::moe_partial_to_bf16(partial[at(r)], z[at(r)], s(r)); });
            ops::allreduce_sum(z, staging, *tp_.execution, *tp_.events);
        }
    } else {
        auto staging = alloc(DType::FP32, {H, T_});
        if constexpr (kExecute) {
            ops::allreduce_sum_f32(partial, staging, &z, *tp_.execution, *tp_.events);
        }
    }
}

template <class Arena>
void Forward<Arena>::moe_partial(std::uint32_t index, const RankTensors& u, RankTensors& partial) {
    const Visit& v  = config_.visits[index];
    const auto H    = hidden();
    const auto k    = static_cast<std::int32_t>(config_.experts_per_token);
    const auto F    = static_cast<std::int32_t>(config_.expert_intermediate);
    const auto held = scopes();
    auto ids        = alloc(DType::I32, {k, T_});
    auto weights    = alloc(DType::FP32, {k, T_});
    const ops::CappedSwiglu activation{config_.gate_softcap, config_.up_softcap};
    each([&](int r) {
        const auto& p = layer(r, v.layer).moe;
        if constexpr (kExecute) {
            ops::moe_sigmoid_route(u[at(r)], p.router, p.router_bias, k, config_.route_scale,
                                   ids[at(r)], weights[at(r)], s(r));
            for (std::size_t i = 0; i < p.banks.size(); ++i) {
                const auto& bank = p.banks[i];
                auto scope       = work(r).scope();
                ops::moe_bank_capped_swiglu(
                    u[at(r)], ids[at(r)], weights[at(r)],
                    ops::MoeBank{bank.gate_up, bank.down, bank.first_expert}, activation,
                    i == 0 ? ops::MoeBankOutput::Store : ops::MoeBankOutput::Accumulate,
                    partial[at(r)], work(r), s(r));
            }
            prefetch(r, index + 1, Prefetch::Inputs);
        } else {
            (void)p, (void)activation, (void)partial;
            reserve(ops::moe_bank_capped_swiglu_workspace_bytes(k, F, H, max_tokens_), r);
        }
    });
    tap(static_cast<std::int32_t>(index), "route_ids", ids);
    tap(static_cast<std::int32_t>(index), "route_weights", weights);
}

} // namespace ninfer::models::inkling2::execution
