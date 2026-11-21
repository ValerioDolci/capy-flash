#pragma once

// The forward's construction and the operands every visit uses: the rank loop, the arena, the
// per-rank Linear and the taps.
//
// Member definitions of Forward (forward.h), included only by forward.cpp, which instantiates
// them for execution and for planning.

#include "models/inkling2/execution/forward.h"

#include "core/device_scope.h"
#include "core/weight_view.h"
#include "ninfer/ops/linear.h"

#include <optional>
#include <stdexcept>
#include <string>

namespace ninfer::models::inkling2::execution {

template <class Arena>
Forward<Arena>::Forward(const TextConfig& config, std::array<Rank, 2> ranks, int width,
                        ForwardCollectives collectives, std::int32_t max_tokens,
                        std::int32_t max_logit_columns)
    : config_(config), r_(ranks), width_(width), tp_(collectives), max_tokens_(max_tokens),
      max_logit_columns_(max_logit_columns) {
    if (width_ != 1 && width_ != 2) {
        throw std::invalid_argument("Inkling 2 forward width must be 1 or 2");
    }
    if (max_tokens_ <= 0 || max_logit_columns_ < 0 || max_logit_columns_ > max_tokens_) {
        throw std::invalid_argument("Inkling 2 forward column bounds are invalid");
    }
    for (int r = 0; r < width_; ++r) {
        const Rank& rank = r_[at(r)];
        if (rank.config == nullptr || rank.parameters == nullptr || rank.work == nullptr ||
            rank.parameters->text.layers.size() != config_.num_hidden_layers ||
            rank.config->visits.size() != config_.visits.size()) {
            throw std::invalid_argument("Inkling 2 forward rank " + std::to_string(r) +
                                        " has no Text Parameters");
        }
        const Model& model = rank.parameters->model;
        embedding_[at(r)]  = native_weight(
            model.weight(model.weights().text.token_embedding, rank.parameters->device).view);
    }
    if constexpr (kExecute) {
        if (width_ == 2 &&
            (tp_.execution == nullptr || tp_.events == nullptr || tp_.execution->tp != 2)) {
            throw std::invalid_argument("Inkling 2 forward at tp 2 has no collectives");
        }
    }
}

template <class Arena>
std::optional<VisitState::ConvTails> Forward<Arena>::records(int r, std::uint32_t slot) const {
    const ConvRecord* record = r_[at(r)].record;
    if (record == nullptr) { return std::nullopt; }
    return record->records(slot, T_);
}

template <class Arena>
auto Forward<Arena>::sink(std::int32_t block) const -> std::optional<RankTensors> {
    if constexpr (kExecute) {
        if (r_[0].features == nullptr) { return std::nullopt; }
        RankTensors out;
        for (int r = 0; r < width_; ++r) {
            out[at(r)] = r_[at(r)].features->features.slice(2, block, 1).view({hidden(), T_});
        }
        return out;
    } else {
        (void)block;
        return std::nullopt;
    }
}

template <class Arena>
std::int32_t Forward<Arena>::hidden() const {
    return static_cast<std::int32_t>(config_.hidden_size);
}

template <class Arena>
template <class Body>
void Forward<Arena>::each(Body&& body) {
    if constexpr (kExecute) {
        if (width_ == 2) {
            const ScopedCurrentDevice restore;
            for (int r = 0; r < width_; ++r) {
                ScopedCurrentDevice::select(r_[at(r)].device);
                body(r);
            }
            return;
        }
    }
    for (int r = 0; r < width_; ++r) { body(r); }
}

template <class Arena>
auto Forward<Arena>::scopes() -> Scopes {
    Scopes out;
    for (int r = 0; r < width_; ++r) { out.held[at(r)].emplace(work(r).scope()); }
    return out;
}

template <class Arena>
auto Forward<Arena>::alloc(DType dtype, std::initializer_list<std::int32_t> shape) -> RankTensors {
    RankTensors out;
    for (int r = 0; r < width_; ++r) { out[at(r)] = work(r).alloc(dtype, shape); }
    return out;
}

template <class Arena>
void Forward<Arena>::reserve(std::size_t bytes, int rank) {
    if constexpr (!kExecute) {
        if (bytes == 0) { return; }
        for (int r = 0; r < width_; ++r) {
            if (rank >= 0 && rank != r) { continue; }
            auto scope = work(r).scope();
            (void)work(r).alloc_bytes(bytes);
        }
    } else {
        (void)bytes, (void)rank;
    }
}

template <class Arena>
void Forward<Arena>::linear(int r, const Tensor& x, const LinearParameters& p, Tensor& out) {
    if constexpr (kExecute) {
        ops::linear(x, p.weight, out, p.policy, work(r), s(r));
    } else {
        (void)x, (void)out;
        reserve(ops::linear_workspace_capacity_bytes(p.weight.qtype, p.weight.n, p.weight.k,
                                                     p.policy, 1, max_tokens_),
                r);
    }
}

template <class Arena>
void Forward<Arena>::tap(std::int32_t visit, std::string_view part, const RankTensors& value) {
    if constexpr (kExecute) {
        if (tap_ == nullptr || !tap_->wants(visit, part)) { return; }
        std::optional<ScopedCurrentDevice> rank0;
        if (width_ == 2) { rank0.emplace(r_[0].device); }
        tap_->capture(visit, part, value[0], first_position_, s(0));
        if (width_ == 2) { tap_->compare_ranks(visit, part, value[0], value[1], s(0), s(1)); }
    } else {
        (void)visit, (void)part, (void)value;
    }
}

} // namespace ninfer::models::inkling2::execution
