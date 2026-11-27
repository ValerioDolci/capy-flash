#include "models/inkling2/program/drafter_choice.h"

#include <algorithm>

namespace ninfer::models::inkling2::detail {
namespace {

// The 4-gram ending at `end` as one 64-bit key (token ids are below 2^24; a collision only blurs
// the signal).
[[nodiscard]] std::uint64_t gram(std::span<const TokenId> tokens, std::uint32_t end) {
    std::uint64_t key = 0;
    for (std::uint32_t i = end - 3; i <= end; ++i) {
        key = key * 0x100000001b3ULL ^ static_cast<std::uint32_t>(tokens[i]);
    }
    return key;
}

[[nodiscard]] SpeculativeBackend backend(std::size_t drafter) noexcept {
    return drafter == 0 ? SpeculativeBackend::Mtp : SpeculativeBackend::DFlash2;
}

[[nodiscard]] StepKind round_kind(std::size_t drafter) noexcept {
    return drafter == 0 ? StepKind::RoundMtp : StepKind::RoundDFlash2;
}

} // namespace

void DrafterChoice::start(std::uint32_t prompt) {
    *this    = DrafterChoice{};
    prompt_  = prompt;
    prior_   = prompt >= kLongPrompt ? SpeculativeBackend::Mtp : SpeculativeBackend::DFlash2;
    current_ = prior_ == SpeculativeBackend::Mtp ? 0 : 1;
    choice_  = current_;
}

double DrafterChoice::rate(std::size_t drafter, std::uint32_t frontier,
                           const StepCosts& costs) const noexcept {
    return (1.0 + mean_[drafter]) / costs.ms(round_kind(drafter), frontier);
}

SpeculativeBackend DrafterChoice::choose(std::span<const TokenId> tokens, const StepCosts& costs) {
    for (; seen_ < tokens.size(); ++seen_) {
        if (seen_ < 3) { continue; }
        const std::uint64_t key = gram(tokens, seen_);
        const auto found        = last_.find(key);
        if (seen_ >= prompt_) {
            const bool far = found != last_.end() && found->second + kWindow <= seen_;
            recent_.push_back(far);
            hits_ += far ? 1U : 0U;
            if (recent_.size() > kRecent) {
                hits_ -= recent_.front() ? 1U : 0U;
                recent_.pop_front();
            }
        }
        if (found != last_.end()) {
            found->second = seen_;
        } else {
            last_.emplace(key, seen_);
        }
    }
    share_ = recent_.size() == kRecent ? double(hits_) / double(kRecent) : -1.0;
    if (share_ >= 0.0) {
        if (prior_ == SpeculativeBackend::DFlash2 && share_ >= kEnter) {
            prior_ = SpeculativeBackend::Mtp;
            ++switches_;
        } else if (prior_ == SpeculativeBackend::Mtp && share_ <= kLeave) {
            prior_ = SpeculativeBackend::DFlash2;
            ++switches_;
        }
    }
    if (plain_ && since_ < kProbe) {
        ++since_;
        return SpeculativeBackend::None;
    }
    const auto frontier = static_cast<std::uint32_t>(tokens.size() - 1U);
    std::size_t choice  = prior_ == SpeculativeBackend::Mtp ? 0 : 1;
    if (kCompare && rounds_[0] >= kEvidence && rounds_[1] >= kEvidence) {
        const std::size_t other = 1 - current_;
        choice =
            rate(other, frontier, costs) > rate(current_, frontier, costs) * (1.0 + kHysteresis)
                ? other
                : current_;
        if (kPriorWins && prior_ == SpeculativeBackend::Mtp && share_ >= kEnter) { choice = 0; }
    }
    choice_                 = choice;
    std::size_t drafter     = choice;
    const std::size_t other = 1 - choice;
    // Exploring stops once the other drafter's rate is below the chosen one's by kSettle.
    const bool settled =
        rounds_[0] >= 2 && rounds_[1] >= 2 &&
        rate(other, frontier, costs) < rate(choice, frontier, costs) * (1.0 - kSettle);
    const std::uint32_t started = rounds_[0] + rounds_[1];
    if (prompt_ >= kLongPrompt && !settled) {
        if (started < 2 * kStart) {
            drafter = started % 2;
        } else if (kExplore > 0 && ++explore_ >= kExplore) {
            drafter  = other;
            explore_ = 0;
        }
    }
    return backend(drafter);
}

void DrafterChoice::observe(SpeculativeBackend drafter, std::uint32_t accepted,
                            std::uint32_t frontier, const StepCosts& costs) {
    const std::size_t i = drafter == SpeculativeBackend::Mtp ? 0 : 1;
    ++rounds_[i];
    // A round of the drafter not chosen, or one between ordinary steps, renews a mean seen rarely.
    const double weight =
        std::max(i == choice_ && !plain_ ? kWeight : kOtherWeight, 1.0 / double(rounds_[i]));
    mean_[i] += weight * (double(accepted) - mean_[i]);
    since_ = 0;
    if (i == choice_) { current_ = choice_; }
    double best = -1.0;
    for (std::size_t d = 0; d < 2; ++d) {
        if (rounds_[d] >= kEvidence) { best = std::max(best, rate(d, frontier, costs)); }
    }
    if (best >= 0.0) { plain_ = best < (1.0 - kMargin) / costs.ms(StepKind::Ordinary, frontier); }
}

} // namespace ninfer::models::inkling2::detail
