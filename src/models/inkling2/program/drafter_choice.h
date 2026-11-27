#pragma once

// What a lane's next decode step is under --spec auto: a round of one of the artifact's two
// drafters, or an ordinary step. The DFlash2 drafter sees the last kWindow positions of the
// context, the DSpark drafter's global layer every position. The constants are generated
// (drafter_tuning.h, tools/inkling2_tune: docs/maintainer/inkling2-auto-tuning.md) and mirrored
// by the tool's simulator (tools/inkling2_tune/policy.py), which must change with this file.
//
// The prior: for every committed generated token, whether its 4-gram (the token and the three
// before it) last occurred more than kWindow positions back, i.e. beyond the DFlash2 window and
// not inside it. A request whose prompt holds at least kLongPrompt tokens starts with DSpark,
// otherwise with DFlash2; over the last kRecent generated tokens a share of such tokens of at
// least kEnter moves the prior to DSpark and one of at most kLeave back to DFlash2. Host only:
// one hash map from each 4-gram to the last position it ended at.
//
// The measure: each drafter keeps a mean of the drafts its rounds accepted (the running mean, then
// exponential with weight kWeight, or kOtherWeight for a round of the drafter not chosen and for a
// round between ordinary steps). With
// kCompare, once both hold kEvidence rounds the drafter with the better rate, (1 + mean) per
// millisecond of its round (StepCosts: measured while the engine runs), proposes; the other takes
// over only when its rate is higher by kHysteresis, and with kPriorWins far copies (a share of at
// least kEnter) still give DSpark. On a long prompt the first 2 kStart rounds alternate between the
// drafters and then every kExplore-th round is the other drafter's (real rounds, whose drafts the
// target verifies), unless the other's rate is already below the chosen one's by kSettle.
//
// No drafter: once some drafter holds kEvidence rounds, the lane takes ordinary steps (which feed
// both drafters' contexts) while the best measured rate is below (1 - kMargin) per millisecond of
// an ordinary step, and every kProbe ordinary steps one round renews the means.

#include "models/inkling2/program/drafter_costs.h"
#include "models/inkling2/program/drafter_tuning.h"
#include "ninfer/types.h"

#include <array>
#include <cstdint>
#include <deque>
#include <span>
#include <unordered_map>

namespace ninfer::models::inkling2::detail {

class DrafterChoice {
public:
    static constexpr std::uint32_t kWindow     = tuning::kWindow;
    static constexpr std::uint32_t kRecent     = tuning::kRecent;
    static constexpr std::uint32_t kLongPrompt = tuning::kLongPrompt;
    static constexpr double kEnter             = tuning::kEnter;
    static constexpr double kLeave             = tuning::kLeave;
    static constexpr double kWeight            = tuning::kWeight;
    static constexpr std::uint32_t kEvidence   = tuning::kEvidence;
    static constexpr bool kCompare             = tuning::kCompare;
    static constexpr double kHysteresis        = tuning::kHysteresis;
    static constexpr bool kPriorWins           = tuning::kPriorWins;
    static constexpr std::uint32_t kExplore    = tuning::kExplore;
    static constexpr std::uint32_t kStart      = tuning::kStart;
    static constexpr double kOtherWeight       = tuning::kOtherWeight;
    static constexpr double kSettle            = tuning::kSettle;
    static constexpr double kMargin            = tuning::kMargin;
    static constexpr std::uint32_t kProbe      = tuning::kProbe;

    // A new request whose first `prompt` tokens are its prompt.
    void start(std::uint32_t prompt);

    // The next step of a lane whose committed tokens are `tokens` (prompt, then the generated
    // ones; the last is the anchor): a round of Mtp (DSpark) or DFlash2, or None for an ordinary
    // step.
    [[nodiscard]] SpeculativeBackend choose(std::span<const TokenId> tokens,
                                            const StepCosts& costs);

    // After a round of `drafter` at position `frontier` that accepted `accepted` drafts.
    void observe(SpeculativeBackend drafter, std::uint32_t accepted, std::uint32_t frontier,
                 const StepCosts& costs);

    // The far-copy prior's drafter, and the steps that changed it since start().
    [[nodiscard]] SpeculativeBackend prior() const noexcept { return prior_; }

    [[nodiscard]] std::uint32_t switches() const noexcept { return switches_; }

private:
    [[nodiscard]] double rate(std::size_t drafter, std::uint32_t frontier,
                              const StepCosts& costs) const noexcept;

    std::uint32_t prompt_     = 0;
    std::uint32_t seen_       = 0; // tokens accounted for
    SpeculativeBackend prior_ = SpeculativeBackend::DFlash2;
    double share_             = -1.0; // far copies over the last kRecent tokens (-1: fewer)
    std::unordered_map<std::uint64_t, std::uint32_t> last_; // 4-gram -> position it last ended at
    std::deque<bool> recent_;                               // the last kRecent generated tokens
    std::uint32_t hits_     = 0;                            // set entries of recent_
    std::uint32_t switches_ = 0;
    // Per drafter (DSpark, DFlash2): the mean of accepted drafts and the rounds it holds.
    std::array<double, 2> mean_{};
    std::array<std::uint32_t, 2> rounds_{};
    std::size_t current_   = 1;     // the drafter the measure last chose
    std::size_t choice_    = 1;     // the choice of the last round (before exploring)
    std::uint32_t explore_ = 0;     // rounds since the last round of the other drafter
    bool plain_            = false; // ordinary steps
    std::uint32_t since_   = 0;     // ordinary steps since the last round
};

} // namespace ninfer::models::inkling2::detail
