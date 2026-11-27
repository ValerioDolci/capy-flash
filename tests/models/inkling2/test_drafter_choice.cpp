// The choice between the paired drafters of Inkling 2 (models/inkling2/program/drafter_choice.h),
// host only, on constructed sequences of distinct tokens:
//
// 1. a short prompt starts with DFlash2 and a long one (at least the DFlash2 window) with DSpark;
// 2. a long request that writes new text moves to DFlash2 once kRecent generated tokens hold at
//    most the leave share of far copies, and stays there;
// 3. copying a passage that only the context beyond the window holds moves it to DSpark within
//    kRecent tokens, and new text brings it back;
// 4. repeating text inside the window is not a far copy: DFlash2 stays;
// 5. no drafter: rounds that accept nothing make the lane take kProbe ordinary steps between two
//    rounds once kEvidence rounds are in, rounds that accept again bring it back, and one accepted
//    draft per round keeps drafting at short contexts, unless rounds cost as much as 2.5 steps;
// 6. the step costs: the generated table at first, a startup calibration of one kind scales its
//    whole curve and the kinds not timed, measurements move the knot they fall at and the kind,
//    outliers count as kClamp times the estimate;
// 7. with the generated kExplore, every kExplore-th round of a long prompt is the other drafter's;
//    with kCompare, once both drafters hold kEvidence rounds the better measured one proposes
//    against the prior.

#include "models/inkling2/program/drafter_choice.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::SpeculativeBackend;
using ninfer::TokenId;
using ninfer::models::inkling2::detail::DrafterChoice;
using ninfer::models::inkling2::detail::StepCosts;
using ninfer::models::inkling2::detail::StepKind;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

// Fresh tokens: ids no earlier position used.
struct Fresh {
    TokenId next = 1000;

    void add(std::vector<TokenId>& tokens, std::size_t count) {
        for (std::size_t i = 0; i < count; ++i) { tokens.push_back(next++); }
    }
};

// The generated step costs (drafter_tuning.h), unmeasured.
const StepCosts kTable;

// Commits `tokens[from, to)` one round of `step` tokens at a time; returns the far-copy prior
// after the last round and the round it last changed at (-1: never).
std::pair<SpeculativeBackend, long> decode(DrafterChoice& choice,
                                           const std::vector<TokenId>& tokens, std::size_t from,
                                           std::size_t to, std::size_t step = 3) {
    SpeculativeBackend last = SpeculativeBackend::None;
    long changed            = -1;
    for (std::size_t end = from; end <= to; end += step) {
        (void)choice.choose(std::span<const TokenId>(tokens.data(), std::min(end, to)), kTable);
        const SpeculativeBackend now = choice.prior();
        if (last != SpeculativeBackend::None && now != last) { changed = long(end); }
        last = now;
    }
    return {last, changed};
}

} // namespace

int main() {
    try {
        constexpr std::size_t W = DrafterChoice::kWindow, R = DrafterChoice::kRecent;
        constexpr std::size_t Lp = DrafterChoice::kLongPrompt;
        {
            Fresh fresh;
            std::vector<TokenId> tokens;
            fresh.add(tokens, 300);
            DrafterChoice choice;
            choice.start(300);
            require(choice.choose(tokens, kTable) == SpeculativeBackend::DFlash2,
                    "a short prompt starts with DFlash2");
            fresh.add(tokens, 400);
            require(decode(choice, tokens, 301, tokens.size()).first ==
                            SpeculativeBackend::DFlash2 &&
                        choice.switches() == 0,
                    "new text after a short prompt stays with DFlash2");
        }
        {
            Fresh fresh;
            std::vector<TokenId> tokens;
            fresh.add(tokens, std::max(3 * W, Lp + 300));
            const std::size_t prompt = tokens.size();
            DrafterChoice choice;
            choice.start(std::uint32_t(prompt));
            require(choice.choose(std::span<const TokenId>(tokens.data(), prompt), kTable) ==
                        SpeculativeBackend::Mtp,
                    "a long prompt starts with DSpark");
            // New text: DFlash2 once kRecent generated tokens hold no far copy.
            fresh.add(tokens, 200);
            const auto [written, at] = decode(choice, tokens, prompt + 1, tokens.size());
            require(written == SpeculativeBackend::DFlash2 && at > 0 &&
                        std::size_t(at) <= prompt + R + 3,
                    "new text moves to DFlash2 within kRecent tokens");
            // A copy of a passage from the start of the prompt (beyond the window): DSpark.
            const std::size_t copy_begin = tokens.size();
            for (std::size_t i = 100; i < 100 + 120; ++i) { tokens.push_back(tokens[i]); }
            const auto [copying, at_copy] = decode(choice, tokens, copy_begin + 1, tokens.size());
            require(copying == SpeculativeBackend::Mtp && at_copy > 0 &&
                        std::size_t(at_copy) <= copy_begin + R,
                    "a far copy moves to DSpark within kRecent tokens");
            // New text again: back to DFlash2.
            const std::size_t again = tokens.size();
            fresh.add(tokens, 100);
            require(decode(choice, tokens, again + 1, tokens.size()).first ==
                        SpeculativeBackend::DFlash2,
                    "new text after the copy moves back to DFlash2");
            require(choice.switches() == 3, "three changes of drafter");
            // A repeat of text inside the window is not a far copy.
            const std::size_t near = tokens.size();
            for (std::size_t i = near - 90; i < near - 10; ++i) { tokens.push_back(tokens[i]); }
            require(decode(choice, tokens, near + 1, tokens.size()).first ==
                            SpeculativeBackend::DFlash2 &&
                        choice.switches() == 3,
                    "a copy inside the window keeps DFlash2");
        }
        {
            constexpr std::uint32_t E = DrafterChoice::kEvidence, P = DrafterChoice::kProbe;
            Fresh fresh;
            std::vector<TokenId> tokens;
            fresh.add(tokens, 300);
            DrafterChoice choice;
            choice.start(300);
            const StepCosts costs;
            // A step of the lane: the choice, then (for a round) `accepted` drafts at 1k tokens.
            const auto step = [&](std::uint32_t accepted) {
                fresh.add(tokens, 1);
                const SpeculativeBackend chosen = choice.choose(tokens, costs);
                if (chosen != SpeculativeBackend::None) {
                    choice.observe(chosen, accepted, 1000, costs);
                }
                return chosen;
            };
            for (std::uint32_t i = 0; i + 1 < E; ++i) {
                require(step(0) == SpeculativeBackend::DFlash2, "no decision before kEvidence");
            }
            require(step(0) == SpeculativeBackend::DFlash2, "the kEvidence-th round runs");
            for (std::uint32_t i = 0; i < P; ++i) {
                require(step(0) == SpeculativeBackend::None, "ordinary steps after empty rounds");
            }
            require(step(0) == SpeculativeBackend::DFlash2, "a round after kProbe steps");
            require(step(0) == SpeculativeBackend::None, "still ordinary steps");
            // Rounds that accept every draft: back to drafting within a few probes.
            std::uint32_t rounds = 0, steps = 0;
            for (std::uint32_t i = 0; i < 8 * (P + 1); ++i) {
                (step(3) == SpeculativeBackend::None ? steps : rounds) += 1;
            }
            require(rounds > 8 * P / 2 && steps <= 5 * P,
                    "accepting rounds bring the drafter back");
            DrafterChoice one;
            one.start(300);
            for (std::uint32_t i = 0; i < 4 * E; ++i) {
                fresh.add(tokens, 1);
                const SpeculativeBackend chosen = one.choose(tokens, costs);
                require(chosen == SpeculativeBackend::DFlash2,
                        "one accepted draft per round keeps drafting at short contexts");
                one.observe(chosen, 1, 500, costs);
            }
            // Rounds measured at 2.5 ordinary steps: one accepted draft no longer pays.
            StepCosts slow;
            slow.calibrate(StepKind::Ordinary, 500, 5.0);
            slow.calibrate(StepKind::RoundDFlash2, 500, 12.5);
            DrafterChoice costly;
            costly.start(300);
            SpeculativeBackend last = SpeculativeBackend::DFlash2;
            for (std::uint32_t i = 0; i < E + 1; ++i) {
                fresh.add(tokens, 1);
                last = costly.choose(tokens, slow);
                if (last != SpeculativeBackend::None) { costly.observe(last, 1, 500, slow); }
            }
            require(last == SpeculativeBackend::None,
                    "with rounds at 2.5 steps one accepted draft takes ordinary steps");
        }
        {
            const StepCosts table;
            require(table.ms(StepKind::RoundMtp, 350) > 2.0 * table.ms(StepKind::Ordinary, 350) &&
                        table.ms(StepKind::Ordinary, 150000) > table.ms(StepKind::Ordinary, 350),
                    "the table: rounds about two steps, steps grow with the context");
            StepCosts fast;
            const double round = fast.ms(StepKind::RoundMtp, 4700);
            fast.calibrate(StepKind::RoundMtp, 4700, round / 2.0);
            require(std::abs(fast.ms(StepKind::RoundMtp, 100000) * 2.0 -
                             table.ms(StepKind::RoundMtp, 100000)) < 1e-9 &&
                        std::abs(fast.ms(StepKind::Ordinary, 4700) * 2.0 -
                                 table.ms(StepKind::Ordinary, 4700)) < 1e-9,
                    "a calibration scales the kind's curve and the kinds not timed");
            StepCosts live      = fast;
            const double before = live.ms(StepKind::RoundMtp, 20000);
            for (int i = 0; i < 200; ++i) { live.observe(StepKind::RoundMtp, 4700, round * 0.8); }
            require(std::abs(live.ms(StepKind::RoundMtp, 4700) - round * 0.8) < 0.01 * round,
                    "measurements bring the knot to the measured time");
            require(live.ms(StepKind::RoundMtp, 20000) > before,
                    "the kind's scale follows the measurements elsewhere too");
            StepCosts spike;
            const double at = spike.ms(StepKind::Ordinary, 11600);
            spike.observe(StepKind::Ordinary, 11600, at * 100.0);
            require(spike.ms(StepKind::Ordinary, 11600) <=
                        at * (1.0 + StepCosts::kKnotWeight * (StepCosts::kClamp - 1.0)) + 1e-9,
                    "an outlier counts as kClamp times the estimate");
        }
        {
            // A long prompt whose drafters accept as many drafts: the first 2 kStart rounds
            // alternate, then every kExplore-th round is the drafter not chosen.
            Fresh fresh;
            std::vector<TokenId> tokens;
            fresh.add(tokens, DrafterChoice::kLongPrompt + 10);
            DrafterChoice choice;
            choice.start(std::uint32_t(tokens.size()));
            std::vector<SpeculativeBackend> rounds;
            for (std::uint32_t i = 0; rounds.size() < 2 * DrafterChoice::kStart + 64; ++i) {
                fresh.add(tokens, 1);
                const SpeculativeBackend chosen = choice.choose(tokens, kTable);
                if (chosen == SpeculativeBackend::None) { continue; }
                rounds.push_back(chosen);
                choice.observe(chosen, 2, std::uint32_t(tokens.size() - 1), kTable);
            }
            for (std::uint32_t i = 0; i < 2 * DrafterChoice::kStart; ++i) {
                require(rounds[i] ==
                            (i % 2 == 0 ? SpeculativeBackend::Mtp : SpeculativeBackend::DFlash2),
                        "the first 2 kStart rounds of a long prompt alternate");
            }
            if (DrafterChoice::kExplore > 0) {
                std::size_t dspark = 0;
                for (std::size_t i = 2 * DrafterChoice::kStart; i < rounds.size(); ++i) {
                    dspark += rounds[i] == SpeculativeBackend::Mtp ? 1U : 0U;
                }
                const std::size_t after    = rounds.size() - 2 * DrafterChoice::kStart;
                const std::size_t majority = std::max(dspark, after - dspark);
                require(after - majority <= after / DrafterChoice::kExplore + 1U,
                        "then the other drafter every kExplore rounds");
            }
        }
        if (DrafterChoice::kCompare) {
            // A long prompt whose DFlash2 rounds accept nothing and DSpark's every draft: once both
            // hold kEvidence rounds, DSpark proposes although the prior (new text) says DFlash2.
            Fresh fresh;
            std::vector<TokenId> tokens;
            fresh.add(tokens, DrafterChoice::kLongPrompt + 10);
            DrafterChoice choice;
            choice.start(std::uint32_t(tokens.size()));
            std::uint32_t dspark = 0, late = 0;
            for (std::uint32_t i = 0; i < 400; ++i) {
                fresh.add(tokens, 1);
                const SpeculativeBackend chosen = choice.choose(tokens, kTable);
                if (chosen == SpeculativeBackend::None) { continue; }
                const bool mtp = chosen == SpeculativeBackend::Mtp;
                choice.observe(chosen, mtp ? 3U : 0U, std::uint32_t(tokens.size() - 1), kTable);
                if (i >= 300) {
                    ++late;
                    dspark += mtp ? 1U : 0U;
                }
            }
            const std::uint32_t explored =
                DrafterChoice::kExplore > 0 ? late / DrafterChoice::kExplore + 2U : 0U;
            require(late > 0 && dspark + explored >= late,
                    "the better measured drafter proposes against the prior");
        }
    } catch (const std::exception& error) {
        std::printf("FAIL: %s\n", error.what());
        return 1;
    }
    std::printf("OK inkling2 drafter choice\n");
    return 0;
}
