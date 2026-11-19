#pragma once
// Qwen4Exp graph MTP rounds: n-gram (PLE) row stages, and the check that the rows gathered early,
// right after a round synchronized, are the rows the next round queues (Qwen4PleRuntime).

#include "models/qwen3_5/program/round_buffers.h"
#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>

namespace ninfer::models::qwen3_5::detail {

// One lane's rows of a round: `count` window tokens written from column `first_column`, the
// sequence holding `frontier` tokens before them. An early stage runs before the round's tokens
// are committed, so it carries the context tokens before the window itself (`snapshot`): the
// n-gram size - 1 tokens before the frontier, fewer near the start of a sequence.
struct Qwen4PleStage {
    std::uint32_t lane       = 0;
    std::uint32_t frontier   = 0;
    std::uint32_t count      = 0;
    std::size_t first_column = 0;
    std::array<TokenId, qwen3_5::kMtpDecodeMaximumWidth> tokens{};
    bool snapshot               = false;
    std::uint32_t context_count = 0;
    std::array<TokenId, 8> context{};
};

// What the early staging reads from the Program: the n-gram context (n-gram size - 1), the MTP
// round geometry, the egress of the round that just synchronized and each lane's committed tokens.
struct Qwen4PleEarlySource {
    std::size_t context           = 0;
    std::uint32_t draft_window    = 0;
    std::uint32_t max_concurrency = 0;
    const MtpDecodeEgress* egress = nullptr;
    std::function<std::span<const TokenId>(std::uint32_t lane)> ledger;
};

// True when `early` holds exactly the rows `expected` needs: the same stages in the same order
// (lane, frontier, columns, window tokens) and, for each, a snapshot context equal to the
// `context` committed tokens before the frontier (all of them when the frontier is shorter).
// `ledger(lane)` returns that lane's committed tokens. Any difference (a lane or batch change, a
// count the commit cut, a replayed or stop-trimmed token) means the early rows are not used.
template <class Ledger>
[[nodiscard]] bool qwen4_ple_early_matches(std::span<const Qwen4PleStage> expected,
                                           std::span<const Qwen4PleStage> early,
                                           std::size_t context, Ledger&& ledger) {
    if (expected.size() != early.size()) { return false; }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const Qwen4PleStage& a = expected[i];
        const Qwen4PleStage& b = early[i];
        if (a.lane != b.lane || a.frontier != b.frontier || a.count != b.count ||
            a.first_column != b.first_column || a.count > a.tokens.size() ||
            !std::equal(a.tokens.begin(), a.tokens.begin() + a.count, b.tokens.begin())) {
            return false;
        }
        const std::span<const TokenId> committed = ledger(a.lane);
        const std::size_t before                 = std::min<std::size_t>(context, a.frontier);
        if (!b.snapshot || committed.size() < a.frontier || b.context_count != before ||
            before > b.context.size()) {
            return false;
        }
        if (!std::equal(committed.begin() + (a.frontier - before), committed.begin() + a.frontier,
                        b.context.begin())) {
            return false;
        }
    }
    return true;
}

} // namespace ninfer::models::qwen3_5::detail
