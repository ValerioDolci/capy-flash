// qwen4_ple_early_matches: early n-gram rows are used only when they are exactly the rows the next
// round queues, their context included. Then Qwen4PleRuntime's early staging without a Program
// (a Qwen4PleEarlySource, a stager whose job records the rows it would gather): the rows gathered
// after a round are used when the next round queues exactly them, discarded otherwise, and the
// exception of a discarded run does not reach the round.
#include "models/qwen3_5/program/qwen4_ple_early.h"
#include "models/qwen3_5/program/qwen4_ple_runtime.h"

#include <cstdio>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

using ninfer::TokenId;
using ninfer::models::qwen3_5::MtpDecodeEgress;
using ninfer::models::qwen3_5::detail::Qwen4PleEarlySource;
using ninfer::models::qwen3_5::detail::Qwen4PleRuntime;
using ninfer::models::qwen3_5::detail::Qwen4PleStage;
using ninfer::models::qwen3_5::Qwen4PleStager;
using ninfer::models::qwen3_5::detail::qwen4_ple_early_matches;

namespace {
int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

constexpr std::size_t kContext = 2; // n-gram size 3

// The stage a round queues for `lane` at `frontier` (window tokens), and the early one with the
// context taken from `ledger`.
Qwen4PleStage queued(std::uint32_t lane, std::uint32_t frontier, std::vector<TokenId> window,
                     std::size_t column) {
    Qwen4PleStage s;
    s.lane         = lane;
    s.frontier     = frontier;
    s.count        = static_cast<std::uint32_t>(window.size());
    s.first_column = column;
    std::copy(window.begin(), window.end(), s.tokens.begin());
    return s;
}

Qwen4PleStage early_of(Qwen4PleStage s, const std::vector<TokenId>& ledger) {
    const std::size_t before = std::min<std::size_t>(kContext, s.frontier);
    s.snapshot               = true;
    s.context_count          = static_cast<std::uint32_t>(before);
    for (std::size_t k = 0; k < before; ++k) { s.context[k] = ledger[s.frontier - before + k]; }
    return s;
}
} // namespace

int main() {
    std::vector<std::vector<TokenId>> ledgers{{10, 11, 12, 13, 14}, {20, 21, 22}};
    const auto ledger = [&](std::uint32_t lane) -> std::span<const TokenId> {
        return ledgers[lane];
    };
    const auto match = [&](std::vector<Qwen4PleStage> expected, std::vector<Qwen4PleStage> early) {
        return qwen4_ple_early_matches(expected, early, kContext, ledger);
    };
    const Qwen4PleStage a = queued(0, 5, {15, 16, 17, 18}, 0);
    const Qwen4PleStage b = queued(1, 3, {23, 24, 25, 26}, 4);

    check(match({a, b}, {early_of(a, ledgers[0]), early_of(b, ledgers[1])}), "the same rows match");
    check(match({}, {}), "no rows match no rows");
    check(!match({a, b}, {early_of(b, ledgers[1]), early_of(a, ledgers[0])}),
          "another lane order (batch change)");
    check(!match({a}, {early_of(a, ledgers[0]), early_of(b, ledgers[1])}), "a lane left the batch");

    Qwen4PleStage other = early_of(a, ledgers[0]);
    other.lane          = 1;
    check(!match({a}, {other}), "another lane");

    other          = early_of(a, ledgers[0]);
    other.frontier = 4; // the commit took one token fewer than licensed
    check(!match({a}, {other}), "another frontier");

    other           = early_of(a, ledgers[0]);
    other.tokens[2] = 99; // a replayed target or a stop string changed a window token
    check(!match({a}, {other}), "another window token");

    other              = early_of(a, ledgers[0]);
    other.first_column = 4;
    check(!match({a}, {other}), "another column");

    other       = early_of(a, ledgers[0]);
    other.count = 3;
    check(!match({a}, {other}), "another width");

    other            = early_of(a, ledgers[0]);
    other.context[1] = 77; // the committed context differs from the snapshot
    check(!match({a}, {other}), "another context token");

    other          = early_of(a, ledgers[0]);
    other.snapshot = false;
    check(!match({a}, {other}), "a stage without a snapshot");

    // Near the start of a sequence the context is shorter than the n-gram context.
    ledgers.push_back({30});
    const Qwen4PleStage c = queued(2, 1, {31, 32}, 8);
    check(match({c}, {early_of(c, ledgers[2])}), "frontier below the n-gram context");
    other               = early_of(c, ledgers[2]);
    other.context_count = 2;
    check(!match({c}, {other}), "a context longer than the frontier");

    // The ledger shorter than the frontier (not committed yet).
    const Qwen4PleStage d = queued(1, 4, {27, 28}, 0);
    Qwen4PleStage ahead   = d;
    ahead.snapshot        = true;
    ahead.context_count   = 2;
    ahead.context[0]      = 22;
    ahead.context[1]      = 99;
    check(!match({d}, {ahead}), "a frontier past the committed tokens");

    // The runtime. One lane, n-gram context 2, two drafts (window 3). The round that just
    // synchronized licensed 2 tokens (11, 12) after the committed {1, 2, 3, 4, 5}; its next
    // drafts are 21, 22. After the commit the ledger is {1, 2, 3, 4, 5, 11} and the next round
    // queues frontier 6, window {12, 21, 22}, context {5, 11}.
    std::vector<TokenId> committed{1, 2, 3, 4, 5};
    MtpDecodeEgress egress;
    egress.licensed_counts[0] = 2;
    egress.licensed_tokens[0] = 11;
    egress.licensed_tokens[1] = 12;
    egress.next_extents[0]    = 2;
    egress.next_drafts[0]     = 21; // step-major, one row
    egress.next_drafts[1]     = 22;
    Qwen4PleEarlySource source;
    source.context         = kContext;
    source.draft_window    = 2;
    source.max_concurrency = 1;
    source.egress          = &egress;
    source.ledger          = [&](std::uint32_t) -> std::span<const TokenId> { return committed; };
    const std::vector<std::uint32_t> lanes{0};

    Qwen4PleRuntime ple;
    std::vector<std::vector<Qwen4PleStage>> runs; // the rows each stager run would gather
    bool fail_next = false;
    ple.stager     = std::make_unique<Qwen4PleStager>([&] {
        runs.push_back(ple.queue);
        if (fail_next) {
            fail_next = false;
            throw std::runtime_error("early gather failed");
        }
    });
    // One round: the early run after the previous round, the commit, the round's own rows.
    const auto round = [&](std::vector<TokenId> window) {
        committed = {1, 2, 3, 4, 5};
        ple.start_early(source, lanes);
        committed.push_back(11);
        ple.begin_round(true);
        ple.queue_rows(0, 6, window, 0);
        ple.resolve_early(source);
    };

    runs.clear();
    round({12, 21, 22});
    check(ple.round_prestaged, "matching rows: the early rows are used");
    ple.stager->finish(); // the round's wait
    check(runs.size() == 1 && runs[0].size() == 1 && runs[0][0].snapshot &&
              runs[0][0].frontier == 6 && runs[0][0].count == 3 && runs[0][0].tokens[0] == 12 &&
              runs[0][0].tokens[1] == 21 && runs[0][0].tokens[2] == 22 &&
              runs[0][0].context_count == 2 && runs[0][0].context[0] == 5 &&
              runs[0][0].context[1] == 11,
          "the early run gathers the next round's rows from the snapshot context");

    runs.clear();
    round({12, 21, 99}); // the round's last draft differs
    check(!ple.round_prestaged, "different rows: the early rows are not used");
    check(ple.queue.size() == 1 && !ple.queue[0].snapshot && ple.queue[0].tokens[2] == 99,
          "different rows: the round stages its own rows");
    check(runs.size() == 1, "different rows: the early run was settled before the round's rows");

    fail_next  = true;
    bool threw = false;
    try {
        round({12, 21, 99});
    } catch (...) { threw = true; }
    check(!threw && !ple.round_prestaged,
          "a discarded early run's exception does not reach the round");

    fail_next = true;
    round({12, 21, 22});
    threw = false;
    try {
        ple.stager->finish();
    } catch (const std::runtime_error&) { threw = true; }
    check(ple.round_prestaged && threw, "a used early run's exception reaches the round's wait");
    ple.round_prestaged = false;

    fail_next = true;
    ple.start_early(source, lanes);
    ple.drop_early(); // the Program's destructor: waits, never throws
    check(!ple.early_pending && !ple.collecting, "drop_early clears the early state");

    if (failures != 0) { return 1; }
    std::printf("qwen4 PLE early: OK\n");
    return 0;
}
