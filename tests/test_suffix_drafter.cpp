// SuffixDrafter (core/suffix_drafter.h): the unit tests of Strata's suffix_drafter_test.cpp
// (engine/src/spec), MIT License, Copyright (c) 2026 Niko1221 and the Strata contributors.
#include "core/suffix_drafter.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using ninfer::spec::SuffixDrafter;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}

void basic() {
    { // a repeated passage is proposed verbatim
        SuffixDrafter d;
        const std::vector<std::int32_t> doc = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
        d.append(doc.data(), doc.size());
        const std::int32_t again[] = {10, 11, 12};
        d.append(again, 3);
        std::int32_t out[8];
        const int n = d.propose(5, out);
        check(n == 5 && out[0] == 13 && out[4] == 17, "repeat proposes the continuation");
        check(d.last_match() == 3, "match length is the shared suffix");
    }
    { // no earlier occurrence: nothing proposed
        SuffixDrafter d;
        const std::vector<std::int32_t> doc = {1, 2, 3, 4, 5, 6, 7};
        d.append(doc.data(), doc.size());
        std::int32_t out[4];
        check(d.propose(4, out) == 0, "no match proposes nothing");
    }
    { // two occurrences of the trigram: the longer match wins even though it is older
        SuffixDrafter d;
        const std::vector<std::int32_t> doc = {7, 8,   1,   2,  3, 100, 101, 9, 1, 2,
                                               3, 200, 201, 50, 7, 8,   1,   2, 3};
        d.append(doc.data(), doc.size());
        std::int32_t out[2];
        const int n = d.propose(2, out);
        check(n == 2 && out[0] == 100 && out[1] == 101, "longest match preferred over most recent");
        check(d.last_match() == 5, "longest match length 5");
    }
    { // a match shorter than min_match is ignored (only a bigram in common)
        SuffixDrafter d(4);
        const std::vector<std::int32_t> doc = {1, 2, 3, 9, 5, 2, 3};
        d.append(doc.data(), doc.size());
        std::int32_t out[2];
        check(d.propose(2, out) == 0, "min_match respected");
    }
    { // periodic text: the continuation may run into the current suffix
        SuffixDrafter d;
        const std::vector<std::int32_t> doc = {1, 2, 3, 1, 2, 3, 1, 2, 3};
        d.append(doc.data(), doc.size());
        std::int32_t out[6];
        const int n = d.propose(6, out);
        check(n >= 3 && out[0] == 1 && out[1] == 2 && out[2] == 3, "periodic continuation");
    }
    { // bounded memory: appending beyond the nominal capacity does not crash and still works
        SuffixDrafter d(3, 32, 1024);
        std::vector<std::int32_t> doc;
        for (int i = 0; i < 5000; ++i) { doc.push_back(i % 997); }
        d.append(doc.data(), doc.size());
        std::int32_t out[4];
        check(d.propose(4, out) > 0, "propose after overflow of nominal capacity");
    }
    { // the match continues from tokens appended after the history, which stays unchanged
        SuffixDrafter d(3, 32, 256);
        const std::int32_t hist[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 50, 51, 52, 1, 2};
        d.append(hist, sizeof(hist) / sizeof(hist[0]));
        const std::int32_t drafts[] = {3, 4};
        std::int32_t out[8]         = {};
        const int k                 = d.propose_after(drafts, 2, 3, out);
        check(k == 3 && out[0] == 5 && out[1] == 6 && out[2] == 7,
              "propose_after continues after the drafts");
        check(d.last_match() >= 4, "propose_after match length counts the drafts");
        check(d.size() == sizeof(hist) / sizeof(hist[0]),
              "propose_after leaves the history unchanged");
        const std::int32_t miss[] = {99, 98, 97};
        check(d.propose_after(miss, 3, 3, out) == 0,
              "propose_after without a match proposes nothing");
        check(d.propose(3, out) == 0, "propose after propose_after");
        check(d.propose_after(nullptr, 0, 3, out) == d.propose(3, out),
              "propose_after with no drafts == propose");
    }
    { // reset forgets the history
        SuffixDrafter d;
        const std::vector<std::int32_t> doc = {10, 11, 12, 13, 14, 10, 11, 12};
        d.append(doc.data(), doc.size());
        std::int32_t out[4];
        check(d.propose(2, out) == 2, "before reset: a match");
        d.reset();
        check(d.size() == 0 && d.propose(2, out) == 0 && d.last_match() == 0,
              "reset: nothing left");
    }
}

// Small vocabularies repeat a lot, tiny tables fill up, propose_after is interleaved; every
// proposal must be a real earlier continuation of the matched suffix.
void randomized() {
    std::uint64_t rng = 0x243F6A8885A308D3ULL;
    auto next         = [&rng] {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        return rng;
    };
    long proposals = 0, bad = 0;
    for (int round = 0; round < 300; ++round) {
        const int vocab       = 2 + static_cast<int>(next() % 6);
        const int min_m       = 3 + static_cast<int>(next() % 3);
        const std::size_t cap = 8 + static_cast<std::size_t>(next() % 200);
        SuffixDrafter d(min_m, 4 + static_cast<int>(next() % 40), cap);
        std::vector<std::int32_t> hist;
        for (int step = 0; step < 400; ++step) {
            const int burst = 1 + static_cast<int>(next() % 3);
            for (int b = 0; b < burst; ++b) {
                const std::int32_t t =
                    hist.size() > 8 && (next() % 3) == 0
                        ? hist[hist.size() - 1 - static_cast<std::size_t>(next() % 8)]
                        : static_cast<std::int32_t>(next() % static_cast<std::uint64_t>(vocab));
                hist.push_back(t);
                d.append(t);
            }
            const int max_k       = 1 + static_cast<int>(next() % 8);
            std::int32_t out[16]  = {};
            std::int32_t extra[3] = {};
            const int n_extra     = static_cast<int>(next() % 4);
            for (int e = 0; e < n_extra; ++e) {
                extra[e] = static_cast<std::int32_t>(next() % static_cast<std::uint64_t>(vocab));
            }
            const int k = n_extra > 0 && (next() % 2) != 0
                              ? d.propose_after(extra, n_extra, max_k, out)
                              : d.propose(max_k, out);
            if (k < 0 || k > max_k || d.size() != hist.size()) {
                ++bad;
                continue;
            }
            if (k == 0) { continue; }
            ++proposals;
            if (d.last_match() < min_m) {
                ++bad;
                continue;
            }
            bool found = false;
            for (std::size_t p = 0;
                 p + 1 < hist.size() + static_cast<std::size_t>(n_extra) && !found; ++p) {
                bool match = true;
                for (int t = 0; t < k && match; ++t) {
                    const std::size_t at = p + 1 + static_cast<std::size_t>(t);
                    const std::int32_t v =
                        at < hist.size() ? hist[at]
                                         : (at - hist.size() < static_cast<std::size_t>(n_extra)
                                                ? extra[at - hist.size()]
                                                : -1);
                    match = v == out[t];
                }
                found = match;
            }
            if (!found) { ++bad; }
        }
    }
    check(bad == 0, "randomized proposals are real continuations within bounds");
    check(proposals > 1000, "randomized test exercised proposals");
}

} // namespace

int main() {
    basic();
    randomized();
    std::printf("suffix_drafter: %s\n", g_fail != 0 ? "FAILED" : "PASS");
    return g_fail != 0 ? 1 : 0;
}
