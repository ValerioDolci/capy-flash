// DraftPolicy (core/draft_policy.h): when does a lookup window beat the MTP's? The live-cost tests
// of Strata's draft_policy_test.cpp (engine/src/spec), MIT License, Copyright (c) 2026 Niko1221 and
// the Strata contributors. Simulated rounds with costs of ~10 ms more per token check that
//   1. with no lookup proposal the MTP window is kept;
//   2. lookup drafts that are mostly rejected stop being taken (their bucket's rate falls);
//   3. lookup drafts that are always accepted are taken, and the window grows with them;
//   4. match-length buckets learn separately (short matches failing does not stop long ones);
//   5. the policy never proposes a window beyond its cap or outside its sizes.
#include "core/draft_policy.h"

#include <cstdio>

using ninfer::spec::DraftPolicy;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  %-70s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) { ++g_fail; }
}

double cost(int t) { return 19.0 + 10.5 * (t - 1); } // ms per round, measured shape

} // namespace

int main() {
    std::printf("draft_policy_test\n");
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) { p.observe(false, 4, 2, 0, cost(4)); }
        const DraftPolicy::Pick k = p.choose(4, 0, 0);
        check(!k.lookup && k.t == 4, "no proposal: the MTP window");
    }
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) { p.observe(false, 4, 2, 0, cost(4)); }
        for (int t = 2; t <= 6; ++t) { p.observe(false, t, 0, 0, cost(t)); }
        for (int i = 0; i < 40; ++i) {
            p.observe(true, 6, 0, 4, cost(6));
        } // short matches, rejected
        check(p.lookup_rate(4) < 0.15, "rejected short-match drafts: their rate falls below 0.15");
        check(!p.choose(4, 5, 4).lookup, "rejected short-match drafts: no longer taken");
        for (int i = 0; i < 40; ++i) {
            p.observe(true, 6, 5, 30, cost(6));
        } // long matches, accepted
        check(p.lookup_rate(30) > 0.9, "accepted long-match drafts: their rate rises above 0.9");
        const DraftPolicy::Pick k = p.choose(4, 5, 30);
        check(k.lookup && k.t == 6, "accepted long matches: the full lookup window is taken");
        check(!p.choose(4, 5, 4).lookup, "buckets are separate: short matches still not taken");
        check(p.choose(4, 20, 30).t <= 6, "never beyond the window cap");
    }
    {
        DraftPolicy p(8);
        for (int i = 0; i < 50; ++i) { p.observe(false, 3, 2, 0, cost(3)); } // a very good MTP
        for (int t = 2; t <= 8; ++t) { p.observe(false, t, t - 1, 0, cost(t)); }
        for (int i = 0; i < 40; ++i) { p.observe(true, 4, 2, 8, cost(4)); } // lookup at q ~ 0.67
        check(!p.choose(3, 7, 8).lookup, "a mediocre lookup does not replace a strong MTP window");
    }
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) { p.observe(false, 4, 3, 0, cost(4)); } // only size 4 seen
        const DraftPolicy::Pick k = p.choose(4, 5, 40);
        check(k.lookup && k.t == 6, "an unmeasured size is probed for a confident lookup");
        for (int i = 0; i < 3; ++i) { p.observe(true, 6, 5, 40, 3.0 * cost(6)); } // very expensive
        check(!p.choose(4, 5, 40).lookup, "after the probes, the measured cost decides");
    }
    {
        // A wide size priced high in one slow stretch is measured again after a while.
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) { p.observe(false, 4, 3, 0, cost(4)); }
        for (int t = 2; t <= 6; ++t) {
            for (int i = 0; i < 4; ++i) { p.observe(false, t, t - 1, 0, cost(t)); }
        }
        for (int i = 0; i < 6; ++i) { p.observe(true, 6, 5, 40, 4.0 * cost(6)); }
        for (int i = 0; i < 30; ++i) { p.observe(true, 4, 3, 40, cost(4)); }
        bool saw6 = false;
        for (int i = 0; i < 400 && !saw6; ++i) {
            const DraftPolicy::Pick k = p.choose(4, 5, 40);
            if (k.lookup && k.t == 6) {
                saw6 = true;
                p.observe(true, 6, 5, 40, cost(6));
            } else {
                p.observe(k.lookup, k.t, k.t - 1, 40, cost(k.t));
            }
        }
        check(saw6, "a stale wide size is tried again within a few hundred rounds");
    }
    {
        // Only the allowed sizes are taken; a proposal between two sizes takes the smaller; a
        // disabled size is never proposed again; sizes past 8 work like the others.
        DraftPolicy p(16);
        p.set_sizes((1U << 1) | (1U << 2) | (1U << 3) | (1U << 5) | (1U << 8) | (1U << 16));
        for (int i = 0; i < 50; ++i) { p.observe(false, 3, 2, 0, cost(3)); }
        for (int i = 0; i < 40; ++i) { p.observe(true, 16, 15, 40, 19.0 + 3.0 * 15); }
        for (int i = 0; i < 40; ++i) { p.observe(true, 8, 7, 40, 19.0 + 3.0 * 7); }
        for (int i = 0; i < 40; ++i) { p.observe(true, 5, 4, 40, 19.0 + 3.0 * 4); }
        const DraftPolicy::Pick a = p.choose(3, 15, 40);
        check(a.lookup && a.t == 16, "sizes: a full 16-token lookup window when it pays");
        const DraftPolicy::Pick b = p.choose(3, 10, 40);
        check(b.lookup && b.t == 8, "sizes: a 10-token proposal takes the 8-token size");
        bool ok = true;
        for (int k = 1; k <= 15; ++k) {
            const DraftPolicy::Pick c = p.choose(3, k, 40);
            ok                        = ok && (!c.lookup || p.allowed(c.t));
        }
        check(ok, "sizes: never a size outside the mask");
        p.disable(16);
        const DraftPolicy::Pick d = p.choose(3, 15, 40);
        check(d.lookup && d.t == 8, "sizes: a disabled size falls back to the next one");
        check(p.fit(7) == 5 && p.fit(16) == 8 && p.fit(1) == 1,
              "sizes: fit() finds the largest allowed size");
    }
    {
        // Robust mode: the first round of a new width (a graph capture) does not price it; context
        // buckets start from the nearest measured one.
        DraftPolicy p(8);
        p.set_robust(true);
        for (int i = 0; i < 20; ++i) { p.observe(false, 4, 2, 0, cost(4)); }
        p.observe(true, 8, 7, 40, 50.0 * cost(8)); // the capture round
        check(p.cost_ms(8) < 2.0 * cost(8), "robust: a capture round does not price a width out");
        p.set_context(30000);
        check(p.cost_ms(4) > 0.99 * cost(4) && p.cost_ms(4) < 1.01 * cost(4),
              "robust: a new context bucket starts from the nearest measured one");
    }
    std::printf("draft_policy: %s\n", g_fail != 0 ? "FAILED" : "PASS");
    return g_fail != 0 ? 1 : 0;
}
