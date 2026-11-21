// DraftPolicy (core/draft_policy.h): when does a lookup window beat the MTP's? The live-cost tests
// of Strata's draft_policy_test.cpp (engine/src/spec), MIT License, Copyright (c) 2026 Niko1221 and
// the Strata contributors. Simulated rounds with costs of ~10 ms more per token check that
//   1. with no lookup proposal the MTP window is kept;
//   2. lookup drafts that are mostly rejected stop being taken (their bucket's rate falls);
//   3. lookup drafts that are always accepted are taken, and the window grows with them;
//   4. match-length buckets learn separately (short matches failing does not stop long ones);
//   5. the policy never proposes a window beyond its cap or outside its sizes.
// Drafter-window cases after Basalt's tests (MIT License, Copyright (c) 2026 Jesús Gulfo), for a
// drafter that states each draft's chance:
//   6. drafts that are always accepted fill the window, which grows past a size measured too dear;
//   7. middling chances stop the window where another row costs more than it returns;
//   8. a drafter that overstates its chances is calibrated down, and the window shrinks with it;
//   9. rows that cost little take more drafts at the same chances;
//  10. against a lookup proposal, a long reliable copy wins and a failing short one loses;
//  11. an untimed policy's picks do not depend on the round times.
#include "core/draft_policy.h"

#include <algorithm>
#include <cstdio>
#include <random>

using ninfer::spec::DraftPolicy;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  %-70s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) { ++g_fail; }
}

double cost(int t) { return 19.0 + 10.5 * (t - 1); } // ms per round, measured shape
double ds_cost(int t) { return 13.0 + 3.6 * (t - 1); } // a round: verify + commit + draft

// Rounds of a drafter whose drafts are each accepted with chance `q` (a fixed-seed draw), stating
// `stated`; returns the window the policy picks most often over the last half.
int run(DraftPolicy& p, float stated, float q, int rounds, double (*c)(int)) {
    const float ch[5] = {stated, stated, stated, stated, stated};
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(0.0F, 1.0F);
    int seen[DraftPolicy::kMaxT + 1] = {};
    for (int i = 0; i < rounds; ++i) {
        const int t = p.window(ch, 5);
        int a       = 0;
        while (a < t - 1 && u(rng) < q) { ++a; }
        p.observe_window(ch, t, a, c(t));
        if (2 * i >= rounds) { ++seen[t]; }
    }
    int best = 1;
    for (int t = 1; t <= DraftPolicy::kMaxT; ++t) {
        if (seen[t] > seen[best]) { best = t; }
    }
    return best;
}

void drafter_cases() {
    {
        DraftPolicy p(6);
        const float ch[5] = {0.999F, 0.999F, 0.999F, 0.999F, 0.999F};
        for (int i = 0; i < 20; ++i) { // a cold start: size 6 measured dear beside size 1
            p.observe_window(nullptr, 1, 0, ds_cost(1));
            p.observe_window(ch, 6, 5, 3.0 * ds_cost(6));
        }
        check(p.window(ch, 5) < 6, "drafter: a size measured dear is not taken at first");
        const int t = run(p, 0.999F, 1.0F, 800, ds_cost);
        check(t == 6, "drafter: accepted drafts fill the window once its cost is known");
        check(p.cost_ms(6) < 1.2 * ds_cost(6), "drafter: the dear size's cost is measured again");
    }
    {
        DraftPolicy p(6);
        const float ch[5] = {0.6F, 0.6F, 0.6F, 0.6F, 0.6F};
        const int t       = run(p, 0.6F, 0.6F, 300, ds_cost);
        check(t == 2 || t == 3, "drafter: middling chances stop at 2 or 3 rows");
        check(p.window(ch, 5) <= 3, "drafter: no deeper than the rows pay");
    }
    {
        DraftPolicy p(6);
        const int t0 = run(p, 0.95F, 0.95F, 200, ds_cost);
        DraftPolicy o(6);
        const int t1 = run(o, 0.95F, 0.45F, 300, ds_cost); // stated 0.95, accepted 0.45
        check(o.chance_rate(0.95F) < 0.7, "drafter: an overstated chance is calibrated down");
        check(t1 < t0, "drafter: the window shrinks with the calibrated chance");
    }
    {
        DraftPolicy p(6), q(6);
        const int t_dear  = run(p, 0.75F, 0.75F, 300, ds_cost);
        const int t_cheap = run(q, 0.75F, 0.75F, 300, [](int t) { return 13.0 + 0.8 * (t - 1); });
        check(t_cheap > t_dear, "drafter: cheap rows take more drafts at the same chances");
    }
    {
        DraftPolicy p(4);
        const float ch[5] = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
        check(p.window(ch, 2) <= 3, "drafter: never more rows than drafts + 1");
        check(p.window(ch, 5) <= 4, "drafter: never beyond the cap");
        check(p.window(ch, 0) == 1, "drafter: no drafts, one row");
    }
    {
        DraftPolicy p(8);
        p.set_sizes((1U << 2) | (1U << 4) | (1U << 8));
        const float ch[7] = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
        bool inside = true;
        for (int n = 0; n <= 7; ++n) { inside = inside && p.allowed(p.window(ch, n)); }
        check(inside, "drafter: never a size outside the mask");
    }
    {
        DraftPolicy p(DraftPolicy::kMaxT);
        const float ch[DraftPolicy::kMaxT - 1] = {0.9F, 0.9F, 0.9F, 0.9F, 0.9F, 0.9F, 0.9F};
        for (int i = 0; i < 200; ++i) {
            const int t = 1 + i % DraftPolicy::kMaxT;
            p.observe(true, t, t - 1, 40, 20.0 + 1.5 * (t - 1)); // long matches: all accepted
        }
        const DraftPolicy::Pick none = p.choose_window(ch, 7, 0, 0);
        check(!none.lookup && none.t == p.window(ch, 7),
              "drafter vs lookup: no proposal, the drafter's window");
        const DraftPolicy::Pick k = p.choose_window(ch, 7, 15, 40);
        check(k.lookup && k.t > 8, "drafter vs lookup: a long reliable copy takes a wide window");
        check(p.choose_window(ch, 7, 40, 40).t <= DraftPolicy::kMaxT,
              "drafter vs lookup: never beyond the cap");
        for (int i = 0; i < 200; ++i) { p.observe(true, 6, 0, 4, 20.0 + 1.5 * 5); }
        const float sure[7] = {0.99F, 0.99F, 0.99F, 0.99F, 0.99F, 0.99F, 0.99F};
        check(!p.choose_window(sure, 7, 5, 4).lookup,
              "drafter vs lookup: a failing short copy loses to sure drafts");
    }
    { // an untimed policy ignores round times
        DraftPolicy fast(8, 0.03, false), slow(8, 0.03, false);
        std::mt19937 rng(7);
        std::uniform_int_distribution<int> acc(0, 5);
        bool same = true;
        for (int i = 0; i < 300; ++i) {
            const int t = 2 + i % 6, a2 = std::min(acc(rng), t - 1), m = 3 + (i * 5) % 40;
            const bool lk = i % 3 == 0;
            fast.observe(lk, t, a2, m, 10.0 + 7.0 * t);
            slow.observe(lk, t, a2, m, 55.0 - 3.0 * t + (i % 11));
            const DraftPolicy::Pick x = fast.choose(t, 5 + i % 9, m), y = slow.choose(t, 5 + i % 9, m);
            same = same && x.lookup == y.lookup && x.t == y.t;
        }
        check(same, "untimed: the picks do not depend on the round times");
    }
}

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
    drafter_cases();
    std::printf("draft_policy: %s\n", g_fail != 0 ? "FAILED" : "PASS");
    return g_fail != 0 ? 1 : 0;
}
