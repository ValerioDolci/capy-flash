// Window choice inspired by Basalt's DraftPolicy (MIT License, Copyright (c) 2026 Jesús Gulfo),
// after Strata's (MIT License, Copyright (c) 2026 Niko1221 and the Strata contributors). See
// draft_policy.h.
#include "core/draft_policy.h"

#include <algorithm>
#include <cstdio>

namespace ninfer::spec {
namespace {

// The shape of a round's cost by window size, relative to one token, used only below the smallest
// measured size or while a single size is measured (two measured sizes give the cost of a row;
// measured round times replace both). It sits between the measured curves: steeper with every
// missed expert on the CPU, flatter when the experts are resident.
constexpr double kShape[DraftPolicy::kMaxT + 1] = {
    0.0, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6, 3.95, 4.3, 4.65, 5.0, 5.35, 5.7, 6.05, 6.4};
// A round's cost is scale x base(size): the scale follows every round fast, each size's base
// follows its own rounds slowly.
constexpr double kScaleAlpha = 0.2;
constexpr double kCostAlpha  = 0.1; // EMA weight of a size's new base
// A size's first rounds start from the base the other sizes imply, worth this many rounds: one
// cold round (a request's first windows run while the expert cache settles) cannot rule a size out.
constexpr double kBasePrior = 3.0;
// Rounds after which a size's base is from another time: the sizes measured since stand in for it
// until it is measured again, so a size measured dear once is not ruled out for good.
constexpr long kStale      = 64;
constexpr double kTokAlpha = 0.05; // EMA weight of a new MTP window outcome
constexpr double kDecay    = 0.97; // lookup counts: older windows fade
// Before a bucket has data: the longer the match, the likelier its continuation; worth 4
// observations, so a few real windows override it.
constexpr double kPriorQ[DraftPolicy::kBuckets] = {0.75, 0.88, 0.93, 0.96};
constexpr double kPriorN                        = 4.0;
constexpr int kProbes = 3; // a size is tried this often before its guessed cost can veto it
// Stated chances: the bins' upper edges, finest near 1, where long windows are decided. Each bin
// starts from the stated chance, worth kChancePriorN tested drafts.
constexpr float kChanceEdge[]  = {0.3F, 0.5F, 0.7F, 0.8F, 0.9F, 0.95F, 0.98F, 0.995F};
constexpr double kChanceDecay  = 0.98;
constexpr double kChancePriorN = 10.0;
// The joint chance at which a window one row past the best is probed.
constexpr double kProbeChance = 0.85;
// Robust mode: a round counts at most this times the known cost.
constexpr double kSlowRound = 1.3;
// Drafting more: the EMA weights of its cost and of its balance, and how often it is still tried
// while it does not pay.
constexpr double kExtraAlpha = 0.1;
constexpr double kGainAlpha  = 0.05;
constexpr double kRateAlpha  = 0.02;
constexpr long kMoreProbe    = 16;  // the first probe interval; it doubles after each probe
constexpr long kMoreProbeMax = 512; // while the balance stays below zero, up to this
// Hysteresis: drafting more opens at a balance of kMoreOpen tokens and closes below zero, so a
// balance about zero does not flip it every few rounds.
constexpr double kMoreOpen = 0.1;
// Draft positions (a drafter without stated chances): a position's drafts are tested only when the
// ones before it were accepted, so its own counts can be old. Its rate is its counts shrunk toward
// the rate of all drafts tested lately (kPosShrink tested drafts' worth), which follows every round
// fast; its own counts fade every round, so after a change of text (code to prose) the positions
// not reached since fall back to it. Before any round the pooled rate is kPosPrior.
constexpr double kPosPrior      = 0.7;
constexpr double kPosPriorN     = 2.0;
constexpr double kPosShrink     = 3.0;
constexpr double kPosRoundDecay = 0.9; // a position's own counts, per round
constexpr double kPoolDecay     = 0.7; // all drafts tested, per round

} // namespace

DraftPolicy::DraftPolicy(int max_t, double margin, bool timed)
    : max_t_(std::clamp(max_t, 1, kMaxT)), margin_(margin), timed_(timed) {}

void DraftPolicy::set_sizes(unsigned mask) { sizes_ = mask | 2U; }

void DraftPolicy::disable(int t) {
    if (t > 1 && t <= kMaxT) { sizes_ &= ~(1U << t); }
}

int DraftPolicy::fit(int t) const {
    for (t = std::min(t, max_t_); t > 1; --t) {
        if (allowed(t)) { return t; }
    }
    return 1;
}

int DraftPolicy::bucket(int match) { return match < 6 ? 0 : match < 12 ? 1 : match < 24 ? 2 : 3; }

int DraftPolicy::chance_bin(float chance) {
    int b = 0;
    while (b < kChanceBins - 1 && chance >= kChanceEdge[b]) { ++b; }
    return b;
}

double DraftPolicy::chance_rate(float chance) const {
    const int b    = chance_bin(chance);
    const double c = std::clamp(static_cast<double>(chance), 0.0, 1.0);
    return (c_ok_[b] + kChancePriorN * c) / (c_n_[b] + kChancePriorN);
}

double DraftPolicy::lookup_rate(int match) const {
    const int b = bucket(match);
    return (ok_[b] + kPriorN * kPriorQ[b]) / (ok_[b] + bad_[b] + kPriorN);
}

bool DraftPolicy::recent(int t) const { return base_n_[t] > 0 && round_ - seen_[t] <= kStale; }

bool DraftPolicy::unmeasured(int t) const {
    if (plain_) { return ms_n_[t] < kProbes || round_ - seen_[t] > kStale; }
    return base_n_[t] < kBasePrior + kProbes || !recent(t);
}

double DraftPolicy::base(int t) const {
    if (recent(t)) { return base_[t]; }
    for (const bool lately : {true, false}) {
        const double b = base_from(t, lately);
        if (b > 0) { return b; }
    }
    return kShape[t];
}

double DraftPolicy::base_from(int t, bool lately) const {
    // The measured sizes next to t: between two, the line through them; above the largest, its line
    // on from the two largest (the cost of a row on this machine); below the smallest, or with one,
    // the prior shape scales it.
    int lo = 0, hi = 0, below = 0;
    for (int u = 1; u <= kMaxT; ++u) {
        if (u == t || base_n_[u] <= 0 || (lately && !recent(u))) { continue; }
        if (u < t) {
            below = lo;
            lo    = u;
        } else if (hi == 0) {
            hi = u;
        }
    }
    if (lo > 0 && hi > 0) {
        return base_[lo] + (base_[hi] - base_[lo]) * static_cast<double>(t - lo) / (hi - lo);
    }
    if (lo > 0 && below > 0) {
        return base_[lo] + std::max(0.0, (base_[lo] - base_[below]) / (lo - below)) * (t - lo);
    }
    if (lo > 0) { return base_[lo] * kShape[t] / kShape[lo]; }
    if (hi > 0) { return base_[hi] * kShape[t] / kShape[hi]; }
    return -1.0;
}

double DraftPolicy::cost_ms(int t) const {
    t = std::clamp(t, 1, kMaxT);
    if (plain_) { return plain_cost(t); }
    return (scale_ > 0 ? scale_ : 1.0) * base(t); // before any round: the prior shape
}

double DraftPolicy::plain_cost(int t) const {
    if (ms_n_[t] > 0) { return ms_[t]; }
    // Between two measured sizes, the line through them; above the largest, the line on from the
    // two largest (never falling); below the smallest, the prior shape scales it, which prices a
    // smaller window low until it is measured (the round cost is not a line: on the 27B it steps
    // between 4 and 7 columns, so a line from the wider sizes would price the narrow ones too high
    // ever to be tried).
    int lo = 0, hi = 0, below = 0;
    for (int u = 1; u <= kMaxT; ++u) {
        if (ms_n_[u] <= 0) { continue; }
        if (u < t) {
            below = lo;
            lo    = u;
        } else if (hi == 0) {
            hi = u;
        }
    }
    if (lo > 0 && hi > 0) { return ms_[lo] + (ms_[hi] - ms_[lo]) * (t - lo) / double(hi - lo); }
    if (lo > 0 && below > 0) {
        return ms_[lo] + std::max(0.0, (ms_[lo] - ms_[below]) / (lo - below)) * (t - lo);
    }
    if (lo > 0) { return ms_[lo] * kShape[t] / kShape[lo]; }
    if (hi > 0) { return ms_[hi] * kShape[t] / kShape[hi]; }
    return kShape[t];
}

double DraftPolicy::mtp_tokens(int t) const {
    if (mtp_n_[t] > 0) { return mtp_tok_[t]; }
    return 1.0 + 0.7 * (t - 1); // before any MTP window of this size: a typical acceptance
}

DraftPolicy::Pick DraftPolicy::choose(int t_mtp, int lookup_k, int match) const {
    const int t = fit(std::clamp(t_mtp, 1, max_t_));
    if (lookup_k <= 0) {
        Pick p;
        p.t = t;
        return p;
    }
    return lookup_over(t, mtp_tokens(t) / cost_ms(t), lookup_k, match);
}

DraftPolicy::Pick DraftPolicy::choose_window(const float* chance, int n, int lookup_k, int match,
                                             bool more) const {
    Pick p;
    p.t = window(chance, n, more);
    if (lookup_k <= 0) { return p; }
    return lookup_over(p.t, drafter_tokens(chance, p.t) / cost_ms(p.t), lookup_k, match);
}

double DraftPolicy::drafter_tokens(const float* chance, int t) const {
    double e = 1.0, all = 1.0;
    for (int i = 0; i + 1 < t; ++i) {
        all *= chance_rate(chance[i]);
        e += all;
    }
    return e;
}

DraftPolicy::Pick DraftPolicy::lookup_over(int t, double rate, int lookup_k, int match) const {
    Pick p;
    p.t            = t;
    const double q = lookup_rate(match);
    double e = 1.0, qi = 1.0, best = 0.0;
    int best_t = 0;
    for (int k = 1; k <= std::min(lookup_k, max_t_ - 1); ++k) {
        qi *= q;
        e += qi;
        if (!allowed(k + 1)) { continue; }
        const double r = e / cost_ms(k + 1);
        if (r > best) {
            best   = r;
            best_t = k + 1;
        }
    }
    if (best_t > 0 && best > rate * (1.0 + margin_)) {
        p.lookup = true;
        p.t      = best_t;
        return p;
    }
    // A guessed cost can keep the policy from ever measuring a size: the first few times a
    // confident lookup would need a size not measured yet (or not for long), it is tried
    // (verification keeps the output; only the one round's speed is at stake).
    const int t_full = fit(std::min(lookup_k, max_t_ - 1) + 1);
    if (t_full > p.t && unmeasured(t_full) && q >= 0.85) {
        p.lookup = true;
        p.t      = t_full;
    }
    return p;
}

int DraftPolicy::window(const float* chance, int n, bool more) const {
    const int most = std::clamp(n + 1, 1, max_t_);
    // A window past the base one drafts more for the next round, and its measured cost has that in.
    const double extra = more && base_t_ > 0 ? more_cost_ms() : 0.0;
    const auto cost    = [&](int t) { return cost_ms(t) + (t <= base_t_ ? extra : 0.0); };
    double e = 1.0, all = 1.0, best = 1.0 / cost(1);
    int best_t = 1;
    for (int t = 2; t <= most; ++t) {
        all *= chance_rate(chance[t - 2]);
        e += all;
        if (!allowed(t)) { continue; }
        const double r = e / cost(t);
        if (r > best) {
            best   = r;
            best_t = t;
        }
    }
    // As for lookup windows: a size whose cost is only guessed (or old) is tried a few times when
    // its drafts are likely accepted, so a guessed or stale cost cannot keep the window from
    // growing.
    int next = best_t + 1;
    while (next <= most && !allowed(next)) { ++next; }
    if (next <= most && unmeasured(next)) {
        double joint = 1.0;
        for (int i = 0; i + 1 < next; ++i) { joint *= chance_rate(chance[i]); }
        if (joint >= kProbeChance) { best_t = next; }
    }
    return best_t;
}

double DraftPolicy::pooled_rate() const {
    return (pool_ok_ + kPosPriorN * kPosPrior) / (pool_n_ + kPosPriorN);
}

double DraftPolicy::position_rate(int i) const {
    i = std::clamp(i, 0, kMaxT - 1);
    return (p_ok_[i] + kPosShrink * pooled_rate()) / (p_n_[i] + kPosShrink);
}

int DraftPolicy::position_window(int n, int widen_from) const {
    n = std::clamp(n, 0, max_t_ - 1);
    std::array<float, kMaxT> chance{};
    for (int i = 0; i < n; ++i) { chance[i] = static_cast<float>(position_rate(i)); }
    // No stated chances: the chance bins stay empty, so window() takes these rates as they are.
    int t = window(chance.data(), n);
    if (widen_from > 0) {
        int next = widen_from + 1;
        while (next <= n + 1 && !allowed(next)) { ++next; }
        if (next <= n + 1) { t = std::max(t, next); }
    }
    return t;
}

void DraftPolicy::observe_positions(int t, int accepted, double round_ms) {
    // Drafts 0..accepted-1 were accepted given their prefix and draft `accepted` (if the window had
    // it) was not; the drafts after it were never tested.
    const int tested = std::min(std::min(t - 1, accepted + 1), kMaxT);
    for (int i = 0; i < kMaxT; ++i) {
        p_ok_[i] *= kPosRoundDecay;
        p_n_[i] *= kPosRoundDecay;
    }
    pool_ok_ *= kPoolDecay;
    pool_n_ *= kPoolDecay;
    for (int i = 0; i < tested; ++i) {
        const double ok = i < accepted ? 1.0 : 0.0;
        p_ok_[i] += ok;
        p_n_[i] += 1.0;
        pool_ok_ += ok;
        pool_n_ += 1.0;
    }
    observe(false, t, accepted, 0, round_ms);
}

bool DraftPolicy::draft_more() {
    if (base_t_ <= 0) { return true; }
    if (more_open_ ? gain_ < 0 : gain_ >= kMoreOpen) { more_open_ = !more_open_; }
    if (more_open_) {
        probe_every_ = kMoreProbe;
        more_tries_  = 0;
        return true;
    }
    if (++more_tries_ < probe_every_) { return false; }
    more_tries_  = 0;
    probe_every_ = std::min(2 * probe_every_, kMoreProbeMax);
    return true;
}

void DraftPolicy::observe_window(const float* chance, int t, int accepted, double round_ms,
                                 bool more, int avail) {
    // Drafts 0..accepted-1 were accepted given their prefix and draft `accepted` (if the window had
    // it) was not; the drafts after it were never tested.
    const int tested = std::min(t - 1, accepted + 1);
    for (int i = 0; i < tested; ++i) {
        const int b = chance_bin(chance[i]);
        c_ok_[b]    = kChanceDecay * c_ok_[b] + (i < accepted ? 1.0 : 0.0);
        c_n_[b]     = kChanceDecay * c_n_[b] + 1.0;
    }
    if (base_t_ > 0 && timed_ && round_ms > 0) {
        const double got = accepted + 1.0;
        if (scale_ > 0) {
            const double ms = std::min(round_ms, kSlowRound * cost_ms(t) + more_cost_ms());
            rate_           = rate_ > 0 ? rate_ + kRateAlpha * (got / ms - rate_) : got / ms;
        }
        // What the more drafts brought: tokens past the base window's, less their time (drafting
        // them in the round before, or for a window past the base one its cost past the base
        // window's, which has the drafting of the next round's in it) at the current rate.
        if (avail > base_t_ - 1 && extra_n_ > 0 && rate_ > 0) {
            const double tokens = got - std::min(got, static_cast<double>(base_t_));
            const double ms =
                t > base_t_ ? std::max(0.0, cost_ms(t) - cost_ms(base_t_)) : more_cost_ms();
            gain_ += kGainAlpha * ((tokens - rate_ * ms) - gain_);
        }
        if (more && t <= base_t_) {
            // Not the size's cost: the time past it is the cost of drafting more.
            ++round_;
            if (scale_ > 0 && base_n_[t] > 0) {
                const double over = std::clamp(round_ms - cost_ms(t), 0.0, cost_ms(t));
                extra_ms_ = extra_n_ > 0 ? extra_ms_ + kExtraAlpha * (over - extra_ms_) : over;
                extra_n_ += 1.0;
            }
            const double g = accepted + 1.0;
            mtp_tok_[t]    = mtp_n_[t] > 0 ? (1.0 - kTokAlpha) * mtp_tok_[t] + kTokAlpha * g : g;
            mtp_n_[t] += 1.0;
            return;
        }
    }
    observe(false, t, accepted, 0, round_ms);
}

double DraftPolicy::clamp_sample(int t, double ms) const {
    if (!robust_ || scale_ <= 0) { return ms; }
    const double known = cost_ms(t);
    // The first round of a size captures its graph: it says nothing about its cost, the base the
    // other sizes imply stands in. A slow round (a stall, a host thread descheduled, an expert
    // refill) counts at most 1.3 times the known cost: a size's cost is a typical round's, and one
    // slow round must not price a wider window out of use (measured: the width-4 base stood 9 %
    // above its rounds' median and the policy cut conversation windows to 3). A lasting rise still
    // gets through, 30 % a round.
    if (base_n_[t] <= 0) { return known; }
    return std::min(ms, kSlowRound * known);
}

void DraftPolicy::observe(bool lookup, int t, int accepted, int match, double round_ms) {
    t = std::clamp(t, 1, kMaxT);
    ++round_;
    if (!timed_) { round_ms = kShape[t]; }
    if (round_ms > 0 && plain_) {
        // The first round of a size captures its graph: skipped. A slow round counts at most
        // kSlowRound times the size's cost.
        if (ms_n_[t] <= 0 && robust_ && seen_[t] == 0) {
            seen_[t] = round_;
        } else {
            const double ms =
                ms_n_[t] > 0 && robust_ ? std::min(round_ms, kSlowRound * ms_[t]) : round_ms;
            const double w = std::max(kCostAlpha, 1.0 / (ms_n_[t] + 1.0));
            ms_[t]         = ms_n_[t] > 0 ? (1.0 - w) * ms_[t] + w * ms : ms;
            ms_n_[t] += 1.0;
            seen_[t] = round_;
        }
        round_ms = 0; // the scaled costs below are not used
    }
    if (round_ms > 0) {
        round_ms = clamp_sample(t, round_ms);
        if (scale_ <= 0) { scale_ = round_ms / kShape[t]; } // the first round sets the scale
        if (!recent(t)) { // new, or measured too long ago: start from what the other sizes imply
            base_[t]   = base(t);
            base_n_[t] = kBasePrior;
        }
        scale_ += kScaleAlpha * (round_ms / base_[t] - scale_);
        const double w = std::max(kCostAlpha, 1.0 / (base_n_[t] + 1.0));
        base_[t]       = (1.0 - w) * base_[t] + w * round_ms / scale_;
        base_n_[t] += 1.0;
        seen_[t] = round_;
    }
    if (lookup) {
        const int b = bucket(match);
        ok_[b]      = kDecay * ok_[b] + accepted;
        bad_[b]     = kDecay * bad_[b] + (accepted < t - 1 ? 1.0 : 0.0);
    } else {
        const double got = accepted + 1.0;
        mtp_tok_[t]      = mtp_n_[t] > 0 ? (1.0 - kTokAlpha) * mtp_tok_[t] + kTokAlpha * got : got;
        mtp_n_[t] += 1.0;
    }
}

std::string DraftPolicy::cost_table_text() const {
    std::string out;
    char b[128];
    std::snprintf(b, sizeof b, "scale %.3f ms:", scale_);
    out += b;
    if (base_t_ > 0) {
        std::snprintf(b, sizeof b, " more %.2f ms(n=%.0f) balance %.3f rate %.3f;", more_cost_ms(),
                      extra_n_, gain_, rate_);
        out += b;
    }
    for (int t = 1; t <= kMaxT; ++t) {
        if (plain_ && ms_n_[t] > 0) {
            std::snprintf(b, sizeof b, " T%d:%.2f(n=%.0f)", t, ms_[t], ms_n_[t]);
            out += b;
        } else if (!plain_ && base_n_[t] > 0) {
            std::snprintf(b, sizeof b, " T%d:%.2f(n=%.0f%s)", t, cost_ms(t), base_n_[t],
                          recent(t) ? "" : ",old");
            out += b;
        }
    }
    return out;
}

} // namespace ninfer::spec
