// Ported from Strata (engine/src/spec/draft_policy.cpp, the live-cost core of DraftPolicy), MIT
// License, Copyright (c) 2026 Niko1221 and the Strata contributors. See draft_policy.h.
#include "core/draft_policy.h"

#include <algorithm>
#include <cstdio>

namespace ninfer::spec {
namespace {

// The shape of a round's cost by window size, relative to one token, used only for sizes not
// measured yet (the measured round times replace it). Strata measured it on an RTX 5070 with
// experts in host RAM (sizes 9-16 above the measured curves: +0.4 a token).
constexpr double kShape[DraftPolicy::kMaxT + 1] = {0.0, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6,
                                                   4.0, 4.4, 4.8,  5.2, 5.6,  6.0,  6.4,  6.8};
constexpr double kCostAlpha                     = 0.1;  // EMA weight of a new round time
constexpr double kTokAlpha                      = 0.05; // EMA weight of a new MTP window outcome
constexpr double kDecay                         = 0.97; // lookup counts: older windows fade
// Before a bucket has data: the longer the match, the likelier its continuation (llama.cpp's
// lookup decoding gates on the same thing); worth 4 observations, so a few real windows override
// it.
constexpr double kPriorQ[DraftPolicy::kBuckets] = {0.75, 0.88, 0.93, 0.96};
constexpr double kPriorN                        = 4.0;
constexpr int kProbes =
    3; // a lookup window size is tried this often before its guessed cost can veto it
// A size not measured for this many rounds is tried again when a confident lookup could use it: its
// cost was set in one stretch (a prompt's cache refill, say) and, priced high, it would never be
// measured again.
constexpr double kStaleRounds = 300.0;

} // namespace

DraftPolicy::DraftPolicy(int max_t, double margin)
    : max_t_(std::clamp(max_t, 1, kMaxT)), margin_(margin) {}

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

double DraftPolicy::lookup_rate(int match) const {
    const int b = bucket(match);
    return (ok_[b] + kPriorN * kPriorQ[b]) / (ok_[b] + bad_[b] + kPriorN);
}

double DraftPolicy::cost_ms(int t) const {
    t = std::clamp(t, 1, kMaxT);
    if (cost_n_[bk_][t] > 0) { return cost_[bk_][t]; }
    // Scaled from the measured sizes, each weighted by how often it was seen.
    double num = 0.0, den = 0.0;
    for (int u = 1; u <= kMaxT; ++u) {
        if (cost_n_[bk_][u] > 0) {
            const double w = std::min(cost_n_[bk_][u], 20.0);
            num += w * cost_[bk_][u] * kShape[t] / kShape[u];
            den += w;
        }
    }
    return den > 0 ? num / den : kShape[t];
}

double DraftPolicy::clamp_sample(int t, double ms) const {
    if (!robust_) { return ms; }
    const double known =
        cost_ms(t); // the measured value, or the scaled guess from the other widths
    if (cost_n_[bk_][t] == 0) {
        // The first use of a width captures its graph: that round says nothing about its cost, the
        // guess from the other widths stands in.
        for (int u = 1; u <= kMaxT; ++u) {
            if (cost_n_[bk_][u] > 0) { return known; }
        }
        return ms;
    }
    return std::min(ms, 2.0 * known);
}

void DraftPolicy::set_context(long long pos) {
    if (!robust_) { return; }
    const int nb = pos < 6000 ? 0 : pos < 24000 ? 1 : 2;
    if (nb == bk_) { return; }
    bool has = false;
    for (int t = 1; t <= kMaxT; ++t) { has = has || cost_n_[nb][t] > 0; }
    if (!has) { // nothing measured at this length yet: the nearest bucket with data is the prior
        for (int d = 1; d < kCtxBuckets && !has; ++d) {
            for (const int src : {nb - d, nb + d}) {
                if (src < 0 || src >= kCtxBuckets || has) { continue; }
                for (int t = 1; t <= kMaxT; ++t) {
                    if (cost_n_[src][t] > 0) {
                        cost_[nb][t]   = cost_[src][t];
                        cost_n_[nb][t] = 1.0;
                        last_[nb][t]   = rounds_;
                        has            = true;
                    }
                }
            }
        }
    }
    bk_ = nb;
}

std::string DraftPolicy::cost_table_text() const {
    std::string out;
    char b[64];
    for (int k = 0; k < kCtxBuckets; ++k) {
        bool any = false;
        for (int t = 1; t <= kMaxT; ++t) { any = any || cost_n_[k][t] > 0; }
        if (!any) { continue; }
        out += out.empty() ? "" : " | ";
        out += k == 0 ? "ctx<6K:" : k == 1 ? "ctx<24K:" : "ctx>=24K:";
        for (int t = 1; t <= kMaxT; ++t) {
            if (cost_n_[k][t] > 0) {
                std::snprintf(b, sizeof b, " T%d:%.2f(n=%.0f)", t, cost_[k][t], cost_n_[k][t]);
                out += b;
            }
        }
    }
    return out;
}

bool DraftPolicy::stale(int t) const {
    return cost_n_[bk_][t] > 0 && rounds_ - last_[bk_][t] > kStaleRounds;
}

double DraftPolicy::mtp_tokens(int t) const {
    if (mtp_n_[t] > 0) { return mtp_tok_[t]; }
    return 1.0 + 0.7 * (t - 1); // before any MTP window of this size: a typical acceptance
}

DraftPolicy::Pick DraftPolicy::choose(int t_mtp, int lookup_k, int match) const {
    Pick p;
    p.t = std::clamp(t_mtp, 1, max_t_);
    if (lookup_k <= 0) { return p; }
    const double base = mtp_tokens(p.t) / cost_ms(p.t);
    const double q    = lookup_rate(match);
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
    const int t_full = fit(std::min(lookup_k, max_t_ - 1) + 1);
    if (best_t > 0 && best > base * (1.0 + margin_)) {
        p.lookup = true;
        p.t      = best_t;
        // A wider size whose cost has gone stale is measured again, so one slow stretch cannot
        // price it out for good.
        if (best_t < t_full && q >= 0.85 && stale(t_full)) { p.t = t_full; }
        return p;
    }
    // A guessed cost can keep the policy from ever measuring a size: the first few times a
    // confident lookup would need a size not measured yet, it is tried (verification keeps the
    // output; only the one round's speed is at stake).
    if (t_full > p.t && t_full > 1 && (cost_n_[bk_][t_full] < kProbes || stale(t_full)) &&
        q >= 0.85) {
        p.lookup = true;
        p.t      = t_full;
    }
    return p;
}

void DraftPolicy::observe(bool lookup, int t, int accepted, int match, double round_ms) {
    t = std::clamp(t, 1, kMaxT);
    if (round_ms > 0) {
        round_ms = clamp_sample(t, round_ms);
        // Robust mode: a running mean for the first samples.
        const double alpha =
            robust_ ? std::max(kCostAlpha, 1.0 / (cost_n_[bk_][t] + 1.0)) : kCostAlpha;
        cost_[bk_][t] =
            cost_n_[bk_][t] > 0 ? (1.0 - alpha) * cost_[bk_][t] + alpha * round_ms : round_ms;
        cost_n_[bk_][t] += 1.0;
        last_[bk_][t] = rounds_;
    }
    rounds_ += 1.0;
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

} // namespace ninfer::spec
