// Per verify round: how many drafts to verify, and from which drafter (the MTP layer or a lookup,
// i.e. suffix drafter, proposal)?
//
// Window choice inspired by Basalt's DraftPolicy (MIT License, Copyright (c) 2026 Jesús Gulfo),
// itself after Strata's (MIT License, Copyright (c) 2026 Niko1221 and the Strata contributors).
//
// The policy compares expected committed tokens per millisecond:
//
//   drafter E(t) = 1 + q(c_0) + q(c_0) q(c_1) + ...  (t - 1 terms), c_i = the chance the drafter
//           stated for draft i (the probability of the drafted token), q = the acceptance rate of
//           drafts stated at about that chance (decayed counts per chance bin, the stated chance as
//           the prior)
//   MTP without stated chances: E = the mean tokens a window of that size has committed (EMA)
//   lookup  E(k) = 1 + q + q^2 + ... + q^k for k <= the proposal, q = the acceptance rate of lookup
//           drafts whose match was about as long (4 buckets of match length, decayed counts)
//
// each at the measured cost of its window size, and takes the window with the most tokens per ms;
// a lookup window only when it beats the drafter's best by `margin`. A round's cost is scale x
// base(size): the scale follows every round fast (the context grows, the expert cache moves, a
// request's first windows run cold), each size's base follows its own rounds slowly, so the costs
// of sizes not taken lately move with the ones taken. On a Flash-Next machine a wider window costs
// what its extra rows cost there (more distinct experts on the CPU), so a window grows only where
// its drafts pay.
//
// It only chooses which drafts to verify; verification keeps the output. A different window is a
// different verifier batch, so two runs whose measured round times differ can choose differently.
#pragma once

#include <array>
#include <string>

namespace ninfer::spec {

class DraftPolicy {
public:
    static constexpr int kMaxT    = 16;
    static constexpr int kBuckets = 4;

    // `timed` false: round times are not read, every round costs the prior shape of its size, so
    // the picks follow the drafts' outcomes alone (the same tokens give the same windows every run).
    explicit DraftPolicy(int max_t, double margin = 0.03, bool timed = true);

    struct Pick {
        bool lookup = false;
        int t       = 1; // window size (1 + drafts)
    };

    // MTP without stated chances. `t_mtp`: the MTP's window; `lookup_k`: the lookup proposal's
    // length (0 = none); `match`: its match length.
    [[nodiscard]] Pick choose(int t_mtp, int lookup_k, int match) const;
    // After such a round: the window it used, the drafts accepted, and the round's time (verify +
    // commit + draft).
    void observe(bool lookup, int t, int accepted, int match, double round_ms);

    // A drafter with stated chances: the window (1 + up to `n` drafts) with the most expected tokens
    // per ms.
    [[nodiscard]] int window(const float* chance, int n) const;
    // The drafter's window against a lookup proposal of `lookup_k` tokens behind a match of
    // `match`: the lookup window when its expected tokens per ms beat the drafter's best by the
    // margin.
    [[nodiscard]] Pick choose_window(const float* chance, int n, int lookup_k, int match) const;
    // After a drafter window: the chances of its t - 1 drafts, the drafts accepted, and the round's
    // time.
    void observe_window(const float* chance, int t, int accepted, double round_ms);

    [[nodiscard]] double lookup_rate(int match) const;   // current q for a match length
    [[nodiscard]] double chance_rate(float chance) const; // current q for a draft stated at `chance`
    [[nodiscard]] double cost_ms(int t) const; // measured or inferred round time of a window of t

    // Kept for callers of the earlier port: the scale follows the context, so the position needs
    // no table of its own.
    void set_context(long long /*pos*/) {}

    // Robust mode: the first round of a size (it captures that size's graph) and a round far above
    // what is known for that size (a stalled round) are clamped, so one capture cannot price a size
    // out of use.
    void set_robust(bool on) { robust_ = on; }

    // The measured round cost of every window size seen, for the log.
    [[nodiscard]] std::string cost_table_text() const;

    // The window sizes the policy may choose, as a bit mask (bit t = size t; bit 1 always kept).
    void set_sizes(unsigned mask);
    void disable(int t); // drops one size (its window graph did not fit)

    [[nodiscard]] bool allowed(int t) const {
        return t >= 1 && t <= max_t_ && ((sizes_ >> t) & 1U) != 0;
    }

    [[nodiscard]] int fit(int t) const; // the largest allowed size <= t (1 if none)

    [[nodiscard]] int max_t() const { return max_t_; }

    [[nodiscard]] unsigned sizes() const { return sizes_; }

private:
    static constexpr int kChanceBins = 9;
    static int bucket(int match);
    static int chance_bin(float chance);
    [[nodiscard]] bool unmeasured(int t) const; // measured too few times, or too long ago, to trust
    [[nodiscard]] bool recent(int t) const;  // measured within the last rounds
    [[nodiscard]] double base(int t) const;  // a size's cost relative to the scale, measured or inferred
    [[nodiscard]] double base_from(int t, bool lately) const; // t's base from the other sizes; < 0 without
    [[nodiscard]] double mtp_tokens(int t) const;
    [[nodiscard]] Pick lookup_over(int t, double rate, int lookup_k, int match) const;
    [[nodiscard]] double drafter_tokens(const float* chance, int t) const;
    [[nodiscard]] double clamp_sample(int t, double ms) const;

    int max_t_;
    double margin_;
    bool timed_;
    bool robust_   = false;
    unsigned sizes_ = ~0U;                             // allowed window sizes (bit t)
    double scale_  = 0.0;                              // round ms = scale x base(size)
    std::array<double, kMaxT + 1> base_{}, base_n_{};  // by window size; rounds behind it
    std::array<long, kMaxT + 1> seen_{};               // the round each size was last measured
    long round_ = 0;
    std::array<double, kMaxT + 1> mtp_tok_{}, mtp_n_{}; // tokens committed by MTP windows of that size
    std::array<double, kBuckets> ok_{}, bad_{};         // lookup drafts accepted / windows cut short
    std::array<double, kChanceBins> c_ok_{}, c_n_{};    // drafter drafts accepted / tested, by chance
};

} // namespace ninfer::spec
