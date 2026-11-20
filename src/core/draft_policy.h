// Per verify round: the MTP's window, or a lookup (suffix drafter) window?
//
// Ported from Strata (engine/include/strata/spec/draft_policy.hpp, the live-cost core of
// DraftPolicy), MIT License, Copyright (c) 2026 Niko1221 and the Strata contributors. Not ported:
// cost tables learned offline (CostLearner, cost files, calibration), gated proposals, lookup
// chains and the cost-aware MTP window.
//
// The suffix drafter proposes the tokens that followed an earlier repeat of the context. Taken
// whenever it proposes more than the MTP, it loses on ordinary text: a long lookup window costs
// much more to verify than the MTP's usual 3-4 tokens, and it is only worth that when enough of it
// is accepted. The policy learns both sides online and compares expected committed tokens per
// millisecond:
//
//   MTP     E = the mean tokens a window of that size has committed (EMA), at the measured cost of
//           that size
//   lookup  E(k) = 1 + q + q^2 + ... + q^k for k <= the proposal, q = the acceptance rate of lookup
//           drafts whose match was about as long (4 buckets of match length, decayed counts), at
//           the measured cost of k + 1
//
// and takes the lookup window only when its best E/cost beats the MTP's by `margin`. Costs are the
// measured round times per window size (EMA; sizes not seen yet are scaled from seen ones by a
// prior shape), so the policy adapts to the machine and the context length. It only chooses which
// drafts to verify; a different window is a different verifier batch, so two runs whose measured
// round times differ can choose different windows.
#pragma once

#include <array>
#include <string>

namespace ninfer::spec {

class DraftPolicy {
public:
    static constexpr int kMaxT    = 16;
    static constexpr int kBuckets = 4;
    static constexpr int kCtxBuckets =
        3; // robust mode: row costs per context length (< 6K, < 24K, longer)

    explicit DraftPolicy(int max_t, double margin = 0.03);

    struct Pick {
        bool lookup = false;
        int t       = 1; // window size (1 + drafts)
    };

    // `t_mtp`: the MTP's window; `lookup_k`: the lookup proposal's length (0 = none); `match`: its
    // match length.
    [[nodiscard]] Pick choose(int t_mtp, int lookup_k, int match) const;
    // After the round: the window it used, the drafts accepted, and the round's time (verify +
    // commit + draft).
    void observe(bool lookup, int t, int accepted, int match, double round_ms);

    [[nodiscard]] double lookup_rate(int match) const; // current q for a match length
    [[nodiscard]] double
    cost_ms(int t) const; // measured or scaled round time of a window of t tokens
    // Robust mode: the context position of the next round selects the row-cost table of its length.
    // A bucket with no measurement yet starts from the nearest one that has. A no-op otherwise.
    void set_context(long long pos);

    // Robust mode: a round time far above what is known for that width (the first use of a width
    // captures its graph; a stalled round) is clamped, so one capture cannot price a width out of
    // use. Off by default, as in Strata's default serve.
    void set_robust(bool on) { robust_ = on; }

    // The measured round cost of every window size seen, for the log.
    [[nodiscard]] std::string cost_table_text() const;

    // The window sizes a lookup window may use, as a bit mask (bit t = size t; bit 1 always kept).
    void set_sizes(unsigned mask);
    void disable(int t); // drops one size (its window graph did not fit)

    [[nodiscard]] bool allowed(int t) const {
        return t >= 1 && t <= max_t_ && ((sizes_ >> t) & 1U) != 0;
    }

    [[nodiscard]] int fit(int t) const; // the largest allowed size <= t (1 if none)

    [[nodiscard]] int max_t() const { return max_t_; }

    [[nodiscard]] unsigned sizes() const { return sizes_; }

private:
    static int bucket(int match);
    [[nodiscard]] double mtp_tokens(int t) const;
    [[nodiscard]] bool stale(int t) const; // a size not measured for a long stretch of rounds
    [[nodiscard]] double clamp_sample(int t, double ms) const;

    int max_t_;
    double margin_;
    unsigned sizes_ = ~0U; // allowed window sizes (bit t)
    std::array<std::array<double, kMaxT + 1>, kCtxBuckets> cost_{},
        cost_n_{}; // ms by bucket and size
    int bk_ = 0;   // the current context bucket
    std::array<double, kMaxT + 1> mtp_tok_{},
        mtp_n_{}; // tokens committed by MTP windows of that size
    std::array<double, kBuckets> ok_{},
        bad_{};         // lookup drafts accepted / windows cut short, decayed
    double rounds_ = 0; // rounds observed so far
    bool robust_   = false;
    std::array<std::array<double, kMaxT + 1>, kCtxBuckets>
        last_{}; // round each size was last measured
};

} // namespace ninfer::spec
