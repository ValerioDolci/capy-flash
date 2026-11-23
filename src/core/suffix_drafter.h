// The suffix-lookup drafter (prompt lookup): no weights, no GPU.
//
// Ported from Strata (engine/include/strata/spec/suffix_drafter.hpp), MIT License,
// Copyright (c) 2026 Niko1221 and the Strata contributors.
//
// Proposes the tokens that followed the longest earlier occurrence of the sequence's current
// suffix: when the output quotes its input (edits, refactors, repeated code) the continuation is
// usually exact, and one verify window accepts many tokens at once.
//
// Index: every trigram of the history (prompt + accepted output) maps to its kWays most recent end
// positions in a fixed-size open-addressing table, so memory is bounded (~20 bytes per history
// token) and appends are O(1). A proposal checks those candidates, extends each match backwards up
// to `max_match`, and takes the longest (most recent on ties). Matches shorter than `min_match`
// propose nothing.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::spec {

class SuffixDrafter {
public:
    static constexpr int kWays = 4;

    explicit SuffixDrafter(int min_match = 3, int max_match = 32,
                           std::size_t capacity_tokens = std::size_t{1} << 19);

    void reset();
    // Adds tokens to the history (the prompt, then every accepted token).
    void append(const std::int32_t* tokens, std::size_t n);

    void append(std::int32_t token) { append(&token, 1); }

    // Writes up to `max_k` proposed next tokens to `out`; returns how many (0 = no match of at
    // least min_match).
    int propose(int max_k, std::int32_t* out);

    // As propose(), for the history followed by `extra` (tokens not yet verified): the lookup
    // continues where they end. The history itself is unchanged.
    int propose_after(const std::int32_t* extra, int n_extra, int max_k, std::int32_t* out);

    // Length of the match behind the last proposal (0 if none).
    [[nodiscard]] int last_match() const { return last_match_; }

    [[nodiscard]] std::size_t size() const { return hist_.size(); }

    [[nodiscard]] const std::vector<std::int32_t>& history() const { return hist_; }

private:
    struct Slot {
        std::uint64_t key        = 0;  // trigram hash + 1 (0 = empty)
        std::uint32_t pos[kWays] = {}; // end positions, most recent first
        std::uint8_t n           = 0;
    };

    Slot* find_slot(std::uint64_t key, bool insert);
    [[nodiscard]] std::uint64_t key_at(std::size_t end) const;

    int min_match_, max_match_;
    std::vector<std::int32_t> hist_;
    std::vector<Slot> table_;
    std::size_t mask_ = 0;
    int last_match_   = 0;
};

} // namespace ninfer::spec
