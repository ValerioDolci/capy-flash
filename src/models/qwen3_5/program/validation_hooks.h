#pragma once

#include "core/tensor.h"
#include "models/qwen3_5/execution/qwen4_tfdump.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "ninfer/types.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

class ProgramImpl;

// The validation and diagnostic hooks of the decode and prompt rounds (docs/capy-flash/
// diagnostics.md): NINFER_QWEN4_TFDUMP*, REPLAY, REPLAY_RECORD, TOP2, ROUND_LOG, MTP_CALIB,
// MTP_ORACLE, FORCE_TOKENS, MTP_NO_DRAFTS, MTP_EXTENT_CAP. Built once from the environment when
// the Program is constructed; the Program keeps none when every hook is off, so a round tests one
// pointer per hook point. A new diagnostic is a member and a hook here.
class ValidationHooks {
public:
    // Null when no hook is set. Reads the environment in the order the hooks are listed above
    // (the validation-gated ones refuse to start without NINFER_QWEN4_VALIDATION=1).
    [[nodiscard]] static std::unique_ptr<ValidationHooks>
    from_environment(bool mtp, std::int32_t text_layers, std::size_t public_token_count,
                     const char* meta_engine, const std::string& meta_model);
    ValidationHooks() = default;
    ~ValidationHooks();
    ValidationHooks(const ValidationHooks&)            = delete;
    ValidationHooks& operator=(const ValidationHooks&) = delete;

    // The eager hidden taps of NINFER_QWEN4_TFDUMP (null without it).
    [[nodiscard]] execution::Qwen4Taps* taps() const noexcept { return tfdump.get(); }

    // A hook that replaces or reads the sampled tokens is set: sampling must not be bypassed.
    [[nodiscard]] bool bypasses_sampling() const noexcept;

    [[nodiscard]] bool mtp_calib() const noexcept { return calib_file != nullptr; }

    // Prompt: the first token after the prompt (teacher forcing, replay, recorder, round log).
    void on_first_token(std::span<const TokenId> ledger, std::uint32_t prompt_tokens,
                        TokenId& token);

    // Ordinary round, after the expert window ended: forced and replayed tokens, top-2 and
    // logits dumps of the round (`logits` = the ordinary logits, one column per row).
    void after_ordinary_round(const ProgramImpl& program, std::span<const std::uint32_t> lanes,
                              TokenId* sampled_tokens, const Tensor& logits);
    // Ordinary round, one committed row: the round log line and the teacher-forced token.
    void on_ordinary_commit(const ProgramImpl& program, std::uint32_t base_E, double seconds,
                            std::size_t ledger_size, TokenId& token, TokenId& sampled_token);

    // MTP round, one row before the launch: the replayed window of the row (if any) and the
    // proposed extent it implies (`model_extent` = the model's own draft count).
    struct MtpRowReplay {
        const execution::Qwen4ReplaySequence* ref  = nullptr;
        const execution::Qwen4ReplayWindow* window = nullptr;
    };

    [[nodiscard]] std::uint32_t plan_mtp_row(std::span<const TokenId> ledger,
                                             std::uint32_t frontier, std::uint32_t draft_window,
                                             std::uint32_t model_extent, MtpRowReplay& replay);
    // The verified extent: `proposed` capped by `limit` and by the validation caps.
    [[nodiscard]] std::uint32_t cap_mtp_extent(std::uint32_t proposed,
                                               std::uint32_t limit) const noexcept;
    // Oracle and replayed drafts of the row (`drafts`: draft_window entries) and the replayed
    // targets (`targets`: width entries).
    void override_mtp_drafts(std::span<const TokenId> ledger, std::uint32_t frontier,
                             std::uint32_t extent, std::uint32_t width, const MtpRowReplay& replay,
                             TokenId* drafts, TokenId* targets) const;
    // MTP round, after the expert window ended: forced tokens, logits and top-2 dumps of every
    // committed verification column (`target_logits` = [vocab, width, rows]).
    void after_mtp_round(const ProgramImpl& program, std::span<const std::uint32_t> lanes,
                         std::uint32_t width, const std::int32_t* licensed_counts,
                         TokenId* licensed_tokens, const Tensor& target_logits);
    // MTP round, one committed row: round log, replay recorder and replay check.
    void on_mtp_commit(const ProgramImpl& program, std::uint32_t base_E, std::uint32_t extent,
                       std::int32_t accepted, double seconds, std::span<const TokenId> drafts,
                       std::span<const TokenId> ledger, std::span<const TokenId> row_tokens);
    // MTP commit of a row that drafts again: "extent accepted logp_1 .. logp_K" of its round.
    void write_mtp_calib(std::int32_t extent, std::uint32_t committed,
                         std::span<const float> draft_logprobs);

    // Causal scoring: the scored prompt and each staged tile's logits.
    void on_score_prompt(std::span<const std::int32_t> tokens);
    void on_score_logits(const Tensor& logits, const std::vector<std::int32_t>& positions);

private:
    void log_round(const ProgramImpl& program, char kind, std::uint32_t frontier,
                   std::uint32_t extent, std::int32_t accepted, double round_seconds);
    void force_tokens(std::span<const TokenId> ledger, TokenId* tokens, std::int32_t count) const;
    void write_top2(const Tensor& logits_column, std::uint32_t lane, std::uint32_t position,
                    int source);

    // NINFER_QWEN4_MTP_ORACLE=<file> (validation): one whole token sequence per line (prompt +
    // a reference continuation). A round whose committed ledger is a prefix of a line drafts that
    // line's next tokens instead of the MTP proposals, so verification and the commit of long
    // accepted prefixes are exercised whatever the draft head's quality.
    std::vector<std::vector<TokenId>> mtp_oracle;
    // NINFER_QWEN4_FORCE_TOKENS=<file> (validation, same format): a round of a matching sequence
    // commits the file's tokens instead of its own choices (teacher forcing through the decode or
    // the verification route; with NINFER_QWEN4_MTP_NO_DRAFTS one token per round). The logits and
    // the state the round computed are untouched: they depend only on the consumed tokens.
    std::vector<std::vector<TokenId>> forced_tokens;
    // NINFER_QWEN4_MTP_NO_DRAFTS=1 (validation): every round verifies its anchor alone (extent 0,
    // the fallback step), i.e. greedy decode through the verification route.
    bool mtp_no_drafts = false;
    // NINFER_QWEN4_MTP_EXTENT_CAP=<k> (validation/measurement): verify at most k proposals per
    // round in the K-wide window (the cost of a column at fixed drafting work).
    std::uint32_t mtp_extent_cap = qwen3_5::kMtpMaximumProposalDrafts;
    // NINFER_QWEN4_MTP_CALIB=<file> (measurement): one line per verified MTP round, "extent
    // accepted logp_1 .. logp_K" (the proposals' own softmax log-probabilities).
    std::FILE* calib_file = nullptr;
    std::unique_ptr<execution::Qwen4TfDump> tfdump;
    // Teacher-forced replay of a recorded decode (NINFER_QWEN4_REPLAY, see
    // qwen4_tfdump.h) and its recorder (NINFER_QWEN4_REPLAY_RECORD=<file>: "req <prompt tokens>"
    // at every first token, "win <frontier> <extent> <accepted> <drafts...>" per MTP round).
    std::unique_ptr<execution::Qwen4Replay> replay;
    std::FILE* replay_record = nullptr;
    // Round log (NINFER_QWEN4_ROUND_LOG=<file>, measurement): one line per decode round,
    // "<o|m> <frontier> <extent> <accepted> <round us> <host gap us> <missing> <cpu experts>
    // <gpu-share experts> <expert window wall us> <fill wait us> <elastic expert columns>" (gap =
    // previous round's end -> this round's start, -1 for a request's first round; columns -1
    // without elastic KV).
    std::FILE* round_log = nullptr;
    std::chrono::steady_clock::time_point round_end{};
    bool round_follows = false;
    // NINFER_QWEN4_TOP2=<file> (validation): one line per decoded position, the target's two
    // largest logits: lane position source(0 ordinary, 1 verify) id1 logit1 id2 logit2.
    std::FILE* top2_file    = nullptr;
    std::size_t top2_domain = 0;
};

} // namespace ninfer::models::qwen3_5::detail
