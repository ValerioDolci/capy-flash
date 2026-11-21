#pragma once

#include "ninfer/types.h"
#include "runtime/contract/request.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::inkling2 {
namespace frontend {
class Tokenizer;
} // namespace frontend
class Frontend;

// Deltas of one committed output transaction, in order, at most one per channel transition.
class PublishedOutput {
public:
    using iterator       = std::vector<OutputDelta>::iterator;
    using const_iterator = std::vector<OutputDelta>::const_iterator;

    [[nodiscard]] bool empty() const noexcept { return values_.empty(); }

    [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }

    [[nodiscard]] iterator begin() noexcept { return values_.begin(); }

    [[nodiscard]] const_iterator begin() const noexcept { return values_.begin(); }

    [[nodiscard]] iterator end() noexcept { return values_.end(); }

    [[nodiscard]] const_iterator end() const noexcept { return values_.end(); }

    [[nodiscard]] OutputDelta& back() noexcept { return values_.back(); }

    [[nodiscard]] const OutputDelta& back() const noexcept { return values_.back(); }

    void clear() noexcept { values_.clear(); }

    // Appends to the last delta when the channel is unchanged; empty text is dropped.
    void append(OutputChannel channel, std::string text);

private:
    std::vector<OutputDelta> values_;
};

// One request's generated output: Inkling 2 message channels (frontend/output_parser.h), stop
// tokens and stop strings, the thinking budget, and the Engine's preview/commit transactions.
// The thinking phase of a chat prompt is open from the first generated token: the model always
// begins its turn with <|message_model|><|content_thinking|>, and those tokens count toward the
// thinking budget and the reasoning tokens. A thinking budget ends an open thinking message with
// the model's own continuation
// <|end_message|><|message_model|><|content_text|>, so the reply goes on as text.
class OutputSession {
public:
    OutputSession() noexcept;
    ~OutputSession();
    OutputSession(OutputSession&&) noexcept;
    OutputSession& operator=(OutputSession&&) noexcept;

    OutputSession(const OutputSession&)            = delete;
    OutputSession& operator=(const OutputSession&) = delete;

    [[nodiscard]] runtime::OutputDecision preview_model(std::span<const TokenId> tokens,
                                                        std::uint32_t total_budget_remaining,
                                                        FinishReason limit_reason);
    [[nodiscard]] std::uint32_t
    model_token_budget_remaining(std::uint32_t total_budget_remaining) const noexcept;
    [[nodiscard]] std::span<const TokenId> pending_control_tokens() const noexcept;

    [[nodiscard]] bool control_completes_close() const noexcept { return false; }

    [[nodiscard]] std::uint32_t control_suffix_tokens() const noexcept;
    [[nodiscard]] runtime::OutputDecision preview_control(std::span<const TokenId> tokens,
                                                          std::uint32_t total_budget_remaining);
    void validate_generation_capacity(std::uint32_t effective_output_tokens) const;
    [[nodiscard]] runtime::OutputDecision preview_terminal(FinishReason reason);
    [[nodiscard]] PublishedOutput commit_preview();
    void discard_preview() noexcept;

    // Structured output is not supported: never constrained.
    [[nodiscard]] bool constrained() const noexcept { return false; }

    [[nodiscard]] std::uint32_t grammar_masks(std::span<const TokenId> drafts,
                                              std::span<std::uint32_t> words);
    [[nodiscard]] RequestError constraint_error(std::string dead_end) const;
    [[nodiscard]] std::vector<GeneratedToolCall> take_tool_calls() noexcept;
    [[nodiscard]] ToolCallParseDiagnostics tool_call_parse_diagnostics() const noexcept;
    [[nodiscard]] std::uint32_t reasoning_tokens() const noexcept;
    [[nodiscard]] ThinkingBudgetStats thinking_stats() const noexcept;
    [[nodiscard]] std::optional<std::string> matched_stop_string() const;

private:
    class Impl;
    OutputSession(std::shared_ptr<const frontend::Tokenizer> tokenizer, StopPolicy policy,
                  OutputOptions output, ThinkingControlOptions thinking,
                  std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens,
                  std::vector<std::string> filtered_out_tool_names, bool starts_in_reasoning,
                  std::span<const TokenId> resume_tokens);
    std::unique_ptr<Impl> impl_;

    friend class Frontend;
};

} // namespace ninfer::models::inkling2
