#include "models/inkling2/frontend/output_session.h"

#include "models/inkling2/frontend/output_parser.h"
#include "models/inkling2/frontend/tokenizer.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::models::inkling2 {
namespace {

namespace fi = frontend;

std::size_t channel_index(OutputChannel channel) noexcept {
    return channel == OutputChannel::Reasoning ? 0 : 1;
}

// Longest proper prefix of `marker` that ends `text`.
std::size_t longest_suffix_prefix(std::string_view text, std::string_view marker) {
    const std::size_t maximum = std::min(text.size(), marker.size() - 1);
    for (std::size_t size = maximum; size != 0; --size) {
        if (text.substr(text.size() - size) == marker.substr(0, size)) { return size; }
    }
    return 0;
}

struct StopMatch {
    bool found                      = false;
    std::uint32_t declaration_order = 0;
};

// Committed (or previewed) decoding state; copied to open a preview.
struct State {
    explicit State(const fi::Tokenizer& tokenizer, std::vector<std::string> filtered)
        : parser(tokenizer, std::move(filtered)), raw(tokenizer) {}

    fi::OutputParser parser;
    fi::TokenStreamDecoder raw;
    // Text held per channel because it may begin a stop string.
    std::array<std::string, 2> stop_pending;
    std::optional<OutputChannel> pending_channel;
    // The turn's thinking phase: open from the first generated token of a chat prompt (the
    // model opens its own thinking message), closed by the <|end_message|> of that message or
    // by the opening of a message that is not thinking.
    bool thinking_open                  = false;
    std::uint32_t model_thinking_tokens = 0;
    // Tokens accepted in the thinking phase or inside any later thinking message.
    std::uint32_t reasoning_tokens = 0;
    std::uint32_t injected_tokens  = 0;
    bool control_pending           = false;
    bool applied                   = false;
    bool terminal                  = false;
    std::optional<std::uint32_t> matched_stop_order;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics diagnostics;
};

void flush_pending(State& state, PublishedOutput& out) {
    if (!state.pending_channel) { return; }
    std::string& pending = state.stop_pending[channel_index(*state.pending_channel)];
    out.append(*state.pending_channel, std::move(pending));
    pending.clear();
    state.pending_channel.reset();
}

// Publishes `text` on `channel`, holding a possible stop-string prefix. On a match the output
// ends before the stop string (or after it with include_in_output) and true is returned.
bool feed(State& state, const StopPolicy& policy, PublishedOutput& out, OutputChannel channel,
          std::string_view text, bool match_stops) {
    if (text.empty()) { return false; }
    if (state.pending_channel && *state.pending_channel != channel) { flush_pending(state, out); }
    std::string combined = std::move(state.stop_pending[channel_index(channel)]);
    state.stop_pending[channel_index(channel)].clear();
    combined.append(text);
    if (match_stops) {
        std::size_t best    = std::string::npos;
        std::uint32_t order = 0;
        for (std::size_t index = 0; index < policy.strings.size(); ++index) {
            const StopString& stop = policy.strings[index];
            if (stop.channel != channel) { continue; }
            const std::size_t found = combined.find(stop.text);
            if (found != std::string::npos && (best == std::string::npos || found < best)) {
                best  = found;
                order = static_cast<std::uint32_t>(index);
            }
        }
        if (best != std::string::npos) {
            out.append(channel, combined.substr(0, best));
            if (policy.strings[order].include_in_output) {
                out.append(channel, policy.strings[order].text);
            }
            state.matched_stop_order = order;
            state.pending_channel.reset();
            return true;
        }
    }
    std::size_t hold = 0;
    for (const StopString& stop : policy.strings) {
        if (stop.channel == channel) {
            hold = std::max(hold, longest_suffix_prefix(combined, stop.text));
        }
    }
    out.append(channel, combined.substr(0, combined.size() - hold));
    state.stop_pending[channel_index(channel)] = combined.substr(combined.size() - hold);
    state.pending_channel = hold != 0 ? std::optional<OutputChannel>(channel) : std::nullopt;
    return false;
}

bool feed_deltas(State& state, const StopPolicy& policy, PublishedOutput& out,
                 const std::vector<OutputDelta>& deltas, bool match_stops) {
    for (const OutputDelta& delta : deltas) {
        if (feed(state, policy, out, delta.channel, delta.text, match_stops)) { return true; }
    }
    return false;
}

// Pushes one token through the parser and follows the thinking phase.
fi::OutputStep push_token(State& state, TokenId token) {
    const bool was_reasoning = state.parser.in_reasoning();
    fi::OutputStep step      = state.parser.push(token);
    if (state.thinking_open && ((was_reasoning && !state.parser.in_reasoning()) ||
                                (state.parser.in_body() && !state.parser.in_reasoning()))) {
        state.thinking_open = false;
    }
    return step;
}

// Ends the output: the parser closes an open message (a tool call completes only at the model's
// own stop) and held text is published. After a stop string, nothing more is published.
void terminalize(State& state, const StopPolicy& policy, PublishedOutput& out,
                 fi::ToolCallRegionEnd end, bool after_stop_string) {
    fi::OutputTerminal terminal = state.parser.finish(end);
    if (!after_stop_string) {
        const std::string raw_tail = state.raw.finish();
        (void)feed(state, policy, out, OutputChannel::Content, raw_tail, false);
        (void)feed_deltas(state, policy, out, terminal.deltas, false);
        flush_pending(state, out);
    }
    state.stop_pending = {};
    state.pending_channel.reset();
    state.tool_calls  = std::move(terminal.tool_calls);
    state.diagnostics = terminal.diagnostics;
    state.terminal    = true;
}

} // namespace

void PublishedOutput::append(OutputChannel channel, std::string text) {
    if (text.empty()) { return; }
    if (!values_.empty() && values_.back().channel == channel) {
        values_.back().text += text;
    } else {
        values_.push_back(OutputDelta{.channel = channel, .text = std::move(text)});
    }
}

class OutputSession::Impl {
public:
    Impl(std::shared_ptr<const fi::Tokenizer> tokenizer_, StopPolicy policy_, OutputOptions output,
         ThinkingControlOptions thinking,
         std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens_,
         std::vector<std::string> filtered_out_tool_names, bool starts_in_reasoning,
         std::span<const TokenId> resume_tokens)
        : tokenizer(std::move(tokenizer_)), policy(std::move(policy_)),
          thinking_control_tokens(std::move(thinking_control_tokens_)), raw(output.raw),
          budget(thinking.budget), state(*tokenizer, std::move(filtered_out_tool_names)) {
        if (budget && *budget == 0) {
            throw std::invalid_argument("thinking budget must be positive");
        }
        state.thinking_open = starts_in_reasoning && !output.raw;
        // A continuation resumes after tokens that belong to the prompt: replay them unpublished.
        for (const TokenId token : resume_tokens) {
            if (state.parser.push(token).stop) {
                throw std::invalid_argument("continuation tokens end the assistant turn");
            }
        }
    }

    [[nodiscard]] bool stop_token(TokenId token) const {
        return std::find(policy.token_ids.begin(), policy.token_ids.end(), token) !=
               policy.token_ids.end();
    }

    void open_preview() {
        if (state.terminal) { throw std::logic_error("output session is already terminal"); }
        if (preview) { throw std::logic_error("output session already has a preview"); }
        preview.emplace(state);
        preview_output.clear();
    }

    std::shared_ptr<const fi::Tokenizer> tokenizer;
    StopPolicy policy;
    std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens;
    bool raw = false;
    std::optional<std::uint32_t> budget;
    State state;
    std::optional<State> preview;
    PublishedOutput preview_output;
};

OutputSession::OutputSession() noexcept                           = default;
OutputSession::~OutputSession()                                   = default;
OutputSession::OutputSession(OutputSession&&) noexcept            = default;
OutputSession& OutputSession::operator=(OutputSession&&) noexcept = default;

OutputSession::OutputSession(std::shared_ptr<const frontend::Tokenizer> tokenizer,
                             StopPolicy policy, OutputOptions output,
                             ThinkingControlOptions thinking,
                             std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens,
                             std::vector<std::string> filtered_out_tool_names,
                             bool starts_in_reasoning, std::span<const TokenId> resume_tokens)
    : impl_(std::make_unique<Impl>(std::move(tokenizer), std::move(policy), output, thinking,
                                   std::move(thinking_control_tokens),
                                   std::move(filtered_out_tool_names), starts_in_reasoning,
                                   resume_tokens)) {}

runtime::OutputDecision OutputSession::preview_model(std::span<const TokenId> tokens,
                                                     std::uint32_t total_budget_remaining,
                                                     FinishReason limit_reason) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (impl_->state.control_pending) {
        throw std::logic_error("model output cannot advance while thinking control is pending");
    }
    if (tokens.empty()) {
        throw std::invalid_argument("cannot preview an empty generated-token round");
    }
    if (tokens.size() > total_budget_remaining) {
        throw std::invalid_argument("generated-token round exceeds the remaining budget");
    }
    if (limit_reason != FinishReason::OutputLimit &&
        limit_reason != FinishReason::ContextCapacity) {
        throw std::invalid_argument("generated-token budget has an invalid limit reason");
    }
    impl_->open_preview();
    State& state             = *impl_->preview;
    PublishedOutput& out     = impl_->preview_output;
    const StopPolicy& policy = impl_->policy;
    const auto decision      = [](std::uint32_t count, FinishReason reason,
                             runtime::ContinuationAction continuation =
                                 runtime::ContinuationAction::Decode) {
        return runtime::OutputDecision{
                 .accepted_tokens = count, .finish_reason = reason, .continuation = continuation};
    };

    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const auto count                   = static_cast<std::uint32_t>(index + 1);
        const TokenId token                = tokens[index];
        const fi::DecodedTokenView decoded = impl_->tokenizer->decoded_token(token);
        if (impl_->stop_token(token)) {
            if (impl_->raw && policy.publish_stop_token) {
                (void)feed(state, policy, out, OutputChannel::Content,
                           state.raw.push_bytes(decoded.bytes), false);
            }
            terminalize(state, policy, out, fi::ToolCallRegionEnd::ModelStop, false);
            return decision(count, FinishReason::StopToken);
        }
        if (impl_->raw) {
            if (feed(state, policy, out, OutputChannel::Content,
                     state.raw.push_bytes(decoded.bytes), true)) {
                terminalize(state, policy, out, fi::ToolCallRegionEnd::Cut, true);
                return decision(count, FinishReason::StopString);
            }
        } else {
            if (state.thinking_open) { ++state.model_thinking_tokens; }
            if (state.thinking_open || state.parser.in_reasoning()) { ++state.reasoning_tokens; }
            const fi::OutputStep step = push_token(state, token);
            if (feed_deltas(state, policy, out, step.deltas, true)) {
                terminalize(state, policy, out, fi::ToolCallRegionEnd::Cut, true);
                return decision(count, FinishReason::StopString);
            }
            if (step.stop) {
                terminalize(state, policy, out, fi::ToolCallRegionEnd::ModelStop, false);
                return decision(count, FinishReason::StopToken);
            }
        }
        if (count == total_budget_remaining) {
            terminalize(state, policy, out, fi::ToolCallRegionEnd::Cut, false);
            return decision(count, limit_reason);
        }
        if (impl_->budget && !state.applied && state.thinking_open &&
            state.model_thinking_tokens >= *impl_->budget) {
            state.control_pending = true;
            return decision(count, FinishReason::None,
                            runtime::ContinuationAction::ApplyTargetControl);
        }
    }
    return decision(static_cast<std::uint32_t>(tokens.size()), FinishReason::None);
}

std::uint32_t
OutputSession::model_token_budget_remaining(std::uint32_t total_budget_remaining) const noexcept {
    if (impl_ == nullptr || !impl_->budget || impl_->state.applied) {
        return total_budget_remaining;
    }
    if (impl_->state.control_pending) { return 0; }
    if (!impl_->state.thinking_open) { return total_budget_remaining; }
    return std::min(total_budget_remaining,
                    *impl_->budget - std::min(*impl_->budget, impl_->state.model_thinking_tokens));
}

std::span<const TokenId> OutputSession::pending_control_tokens() const noexcept {
    if (impl_ == nullptr || !impl_->state.control_pending) { return {}; }
    return *impl_->thinking_control_tokens;
}

std::uint32_t OutputSession::control_suffix_tokens() const noexcept {
    if (impl_ == nullptr || !impl_->budget || impl_->state.applied ||
        !(impl_->state.thinking_open || impl_->state.control_pending)) {
        return 0;
    }
    return static_cast<std::uint32_t>(impl_->thinking_control_tokens->size());
}

runtime::OutputDecision OutputSession::preview_control(std::span<const TokenId> tokens,
                                                       std::uint32_t total_budget_remaining) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    const std::span<const TokenId> expected = pending_control_tokens();
    if (expected.empty() ||
        !std::equal(tokens.begin(), tokens.end(), expected.begin(), expected.end())) {
        throw std::invalid_argument("thinking control preview requires the exact pending span");
    }
    if (tokens.size() > total_budget_remaining) {
        throw std::invalid_argument("thinking control span exceeds the remaining output budget");
    }
    impl_->open_preview();
    State& state = *impl_->preview;
    // The control span is not model-origin thinking: it is not counted.
    for (const TokenId token : tokens) {
        (void)feed_deltas(state, impl_->policy, impl_->preview_output,
                          push_token(state, token).deltas, false);
    }
    if (state.thinking_open || state.parser.in_reasoning()) {
        throw std::logic_error("thinking control did not close the thinking message");
    }
    state.control_pending = false;
    state.applied         = true;
    state.injected_tokens = static_cast<std::uint32_t>(tokens.size());
    return runtime::OutputDecision{.accepted_tokens = static_cast<std::uint32_t>(tokens.size())};
}

void OutputSession::validate_generation_capacity(std::uint32_t effective_output_tokens) const {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (!impl_->budget || effective_output_tokens <= *impl_->budget) { return; }
    const std::uint64_t remaining = std::uint64_t{effective_output_tokens} - *impl_->budget;
    if (remaining < impl_->thinking_control_tokens->size() + 1U) {
        throw std::invalid_argument(
            "effective output capacity after the thinking budget must fit the complete control "
            "suffix and one post-close model token");
    }
}

runtime::OutputDecision OutputSession::preview_terminal(FinishReason reason) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (reason == FinishReason::None || reason == FinishReason::StopString ||
        reason == FinishReason::StopToken) {
        throw std::invalid_argument("invalid between-round terminal decoder reason");
    }
    impl_->open_preview();
    impl_->preview->control_pending = false;
    terminalize(*impl_->preview, impl_->policy, impl_->preview_output, fi::ToolCallRegionEnd::Cut,
                false);
    return runtime::OutputDecision{.accepted_tokens = 0, .finish_reason = reason};
}

PublishedOutput OutputSession::commit_preview() {
    if (impl_ == nullptr || !impl_->preview) {
        throw std::logic_error("output session has no preview to commit");
    }
    impl_->state = std::move(*impl_->preview);
    impl_->preview.reset();
    PublishedOutput output = std::move(impl_->preview_output);
    impl_->preview_output.clear();
    return output;
}

void OutputSession::discard_preview() noexcept {
    if (impl_ == nullptr) { return; }
    impl_->preview.reset();
    impl_->preview_output.clear();
}

std::uint32_t OutputSession::grammar_masks(std::span<const TokenId>, std::span<std::uint32_t>) {
    throw std::logic_error("mask requested for unconstrained output");
}

RequestError OutputSession::constraint_error(std::string dead_end) const {
    return RequestError(RequestErrorKind::ConstraintDeadEnd, std::move(dead_end));
}

std::vector<GeneratedToolCall> OutputSession::take_tool_calls() noexcept {
    return impl_ != nullptr ? std::move(impl_->state.tool_calls) : std::vector<GeneratedToolCall>{};
}

ToolCallParseDiagnostics OutputSession::tool_call_parse_diagnostics() const noexcept {
    return impl_ != nullptr ? impl_->state.diagnostics : ToolCallParseDiagnostics{};
}

std::uint32_t OutputSession::reasoning_tokens() const noexcept {
    return impl_ != nullptr ? impl_->state.reasoning_tokens : 0;
}

ThinkingBudgetStats OutputSession::thinking_stats() const noexcept {
    if (impl_ == nullptr) { return {}; }
    return ThinkingBudgetStats{.configured_budget     = impl_->budget,
                               .model_thinking_tokens = impl_->state.model_thinking_tokens,
                               .injected_tokens       = impl_->state.injected_tokens,
                               .applied               = impl_->state.applied};
}

std::optional<std::string> OutputSession::matched_stop_string() const {
    if (impl_ == nullptr || !impl_->state.matched_stop_order) { return std::nullopt; }
    return impl_->policy.strings.at(*impl_->state.matched_stop_order).text;
}

} // namespace ninfer::models::inkling2
