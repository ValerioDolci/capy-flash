#pragma once

#include "models/inkling2/frontend/tokenizer.h"

#include <ninfer/types.h>

#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::models::inkling2::frontend {

// How the generated text ended. A tool call still open at ModelStop (the model's end-of-turn
// token inside the call) is completed when its JSON parses; one open at Cut (token budget,
// context limit, cancellation) stays ordinary content.
enum class ToolCallRegionEnd : std::uint8_t {
    ModelStop,
    Cut,
};

struct OutputStep {
    std::vector<OutputDelta> deltas;
    // The token ends the turn: <|content_model_end_sampling|>, or the header of a message the
    // model does not write (user, system, tool).
    bool stop = false;
};

struct OutputTerminal {
    std::vector<OutputDelta> deltas;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics diagnostics;
};

// Channels of one generated Inkling 2 turn, token by token. The model writes a sequence of
// messages, each `<|message_model|>[name]<|content_KIND|>body<|end_message|>`, and ends the turn
// with <|content_model_end_sampling|>:
//   content_thinking: reasoning (several messages concatenate);
//   content_text, content_tool_error, content_invoke_tool_text: content;
//   content_invoke_tool_json: one tool call, body {"name": ..., "args": {...}}.
// Reasoning and content are published as soon as they form complete UTF-8; a tool-call body is
// held until its <|end_message|> and becomes a call, or, when it is not a JSON object with a
// string name ([A-Za-z0-9_-]) and an object args, ordinary content. Text outside any message
// body is content. The parser is a value: copy it to preview tokens and keep or drop the copy.
class OutputParser {
public:
    explicit OutputParser(const Tokenizer& tokenizer,
                          std::vector<std::string> filtered_out_tool_names = {});

    [[nodiscard]] OutputStep push(int token);
    [[nodiscard]] OutputTerminal finish(ToolCallRegionEnd end);

    [[nodiscard]] bool stopped() const noexcept { return stopped_; }

    // Inside a thinking body: the next model token is reasoning (or the <|end_message|> that
    // closes it).
    [[nodiscard]] bool in_reasoning() const noexcept {
        return state_ == State::Body && body_ == Body::Reasoning;
    }

    // Inside any message body (thinking, text or tool call).
    [[nodiscard]] bool in_body() const noexcept { return state_ == State::Body; }

    // Tokens of thinking bodies (their delimiters excluded).
    [[nodiscard]] std::uint32_t reasoning_tokens() const noexcept { return reasoning_tokens_; }

    // Calls completed so far (each at its <|end_message|>).
    [[nodiscard]] const std::vector<GeneratedToolCall>& tool_calls() const noexcept {
        return tool_calls_;
    }

private:
    enum class State : std::uint8_t { Between, Header, Body };
    enum class Body : std::uint8_t { Reasoning, Content, ToolCall };

    struct Ids {
        int message_model    = -1;
        int message_user     = -1;
        int message_system   = -1;
        int message_tool     = -1;
        int end_message      = -1;
        int end_sampling     = -1;
        int thinking         = -1;
        int text             = -1;
        int tool_error       = -1;
        int invoke_tool_text = -1;
        int invoke_tool_json = -1;
    };

    void open_body(Body body, std::vector<OutputDelta>& deltas);
    void close_body(std::vector<OutputDelta>& deltas, bool complete, ToolCallRegionEnd end);
    void append(std::vector<OutputDelta>& deltas, OutputChannel channel, std::string text) const;

    const Tokenizer* tokenizer_;
    Ids ids_;
    std::vector<std::string> filtered_out_tool_names_;
    State state_ = State::Between;
    Body body_   = Body::Content;
    TokenStreamDecoder text_;
    std::string header_;
    std::string tool_body_;
    std::vector<GeneratedToolCall> tool_calls_;
    ToolCallParseDiagnostics diagnostics_;
    std::uint32_t reasoning_tokens_ = 0;
    bool stopped_                   = false;
    bool finished_                  = false;
};

} // namespace ninfer::models::inkling2::frontend
