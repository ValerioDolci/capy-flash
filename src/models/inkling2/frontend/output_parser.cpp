#include "models/inkling2/frontend/output_parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::models::inkling2::frontend {
namespace {

int require_added_token(const Tokenizer& tokenizer, std::string_view content) {
    const auto id = tokenizer.added_token_id(content);
    if (!id) {
        throw std::invalid_argument("Inkling 2 tokenizer lacks the control token " +
                                    std::string(content));
    }
    return *id;
}

// The tool-name charset the request parsers accept, so a call can be echoed back next turn.
bool valid_tool_name(std::string_view name) {
    return !name.empty() && std::all_of(name.begin(), name.end(), [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
               c == '_' || c == '-';
    });
}

} // namespace

OutputParser::OutputParser(const Tokenizer& tokenizer,
                           std::vector<std::string> filtered_out_tool_names)
    : tokenizer_(&tokenizer), filtered_out_tool_names_(std::move(filtered_out_tool_names)),
      text_(tokenizer) {
    ids_.message_model    = require_added_token(tokenizer, "<|message_model|>");
    ids_.message_user     = require_added_token(tokenizer, "<|message_user|>");
    ids_.message_system   = require_added_token(tokenizer, "<|message_system|>");
    ids_.message_tool     = require_added_token(tokenizer, "<|message_tool|>");
    ids_.end_message      = require_added_token(tokenizer, "<|end_message|>");
    ids_.end_sampling     = require_added_token(tokenizer, "<|content_model_end_sampling|>");
    ids_.thinking         = require_added_token(tokenizer, "<|content_thinking|>");
    ids_.text             = require_added_token(tokenizer, "<|content_text|>");
    ids_.tool_error       = require_added_token(tokenizer, "<|content_tool_error|>");
    ids_.invoke_tool_text = require_added_token(tokenizer, "<|content_invoke_tool_text|>");
    ids_.invoke_tool_json = require_added_token(tokenizer, "<|content_invoke_tool_json|>");
}

void OutputParser::append(std::vector<OutputDelta>& deltas, OutputChannel channel,
                          std::string text) const {
    if (text.empty()) { return; }
    if (!deltas.empty() && deltas.back().channel == channel) {
        deltas.back().text += text;
    } else {
        deltas.push_back(OutputDelta{.channel = channel, .text = std::move(text)});
    }
}

void OutputParser::open_body(Body body, std::vector<OutputDelta>& deltas) {
    if (state_ == State::Body) { close_body(deltas, false, ToolCallRegionEnd::ModelStop); }
    // Text written outside a message body was published as content; finish its last character.
    append(deltas, OutputChannel::Content, text_.finish());
    state_ = State::Body;
    body_  = body;
    if (body == Body::ToolCall) {
        tool_body_.clear();
        diagnostics_.marker_seen = true;
    }
}

void OutputParser::close_body(std::vector<OutputDelta>& deltas, bool complete,
                              ToolCallRegionEnd end) {
    if (state_ != State::Body) { return; }
    state_ = State::Between;
    // The header repeats the tool name of a call; the JSON body is authoritative.
    header_.clear();
    if (body_ != Body::ToolCall) {
        append(deltas, body_ == Body::Reasoning ? OutputChannel::Reasoning : OutputChannel::Content,
               text_.finish());
        return;
    }
    const auto fallback = [&](ToolCallParseFallbackReason reason) {
        if (diagnostics_.fallback_reason == ToolCallParseFallbackReason::None) {
            diagnostics_.fallback_reason = reason;
        }
        TokenStreamDecoder text(*tokenizer_);
        std::string content = text.push_bytes(tool_body_);
        content += text.finish();
        append(deltas, OutputChannel::Content, std::move(content));
    };
    if (!complete && end == ToolCallRegionEnd::Cut) {
        fallback(ToolCallParseFallbackReason::MalformedStructure);
        return;
    }
    const auto call = nlohmann::ordered_json::parse(tool_body_, nullptr, false);
    if (call.is_discarded() || !call.is_object() || !call.contains("name") ||
        !call["name"].is_string() || !call.contains("args") || !call["args"].is_object()) {
        fallback(ToolCallParseFallbackReason::MalformedStructure);
        return;
    }
    const std::string name = call["name"].get<std::string>();
    if (!valid_tool_name(name)) {
        fallback(ToolCallParseFallbackReason::InvalidToolName);
        return;
    }
    if (std::find(filtered_out_tool_names_.begin(), filtered_out_tool_names_.end(), name) !=
        filtered_out_tool_names_.end()) {
        fallback(ToolCallParseFallbackReason::FilteredTool);
        return;
    }
    tool_calls_.push_back(GeneratedToolCall{.name = name, .arguments_json = call["args"].dump()});
    ++diagnostics_.structured_call_count;
    if (!complete) { ++diagnostics_.structural_salvage_count; }
}

OutputStep OutputParser::push(int token) {
    if (finished_ || stopped_) {
        throw std::logic_error("Inkling 2 output parser received a token after the turn ended");
    }
    OutputStep step;
    if (token == ids_.end_sampling || token == ids_.message_user || token == ids_.message_system ||
        token == ids_.message_tool) {
        close_body(step.deltas, false, ToolCallRegionEnd::ModelStop);
        append(step.deltas, OutputChannel::Content, text_.finish());
        stopped_  = true;
        step.stop = true;
        return step;
    }
    if (token == ids_.message_model) {
        close_body(step.deltas, false, ToolCallRegionEnd::ModelStop);
        append(step.deltas, OutputChannel::Content, text_.finish());
        state_ = State::Header;
        header_.clear();
        return step;
    }
    if (token == ids_.end_message) {
        close_body(step.deltas, true, ToolCallRegionEnd::ModelStop);
        state_ = State::Between;
        header_.clear();
        return step;
    }
    if (token == ids_.thinking) {
        open_body(Body::Reasoning, step.deltas);
        return step;
    }
    if (token == ids_.text || token == ids_.tool_error || token == ids_.invoke_tool_text) {
        open_body(Body::Content, step.deltas);
        return step;
    }
    if (token == ids_.invoke_tool_json) {
        open_body(Body::ToolCall, step.deltas);
        return step;
    }
    const DecodedTokenView decoded = tokenizer_->decoded_token(token);
    if (decoded.special) { return step; } // other control tokens carry no text
    switch (state_) {
    case State::Header:
        header_.append(decoded.bytes);
        break;
    case State::Between:
        append(step.deltas, OutputChannel::Content, text_.push_bytes(decoded.bytes));
        break;
    case State::Body:
        if (body_ == Body::ToolCall) {
            tool_body_.append(decoded.bytes);
        } else if (body_ == Body::Reasoning) {
            ++reasoning_tokens_;
            append(step.deltas, OutputChannel::Reasoning, text_.push_bytes(decoded.bytes));
        } else {
            append(step.deltas, OutputChannel::Content, text_.push_bytes(decoded.bytes));
        }
        break;
    }
    return step;
}

OutputTerminal OutputParser::finish(ToolCallRegionEnd end) {
    if (finished_) { throw std::logic_error("Inkling 2 output parser finished twice"); }
    finished_ = true;
    OutputTerminal terminal;
    close_body(terminal.deltas, false, end);
    append(terminal.deltas, OutputChannel::Content, text_.finish());
    terminal.tool_calls  = std::move(tool_calls_);
    terminal.diagnostics = diagnostics_;
    return terminal;
}

} // namespace ninfer::models::inkling2::frontend
