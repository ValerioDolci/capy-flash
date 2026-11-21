#include "models/inkling2/frontend/prompt_layout.h"

#include <stdexcept>
#include <string>

namespace ninfer::models::inkling2::frontend {
namespace {

bool instruction(ChatRole role) { return role == ChatRole::System || role == ChatRole::Developer; }

int require(const Tokenizer& tokenizer, std::string_view content) {
    const auto id = tokenizer.added_token_id(content);
    if (!id) { throw std::invalid_argument("tokenizer lacks " + std::string(content)); }
    return *id;
}

} // namespace

LayoutTokens layout_tokens(const Tokenizer& tokenizer) {
    return LayoutTokens{.message_system = require(tokenizer, "<|message_system|>"),
                        .end_message    = require(tokenizer, "<|end_message|>"),
                        .end_sampling   = require(tokenizer, "<|content_model_end_sampling|>")};
}

PromptLayout inspect_prompt_layout(std::span<const int> ids, std::span<const ChatMessage> messages,
                                   bool has_tools, const LayoutTokens& tokens) {
    std::size_t pos = 0;
    // Index after the next occurrence of `token` at or after pos.
    const auto after_next = [&](int token) {
        for (std::size_t i = pos; i < ids.size(); ++i) {
            if (ids[i] == token) { return i + 1; }
        }
        throw std::logic_error("prompt tokens lack the block structure of the chat template");
    };
    const auto frontier = [](std::size_t value) { return static_cast<std::uint32_t>(value); };

    PromptLayout layout;
    layout.message_frontiers.assign(messages.size() + 1, std::nullopt);
    if (has_tools) {
        if (ids.empty() || ids[0] != tokens.message_system) {
            throw std::logic_error("prompt does not open with the tool declaration block");
        }
        pos                  = after_next(tokens.end_message);
        layout.tool_frontier = frontier(pos);
    }
    std::size_t first = 0;
    while (first < messages.size() && instruction(messages[first].role)) { ++first; }

    const auto take_effort = [&] {
        if (pos >= ids.size() || ids[pos] != tokens.message_system) {
            throw std::logic_error("prompt lacks the thinking-effort block");
        }
        pos                     = after_next(tokens.end_message);
        layout.leading_frontier = frontier(pos);
    };
    for (std::size_t index = 0; index <= messages.size(); ++index) {
        if (index == first) {
            take_effort();
            layout.message_frontiers[index] = frontier(pos);
        } else if (index > first) {
            layout.message_frontiers[index] = frontier(pos);
        }
        if (index == messages.size()) { break; }
        const ChatMessage& message = messages[index];
        if (message.role == ChatRole::Assistant) {
            pos = after_next(tokens.end_sampling);
        } else {
            const std::size_t blocks = message.parts.size() <= 1 ? 1 : message.parts.size();
            for (std::size_t block = 0; block < blocks; ++block) {
                pos = after_next(tokens.end_message);
            }
        }
    }
    if (pos != ids.size()) {
        throw std::logic_error("prompt tokens continue past the last chat template block");
    }
    return layout;
}

} // namespace ninfer::models::inkling2::frontend
