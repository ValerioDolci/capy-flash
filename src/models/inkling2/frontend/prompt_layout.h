#pragma once

#include "models/inkling2/frontend/tokenizer.h"

#include <ninfer/types.h>

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::inkling2::frontend {

// Token frontiers of a rendered chat prompt where a shorter conversation's prompt ends, i.e. the
// prefixes reusable across turns. The template renders, in order: the tool declaration block
// (when tools are declared), the messages, and the thinking-effort block right before the first
// non-system message (after all messages when there is none). Every block ends with
// <|end_message|>; an assistant message ends with <|content_model_end_sampling|>. Client text is
// encoded literally, so these control tokens appear only as block delimiters.
struct PromptLayout {
    // End of the tool declaration block.
    std::optional<std::uint32_t> tool_frontier;
    // End of the effort block: the shared preamble of tools, leading system messages and effort.
    std::uint32_t leading_frontier = 0;
    // Index n: the prompt of the first n messages (rendered alone) is exactly this token prefix;
    // absent when it is not a prefix (between leading system messages, whose shorter rendering
    // places the effort block earlier).
    std::vector<std::optional<std::uint32_t>> message_frontiers;
};

struct LayoutTokens {
    int message_system = -1;
    int end_message    = -1;
    int end_sampling   = -1;
};

[[nodiscard]] LayoutTokens layout_tokens(const Tokenizer& tokenizer);

// Throws std::logic_error when the ids do not have the block structure of `messages`.
[[nodiscard]] PromptLayout inspect_prompt_layout(std::span<const int> ids,
                                                 std::span<const ChatMessage> messages,
                                                 bool has_tools, const LayoutTokens& tokens);

} // namespace ninfer::models::inkling2::frontend
