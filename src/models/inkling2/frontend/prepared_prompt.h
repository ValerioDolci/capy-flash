#pragma once

#include "models/inkling2/frontend/frontend.h"
#include "models/inkling2/frontend/prompt_layout.h"

#include <string>
#include <vector>

namespace ninfer::models::inkling2 {

struct PromptIdentity {
    // False for raw token prompts that must not seed or match reusable prefixes.
    bool reusable = true;
};

struct PrepareStats {
    double seconds          = 0.0;
    double tokenize_seconds = 0.0;
};

struct PreparedPromptData {
    std::vector<TokenId> token_ids;
    PromptIdentity identity;
    // Reusable-prefix frontiers of a chat prompt (empty layout for token prompts).
    frontend::PromptLayout layout;
    ContextCacheHints context_cache;
    // Tool names declared but removed for this turn: calls naming them stay text.
    std::vector<std::string> filtered_out_tool_names;
    // Continuation: the tokens appended after the chat prompt, replayed into the output session
    // so decoding resumes in the message they leave open.
    std::vector<TokenId> resume_tokens;
    // A chat prompt: the model's turn starts with its thinking message (always, at any effort).
    bool starts_in_reasoning = false;
    bool has_active_tools    = false;
    bool continuation        = false;
    PrepareStats prepare;
};

class PreparedPromptAccess {
public:
    [[nodiscard]] static const PreparedPromptData& view(const PreparedPrompt& prompt);
    [[nodiscard]] static PreparedPromptData take(PreparedPrompt&& prompt);
};

} // namespace ninfer::models::inkling2
