#pragma once

#include "text/byte_span.h"
#include "text/jinja.h"

#include <ninfer/types.h>

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::inkling2::frontend {

struct RenderedChat {
    std::string text;
    // Client-supplied text (message content, reasoning, tool names and arguments): encoded as
    // ordinary text even where it spells an added token.
    std::vector<text::ByteSpan> literal_spans;
};

// The checkpoint's chat_template.jinja, rendered with the engine's Jinja interpreter from the
// public prompt input. The template has no generation prompt: the rendering ends with the
// <|end_message|> of the last message and the model writes its own <|message_model|> header.
//
// Input mapping: Developer messages render as system messages; one text part renders as a string
// content, several text parts as a list (one <|content_text|> block each). The reasoning effort
// is PromptOptions::reasoning_effort (its name) or a "reasoning_effort" string or number in
// chat_template_kwargs_json; enable_thinking=false selects effort "none" (reasoning cannot be
// switched off, only minimised). Media parts and continuation of a final assistant message are
// rejected.
class CompiledChatTemplate {
public:
    [[nodiscard]] static CompiledChatTemplate
    resolve(std::string_view source, std::string source_name = "chat_template.jinja");
    [[nodiscard]] RenderedChat render(std::span<const ChatMessage> messages,
                                      const PromptOptions& options = {}) const;

private:
    explicit CompiledChatTemplate(text::JinjaTemplate compiled) : compiled_(std::move(compiled)) {}

    text::JinjaTemplate compiled_;
};

} // namespace ninfer::models::inkling2::frontend
