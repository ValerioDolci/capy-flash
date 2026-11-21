#include "models/inkling2/frontend/chat_template.h"

#include <nlohmann/json.hpp>

#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::inkling2::frontend {
namespace {

using Json = nlohmann::ordered_json;

std::string_view role_name(ChatRole role) {
    switch (role) {
    case ChatRole::System:
    case ChatRole::Developer:
        return "system";
    case ChatRole::User:
        return "user";
    case ChatRole::Assistant:
        return "assistant";
    case ChatRole::Tool:
        return "tool";
    }
    throw std::invalid_argument("invalid chat role");
}

// The template renders the arguments of a past call as an object (top-level keys sorted, values
// compact in their given order).
Json history_tool_arguments(const ToolCall& call) {
    if (call.arguments_json.empty()) { return Json::object(); }
    Json arguments = Json::parse(call.arguments_json, nullptr, false);
    if (arguments.is_discarded() || !arguments.is_object()) {
        throw std::invalid_argument("tool call '" + call.name +
                                    "' in the history has arguments that are not a JSON object");
    }
    return arguments;
}

Json message_value(const ChatMessage& message) {
    Json value{{"role", role_name(message.role)}};
    for (const MessagePart& part : message.parts) {
        if (part.kind != MessagePartKind::Text) {
            throw std::invalid_argument("the Inkling 2 frontend does not accept media input yet");
        }
    }
    if (message.parts.size() <= 1) {
        value["content"] = message.parts.empty() ? std::string() : message.parts.front().text;
    } else {
        value["content"] = Json::array();
        for (const MessagePart& part : message.parts) {
            value["content"].push_back({{"type", "text"}, {"text", part.text}});
        }
    }
    if (!message.reasoning_content.empty()) {
        value["reasoning_content"] = message.reasoning_content;
    }
    if (!message.tool_calls.empty()) {
        value["tool_calls"] = Json::array();
        for (const ToolCall& call : message.tool_calls) {
            value["tool_calls"].push_back(
                {{"id", call.id},
                 {"type", "function"},
                 {"function", {{"name", call.name}, {"arguments", history_tool_arguments(call)}}}});
        }
    }
    if (!message.tool_call_id.empty()) { value["tool_call_id"] = message.tool_call_id; }
    return value;
}

void merge_option(Json& context, const std::string& name, const Json& value) {
    if (context.contains(name) && !context[name].is_null() && context[name] != value) {
        throw std::invalid_argument("conflicting chat template parameter: " + name);
    }
    context[name] = value;
}

Json template_parameters(const PromptOptions& options) {
    Json context = options.chat_template_kwargs_json.empty()
                       ? Json::object()
                       : Json::parse(options.chat_template_kwargs_json);
    if (!context.is_object()) {
        throw std::invalid_argument("chat_template_kwargs must be an object");
    }
    for (const auto& item : context.items()) {
        const auto& key = item.key();
        if (key == "messages" || key == "tools" || key == "add_generation_prompt" ||
            key == "continue_final_message" || key == "bos_token" || key == "eos_token" ||
            key == "pad_token") {
            throw std::invalid_argument("chat_template_kwargs cannot override " + key);
        }
    }
    for (const char* name : {"enable_thinking", "reasoning_effort"}) {
        if (context.contains(name) && context[name].is_null()) { context.erase(name); }
    }
    if (options.reasoning_effort) {
        const auto name = reasoning_effort_name(*options.reasoning_effort);
        if (name.empty()) { throw std::invalid_argument("invalid reasoning effort"); }
        merge_option(context, "reasoning_effort", std::string(name));
    }
    if (options.enable_thinking) {
        merge_option(context, "enable_thinking", *options.enable_thinking);
    }
    if (context.contains("enable_thinking")) {
        if (!context["enable_thinking"].is_boolean()) {
            throw std::invalid_argument("enable_thinking must be a boolean");
        }
        if (!context["enable_thinking"].get<bool>()) {
            if (context.contains("reasoning_effort") && context["reasoning_effort"] != "none" &&
                context["reasoning_effort"] != 0) {
                throw std::invalid_argument("reasoning_effort conflicts with enable_thinking");
            }
            context["reasoning_effort"] = "none";
        }
    }
    if (context.contains("reasoning_effort")) {
        const Json& effort = context["reasoning_effort"];
        if (effort.is_string()) {
            const auto name = effort.get<std::string>();
            if (name != "none" && name != "minimal" && name != "low" && name != "medium" &&
                name != "high" && name != "xhigh" && name != "max") {
                throw std::invalid_argument("invalid reasoning_effort template parameter");
            }
        } else if (!effort.is_number() || effort.get<double>() < 0.0 ||
                   effort.get<double>() > 0.99) {
            throw std::invalid_argument("reasoning_effort must be an effort name or a number in "
                                        "[0, 0.99]");
        }
    }
    return context;
}

} // namespace

CompiledChatTemplate CompiledChatTemplate::resolve(std::string_view source,
                                                   std::string source_name) {
    return CompiledChatTemplate(text::JinjaTemplate(std::string(source), std::move(source_name)));
}

RenderedChat CompiledChatTemplate::render(std::span<const ChatMessage> messages,
                                          const PromptOptions& options) const {
    if (messages.empty()) { throw std::invalid_argument("chat requires at least one message"); }
    if (options.continuation == PromptContinuationMode::ContinueFinalAssistant) {
        throw std::invalid_argument(
            "the Inkling 2 frontend cannot continue a final assistant message");
    }
    Json context                     = template_parameters(options);
    context["add_generation_prompt"] = true;
    context["messages"]              = Json::array();
    for (const ChatMessage& message : messages) {
        context["messages"].push_back(message_value(message));
    }
    if (!options.tool_jsons.empty()) {
        context["tools"] = Json::array();
        for (const std::string& tool : options.tool_jsons) {
            context["tools"].push_back(Json::parse(tool));
        }
    }
    text::TemplateOutput output = compiled_.render(context);
    return RenderedChat{.text          = std::move(output.text),
                        .literal_spans = std::move(output.literal_spans)};
}

} // namespace ninfer::models::inkling2::frontend
