#include "serve/tool_call_retry.h"

#include <algorithm>

namespace ninfer::serve {
namespace {

bool contains(std::span<const TokenId> haystack, std::span<const TokenId> needle) {
    if (needle.empty() || needle.size() > haystack.size()) { return false; }
    return std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end()) !=
           haystack.end();
}

} // namespace

std::string tool_call_retry_steering(std::string_view function_name) {
    std::string text =
        "\n\nThe call above never reached the client: a parameter value contained a literal tag, "
        "tags pair by depth, and this format has no way to escape one. I will re-send the call; "
        "if a value must hold tag-shaped text I will split it across calls so no literal tag "
        "appears in any value.";
    if (!function_name.empty()) {
        text += "\n\n<tool_call>\n<function=";
        text.append(function_name);
        text += ">\n";
    }
    return text;
}

std::string first_dead_function_name(std::string_view reply) noexcept {
    static constexpr std::string_view kOpen = "<function=";
    const std::size_t open                  = reply.find(kOpen);
    if (open == std::string_view::npos) { return {}; }
    const std::size_t begin = open + kOpen.size();
    const std::size_t end   = reply.find('>', begin);
    if (end == std::string_view::npos || end == begin) { return {}; }
    return std::string(reply.substr(begin, end - begin));
}

ToolCallRetryPlan plan_tool_call_retry(const ToolCallRetryInput& input,
                                       const ReasoningGuardTokens& tokens,
                                       std::span<const TokenId> steering) {
    ToolCallRetryPlan plan;
    // The dead-round signature is the gate; anything that produced a call is the client's to
    // answer, not ours to redo.
    if (!tool_call_region_is_dead(input.parse) || input.finish_reason != FinishReason::StopToken ||
        input.generated.empty() || steering.empty() || tokens.close_suffix.empty() ||
        tokens.think_close.empty()) {
        return plan;
    }
    const TokenId last = input.generated.back();
    if (std::find(tokens.stop_token_ids.begin(), tokens.stop_token_ids.end(), last) ==
        tokens.stop_token_ids.end()) {
        return plan; // a caller stop token, not the model's end of turn
    }
    const std::span<const TokenId> kept = input.generated.first(input.generated.size() - 1U);
    const bool closed                   = contains(kept, tokens.think_close);
    // The dead markup stays in front of the steering line: the model wrote it, and re-sending from
    // that point tells it which call to redo instead of losing the intent with it.
    const std::uint64_t injected = (closed ? 0U : tokens.close_suffix.size()) + steering.size();
    if (input.requested_output_tokens != 0) {
        const std::uint64_t used = kept.size() + injected;
        // At least one model token must fit after the injected text.
        if (used + 1U > input.requested_output_tokens) { return plan; }
        plan.remaining_output_tokens =
            static_cast<std::uint32_t>(input.requested_output_tokens - used);
    }
    plan.retry = true;
    plan.appended.assign(kept.begin(), kept.end());
    if (!closed) {
        plan.appended.insert(plan.appended.end(), tokens.close_suffix.begin(),
                             tokens.close_suffix.end());
    }
    plan.appended.insert(plan.appended.end(), steering.begin(), steering.end());
    plan.kept_tokens     = static_cast<std::uint32_t>(kept.size());
    plan.injected_tokens = static_cast<std::uint32_t>(injected);
    return plan;
}

} // namespace ninfer::serve
