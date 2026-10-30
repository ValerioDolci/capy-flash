#include "serve/tool_call_retry.h"

#include <algorithm>

namespace ninfer::serve {
namespace {

std::string describe_first_function(std::string_view reply) {
    static constexpr std::string_view kOpen = "<function=";
    const std::size_t open                  = reply.find(kOpen);
    if (open == std::string_view::npos) { return "the tool"; }
    const std::size_t begin = open + kOpen.size();
    const std::size_t end   = reply.find('>', begin);
    if (end == std::string_view::npos || end == begin) { return "the tool"; }
    return "`" + std::string(reply.substr(begin, end - begin)) + "`";
}

} // namespace

std::string tool_call_retry_steering(std::string_view dead_reply) {
    // The dead reply is not kept in the continuation context and the steering is the model's own
    // voice, so it describes the failure without pointing at markup that is no longer there. Its
    // tags are named in words only: the continuation is a fresh output session whose region gate
    // fires only on a tool marker in the model's own stream, so the model must emit the whole
    // call. Teacher-forcing the open tags instead measured as a half-call decoded as plain
    // content (marker_seen=false), and keeping the dead reply in front of the steering measured
    // as an immediate stop token.
    return "\n\nMy tool call to " + describe_first_function(dead_reply) +
           " could not be delivered: a parameter value contained a literal call tag, and this "
           "format pairs those tags by depth with no way to escape one. I will call it again "
           "from the start, splitting any tag-shaped text across separate calls so no parameter "
           "value contains a literal tag.";
}

ToolCallRetryPlan plan_tool_call_retry(const ToolCallRetryInput& input,
                                       const ReasoningGuardTokens& tokens,
                                       std::span<const TokenId> steering) {
    ToolCallRetryPlan plan;
    // The dead-round signature is the gate; anything that produced a call is the client's to
    // answer, not ours to redo.
    if (!tool_call_region_is_dead(input.parse) || input.finish_reason != FinishReason::StopToken ||
        steering.empty() || tokens.close_suffix.empty()) {
        return plan;
    }
    // The dead reply is NOT kept in front of the steering: with the (complete-looking) reply in
    // context, the model answered the steering line with an immediate stop token. The steering
    // text quotes the dead call's tool name instead.
    const std::uint64_t injected = tokens.close_suffix.size() + steering.size();
    if (input.requested_output_tokens != 0) {
        // Room for the steering line plus at least one model token; the dead reply's tokens stay
        // billed to the request even though the reply is dropped.
        const std::uint64_t used = input.generated.size() + injected;
        if (used + 1U > input.requested_output_tokens) { return plan; }
        plan.remaining_output_tokens =
            static_cast<std::uint32_t>(input.requested_output_tokens - used);
    }
    plan.retry    = true;
    plan.appended = tokens.close_suffix;
    plan.appended.insert(plan.appended.end(), steering.begin(), steering.end());
    plan.reply_tokens    = static_cast<std::uint32_t>(input.generated.size());
    plan.injected_tokens = static_cast<std::uint32_t>(injected);
    return plan;
}

} // namespace ninfer::serve
