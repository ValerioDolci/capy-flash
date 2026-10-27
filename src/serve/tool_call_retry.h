#pragma once

// Tool-call retry (--tool-call-retry, on with --profile flash-next). The Qwen tool-call wire format
// pairs parameter tags by depth and has no escaping: a reply whose parameter value contains a
// literal parameter tag - the model writing tool markup as file content, or an extra open tag -
// fails the strict parse as malformed_structure with zero structured calls, and the whole marker
// region is returned as ordinary content with end_turn: the agent loop dies silently and the client
// has nothing to react to (measured: 44 such dead rounds in the production request log, 43 of them
// text-only with thinking open, i.e. continuable). When that happens the service appends the reply
// so far, the canonical thinking close when the reply left thinking open, and steering that ends
// with the open tags of a fresh call to the same tool, and lets the
// model continue the call once, from that point, with a different seed. Nothing changes for a reply
// that does not end this way: the first generation runs exactly as without the option.

#include "ninfer/types.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// One server-side retry per request; the retry's own reply stands as it is even if it dies the
// same way.
constexpr int kToolCallRetryLimit = 1;

// The dead-round signature: markup was emitted and parsed to nothing. Anything that produced a
// call, salvaged or not, is left alone.
[[nodiscard]] constexpr bool
tool_call_region_is_dead(const ninfer::ToolCallParseDiagnostics& parse) noexcept {
    return parse.marker_seen && parse.structured_call_count == 0U &&
           parse.fallback_reason != ninfer::ToolCallParseFallbackReason::None;
}

struct ToolCallRetryInput {
    ninfer::ToolCallParseDiagnostics parse;
    FinishReason finish_reason = FinishReason::None;
    std::span<const TokenId> generated; // the reply's token ids, terminal token included
    // Request output budget (0 = the server default, not bounded here).
    std::uint32_t requested_output_tokens = 0;
};

struct ToolCallRetryPlan {
    bool retry = false;
    // Tokens appended to the chat prompt: the reply without its terminal token, the thinking close
    // if the reply left thinking open, then the steering line.
    std::vector<TokenId> appended;
    std::uint32_t kept_tokens     = 0; // reply tokens kept (terminal token dropped)
    std::uint32_t injected_tokens = 0; // thinking close + steering line tokens
    // Output budget left for the continuation when the request set one.
    std::optional<std::uint32_t> remaining_output_tokens;
};

// Decision and token plan for the retry: pure so it can be tested without a model. A marker region
// that produced no structured call at all is the dead-round signature; anything that produced a
// call, salvaged or not, is left alone.
[[nodiscard]] ToolCallRetryPlan plan_tool_call_retry(const ToolCallRetryInput& input,
                                                     const ReasoningGuardTokens& tokens,
                                                     std::span<const TokenId> steering);

// Steering appended after the dead reply, in the model's own voice. Prose alone measured an
// immediate stop token (the assistant turn already looked finished); when the dead reply spelled
// out a call, the steering ends by teacher-forcing the opening tags of a fresh call to the first
// named tool, so the continuation goes through the parameters. Tag names in prose are words; the
// wire tag appears only in the primed prefix.
[[nodiscard]] std::string tool_call_retry_steering(std::string_view function_name);

// The first function name spelled out in a dead reply, empty when none is complete.
[[nodiscard]] std::string first_dead_function_name(std::string_view reply) noexcept;

} // namespace ninfer::serve
