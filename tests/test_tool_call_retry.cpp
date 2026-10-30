// plan_tool_call_retry (serve --tool-call-retry): when a dead tool-call round gets continued,
// with which tokens, and when it is left alone.
#include "serve/tool_call_retry.h"

#include <iostream>
#include <vector>

namespace {

using ninfer::FinishReason;
using ninfer::TokenId;
using ninfer::ToolCallParseDiagnostics;
using ninfer::ToolCallParseFallbackReason;
using ninfer::serve::plan_tool_call_retry;
using ninfer::serve::ToolCallRetryInput;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << "\n";
    return 1;
}

// Stop and think-close ids as in the Qwen3.8 tokenizer; the other ids are illustrative.
ninfer::ReasoningGuardTokens qwen_tokens() {
    ninfer::ReasoningGuardTokens tokens;
    tokens.close_suffix   = {198, 248069, 271};
    tokens.stop_token_ids = {248046, 248044};
    tokens.think_close    = {248069};
    return tokens;
}

ToolCallParseDiagnostics dead() {
    return ToolCallParseDiagnostics{.marker_seen           = true,
                                    .structured_call_count = 0,
                                    .fallback_reason =
                                        ToolCallParseFallbackReason::MalformedStructure};
}

ToolCallRetryInput stopped(const std::vector<TokenId>& generated,
                           const ToolCallParseDiagnostics& parse = dead()) {
    return ToolCallRetryInput{.parse                   = parse,
                              .finish_reason           = FinishReason::StopToken,
                              .generated               = generated,
                              .requested_output_tokens = 32000};
}

const std::vector<TokenId> kSteering = {501, 502, 503};

} // namespace

int main() {
    int failures      = 0;
    const auto tokens = qwen_tokens();

    // A dead region ending in the model's terminal token is retried: the canonical close and the
    // steering line are appended, the dead reply itself stays out of the continuation context.
    {
        const std::vector<TokenId> generated = {248067, 101, 102, 248069, 103, 248046};
        const auto plan = plan_tool_call_retry(stopped(generated), tokens, kSteering);
        failures += check(plan.retry, "dead round was not retried");
        const std::vector<TokenId> expected = {198, 248069, 271, 501, 502, 503};
        failures += check(plan.appended == expected, "appended tokens are not close + steering");
        failures += check(plan.reply_tokens == 6U, "dead reply token count wrong");
        failures += check(plan.injected_tokens == 3U + 3U, "injected token count wrong");
    }

    // The dead reply never re-enters the prompt: with it kept in context the model answered the
    // steering with an immediate stop token (measured).
    {
        const std::vector<TokenId> generated = {101, 102, 248046};
        const auto plan = plan_tool_call_retry(stopped(generated), tokens, kSteering);
        failures += check(plan.retry, "open-thinking dead round was not retried");
        const std::vector<TokenId> expected = {198, 248069, 271, 501, 502, 503};
        failures += check(plan.appended == expected, "the dead reply leaked into the append");
    }

    // A reply that produced a call (even salvaged or schema-mismatched) is the client's to answer.
    {
        auto parse                           = dead();
        parse.structured_call_count          = 1;
        const std::vector<TokenId> generated = {101, 248046};
        failures += check(!plan_tool_call_retry(stopped(generated, parse), tokens, kSteering).retry,
                          "round with a structured call was retried");
    }
    // No markup at all: nothing to redo.
    {
        auto parse                           = dead();
        parse.marker_seen                    = false;
        parse.fallback_reason                = ToolCallParseFallbackReason::None;
        const std::vector<TokenId> generated = {101, 248046};
        failures += check(!plan_tool_call_retry(stopped(generated, parse), tokens, kSteering).retry,
                          "reply without markup was retried");
    }
    // Cut by the output limit: the budget is spent, a continuation cannot fix it. A caller stop
    // string ends the reply with a different finish reason and is rejected the same way.
    {
        const std::vector<TokenId> generated = {101, 248046};
        auto input                           = stopped(generated);
        input.finish_reason                  = FinishReason::OutputLimit;
        failures += check(!plan_tool_call_retry(input, tokens, kSteering).retry,
                          "output-limit truncation was retried");
        input.finish_reason = FinishReason::StopString;
        failures += check(!plan_tool_call_retry(input, tokens, kSteering).retry,
                          "caller-stopped reply was retried");
    }
    // Requested output budget: the continuation must still have room after the injected text.
    {
        const std::vector<TokenId> generated = {101, 102, 248046};
        auto input                           = stopped(generated);
        input.requested_output_tokens =
            3U + 3U + 3U; // dead reply + close suffix + steering, no model token left
        failures += check(!plan_tool_call_retry(input, tokens, kSteering).retry,
                          "retry allowed with no room for a model token");
        input.requested_output_tokens = 3U + 3U + 3U + 1U;
        const auto plan               = plan_tool_call_retry(input, tokens, kSteering);
        failures += check(plan.retry && plan.remaining_output_tokens == 1U,
                          "budget accounting wrong for the tightest retry");
    }

    // The steering names the tool from the dead reply's first function tag and never spells a
    // wire tag itself: the model has to emit the whole call in its own stream.
    {
        const std::string reply = "working...\n<function=write_note>\nvalue\n<function=other>\n";
        const auto steering     = ninfer::serve::tool_call_retry_steering(reply);
        failures += check(steering.find("`write_note`") != std::string::npos,
                          "steering does not name the dead call's tool");
        failures += check(steering.find("<tool_call>") == std::string::npos &&
                              steering.find("<function=") == std::string::npos,
                          "steering spells a wire tag");
        const auto bare = ninfer::serve::tool_call_retry_steering("no call here");
        failures += check(bare.find("the tool") != std::string::npos,
                          "nameless steering has no fallback subject");
    }

    if (failures == 0) { std::cout << "tool_call_retry tests passed\n"; }
    return failures;
}
