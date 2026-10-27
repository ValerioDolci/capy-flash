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

bool steering_bare_prefix_free(const std::string& text) {
    return text.find("<tool_call>") == std::string::npos;
}

} // namespace

int main() {
    int failures      = 0;
    const auto tokens = qwen_tokens();

    // A dead region ending in the model's terminal token is retried: reply tokens minus the
    // terminal, then the steering line (thinking was closed by the reply itself).
    {
        const std::vector<TokenId> generated = {248067, 101, 102, 248069, 103, 248046};
        const auto plan = plan_tool_call_retry(stopped(generated), tokens, kSteering);
        failures += check(plan.retry, "dead round was not retried");
        const std::vector<TokenId> expected = {248067, 101, 102, 248069, 103, 501, 502, 503};
        failures += check(plan.appended == expected, "appended tokens are not reply + steering");
        failures += check(plan.kept_tokens == 5U, "kept token count wrong");
        failures += check(plan.injected_tokens == 3U, "injected token count wrong");
    }

    // Thinking left open: the canonical close is injected before the steering line.
    {
        const std::vector<TokenId> generated = {101, 102, 248046};
        const auto plan = plan_tool_call_retry(stopped(generated), tokens, kSteering);
        failures += check(plan.retry, "open-thinking dead round was not retried");
        const std::vector<TokenId> expected = {101, 102, 198, 248069, 271, 501, 502, 503};
        failures +=
            check(plan.appended == expected, "open thinking was not closed before steering");
        failures +=
            check(plan.injected_tokens == 3U + 3U, "injected count misses the close suffix");
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
    // Cut by the output limit: the budget is spent, a continuation cannot fix it.
    {
        const std::vector<TokenId> generated = {101, 248046};
        auto input                           = stopped(generated);
        input.finish_reason                  = FinishReason::OutputLimit;
        failures += check(!plan_tool_call_retry(input, tokens, kSteering).retry,
                          "output-limit truncation was retried");
    }
    // A caller stop string ended the reply: the caller chose the cut, not the model.
    {
        const std::vector<TokenId> generated = {101, 777};
        failures += check(!plan_tool_call_retry(stopped(generated), tokens, kSteering).retry,
                          "caller-stopped reply was retried");
    }
    // Requested output budget: the continuation must still have room after the injected text.
    {
        const std::vector<TokenId> generated = {101, 102, 248046};
        auto input                           = stopped(generated);
        input.requested_output_tokens =
            2U + 3U + 3U; // kept + close suffix + steering, no model token left
        failures += check(!plan_tool_call_retry(input, tokens, kSteering).retry,
                          "retry allowed with no room for a model token");
        input.requested_output_tokens = 2U + 3U + 3U + 1U;
        const auto plan               = plan_tool_call_retry(input, tokens, kSteering);
        failures += check(plan.retry && plan.remaining_output_tokens == 1U,
                          "budget accounting wrong for the tightest retry");
    }

    // The steering text names the failed tool in a primed open prefix when the dead reply spelled
    // one out, and stays prose-only when it did not.
    {
        const std::string reply = "working...\n<function=write_note>\nvalue\n<function=other>\n";
        const auto name         = ninfer::serve::first_dead_function_name(reply);
        failures += check(name == "write_note", "first dead function name wrong");
        const auto steering = ninfer::serve::tool_call_retry_steering(name);
        failures += check(steering.ends_with("<tool_call>\n<function=write_note>\n"),
                          "steering does not end with the primed open call prefix");
        const auto bare = ninfer::serve::tool_call_retry_steering(
            ninfer::serve::first_dead_function_name("no call here"));
        failures += check(ninfer::serve::first_dead_function_name("no call here").empty() &&
                              steering_bare_prefix_free(bare),
                          "nameless steering still primes a call");
    }

    if (failures == 0) { std::cout << "tool_call_retry tests passed\n"; }
    return failures;
}
