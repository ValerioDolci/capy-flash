// Inkling 2 Frontend behind the Engine contract: prompt preparation (ids and reusable-prefix
// frontiers against re-rendered prefixes, context limit, rejections), and output sessions driven
// by recorded token sequences: the real greedy turns of the model
// (tests/models/inkling2/greedy_fixture.json, tools/ref_ink/frontend_greedy_fixture.py), two tool
// calls in one turn, a truncated call, a budget cut inside a character, stop strings, the thinking
// budget, preview/discard/commit and continuation. Needs the files of an Inkling 2 Mini
// checkpoint in NINFER_TEST_INKLING2_TOKENIZER_DIR; skips (77) without them. CPU only.

#include "models/inkling2/frontend/frontend.h"
#include "models/inkling2/frontend/prepared_prompt.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace ink = ninfer::models::inkling2;
using Json    = nlohmann::ordered_json;
using ninfer::ChatMessage;
using ninfer::ChatRole;
using ninfer::FinishReason;
using ninfer::OutputChannel;
using ninfer::PromptInput;
using ninfer::RequestError;
using ninfer::RequestErrorKind;
using ninfer::TokenId;

constexpr TokenId kMessageModel = 200'001;
constexpr TokenId kContentText  = 200'004;
constexpr TokenId kEndSampling  = 200'006;
constexpr TokenId kThinking     = 200'008;
constexpr TokenId kEndMessage   = 200'010;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("cannot open " + path); }
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

bool valid_utf8(const std::string& text) {
    try {
        (void)nlohmann::json(text).dump();
        return true;
    } catch (const nlohmann::json::exception&) { return false; }
}

ChatRole parse_role(const std::string& role) {
    if (role == "system") { return ChatRole::System; }
    if (role == "user") { return ChatRole::User; }
    if (role == "assistant") { return ChatRole::Assistant; }
    if (role == "tool") { return ChatRole::Tool; }
    throw std::invalid_argument("fixture role " + role);
}

PromptInput prompt_input(const Json& item) {
    PromptInput input;
    for (const Json& source : item.at("messages")) {
        ChatMessage message;
        message.role = parse_role(source.at("role").get<std::string>());
        if (source.contains("content")) {
            const Json& content = source.at("content");
            if (content.is_string()) {
                message.parts.push_back({.text = content.get<std::string>()});
            } else {
                for (const Json& part : content) {
                    message.parts.push_back({.text = part.at("text").get<std::string>()});
                }
            }
        }
        message.reasoning_content = source.value("reasoning_content", std::string());
        message.tool_call_id      = source.value("tool_call_id", std::string());
        for (const Json& call : source.value("tool_calls", Json::array())) {
            message.tool_calls.push_back(
                {.id             = call.at("id").get<std::string>(),
                 .name           = call.at("function").at("name").get<std::string>(),
                 .arguments_json = call.at("function").at("arguments").dump()});
        }
        input.messages.push_back(std::move(message));
    }
    for (const Json& tool : item.value("tools", Json::array())) {
        input.options.tool_jsons.push_back(tool.dump());
    }
    if (item.contains("kwargs")) {
        input.options.chat_template_kwargs_json = item.at("kwargs").dump();
    } else if (item.contains("effort")) {
        input.options.chat_template_kwargs_json =
            Json{{"reasoning_effort", item.at("effort")}}.dump();
    }
    return input;
}

struct Turn {
    std::string reasoning;
    std::string content;
    std::vector<ninfer::GeneratedToolCall> calls;
    ninfer::ToolCallParseDiagnostics diagnostics;
    FinishReason finish  = FinishReason::None;
    std::size_t accepted = 0;
    bool utf8_ok         = true;
    std::optional<std::string> stop_string;
    ninfer::ThinkingBudgetStats thinking;
};

void collect(Turn& turn, ink::PublishedOutput output) {
    for (const auto& delta : output) {
        turn.utf8_ok = turn.utf8_ok && valid_utf8(delta.text) && !delta.text.empty();
        (delta.channel == OutputChannel::Reasoning ? turn.reasoning : turn.content) += delta.text;
    }
}

// Engine-style driving: one token per round, the remaining budget shrinking; a thinking control
// span is applied when the session asks for it; preview_terminal when the tokens run out first.
Turn drive(ink::OutputSession& session, const std::vector<TokenId>& tokens, std::uint32_t budget,
           const std::function<void(std::size_t)>& before_round = {}) {
    Turn turn;
    std::uint32_t remaining = budget;
    for (std::size_t i = 0; i < tokens.size() && turn.finish == FinishReason::None; ++i) {
        if (before_round) { before_round(i); }
        const auto decision = session.preview_model(std::span<const TokenId>(&tokens[i], 1),
                                                    remaining, FinishReason::OutputLimit);
        collect(turn, session.commit_preview());
        turn.accepted += decision.accepted_tokens;
        remaining -= decision.accepted_tokens;
        turn.finish = decision.finish_reason;
        if (decision.continuation == ninfer::runtime::ContinuationAction::ApplyTargetControl) {
            const auto control = session.pending_control_tokens();
            const std::vector<TokenId> span(control.begin(), control.end());
            const auto applied = session.preview_control(span, remaining);
            collect(turn, session.commit_preview());
            remaining -= applied.accepted_tokens;
        }
    }
    if (turn.finish == FinishReason::None) {
        turn.finish = session.preview_terminal(FinishReason::Cancelled).finish_reason;
        collect(turn, session.commit_preview());
    }
    turn.calls       = session.take_tool_calls();
    turn.diagnostics = session.tool_call_parse_diagnostics();
    turn.stop_string = session.matched_stop_string();
    turn.thinking    = session.thinking_stats();
    return turn;
}

// Reasoning and content of a decoded turn, read from its text form (independent of the parser).
std::pair<std::string, std::string> channels_of_text(const std::string& text) {
    std::string reasoning;
    std::string content;
    std::size_t pos = 0;
    while (true) {
        const std::size_t open = text.find("<|content_", pos);
        if (open == std::string::npos) { break; }
        const std::size_t open_end = text.find("|>", open) + 2;
        const std::string kind     = text.substr(open, open_end - open);
        std::size_t close          = text.find("<|", open_end);
        const std::string body =
            text.substr(open_end, (close == std::string::npos ? text.size() : close) - open_end);
        if (kind == "<|content_thinking|>") { reasoning += body; }
        if (kind == "<|content_text|>") { content += body; }
        if (close == std::string::npos) { break; }
        pos = close;
    }
    return {reasoning, content};
}

void preparation(const ink::Frontend& frontend, const Json& fixture) {
    std::size_t identical = 0;
    std::size_t frontiers = 0;
    for (const Json& item : fixture.at("conversations")) {
        const std::string name  = item.at("name").get<std::string>();
        const PromptInput input = prompt_input(item);
        const auto prompt       = frontend.prepare(input);
        const std::vector<TokenId> ids(prompt.token_ids().begin(), prompt.token_ids().end());
        if (name != "special_tokens_in_user_text") {
            check(ids == item.at("ids").get<std::vector<TokenId>>(), name + ": prepare ids");
            if (ids == item.at("ids").get<std::vector<TokenId>>()) { ++identical; }
        }
        check(frontend.count_tokens(input) == ids.size(), name + ": count_tokens");
        const auto summary = prompt.summary();
        check(summary.prompt_tokens == ids.size() && summary.starts_in_reasoning &&
                  !summary.has_media,
              name + ": summary");
        // Every frontier is the exact token prompt of the shorter conversation; every absent one
        // is not a prefix.
        const auto& layout = ink::PreparedPromptAccess::view(prompt).layout;
        check(layout.message_frontiers.size() == input.messages.size() + 1, name + ": frontiers");
        check(layout.message_frontiers.back() == ids.size(), name + ": last frontier");
        for (std::size_t n = 1; n < input.messages.size(); ++n) {
            PromptInput shorter = input;
            shorter.messages.resize(n);
            const auto prefix = frontend.prepare(shorter);
            const std::vector<TokenId> head(prefix.token_ids().begin(), prefix.token_ids().end());
            const bool is_prefix =
                head.size() <= ids.size() && std::equal(head.begin(), head.end(), ids.begin());
            const auto& frontier = layout.message_frontiers[n];
            check(frontier ? (is_prefix && *frontier == head.size()) : !is_prefix,
                  name + ": frontier after " + std::to_string(n) + " messages");
            ++frontiers;
        }
        if (layout.tool_frontier) {
            check(ids[*layout.tool_frontier - 1] == kEndMessage, name + ": tool frontier");
        }
    }
    std::cout << "prepare: " << identical << " conversations identical, " << frontiers
              << " prefix frontiers checked against re-rendering\n";
}

void greedy_turns(const ink::Frontend& frontend, const Json& greedy) {
    for (const Json& item : greedy.at("cases")) {
        const std::string name = item.at("name").get<std::string>();
        const auto prompt      = frontend.prepare(prompt_input(item));
        const std::vector<TokenId> ids(prompt.token_ids().begin(), prompt.token_ids().end());
        check(ids == item.at("prompt_ids").get<std::vector<TokenId>>(),
              name + ": prompt ids differ from the reference run");
        const auto generated = item.at("generated_ids").get<std::vector<TokenId>>();
        // The model opens its own message: header, then thinking.
        check(generated.size() >= 2 && generated[0] == kMessageModel && generated[1] == kThinking,
              name + ": first tokens are not <|message_model|><|content_thinking|>");
        auto session    = frontend.make_output_session(prompt, {});
        const Turn turn = drive(session, generated, 128);
        const bool eos  = item.at("stopped_on_eos").get<bool>();
        check(turn.finish == (eos ? FinishReason::StopToken : FinishReason::OutputLimit) &&
                  turn.accepted == generated.size(),
              name + ": finish reason / accepted tokens");
        check(turn.utf8_ok, name + ": a delta is empty or not UTF-8");
        const auto [reasoning, content] = channels_of_text(item.at("text").get<std::string>());
        check(turn.reasoning == reasoning, name + ": reasoning differs:\n" + turn.reasoning);
        check(turn.content == content, name + ": content differs:\n" + turn.content);
        // Reasoning tokens: the whole thinking phase, from <|message_model|> to the
        // <|end_message|> that closes the thinking message (or the end of the run).
        const auto close = std::find(generated.begin(), generated.end(), kEndMessage);
        const auto phase = static_cast<std::uint32_t>(
            close == generated.end() ? generated.size() : close - generated.begin() + 1);
        check(session.reasoning_tokens() == phase && turn.thinking.model_thinking_tokens == phase &&
                  !turn.thinking.applied,
              name + ": reasoning tokens " + std::to_string(session.reasoning_tokens()) +
                  " != thinking phase " + std::to_string(phase));
        check(prompt.summary().starts_in_reasoning, name + ": effort 0 prompt is not thinking");
        // Thinking budget on the real turn: after N tokens of the phase the three control tokens
        // close it and the reply goes on as text (a recorded answer stands in for the model).
        constexpr std::uint32_t kBudget = 10;
        if (phase > kBudget) {
            auto limited = frontend.make_output_session(prompt, {}, {}, {.budget = kBudget});
            std::vector<TokenId> replay(generated.begin(), generated.begin() + kBudget);
            const auto answer =
                frontend.tokenize_text("Risposta.<|end_message|><|content_model_end_sampling|>");
            replay.insert(replay.end(), answer.begin(), answer.end());
            const Turn cut = drive(limited, replay, 128);
            std::vector<TokenId> thinking(generated.begin() + 2, generated.begin() + kBudget);
            check(cut.finish == FinishReason::StopToken && cut.thinking.applied &&
                      cut.thinking.injected_tokens == 3 &&
                      cut.thinking.model_thinking_tokens == kBudget &&
                      limited.reasoning_tokens() == kBudget &&
                      cut.reasoning == frontend.detokenize_text(thinking) &&
                      cut.content == "Risposta.",
                  name + ": thinking budget on the real turn: [" + cut.reasoning + "|" +
                      cut.content + "]");
        }
        if (name == "tool1") {
            check(
                eos && generated.back() == kEndSampling &&
                    generated[generated.size() - 2] == kEndMessage,
                "tool1: the turn does not end with <|end_message|><|content_model_end_sampling|>");
            check(turn.calls.size() == 1 && turn.calls[0].name == "get_weather" &&
                      turn.calls[0].arguments_json == R"({"city":"Rome","unit":"celsius"})" &&
                      turn.content.empty(),
                  "tool1: tool call");
        }
    }
    std::cout << "greedy: " << greedy.at("cases").size() << " real turns replayed\n";
}

std::vector<TokenId> ids_of(const ink::Frontend& frontend, const std::string& text) {
    return frontend.tokenize_text(text);
}

void synthetic_turns(const ink::Frontend& frontend) {
    PromptInput input;
    input.messages.push_back({.role = ChatRole::User, .parts = {{.text = "ciao"}}});
    const auto prompt = frontend.prepare(input);

    // Two tool calls in one turn.
    {
        auto session    = frontend.make_output_session(prompt, {});
        const Turn turn = drive(
            session,
            ids_of(
                frontend,
                "<|message_model|><|content_thinking|>Two.<|end_message|>"
                "<|message_model|>a<|content_invoke_tool_json|>{\"name\":\"a\",\"args\":{\"x\":1}}"
                "<|end_message|><|message_model|>b<|content_invoke_tool_json|>"
                "{\"name\":\"b\",\"args\":{\"y\":[true,null]}}<|end_message|>"
                "<|content_model_end_sampling|>"),
            100);
        check(turn.finish == FinishReason::StopToken && turn.calls.size() == 2 &&
                  turn.calls[1].arguments_json == R"({"y":[true,null]})" &&
                  turn.reasoning == "Two." && turn.content.empty(),
              "two tool calls in one turn");
    }
    // A call cut by the output budget stays text.
    {
        const auto tokens = ids_of(frontend, "<|message_model|>a<|content_invoke_tool_json|>"
                                             "{\"name\":\"a\",\"args\":{\"x\":12345");
        auto session      = frontend.make_output_session(prompt, {});
        const Turn turn   = drive(session, tokens, static_cast<std::uint32_t>(tokens.size()));
        check(turn.finish == FinishReason::OutputLimit && turn.calls.empty() &&
                  turn.content == "{\"name\":\"a\",\"args\":{\"x\":12345" &&
                  turn.diagnostics.fallback_reason ==
                      ninfer::ToolCallParseFallbackReason::MalformedStructure,
              "truncated call is content: " + turn.content);
    }
    // The budget ends between the tokens of one character: U+FFFD, never a broken sequence.
    {
        std::vector<TokenId> tokens = {kMessageModel, kContentText};
        const auto text             = ids_of(frontend, "ok 𝔘𝔘");
        tokens.insert(tokens.end(), text.begin(), text.end());
        std::size_t cut = 0;
        for (std::size_t i = 2; i < tokens.size() && cut == 0; ++i) {
            std::vector<int> head(tokens.begin() + 2, tokens.begin() + static_cast<long>(i) + 1);
            try {
                (void)frontend.detokenize_text(head);
            } catch (const std::exception&) { cut = i + 1; }
        }
        check(cut != 0, "no token splits the character");
        auto session    = frontend.make_output_session(prompt, {});
        const Turn turn = drive(
            session, std::vector<TokenId>(tokens.begin(), tokens.begin() + static_cast<long>(cut)),
            static_cast<std::uint32_t>(cut));
        check(turn.finish == FinishReason::OutputLimit && turn.utf8_ok &&
                  turn.content.ends_with("\xef\xbf\xbd") && turn.content.starts_with("ok "),
              "cut inside a character: " + turn.content);
    }
    // Stop strings: cut before the match, across tokens and on the right channel.
    {
        ninfer::StopPolicy stop;
        stop.strings.push_back({.text = "world", .channel = OutputChannel::Content});
        stop.strings.push_back({.text = "never", .channel = OutputChannel::Reasoning});
        auto session = frontend.make_output_session(prompt, stop);
        const Turn turn =
            drive(session,
                  ids_of(frontend, "<|message_model|><|content_thinking|>say world<|end_message|>"
                                   "<|message_model|><|content_text|>Hello, world! More."
                                   "<|end_message|><|content_model_end_sampling|>"),
                  100);
        check(turn.finish == FinishReason::StopString && turn.content == "Hello, " &&
                  turn.reasoning == "say world" && turn.stop_string == "world",
              "stop string: [" + turn.reasoning + "|" + turn.content + "]");
    }
    // Thinking budget: after 3 model thinking tokens the model's own close is injected and the
    // reply continues as text.
    {
        auto session = frontend.make_output_session(prompt, {}, {}, {.budget = 3});
        check(session.control_suffix_tokens() == 3, "control suffix size");
        // The thinking phase is open from the first token: the header counts.
        check(session.model_token_budget_remaining(50) == 3, "budget before the first token");
        session.validate_generation_capacity(7);
        bool rejected = false;
        try {
            session.validate_generation_capacity(6);
        } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "capacity without room for the control suffix accepted");
        const auto tokens =
            ids_of(frontend, "<|message_model|><|content_thinking|>a b c d e f g h");
        const Turn turn = drive(session, tokens, 50);
        // Rounds stop at the budget; the control was applied inside drive().
        check(turn.thinking.applied && turn.thinking.injected_tokens == 3 &&
                  turn.thinking.model_thinking_tokens == 3,
              "thinking budget applied");
    }
    {
        auto session = frontend.make_output_session(prompt, {}, {}, {.budget = 4});
        std::vector<TokenId> tokens =
            ids_of(frontend, "<|message_model|><|content_thinking|>alpha beta gamma");
        std::vector<TokenId> rest =
            ids_of(frontend, "Ciao.<|end_message|><|content_model_end_sampling|>");
        Turn turn;
        std::uint32_t remaining = 64;
        std::size_t i           = 0;
        for (; i < tokens.size(); ++i) {
            const auto decision = session.preview_model(std::span<const TokenId>(&tokens[i], 1),
                                                        remaining, FinishReason::OutputLimit);
            collect(turn, session.commit_preview());
            remaining -= decision.accepted_tokens;
            if (decision.continuation == ninfer::runtime::ContinuationAction::ApplyTargetControl) {
                check(session.model_token_budget_remaining(remaining) == 0, "budget while pending");
                const auto control = session.pending_control_tokens();
                check(std::vector<TokenId>(control.begin(), control.end()) ==
                          std::vector<TokenId>{kEndMessage, kMessageModel, kContentText},
                      "control span is the model's own close");
                (void)session.preview_control(std::vector<TokenId>(control.begin(), control.end()),
                                              remaining);
                collect(turn, session.commit_preview());
                break;
            }
        }
        for (const TokenId token : rest) {
            const auto decision = session.preview_model(std::span<const TokenId>(&token, 1),
                                                        remaining, FinishReason::OutputLimit);
            collect(turn, session.commit_preview());
            if (decision.finished()) { turn.finish = decision.finish_reason; }
        }
        const auto stats = session.thinking_stats();
        check(turn.finish == FinishReason::StopToken && turn.content == "Ciao." &&
                  turn.reasoning == "alpha beta" && stats.applied &&
                  stats.model_thinking_tokens == 4 && stats.injected_tokens == 3 &&
                  session.reasoning_tokens() == 4,
              "after the thinking control the reply is text: [" + turn.reasoning + "|" +
                  turn.content + "]");
    }
    // Preview then discard leaves no trace.
    {
        auto session      = frontend.make_output_session(prompt, {});
        const auto tokens = ids_of(frontend, "<|message_model|><|content_text|>abc");
        Turn turn;
        for (const TokenId token : tokens) {
            (void)session.preview_model(std::span<const TokenId>(&token, 1), 10,
                                        FinishReason::OutputLimit);
            session.discard_preview();
            (void)session.preview_model(std::span<const TokenId>(&token, 1), 10,
                                        FinishReason::OutputLimit);
            collect(turn, session.commit_preview());
        }
        check(turn.content == "abc", "preview/discard/commit: " + turn.content);
    }
    // Continuation: the reply so far plus the close suffix; decoding resumes inside a text body.
    {
        auto appended    = ids_of(frontend, "<|message_model|><|content_thinking|>thinking");
        const auto guard = frontend.reasoning_guard_tokens();
        check(guard.close_suffix ==
                      std::vector<TokenId>{kEndMessage, kMessageModel, kContentText} &&
                  guard.think_close == std::vector<TokenId>{kEndMessage} &&
                  guard.stop_token_ids == std::vector<TokenId>{kEndSampling},
              "reasoning guard tokens");
        appended.insert(appended.end(), guard.close_suffix.begin(), guard.close_suffix.end());
        const auto resumed = frontend.prepare_continuation(input, appended);
        check(resumed.token_ids().size() == prompt.token_ids().size() + appended.size(),
              "continuation length");
        auto session = frontend.make_output_session(resumed, {});
        const Turn turn =
            drive(session, ids_of(frontend, "Hi<|end_message|><|content_model_end_sampling|>"), 10);
        check(turn.content == "Hi" && turn.reasoning.empty() &&
                  turn.finish == FinishReason::StopToken,
              "continuation resumes in text: " + turn.content);
    }
}

void contract(const ink::Frontend& frontend) {
    check(frontend.default_stop_policy().token_ids == std::vector<TokenId>{kEndSampling},
          "default stop policy");
    const auto& sampling = frontend.sampling_defaults();
    check(sampling.thinking.temperature == 1.0F && sampling.thinking.top_p == 1.0F &&
              sampling.thinking.top_k == 0 && sampling.non_thinking.temperature == 1.0F,
          "sampling defaults from generation_config.json");
    const auto greedy = ink::sampling_defaults_from(R"({"do_sample":false,"temperature":0.6})");
    check(greedy.thinking.temperature == 0.0F, "do_sample=false is greedy");
    check(frontend.detokenize_text(frontend.tokenize_text("città 👍")) == "città 👍",
          "text round trip");

    PromptInput input;
    input.messages.push_back({.role = ChatRole::User, .parts = {{.text = "ciao"}}});
    const auto prompt       = frontend.prepare(input);
    const auto expect_error = [&](const std::function<void()>& call, RequestErrorKind kind,
                                  const std::string& what) {
        try {
            call();
            check(false, what + " accepted");
        } catch (const RequestError& error) {
            check(error.kind() == kind, what + ": wrong error kind");
        }
    };
    expect_error(
        [&] {
            (void)frontend.make_output_session(prompt, {}, {}, {},
                                               ninfer::OutputConstraint::grammar("root ::= \"a\""));
        },
        RequestErrorKind::UnsupportedOutputCombination, "grammar");
    expect_error(
        [&] {
            (void)frontend.make_output_session(prompt, {}, {}, {}, {},
                                               {.mode = ninfer::ToolChoiceMode::Required});
        },
        RequestErrorKind::InvalidToolConstraint, "tool_choice required");
    std::string many_words;
    for (int i = 0; i < 5000; ++i) { many_words += " w" + std::to_string(i); }
    PromptInput longer;
    longer.messages.push_back({.role = ChatRole::User, .parts = {{.text = many_words}}});
    expect_error([&] { (void)frontend.prepare(longer); }, RequestErrorKind::ContextLengthExceeded,
                 "prompt over max_context");
    bool media = false;
    try {
        PromptInput image;
        image.messages.push_back(
            {.role = ChatRole::User, .parts = {{.kind = ninfer::MessagePartKind::Media}}});
        (void)frontend.prepare(image);
    } catch (const std::invalid_argument&) { media = true; }
    check(media, "media accepted");
    bool invalid = false;
    try {
        (void)frontend.prepare_tokens({1, 2, 999'999});
    } catch (const std::out_of_range&) { invalid = true; }
    check(invalid, "prepare_tokens accepted an id outside the vocabulary");
    const auto raw = frontend.prepare_tokens({1, 2, 3}, false);
    check(raw.token_ids().size() == 3 && !ink::PreparedPromptAccess::view(raw).identity.reusable,
          "prepare_tokens identity");
    check(!ink::PreparedPrompt() && frontend.media_cache_summary().entries == 0, "empty values");
}

} // namespace

int main() {
    const char* directory = std::getenv("NINFER_TEST_INKLING2_TOKENIZER_DIR");
    if (directory == nullptr || *directory == '\0') {
        std::cout << "skip: NINFER_TEST_INKLING2_TOKENIZER_DIR is not set\n";
        return 77;
    }
    try {
        const std::string dir                    = directory;
        const std::string tokenizer_json         = read_file(dir + "/tokenizer.json");
        const std::string tokenizer_config_json  = read_file(dir + "/tokenizer_config.json");
        const std::string chat_template          = read_file(dir + "/chat_template.jinja");
        const std::string generation_config_json = read_file(dir + "/generation_config.json");
        const ink::Frontend frontend =
            ink::make_frontend({.tokenizer_json         = tokenizer_json,
                                .tokenizer_config_json  = tokenizer_config_json,
                                .chat_template_jinja    = chat_template,
                                .generation_config_json = generation_config_json},
                               {.max_context = 4096});
        const std::string base = std::string(NINFER_SOURCE_DIR) + "/tests/models/inkling2/";
        contract(frontend);
        preparation(frontend, Json::parse(read_file(base + "tokenizer_fixture.json")));
        greedy_turns(frontend, Json::parse(read_file(base + "greedy_fixture.json")));
        synthetic_turns(frontend);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout
        << "inkling2 frontend contract: preparation, frontiers, real greedy turns, tool calls, "
           "truncation, split UTF-8, stop strings, thinking budget, preview, continuation "
           "passed\n";
    return 0;
}
