// Inkling 2 frontend against the Hugging Face reference
// (tests/models/inkling2/tokenizer_fixture.json, written by tools/ref_ink/tokenizer_fixtures.py):
// identical token ids for texts and for rendered conversations, encode/decode round trip, streaming
// decode of UTF-8 split across tokens, and the output channels of generated turns (thinking, text,
// JSON tool calls, stop, truncated calls). Needs the tokenizer files of an Inkling 2 Mini
// checkpoint (tokenizer.json, tokenizer_config.json, generation_config.json, chat_template.jinja)
// in NINFER_TEST_INKLING2_TOKENIZER_DIR; skips (77) without them. CPU only.

#include "models/inkling2/frontend/chat_template.h"
#include "models/inkling2/frontend/output_parser.h"
#include "models/inkling2/frontend/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace ink = ninfer::models::inkling2::frontend;
using Json    = nlohmann::ordered_json;

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

std::string describe(const std::vector<int>& ids) {
    std::string text = "[";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) { text += ","; }
        text += std::to_string(ids[i]);
    }
    return text + "]";
}

std::string first_difference(const std::vector<int>& got, const std::vector<int>& want) {
    std::size_t i = 0;
    while (i < got.size() && i < want.size() && got[i] == want[i]) { ++i; }
    const auto window = [&](const std::vector<int>& ids) {
        const std::size_t begin = i < 3 ? 0 : i - 3;
        const std::size_t end   = std::min(ids.size(), i + 5);
        return describe(std::vector<int>(ids.begin() + static_cast<std::ptrdiff_t>(begin),
                                         ids.begin() + static_cast<std::ptrdiff_t>(end)));
    };
    return "first difference at " + std::to_string(i) + " (sizes " + std::to_string(got.size()) +
           "/" + std::to_string(want.size()) + "): got " + window(got) + " want " + window(want);
}

void texts(const ink::Tokenizer& tokenizer, const Json& fixture) {
    std::size_t passed = 0;
    for (const Json& item : fixture.at("texts")) {
        const std::string name = item.at("name").get<std::string>();
        const std::string text = item.at("text").get<std::string>();
        const auto want        = item.at("ids").get<std::vector<int>>();
        const auto got         = tokenizer.encode(text);
        check(got == want, "text " + name + ": " + first_difference(got, want));
        check(tokenizer.decode(got) == text, "text " + name + ": decode(encode(x)) != x");

        // Streaming: one token at a time, every piece complete UTF-8, the concatenation exact.
        ink::TokenStreamDecoder stream(tokenizer);
        std::string streamed;
        for (const int id : got) { streamed += stream.push(id); }
        streamed += stream.finish();
        check(streamed == text, "text " + name + ": streaming decode differs");
        if (got == want) { ++passed; }
    }
    std::cout << "texts: " << passed << "/" << fixture.at("texts").size() << " identical\n";
}

ninfer::ChatRole parse_role(const std::string& role) {
    if (role == "system") { return ninfer::ChatRole::System; }
    if (role == "user") { return ninfer::ChatRole::User; }
    if (role == "assistant") { return ninfer::ChatRole::Assistant; }
    if (role == "tool") { return ninfer::ChatRole::Tool; }
    throw std::invalid_argument("fixture role " + role);
}

std::optional<ninfer::ReasoningEffort> effort_by_name(const std::string& name) {
    using ninfer::ReasoningEffort;
    for (const ReasoningEffort effort :
         {ReasoningEffort::None, ReasoningEffort::Minimal, ReasoningEffort::Low,
          ReasoningEffort::Medium, ReasoningEffort::High, ReasoningEffort::XHigh,
          ReasoningEffort::Max}) {
        if (ninfer::reasoning_effort_name(effort) == name) { return effort; }
    }
    return std::nullopt;
}

// The fixture's OpenAI-style messages as the public prompt input the server builds.
std::vector<ninfer::ChatMessage> prompt_messages(const Json& messages) {
    std::vector<ninfer::ChatMessage> result;
    for (const Json& item : messages) {
        ninfer::ChatMessage message;
        message.role = parse_role(item.at("role").get<std::string>());
        if (item.contains("content")) {
            const Json& content = item.at("content");
            if (content.is_string()) {
                message.parts.push_back({.text = content.get<std::string>()});
            } else {
                for (const Json& part : content) {
                    message.parts.push_back({.text = part.at("text").get<std::string>()});
                }
            }
        }
        message.reasoning_content = item.value("reasoning_content", std::string());
        message.tool_call_id      = item.value("tool_call_id", std::string());
        for (const Json& call : item.value("tool_calls", Json::array())) {
            message.tool_calls.push_back(
                {.id             = call.at("id").get<std::string>(),
                 .name           = call.at("function").at("name").get<std::string>(),
                 .arguments_json = call.at("function").at("arguments").dump()});
        }
        result.push_back(std::move(message));
    }
    return result;
}

void conversations(const ink::Tokenizer& tokenizer, const ink::CompiledChatTemplate& chat,
                   const Json& fixture) {
    std::size_t passed = 0;
    for (const Json& item : fixture.at("conversations")) {
        const std::string name = item.at("name").get<std::string>();
        const auto messages    = prompt_messages(item.at("messages"));
        ninfer::PromptOptions options;
        for (const Json& tool : item.value("tools", Json::array())) {
            options.tool_jsons.push_back(tool.dump());
        }
        // A named effort goes through the typed option; a number through the template kwargs.
        std::optional<ninfer::PromptOptions> kwargs_options;
        if (item.contains("kwargs")) {
            const Json& effort                        = item.at("kwargs").at("reasoning_effort");
            kwargs_options                            = options;
            kwargs_options->chat_template_kwargs_json = item.at("kwargs").dump();
            if (effort.is_string()) {
                options.reasoning_effort = effort_by_name(effort.get<std::string>());
                check(options.reasoning_effort.has_value(), name + ": unknown effort name");
            } else {
                options = *kwargs_options;
                kwargs_options.reset();
            }
        }
        const ink::RenderedChat rendered = chat.render(messages, options);
        check(rendered.text == item.at("text").get<std::string>(),
              "conversation " + name + ": rendered text differs:\n" + rendered.text);
        if (kwargs_options) {
            check(chat.render(messages, *kwargs_options).text == rendered.text,
                  "conversation " + name + ": effort name through kwargs differs");
        }
        const auto got = tokenizer.encode(rendered.text, {}, rendered.literal_spans);
        auto want      = item.at("ids").get<std::vector<int>>();
        if (name == "special_tokens_in_user_text") {
            // Hugging Face turns added-token spellings inside user content into control tokens.
            // The engine keeps client text literal: the content is ordinary text, the rest equal.
            const std::string content = item.at("messages").at(0).at("content").get<std::string>();
            const std::size_t at      = rendered.text.find(content);
            check(at != std::string::npos, name + ": content not found");
            const std::string before = rendered.text.substr(0, at);
            const std::string after  = rendered.text.substr(at + content.size());
            std::vector<int> literal = tokenizer.encode(before);
            const auto middle        = tokenizer.encode(content, {.parse_added_tokens = false});
            const auto tail          = tokenizer.encode(after);
            literal.insert(literal.end(), middle.begin(), middle.end());
            literal.insert(literal.end(), tail.begin(), tail.end());
            check(want != literal, name + ": the reference no longer splits user specials");
            check(std::count(want.begin(), want.end(), 200'010) == 3 &&
                      std::count(got.begin(), got.end(), 200'010) == 2,
                  name + ": <|end_message|> counts");
            want = literal;
        }
        check(got == want, "conversation " + name + ": " + first_difference(got, want));
        if (got == want && rendered.text == item.at("text").get<std::string>()) { ++passed; }
    }
    std::cout << "conversations: " << passed << "/" << fixture.at("conversations").size()
              << " identical\n";
}

void template_rejections(const ink::CompiledChatTemplate& chat) {
    const auto rejects = [&](const std::vector<ninfer::ChatMessage>& messages,
                             const ninfer::PromptOptions& options, const std::string& what) {
        try {
            (void)chat.render(messages, options);
            check(false, "accepted " + what);
        } catch (const std::exception&) {}
    };
    const std::vector<ninfer::ChatMessage> user{
        {.role = ninfer::ChatRole::User, .parts = {{.text = "hi"}}}};
    rejects({}, {}, "no messages");
    rejects(user, {.chat_template_kwargs_json = R"({"reasoning_effort":1.5})"}, "effort 1.5");
    rejects(user, {.chat_template_kwargs_json = R"({"reasoning_effort":"huge"})"}, "effort name");
    rejects(user, {.enable_thinking = false, .reasoning_effort = ninfer::ReasoningEffort::High},
            "thinking off with high effort");
    rejects(user, {.continuation = ninfer::PromptContinuationMode::ContinueFinalAssistant},
            "continuation");
    // Thinking off is the minimum effort, and Developer is a system message.
    const auto off = chat.render(user, {.enable_thinking = false}).text;
    check(off.find("Thinking effort level: 0<|end_message|>") != std::string::npos,
          "enable_thinking=false is not effort 0: " + off);
    const std::vector<ninfer::ChatMessage> developer{
        {.role = ninfer::ChatRole::Developer, .parts = {{.text = "rules"}}},
        {.role = ninfer::ChatRole::User, .parts = {{.text = "hi"}}}};
    check(chat.render(developer).text.starts_with("<|message_system|><|content_text|>rules"),
          "developer message is not a system message");
}

// Token ids of a generated turn written as text with control tokens.
std::vector<int> turn(const ink::Tokenizer& tokenizer, const std::string& text) {
    return tokenizer.encode(text);
}

struct Collected {
    std::string reasoning;
    std::string content;
    std::vector<ninfer::GeneratedToolCall> calls;
    ninfer::ToolCallParseDiagnostics diagnostics;
    bool stopped           = false;
    std::size_t stop_index = 0;
    bool split_utf8_ok     = true;
};

Collected run(const ink::Tokenizer& tokenizer, const std::vector<int>& ids,
              ink::ToolCallRegionEnd end = ink::ToolCallRegionEnd::ModelStop) {
    ink::OutputParser parser(tokenizer, {"blocked"});
    Collected out;
    const auto take = [&](const std::vector<ninfer::OutputDelta>& deltas) {
        for (const auto& delta : deltas) {
            // Every published piece is complete UTF-8 on its own.
            ink::TokenStreamDecoder check_utf8(tokenizer);
            if (check_utf8.push_bytes(delta.text) != delta.text || check_utf8.pending()) {
                out.split_utf8_ok = false;
            }
            (delta.channel == ninfer::OutputChannel::Reasoning ? out.reasoning : out.content) +=
                delta.text;
        }
    };
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const auto step = parser.push(ids[i]);
        take(step.deltas);
        if (step.stop) {
            out.stopped    = true;
            out.stop_index = i;
            break;
        }
    }
    auto terminal = parser.finish(end);
    take(terminal.deltas);
    out.calls       = std::move(terminal.tool_calls);
    out.diagnostics = terminal.diagnostics;
    return out;
}

void output_channels(const ink::Tokenizer& tokenizer, const ink::CompiledChatTemplate& chat) {
    using ninfer::ToolCallParseFallbackReason;
    // Thinking in two messages, then text, then the end-of-turn token.
    const std::string plain =
        "<|message_model|><|content_thinking|>Penso: città, 👍.<|end_message|>"
        "<|message_model|><|content_thinking|> Poi basta.<|end_message|>"
        "<|message_model|><|content_text|>Risposta: 42 — più o meno.<|end_message|>"
        "<|content_model_end_sampling|>ignored";
    auto out = run(tokenizer, turn(tokenizer, plain));
    check(out.reasoning == "Penso: città, 👍. Poi basta.", "reasoning: " + out.reasoning);
    check(out.content == "Risposta: 42 — più o meno.", "content: " + out.content);
    check(out.stopped && out.calls.empty() && out.split_utf8_ok, "plain turn stop/UTF-8");

    // Two calls in one turn, the second with nested arguments.
    const std::string calls =
        "<|message_model|><|content_thinking|>Two lookups.<|end_message|>"
        "<|message_model|>get_weather<|content_invoke_tool_json|>"
        R"({"name":"get_weather","args":{"city":"Roma","unit":"celsius"}})"
        "<|end_message|>"
        "<|message_model|>search_docs<|content_invoke_tool_json|>"
        R"({"name":"search_docs","args":{"query":"«BPE»","filters":{"lang":"en"},"limit":5}})"
        "<|end_message|><|content_model_end_sampling|>";
    out = run(tokenizer, turn(tokenizer, calls));
    check(out.calls.size() == 2 && out.calls[0].name == "get_weather" &&
              out.calls[0].arguments_json == R"({"city":"Roma","unit":"celsius"})" &&
              out.calls[1].name == "search_docs" &&
              out.calls[1].arguments_json ==
                  R"({"query":"«BPE»","filters":{"lang":"en"},"limit":5})",
          "two tool calls");
    check(out.content.empty() && out.reasoning == "Two lookups." &&
              out.diagnostics.structured_call_count == 2 &&
              out.diagnostics.fallback_reason == ToolCallParseFallbackReason::None,
          "tool call turn channels");

    // A call cut by the token budget stays text; one ended by the model's stop is completed.
    const std::string open_call = "<|message_model|>get_weather<|content_invoke_tool_json|>"
                                  R"({"name":"get_weather","args":{"city":"Ro)";
    out = run(tokenizer, turn(tokenizer, open_call), ink::ToolCallRegionEnd::Cut);
    check(out.calls.empty() && out.content == R"({"name":"get_weather","args":{"city":"Ro)" &&
              out.diagnostics.marker_seen &&
              out.diagnostics.fallback_reason == ToolCallParseFallbackReason::MalformedStructure,
          "cut call is content: " + out.content);
    const std::string closed_by_stop = "<|message_model|>get_weather<|content_invoke_tool_json|>"
                                       R"({"name":"get_weather","args":{}})"
                                       "<|content_model_end_sampling|>";
    out                              = run(tokenizer, turn(tokenizer, closed_by_stop));
    check(out.calls.size() == 1 && out.calls[0].arguments_json == "{}" &&
              out.diagnostics.structural_salvage_count == 1,
          "call closed by the end-of-turn token");
    // Malformed, invalid name, filtered tool: the body comes back as content.
    for (const auto& [body, reason] :
         std::vector<std::pair<std::string, ToolCallParseFallbackReason>>{
             {R"({"name":"x","args":[1]})", ToolCallParseFallbackReason::MalformedStructure},
             {R"({"name":"bad name","args":{}})", ToolCallParseFallbackReason::InvalidToolName},
             {R"({"name":"blocked","args":{}})", ToolCallParseFallbackReason::FilteredTool}}) {
        out =
            run(tokenizer, turn(tokenizer, "<|message_model|>x<|content_invoke_tool_json|>" + body +
                                               "<|end_message|><|content_model_end_sampling|>"));
        check(out.calls.empty() && out.content == body && out.diagnostics.fallback_reason == reason,
              "fallback for " + body);
    }
    // The header of a message the model does not write ends the turn.
    out = run(tokenizer, turn(tokenizer, "<|message_model|><|content_text|>ok<|end_message|>"
                                         "<|message_user|><|content_text|>no"));
    check(out.stopped && out.stop_index == 4 && out.content == "ok", "user header stops");

    // History round trip: the assistant turn rendered by the template parses back to the same
    // reasoning and calls, so a generated turn and its re-rendered history are the same tokens.
    const std::vector<ninfer::ChatMessage> prompt{
        {.role = ninfer::ChatRole::User, .parts = {{.text = "Meteo a Roma e Oslo?"}}}};
    auto history = prompt;
    history.push_back(
        {.role              = ninfer::ChatRole::Assistant,
         .parts             = {{.text = ""}},
         .reasoning_content = "Due chiamate.",
         .tool_calls = {{.id = "a", .name = "get_weather", .arguments_json = R"({"city":"Roma"})"},
                        {.id             = "b",
                         .name           = "get_weather",
                         .arguments_json = R"({"unit":"celsius","city":"Oslo"})"}}});
    const auto before = chat.render(prompt);
    const auto after  = chat.render(history);
    check(after.text.starts_with(before.text), "assistant turn does not extend the prompt");
    const auto prompt_ids = tokenizer.encode(before.text, {}, before.literal_spans);
    const auto all_ids    = tokenizer.encode(after.text, {}, after.literal_spans);
    const std::vector<int> generated(
        all_ids.begin() + static_cast<std::ptrdiff_t>(prompt_ids.size()), all_ids.end());
    out = run(tokenizer, generated);
    check(out.stopped && out.stop_index + 1 == generated.size() &&
              out.reasoning == "Due chiamate." && out.content.empty() && out.calls.size() == 2 &&
              out.calls[1].arguments_json == R"({"city":"Oslo","unit":"celsius"})",
          "history round trip");
}

void special_tokens(const ink::Tokenizer& tokenizer) {
    check(tokenizer.vocab_size() == 200'058, "vocab size is not 200058");
    check(tokenizer.has_exact_token_domain(200'058), "token domain has holes");
    check(tokenizer.default_stop_token_ids() == std::vector<int>{200'006},
          "default stop is not <|content_model_end_sampling|>");
    check(tokenizer.added_token_id("<|end_message|>") == 200'010, "<|end_message|> id");
    check(tokenizer.added_token_id("<|content_thinking|>") == 200'008, "<|content_thinking|> id");
    check(tokenizer.added_token_id("<|content_invoke_tool_json|>") == 200'049,
          "<|content_invoke_tool_json|> id");
    check(tokenizer.added_token_id("<|begin_of_text|>") == 200'028, "<|begin_of_text|> id");
    check(tokenizer.is_special_token(200'006) && !tokenizer.is_special_token(1000),
          "special token flags");

    // Literal spans: the same bytes as ordinary text, encoded together with their neighbours.
    const std::string text = "a <|end_message|> b";
    const std::vector<ninfer::text::ByteSpan> literal{{.begin = 0, .end = text.size()}};
    check(tokenizer.encode(text, {}, literal) ==
              tokenizer.encode(text, {.parse_added_tokens = false}),
          "literal span is not ordinary text");
    check(tokenizer.encode(text).size() == 4,
          "added token not split out: " + describe(tokenizer.encode(text)));
    check(tokenizer.decode(tokenizer.encode(text), {.skip_special_tokens = true}) == "a  b",
          "skip_special_tokens");
    check(tokenizer.encode(text, {.max_tokens = 2}).size() == 2, "max_tokens");
}

void streaming_split_utf8(const ink::Tokenizer& tokenizer) {
    // A 4-byte emoji fed one byte at a time: nothing until the last byte, then the character.
    const std::string emoji = "\xf0\x9f\x91\x8d";
    ink::TokenStreamDecoder stream(tokenizer);
    for (std::size_t i = 0; i + 1 < emoji.size(); ++i) {
        check(stream.push_bytes(emoji.substr(i, 1)).empty(), "partial UTF-8 published");
    }
    check(stream.push_bytes(emoji.substr(3)) == emoji, "completed UTF-8 not published");
    // A cut sequence at the end becomes one replacement character.
    check(stream.push_bytes("x\xe2\x82").empty() == false, "ASCII before a cut not published");
    check(stream.finish() == "\xef\xbf\xbd", "incomplete tail is not U+FFFD");
    // Single-byte tokens that split one character, decoded through token ids.
    std::vector<int> bytes;
    for (const unsigned char byte : std::string("€")) {
        for (int id = 0; id < static_cast<int>(tokenizer.vocab_size()); ++id) {
            if (tokenizer.decode_token_bytes(id) == std::string(1, static_cast<char>(byte))) {
                bytes.push_back(id);
                break;
            }
        }
    }
    check(bytes.size() == 3, "single-byte tokens of the euro sign");
    ink::TokenStreamDecoder split(tokenizer);
    std::string out;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        const std::string piece = split.push(bytes[i]);
        check(i + 1 == bytes.size() || piece.empty(), "partial euro published");
        out += piece;
    }
    check(out == "€" && split.finish().empty(), "euro from byte tokens");
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
        const std::string generation_config_json = read_file(dir + "/generation_config.json");
        const auto chat =
            ink::CompiledChatTemplate::resolve(read_file(dir + "/chat_template.jinja"));
        const ink::Tokenizer tokenizer(
            ink::TokenizerResources{.tokenizer_json         = tokenizer_json,
                                    .tokenizer_config_json  = tokenizer_config_json,
                                    .generation_config_json = generation_config_json});
        const Json fixture = Json::parse(read_file(
            std::string(NINFER_SOURCE_DIR) + "/tests/models/inkling2/tokenizer_fixture.json"));
        special_tokens(tokenizer);
        texts(tokenizer, fixture);
        conversations(tokenizer, chat, fixture);
        template_rejections(chat);
        output_channels(tokenizer, chat);
        streaming_split_utf8(tokenizer);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "inkling2 frontend: fixture ids of texts and conversations, round trip, streaming "
                 "decode, template options, output channels passed\n";
    return 0;
}
