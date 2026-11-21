#include "models/inkling2/frontend/frontend.h"

#include "models/inkling2/frontend/chat_template.h"
#include "models/inkling2/frontend/prepared_prompt.h"
#include "models/inkling2/frontend/prompt_layout.h"
#include "models/inkling2/frontend/tokenizer.h"
#include "text/unicode.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::inkling2 {
namespace {

namespace fi = frontend;
using Clock  = std::chrono::steady_clock;

// The model's own continuation after a thinking message: close it and open a text message.
constexpr std::string_view kThinkingControl = "<|end_message|><|message_model|><|content_text|>";
constexpr std::string_view kThinkingClose   = "<|end_message|>";

void check_preparation_control(const PreparationControl& control,
                               std::string_view stage = "preparation") {
    if (control.cancellation.requested()) {
        throw RequestError(RequestErrorKind::Cancelled,
                           "inference request was cancelled during " + std::string(stage));
    }
    if (control.deadline != Clock::time_point{} && Clock::now() >= control.deadline) {
        throw RequestError(RequestErrorKind::QueueTimeout,
                           "inference request expired during " + std::string(stage));
    }
}

[[noreturn]] void throw_context_length_exceeded(std::size_t prompt_tokens,
                                                std::uint32_t max_context, bool at_least = false) {
    throw RequestError(RequestErrorKind::ContextLengthExceeded,
                       context_length_message(prompt_tokens, max_context, at_least));
}

std::uint32_t checked_token_count(std::size_t count) {
    if (count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("prompt token count exceeds uint32_t");
    }
    return static_cast<std::uint32_t>(count);
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::invalid_argument("cannot read chat template " + path.string()); }
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

StopPolicy merge_stop_policy(const fi::Tokenizer& tokenizer, const StopPolicy& caller) {
    StopPolicy result;
    result.publish_stop_token = caller.publish_stop_token;
    const auto append_token   = [&](TokenId token) {
        if (!tokenizer.is_valid_token(token)) {
            throw std::invalid_argument("stop token id is outside the checkpoint vocabulary: " +
                                          std::to_string(token));
        }
        if (std::find(result.token_ids.begin(), result.token_ids.end(), token) ==
            result.token_ids.end()) {
            result.token_ids.push_back(token);
        }
    };
    if (caller.include_model_defaults) {
        for (const int token : tokenizer.default_stop_token_ids()) { append_token(token); }
    }
    for (const TokenId token : caller.token_ids) { append_token(token); }
    for (const StopString& stop : caller.strings) {
        if (stop.text.empty()) { throw std::invalid_argument("stop string must not be empty"); }
        (void)text::unicode_internal::utf8_codepoints(stop.text, "stop string");
        const auto duplicate =
            std::find_if(result.strings.begin(), result.strings.end(), [&](const StopString& s) {
                return s.text == stop.text && s.channel == stop.channel &&
                       s.include_in_output == stop.include_in_output;
            });
        if (duplicate == result.strings.end()) { result.strings.push_back(stop); }
    }
    return result;
}

void validate_tool_choice(const ToolChoice& tool_choice) {
    if (tool_choice.mode == ToolChoiceMode::Required || tool_choice.allowed_names ||
        !tool_choice.parallel || tool_choice.constraints != ToolConstraintMode::Automatic) {
        throw RequestError(RequestErrorKind::InvalidToolConstraint,
                           "tool_choice selection, cardinality and strict tool constraints are not "
                           "supported for Inkling 2",
                           {}, RequestErrorSource::Tools);
    }
}

} // namespace

ModelSamplingDefaults sampling_defaults_from(std::string_view generation_config_json) {
    nlohmann::json config;
    try {
        config = nlohmann::json::parse(generation_config_json);
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument(std::string("malformed generation_config.json: ") +
                                    error.what());
    }
    if (!config.is_object()) {
        throw std::invalid_argument("generation_config.json must be an object");
    }
    const auto number = [&](const char* key, double fallback) {
        if (!config.contains(key) || config[key].is_null()) { return fallback; }
        if (!config[key].is_number()) {
            throw std::invalid_argument(std::string("generation_config.json ") + key +
                                        " must be a number");
        }
        return config[key].get<double>();
    };
    const bool sample = config.value("do_sample", true);
    SamplingPreset preset;
    preset.temperature = sample ? static_cast<float>(number("temperature", 1.0)) : 0.0F;
    preset.top_p       = static_cast<float>(number("top_p", 1.0));
    preset.top_k       = static_cast<std::int32_t>(number("top_k", 0.0));
    preset.min_p       = static_cast<float>(number("min_p", 0.0));
    return ModelSamplingDefaults{.thinking = preset, .non_thinking = preset};
}

class Frontend::Impl {
public:
    Impl(const FrontendResources& resources, const FrontendOptions& options)
        : chat_template(fi::CompiledChatTemplate::resolve(
              options.chat_template_path.empty() ? std::string(resources.chat_template_jinja)
                                                 : read_text_file(options.chat_template_path),
              options.chat_template_path.empty() ? "chat_template.jinja"
                                                 : options.chat_template_path.string())),
          tokenizer(std::make_shared<const fi::Tokenizer>(
              fi::TokenizerResources{.tokenizer_json         = resources.tokenizer_json,
                                     .tokenizer_config_json  = resources.tokenizer_config_json,
                                     .generation_config_json = resources.generation_config_json})),
          layout(fi::layout_tokens(*tokenizer)),
          sampling(sampling_defaults_from(resources.generation_config_json)),
          max_context(options.max_context) {
        if (options.max_context == 0) {
            throw std::invalid_argument("frontend max_context must be nonzero");
        }
        for (const int token : tokenizer->default_stop_token_ids()) {
            if (!tokenizer->is_valid_token(token)) {
                throw std::invalid_argument(
                    "generation_config.json contains a stop token outside the vocabulary");
            }
            defaults.token_ids.push_back(token);
        }
        auto control = tokenizer->encode(kThinkingControl);
        if (control.size() != 3 || std::any_of(control.begin(), control.end(), [&](int token) {
                return !tokenizer->is_special_token(token) ||
                       std::find(defaults.token_ids.begin(), defaults.token_ids.end(), token) !=
                           defaults.token_ids.end();
            })) {
            throw std::invalid_argument(
                "Inkling 2 tokenizer cannot encode the thinking control as three control tokens");
        }
        thinking_control = std::make_shared<const std::vector<TokenId>>(std::move(control));
    }

    fi::CompiledChatTemplate chat_template;
    std::shared_ptr<const fi::Tokenizer> tokenizer;
    fi::LayoutTokens layout;
    StopPolicy defaults;
    ModelSamplingDefaults sampling;
    std::shared_ptr<const std::vector<TokenId>> thinking_control;
    std::uint32_t max_context = 0;
};

PreparedPrompt::PreparedPrompt() noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept
    : data_(std::move(data)) {}

PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PromptSummary PreparedPrompt::summary() const {
    if (data_ == nullptr) { throw std::logic_error("prepared prompt is empty"); }
    // A chat prompt ends after a complete message and the model always opens its turn with a
    // thinking message: reasoning is on from the first generated token (the effort only sets
    // its length), so the Engine treats the turn as thinking.
    return PromptSummary{.starts_in_reasoning = data_->starts_in_reasoning,
                         .prompt_tokens       = checked_token_count(data_->token_ids.size()),
                         .has_media           = false};
}

PromptPreparationStats PreparedPrompt::preparation_stats() const noexcept {
    if (data_ == nullptr) { return {}; }
    return PromptPreparationStats{.seconds          = data_->prepare.seconds,
                                  .tokenize_seconds = data_->prepare.tokenize_seconds};
}

std::span<const TokenId> PreparedPrompt::token_ids() const noexcept {
    return data_ != nullptr ? std::span<const TokenId>(data_->token_ids)
                            : std::span<const TokenId>{};
}

PreparedPrompt::operator bool() const noexcept { return data_ != nullptr; }

const PreparedPromptData& PreparedPromptAccess::view(const PreparedPrompt& prompt) {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    return *prompt.data_;
}

PreparedPromptData PreparedPromptAccess::take(PreparedPrompt&& prompt) {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    auto data = std::move(prompt.data_);
    return std::move(*data);
}

Frontend::Frontend(std::shared_ptr<const Impl> impl) noexcept : impl_(std::move(impl)) {}

Frontend::Frontend(const Frontend&)                = default;
Frontend& Frontend::operator=(const Frontend&)     = default;
Frontend::Frontend(Frontend&&) noexcept            = default;
Frontend& Frontend::operator=(Frontend&&) noexcept = default;
Frontend::~Frontend()                              = default;

Frontend make_frontend(const FrontendResources& resources, FrontendOptions options) {
    return Frontend(std::make_shared<const Frontend::Impl>(resources, options));
}

PreparedPrompt Frontend::prepare(PromptInput input, const PreparationControl& control) const {
    check_preparation_control(control);
    const auto start                = Clock::now();
    auto prepared                   = std::make_unique<PreparedPromptData>();
    PreparedPromptData& result      = *prepared;
    const fi::RenderedChat rendered = impl_->chat_template.render(input.messages, input.options);
    check_preparation_control(control, "chat template");
    const auto tokenize_started = Clock::now();
    result.token_ids            = impl_->tokenizer->encode(
        rendered.text, {.max_tokens = std::size_t{impl_->max_context} + 1U},
        rendered.literal_spans);
    result.prepare.tokenize_seconds =
        std::chrono::duration<double>(Clock::now() - tokenize_started).count();
    check_preparation_control(control, "tokenization");
    if (result.token_ids.size() > impl_->max_context) {
        // Tokenization stopped past the limit: count on, up to four times it, for the message.
        const std::size_t bound = std::size_t{impl_->max_context} * 4U;
        const std::size_t counted =
            impl_->tokenizer
                ->encode(rendered.text, {.max_tokens = bound + 1U}, rendered.literal_spans)
                .size();
        throw_context_length_exceeded(counted, impl_->max_context, counted > bound);
    }
    result.layout                  = fi::inspect_prompt_layout(result.token_ids, input.messages,
                                                               !input.options.tool_jsons.empty(), impl_->layout);
    result.starts_in_reasoning     = true;
    result.has_active_tools        = !input.options.tool_jsons.empty();
    result.filtered_out_tool_names = std::move(input.options.filtered_out_tool_names);
    result.context_cache           = std::move(input.context_cache);
    result.prepare.seconds         = std::chrono::duration<double>(Clock::now() - start).count();
    return PreparedPrompt(std::move(prepared));
}

std::uint32_t Frontend::count_tokens(PromptInput input, const PreparationControl& control) const {
    check_preparation_control(control);
    const fi::RenderedChat rendered = impl_->chat_template.render(input.messages, input.options);
    const auto count                = checked_token_count(
        impl_->tokenizer->encode(rendered.text, {}, rendered.literal_spans).size());
    check_preparation_control(control, "tokenization");
    return count;
}

PreparedPrompt Frontend::prepare_tokens(std::vector<TokenId> token_ids,
                                        bool allow_prefix_identity) const {
    const auto start = Clock::now();
    if (token_ids.size() > impl_->max_context) {
        throw_context_length_exceeded(token_ids.size(), impl_->max_context);
    }
    for (const TokenId token : token_ids) {
        if (!impl_->tokenizer->is_valid_token(token)) {
            throw std::out_of_range("prompt token is outside the checkpoint vocabulary: " +
                                    std::to_string(token));
        }
    }
    auto prepared               = std::make_unique<PreparedPromptData>();
    prepared->token_ids         = std::move(token_ids);
    prepared->identity.reusable = allow_prefix_identity;
    prepared->prepare.seconds   = std::chrono::duration<double>(Clock::now() - start).count();
    return PreparedPrompt(std::move(prepared));
}

PreparedPrompt Frontend::prepare_continuation(PromptInput input, std::span<const TokenId> appended,
                                              const PreparationControl& control) const {
    PreparedPrompt prompt    = prepare(std::move(input), control);
    PreparedPromptData& data = *prompt.data_;
    if (appended.size() > impl_->max_context ||
        data.token_ids.size() > impl_->max_context - appended.size()) {
        throw_context_length_exceeded(appended.size() + data.token_ids.size(), impl_->max_context);
    }
    for (const TokenId token : appended) {
        if (!impl_->tokenizer->is_valid_token(token)) {
            throw std::out_of_range("continuation token is outside the checkpoint vocabulary: " +
                                    std::to_string(token));
        }
    }
    data.token_ids.insert(data.token_ids.end(), appended.begin(), appended.end());
    data.resume_tokens.assign(appended.begin(), appended.end());
    // Decoding resumes after the appended reply: no longer at the start of the thinking phase.
    data.starts_in_reasoning = false;
    data.continuation        = true;
    return prompt;
}

ReasoningGuardTokens Frontend::reasoning_guard_tokens() const {
    if (impl_ == nullptr) { throw std::logic_error("frontend is empty"); }
    return ReasoningGuardTokens{.close_suffix   = *impl_->thinking_control,
                                .stop_token_ids = impl_->defaults.token_ids,
                                .think_close    = impl_->tokenizer->encode(kThinkingClose)};
}

std::vector<TokenId> Frontend::tokenize_text(std::string_view text) const {
    if (impl_ == nullptr) { throw std::logic_error("frontend is empty"); }
    return impl_->tokenizer->encode(text);
}

std::string Frontend::detokenize_text(std::span<const TokenId> token_ids) const {
    if (impl_ == nullptr) { throw std::logic_error("frontend is empty"); }
    return impl_->tokenizer->decode(token_ids);
}

MediaCacheSummary Frontend::media_cache_summary() const { return {}; }

OutputSession Frontend::make_output_session(const PreparedPrompt& prompt,
                                            const StopPolicy& caller_stop,
                                            const OutputOptions& output,
                                            const ThinkingControlOptions& thinking,
                                            const std::optional<OutputConstraint>& constraint,
                                            const ToolChoice& tool_choice) const {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    if (constraint) {
        throw RequestError(RequestErrorKind::UnsupportedOutputCombination,
                           "structured output is not supported for Inkling 2");
    }
    validate_tool_choice(tool_choice);
    StopPolicy policy = merge_stop_policy(*impl_->tokenizer, caller_stop);
    if (output.raw) { policy.publish_stop_token = true; }
    return OutputSession(impl_->tokenizer, std::move(policy), output, thinking,
                         impl_->thinking_control, prompt.data_->filtered_out_tool_names,
                         prompt.data_->starts_in_reasoning, prompt.data_->resume_tokens);
}

const StopPolicy& Frontend::default_stop_policy() const noexcept { return impl_->defaults; }

const ModelSamplingDefaults& Frontend::sampling_defaults() const noexcept {
    return impl_->sampling;
}

} // namespace ninfer::models::inkling2
