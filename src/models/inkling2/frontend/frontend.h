#pragma once

#include "models/inkling2/frontend/output_session.h"
#include "ninfer/types.h"
#include "runtime/contract/request.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::inkling2 {

// The Text component's frontend files (Model::resources() of the artifact); the views must stay
// valid during make_frontend only.
struct FrontendResources {
    std::string_view tokenizer_json;
    std::string_view tokenizer_config_json;
    std::string_view chat_template_jinja;
    std::string_view generation_config_json;
};

struct FrontendOptions {
    // Replaces the artifact's chat template when set.
    std::filesystem::path chat_template_path;
    std::uint32_t max_context = 2'048;
};

// Sampling of generation_config.json (do_sample, temperature, top_p, top_k), the same for the
// thinking and non-thinking presets: Inkling 2 always thinks.
[[nodiscard]] ModelSamplingDefaults sampling_defaults_from(std::string_view generation_config_json);

struct PreparedPromptData;
class Frontend;
class PreparedPromptAccess;

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();
    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] PromptSummary summary() const;
    [[nodiscard]] PromptPreparationStats preparation_stats() const noexcept;
    [[nodiscard]] std::span<const TokenId> token_ids() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    explicit PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept;
    std::unique_ptr<PreparedPromptData> data_;

    friend class Frontend;
    friend class PreparedPromptAccess;
};

// Inkling 2 input and output semantics behind the Engine's frontend contract. Images, structured
// output (grammars) and tool_choice selection are rejected with RequestError.
class Frontend {
public:
    Frontend(const Frontend&);
    Frontend& operator=(const Frontend&);
    Frontend(Frontend&&) noexcept;
    Frontend& operator=(Frontend&&) noexcept;
    ~Frontend();

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;
    // prepare(input) followed by `appended` tokens (the reply so far plus a closing suffix); the
    // output session resumes in the state those tokens leave, e.g. inside a text message.
    [[nodiscard]] PreparedPrompt prepare_continuation(PromptInput input,
                                                      std::span<const TokenId> appended,
                                                      const PreparationControl& control = {}) const;
    [[nodiscard]] ReasoningGuardTokens reasoning_guard_tokens() const;
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;
    [[nodiscard]] std::string detokenize_text(std::span<const TokenId> token_ids) const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    [[nodiscard]] OutputSession
    make_output_session(const PreparedPrompt& prompt, const StopPolicy& caller_stop,
                        const OutputOptions& output                       = {},
                        const ThinkingControlOptions& thinking            = {},
                        const std::optional<OutputConstraint>& constraint = {},
                        const ToolChoice& tool_choice                     = {}) const;
    [[nodiscard]] const StopPolicy& default_stop_policy() const noexcept;
    [[nodiscard]] const ModelSamplingDefaults& sampling_defaults() const noexcept;

private:
    class Impl;
    explicit Frontend(std::shared_ptr<const Impl> impl) noexcept;
    std::shared_ptr<const Impl> impl_;

    friend Frontend make_frontend(const FrontendResources& resources, FrontendOptions options);
};

[[nodiscard]] Frontend make_frontend(const FrontendResources& resources, FrontendOptions options);

} // namespace ninfer::models::inkling2
