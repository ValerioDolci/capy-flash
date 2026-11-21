#pragma once

// The Engine of an Inkling 2 artifact. ninfer::Engine (src/runtime/engine/engine.cpp) selects it
// before any Qwen work when the artifact's Text component is Inkling 2, and each of its public
// methods then delegates here; Engine keeps the family-independent request handling (option
// resolution, the zero-output path, handles). The cores are the common ones over
// inkling2::ModelInstance.

#include "models/inkling2/instance.h"
#include "ninfer/types.h"
#include "runtime/contract/request.h"
#include "runtime/engine/causal_score_core.h"
#include "runtime/engine/engine_core.h"

#include <chrono>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::models::inkling2 {

// A prepared prompt with what ninfer::PreparedPrompt publishes about it.
struct EnginePrompt {
    PromptSummary summary;
    PromptPreparationStats preparation;
    SamplingMode sampling_mode = SamplingMode::Thinking;
    PreparedPrompt value;
};

class EngineFamily {
public:
    using GenerationCore = runtime::EngineCore<ModelInstance>;
    using ScoringCore    = runtime::CausalScoreCore<ModelInstance>;
    using Submission     = GenerationCore::Submission;

    // `options` normalized by runtime::normalize_engine_options; `execution` outlives the family.
    // Fills `load` and `sampling_defaults` of the Engine.
    [[nodiscard]] static std::unique_ptr<EngineFamily>
    start(const EngineOptions& options, ExecutionContext& execution, bool peer_access,
          LoadSummary& load, ModelSamplingDefaults& sampling_defaults);

    EngineFamily(const EngineOptions& options, ExecutionContext& execution, bool peer_access);
    ~EngineFamily() noexcept;
    EngineFamily(const EngineFamily&)            = delete;
    EngineFamily& operator=(const EngineFamily&) = delete;

    [[nodiscard]] EnginePrompt prepare(PromptInput input, const PreparationControl& control) const;
    [[nodiscard]] EnginePrompt prepare_tokens(std::vector<TokenId> token_ids,
                                              bool allow_prefix_identity) const;
    [[nodiscard]] EnginePrompt prepare_continuation(PromptInput input,
                                                    std::span<const TokenId> appended,
                                                    const PreparationControl& control) const;
    [[nodiscard]] ReasoningGuardTokens reasoning_guard_tokens() const;
    void configure_reasoning_eos_guard(bool enabled);
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;
    [[nodiscard]] std::string detokenize_text(std::span<const TokenId> token_ids) const;
    // The scores of a prompt from prepare_tokens; `expected` is the count Engine requires.
    [[nodiscard]] std::vector<float> score(PreparedPrompt prompt, std::uint32_t first_target,
                                           std::size_t expected);
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control) const;
    [[nodiscard]] Submission submit(PreparedPrompt prompt, PromptSummary prompt_summary,
                                    double prepare_seconds, runtime::ResolvedRequestOptions options,
                                    OutputConsumerMode consumer_mode,
                                    GenerationObservationOptions observation,
                                    std::chrono::steady_clock::time_point pending_deadline);
    [[nodiscard]] MemorySummary memory_summary() const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    [[nodiscard]] RuntimeStats runtime_stats() const;
    [[nodiscard]] bool is_available() const;
    void reset_memory_peaks() noexcept;

    [[nodiscard]] const LoadSummary& load_summary() const noexcept { return load_; }

    [[nodiscard]] const ModelSamplingDefaults& sampling_defaults() const noexcept {
        return sampling_defaults_;
    }

private:
    [[nodiscard]] EnginePrompt engine_prompt(PreparedPrompt prepared, SamplingMode mode) const;

    EngineOptions options_;
    ExecutionContext& execution_;
    DeviceContext& device_;
    std::unique_ptr<ModelInstance> active_;
    LoadSummary load_;
    ModelSamplingDefaults sampling_defaults_;
    std::variant<std::monostate, std::unique_ptr<GenerationCore>, std::unique_ptr<ScoringCore>>
        core_;
};

} // namespace ninfer::models::inkling2
