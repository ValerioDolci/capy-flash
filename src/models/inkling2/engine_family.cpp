#include "models/inkling2/engine_family.h"

#include "core/startup.h"

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ninfer::models::inkling2 {

std::unique_ptr<EngineFamily> EngineFamily::start(const EngineOptions& options,
                                                  ExecutionContext& execution, bool peer_access,
                                                  LoadSummary& load,
                                                  ModelSamplingDefaults& sampling_defaults) {
    auto family       = std::make_unique<EngineFamily>(options, execution, peer_access);
    load              = family->load_summary();
    sampling_defaults = family->sampling_defaults();
    return family;
}

EngineFamily::EngineFamily(const EngineOptions& options, ExecutionContext& execution,
                           bool peer_access)
    : options_(options), execution_(execution), device_(execution.primary()) {
    auto constructed     = construct_model(options_, execution_);
    active_              = std::move(constructed.instance);
    load_                = std::move(constructed.load);
    load_.cuda_sync_mode = device_.sync_mode();
    load_.peer_access    = peer_access;
    sampling_defaults_   = active_->frontend.sampling_defaults();
    StartupPhaseScope finalize_phase(options_.startup_observer, StartupPhase::EngineFinalize);
    if (options_.purpose == EnginePurpose::CausalScoring) {
        core_ = std::make_unique<ScoringCore>(*active_, device_);
    } else {
        core_ = std::make_unique<GenerationCore>(*active_, device_, options_,
                                                 std::move(constructed.context_cost));
    }
    finalize_phase.complete();
}

EngineFamily::~EngineFamily() noexcept {
    device_.bind_to_current_thread_noexcept();
    core_.emplace<std::monostate>();
    try {
        for (int rank = 1; rank < execution_.tp; ++rank) {
            execution_.dev[static_cast<std::size_t>(rank)]->synchronize();
        }
        device_.synchronize();
    } catch (...) {}
}

EnginePrompt EngineFamily::engine_prompt(PreparedPrompt prepared, SamplingMode mode) const {
    const PromptSummary summary = prepared.summary();
    if (summary.prompt_tokens > active_->capacity) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           context_length_message(summary.prompt_tokens, active_->capacity));
    }
    return {.summary       = summary,
            .preparation   = prepared.preparation_stats(),
            .sampling_mode = mode,
            .value         = std::move(prepared)};
}

EnginePrompt EngineFamily::prepare(PromptInput input, const PreparationControl& control) const {
    auto prepared = active_->frontend.prepare(std::move(input), control);
    const SamplingMode mode =
        prepared.summary().starts_in_reasoning ? SamplingMode::Thinking : SamplingMode::NonThinking;
    return engine_prompt(std::move(prepared), mode);
}

EnginePrompt EngineFamily::prepare_tokens(std::vector<TokenId> token_ids,
                                          bool allow_prefix_identity) const {
    if (token_ids.size() > active_->capacity) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           context_length_message(token_ids.size(), active_->capacity));
    }
    return engine_prompt(
        active_->frontend.prepare_tokens(std::move(token_ids), allow_prefix_identity),
        SamplingMode::Thinking);
}

EnginePrompt EngineFamily::prepare_continuation(PromptInput input,
                                                std::span<const TokenId> appended,
                                                const PreparationControl& control) const {
    return engine_prompt(
        active_->frontend.prepare_continuation(std::move(input), appended, control),
        SamplingMode::Thinking);
}

ReasoningGuardTokens EngineFamily::reasoning_guard_tokens() const {
    return active_->frontend.reasoning_guard_tokens();
}

void EngineFamily::configure_reasoning_eos_guard(bool enabled) {
    if (enabled) { throw std::invalid_argument("Inkling 2 has no reasoning EOS guard"); }
}

std::vector<TokenId> EngineFamily::tokenize_text(std::string_view text) const {
    return active_->frontend.tokenize_text(text);
}

std::string EngineFamily::detokenize_text(std::span<const TokenId> token_ids) const {
    return active_->frontend.detokenize_text(token_ids);
}

std::vector<float> EngineFamily::score(PreparedPrompt prompt, std::uint32_t first_target,
                                       std::size_t expected) {
    auto* core = std::get_if<std::unique_ptr<ScoringCore>>(&core_);
    if (core == nullptr || *core == nullptr) {
        throw std::logic_error("Engine scoring core is unavailable");
    }
    std::vector<float> result = (*core)->score(std::move(prompt), first_target);
    if (result.size() != expected) {
        throw std::logic_error("target Program returned an invalid causal score count");
    }
    return result;
}

std::uint32_t EngineFamily::count_tokens(PromptInput input,
                                         const PreparationControl& control) const {
    return active_->frontend.count_tokens(std::move(input), control);
}

EngineFamily::Submission
EngineFamily::submit(PreparedPrompt prompt, PromptSummary prompt_summary, double prepare_seconds,
                     runtime::ResolvedRequestOptions options, OutputConsumerMode consumer_mode,
                     GenerationObservationOptions observation,
                     std::chrono::steady_clock::time_point pending_deadline) {
    if (!prompt) { throw std::invalid_argument("PreparedPrompt is not an Inkling 2 prompt"); }
    auto* core = std::get_if<std::unique_ptr<GenerationCore>>(&core_);
    if (core == nullptr || *core == nullptr) {
        throw std::logic_error("Engine generation core is unavailable");
    }
    return (*core)->submit(std::move(prompt), prompt_summary, prepare_seconds, std::move(options),
                           consumer_mode, observation, pending_deadline);
}

MemorySummary EngineFamily::memory_summary() const {
    return std::visit(
        [](const auto& core) -> MemorySummary {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else {
                return core->memory_summary();
            }
        },
        core_);
}

MediaCacheSummary EngineFamily::media_cache_summary() const {
    return active_->frontend.media_cache_summary();
}

RuntimeStats EngineFamily::runtime_stats() const {
    return std::visit(
        [](const auto& core) -> RuntimeStats {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else {
                return core->runtime_stats();
            }
        },
        core_);
}

bool EngineFamily::is_available() const {
    return std::visit(
        [](const auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                return false;
            } else {
                return core != nullptr && core->is_available();
            }
        },
        core_);
}

void EngineFamily::reset_memory_peaks() noexcept {
    std::visit(
        [](auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (!std::is_same_v<CoreState, std::monostate>) {
                core->reset_memory_peaks();
            }
        },
        core_);
}

} // namespace ninfer::models::inkling2
