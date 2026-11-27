#pragma once

#include "ninfer/types.h"

#include <chrono>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer {

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();

    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] const PromptSummary& summary() const noexcept;
    [[nodiscard]] const PromptPreparationStats& preparation_stats() const noexcept;
    // The model input exactly as Engine executes it: chat template, tool declarations, earlier
    // turns and one placeholder token per Vision token included. Empty for an empty prompt; valid
    // until the prompt is submitted or destroyed.
    [[nodiscard]] std::span<const TokenId> token_ids() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    class Impl;
    explicit PreparedPrompt(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class GenerationHandle {
public:
    GenerationHandle() noexcept;
    ~GenerationHandle();

    GenerationHandle(GenerationHandle&&) noexcept;
    GenerationHandle& operator=(GenerationHandle&&) noexcept;

    GenerationHandle(const GenerationHandle&)            = delete;
    GenerationHandle& operator=(const GenerationHandle&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept;
    // The Engine's id of the request, as RuntimeStats::active_requests names it; 0 for a request
    // that never reaches the Engine queue (no output tokens requested) and once wait() consumed it.
    [[nodiscard]] std::uint64_t request_id() const noexcept;

    GenerationResult wait(OutputSink* sink = nullptr, const CancellationView& cancellation = {});

private:
    class Impl;
    explicit GenerationHandle(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class Engine {
public:
    explicit Engine(EngineOptions options);
    ~Engine();

    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;

    // Raw token input is retained for repeatable correctness and performance measurement.
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;

    // The chat prompt of `input` (which must start in thinking and carry no media) followed by
    // `appended` tokens, prepared to continue in content: tool declarations, cache identity and
    // markers are those of the chat prompt. Used to close a reply that stopped inside thinking.
    [[nodiscard]] PreparedPrompt prepare_continuation(PromptInput input,
                                                      std::span<const TokenId> appended,
                                                      const PreparationControl& control = {}) const;
    [[nodiscard]] ReasoningGuardTokens reasoning_guard_tokens() const;
    // Reasoning EOS guard (serve --reasoning-eos-guard): the first token of a reply that starts in
    // thinking is never a terminal token. Process-wide; call before the first request.
    void configure_reasoning_eos_guard(bool enabled);
    // First tokens redrawn by the guard since the process started.
    [[nodiscard]] std::uint64_t reasoning_eos_guard_replacements() const noexcept;

    // Artifact-tokenizer raw-text encoding. No chat template or implicit special token is added.
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;
    // Artifact-tokenizer decoding of token IDs to text, special tokens included.
    [[nodiscard]] std::string detokenize_text(std::span<const TokenId> token_ids) const;

    // Returns log p(tokens[i] | tokens[0..i)) for i in [first_target,tokens.size()).
    [[nodiscard]] std::vector<float> score_tokens(std::vector<TokenId> tokens,
                                                  std::uint32_t first_target);

    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] ModelSamplingDefaults sampling_defaults() const;

    // Establishes queue membership synchronously with a fixed output consumer mode. Destroying an
    // unconsumed handle cancels its request; wait() owns result consumption and may run
    // independently from GPU execution. Streaming mode requires a non-null sink in wait() and
    // publishes one exact GenerationStart before output deltas; Aggregate mode requires a null
    // sink. Observation options request protocol-neutral publication facts without changing the
    // execution request.
    [[nodiscard]] GenerationHandle
    submit(PreparedPrompt prompt, RequestOptions options,
           OutputConsumerMode consumer_mode                       = OutputConsumerMode::Aggregate,
           GenerationObservationOptions observation               = {},
           std::chrono::steady_clock::time_point pending_deadline = {});

    GenerationResult generate(PreparedPrompt prompt, RequestOptions options,
                              OutputSink* sink                     = nullptr,
                              const CancellationView& cancellation = {});

    [[nodiscard]] const EngineOptions& options() const;
    [[nodiscard]] LoadSummary load_summary() const;
    [[nodiscard]] MemorySummary memory_summary() const;
    [[nodiscard]] RuntimeStats runtime_stats() const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    [[nodiscard]] bool is_available() const;

    void reset_memory_peaks() noexcept;

    // Disk tier of the context cache (ContextCacheOptions::disk_directory; no-op when off): at a
    // clean shutdown, writes the retained conversations that are not on disk yet, within
    // ContextCacheOptions::disk_flush_seconds.
    void persist_context_cache() noexcept;

    // Unload and load: hands the host-mapped weights' bytes (expert banks, embedding_host) out so
    // they outlive this Engine, to give to the next one (EngineOptions::retained_host_weights);
    // null when there are none or they cannot be kept (the exclusive tier writes them). Call it
    // with nothing running, then free this Engine.
    [[nodiscard]] std::shared_ptr<artifact::RetainedHostWeights> release_host_weights();

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace ninfer
