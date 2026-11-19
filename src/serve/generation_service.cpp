#include "serve/generation_service.h"
#include "serve/prompt_overlap.h"
#include "serve/turn_anchors.h"

#include "product/media_acquire/acquire.h"
#include "serve/translate.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::serve {

struct RequestCapacity {
    explicit RequestCapacity(std::size_t limit) : maximum(limit) {}

    std::mutex mutex;
    std::size_t active = 0;
    const std::size_t maximum;
    // Engine calls outside a request (token counting): they hold the Engine like a request, so
    // an unload waits for neither and refuses both.
    std::size_t users = 0;
    // When a request or a use last started or ended (steady clock ns): --idle-unload-seconds.
    std::atomic<std::int64_t> last_activity_ns{0};

    void touch() noexcept {
        last_activity_ns.store(std::chrono::steady_clock::now().time_since_epoch().count(),
                               std::memory_order_relaxed);
    }
};

// An Engine call outside a request reservation (token counting): counted in
// RequestCapacity::users while it runs.
struct EngineUse {
    explicit EngineUse(std::shared_ptr<RequestCapacity> owner) : capacity(std::move(owner)) {
        std::lock_guard lock(capacity->mutex);
        ++capacity->users;
    }

    ~EngineUse() {
        std::lock_guard lock(capacity->mutex);
        --capacity->users;
        capacity->touch();
    }

    EngineUse(const EngineUse&)            = delete;
    EngineUse& operator=(const EngineUse&) = delete;

    std::shared_ptr<RequestCapacity> capacity;
};

struct RequestLifetime {
    RequestLifetime(std::shared_ptr<RequestCapacity> owner,
                    std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point limit)
        : capacity(std::move(owner)), started(begin), deadline(limit) {}

    ~RequestLifetime() {
        std::lock_guard lock(capacity->mutex);
        --capacity->active;
        capacity->touch();
    }

    std::shared_ptr<RequestCapacity> capacity;
    std::chrono::steady_clock::time_point started;
    std::chrono::steady_clock::time_point deadline;
};

namespace {
// The grammar vendor prefixes its errors with a timestamp and the server's source path
// ("[21:04:30] /home/.../regex_converter.cc:85: Regex parsing error..."): clients get the message
// without them.
std::string client_message(std::string_view text) {
    std::string result;
    result.reserve(text.size());
    std::size_t at    = 0;
    const auto digits = [&](std::size_t from, std::size_t count) {
        if (from + count > text.size()) return false;
        for (std::size_t i = 0; i < count; ++i)
            if (text[from + i] < '0' || text[from + i] > '9') return false;
        return true;
    };
    while (at < text.size()) {
        // "[hh:mm:ss] " + a path without spaces + ":<line>: "
        if (text[at] == '[' && digits(at + 1, 2) && text.size() > at + 10 && text[at + 3] == ':' &&
            digits(at + 4, 2) && text[at + 6] == ':' && digits(at + 7, 2) &&
            text.substr(at + 9, 2) == "] ") {
            std::size_t end = text.find(' ', at + 11);
            if (end == std::string_view::npos) end = text.size();
            const auto location = text.substr(at + 11, end - at - 11);
            const auto colon = location.rfind(':', location.size() >= 2 ? location.size() - 2 : 0);
            if (colon != std::string_view::npos && location.ends_with(":") &&
                digits(at + 11 + colon + 1, location.size() - colon - 2)) {
                at = std::min(text.size(), end + 1);
                continue;
            }
        }
        result += text[at++];
    }
    return result;
}
} // namespace

ApiError request_error_to_api_error(const ninfer::RequestError& exception,
                                    std::string_view constraint_param) {
    ApiError error;
    error.param   = "messages";
    error.message = client_message(exception.what());
    switch (exception.kind()) {
    // Structured output. The param is the wire field of the constraint; JSON Schema errors append
    // their JSON Pointer to it.
    case ninfer::RequestErrorKind::UnsupportedOutputCombination:
        error.status = 400;
        error.code   = "unsupported_output_combination";
        error.param =
            exception.pointer().empty() ? std::string(constraint_param) : exception.pointer();
        break;
    case ninfer::RequestErrorKind::InvalidToolOutput:
        error.param.clear();
        error.status = 500;
        error.type   = "server_error";
        error.code   = "invalid_tool_output";
        break;
    case ninfer::RequestErrorKind::InvalidToolConstraint:
        error.status = 400;
        error.code   = "invalid_tool_constraint";
        error.param  = "tools" + exception.pointer();
        break;
    case ninfer::RequestErrorKind::InvalidGrammar:
    case ninfer::RequestErrorKind::InvalidChoice:
    case ninfer::RequestErrorKind::InvalidRegex:
    case ninfer::RequestErrorKind::ConstraintDeadEnd:
        error.status = 400;
        error.param  = constraint_param;
        error.code =
            exception.kind() == ninfer::RequestErrorKind::InvalidGrammar  ? "invalid_grammar"
            : exception.kind() == ninfer::RequestErrorKind::InvalidChoice ? "invalid_choice"
            : exception.kind() == ninfer::RequestErrorKind::InvalidRegex  ? "invalid_regex"
                                                                          : "constraint_dead_end";
        break;
    // OutputConstraintLimits: an input or compilation over a limit, or a generating request whose
    // matcher exceeded its time budget (only that request fails).
    case ninfer::RequestErrorKind::ConstraintLimitExceeded:
        error.status = 400;
        error.code   = "constraint_limit_exceeded";
        error.param  = std::string(constraint_param) + exception.pointer();
        break;
    case ninfer::RequestErrorKind::ConstraintBudgetExceeded:
        error.status = 400;
        error.code   = "constraint_budget_exceeded";
        error.param  = constraint_param;
        break;
    case ninfer::RequestErrorKind::InvalidJsonSchema:
    case ninfer::RequestErrorKind::UnsupportedJsonSchema:
    case ninfer::RequestErrorKind::UnsatisfiableJsonSchema:
        error.status = 400;
        error.param  = exception.source() == ninfer::RequestErrorSource::Tools
                           ? "tools" + exception.pointer()
                           : std::string(constraint_param) + exception.pointer();
        error.code   = exception.kind() == ninfer::RequestErrorKind::InvalidJsonSchema
                           ? "invalid_json_schema"
                       : exception.kind() == ninfer::RequestErrorKind::UnsupportedJsonSchema
                           ? "unsupported_json_schema"
                           : "unsatisfiable_json_schema";
        break;
    case ninfer::RequestErrorKind::ContextLengthExceeded:
        error.status = 400;
        error.code   = "context_length_exceeded";
        break;
    case ninfer::RequestErrorKind::ThinkingBudgetCapacityInsufficient:
        error.param.clear();
        error.status = 400;
        error.code   = "thinking_budget_capacity_insufficient";
        break;
    case ninfer::RequestErrorKind::MediaBudgetExceeded:
        error.status = 400;
        error.code   = "media_budget_exceeded";
        break;
    case ninfer::RequestErrorKind::InvalidMedia:
        error.status = 400;
        error.code   = "invalid_media";
        break;
    case ninfer::RequestErrorKind::Overloaded:
        error.param.clear();
        error.status = 429;
        error.type   = "rate_limit_error";
        error.code   = "server_overloaded";
        break;
    case ninfer::RequestErrorKind::QueueTimeout:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case ninfer::RequestErrorKind::Cancelled:
        error.param.clear();
        error.status = 499;
        error.type   = "request_cancelled";
        error.code   = "client_disconnected";
        break;
    case ninfer::RequestErrorKind::Unavailable:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "service_unavailable";
        break;
    }
    return error;
}

namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void throw_preparation_cancelled();

[[noreturn]] void throw_media_error(const ninfer::product::media_acquire::Error& exception) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case ninfer::product::media_acquire::ErrorKind::BudgetExceeded:
        error.status = 400;
        error.code   = "media_budget_exceeded";
        break;
    case ninfer::product::media_acquire::ErrorKind::RemoteUnavailable:
        error.status = 502;
        error.type   = "server_error";
        error.code   = "media_fetch_failed";
        break;
    case ninfer::product::media_acquire::ErrorKind::RemoteTimeout:
        error.status = 504;
        error.type   = "server_error";
        error.code   = "media_fetch_timeout";
        break;
    case ninfer::product::media_acquire::ErrorKind::DeadlineExceeded:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case ninfer::product::media_acquire::ErrorKind::Cancelled:
        throw_preparation_cancelled();
    }
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_invalid_input(const std::exception& exception, const char* code) {
    ApiError error;
    error.status  = 400;
    error.param   = "messages";
    error.code    = code;
    error.message = exception.what();
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_preparation_cancelled() {
    ApiError error;
    error.status  = 499;
    error.type    = "request_cancelled";
    error.code    = "client_disconnected";
    error.message = "client disconnected during media preparation";
    throw ApiException(std::move(error));
}

ninfer::OwnedMedia acquire_media(const ContentPart& part, Clock::time_point deadline,
                                 const std::function<bool()>& is_cancelled,
                                 std::size_t& remaining_bytes) {
    if (remaining_bytes == 0) {
        throw_media_error(ninfer::product::media_acquire::Error(
            ninfer::product::media_acquire::ErrorKind::BudgetExceeded,
            "request media exceeds aggregate byte limit"));
    }
    ninfer::product::media_acquire::Policy policy;
    policy.max_bytes    = std::min(policy.max_bytes, remaining_bytes);
    policy.deadline     = deadline;
    policy.is_cancelled = is_cancelled;
    std::vector<std::uint8_t> source_bytes;
    try {
        source_bytes = ninfer::product::media_acquire::acquire_bytes(part.source, policy);
    } catch (const ninfer::product::media_acquire::Error& exception) {
        throw_media_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_media");
    }

    remaining_bytes -= source_bytes.size();
    ninfer::OwnedMedia media;
    media.kind =
        part.kind == ContentKind::Image ? ninfer::MediaKind::Image : ninfer::MediaKind::Video;
    media.media_type = part.source.media_type;
    switch (part.source.kind) {
    case ninfer::product::media_acquire::SourceKind::Path:
    case ninfer::product::media_acquire::SourceKind::Url:
        media.source_name = part.source.value;
        break;
    case ninfer::product::media_acquire::SourceKind::Data:
        media.source_name = "inline-data";
        break;
    case ninfer::product::media_acquire::SourceKind::Bytes:
        media.source_name = "inline-bytes";
        break;
    }
    media.bytes               = std::move(source_bytes);
    media.image_resize_policy = part.image_resize_policy;
    return media;
}

[[noreturn]] void throw_request_error(const ninfer::RequestError& exception,
                                      std::string_view constraint_param = {}) {
    throw ApiException(request_error_to_api_error(exception, constraint_param));
}

void check_preparation_control(Clock::time_point deadline,
                               const std::function<bool()>& is_cancelled) {
    if (is_cancelled && is_cancelled()) { throw_preparation_cancelled(); }
    if (Clock::now() >= deadline) {
        throw_request_error(ninfer::RequestError(RequestErrorKind::QueueTimeout,
                                                 "inference request expired during preparation"));
    }
}

class ServiceOutputSink final : public ninfer::OutputSink {
public:
    explicit ServiceOutputSink(const StreamSink& sink) : sink_(&sink) {}

    void start(ninfer::GenerationStart start) override {
        if (sink_->on_start) { sink_->on_start(start); }
    }

    void progress(ninfer::PromptProgress progress) override {
        if (sink_->on_progress) { sink_->on_progress(progress); }
    }

    void timing(ninfer::GenerationTimingObservation timing) override {
        if (sink_->on_timing) { sink_->on_timing(timing); }
    }

    void publish(ninfer::OutputDelta delta) override {
        if (delta.text.empty()) { return; }
        if (delta.channel == ninfer::OutputChannel::Reasoning) {
            if (sink_->on_reasoning) { sink_->on_reasoning(delta.text); }
        } else {
            if (sink_->on_content) { sink_->on_content(delta.text); }
        }
    }

private:
    const StreamSink* sink_ = nullptr;
};

// The continuation of a reasoning-close retry publishes into the stream the client already
// receives: its own start, progress and timing facts are not forwarded (the first reply's are
// the response's), only its output deltas.
class ContinuationOutputSink final : public ninfer::OutputSink {
public:
    explicit ContinuationOutputSink(const StreamSink& sink) : sink_(&sink) {}

    void start(ninfer::GenerationStart) override {}

    void progress(ninfer::PromptProgress) override {}

    void timing(ninfer::GenerationTimingObservation) override {}

    void publish(ninfer::OutputDelta delta) override {
        if (delta.text.empty()) { return; }
        if (delta.channel == ninfer::OutputChannel::Reasoning) {
            if (sink_->on_reasoning) { sink_->on_reasoning(delta.text); }
        } else {
            if (sink_->on_content) { sink_->on_content(delta.text); }
        }
    }

private:
    const StreamSink* sink_ = nullptr;
};

} // namespace

namespace {

ninfer::EngineOptions to_engine_options(const ServeOptions& options) {
    ninfer::EngineOptions out;
    out.artifact_path          = options.artifact_path;
    out.chat_template_path     = options.chat_template_path;
    out.device                 = options.device;
    out.tp                     = options.tp;
    out.devices                = options.devices;
    out.max_context            = options.max_context;
    out.kv_capacity            = options.kv_capacity;
    out.max_concurrency        = options.max_concurrency;
    out.max_pending_requests   = options.max_pending_requests;
    out.pending_timeout_ms     = options.pending_timeout_ms;
    out.decode_share           = options.decode_share;
    out.prefill_chunk          = options.prefill_chunk;
    out.kv_lease_window_tokens = options.kv_lease_window_tokens;
    out.kv_cache               = options.kv_cache;
    out.elastic_kv             = options.elastic_kv;
    out.kv_ram_active          = options.kv_ram_active;
    if (options.miss_split) {
        out.routed_experts.miss_split.mode      = *options.miss_split;
        out.routed_experts.miss_split.calibrate = options.miss_split_calibrate;
    }
    if (options.expert_cpu_threads) {
        out.routed_experts.cpu_threads      = *options.expert_cpu_threads;
        out.routed_experts.gguf_cpu_threads = *options.expert_cpu_threads;
    }
    if (options.stream_min_tokens) {
        out.routed_experts.stream_min_tokens     = *options.stream_min_tokens;
        out.routed_experts.stream_min_tokens_set = true;
    }
    out.routed_experts.cpus             = options.expert_cpus;
    out.routed_experts.exclusive_layers = options.expert_exclusive_layers;
    out.routed_experts.gguf_hybrid_cost = options.gguf_hybrid_cost;
    if (options.expert_half_life) {
        out.routed_experts.half_life_tokens = *options.expert_half_life;
    }
    out.kv_ram_region_tokens            = options.kv_ram_region_tokens;
    out.kv_ram_cache_blocks             = options.kv_ram_cache_blocks;

    out.rope_yarn_factor           = options.rope_yarn_factor;
    out.rope_original_max_position = options.rope_original_max_position;
    out.enable_vision              = options.enable_vision;
    out.vision_device              = options.vision_device;
    out.max_vision_tokens          = options.max_vision_tokens;
    out.use_cuda_graph             = options.use_cuda_graph;
    out.tp_mailbox                 = options.tp_mailbox;
    out.embedding_host             = options.embedding_host;
    out.host_pinned_max_bytes      = options.host_pinned_max_bytes;
    out.lossy_prefill              = options.lossy_prefill;
    out.speculative                = options.speculative;
    out.context_cache              = options.context_cache;
    out.context_cost.preset_path   = options.context_cost_presets;
    out.media_cache_bytes          = options.media_cache_bytes;
    out.media_live_bytes           = options.media_live_bytes;
    out.media_preprocess_threads   = options.media_preprocess_threads;
    out.constraint_limits          = options.constraint_limits;
    return out;
}

} // namespace

namespace {

std::string format_seconds(double seconds) {
    char text[32];
    std::snprintf(text, sizeof text, "%.2f s", seconds);
    return text;
}

} // namespace

const char* engine_state_name(EngineState state) noexcept {
    switch (state) {
    case EngineState::Loaded:
        return "loaded";
    case EngineState::Unloading:
        return "unloading";
    case EngineState::Unloaded:
        return "unloaded";
    case EngineState::Loading:
        return "loading";
    }
    return "unknown";
}

GenerationService::GenerationService(ServeOptions options, StartupObserver startup_observer)
    : options_(std::move(options)), engine_input_(to_engine_options(options_)) {
    request_capacity_ = std::make_shared<RequestCapacity>(
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests);
    // --load-min-free-mib auto: what the first Engine takes on each device, measured around its
    // construction (the CUDA contexts exist before: the first query creates them), less what a
    // load can do without:
    // - half the routed-expert slot pool. The slots are sized from the free memory at every
    //   load, so they fill whatever the first Engine found free; a load with less memory builds
    //   fewer slots and stays correct. Below half the pool more experts run on the CPU, and the
    //   load waits or answers 503 instead.
    // - kLoadMargin for what other programs keep after giving their memory back (an image
    //   generator's CUDA context) and for a desktop on the device.
    const std::vector<std::size_t> free_before =
        options_.load_min_free_auto ? free_device_memory() : std::vector<std::size_t>{};
    build_engine(std::move(startup_observer));
    engine_options_ = engine_->options();
    if (options_.load_min_free_auto) {
        constexpr std::size_t kLoadMargin         = std::size_t(512) << 20;
        const std::size_t elastic                 = memory_summary_.routed_expert_pool_bytes / 2;
        const std::vector<std::size_t> free_after = free_device_memory();
        for (std::size_t i = 0; i < free_before.size(); ++i) {
            const std::size_t taken =
                free_before[i] > free_after[i] ? free_before[i] - free_after[i] : 0;
            footprint_bytes_.push_back(taken > elastic + kLoadMargin ? taken - elastic - kLoadMargin
                                                                     : 0);
        }
    }
    if (!options_.prefill_miss_dump_dir.empty()) {
        prefill_miss_dump_ = std::make_shared<PrefillMissDump>(
            PrefillMissDumpOptions{.directory  = options_.prefill_miss_dump_dir,
                                   .min_tokens = options_.prefill_miss_min_tokens,
                                   .min_match  = options_.prefill_miss_min_match},
            // The dump runs at the end of a request, which holds the Engine.
            [this](std::span<const ninfer::TokenId> tokens) {
                std::shared_lock lock(engine_mutex_);
                return engine_ != nullptr ? engine_->detokenize_text(tokens) : std::string{};
            });
    }
    note_activity();
    if (options_.idle_unload_seconds != 0) {
        idle_thread_ = std::thread([this] { run_idle_unload(); });
    }
}

GenerationService::~GenerationService() {
    if (idle_thread_.joinable()) {
        {
            std::lock_guard lock(idle_mutex_);
            idle_stopping_ = true;
        }
        idle_cv_.notify_one();
        idle_thread_.join();
    }
}

void GenerationService::build_engine(StartupObserver observer) {
    ninfer::EngineOptions input = engine_input_;
    input.startup_observer      = std::move(observer);
    auto engine                 = std::make_unique<ninfer::Engine>(std::move(input));
    const std::uint32_t anchors =
        resolve_turn_anchors(options_.turn_anchors, engine->load_summary().turn_anchors,
                             static_cast<std::uint32_t>(options_.tp));
    ninfer::ReasoningGuardTokens guard_tokens;
    if (options_.reasoning_close_retry) { guard_tokens = engine->reasoning_guard_tokens(); }
    // On by default: a model whose stop configuration the guard cannot handle (no or more than two
    // terminal ids, no thinking opener) serves without it instead of not starting. The guard's
    // configuration is process-wide: every load sets it again.
    try {
        engine->configure_reasoning_eos_guard(options_.reasoning_eos_guard);
    } catch (const std::invalid_argument& error) {
        std::fprintf(stderr, "reasoning EOS guard off for this model: %s\n", error.what());
        engine->configure_reasoning_eos_guard(false);
    }
    ninfer::LoadSummary load               = engine->load_summary();
    ninfer::MemorySummary memory           = engine->memory_summary();
    ninfer::MediaCacheSummary media        = engine->media_cache_summary();
    ninfer::ModelSamplingDefaults sampling = engine->sampling_defaults();
    std::unique_lock lock(engine_mutex_);
    engine_                 = std::move(engine);
    turn_anchors_           = anchors;
    reasoning_guard_tokens_ = std::move(guard_tokens);
    load_summary_           = std::move(load);
    memory_summary_         = std::move(memory);
    media_cache_summary_    = std::move(media);
    sampling_defaults_      = std::move(sampling);
    generation_.fetch_add(1, std::memory_order_acq_rel);
}

std::vector<std::size_t> GenerationService::free_device_memory() const {
    const std::vector<int> devices = engine_input_.devices.empty()
                                         ? std::vector<int>{engine_input_.device}
                                         : engine_input_.devices;
    int previous                   = 0;
    if (cudaGetDevice(&previous) != cudaSuccess) { previous = 0; }
    std::vector<std::size_t> free_bytes;
    for (const int device : devices) {
        std::size_t free  = 0;
        std::size_t total = 0;
        if (cudaSetDevice(device) != cudaSuccess || cudaMemGetInfo(&free, &total) != cudaSuccess) {
            (void)cudaSetDevice(previous);
            throw std::runtime_error("cannot query the free memory of device " +
                                     std::to_string(device));
        }
        free_bytes.push_back(free);
    }
    (void)cudaSetDevice(previous);
    return free_bytes;
}

void GenerationService::note_activity() const noexcept { request_capacity_->touch(); }

void GenerationService::log_transition(const std::string& line) const {
    std::function<void(const std::string&)> log;
    {
        std::lock_guard lock(lifecycle_mutex_);
        log = transition_log_;
    }
    if (log) { log(line); }
}

ninfer::LoadSummary GenerationService::load_summary() const {
    std::shared_lock lock(engine_mutex_);
    return engine_ != nullptr ? engine_->load_summary() : load_summary_;
}

ninfer::MemorySummary GenerationService::memory_summary() const {
    std::shared_lock lock(engine_mutex_);
    return engine_ != nullptr ? engine_->memory_summary() : memory_summary_;
}

ninfer::RuntimeStats GenerationService::runtime_stats() const {
    std::shared_lock lock(engine_mutex_);
    return engine_ != nullptr ? engine_->runtime_stats() : ninfer::RuntimeStats{};
}

bool GenerationService::is_available() const {
    std::shared_lock lock(engine_mutex_);
    return engine_ == nullptr || engine_->is_available();
}

void GenerationService::persist_context_cache() noexcept {
    std::shared_lock lock(engine_mutex_);
    if (engine_ != nullptr) { engine_->persist_context_cache(); }
}

ninfer::MediaCacheSummary GenerationService::media_cache_summary() const {
    std::shared_lock lock(engine_mutex_);
    return engine_ != nullptr ? engine_->media_cache_summary() : media_cache_summary_;
}

ninfer::ModelSamplingDefaults GenerationService::sampling_defaults() const {
    std::shared_lock lock(engine_mutex_);
    return engine_ != nullptr ? engine_->sampling_defaults() : sampling_defaults_;
}

EngineTransition GenerationService::unload(std::string_view reason) {
    const auto started = Clock::now();
    {
        std::unique_lock lock(lifecycle_mutex_);
        lifecycle_cv_.wait(lock, [this] {
            const EngineState state = state_.load(std::memory_order_acquire);
            return state == EngineState::Loaded || state == EngineState::Unloaded;
        });
        if (state_.load(std::memory_order_acquire) == EngineState::Unloaded) {
            return EngineTransition{.result = EngineTransition::Result::Already};
        }
        std::lock_guard capacity(request_capacity_->mutex);
        const std::size_t holding = request_capacity_->active + request_capacity_->users;
        if (holding != 0) {
            return EngineTransition{.result = EngineTransition::Result::Busy,
                                    .message =
                                        std::to_string(holding) + " request(s) hold the Engine"};
        }
        // A request that arrives from here on reserves its lifetime and then waits in
        // ensure_loaded() until the Engine is unloaded, and loads it again.
        state_.store(EngineState::Unloading, std::memory_order_release);
    }
    log_transition("engine unloading | " + std::string(reason));
    // The retained conversations go to the disk tier (bounded by --disk-cache-flush-seconds)
    // while the Engine still serves the readers; then it is taken out and freed outside the lock.
    engine_->persist_context_cache();
    // The page-locked host weights stay for the next load unless --unload-release-host.
    if (!options_.unload_release_host) { retained_host_ = engine_->release_host_weights(); }
    std::unique_ptr<ninfer::Engine> released;
    {
        std::unique_lock lock(engine_mutex_);
        released = std::move(engine_);
        generation_.fetch_add(1, std::memory_order_acq_rel);
    }
    released.reset();
    const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
    log_transition("engine unloaded | " + format_seconds(seconds) + " | " + std::string(reason) +
                   (retained_host_ != nullptr ? " | host weights kept page-locked" : ""));
    {
        std::lock_guard lock(lifecycle_mutex_);
        state_.store(EngineState::Unloaded, std::memory_order_release);
    }
    lifecycle_cv_.notify_all();
    return EngineTransition{.result = EngineTransition::Result::Done, .seconds = seconds};
}

EngineTransition GenerationService::load(std::string_view reason) {
    const auto started = Clock::now();
    // --load-wait-seconds: while another program still holds the GPUs, the load waits for their
    // memory instead of answering InsufficientMemory at once.
    const auto wait_until = started + std::chrono::seconds(options_.load_wait_seconds);
    for (bool logged_wait = false;;) {
        std::unique_lock lock(lifecycle_mutex_);
        lifecycle_cv_.wait(lock, [this] {
            const EngineState state = state_.load(std::memory_order_acquire);
            return state == EngineState::Loaded || state == EngineState::Unloaded;
        });
        if (state_.load(std::memory_order_acquire) == EngineState::Loaded) {
            return EngineTransition{.result = EngineTransition::Result::Already};
        }
        // Another program may still hold the GPUs: refuse instead of building a smaller Engine
        // (KV capacity and expert slots are planned from the free memory) or failing in the
        // middle of the load.
        const std::string shortage = memory_shortage();
        if (shortage.empty()) {
            state_.store(EngineState::Loading, std::memory_order_release);
            break;
        }
        if (Clock::now() >= wait_until) {
            return EngineTransition{.result  = EngineTransition::Result::InsufficientMemory,
                                    .message = shortage};
        }
        lock.unlock();
        if (!logged_wait) {
            log_transition("engine load waiting for device memory | " + shortage);
            logged_wait = true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    log_transition("engine loading | " + std::string(reason));
    engine_input_.retained_host_weights = std::move(retained_host_);
    try {
        build_engine({});
        engine_input_.retained_host_weights.reset();
        warmup();
    } catch (const std::exception& error) {
        engine_input_.retained_host_weights.reset();
        {
            std::unique_lock lock(engine_mutex_);
            engine_.reset();
        }
        {
            std::lock_guard lock(lifecycle_mutex_);
            state_.store(EngineState::Unloaded, std::memory_order_release);
        }
        lifecycle_cv_.notify_all();
        log_transition("engine load failed | " + std::string(reason) + " | " + error.what());
        return EngineTransition{.result  = EngineTransition::Result::Failed,
                                .message = error.what()};
    }
    const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
    log_transition("engine loaded | " + format_seconds(seconds) + " | " + std::string(reason));
    note_activity();
    {
        std::lock_guard lock(lifecycle_mutex_);
        state_.store(EngineState::Loaded, std::memory_order_release);
    }
    lifecycle_cv_.notify_all();
    return EngineTransition{.result = EngineTransition::Result::Done, .seconds = seconds};
}

std::string GenerationService::memory_shortage() const {
    if (!options_.load_min_free_mib && !options_.load_min_free_auto) { return {}; }
    const std::vector<std::size_t> free = free_device_memory();
    const std::vector<int> devices      = engine_input_.devices.empty()
                                              ? std::vector<int>{engine_input_.device}
                                              : engine_input_.devices;
    for (std::size_t i = 0; i < free.size(); ++i) {
        const std::size_t need = options_.load_min_free_mib    ? *options_.load_min_free_mib << 20
                                 : i < footprint_bytes_.size() ? footprint_bytes_[i]
                                                               : 0;
        if (free[i] < need) {
            return "device " + std::to_string(devices[i]) + " has " +
                   std::to_string(free[i] >> 20) + " MiB free, the Engine needs " +
                   std::to_string(need >> 20) + " MiB";
        }
    }
    return {};
}

void GenerationService::ensure_loaded(std::string_view reason) const {
    if (state_.load(std::memory_order_acquire) == EngineState::Loaded) { return; }
    // A request (it holds a reservation, so no unload starts meanwhile) loads the Engine the way
    // POST /load does. Loading is not const, the request path is: the service owns the Engine.
    const EngineTransition loaded = const_cast<GenerationService*>(this)->load(reason);
    if (loaded.result == EngineTransition::Result::Done ||
        loaded.result == EngineTransition::Result::Already) {
        return;
    }
    throw_request_error(
        ninfer::RequestError(RequestErrorKind::Unavailable,
                             "the model is unloaded and could not be loaded: " + loaded.message));
}

void GenerationService::run_idle_unload() {
    const auto idle = std::chrono::seconds(options_.idle_unload_seconds);
    for (;;) {
        {
            std::unique_lock lock(idle_mutex_);
            if (idle_cv_.wait_for(lock, std::chrono::seconds(1),
                                  [this] { return idle_stopping_; })) {
                return;
            }
        }
        if (state_.load(std::memory_order_acquire) != EngineState::Loaded) { continue; }
        const auto last = Clock::time_point(
            Clock::duration(request_capacity_->last_activity_ns.load(std::memory_order_relaxed)));
        if (Clock::now() - last < idle) { continue; }
        // Busy (a request just arrived) is fine: its end counts as activity.
        (void)unload("idle " + std::to_string(options_.idle_unload_seconds) + " s");
        note_activity();
    }
}

std::shared_ptr<RequestLifetime>
GenerationService::acquire_request_lifetime(DeadlinePolicy deadline_policy) const {
    const auto started = Clock::now();
    {
        std::lock_guard lock(request_capacity_->mutex);
        if (request_capacity_->active >= request_capacity_->maximum) {
            throw_request_error(ninfer::RequestError(RequestErrorKind::Overloaded,
                                                     "inference request queue is full"));
        }
        ++request_capacity_->active;
        request_capacity_->touch();
    }
    try {
        const Clock::time_point deadline =
            deadline_policy == DeadlinePolicy::UnboundedStartup
                ? Clock::time_point::max()
                : started + std::chrono::milliseconds(options_.pending_timeout_ms);
        return std::make_shared<RequestLifetime>(request_capacity_, started, deadline);
    } catch (...) {
        std::lock_guard lock(request_capacity_->mutex);
        --request_capacity_->active;
        throw;
    }
}

PreparedRequest GenerationService::prepare(const GenerationRequest& request,
                                           GenerationConsumerMode consumer_mode,
                                           ninfer::GenerationObservationOptions observation,
                                           std::function<bool()> is_cancelled,
                                           ContextCacheHints context_cache) const {
    if (options_.log_speculation_detail) {
        observation.accepted_length_steps = options_.speculation_detail_max_steps;
    }
    return prepare_impl(
        request, consumer_mode, observation, std::move(is_cancelled), std::move(context_cache),
        options_.allow_prefix_reuse ? CacheParticipation::ReadWrite : CacheParticipation::Disabled,
        DeadlinePolicy::ClientPendingTimeout, options_.log_speculation_detail);
}

PreparedRequest GenerationService::prepare_impl(
    const GenerationRequest& original, GenerationConsumerMode consumer_mode,
    ninfer::GenerationObservationOptions observation, std::function<bool()> is_cancelled,
    ContextCacheHints context_cache, CacheParticipation cache_participation,
    DeadlinePolicy deadline_policy, bool speculation_detail) const {
    PreparedRequest prepared;
    prepared.media_omitted = original.media_item_count() != 0 && !options_.enable_vision;
    const GenerationRequest without_media =
        prepared.media_omitted ? media_as_text(original) : GenerationRequest{};
    const GenerationRequest& request        = prepared.media_omitted ? without_media : original;
    prepared.speculation_detail             = speculation_detail;
    prepared.constraint_param               = request.constraint_param;
    const ResolvedPromptSemantics semantics = resolve_prompt_semantics(request, options_);
    ninfer::RequestOptions request_options  = to_request_options(
        request, options_, semantics, cache_participation == CacheParticipation::ReadWrite);
    prepared.thinking_budget   = request_options.execution.thinking.budget;
    prepared.reasoning_effort  = semantics.reasoning_effort;
    prepared.preserve_thinking = semantics.preserve_thinking;
    prepared.lifetime          = acquire_request_lifetime(deadline_policy);
    if (deadline_policy == DeadlinePolicy::ClientPendingTimeout &&
        engine_state() != EngineState::Loaded) {
        // An unloaded Engine is loaded for this request; the load does not count against the
        // request's pending timeout.
        ensure_loaded("request");
        prepared.lifetime->deadline =
            std::max(prepared.lifetime->deadline,
                     Clock::now() + std::chrono::milliseconds(options_.pending_timeout_ms));
    }

    try {
        const auto acquisition_started = Clock::now();
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, ninfer::kMaximumPromptMediaBytes);
        ninfer::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, prepared.lifetime->deadline, is_cancelled,
                                     remaining_media_bytes);
            });
        std::vector<PromptCacheMarker> protocol_markers = std::move(input.context_cache.markers);
        const bool protocol_allows_engine_automatic =
            input.context_cache.allow_engine_automatic_shared_prefixes;
        input.context_cache = std::move(context_cache);
        input.context_cache.markers.insert(input.context_cache.markers.end(),
                                           std::make_move_iterator(protocol_markers.begin()),
                                           std::make_move_iterator(protocol_markers.end()));
        input.context_cache.allow_engine_automatic_shared_prefixes =
            input.context_cache.allow_engine_automatic_shared_prefixes &&
            protocol_allows_engine_automatic;
        add_turn_anchor_markers(input, turn_anchors_);
        prepared.acquisition_seconds =
            std::chrono::duration<double>(Clock::now() - acquisition_started).count();
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        const PreparationControl control{
            .deadline     = prepared.lifetime->deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        std::shared_ptr<ReasoningRetryContext> retry;
        // A constrained request is never retried: the continuation would run without the
        // grammar state of the first reply. Media prompts are: the continuation
        // keeps their M-RoPE positions.
        if (options_.reasoning_close_retry && !request.constraint &&
            cache_participation == CacheParticipation::ReadWrite) {
            retry                = std::make_shared<ReasoningRetryContext>();
            retry->input         = input;
            retry->consumer_mode = consumer_mode;
            retry->observation   = observation;
        }
        ninfer::PreparedPrompt prompt = engine_->prepare(std::move(input), control);
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        prepared.enable_thinking = prompt.summary().starts_in_reasoning;
        if (retry && prepared.enable_thinking) {
            retry->request_options   = request_options;
            prepared.reasoning_retry = std::move(retry);
        }
        if (!prepared.enable_thinking) {
            request_options.execution.thinking.budget.reset();
            prepared.thinking_budget.reset();
        }
        prepared.prompt_tokens = static_cast<int>(prompt.summary().prompt_tokens);
        prepared.preparation   = prompt.preparation_stats();
        prepared.prepare_seconds =
            std::chrono::duration<double>(Clock::now() - prepared.lifetime->started).count();
        if (speculation_detail || (prefill_miss_dump_ != nullptr &&
                                   cache_participation == CacheParticipation::ReadWrite)) {
            const std::span<const ninfer::TokenId> prompt_ids = prompt.token_ids();
            prepared.prompt_token_ids.assign(prompt_ids.begin(), prompt_ids.end());
        }
        prepared.generation = engine_->submit(std::move(prompt), std::move(request_options),
                                              consumer_mode == GenerationConsumerMode::Streaming
                                                  ? ninfer::OutputConsumerMode::Streaming
                                                  : ninfer::OutputConsumerMode::Aggregate,
                                              observation, prepared.lifetime->deadline);
        prepared.sampling   = prepared.generation.resolved_sampling();
    } catch (const ApiException&) { throw; } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception, request.constraint_param);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_prompt");
    }
    return prepared;
}

int GenerationService::count_prompt_tokens(const GenerationRequest& original,
                                           std::function<bool()> is_cancelled) const {
    const bool omit_media = original.media_item_count() != 0 && !options_.enable_vision;
    const GenerationRequest without_media =
        omit_media ? media_as_text(original) : GenerationRequest{};
    const GenerationRequest& request = omit_media ? without_media : original;
    const EngineUse use(request_capacity_);
    ensure_loaded("token count");
    const Clock::time_point deadline =
        Clock::now() + std::chrono::milliseconds(options_.pending_timeout_ms);
    const ResolvedPromptSemantics semantics = resolve_prompt_semantics(request, options_);
    try {
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, ninfer::kMaximumPromptMediaBytes);
        ninfer::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, deadline, is_cancelled, remaining_media_bytes);
            });
        check_preparation_control(deadline, is_cancelled);
        const PreparationControl control{
            .deadline     = deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        const int prompt_tokens =
            static_cast<int>(engine_->count_tokens(std::move(input), control));
        check_preparation_control(deadline, is_cancelled);
        return prompt_tokens;
    } catch (const ApiException&) { throw; } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_prompt");
    }
}

GenerationOutcome GenerationService::run(PreparedRequest& prepared, const StreamSink* sink,
                                         std::function<bool()> is_cancelled) {
    std::unique_ptr<ServiceOutputSink> output_sink;
    if (sink != nullptr) { output_sink = std::make_unique<ServiceOutputSink>(*sink); }
    ninfer::OutputSink* public_sink = output_sink.get();
    ninfer::CancellationView cancellation;
    if (is_cancelled || (sink != nullptr && sink->is_cancelled)) {
        cancellation = ninfer::CancellationView([external = std::move(is_cancelled), sink]() {
            return (external && external()) ||
                   (sink != nullptr && sink->is_cancelled && sink->is_cancelled());
        });
    }

    ninfer::GenerationResult result;
    try {
        result = prepared.generation.wait(public_sink, cancellation);
    } catch (...) {
        // A request that ends without a result still leaves its prompt as a context to compare
        // the next one against.
        if (prefill_miss_dump_ != nullptr && !prepared.prompt_token_ids.empty()) {
            (void)prefill_miss_dump_->observe(
                prepared.prompt_token_ids, {},
                static_cast<std::uint32_t>(prepared.prompt_token_ids.size()), "failed");
        }
        try {
            throw;
        } catch (const ninfer::RequestError& exception) {
            throw_request_error(exception, prepared.constraint_param);
        }
    }
    GenerationOutcome outcome;
    outcome.text                = std::move(result.content);
    outcome.reasoning           = std::move(result.reasoning);
    outcome.prompt_tokens       = static_cast<int>(result.prompt.prompt_tokens);
    outcome.completion_tokens   = static_cast<int>(result.generated_token_ids.size());
    outcome.reasoning_tokens    = static_cast<int>(result.reasoning_tokens);
    outcome.thinking            = result.thinking;
    outcome.finish_reason       = result.finish_reason;
    outcome.matched_stop_string = std::move(result.matched_stop_string);

    outcome.metrics.prepare_seconds = prepared.prepare_seconds;
    outcome.metrics.ttft_seconds =
        prepared.prepare_seconds +
        std::max(0.0, result.timings.first_token_seconds - result.timings.prepare_seconds);
    outcome.metrics.vision_seconds          = result.timings.vision_seconds;
    outcome.metrics.prefill_seconds         = result.timings.prefill_seconds;
    outcome.metrics.decode_seconds          = result.timings.decode_seconds;
    outcome.metrics.prompt_wall_seconds     = result.timings.prompt_wall_seconds;
    outcome.metrics.generation_wall_seconds = result.timings.generation_wall_seconds;
    outcome.metrics.total_seconds =
        prepared.prepare_seconds +
        std::max(0.0, result.timings.total_seconds - result.timings.prepare_seconds);
    outcome.metrics.engine_timing               = result.engine_timing;
    outcome.metrics.prefix_cache_hit_tokens     = result.reused_prompt_tokens;
    outcome.metrics.prefix_reuse_path           = result.prefix_reuse_path;
    outcome.metrics.materialization             = result.materialization;
    outcome.metrics.speculative_backend         = result.speculative.backend;
    outcome.metrics.speculative_draft_window    = result.speculative.draft_window;
    outcome.metrics.speculative_rounds          = result.speculative.rounds;
    outcome.metrics.speculative_draft_tokens    = result.speculative.drafted_tokens;
    outcome.metrics.speculative_accepted_tokens = result.speculative.accepted_tokens;
    outcome.metrics.speculative_fallback_steps  = result.speculative.fallback_steps;
    outcome.metrics.speculative_accepted_per_position =
        std::move(result.speculative.accepted_per_position);
    if (prepared.speculation_detail) {
        outcome.metrics.speculative_accepted_lengths =
            std::move(result.speculative.accepted_lengths);
        outcome.metrics.prompt_ngram_overlap =
            prompt_ngram_overlap(prepared.prompt_token_ids, result.generated_token_ids);
    }
    if (prefill_miss_dump_ != nullptr && !prepared.prompt_token_ids.empty()) {
        outcome.metrics.prefill_miss = prefill_miss_dump_->observe(
            prepared.prompt_token_ids, result.generated_token_ids, result.reused_prompt_tokens,
            result.finish_reason == ninfer::FinishReason::Cancelled ? "cancelled" : "finished");
    }
    prepared.prompt_token_ids = {};

    outcome.tool_calls      = std::move(result.tool_calls);
    outcome.tool_call_parse = result.tool_call_parse;
    if (prepared.reasoning_retry) {
        apply_reasoning_retry(prepared, result, outcome, sink, cancellation);
        prepared.reasoning_retry.reset();
    }
    if (options_.reasoning_eos_guard) {
        outcome.reasoning_eos_guard_total = engine_->reasoning_eos_guard_replacements();
    }
    return outcome;
}

void GenerationService::apply_reasoning_retry(PreparedRequest& prepared,
                                              const ninfer::GenerationResult& first,
                                              GenerationOutcome& outcome, const StreamSink* sink,
                                              const ninfer::CancellationView& cancellation) {
    ReasoningRetryContext& retry  = *prepared.reasoning_retry;
    const ReasoningRetryPlan plan = plan_reasoning_retry(
        ReasoningRetryInput{.starts_in_reasoning      = prepared.enable_thinking,
                            .finish_reason            = outcome.finish_reason,
                            .generated                = first.generated_token_ids,
                            .content                  = outcome.text,
                            .tool_calls               = outcome.tool_calls.size(),
                            .thinking_control_applied = outcome.thinking.applied,
                            .requested_output_tokens =
                                retry.request_options.execution.requested_output_tokens},
        reasoning_guard_tokens_);
    if (plan.trigger == ReasoningRetryTrigger::None) { return; }
    if (cancellation.requested()) { return; }
    ReasoningRetryStats& stats = outcome.reasoning_retry;
    stats.trigger              = plan.trigger;
    stats.kept_tokens          = plan.kept_tokens;
    stats.injected_tokens      = plan.injected_tokens;

    ninfer::RequestOptions options = std::move(retry.request_options);
    if (plan.remaining_output_tokens) {
        options.execution.requested_output_tokens = *plan.remaining_output_tokens;
    }
    // A fresh draw: the continuation of an EmptyAfterClose reply samples the same position again.
    options.execution.sampling.seed = prepared.sampling.seed + 1U;
    options.execution.thinking.budget.reset();
    std::unique_ptr<ContinuationOutputSink> continuation_sink;
    if (sink != nullptr) { continuation_sink = std::make_unique<ContinuationOutputSink>(*sink); }
    ninfer::GenerationResult second;
    try {
        const auto deadline = Clock::now() + std::chrono::milliseconds(options_.pending_timeout_ms);
        const PreparationControl control{.deadline = deadline, .cancellation = cancellation};
        ninfer::PreparedPrompt prompt =
            engine_->prepare_continuation(std::move(retry.input), plan.appended, control);
        ninfer::GenerationHandle generation =
            engine_->submit(std::move(prompt), std::move(options),
                            retry.consumer_mode == GenerationConsumerMode::Streaming
                                ? ninfer::OutputConsumerMode::Streaming
                                : ninfer::OutputConsumerMode::Aggregate,
                            retry.observation, deadline);
        second = generation.wait(continuation_sink.get(), cancellation);
    } catch (const std::exception& exception) {
        // The first reply stands as it was.
        stats.error = exception.what();
        return;
    }

    stats.continuation_prompt     = second.prompt.prompt_tokens;
    stats.continuation_reused     = second.reused_prompt_tokens;
    stats.continuation_completion = static_cast<std::uint32_t>(second.generated_token_ids.size());
    stats.continuation_path       = second.prefix_reuse_path;

    outcome.text = std::move(second.content);
    outcome.reasoning += second.reasoning;
    outcome.tool_calls          = std::move(second.tool_calls);
    outcome.tool_call_parse     = second.tool_call_parse;
    outcome.finish_reason       = second.finish_reason;
    outcome.matched_stop_string = std::move(second.matched_stop_string);
    outcome.completion_tokens   = static_cast<int>(plan.kept_tokens + plan.injected_tokens +
                                                   second.generated_token_ids.size());
    outcome.reasoning_tokens += static_cast<int>(second.reasoning_tokens);

    GenerationMetrics& metrics = outcome.metrics;
    metrics.prefill_seconds += second.timings.prefill_seconds;
    metrics.decode_seconds += second.timings.decode_seconds;
    metrics.prompt_wall_seconds += second.timings.prompt_wall_seconds;
    metrics.generation_wall_seconds += second.timings.generation_wall_seconds;
    metrics.total_seconds += second.timings.total_seconds;
    metrics.speculative_rounds += second.speculative.rounds;
    metrics.speculative_draft_tokens += second.speculative.drafted_tokens;
    metrics.speculative_accepted_tokens += second.speculative.accepted_tokens;
    metrics.speculative_fallback_steps += second.speculative.fallback_steps;
    const auto& positions = second.speculative.accepted_per_position;
    if (metrics.speculative_accepted_per_position.size() < positions.size()) {
        metrics.speculative_accepted_per_position.resize(positions.size(), 0);
    }
    for (std::size_t index = 0; index < positions.size(); ++index) {
        metrics.speculative_accepted_per_position[index] += positions[index];
    }
}

void GenerationService::warmup() {
    GenerationRequest request;
    ChatTurn turn;
    turn.role = ChatRole::User;
    ContentPart content;
    content.kind     = ContentKind::Text;
    content.text     = "hi";
    content.type_raw = "text";
    turn.content.push_back(std::move(content));
    request.messages.push_back(std::move(turn));
    request.max_tokens = 4;
    // Deterministic expert publication: greedy, because with sampling the drafts accepted, hence
    // the decode rounds run, vary from start to start, and the Qwen4 expert cache learns its first
    // residency from them.
    if (engine_->memory_summary().routed_expert_publish_lag >= 1) {
        request.sampling.temperature = 0.0;
    }
    PreparedRequest prepared =
        prepare_impl(request, GenerationConsumerMode::Aggregate, {}, {}, {},
                     CacheParticipation::Disabled, DeadlinePolicy::UnboundedStartup, false);
    run(prepared, nullptr);
}

} // namespace ninfer::serve
