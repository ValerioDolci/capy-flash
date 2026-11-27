#include "serve/live_stats.h"

#include <algorithm>
#include <utility>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

const char* finish_reason_name(ninfer::FinishReason reason) noexcept {
    switch (reason) {
    case ninfer::FinishReason::None:
        return "none";
    case ninfer::FinishReason::OutputLimit:
        return "output_limit";
    case ninfer::FinishReason::ContextCapacity:
        return "context_capacity";
    case ninfer::FinishReason::StopToken:
        return "stop_token";
    case ninfer::FinishReason::StopString:
        return "stop_string";
    case ninfer::FinishReason::Cancelled:
        return "cancelled";
    }
    return "unknown";
}

const char* failure_class_name(RequestFailureClass classification) noexcept {
    switch (classification) {
    case RequestFailureClass::ClientInput:
        return "client_input";
    case RequestFailureClass::ClientDisconnected:
        return "client_disconnected";
    case RequestFailureClass::Overload:
        return "overload";
    case RequestFailureClass::Timeout:
        return "timeout";
    case RequestFailureClass::Unavailable:
        return "unavailable";
    case RequestFailureClass::Upstream:
        return "upstream";
    case RequestFailureClass::Internal:
        return "internal";
    }
    return "unknown";
}

double unix_seconds(LiveStats::SystemClock::time_point wall) {
    return std::chrono::duration<double>(wall.time_since_epoch()).count();
}

double seconds_between(std::int64_t from_ns, LiveStats::SteadyClock::time_point to) {
    const std::int64_t to_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(to.time_since_epoch()).count();
    return to_ns > from_ns ? static_cast<double>(to_ns - from_ns) * 1e-9 : 0.0;
}

// A counter difference; a counter that went back means the Engine was rebuilt (unload and load).
std::uint64_t grown(std::uint64_t from, std::uint64_t to) noexcept {
    return to >= from ? to - from : 0;
}

std::optional<double> ratio(std::uint64_t part, std::uint64_t whole) noexcept {
    if (whole == 0) { return std::nullopt; }
    return static_cast<double>(part) / static_cast<double>(whole);
}

template <class T>
Json optional_json(const std::optional<T>& value) {
    return value ? Json(*value) : Json(nullptr);
}

Json client_json(const std::optional<std::string>& client) {
    return client ? Json(*client) : Json(nullptr);
}

} // namespace

LiveStats::~LiveStats() { stop(); }

void LiveStats::configure(LiveServerInfo info, std::function<LiveEngineReading()> read,
                          LiveGpuSource gpus, std::function<HostTelemetrySample()> host) {
    std::lock_guard lock(mutex_);
    info_         = std::move(info);
    read_         = std::move(read);
    gpus_         = std::move(gpus);
    host_         = std::move(host);
    started_      = SteadyClock::now();
    started_unix_ = unix_seconds(SystemClock::now());
}

void LiveStats::start() {
    if (thread_.joinable() || !read_) { return; }
    {
        std::lock_guard lock(thread_mutex_);
        stopping_ = false;
    }
    thread_ = std::thread([this] { run(); });
}

void LiveStats::stop() {
    if (!thread_.joinable()) { return; }
    {
        std::lock_guard lock(thread_mutex_);
        stopping_ = true;
    }
    thread_cv_.notify_one();
    thread_.join();
}

void LiveStats::run() {
    SteadyClock::time_point next = SteadyClock::now();
    for (;;) {
        tick(SteadyClock::now(), SystemClock::now());
        next += kTick;
        const SteadyClock::time_point now = SteadyClock::now();
        if (next <= now) { next = now + kTick; }
        std::unique_lock lock(thread_mutex_);
        if (thread_cv_.wait_until(lock, next, [this] { return stopping_; })) { return; }
    }
}

LiveStats::Reading LiveStats::reading_of(const LiveEngineReading& engine,
                                         SteadyClock::time_point at) {
    const ninfer::RuntimeStats& stats             = engine.stats;
    const ninfer::RuntimeHostWorkStats& work      = stats.host_work;
    const ninfer::ExpertCacheRuntimeStats& expert = stats.expert_cache;
    Reading reading;
    reading.at                 = at;
    reading.generation         = engine.generation;
    reading.prefill_tokens     = stats.computed_prefill_tokens;
    reading.reused_tokens      = stats.reused_prompt_tokens;
    reading.decode_tokens      = stats.committed_decode_tokens;
    reading.decode_rounds      = stats.decode_rounds;
    reading.decode_row_rounds  = stats.decode_row_rounds;
    reading.drafted_tokens     = stats.speculative_drafted_tokens;
    reading.accepted_tokens    = stats.speculative_accepted_tokens;
    reading.speculative_rounds = stats.speculative_rounds;
    reading.decode_host_ns     = work.decode_host_ns;
    reading.decode_wait_ns     = work.decode_device_wait_ns;
    reading.prefill_ns         = work.prefill_host_ns + work.prefill_device_wait_ns;
    reading.prefill_units      = work.prefill_units;
    reading.to_device_bytes =
        stats.state_h2d_bytes + stats.main_kv_h2d_bytes + stats.backend_kv_h2d_bytes;
    reading.to_host_bytes =
        stats.state_d2h_bytes + stats.main_kv_d2h_bytes + stats.backend_kv_d2h_bytes;
    reading.expert_routes        = expert.routes;
    reading.expert_hits          = expert.device_hits;
    reading.expert_decode_routes = expert.decode_routes;
    reading.expert_decode_hits   = expert.decode_device_hits;
    reading.expert_fills         = expert.fills;
    reading.expert_fill_bytes    = expert.fill_bytes;
    reading.expert_cpu           = expert.decode_cpu_experts;
    reading.expert_gpu           = expert.decode_gpu_experts;
    return reading;
}

Json LiveStats::rates_between(const Reading& from, const Reading& to) {
    const double seconds = std::chrono::duration<double>(to.at - from.at).count();
    if (seconds <= 0.0 || from.generation != to.generation) { return Json(nullptr); }
    const auto delta = [&](std::uint64_t Reading::* field) {
        return grown(from.*field, to.*field);
    };
    const auto per_second = [&](std::uint64_t Reading::* field) {
        return static_cast<double>(delta(field)) / seconds;
    };
    const std::uint64_t rounds   = delta(&Reading::decode_rounds);
    const std::uint64_t rows     = delta(&Reading::decode_row_rounds);
    const std::uint64_t computed = delta(&Reading::prefill_tokens);
    const std::uint64_t reused   = delta(&Reading::reused_tokens);
    const std::uint64_t host     = delta(&Reading::decode_host_ns);
    const std::uint64_t wait     = delta(&Reading::decode_wait_ns);
    const std::uint64_t units    = delta(&Reading::prefill_units);
    std::optional<double> step;
    if (rounds != 0) {
        step = static_cast<double>(host + wait) * 1e-9 / static_cast<double>(rounds);
    }
    std::optional<double> prefill_step;
    if (units != 0) {
        prefill_step =
            static_cast<double>(delta(&Reading::prefill_ns)) * 1e-9 / static_cast<double>(units);
    }
    return Json{
        {"seconds", seconds},
        {"prefill_tokens_per_second", static_cast<double>(computed) / seconds},
        {"decode_tokens_per_second", per_second(&Reading::decode_tokens)},
        {"decode_rounds_per_second", static_cast<double>(rounds) / seconds},
        {"decode_batch", optional_json(ratio(rows, rounds))},
        {"tokens_per_decode_step", optional_json(ratio(delta(&Reading::decode_tokens), rows))},
        {"decode_step_seconds", optional_json(step)},
        {"decode_host_share", optional_json(ratio(host, host + wait))},
        {"prefill_step_seconds", optional_json(prefill_step)},
        {"draft_acceptance",
         optional_json(ratio(delta(&Reading::accepted_tokens), delta(&Reading::drafted_tokens)))},
        {"accepted_per_round", optional_json(ratio(delta(&Reading::accepted_tokens),
                                                   delta(&Reading::speculative_rounds)))},
        {"draft_window_mean", optional_json(ratio(delta(&Reading::drafted_tokens),
                                                  delta(&Reading::speculative_rounds)))},
        {"prefix_hit_ratio", optional_json(ratio(reused, reused + computed))},
        {"to_device_bytes_per_second", per_second(&Reading::to_device_bytes)},
        {"to_host_bytes_per_second", per_second(&Reading::to_host_bytes)},
        {"expert_hit_ratio",
         optional_json(ratio(delta(&Reading::expert_hits), delta(&Reading::expert_routes)))},
        {"expert_decode_hit_ratio", optional_json(ratio(delta(&Reading::expert_decode_hits),
                                                        delta(&Reading::expert_decode_routes)))},
        {"expert_fills_per_second", per_second(&Reading::expert_fills)},
        {"expert_fill_bytes_per_second", per_second(&Reading::expert_fill_bytes)},
        {"expert_cpu_per_second", per_second(&Reading::expert_cpu)},
        {"expert_gpu_read_per_second", per_second(&Reading::expert_gpu)},
    };
}

void LiveStats::tick(SteadyClock::time_point now, SystemClock::time_point wall) {
    if (!read_) { return; }
    const LiveEngineReading engine = read_();
    const Reading reading          = reading_of(engine, now);
    std::unique_lock lock(mutex_);
    recent_.push_back(reading);
    while (recent_.size() > kRecentTicks) { recent_.pop_front(); }
    if (!point_base_ || point_base_->generation != reading.generation) {
        point_base_ = reading;
        ticks_      = 0;
        return;
    }
    if (++ticks_ < kTicksPerPoint) { return; }
    ticks_ = 0;

    // NVML and /proc once a second, outside the lock (a read can take a millisecond).
    std::optional<std::vector<GpuTelemetrySample>> gpus;
    std::optional<HostTelemetrySample> host;
    if (gpus_.sample || host_) {
        lock.unlock();
        if (gpus_.sample) { gpus = gpus_.sample(); }
        if (host_) { host = host_(); }
        lock.lock();
    }

    const Reading& from  = *point_base_;
    const double seconds = std::chrono::duration<double>(reading.at - from.at).count();
    const auto delta     = [&](std::uint64_t Reading::* field) {
        return grown(from.*field, reading.*field);
    };
    const auto per_second = [&](std::uint64_t Reading::* field) {
        return seconds > 0.0 ? static_cast<double>(delta(field)) / seconds : 0.0;
    };
    Point point;
    point.seq     = next_seq_++;
    point.t       = unix_seconds(wall);
    point.prefill = per_second(&Reading::prefill_tokens);
    point.decode  = per_second(&Reading::decode_tokens);
    point.batch =
        ratio(delta(&Reading::decode_row_rounds), delta(&Reading::decode_rounds)).value_or(0.0);
    point.acceptance = ratio(delta(&Reading::accepted_tokens), delta(&Reading::drafted_tokens));
    if (const std::uint64_t rounds = delta(&Reading::decode_rounds); rounds != 0) {
        point.decode_step_ms =
            static_cast<double>(delta(&Reading::decode_host_ns) + delta(&Reading::decode_wait_ns)) *
            1e-6 / static_cast<double>(rounds);
    }
    const std::uint64_t reused = delta(&Reading::reused_tokens);
    point.prefix_hit           = ratio(reused, reused + delta(&Reading::prefill_tokens));
    point.expert_hit =
        ratio(delta(&Reading::expert_decode_hits), delta(&Reading::expert_decode_routes));
    point.expert_fill_bytes = per_second(&Reading::expert_fill_bytes);
    point.transfer_bytes =
        per_second(&Reading::to_device_bytes) + per_second(&Reading::to_host_bytes);
    point.running  = engine.stats.running_requests;
    point.waiting  = engine.stats.waiting_requests;
    point.kv_pages = engine.stats.device_main_kv_occupied_pages;
    if (gpus) {
        for (const GpuTelemetrySample& gpu : *gpus) {
            std::optional<double> utilization;
            if (gpu.utilization_percent) {
                utilization = static_cast<double>(*gpu.utilization_percent);
            }
            point.gpu_utilization.push_back(utilization);
            point.gpu_power.push_back(gpu.power_watts);
        }
        gpu_samples_ = std::move(gpus);
    }
    if (host) {
        point.cpu_utilization = host->cpu_utilization;
        point.cpu_temperature = host->cpu_temperature_celsius;
        point.process_cpus    = host->process_cpus;
        host_sample_          = std::move(host);
    }
    history_.push_back(std::move(point));
    while (history_.size() > kHistoryPoints) { history_.pop_front(); }
    point_base_ = reading;
}

std::optional<LiveStats::Reading> LiveStats::reading_before(SteadyClock::time_point now,
                                                            std::chrono::milliseconds age) const {
    // The newest reading at least `age` old, or the oldest one kept.
    for (auto it = recent_.rbegin(); it != recent_.rend(); ++it) {
        if (now - it->at >= age) { return *it; }
    }
    if (!recent_.empty()) { return recent_.front(); }
    return std::nullopt;
}

void LiveStats::finish(Json record) {
    finished_.push_back(std::move(record));
    while (finished_.size() > kFinishedRequests) { finished_.pop_front(); }
}

void LiveStats::request_started(const RequestLogContext& context) {
    std::lock_guard lock(mutex_);
    in_flight_[context.id] = InFlight{
        .context      = context,
        .started      = SteadyClock::now(),
        .started_unix = unix_seconds(SystemClock::now()),
    };
}

void LiveStats::request_finished(const RequestLogContext& context,
                                 const GenerationOutcome& outcome) {
    const GenerationMetrics& metrics = outcome.metrics;
    const double computed_prompt     = static_cast<double>(
        std::max(0, outcome.prompt_tokens - static_cast<int>(metrics.prefix_cache_hit_tokens)));
    const double decode_tokens =
        outcome.completion_tokens > 0 ? static_cast<double>(outcome.completion_tokens - 1) : 0.0;
    Json record{
        {"id", context.id},
        {"protocol", context.protocol},
        {"client", client_json(context.client)},
        {"stream", context.stream},
        {"end", unix_seconds(SystemClock::now())},
        {"finish",
         outcome.tool_calls.empty() ? finish_reason_name(outcome.finish_reason) : "tool_calls"},
        {"prompt_tokens", outcome.prompt_tokens},
        {"cached_tokens", metrics.prefix_cache_hit_tokens},
        {"output_tokens", outcome.completion_tokens},
        {"reasoning_tokens", outcome.reasoning_tokens},
        {"ttft_seconds", metrics.ttft_seconds},
        {"total_seconds", metrics.total_seconds},
        {"queue_seconds", metrics.engine_timing.queue_wait_seconds},
        {"prefill_tokens_per_second", metrics.prefill_seconds > 0.0
                                          ? Json(computed_prompt / metrics.prefill_seconds)
                                          : Json(nullptr)},
        {"decode_tokens_per_second", metrics.decode_seconds > 0.0
                                         ? Json(decode_tokens / metrics.decode_seconds)
                                         : Json(nullptr)},
        {"drafted_tokens", metrics.speculative_draft_tokens},
        {"accepted_tokens", metrics.speculative_accepted_tokens},
    };
    std::lock_guard lock(mutex_);
    in_flight_.erase(context.id);
    ++requests_done_;
    finish(std::move(record));
}

void LiveStats::request_failed(const RequestLogContext& context, const RequestFailure& failure) {
    const char* failure_class = failure_class_name(failure.classification);
    Json record{
        {"id", context.id},
        {"protocol", context.protocol},
        {"client", client_json(context.client)},
        {"stream", context.stream},
        {"end", unix_seconds(SystemClock::now())},
        {"finish", "error"},
        {"error", failure.error_code.empty() ? failure.error_type : failure.error_code},
        {"failure", failure_class},
        {"status", failure.http_status},
        {"prompt_tokens", context.prompt_tokens},
    };
    std::lock_guard lock(mutex_);
    const auto it = in_flight_.find(context.id);
    if (it != in_flight_.end()) {
        record["total_seconds"] =
            std::chrono::duration<double>(SteadyClock::now() - it->second.started).count();
        in_flight_.erase(it);
    }
    ++failures_[failure_class];
    finish(std::move(record));
}

void LiveStats::request_rejected(const RequestRejectionLogContext& context) {
    const std::string code = context.error.code.empty() ? context.error.type : context.error.code;
    Json record{
        {"id", context.id},
        {"protocol", context.protocol},
        {"client", client_json(context.client)},
        {"stream", context.stream},
        {"end", unix_seconds(SystemClock::now())},
        {"finish", "rejected"},
        {"error", code},
        {"status", context.error.status},
    };
    std::lock_guard lock(mutex_);
    ++rejections_[code];
    finish(std::move(record));
}

Json LiveStats::gpus_json() const {
    Json out = Json::array();
    if (!gpu_samples_) { return out; }
    for (std::size_t i = 0; i < gpu_samples_->size(); ++i) {
        const GpuTelemetrySample& gpu = (*gpu_samples_)[i];
        Json reasons                  = Json::array();
        if (gpu.clock_event_reasons) {
            for (const std::string& name : clock_event_reason_names(*gpu.clock_event_reasons)) {
                // An idle GPU reports gpu_idle; that is not a slowdown.
                if (name != "gpu_idle") { reasons.push_back(name); }
            }
        }
        out.push_back(Json{
            {"device", gpu.device},
            {"name", i < gpus_.names.size() ? Json(gpus_.names[i]) : Json(nullptr)},
            {"utilization_percent", optional_json(gpu.utilization_percent)},
            {"memory_utilization_percent", optional_json(gpu.memory_utilization_percent)},
            {"memory_used_bytes", optional_json(gpu.memory_used_bytes)},
            {"memory_total_bytes", optional_json(gpu.memory_total_bytes)},
            {"power_watts", optional_json(gpu.power_watts)},
            {"power_limit_watts", optional_json(gpu.power_limit_watts)},
            {"temperature_celsius", optional_json(gpu.temperature_celsius)},
            {"fan_percent", optional_json(gpu.fan_percent)},
            {"sm_clock_mhz", optional_json(gpu.sm_clock_mhz)},
            {"memory_clock_mhz", optional_json(gpu.memory_clock_mhz)},
            {"slowdown_reasons", std::move(reasons)},
        });
    }
    return out;
}

namespace {

Json prefix_cache_json(const ninfer::RuntimeStats& stats) {
    const std::uint64_t prompt = stats.reused_prompt_tokens + stats.computed_prefill_tokens;
    return Json{
        {"reused_tokens", stats.reused_prompt_tokens},
        {"computed_tokens", stats.computed_prefill_tokens},
        {"hit_ratio", optional_json(ratio(stats.reused_prompt_tokens, prompt))},
        {"selections",
         Json{
             {"none", stats.root_selections},
             {"private_endpoint", stats.private_endpoint_selections},
             {"private_turn_closure", stats.private_turn_closure_selections},
             {"private_response_replay", stats.private_response_replay_selections},
             {"private_long_anchor", stats.private_long_anchor_selections},
             {"shared_stable_prefix", stats.shared_stable_prefix_selections},
         }},
        {"historical_fork_hits", stats.historical_fork_hits},
        {"pressure",
         Json{
             {"private_degraded", stats.pressure_private_owners_degraded},
             {"private_evicted", stats.pressure_private_owners_evicted},
             {"shared_degraded", stats.pressure_shared_owners_degraded},
             {"shared_evicted", stats.pressure_shared_owners_evicted},
             {"checkpoints_dropped", stats.pressure_checkpoints_dropped},
             {"spilled_pages", stats.pressure_spill_pages},
         }},
    };
}

Json speculation_json(const ninfer::RuntimeStats& stats) {
    Json positions = Json::array();
    for (std::size_t i = 0; i < ninfer::kMaximumDraftPositions; ++i) {
        const std::uint64_t reached = stats.speculative_reached_per_position[i];
        if (reached == 0) { break; }
        const std::uint64_t accepted = stats.speculative_accepted_per_position[i];
        positions.push_back(Json{{"reached", reached},
                                 {"accepted", accepted},
                                 {"acceptance", optional_json(ratio(accepted, reached))}});
    }
    return Json{
        {"rounds", stats.speculative_rounds},
        {"drafted_tokens", stats.speculative_drafted_tokens},
        {"accepted_tokens", stats.speculative_accepted_tokens},
        {"fallback_steps", stats.speculative_fallback_steps},
        {"acceptance",
         optional_json(ratio(stats.speculative_accepted_tokens, stats.speculative_drafted_tokens))},
        {"draft_window_mean",
         optional_json(ratio(stats.speculative_drafted_tokens, stats.speculative_rounds))},
        {"by_position", std::move(positions)},
    };
}

Json experts_json(const ninfer::ExpertCacheRuntimeStats& expert) {
    if (!expert.present) { return Json(nullptr); }
    return Json{
        {"base_slots", expert.base_slots},
        {"elastic_columns", expert.elastic_columns},
        {"active_columns", expert.active_columns},
        {"windows", expert.windows},
        {"routes", expert.routes},
        {"device_hits", expert.device_hits},
        {"hit_ratio", optional_json(ratio(expert.device_hits, expert.routes))},
        {"missing", expert.missing},
        {"inserts", expert.inserts},
        {"fills", expert.fills},
        {"fill_bytes", expert.fill_bytes},
        {"shrink_evicted", expert.shrink_evicted},
        {"decode_routes", expert.decode_routes},
        {"decode_device_hits", expert.decode_device_hits},
        {"decode_hit_ratio", optional_json(ratio(expert.decode_device_hits, expert.decode_routes))},
        {"decode_gpu_read_experts", expert.decode_gpu_experts},
        {"decode_cpu_experts", expert.decode_cpu_experts},
        {"elastic_shrinks", expert.elastic_shrinks},
        {"elastic_returns", expert.elastic_returns},
    };
}

Json transfers_json(const ninfer::RuntimeStats& stats) {
    return Json{
        {"state_to_device_bytes", stats.state_h2d_bytes},
        {"state_to_host_bytes", stats.state_d2h_bytes},
        {"kv_to_device_bytes", stats.main_kv_h2d_bytes + stats.backend_kv_h2d_bytes},
        {"kv_to_host_bytes", stats.main_kv_d2h_bytes + stats.backend_kv_d2h_bytes},
        {"device_to_device_bytes",
         stats.state_d2d_bytes + stats.main_kv_d2d_bytes + stats.backend_kv_d2d_bytes},
    };
}

} // namespace

Json LiveStats::render(SteadyClock::time_point now, SystemClock::time_point wall,
                       std::optional<std::uint64_t> since) const {
    // The Engine figures at this instant, read outside the lock (the read never waits for the
    // worker, but it does take the Engine's own statistics lock).
    std::optional<LiveEngineReading> engine;
    if (read_) { engine = read_(); }

    std::lock_guard lock(mutex_);
    Json out;
    out["time"]           = unix_seconds(wall);
    out["started"]        = started_unix_;
    out["uptime_seconds"] = std::chrono::duration<double>(now - started_).count();
    out["server"]         = Json{
                {"model", info_.model},
                {"max_concurrency", info_.max_concurrency},
                {"max_context", info_.max_context},
                {"tensor_parallel", info_.tensor_parallel},
                {"speculative",
         info_.speculative_backend.empty()
                     ? Json(nullptr)
                     : Json{{"backend", info_.speculative_backend}, {"draft_tokens", info_.draft_tokens}}},
    };

    // Requests: the server's own in-flight table, joined with the Engine's lanes by request id.
    Json requests = Json::array();
    for (const auto& [id, flight] : in_flight_) {
        Json request{
            {"id", id},
            {"protocol", flight.context.protocol},
            {"client", client_json(flight.context.client)},
            {"stream", flight.context.stream},
            {"start", flight.started_unix},
            {"age_seconds", std::chrono::duration<double>(now - flight.started).count()},
            {"prompt_tokens", flight.context.prompt_tokens},
            {"phase", "queued"},
        };
        if (engine && flight.context.engine_request_id != 0) {
            const ninfer::RuntimeStats& stats = engine->stats;
            for (std::uint32_t i = 0; i < stats.active_request_count; ++i) {
                const ninfer::ActiveRequestStats& active = stats.active_requests[i];
                if (active.id != flight.context.engine_request_id) { continue; }
                const std::uint32_t to_compute =
                    active.prompt_tokens > active.reused_prompt_tokens
                        ? active.prompt_tokens - active.reused_prompt_tokens
                        : 0;
                request["phase"]            = ninfer::active_request_phase_name(active.phase);
                request["cached_tokens"]    = active.reused_prompt_tokens;
                request["prefilled_tokens"] = active.computed_prompt_tokens;
                request["output_tokens"]    = active.generated_tokens;
                request["prefill_progress"] =
                    to_compute != 0 ? static_cast<double>(active.computed_prompt_tokens) /
                                          static_cast<double>(to_compute)
                                    : 1.0;
                if (active.first_token_ns != 0) {
                    request["ttft_seconds"] =
                        static_cast<double>(active.first_token_ns - active.submitted_ns) * 1e-9;
                    const double decoding = seconds_between(active.first_token_ns, now);
                    if (decoding > 0.0 && active.generated_tokens > 1) {
                        request["decode_tokens_per_second"] =
                            static_cast<double>(active.generated_tokens - 1) / decoding;
                    }
                }
                break;
            }
        }
        requests.push_back(std::move(request));
    }
    out["requests"] = std::move(requests);

    Json current{{"engine", engine ? Json(engine->engine_state) : Json(nullptr)}};
    if (engine) {
        const ninfer::RuntimeStats& stats = engine->stats;
        current["running"]                = stats.running_requests;
        current["prefilling"]             = stats.prefilling_requests;
        current["decoding"]               = stats.decode_ready_requests;
        current["waiting"]                = stats.waiting_requests;
        current["materializing"]          = stats.materializing_requests;
        current["unit"]                   = Json{
                              {"kind", ninfer::runtime_unit_kind_name(stats.open_unit)},
                              {"seconds", static_cast<double>(stats.open_unit_elapsed_ns) * 1e-9},
        };

        const Reading at_now = reading_of(*engine, now);
        Json rates;
        for (const auto& [name, age] : {std::pair{"1s", std::chrono::milliseconds(1000)},
                                        std::pair{"10s", std::chrono::milliseconds(10'000)},
                                        std::pair{"60s", std::chrono::milliseconds(60'000)}}) {
            const std::optional<Reading> from = reading_before(now, age);
            rates[name]                       = from ? rates_between(*from, at_now) : Json(nullptr);
        }
        current["rates"] = std::move(rates);

        const ninfer::ExpertCacheRuntimeStats& expert = stats.expert_cache;
        const std::uint32_t page_tokens =
            info_.kv_capacity_pages != 0 ? info_.kv_capacity_tokens / info_.kv_capacity_pages : 0;
        current["kv"] = Json{
            {"page_tokens", page_tokens},
            {"capacity_pages", info_.kv_capacity_pages},
            {"device_pages", stats.device_main_kv_occupied_pages},
            {"device_lease_pages", stats.device_main_kv_lease_pages},
            {"backed_pages", expert.present && expert.elastic_kv_pages != 0
                                 ? Json(expert.elastic_kv_pages)
                                 : Json(nullptr)},
            {"device_state_slots", stats.device_state_occupied_slots},
            {"host_state_slots", stats.host_state_occupied_slots},
            {"host_bytes", stats.host_kv_occupied_bytes},
            {"host_capacity_bytes", info_.host_kv_capacity_bytes},
        };
        current["prefix_cache"] = prefix_cache_json(stats);
        current["speculation"]  = speculation_json(stats);
        current["experts"]      = experts_json(expert);
        current["transfers"]    = transfers_json(stats);
        current["totals"]       = Json{
                  {"prefill_tokens", stats.computed_prefill_tokens},
                  {"decode_tokens", stats.committed_decode_tokens},
                  {"decode_rounds", stats.decode_rounds},
                  {"prefill_units", stats.host_work.prefill_units},
        };
    }
    current["gpus"] = gpus_json();
    if (host_sample_) {
        const HostTelemetrySample& host = *host_sample_;
        current["host"]                 = Json{
                            {"cpu_count", host.cpu_count},
                            {"cpu_utilization", optional_json(host.cpu_utilization)},
                            {"process_cpus", optional_json(host.process_cpus)},
                            {"cpu_temperature_celsius", optional_json(host.cpu_temperature_celsius)},
                            {"hottest_core_celsius", optional_json(host.hottest_core_celsius)},
                            {"cpu_frequency_mhz", optional_json(host.cpu_frequency_mhz)},
                            {"memory_total_bytes", optional_json(host.memory_total_bytes)},
                            {"memory_available_bytes", optional_json(host.memory_available_bytes)},
                            {"swap_used_bytes", optional_json(host.swap_used_bytes)},
                            {"process_resident_bytes", optional_json(host.process_resident_bytes)},
        };
    } else {
        current["host"] = nullptr;
    }
    current["requests"] = Json{
        {"done", requests_done_},
        {"failed", failures_},
        {"rejected", rejections_},
    };
    out["now"] = std::move(current);

    Json finished = Json::array();
    for (auto it = finished_.rbegin(); it != finished_.rend(); ++it) { finished.push_back(*it); }
    out["finished"] = std::move(finished);

    // History, column-wise: one point a second, the oldest first.
    Json t = Json::array(), prefill = Json::array(), decode = Json::array(), batch = Json::array(),
         acceptance = Json::array(), step = Json::array(), prefix = Json::array(),
         expert = Json::array(), fills = Json::array(), transfer = Json::array(),
         running = Json::array(), waiting = Json::array(), kv_pages = Json::array(),
         cpu_utilization = Json::array(), cpu_temperature = Json::array(),
         process_cpus = Json::array();
    std::vector<Json> gpu_utilization;
    std::vector<Json> gpu_power;
    std::size_t points = 0;
    for (const Point& point : history_) {
        if (since && point.seq <= *since) { continue; }
        t.push_back(point.t);
        prefill.push_back(point.prefill);
        decode.push_back(point.decode);
        batch.push_back(point.batch);
        acceptance.push_back(optional_json(point.acceptance));
        step.push_back(optional_json(point.decode_step_ms));
        prefix.push_back(optional_json(point.prefix_hit));
        expert.push_back(optional_json(point.expert_hit));
        fills.push_back(point.expert_fill_bytes);
        transfer.push_back(point.transfer_bytes);
        running.push_back(point.running);
        waiting.push_back(point.waiting);
        kv_pages.push_back(point.kv_pages);
        cpu_utilization.push_back(optional_json(point.cpu_utilization));
        cpu_temperature.push_back(optional_json(point.cpu_temperature));
        process_cpus.push_back(optional_json(point.process_cpus));
        const std::size_t devices = std::max(point.gpu_utilization.size(), gpu_utilization.size());
        while (gpu_utilization.size() < devices) {
            // A device first seen at this point: the earlier points have no value for it.
            gpu_utilization.emplace_back(Json::array());
            gpu_power.emplace_back(Json::array());
            for (std::size_t i = 0; i < points; ++i) {
                gpu_utilization.back().push_back(nullptr);
                gpu_power.back().push_back(nullptr);
            }
        }
        for (std::size_t d = 0; d < devices; ++d) {
            gpu_utilization[d].push_back(d < point.gpu_utilization.size()
                                             ? optional_json(point.gpu_utilization[d])
                                             : Json(nullptr));
            gpu_power[d].push_back(d < point.gpu_power.size() ? optional_json(point.gpu_power[d])
                                                              : Json(nullptr));
        }
        ++points;
    }
    Json gpus = Json::array();
    for (std::size_t d = 0; d < gpu_utilization.size(); ++d) {
        gpus.push_back(Json{{"utilization_percent", std::move(gpu_utilization[d])},
                            {"power_watts", std::move(gpu_power[d])}});
    }
    out["history"] = Json{
        {"interval_seconds", std::chrono::duration<double>(kTick * kTicksPerPoint).count()},
        {"next", next_seq_ - 1},
        {"t", std::move(t)},
        {"prefill_tokens_per_second", std::move(prefill)},
        {"decode_tokens_per_second", std::move(decode)},
        {"decode_batch", std::move(batch)},
        {"draft_acceptance", std::move(acceptance)},
        {"decode_step_ms", std::move(step)},
        {"prefix_hit_ratio", std::move(prefix)},
        {"expert_decode_hit_ratio", std::move(expert)},
        {"expert_fill_bytes_per_second", std::move(fills)},
        {"transfer_bytes_per_second", std::move(transfer)},
        {"running", std::move(running)},
        {"waiting", std::move(waiting)},
        {"kv_device_pages", std::move(kv_pages)},
        {"cpu_utilization", std::move(cpu_utilization)},
        {"cpu_temperature_celsius", std::move(cpu_temperature)},
        {"process_cpus", std::move(process_cpus)},
        {"gpus", std::move(gpus)},
    };
    return out;
}

} // namespace ninfer::serve
