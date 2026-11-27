#pragma once

// GET /stats: what the server is doing now and did over the last hour, for a dashboard or a
// monitor. Everything comes from figures the Engine already publishes (RuntimeStats), the GPUs'
// NVML telemetry and the request lifecycle the server already logs; reading it never waits for
// the Engine worker.

#include "ninfer/types.h"
#include "serve/generation_service.h"
#include "serve/gpu_telemetry.h"
#include "serve/host_telemetry.h"
#include "serve/request_events.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace ninfer::serve {

// What the sampler reads from the Engine at one instant.
struct LiveEngineReading {
    ninfer::RuntimeStats stats;
    // Changes at every unload and load: the counters restarted (GenerationService).
    std::uint64_t generation = 0;
    std::string engine_state = "loaded";
};

// Static facts shown beside the figures.
struct LiveServerInfo {
    std::string model;
    std::uint32_t max_concurrency = 0;
    std::uint32_t max_context     = 0;
    std::string speculative_backend; // empty without speculative decoding
    std::uint32_t draft_tokens = 0;
    // Main KV capacity in tokens and pages (MemorySummary), and the host KV tier's bytes.
    std::uint32_t kv_capacity_tokens   = 0;
    std::uint32_t kv_capacity_pages    = 0;
    std::size_t host_kv_capacity_bytes = 0;
    std::uint32_t tensor_parallel      = 1;
};

// The GPUs: their names, and a sampler (NVML) called once a second from the sampler thread;
// empty when NVML is unavailable.
struct LiveGpuSource {
    std::vector<std::string> names;
    std::function<std::optional<std::vector<GpuTelemetrySample>>()> sample;
};

class LiveStats {
public:
    using SteadyClock = std::chrono::steady_clock;
    using SystemClock = std::chrono::system_clock;

    // One history point a second, an hour of them; a reading every 250 ms for the live rates.
    static constexpr std::chrono::milliseconds kTick{250};
    static constexpr std::size_t kTicksPerPoint    = 4;
    static constexpr std::size_t kHistoryPoints    = 3600;
    static constexpr std::size_t kRecentTicks      = 4 * 60 + 1;
    static constexpr std::size_t kFinishedRequests = 100;

    LiveStats() = default;
    ~LiveStats();
    LiveStats(const LiveStats&)            = delete;
    LiveStats& operator=(const LiveStats&) = delete;

    // `read` is called from the sampler thread and from render(); it must not block on the
    // Engine worker (GenerationService::runtime_stats() does not).
    // `gpus` and `host` are sampled once a second from the sampler thread.
    void configure(LiveServerInfo info, std::function<LiveEngineReading()> read,
                   LiveGpuSource gpus = {}, std::function<HostTelemetrySample()> host = {});
    void start();
    void stop();

    // The request lifecycle (HttpServer::record_request_*).
    void request_started(const RequestLogContext& context);
    void request_finished(const RequestLogContext& context, const GenerationOutcome& outcome);
    void request_failed(const RequestLogContext& context, const RequestFailure& failure);
    void request_rejected(const RequestRejectionLogContext& context);

    // One sampler step; public for the tests, which drive the clocks themselves.
    void tick(SteadyClock::time_point now, SystemClock::time_point wall);

    // The JSON of GET /stats. `since`: the history sequence number the client already has (the
    // `next` of its previous answer); history then carries only the points after it.
    [[nodiscard]] nlohmann::json render(SteadyClock::time_point now, SystemClock::time_point wall,
                                        std::optional<std::uint64_t> since) const;

private:
    // The counters rates are taken from, at one instant.
    struct Reading {
        SteadyClock::time_point at;
        std::uint64_t generation           = 0;
        std::uint64_t prefill_tokens       = 0;
        std::uint64_t reused_tokens        = 0;
        std::uint64_t decode_tokens        = 0;
        std::uint64_t decode_rounds        = 0;
        std::uint64_t decode_row_rounds    = 0;
        std::uint64_t drafted_tokens       = 0;
        std::uint64_t accepted_tokens      = 0;
        std::uint64_t speculative_rounds   = 0;
        std::uint64_t decode_host_ns       = 0;
        std::uint64_t decode_wait_ns       = 0;
        std::uint64_t prefill_ns           = 0;
        std::uint64_t prefill_units        = 0;
        std::uint64_t to_device_bytes      = 0;
        std::uint64_t to_host_bytes        = 0;
        std::uint64_t expert_routes        = 0;
        std::uint64_t expert_hits          = 0;
        std::uint64_t expert_decode_routes = 0;
        std::uint64_t expert_decode_hits   = 0;
        std::uint64_t expert_fills         = 0;
        std::uint64_t expert_fill_bytes    = 0;
        std::uint64_t expert_cpu           = 0;
        std::uint64_t expert_gpu           = 0;
    };

    struct Point {
        std::uint64_t seq = 0;
        double t          = 0.0; // unix seconds
        double prefill    = 0.0; // tokens/s
        double decode     = 0.0; // tokens/s
        double batch      = 0.0; // mean decode rows per round
        std::optional<double> acceptance;
        std::optional<double> decode_step_ms;
        std::optional<double> prefix_hit;
        std::optional<double> expert_hit; // decode routes served from a slot
        double expert_fill_bytes = 0.0;   // per second
        double transfer_bytes    = 0.0;   // KV and state, both directions, per second
        std::uint32_t running    = 0;
        std::uint32_t waiting    = 0;
        std::uint32_t kv_pages   = 0;
        std::vector<std::optional<double>> gpu_utilization;
        std::vector<std::optional<double>> gpu_power;
        std::optional<double> cpu_utilization;
        std::optional<double> cpu_temperature;
        std::optional<double> process_cpus;
    };

    struct InFlight {
        RequestLogContext context;
        SteadyClock::time_point started;
        double started_unix = 0.0;
    };

    [[nodiscard]] static Reading reading_of(const LiveEngineReading& engine,
                                            SteadyClock::time_point at);
    [[nodiscard]] static nlohmann::json rates_between(const Reading& from, const Reading& to);
    [[nodiscard]] std::optional<Reading> reading_before(SteadyClock::time_point now,
                                                        std::chrono::milliseconds age) const;
    [[nodiscard]] nlohmann::json gpus_json() const;
    void finish(nlohmann::json record);
    void run();

    LiveServerInfo info_;
    std::function<LiveEngineReading()> read_;
    LiveGpuSource gpus_;
    std::function<HostTelemetrySample()> host_;
    SteadyClock::time_point started_{SteadyClock::now()};
    double started_unix_ = 0.0;

    mutable std::mutex mutex_;
    std::deque<Reading> recent_; // kTick apart, the newest last
    std::deque<Point> history_;  // a second apart, the newest last
    std::uint64_t next_seq_ = 1;
    std::size_t ticks_      = 0;
    std::optional<Reading> point_base_;
    std::optional<std::vector<GpuTelemetrySample>> gpu_samples_; // the latest
    std::optional<HostTelemetrySample> host_sample_;             // the latest
    std::map<std::uint64_t, InFlight> in_flight_;
    std::deque<nlohmann::json> finished_;
    std::uint64_t requests_done_ = 0;
    std::map<std::string, std::uint64_t> failures_;   // by failure class
    std::map<std::string, std::uint64_t> rejections_; // by error code

    std::mutex thread_mutex_;
    std::condition_variable thread_cv_;
    std::thread thread_;
    bool stopping_ = false;
};

} // namespace ninfer::serve
