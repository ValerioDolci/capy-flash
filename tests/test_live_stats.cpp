#include "serve/live_stats.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <unistd.h>

namespace {

using namespace ninfer::serve;
using Json = nlohmann::json;
using std::chrono::milliseconds;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

bool near(const Json& value, double expected) {
    return value.is_number() && std::abs(value.get<double>() - expected) < 1e-9;
}

struct FakeEngine {
    LiveEngineReading reading;
};

int rate_and_history_tests() {
    int failures = 0;
    FakeEngine engine;
    LiveStats live;
    live.configure(
        LiveServerInfo{.model               = "m",
                       .max_concurrency     = 2,
                       .max_context         = 1024,
                       .speculative_backend = "mtp",
                       .draft_tokens        = 3},
        [&engine] { return engine.reading; },
        LiveGpuSource{.names = {"Test GPU"}, .sample = [] {
                          GpuTelemetrySample gpu;
                          gpu.power_watts         = 120.0;
                          gpu.utilization_percent = 90;
                          gpu.clock_event_reasons = 0x1 | 0x4; // idle, sw power cap
                          return std::optional<std::vector<GpuTelemetrySample>>(
                              std::vector<GpuTelemetrySample>{gpu});
                      }});

    const auto t0   = LiveStats::SteadyClock::time_point{} + std::chrono::hours(1);
    const auto wall = LiveStats::SystemClock::time_point{} + std::chrono::hours(500'000);
    ninfer::RuntimeStats& stats = engine.reading.stats;
    // Eight ticks of 250 ms: two history points. Per tick: 100 prefill tokens, 10 decode tokens in
    // 4 rounds of 2 rows, 6 drafts of which 3 accepted in 2 verified rounds.
    for (int i = 0; i <= 8; ++i) {
        stats.computed_prefill_tokens         = 100ULL * i;
        stats.committed_decode_tokens         = 10ULL * i;
        stats.decode_rounds                   = 4ULL * i;
        stats.decode_row_rounds               = 8ULL * i;
        stats.speculative_drafted_tokens      = 6ULL * i;
        stats.speculative_accepted_tokens     = 3ULL * i;
        stats.speculative_rounds              = 2ULL * i;
        stats.reused_prompt_tokens            = 300ULL * i;
        stats.host_work.decode_host_ns        = 2'000'000ULL * i; // 4 rounds: 0.5 ms host each
        stats.host_work.decode_device_wait_ns = 6'000'000ULL * i; // and 1.5 ms waiting
        stats.main_kv_h2d_bytes               = 1000ULL * i;
        stats.expert_cache.present            = true;
        stats.expert_cache.decode_routes      = 80ULL * i;
        stats.expert_cache.decode_device_hits = 60ULL * i;
        stats.expert_cache.fill_bytes         = 4096ULL * i;
        stats.running_requests                = 1;
        live.tick(t0 + milliseconds(250 * i), wall + milliseconds(250 * i));
    }
    const auto now = t0 + milliseconds(250 * 8);
    Json out       = live.render(now, wall + milliseconds(2000), std::nullopt);

    const Json& history = out.at("history");
    failures += check(history.at("t").size() == 2, "eight ticks must make two history points");
    failures += check(near(history.at("prefill_tokens_per_second").at(0), 400.0),
                      "history prefill rate is tokens over the second");
    failures += check(near(history.at("decode_tokens_per_second").at(1), 40.0),
                      "history decode rate is tokens over the second");
    failures +=
        check(near(history.at("decode_batch").at(0), 2.0), "history batch is rows per round");
    failures += check(near(history.at("draft_acceptance").at(0), 0.5),
                      "history acceptance is accepted over drafted");
    failures += check(history.at("next") == 2, "next names the newest point");
    failures += check(near(history.at("decode_step_ms").at(0), 2.0),
                      "history step time is host and device wait over rounds");
    failures += check(near(history.at("prefix_hit_ratio").at(0), 0.75),
                      "history prefix hits are reused over reused and computed tokens");
    failures += check(near(history.at("expert_decode_hit_ratio").at(0), 0.75),
                      "history expert hits are decode routes served from a slot");
    failures += check(near(history.at("transfer_bytes_per_second").at(0), 4000.0),
                      "history transfers are bytes per second");
    failures += check(history.at("gpus").size() == 1 &&
                          near(history.at("gpus").at(0).at("power_watts").at(1), 120.0),
                      "history keeps each GPU's power");

    const Json& one = out.at("now").at("rates").at("1s");
    failures += check(near(one.at("seconds"), 1.0), "the 1 s window spans four ticks");
    failures += check(near(one.at("prefill_tokens_per_second"), 400.0), "live prefill rate");
    failures += check(near(one.at("decode_tokens_per_second"), 40.0), "live decode rate");
    failures += check(near(one.at("accepted_per_round"), 1.5), "accepted drafts per round");
    failures += check(near(one.at("draft_window_mean"), 3.0), "drafts per verified round");
    failures += check(near(one.at("decode_step_seconds"), 0.002), "live decode step time");
    failures += check(near(one.at("decode_host_share"), 0.25), "host share of the decode step");
    failures += check(near(one.at("expert_fill_bytes_per_second"), 16384.0), "live expert fills");
    const Json& gpu = out.at("now").at("gpus").at(0);
    failures += check(gpu.at("name") == "Test GPU" && near(gpu.at("power_watts"), 120.0) &&
                          gpu.at("slowdown_reasons") == Json::array({"sw_power_cap"}),
                      "the latest GPU sample, its name, and slowdowns without gpu_idle");
    failures += check(out.at("now").at("rates").at("60s").at("seconds").get<double>() <= 2.0,
                      "a window longer than the kept readings starts at the oldest");

    Json later = live.render(now, wall + milliseconds(2000), std::uint64_t{1});
    failures +=
        check(later.at("history").at("t").size() == 1, "since returns only the points after it");
    failures +=
        check(out.at("server").at("speculative").at("draft_tokens") == 3, "server facts are shown");

    // An Engine rebuilt by unload and load: the counters restarted; no rate across the two.
    engine.reading.generation = 1;
    stats                     = ninfer::RuntimeStats{};
    live.tick(now + milliseconds(250), wall + milliseconds(2250));
    Json rebuilt = live.render(now + milliseconds(250), wall + milliseconds(2250), std::nullopt);
    failures +=
        check(rebuilt.at("now").at("rates").at("1s").is_null(), "no rate across two Engines");
    return failures;
}

void write_file(const std::filesystem::path& path, const std::string& text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path) << text;
}

int host_tests() {
    int failures   = 0;
    const auto cpu = parse_proc_stat_cpu("cpu  100 0 50 800 50 0 0 0 0 0\ncpu0 1 2 3 4\n");
    failures += check(cpu && cpu->total == 1000 && cpu->idle == 850, "aggregate cpu line");
    failures += check(!parse_proc_stat_cpu("cpu0 1 2 3 4\n"), "only the aggregate line counts");
    const auto ticks = parse_process_cpu_ticks(
        "4242 (ninfer serve (x)) S 1 4242 4242 0 -1 4194560 100 0 0 0 700 300 0 0 20 0 40\n");
    failures += check(ticks == 1000U, "utime + stime after a comm with spaces and parentheses");
    failures += check(parse_kib_field("MemTotal:  1000 kB\nMemAvailable:    250 kB\n",
                                      "MemAvailable") == 250U * 1024U,
                      "meminfo field in bytes");
    failures += check(!parse_kib_field("MemTotal: 1 kB\n", "Mem"), "a field matches whole");

    // A fake /proc and /sys: Intel coretemp with a package and two cores, two CPUs' frequency.
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / ("ninfer-host-" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);
    write_file(root / "sys/class/hwmon/hwmon0/name", "acpitz\n");
    write_file(root / "sys/class/hwmon/hwmon0/temp1_label", "Package id 0\n");
    write_file(root / "sys/class/hwmon/hwmon0/temp1_input", "99000\n"); // not a CPU driver
    write_file(root / "sys/class/hwmon/hwmon1/name", "coretemp\n");
    write_file(root / "sys/class/hwmon/hwmon1/temp1_label", "Package id 0\n");
    write_file(root / "sys/class/hwmon/hwmon1/temp1_input", "61000\n");
    write_file(root / "sys/class/hwmon/hwmon1/temp2_label", "Core 0\n");
    write_file(root / "sys/class/hwmon/hwmon1/temp2_input", "58000\n");
    write_file(root / "sys/class/hwmon/hwmon1/temp3_label", "Core 4\n");
    write_file(root / "sys/class/hwmon/hwmon1/temp3_input", "64000\n");
    write_file(root / "sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "3000000\n");
    write_file(root / "sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq", "5000000\n");
    write_file(root / "proc/stat", "cpu  100 0 0 900 0 0 0 0 0 0\n");
    write_file(root / "proc/meminfo",
               "MemTotal: 1000 kB\nMemAvailable: 400 kB\nSwapTotal: 100 kB\nSwapFree: 60 kB\n");
    write_file(root / "proc/self/status", "Name: x\nVmRSS:    300 kB\n");
    HostTelemetry host(root);
    HostTelemetrySample first = host.sample();
    failures += check(!first.cpu_utilization, "the first sample only sets the baseline");
    failures += check(first.cpu_temperature_celsius == 61.0, "the CPU driver's package sensor");
    failures += check(first.hottest_core_celsius == 64.0, "the hottest core");
    failures += check(first.cpu_frequency_mhz == 4000.0, "mean frequency in MHz");
    failures += check(first.memory_available_bytes == 400U * 1024U &&
                          first.swap_used_bytes == 40U * 1024U &&
                          first.process_resident_bytes == 300U * 1024U,
                      "memory, swap and resident bytes");
    write_file(root / "proc/stat", "cpu  400 0 0 1600 0 0 0 0 0 0\n");
    const HostTelemetrySample second = host.sample();
    failures += check(second.cpu_utilization && std::abs(*second.cpu_utilization - 0.3) < 1e-9,
                      "utilization is busy over all time since the previous sample");
    std::filesystem::remove_all(root);
    return failures;
}

int position_tests() {
    int failures = 0;
    ninfer::SpeculativeTotals totals;
    totals.record(3, 3); // all three accepted
    totals.record(3, 1); // the second rejected: the third is never reached
    totals.record(2, 0); // the first rejected
    failures += check(totals.reached[0] == 3 && totals.accepted[0] == 2, "position 1");
    failures += check(totals.reached[1] == 2 && totals.accepted[1] == 1, "position 2");
    failures += check(totals.reached[2] == 1 && totals.accepted[2] == 1,
                      "position 3 is reached only after two accepted drafts");
    failures += check(totals.drafted_tokens == 8 && totals.accepted_tokens == 4, "token totals");

    FakeEngine engine;
    engine.reading.stats.speculative_reached_per_position  = totals.reached;
    engine.reading.stats.speculative_accepted_per_position = totals.accepted;
    LiveStats live;
    live.configure(LiveServerInfo{.model = "m"}, [&engine] { return engine.reading; });
    const Json out =
        live.render(LiveStats::SteadyClock::now(), LiveStats::SystemClock::now(), std::nullopt);
    const Json& positions = out.at("now").at("speculation").at("by_position");
    failures += check(positions.size() == 3, "positions end at the last one reached");
    failures += check(near(positions.at(1).at("acceptance"), 0.5), "acceptance at position 2");
    failures += check(out.at("now").at("experts").is_null(), "no slot cache, no experts");
    return failures;
}

int request_tests() {
    int failures = 0;
    FakeEngine engine;
    LiveStats live;
    live.configure(LiveServerInfo{.model = "m"}, [&engine] { return engine.reading; });

    RequestLogContext context;
    context.id                = 7;
    context.protocol          = "openai-chat";
    context.prompt_tokens     = 1000;
    context.engine_request_id = 42;
    live.request_started(context);

    const auto now              = LiveStats::SteadyClock::now();
    ninfer::RuntimeStats& stats = engine.reading.stats;
    stats.active_request_count  = 1;
    stats.active_requests[0]    = ninfer::ActiveRequestStats{
           .id                     = 42,
           .phase                  = ninfer::ActiveRequestPhase::Prefill,
           .prompt_tokens          = 1000,
           .reused_prompt_tokens   = 200,
           .computed_prompt_tokens = 400,
    };
    stats.open_unit            = ninfer::RuntimeUnitKind::Prefill;
    stats.open_unit_elapsed_ns = 1'500'000'000;
    Json out                   = live.render(now, LiveStats::SystemClock::now(), std::nullopt);
    const Json& request        = out.at("requests").at(0);
    failures += check(request.at("phase") == "prefill", "the Engine phase is joined by id");
    failures += check(near(request.at("prefill_progress"), 0.5),
                      "progress counts only the prompt tokens to compute");
    failures += check(out.at("now").at("unit").at("kind") == "prefill", "the open unit is shown");
    failures += check(near(out.at("now").at("unit").at("seconds"), 1.5), "the open unit's age");

    stats.active_requests[0].phase            = ninfer::ActiveRequestPhase::Decode;
    stats.active_requests[0].generated_tokens = 11;
    const auto ns                             = [](LiveStats::SteadyClock::time_point at) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(at.time_since_epoch()).count();
    };
    stats.active_requests[0].submitted_ns   = ns(now - std::chrono::seconds(3));
    stats.active_requests[0].first_token_ns = ns(now - std::chrono::seconds(2));
    out = live.render(now, LiveStats::SystemClock::now(), std::nullopt);
    failures +=
        check(near(out.at("requests").at(0).at("ttft_seconds"), 1.0), "live time to first token");
    failures += check(near(out.at("requests").at(0).at("decode_tokens_per_second"), 5.0),
                      "live decode rate counts the tokens after the first");

    GenerationOutcome outcome;
    outcome.prompt_tokens                       = 1000;
    outcome.completion_tokens                   = 101;
    outcome.finish_reason                       = ninfer::FinishReason::StopToken;
    outcome.metrics.prefix_cache_hit_tokens     = 200;
    outcome.metrics.prefill_seconds             = 2.0;
    outcome.metrics.decode_seconds              = 4.0;
    outcome.metrics.speculative_draft_tokens    = 90;
    outcome.metrics.speculative_accepted_tokens = 60;
    live.request_finished(context, outcome);
    out = live.render(now, LiveStats::SystemClock::now(), std::nullopt);
    failures += check(out.at("requests").empty(), "a finished request leaves the live table");
    const Json& finished = out.at("finished").at(0);
    failures += check(finished.at("finish") == "stop_token", "finish reason is recorded");
    failures += check(near(finished.at("prefill_tokens_per_second"), 400.0),
                      "finished prefill rate counts computed tokens");
    failures += check(near(finished.at("decode_tokens_per_second"), 25.0),
                      "finished decode rate counts the tokens after the first");
    failures += check(out.at("now").at("requests").at("done") == 1, "done requests count");

    RequestLogContext failed = context;
    failed.id                = 8;
    live.request_started(failed);
    RequestFailure failure;
    failure.error_code  = "context_length_exceeded";
    failure.http_status = 400;
    live.request_failed(failed, failure);
    out = live.render(now, LiveStats::SystemClock::now(), std::nullopt);
    failures += check(out.at("finished").at(0).at("finish") == "error", "failures are listed");
    failures += check(out.at("now").at("requests").at("failed").at("internal") == 1,
                      "failed requests count by class");

    RequestRejectionLogContext rejected;
    rejected.id           = 9;
    rejected.protocol     = "openai_chat_completions";
    rejected.error.code   = "invalid_parameter";
    rejected.error.status = 400;
    live.request_rejected(rejected);
    out = live.render(now, LiveStats::SystemClock::now(), std::nullopt);
    failures += check(out.at("finished").at(0).at("finish") == "rejected", "rejections are listed");
    failures += check(out.at("now").at("requests").at("rejected").at("invalid_parameter") == 1,
                      "rejections count by code");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += rate_and_history_tests();
    failures += position_tests();
    failures += host_tests();
    failures += request_tests();
    if (failures != 0) {
        std::cerr << failures << " live stats check(s) failed\n";
        return 1;
    }
    std::cout << "live stats tests passed\n";
    return 0;
}
