#pragma once

// CPU and host memory samples for GET /stats, read from Linux's /proc and /sys (hwmon, cpufreq):
// how busy the CPUs are (the routed experts that are not in a GPU slot are computed there), how
// hot, and how much memory is left. A value is empty when the host does not expose it. Sampling
// runs on the live statistics thread, never on the request path.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

struct HostTelemetrySample {
    std::uint32_t cpu_count = 0; // online logical CPUs
    // Busy share of every CPU since the previous sample (0..1).
    std::optional<double> cpu_utilization;
    // This process's CPU time over the same wall time, in CPUs (2.5 = two and a half busy).
    std::optional<double> process_cpus;
    // The package (Intel coretemp "Package id N", the hottest; AMD k10temp Tctl/Tdie) and the
    // hottest core.
    std::optional<double> cpu_temperature_celsius;
    std::optional<double> hottest_core_celsius;
    // Mean current frequency of the CPUs cpufreq reports.
    std::optional<double> cpu_frequency_mhz;
    std::optional<std::uint64_t> memory_total_bytes;
    std::optional<std::uint64_t> memory_available_bytes;
    std::optional<std::uint64_t> swap_used_bytes;
    std::optional<std::uint64_t> process_resident_bytes;
};

// Parsing, public for the tests.
struct ProcStatCpu {
    std::uint64_t total = 0; // every field of the aggregate "cpu" line
    std::uint64_t idle  = 0; // idle + iowait
};

[[nodiscard]] std::optional<ProcStatCpu> parse_proc_stat_cpu(std::string_view proc_stat);
// utime + stime of /proc/self/stat, in clock ticks.
[[nodiscard]] std::optional<std::uint64_t> parse_process_cpu_ticks(std::string_view self_stat);
// A /proc/meminfo or /proc/self/status field ("MemAvailable", "VmRSS"), in bytes.
[[nodiscard]] std::optional<std::uint64_t> parse_kib_field(std::string_view text,
                                                           std::string_view key);

class HostTelemetry {
public:
    // `root` replaces "/" for the tests (a directory holding proc/ and sys/).
    explicit HostTelemetry(std::filesystem::path root = "/");

    // The rates cover the time since the previous call, so the first call has none. Not
    // thread-safe: one caller owns that baseline.
    [[nodiscard]] HostTelemetrySample sample();

private:
    [[nodiscard]] std::string read(const std::filesystem::path& path) const;

    std::filesystem::path root_;
    long clock_ticks_per_second_ = 100;
    std::vector<std::filesystem::path> package_sensors_; // temp*_input
    std::vector<std::filesystem::path> core_sensors_;
    std::vector<std::filesystem::path> frequency_files_; // scaling_cur_freq, kHz
    std::optional<ProcStatCpu> previous_cpu_;
    std::optional<std::uint64_t> previous_process_ticks_;
    std::optional<std::uint64_t> previous_ns_;
};

} // namespace ninfer::serve
