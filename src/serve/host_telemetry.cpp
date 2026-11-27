#include "serve/host_telemetry.h"

#include <unistd.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <fstream>
#include <iterator>
#include <sstream>
#include <system_error>

namespace ninfer::serve {
namespace {

namespace fs = std::filesystem;

std::optional<std::uint64_t> to_number(std::string_view text) {
    std::uint64_t value     = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end == text.data()) { return std::nullopt; }
    return value;
}

std::string_view trim(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) {
        text.remove_prefix(1);
    }
    while (!text.empty() && (text.back() == '\n' || text.back() == ' ')) { text.remove_suffix(1); }
    return text;
}

// The hwmon drivers whose readings are the CPU's: Intel coretemp, AMD k10temp and zenpower.
bool cpu_sensor_driver(std::string_view name) {
    return name == "coretemp" || name == "k10temp" || name == "zenpower";
}

bool package_label(std::string_view label) {
    return label.starts_with("Package id") || label == "Tctl" || label == "Tdie";
}

} // namespace

std::optional<ProcStatCpu> parse_proc_stat_cpu(std::string_view proc_stat) {
    // "cpu  user nice system idle iowait irq softirq steal guest guest_nice"; guest time is
    // already counted in user and nice.
    if (!proc_stat.starts_with("cpu ")) { return std::nullopt; }
    std::string_view line = proc_stat.substr(0, proc_stat.find('\n')).substr(4);
    ProcStatCpu out;
    for (int field = 0; field < 8; ++field) {
        line                    = trim(line);
        const std::size_t space = line.find(' ');
        const auto value        = to_number(line.substr(0, space));
        if (!value) { return field >= 4 ? std::optional<ProcStatCpu>(out) : std::nullopt; }
        out.total += *value;
        if (field == 3 || field == 4) { out.idle += *value; }
        if (space == std::string_view::npos) { break; }
        line = line.substr(space);
    }
    return out;
}

std::optional<std::uint64_t> parse_process_cpu_ticks(std::string_view self_stat) {
    // "pid (comm) state ppid ...": comm may hold spaces and parentheses, so count fields after
    // the last ')'. utime and stime are fields 14 and 15, the 12th and 13th after it.
    const std::size_t close = self_stat.rfind(')');
    if (close == std::string_view::npos) { return std::nullopt; }
    std::string_view rest = self_stat.substr(close + 1);
    std::optional<std::uint64_t> utime;
    for (int field = 1; field <= 13; ++field) {
        rest                         = trim(rest);
        const std::size_t space      = rest.find(' ');
        const std::string_view token = rest.substr(0, space);
        if (field == 12) { utime = to_number(token); }
        if (field == 13) {
            const auto stime = to_number(token);
            if (!utime || !stime) { return std::nullopt; }
            return *utime + *stime;
        }
        if (space == std::string_view::npos) { return std::nullopt; }
        rest = rest.substr(space);
    }
    return std::nullopt;
}

std::optional<std::uint64_t> parse_kib_field(std::string_view text, std::string_view key) {
    std::size_t at = 0;
    while (at < text.size()) {
        const std::size_t end      = std::min(text.find('\n', at), text.size());
        const std::string_view row = text.substr(at, end - at);
        if (row.starts_with(key) && row.size() > key.size() && row[key.size()] == ':') {
            const std::string_view value = trim(row.substr(key.size() + 1));
            const auto kib               = to_number(value.substr(0, value.find(' ')));
            if (!kib) { return std::nullopt; }
            return *kib * 1024;
        }
        at = end + 1;
    }
    return std::nullopt;
}

HostTelemetry::HostTelemetry(fs::path root) : root_(std::move(root)) {
    if (const long ticks = ::sysconf(_SC_CLK_TCK); ticks > 0) { clock_ticks_per_second_ = ticks; }
    std::error_code error;
    const fs::path hwmon = root_ / "sys/class/hwmon";
    for (const auto& entry : fs::directory_iterator(hwmon, error)) {
        if (!cpu_sensor_driver(trim(read(entry.path() / "name")))) { continue; }
        for (const auto& file : fs::directory_iterator(entry.path(), error)) {
            const std::string name = file.path().filename().string();
            if (!name.starts_with("temp") || !name.ends_with("_label")) { continue; }
            const std::string label = std::string(trim(read(file.path())));
            const fs::path input =
                file.path().parent_path() / (name.substr(0, name.size() - 6) + "_input");
            if (package_label(label)) {
                package_sensors_.push_back(input);
            } else if (label.starts_with("Core")) {
                core_sensors_.push_back(input);
            }
        }
    }
    for (const auto& entry : fs::directory_iterator(root_ / "sys/devices/system/cpu", error)) {
        const std::string name = entry.path().filename().string();
        if (name.size() > 3 && name.starts_with("cpu") &&
            std::all_of(name.begin() + 3, name.end(),
                        [](char c) { return c >= '0' && c <= '9'; })) {
            const fs::path frequency = entry.path() / "cpufreq/scaling_cur_freq";
            if (fs::exists(frequency, error)) { frequency_files_.push_back(frequency); }
        }
    }
}

std::string HostTelemetry::read(const fs::path& path) const {
    std::ifstream in(path);
    if (!in) { return {}; }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

HostTelemetrySample HostTelemetry::sample() {
    HostTelemetrySample out;
    const std::uint64_t now_ns =
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now().time_since_epoch())
                                       .count());
    if (const long online = ::sysconf(_SC_NPROCESSORS_ONLN); online > 0) {
        out.cpu_count = static_cast<std::uint32_t>(online);
    }

    const std::optional<ProcStatCpu> cpu = parse_proc_stat_cpu(read(root_ / "proc/stat"));
    if (cpu && previous_cpu_ && cpu->total > previous_cpu_->total &&
        cpu->idle >= previous_cpu_->idle) {
        const double total  = static_cast<double>(cpu->total - previous_cpu_->total);
        const double idle   = static_cast<double>(cpu->idle - previous_cpu_->idle);
        out.cpu_utilization = std::clamp(1.0 - idle / total, 0.0, 1.0);
    }
    previous_cpu_ = cpu;

    const std::optional<std::uint64_t> ticks =
        parse_process_cpu_ticks(read(root_ / "proc/self/stat"));
    if (ticks && previous_process_ticks_ && previous_ns_ && *ticks >= *previous_process_ticks_ &&
        now_ns > *previous_ns_) {
        const double cpu_seconds = static_cast<double>(*ticks - *previous_process_ticks_) /
                                   static_cast<double>(clock_ticks_per_second_);
        out.process_cpus = cpu_seconds / (static_cast<double>(now_ns - *previous_ns_) * 1e-9);
    }
    previous_process_ticks_ = ticks;
    previous_ns_            = now_ns;

    const auto hottest = [this](const std::vector<fs::path>& sensors) -> std::optional<double> {
        std::optional<double> best;
        for (const fs::path& sensor : sensors) {
            if (const auto millidegrees = to_number(trim(read(sensor)))) {
                const double celsius = static_cast<double>(*millidegrees) * 1e-3;
                if (!best || celsius > *best) { best = celsius; }
            }
        }
        return best;
    };
    out.cpu_temperature_celsius = hottest(package_sensors_);
    out.hottest_core_celsius    = hottest(core_sensors_);

    double khz        = 0.0;
    std::size_t count = 0;
    for (const fs::path& file : frequency_files_) {
        if (const auto value = to_number(trim(read(file)))) {
            khz += static_cast<double>(*value);
            ++count;
        }
    }
    if (count != 0) { out.cpu_frequency_mhz = khz / static_cast<double>(count) * 1e-3; }

    const std::string meminfo  = read(root_ / "proc/meminfo");
    out.memory_total_bytes     = parse_kib_field(meminfo, "MemTotal");
    out.memory_available_bytes = parse_kib_field(meminfo, "MemAvailable");
    const auto swap_total      = parse_kib_field(meminfo, "SwapTotal");
    const auto swap_free       = parse_kib_field(meminfo, "SwapFree");
    if (swap_total && swap_free && *swap_total >= *swap_free) {
        out.swap_used_bytes = *swap_total - *swap_free;
    }
    out.process_resident_bytes = parse_kib_field(read(root_ / "proc/self/status"), "VmRSS");
    return out;
}

} // namespace ninfer::serve
