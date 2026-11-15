// Performance bench for the BF16 depthwise causal width-4 convolution + SiLU split-output Op.
//
// Example:
//   ./ninfer_causal_conv1d_silu_bench --cache cold --channels 10240 --tokens 1,16,1024
// Printed logical GB/s is informational; NCU determines the applicable resource roofline.
#include "ninfer/ops/causal_conv1d_silu.h"
#include "core/device.h"
#include "ninfer_bench_common.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::size_t kFlushBytes = std::size_t{256} << 20;

struct Options {
    std::int32_t channels = 10240;
    std::int32_t tokens   = 1024;
    std::vector<std::int32_t> token_list;
    bool cold = false;
};

// The candidate partition for a channel extent: q and k of one fifth of the channels each, v of
// the rest, as in the GDN caller. Which partitions are registered is the entry's business, not the
// benchmark's: this builds the obvious one and lets the entry accept or reject it.
bool split_partition(std::int32_t channels, std::int32_t& key_dim, std::int32_t& value_dim) {
    key_dim   = channels / 5;
    value_dim = channels - 2 * key_dim;
    return key_dim > 0;
}

Result time_stage(const Options& options, const launch_fn& launch, double bytes_moved) {
    if (!options.cold) { return bench_loop(launch, bytes_moved); }
    constexpr int kColdWarmup = 20;
    constexpr int kColdRepeat = 40;
    bench::L2FlushBuffer flush(kFlushBytes);
    const ColdTiming timing = measure_cold_launch(launch, flush, nullptr, kColdWarmup, kColdRepeat);
    Result result;
    result.n_runs        = kColdRepeat;
    result.median_us     = timing.median_us;
    result.min_us        = timing.min_us;
    result.p95_us        = timing.p95_us;
    const double seconds = timing.median_us * 1e-6;
    result.gbs           = seconds > 0.0 ? bytes_moved / seconds / 1e9 : 0.0;
    return result;
}

std::string cache_tag(const Options& options, const char* entry) {
    return options.cold ? std::string(entry) + "-cold" : std::string(entry);
}

DeviceBuffer make_varied_bf16(std::size_t n, std::uint32_t seed) {
    std::vector<std::uint16_t> h(n);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < n; ++i) {
        state         = state * 1664525u + 1013904223u;
        const float u = static_cast<float>((state >> 8) & 0x00ffffffu) * (1.0f / 16777216.0f);
        h[i]          = f32_to_bf16(2.0f * u - 1.0f);
    }
    DeviceBuffer d(n * 2u);
    cudaMemcpy(d.p, h.data(), n * 2u, cudaMemcpyHostToDevice);
    return d;
}

std::string shape_tag(const char* mode, std::int32_t channels, std::int32_t tokens) {
    return std::string("causal_conv1d ") + mode + " [" + std::to_string(channels) + "," +
           std::to_string(tokens) + ",1]";
}

// Split-output prefill. The destinations partition the channel extent the way the GDN caller
// does: two key-sized ranges followed by the value range.
void run_split(const Options& options) {
    std::int32_t key_dim   = 0;
    std::int32_t value_dim = 0;
    if (!split_partition(options.channels, key_dim, value_dim)) {
        std::printf("SKIP: --channels %d leaves no value rows\n", options.channels);
        return;
    }
    const std::size_t n       = static_cast<std::size_t>(options.channels) * options.tokens;
    const std::size_t state_n = static_cast<std::size_t>(options.channels) * 3u;

    DeviceBuffer x = make_varied_bf16(n, 0x12345678U);
    DeviceBuffer weight =
        make_varied_bf16(static_cast<std::size_t>(options.channels) * 4u, 0x87654321U);
    DeviceBuffer state_in  = make_varied_bf16(state_n, 0x31415926U);
    DeviceBuffer state_out = make_zeros(state_n * 2u);
    DeviceBuffer q         = make_zeros(static_cast<std::size_t>(key_dim) * options.tokens * 2u);
    DeviceBuffer k         = make_zeros(static_cast<std::size_t>(key_dim) * options.tokens * 2u);
    DeviceBuffer v         = make_zeros(static_cast<std::size_t>(value_dim) * options.tokens * 2u);

    Tensor tx(x.p, DType::BF16, {options.channels, options.tokens});
    Tensor tw(weight.p, DType::BF16, {options.channels, 4});
    Tensor tin(state_in.p, DType::BF16, {options.channels, 3});
    Tensor tout_state(state_out.p, DType::BF16, {options.channels, 3});
    Tensor tq(q.p, DType::BF16, {key_dim, options.tokens});
    Tensor tk(k.p, DType::BF16, {key_dim, options.tokens});
    Tensor tv(v.p, DType::BF16, {value_dim, options.tokens});

    // x in, q+k+v out, the four-tap weight and both states. NCU counters remain the performance
    // authority.
    const double bytes = 4.0 * static_cast<double>(n) + 20.0 * options.channels;
    const Result r     = time_stage(
        options,
        [&](cudaStream_t s) {
            ops::causal_conv1d_silu_split(tx, tw, tin, tout_state, tq, tk, tv, s);
        },
        bytes);
    const std::string tag =
        shape_tag(cache_tag(options, "split").c_str(), options.channels, options.tokens);
    print_result(tag.c_str(), r);
}

void print_usage(const char* program) {
    std::fprintf(stderr, "usage: %s [--cache warm|cold] [--channels C] [--tokens T[,T...]]\n",
                 program);
}

bool parse_positive(const char* text, std::int32_t& value) {
    const long parsed = std::strtol(text, nullptr, 10);
    if (parsed <= 0 || parsed > INT32_MAX) { return false; }
    value = static_cast<std::int32_t>(parsed);
    return true;
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--cache") && i + 1 < argc) {
            const char* value = argv[++i];
            if (!std::strcmp(value, "cold")) {
                options.cold = true;
            } else if (!std::strcmp(value, "warm")) {
                options.cold = false;
            } else {
                return false;
            }
        } else if (!std::strcmp(argv[i], "--channels") && i + 1 < argc) {
            if (!parse_positive(argv[++i], options.channels)) { return false; }
        } else if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) {
            options.token_list.clear();
            const char* cursor = argv[++i];
            while (*cursor != 0) {
                char* end         = nullptr;
                const long parsed = std::strtol(cursor, &end, 10);
                if (end == cursor || parsed <= 0 || parsed > INT32_MAX) { return false; }
                options.token_list.push_back(static_cast<std::int32_t>(parsed));
                cursor = (*end == ',') ? end + 1 : end;
            }
            if (options.token_list.empty()) { return false; }
        } else {
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    Options options;
    if (!parse_options(argc, argv, options)) {
        print_usage(argv[0]);
        return 2;
    }

    std::vector<std::int32_t> tokens = options.token_list;
    if (tokens.empty()) { tokens.push_back(options.tokens); }
    for (const std::int32_t t : tokens) {
        Options point = options;
        point.tokens  = t;
        run_split(point);
    }
    return 0;
}
