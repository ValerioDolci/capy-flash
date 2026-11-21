// Public-Op benchmark for relative-sink attention. Every measured graph is one public
// relative_sink_attention_windowed() or relative_sink_attention_paged() call, timed cold
// (L2 evicted before each launch) like the layer-after-layer model schedule sees it.

#include "ninfer/ops/relative_sink_attention.h"

#include "core/cyclic_kv_cache.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "ninfer_bench_common.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kHeadDim     = 128;
constexpr std::int32_t kRelativeDim = 16;
constexpr std::int32_t kExtent      = 1024;
constexpr std::int32_t kWindow      = 1024;
// Ring slots as the model plans them: the window plus the most drafts one verification writes.
constexpr std::int32_t kRingSlots = kWindow + 7;
constexpr std::size_t kFlushBytes = std::size_t{256} << 20;

struct Options {
    bool paged               = false;
    std::int32_t query_heads = 12;
    std::int32_t kv_heads    = 4;
    std::vector<std::int32_t> tokens{1};
    std::vector<std::int32_t> contexts{4096};
    int warmup = 5;
    int repeat = 50;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\n"
                 "usage: ninfer_relative_sink_attention_bench --entry windowed|paged "
                 "[--heads 12/4|24/8] [--tokens 1,...] [--context L,...] "
                 "[--warmup N] [--repeat N]\n",
                 message);
    std::exit(2);
}

std::vector<std::int32_t> parse_list(const char* text, const char* flag) {
    std::vector<std::int32_t> result;
    std::string_view remaining(text);
    while (!remaining.empty()) {
        const std::size_t comma = remaining.find(',');
        const std::string item(remaining.substr(0, comma));
        errno            = 0;
        char* end        = nullptr;
        const long value = std::strtol(item.c_str(), &end, 10);
        if (errno != 0 || end == item.c_str() || *end != '\0' || value < 0 || value > (1 << 22))
            usage(flag);
        result.push_back(static_cast<std::int32_t>(value));
        if (comma == std::string_view::npos) break;
        remaining.remove_prefix(comma + 1);
    }
    if (result.empty()) usage(flag);
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* flag) -> const char* {
            if (++index == argc) usage(flag);
            return argv[index];
        };
        if (argument == "--entry") {
            const std::string_view value(next("--entry requires a value"));
            if (value != "windowed" && value != "paged") usage("--entry windowed|paged");
            options.paged = value == "paged";
        } else if (argument == "--heads") {
            const std::string_view value(next("--heads requires a value"));
            if (value == "12/4") {
                options.query_heads = 12;
                options.kv_heads    = 4;
            } else if (value == "24/8") {
                options.query_heads = 24;
                options.kv_heads    = 8;
            } else {
                usage("--heads 12/4|24/8");
            }
        } else if (argument == "--tokens") {
            options.tokens = parse_list(next("--tokens requires a value"), "--tokens");
        } else if (argument == "--context") {
            options.contexts = parse_list(next("--context requires a value"), "--context");
        } else if (argument == "--warmup") {
            options.warmup = parse_list(next("--warmup requires a value"), "--warmup").front();
        } else if (argument == "--repeat") {
            options.repeat = parse_list(next("--repeat requires a value"), "--repeat").front();
        } else {
            usage("unknown argument");
        }
    }
    for (const std::int32_t tokens : options.tokens)
        if (tokens < 1) usage("--tokens must be positive");
    if (options.repeat < 1) usage("--repeat must be positive");
    return options;
}

class Case {
public:
    Case(const Options& options, std::int32_t tokens, std::int32_t context)
        : paged_(options.paged), tokens_(tokens), context_(context),
          geometry_{kHeadDim, options.query_heads, options.kv_heads},
          q_(bench::make_bf16(static_cast<std::size_t>(kHeadDim) * geometry_.query_heads * tokens,
                              101U, -2.0F, 2.0F)),
          k_(bench::make_bf16(static_cast<std::size_t>(kHeadDim) * geometry_.kv_heads * tokens,
                              103U, -2.0F, 2.0F)),
          v_(bench::make_bf16(static_cast<std::size_t>(kHeadDim) * geometry_.kv_heads * tokens,
                              105U, -2.0F, 2.0F)),
          r_(bench::make_bf16(static_cast<std::size_t>(kRelativeDim) * geometry_.query_heads *
                                  tokens,
                              107U, -2.0F, 2.0F)),
          sink_(bench::make_bf16(static_cast<std::size_t>(geometry_.query_heads) * tokens, 109U)),
          q_scale_(bench::make_bf16(static_cast<std::size_t>(kHeadDim) * geometry_.query_heads,
                                    111U, 0.5F, 2.0F)),
          r_scale_(bench::make_bf16(static_cast<std::size_t>(kRelativeDim) * geometry_.query_heads,
                                    113U, 0.5F, 2.0F)),
          rel_proj_(bench::make_bf16(static_cast<std::size_t>(kExtent) * kRelativeDim, 115U, -1.0F,
                                     1.0F)),
          positions_(static_cast<std::size_t>(tokens) * sizeof(std::int32_t)),
          rows_(sizeof(std::int32_t)),
          output_(bench::make_zeros(static_cast<std::size_t>(kHeadDim) * geometry_.query_heads *
                                    tokens * 2)),
          workspace_bytes_(ops::relative_sink_attention_workspace_capacity_bytes(
              geometry_, kRelativeDim, 1, tokens, tokens)),
          workspace_(std::max<std::size_t>(workspace_bytes_, 1)) {
        std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
        std::iota(positions.begin(), positions.end(), context);
        CUDA_CHECK(
            cudaMemcpy(positions_.p, positions.data(), positions_.bytes, cudaMemcpyHostToDevice));
        const std::int32_t row = 0;
        CUDA_CHECK(cudaMemcpy(rows_.p, &row, sizeof(row), cudaMemcpyHostToDevice));
        const std::int32_t slots = paged_ ? 0 : kRingSlots;
        pages_                   = paged_ ? static_cast<std::int32_t>(
                              kv_pages_for_tokens(static_cast<std::uint32_t>(context + tokens)))
                                          : 0;
        const std::size_t rows_of_cache =
            paged_ ? static_cast<std::size_t>(pages_) * kPagedKVPageSize : slots;
        const std::size_t elements =
            static_cast<std::size_t>(kHeadDim) * rows_of_cache * geometry_.kv_heads;
        cache_k_ = bench::make_bf16(elements, 117U, -0.1F, 0.1F);
        // FP16 bit patterns of small BF16 values are finite FP16 values.
        cache_v_ = bench::make_bf16(elements, 119U, -0.1F, 0.1F);
        if (paged_) {
            std::vector<std::int32_t> table(static_cast<std::size_t>(pages_));
            std::iota(table.begin(), table.end(), 0);
            table_ = DeviceBuffer(table.size() * sizeof(std::int32_t));
            CUDA_CHECK(cudaMemcpy(table_.p, table.data(), table_.bytes, cudaMemcpyHostToDevice));
        }
        params_.relative_dim    = kRelativeDim;
        params_.relative_extent = kExtent;
        params_.qk_clip         = 22.627416997969522F;
        params_.relative_clip   = 8.0F;
        params_.clip_epsilon    = 1e-6F;
        params_.value_epsilon   = 1e-6F;
        params_.scale           = 0.08838834764831844055F;
        params_.log_scale_alpha = paged_ ? 0.1F : 0.0F;
        params_.log_scale_floor = 8192.0F;
    }

    void launch(cudaStream_t stream) {
        const std::int32_t D = kHeadDim, Hq = geometry_.query_heads, Hk = geometry_.kv_heads;
        Tensor q(q_.p, DType::BF16, {D, Hq, tokens_, 1}), k(k_.p, DType::BF16, {D, Hk, tokens_, 1}),
            v(v_.p, DType::BF16, {D, Hk, tokens_, 1}),
            r(r_.p, DType::BF16, {kRelativeDim, Hq, tokens_, 1}),
            sink(sink_.p, DType::BF16, {Hq, tokens_, 1}),
            positions(positions_.p, DType::I32, {tokens_, 1}), rows(rows_.p, DType::I32, {1}),
            q_scale(q_scale_.p, DType::BF16, {D, Hq}),
            r_scale(r_scale_.p, DType::BF16, {kRelativeDim, Hq}),
            rel_proj(rel_proj_.p, DType::BF16, {kExtent, kRelativeDim}),
            out(output_.p, DType::BF16, {D, Hq, tokens_, 1});
        if (paged_) {
            PagedKVBatchLayerView cache;
            cache.k_pages      = Tensor(cache_k_.p, DType::BF16, {D, kPagedKVPageSize, Hk, pages_});
            cache.v_pages      = Tensor(cache_v_.p, DType::FP16, {D, kPagedKVPageSize, Hk, pages_});
            cache.block_tables = Tensor(table_.p, DType::I32, {pages_, 1});
            cache.head_dim     = D;
            cache.num_kv_heads = Hk;
            cache.storage      = KvCacheStorage::BFloat16;
            ops::relative_sink_attention_paged(q, k, v, r, sink, positions, rows, q_scale, r_scale,
                                               rel_proj, geometry_, params_, cache, nullptr,
                                               workspace_, out, stream);
        } else {
            const CyclicKVCacheLayerView cache{
                .k               = Tensor(cache_k_.p, DType::BF16, {D, kRingSlots, Hk, 1}),
                .v               = Tensor(cache_v_.p, DType::FP16, {D, kRingSlots, Hk, 1}),
                .capacity        = static_cast<std::uint32_t>(kRingSlots),
                .padded_capacity = static_cast<std::uint32_t>(kRingSlots),
                .num_kv_heads    = Hk,
                .head_dim        = D,
                .lane_capacity   = 1};
            ops::relative_sink_attention_windowed(q, k, v, r, sink, positions, rows, q_scale,
                                                  r_scale, rel_proj, geometry_, params_, kWindow,
                                                  cache, nullptr, workspace_, out, stream);
        }
    }

    // K and V bytes of every key some query of the call sees: the bandwidth floor.
    [[nodiscard]] double useful_bytes() const {
        const std::int64_t first = paged_ ? 0 : std::max(0, context_ - kWindow + 1);
        const std::int64_t keys  = static_cast<std::int64_t>(context_) + tokens_ - first;
        return static_cast<double>(keys) * geometry_.kv_heads * kHeadDim * 4.0;
    }

    [[nodiscard]] std::size_t workspace_bytes() const noexcept { return workspace_bytes_; }

private:
    bool paged_;
    std::int32_t tokens_;
    std::int32_t context_;
    ops::AttentionHeadGeometry geometry_;
    DeviceBuffer q_, k_, v_, r_, sink_, q_scale_, r_scale_, rel_proj_, positions_, rows_, output_;
    DeviceBuffer cache_k_, cache_v_, table_;
    std::int32_t pages_ = 0;
    std::size_t workspace_bytes_;
    WorkspaceArena workspace_;
    ops::RelativeSinkAttentionParams params_;
};

} // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }
        const Options options = parse_options(argc, argv);
        DeviceContext device;
        bench::L2FlushBuffer flush(kFlushBytes);
        for (const std::int32_t context : options.contexts) {
            for (const std::int32_t tokens : options.tokens) {
                Case data(options, tokens, context);
                data.launch(device.stream);
                CUDA_CHECK(cudaStreamSynchronize(device.stream));
                bench::TimedGraph graph;
                graph.capture(device.stream, [&](cudaStream_t stream) { data.launch(stream); });
                const bench::ColdTiming timing = bench::measure_cold_graph(
                    graph, flush, device.stream, options.warmup, options.repeat);
                const double gbps = data.useful_bytes() / (timing.median_us * 1.0e-6) / 1.0e9;
                std::printf("entry=%s heads=%d/%d T=%d L=%d graph_nodes=%zu workspace=%zu "
                            "median=%.3f us min=%.3f us p95=%.3f us useful=%.1f GB/s\n",
                            options.paged ? "paged" : "windowed", options.query_heads,
                            options.kv_heads, tokens, context, graph.nodes(),
                            data.workspace_bytes(), timing.median_us, timing.min_us, timing.p95_us,
                            gbps);
                std::fflush(stdout);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "relative_sink_attention_bench: %s\n", error.what());
        return 1;
    }
}
