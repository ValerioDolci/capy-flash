// Public-Op benchmark for the full D256 KV cache append contract.
// Cache encoding, launch geometry, and route selection remain private to the public wrappers.

#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/kv_cache_append.h"

#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "ninfer_bench_common.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kFullHeadDim = 256;
constexpr std::int32_t kMaxTokens   = 4096;
constexpr std::size_t kFlushBytes   = std::size_t{256} << 20;
constexpr double kRtx5090DramGBs    = 1792.0;

enum class FullGeometryChoice : std::uint8_t { Kv4, Kv2, All };
enum class KvChoice : std::uint8_t { Bf16, Int8, Fp8, Nvfp4, K8V4, All };
enum class Execution : std::uint8_t { Eager, Graph, Both };
enum class CacheMode : std::uint8_t { Cold, Warm, Both };
enum class CacheState : std::uint8_t { Cold, Warm };

struct FullGeometry {
    const char* name;
    std::int32_t kv_heads;
};

constexpr FullGeometry kFullKv4{"d256-kv4", 4};
constexpr FullGeometry kFullKv2{"d256-kv2", 2};

struct Options {
    FullGeometryChoice full_geometry = FullGeometryChoice::All;
    KvChoice kv                      = KvChoice::All;
    Execution execution              = Execution::Graph;
    CacheMode cache                  = CacheMode::Cold;
    std::vector<std::int32_t> tokens{1, 2, 4, 8, 16, 1024};
    std::int32_t context = 128;
    int warmup           = 5;
    int repeat           = 50;
    int graph_calls      = 1;
    bool profile         = false;
    std::string csv_out;
};

struct Result {
    const char* geometry;
    KvCacheStorage storage;
    Execution execution;
    CacheState cache;
    std::int32_t tokens;
    std::size_t graph_nodes;
    int graph_calls;
    double logical_cache_bytes;
    double key_vector_bytes;
    double value_vector_bytes;
    double physical_cache_bytes;
    double useful_bytes;
    bench::ColdTiming timing;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\n"
                 "usage: ninfer_kv_cache_append_bench "
                 "[--full-geometry d256-kv4|d256-kv2|all] "
                 "[--kv-dtype bf16|int8|fp8|nvfp4|k8v4|all] [--tokens T,...] "
                 "[--context L] [--execution eager|graph|both] [--cache cold|warm|both] "
                 "[--warmup N] [--repeat N] [--graph-calls N] [--profile] "
                 "[--csv-out PATH]\n",
                 message);
    std::exit(2);
}

std::int32_t parse_i32(std::string_view text, std::int32_t minimum, std::int32_t maximum,
                       const char* flag) {
    const std::string value(text);
    errno       = 0;
    char* end   = nullptr;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || parsed < minimum ||
        parsed > maximum) {
        usage(flag);
    }
    return static_cast<std::int32_t>(parsed);
}

std::vector<std::int32_t> parse_list(const char* text, std::int32_t minimum, std::int32_t maximum,
                                     const char* flag) {
    std::vector<std::int32_t> result;
    std::string_view remaining(text);
    while (!remaining.empty()) {
        const std::size_t comma     = remaining.find(',');
        const std::string_view item = remaining.substr(0, comma);
        if (item.empty()) { usage(flag); }
        result.push_back(parse_i32(item, minimum, maximum, flag));
        if (comma == std::string_view::npos) { break; }
        remaining.remove_prefix(comma + 1);
    }
    if (result.empty()) { usage(flag); }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* flag) -> const char* {
            if (++index == argc) { usage(flag); }
            return argv[index];
        };
        if (argument == "--full-geometry") {
            const std::string_view value(next("--full-geometry requires a value"));
            if (value == "d256-kv4")
                options.full_geometry = FullGeometryChoice::Kv4;
            else if (value == "d256-kv2")
                options.full_geometry = FullGeometryChoice::Kv2;
            else if (value == "all")
                options.full_geometry = FullGeometryChoice::All;
            else
                usage("--full-geometry expects d256-kv4, d256-kv2, or all");
        } else if (argument == "--kv-dtype") {
            const std::string_view value(next("--kv-dtype requires a value"));
            if (value == "bf16")
                options.kv = KvChoice::Bf16;
            else if (value == "int8")
                options.kv = KvChoice::Int8;
            else if (value == "fp8")
                options.kv = KvChoice::Fp8;
            else if (value == "nvfp4")
                options.kv = KvChoice::Nvfp4;
            else if (value == "k8v4")
                options.kv = KvChoice::K8V4;
            else if (value == "all")
                options.kv = KvChoice::All;
            else
                usage("--kv-dtype expects bf16, int8, fp8, nvfp4, k8v4, or all");
        } else if (argument == "--tokens") {
            options.tokens =
                parse_list(next("--tokens requires a value"), 1, kMaxTokens, "--tokens");
        } else if (argument == "--context") {
            options.context = parse_i32(next("--context requires a value"), 0, 262144, "--context");
        } else if (argument == "--execution") {
            const std::string_view value(next("--execution requires a value"));
            if (value == "eager")
                options.execution = Execution::Eager;
            else if (value == "graph")
                options.execution = Execution::Graph;
            else if (value == "both")
                options.execution = Execution::Both;
            else
                usage("--execution expects eager, graph, or both");
        } else if (argument == "--cache") {
            const std::string_view value(next("--cache requires a value"));
            if (value == "cold")
                options.cache = CacheMode::Cold;
            else if (value == "warm")
                options.cache = CacheMode::Warm;
            else if (value == "both")
                options.cache = CacheMode::Both;
            else
                usage("--cache expects cold, warm, or both");
        } else if (argument == "--warmup") {
            options.warmup = parse_i32(next("--warmup requires a value"), 0, 10000, "--warmup");
        } else if (argument == "--repeat") {
            options.repeat = parse_i32(next("--repeat requires a value"), 1, 10000, "--repeat");
        } else if (argument == "--graph-calls") {
            options.graph_calls =
                parse_i32(next("--graph-calls requires a value"), 1, 64, "--graph-calls");
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--csv-out") {
            options.csv_out = next("--csv-out requires a path");
        } else if (argument == "--help" || argument == "-h") {
            usage("help");
        } else {
            usage("unknown argument");
        }
    }
    if (options.context > std::numeric_limits<std::int32_t>::max() -
                              *std::max_element(options.tokens.begin(), options.tokens.end())) {
        usage("context + tokens exceeds int32");
    }
    if (options.profile && (options.tokens.size() != 1 || options.execution == Execution::Both ||
                            options.cache == CacheMode::Both)) {
        usage("--profile requires one T, one execution, and one cache state");
    }
    if (options.profile &&
        (options.full_geometry == FullGeometryChoice::All || options.kv == KvChoice::All)) {
        usage("--profile requires one full geometry and one KV dtype");
    }
    if (options.graph_calls > 1 && (options.execution != Execution::Graph || options.profile))
        usage("graph bundles require --execution graph and no --profile");
    return options;
}

std::int32_t align_context(std::int32_t context) {
    return ((std::max(context, 1) + 127) / 128) * 128;
}

std::size_t full_data_bytes(const FullGeometry& geometry, const PagedKVVectorLayout& layout,
                            std::int32_t padded) {
    return static_cast<std::size_t>(layout.data_leading_extent) * geometry.kv_heads * padded *
           dtype_size(layout.data_dtype);
}

std::size_t full_scale_bytes(const FullGeometry& geometry, const PagedKVVectorLayout& layout,
                             std::int32_t padded) {
    return static_cast<std::size_t>(layout.scale_leading_extent) * geometry.kv_heads * padded *
           dtype_size(layout.scale_dtype);
}

PagedKVLayerView make_full_view(DeviceBuffer& k, DeviceBuffer& v, DeviceBuffer& k_scale,
                                DeviceBuffer& v_scale, DeviceBuffer& block_table,
                                const FullGeometry& geometry, KvCacheStorage storage,
                                std::int32_t padded) {
    const PagedKVStorageLayout layout = paged_kv_storage_layout(storage, kFullHeadDim);
    const std::int32_t pages          = padded / kPagedKVPageSize;
    return {
        .k_pages =
            Tensor(k.p, layout.key.data_dtype,
                   {layout.key.data_leading_extent, kPagedKVPageSize, geometry.kv_heads, pages}),
        .v_pages =
            Tensor(v.p, layout.value.data_dtype,
                   {layout.value.data_leading_extent, kPagedKVPageSize, geometry.kv_heads, pages}),
        .k_scale_pages = layout.key.has_scale()
                             ? Tensor(k_scale.p, layout.key.scale_dtype,
                                      {layout.key.scale_leading_extent, kPagedKVPageSize,
                                       geometry.kv_heads, pages})
                             : Tensor(),
        .v_scale_pages = layout.value.has_scale()
                             ? Tensor(v_scale.p, layout.value.scale_dtype,
                                      {layout.value.scale_leading_extent, kPagedKVPageSize,
                                       geometry.kv_heads, pages})
                             : Tensor(),
        .block_table   = Tensor(block_table.p, DType::I32, {pages}),
        .head_dim      = kFullHeadDim,
        .num_kv_heads  = geometry.kv_heads,
        .storage       = storage,
    };
}

class FullCase {
public:
    FullCase(FullGeometry geometry, KvCacheStorage storage, std::int32_t tokens,
             std::int32_t context)
        : geometry_(geometry), storage_(storage),
          storage_layout_(paged_kv_storage_layout(storage, kFullHeadDim)), tokens_(tokens),
          capacity_(context + tokens), padded_(align_context(capacity_)),
          k_(bench::make_bf16(static_cast<std::size_t>(kFullHeadDim) * geometry.kv_heads * tokens,
                              101U)),
          v_(bench::make_bf16(static_cast<std::size_t>(kFullHeadDim) * geometry.kv_heads * tokens,
                              103U)),
          positions_(static_cast<std::size_t>(tokens) * sizeof(std::int32_t)),
          cache_k_(bench::make_zeros(full_data_bytes(geometry, storage_layout_.key, padded_))),
          cache_v_(bench::make_zeros(full_data_bytes(geometry, storage_layout_.value, padded_))),
          cache_k_scale_(
              bench::make_zeros(storage_layout_.key.has_scale()
                                    ? full_scale_bytes(geometry, storage_layout_.key, padded_)
                                    : std::size_t{1})),
          cache_v_scale_(
              bench::make_zeros(storage_layout_.value.has_scale()
                                    ? full_scale_bytes(geometry, storage_layout_.value, padded_)
                                    : std::size_t{1})),
          block_table_(static_cast<std::size_t>(padded_ / kPagedKVPageSize) * sizeof(std::int32_t)),
          k_tensor_(k_.p, DType::BF16, {kFullHeadDim, geometry.kv_heads, tokens}),
          v_tensor_(v_.p, DType::BF16, {kFullHeadDim, geometry.kv_heads, tokens}),
          positions_tensor_(positions_.p, DType::I32, {tokens}),
          cache_view_(make_full_view(cache_k_, cache_v_, cache_k_scale_, cache_v_scale_,
                                     block_table_, geometry, storage, padded_)) {
        std::vector<std::int32_t> host_positions(static_cast<std::size_t>(tokens));
        for (std::int32_t token = 0; token < tokens; ++token) {
            host_positions[static_cast<std::size_t>(token)] = context + token;
        }
        std::vector<std::int32_t> host_table(static_cast<std::size_t>(padded_ / kPagedKVPageSize));
        for (std::int32_t page = 0; page < static_cast<std::int32_t>(host_table.size()); ++page) {
            host_table[static_cast<std::size_t>(page)] = page;
        }
        CUDA_CHECK(cudaMemcpy(positions_.p, host_positions.data(), positions_.bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(block_table_.p, host_table.data(), block_table_.bytes,
                              cudaMemcpyHostToDevice));
    }

    void launch(cudaStream_t stream) {
        ops::kv_cache_append(k_tensor_, v_tensor_, positions_tensor_, cache_view_, stream);
    }

private:
    FullGeometry geometry_;
    KvCacheStorage storage_;
    PagedKVStorageLayout storage_layout_;
    std::int32_t tokens_;
    std::int32_t capacity_;
    std::int32_t padded_;
    DeviceBuffer k_;
    DeviceBuffer v_;
    DeviceBuffer positions_;
    DeviceBuffer cache_k_;
    DeviceBuffer cache_v_;
    DeviceBuffer cache_k_scale_;
    DeviceBuffer cache_v_scale_;
    DeviceBuffer block_table_;
    Tensor k_tensor_;
    Tensor v_tensor_;
    Tensor positions_tensor_;
    PagedKVLayerView cache_view_;
};

const char* storage_name(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return "bf16";
    case KvCacheStorage::Int8Group64:
        return "int8";
    case KvCacheStorage::Fp8E4M3Row256:
        return "fp8";
    case KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    return "unknown";
}

const char* execution_name(Execution execution) {
    return execution == Execution::Eager ? "eager" : "graph";
}

const char* cache_name(CacheState cache) { return cache == CacheState::Cold ? "cold" : "warm"; }

double full_vector_bytes(const PagedKVVectorLayout& layout) {
    return static_cast<double>(layout.physical_bytes());
}

double full_logical_cache_bytes(const FullGeometry& geometry, std::int32_t tokens) {
    const auto layout = paged_kv_storage_layout(KvCacheStorage::BFloat16, kFullHeadDim);
    return static_cast<double>(layout.logical_bytes_per_token_head()) * geometry.kv_heads * tokens;
}

double full_physical_cache_bytes(const FullGeometry& geometry, KvCacheStorage storage,
                                 std::int32_t tokens) {
    const auto layout = paged_kv_storage_layout(storage, kFullHeadDim);
    return geometry.kv_heads * tokens *
           (full_vector_bytes(layout.key) + full_vector_bytes(layout.value));
}

double full_useful_bytes(const FullGeometry& geometry, KvCacheStorage storage,
                         std::int32_t tokens) {
    const auto layout   = paged_kv_storage_layout(storage, kFullHeadDim);
    const double input  = 2.0 * kFullHeadDim * geometry.kv_heads * tokens * 2.0;
    const double output = geometry.kv_heads * tokens *
                          (full_vector_bytes(layout.key) + full_vector_bytes(layout.value));
    return input + output;
}

template <class Case>
bench::ColdTiming measure(Case& data, Execution execution, CacheState cache,
                          bench::TimedGraph* graph, bench::L2FlushBuffer& flush,
                          cudaStream_t stream, int warmup, int repeat) {
    if (execution == Execution::Eager) {
        const auto launch = [&](cudaStream_t launch_stream) { data.launch(launch_stream); };
        return cache == CacheState::Cold
                   ? bench::measure_cold_launch(launch, flush, stream, warmup, repeat)
                   : bench::measure_launch(launch, stream, warmup, repeat);
    }
    return cache == CacheState::Cold
               ? bench::measure_cold_graph(*graph, flush, stream, warmup, repeat)
               : bench::measure_graph(*graph, stream, warmup, repeat);
}

void report(const Result& result) {
    const double seconds = result.timing.median_us * 1.0e-6;
    const double gbps    = seconds > 0 ? result.useful_bytes / seconds / 1.0e9 : 0.0;
    std::printf("geometry=%-9s kv=%-6s execution=%-5s cache=%-4s T=%4d graph_nodes=%zu "
                "graph_calls=%d median=%8.3f us min=%8.3f us p95=%8.3f us "
                "logical=%.0f physical=%.0f (K=%.0f V=%.0f) useful=%8.1f GB/s "
                "(%5.1f%% of %.0f)\n",
                result.geometry, storage_name(result.storage), execution_name(result.execution),
                cache_name(result.cache), result.tokens, result.graph_nodes, result.graph_calls,
                result.timing.median_us, result.timing.min_us, result.timing.p95_us,
                result.logical_cache_bytes, result.physical_cache_bytes, result.key_vector_bytes,
                result.value_vector_bytes, gbps, gbps / kRtx5090DramGBs * 100.0, kRtx5090DramGBs);
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream output(path);
    if (!output) { throw std::runtime_error("failed to open CSV output"); }
    output << "geometry,kv_dtype,execution,cache,T,graph_nodes,graph_calls,"
              "logical_cache_bytes,key_vector_bytes,value_vector_bytes,physical_cache_bytes,"
              "useful_bytes,"
              "median_us,min_us,p95_us\n";
    for (const Result& result : results) {
        output << result.geometry << ',' << storage_name(result.storage) << ','
               << execution_name(result.execution) << ',' << cache_name(result.cache) << ','
               << result.tokens << ',' << result.graph_nodes << ',' << result.graph_calls << ','
               << result.logical_cache_bytes << ',' << result.key_vector_bytes << ','
               << result.value_vector_bytes << ',' << result.physical_cache_bytes << ','
               << result.useful_bytes << ',' << result.timing.median_us << ','
               << result.timing.min_us << ',' << result.timing.p95_us << '\n';
    }
}

void profile_case(FullCase& data, const char* label, const Options& options,
                  bench::L2FlushBuffer& flush, cudaStream_t stream) {
    const Execution execution = options.execution;
    const CacheState cache = options.cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm;
    bench::TimedGraph graph;
    if (execution == Execution::Graph) {
        data.launch(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        graph.capture(stream, [&](cudaStream_t launch_stream) { data.launch(launch_stream); });
        for (int index = 0; index < options.warmup; ++index) { graph.launch(stream); }
    } else {
        for (int index = 0; index < options.warmup; ++index) { data.launch(stream); }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (cache == CacheState::Cold) {
        bench::flush_l2(flush, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::printf("PROFILE %s dispatch=public execution=%s cache=%s\n", label,
                execution_name(execution), cache_name(cache));
    std::fflush(stdout);
    CUDA_CHECK(cudaProfilerStart());
    if (execution == Execution::Graph)
        graph.launch(stream);
    else
        data.launch(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaProfilerStop());
}

std::vector<FullGeometry> selected_geometries(FullGeometryChoice choice) {
    if (choice == FullGeometryChoice::Kv4) return {kFullKv4};
    if (choice == FullGeometryChoice::Kv2) return {kFullKv2};
    return {kFullKv4, kFullKv2};
}

std::vector<KvCacheStorage> selected_storages(KvChoice choice) {
    if (choice == KvChoice::Bf16) return {KvCacheStorage::BFloat16};
    if (choice == KvChoice::Int8) return {KvCacheStorage::Int8Group64};
    if (choice == KvChoice::Fp8) return {KvCacheStorage::Fp8E4M3Row256};
    if (choice == KvChoice::Nvfp4) return {KvCacheStorage::Nvfp4Group16};
    if (choice == KvChoice::K8V4) return {KvCacheStorage::Fp8KeyNvfp4Value};
    return {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
            KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value};
}

void collect_case(FullCase& data, const char* geometry, KvCacheStorage storage, std::int32_t tokens,
                  double logical_cache_bytes, double key_vector_bytes, double value_vector_bytes,
                  double physical_cache_bytes, double useful_bytes, const Options& options,
                  bench::L2FlushBuffer& flush, cudaStream_t stream, std::vector<Result>& results) {
    bench::TimedGraph graph;
    if (options.execution != Execution::Eager) {
        data.launch(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        graph.capture(stream, [&](cudaStream_t launch_stream) {
            for (int call = 0; call < options.graph_calls; ++call) data.launch(launch_stream);
        });
    }
    for (const Execution execution : {Execution::Eager, Execution::Graph}) {
        if ((options.execution == Execution::Eager && execution != Execution::Eager) ||
            (options.execution == Execution::Graph && execution != Execution::Graph)) {
            continue;
        }
        for (const CacheState cache : {CacheState::Cold, CacheState::Warm}) {
            if ((options.cache == CacheMode::Cold && cache != CacheState::Cold) ||
                (options.cache == CacheMode::Warm && cache != CacheState::Warm)) {
                continue;
            }
            Result result{geometry,
                          storage,
                          execution,
                          cache,
                          tokens,
                          execution == Execution::Graph ? graph.nodes() : 0,
                          options.graph_calls,
                          logical_cache_bytes,
                          key_vector_bytes,
                          value_vector_bytes,
                          physical_cache_bytes,
                          useful_bytes,
                          measure(data, execution, cache, &graph, flush, stream, options.warmup,
                                  options.repeat)};
            result.timing.median_us /= options.graph_calls;
            result.timing.min_us /= options.graph_calls;
            result.timing.p95_us /= options.graph_calls;
            report(result);
            results.push_back(result);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }
        const Options options = parse_options(argc, argv);
        cudaStream_t stream   = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        bench::L2FlushBuffer flush(kFlushBytes);
        const std::vector<FullGeometry> geometries = selected_geometries(options.full_geometry);
        const std::vector<KvCacheStorage> storages = selected_storages(options.kv);

        if (options.profile) {
            const FullGeometry geometry  = geometries.front();
            const KvCacheStorage storage = storages.front();
            FullCase data(geometry, storage, options.tokens.front(), options.context);
            const std::string label =
                std::string("geometry=") + geometry.name + " kv=" + storage_name(storage);
            profile_case(data, label.c_str(), options, flush, stream);
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        std::vector<Result> results;
        for (const FullGeometry& geometry : geometries) {
            for (const KvCacheStorage storage : storages) {
                for (const std::int32_t tokens : options.tokens) {
                    const auto storage_layout     = paged_kv_storage_layout(storage, kFullHeadDim);
                    const double key_vector_bytes = full_vector_bytes(storage_layout.key);
                    const double value_vector_bytes = full_vector_bytes(storage_layout.value);
                    const double physical_cache_bytes =
                        full_physical_cache_bytes(geometry, storage, tokens);
                    FullCase data(geometry, storage, tokens, options.context);
                    collect_case(data, geometry.name, storage, tokens,
                                 full_logical_cache_bytes(geometry, tokens), key_vector_bytes,
                                 value_vector_bytes, physical_cache_bytes,
                                 full_useful_bytes(geometry, storage, tokens), options, flush,
                                 stream, results);
                }
            }
        }
        write_csv(options, results);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_kv_cache_append_bench: %s\n", error.what());
        return 1;
    }
}
