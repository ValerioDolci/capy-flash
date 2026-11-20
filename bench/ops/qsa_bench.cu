// Qwen Sparse Attention decode/verify microbenchmark at the Qwen3.8-Flash-Next geometry.
//
// One QSA layer of one sequence holding `depth` cached tokens appends a window of W columns
// (positions depth..depth+W-1), selects and attends. Each Op and the whole chain run as a
// captured CUDA Graph; timings are medians of per-replay CUDA event intervals, warm (operands
// resident in L2 from the previous replay) and cold (a 256 MiB read evicts L2 before each replay;
// the eviction is outside the timed interval).
//
// `ext-select` is a bench-local reference: ninfer-ext's selection route (giveen/ninfer-ext
// src/ops/sparse_attention/kernels.cu @9c933f01, Apache-2.0: qsa_score_kernel recomputing pooled
// keys from the raw records at every call, one-CTA-per-column radix qsa_select_kernel) run on the
// same index records, so the two selection designs are compared on identical data.
#include "ninfer/ops/sparse_attention.h"
#include "ninfer_bench_common.h"

#include "core/arena.h"
#include "core/paged_kv_storage.h"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <functional>
#include <iterator>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr int kHeadDim  = 256;
constexpr int kIndexDim = 128;
constexpr int kPage     = kPagedKVPageSize;

// ---------------------------------------------------------------------------------------------
// ninfer-ext selection reference (verbatim arithmetic; Host arm removed).
namespace ext {

constexpr int kRecord       = ops::kQsaIndexRecordWords;
constexpr int kMaxIndexHead = 8;

struct KernelGeometry {
    std::int32_t index_heads, index_dim, budget, ratio, rotary_dim;
    float rope_theta, rms_norm_eps;
    std::uint8_t pair_axes[64];

    __device__ std::int32_t block_topk() const { return budget / ratio; }
};

struct SelectArgs {
    const __nv_bfloat16* index_query;
    const std::int32_t* query_positions;
    std::int32_t query_axes;
    const std::int32_t* cache;
    const std::int32_t* table_rows;
    const std::int32_t* tables;
    std::int32_t table_stride;
    const __nv_bfloat16* records;
    const __nv_bfloat16* query_norm;
    const __nv_bfloat16* key_norm;
    float* scores;
    std::int32_t max_blocks;
    std::int32_t* selected;
    std::int32_t* counts;
    std::int32_t width;
    std::int32_t lanes;
    std::int32_t max_selected;
    KernelGeometry geometry;
};

__device__ __forceinline__ float rope_angle(const KernelGeometry& g, int pair,
                                            const std::int32_t* p) {
    const float exponent = -2.0F * static_cast<float>(pair) / static_cast<float>(g.rotary_dim);
    return static_cast<float>(p[g.pair_axes[pair]]) * powf(g.rope_theta, exponent);
}

__device__ __forceinline__ std::uint32_t order_key(float x) {
    const std::uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ std::int32_t read_position(const __nv_bfloat16* record, int axis) {
    const auto* words = reinterpret_cast<const std::uint16_t*>(record + kIndexDim);
    return static_cast<std::int32_t>(static_cast<std::uint32_t>(words[2 * axis]) |
                                     (static_cast<std::uint32_t>(words[2 * axis + 1]) << 16));
}

__global__ void __launch_bounds__(256) score_kernel(SelectArgs a, std::int32_t split_blocks) {
    __shared__ float query[kMaxIndexHead * kIndexDim];
    const KernelGeometry& g    = a.geometry;
    const int tid              = static_cast<int>(threadIdx.x);
    const int warp             = tid >> 5;
    const int lane_id          = tid & 31;
    const int column           = static_cast<int>(blockIdx.x);
    const int lane             = column / a.width;
    const std::int32_t visible = a.cache[column] + 1;
    const std::int32_t blocks  = visible / g.ratio;
    const std::int32_t first   = static_cast<int>(blockIdx.y) * split_blocks;
    const std::int32_t last    = min(blocks, first + split_blocks);
    if (blocks <= g.block_topk() || first >= last) { return; }
    const std::int64_t tokens = static_cast<std::int64_t>(a.width) * a.lanes;
    std::int32_t qpos[3];
    for (int axis = 0; axis < 3; ++axis) {
        qpos[axis] = a.query_positions[(a.query_axes == 3 ? axis : 0) * tokens + column];
    }
    const int di = g.index_dim;
    for (int h = warp; h < g.index_heads; h += blockDim.x / 32) {
        const __nv_bfloat16* raw =
            a.index_query + (static_cast<std::int64_t>(column) * g.index_heads + h) * di;
        float sum = 0.0F;
        for (int d = lane_id; d < di; d += 32) {
            const float x = __bfloat162float(raw[d]);
            sum += x * x;
        }
        const float inv = rsqrtf(ops::warp_sum(sum) / static_cast<float>(di) + g.rms_norm_eps);
        for (int d = lane_id; d < di; d += 32) {
            query[h * di + d] =
                __bfloat162float(raw[d]) * inv * (1.0F + __bfloat162float(a.query_norm[d]));
        }
        __syncwarp();
        if (lane_id < g.rotary_dim / 2) {
            const int i   = lane_id;
            const int j   = i + g.rotary_dim / 2;
            const float t = rope_angle(g, i, qpos);
            const float c = cosf(t), s = sinf(t);
            const float x0 = query[h * di + i], x1 = query[h * di + j];
            query[h * di + i] = x0 * c - x1 * s;
            query[h * di + j] = x1 * c + x0 * s;
        }
    }
    __syncthreads();
    const std::int32_t* table =
        a.tables + static_cast<std::int64_t>(a.table_rows[lane]) * a.table_stride;
    float* scores     = a.scores + static_cast<std::int64_t>(blockIdx.x) * a.max_blocks;
    const float scale = rsqrtf(static_cast<float>(di));
    const int per     = di / 32;
    for (int b = first + warp; b < last; b += blockDim.x / 32) {
        float key[8];
        for (int r = 0; r < per; ++r) { key[r] = 0.0F; }
        std::int32_t kpos[3] = {0, 0, 0};
        for (int t = 0; t < g.ratio; ++t) {
            const std::int32_t token = b * g.ratio + t;
            const __nv_bfloat16* record =
                a.records +
                (static_cast<std::int64_t>(ops::paged_kv_physical_page(table, token)) * kPage +
                 (token & ops::kPagedKVPageMask)) *
                    kRecord;
            for (int r = 0; r < per; ++r) { key[r] += __bfloat162float(record[lane_id * per + r]); }
            if (t == 0) {
                for (int axis = 0; axis < 3; ++axis) { kpos[axis] = read_position(record, axis); }
            }
        }
        float sum = 0.0F;
        for (int r = 0; r < per; ++r) {
            key[r] = __bfloat162float(__float2bfloat16_rn(key[r] / static_cast<float>(g.ratio)));
            sum += key[r] * key[r];
        }
        const float inv = rsqrtf(ops::warp_sum(sum) / static_cast<float>(di) + g.rms_norm_eps);
        for (int r = 0; r < per; ++r) {
            key[r] *= inv * (1.0F + __bfloat162float(a.key_norm[lane_id * per + r]));
        }
        const int half  = g.rotary_dim / 2;
        const int shift = half / per;
        for (int r = 0; r < per; ++r) {
            const int d       = lane_id * per + r;
            const float other = __shfl_xor_sync(0xffffffffu, key[r], shift);
            if (d < g.rotary_dim) {
                const int pair = d < half ? d : d - half;
                const float t  = rope_angle(g, pair, kpos);
                const float c = cosf(t), s = sinf(t);
                key[r] = d < half ? key[r] * c - other * s : key[r] * c + other * s;
            }
        }
        float score = 0.0F;
        for (int h = 0; h < g.index_heads; ++h) {
            float dot = 0.0F;
            for (int r = 0; r < per; ++r) { dot += query[h * di + lane_id * per + r] * key[r]; }
            score += fmaxf(ops::warp_sum(dot), 0.0F);
        }
        if (lane_id == 0) { scores[b] = score * scale; }
    }
}

__global__ void __launch_bounds__(256) select_kernel(SelectArgs a) {
    __shared__ std::uint32_t histogram[256];
    __shared__ std::int32_t scan[256];
    __shared__ std::uint32_t prefix_s;
    __shared__ std::int32_t remaining_s, emitted_s, ties_s;
    const KernelGeometry& g    = a.geometry;
    const int tid              = static_cast<int>(threadIdx.x);
    const int column           = static_cast<int>(blockIdx.x);
    const std::int32_t visible = a.cache[column] + 1;
    const std::int32_t ratio   = g.ratio;
    const std::int32_t blocks  = visible / ratio;
    const std::int32_t keep    = min(g.block_topk(), blocks);
    const std::int32_t tail    = visible - blocks * ratio;
    std::int32_t* selected     = a.selected + static_cast<std::int64_t>(column) * a.max_selected;
    if (blocks <= g.block_topk()) {
        for (int i = tid; i < visible; i += blockDim.x) { selected[i] = i; }
        if (tid == 0) { a.counts[column] = visible; }
        return;
    }
    const float* scores = a.scores + static_cast<std::int64_t>(blockIdx.x) * a.max_blocks;
    if (tid == 0) {
        prefix_s    = 0;
        remaining_s = keep;
    }
    __syncthreads();
    std::uint32_t mask = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        histogram[tid] = 0;
        __syncthreads();
        for (int b = tid; b < blocks; b += blockDim.x) {
            const std::uint32_t key = order_key(scores[b]);
            if ((key & mask) == prefix_s) { atomicAdd(&histogram[(key >> shift) & 0xFFu], 1u); }
        }
        __syncthreads();
        if (tid == 0) {
            std::int32_t need = remaining_s;
            for (int bin = 255; bin >= 0; --bin) {
                const std::int32_t count = static_cast<std::int32_t>(histogram[bin]);
                if (count >= need) {
                    prefix_s |= static_cast<std::uint32_t>(bin) << shift;
                    remaining_s = need;
                    break;
                }
                need -= count;
            }
        }
        mask |= 0xFFu << shift;
        __syncthreads();
    }
    const std::uint32_t threshold = prefix_s;
    if (tid == 0) {
        emitted_s = 0;
        ties_s    = remaining_s;
    }
    __syncthreads();
    for (int base = 0; base < blocks; base += blockDim.x) {
        const int b             = base + tid;
        const std::uint32_t key = b < blocks ? order_key(scores[b]) : 0u;
        const int greater       = b < blocks && key > threshold ? 1 : 0;
        const int tie           = b < blocks && key == threshold ? 1 : 0;
        scan[tid]               = greater | (tie << 16);
        __syncthreads();
        for (int offset = 1; offset < static_cast<int>(blockDim.x); offset <<= 1) {
            const std::int32_t add = tid >= offset ? scan[tid - offset] : 0;
            __syncthreads();
            scan[tid] += add;
            __syncthreads();
        }
        const std::int32_t inclusive = scan[tid];
        const std::int32_t total     = scan[blockDim.x - 1];
        const std::int32_t tie_rank  = (inclusive >> 16) - tie;
        const std::int32_t ties_left = ties_s;
        const bool take_tie          = tie && tie_rank < ties_left;
        __syncthreads();
        scan[tid] = (greater || take_tie) ? 1 : 0;
        __syncthreads();
        for (int offset = 1; offset < static_cast<int>(blockDim.x); offset <<= 1) {
            const std::int32_t add = tid >= offset ? scan[tid - offset] : 0;
            __syncthreads();
            scan[tid] += add;
            __syncthreads();
        }
        if (greater || take_tie) {
            const std::int32_t slot = emitted_s + scan[tid] - 1;
            for (int t = 0; t < ratio; ++t) { selected[slot * ratio + t] = b * ratio + t; }
        }
        __syncthreads();
        if (tid == 0) {
            emitted_s += scan[blockDim.x - 1];
            ties_s = max(0, ties_left - (total >> 16));
        }
        __syncthreads();
    }
    for (int t = tid; t < tail; t += blockDim.x) {
        selected[keep * ratio + t] = blocks * ratio + t;
    }
    if (tid == 0) { a.counts[column] = keep * ratio + tail; }
}

void launch(SelectArgs args, std::int32_t columns, cudaStream_t stream) {
    constexpr std::int32_t kScoreTargetCtas = 256, kMinScoreBlocks = 32;
    const std::int32_t by_blocks = (args.max_blocks + kMinScoreBlocks - 1) / kMinScoreBlocks;
    const std::int32_t splits =
        std::max(1, std::min((kScoreTargetCtas + columns - 1) / columns, by_blocks));
    const std::int32_t split_blocks = (args.max_blocks + splits - 1) / splits;
    score_kernel<<<dim3(columns, splits), 256, 0, stream>>>(args, split_blocks);
    CUDA_CHECK(cudaGetLastError());
    select_kernel<<<columns, 256, 0, stream>>>(args);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ext

// ---------------------------------------------------------------------------------------------

struct Options {
    std::vector<int> depths              = {0, 16384, 65536};
    std::vector<int> widths              = {1, 4};
    std::vector<KvCacheStorage> storages = {KvCacheStorage::BFloat16,
                                            KvCacheStorage::Fp8E4M3Row256};
    int kv_heads                         = 2;
    int warmup                           = 20;
    int repeat                           = 100;
};

std::vector<int> parse_list(std::string_view text) {
    std::vector<int> out;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const std::size_t end = std::min(text.find(',', begin), text.size());
        out.push_back(std::atoi(std::string(text.substr(begin, end - begin)).c_str()));
        begin = end + 1;
    }
    return out;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a(argv[i]);
        const auto next = [&] {
            if (++i >= argc) { throw std::invalid_argument("missing value"); }
            return std::string_view(argv[i]);
        };
        if (a == "--depth") {
            o.depths = parse_list(next());
        } else if (a == "--width") {
            o.widths = parse_list(next());
        } else if (a == "--storage") {
            const auto v = next();
            o.storages.clear();
            if (v == "bf16" || v == "all") o.storages.push_back(KvCacheStorage::BFloat16);
            if (v == "fp8" || v == "all") o.storages.push_back(KvCacheStorage::Fp8E4M3Row256);
            if (v == "int8" || v == "all") o.storages.push_back(KvCacheStorage::Int8Group64);
        } else if (a == "--kv-heads") {
            o.kv_heads = std::atoi(std::string(next()).c_str());
        } else if (a == "--warmup") {
            o.warmup = std::atoi(std::string(next()).c_str());
        } else if (a == "--repeat") {
            o.repeat = std::atoi(std::string(next()).c_str());
        } else {
            std::printf(
                "usage: %s [--depth 0,16384,65536] [--width 1,4] [--storage bf16|fp8|int8|all]\n"
                "          [--kv-heads 2|1] [--warmup N] [--repeat N]\n",
                argv[0]);
            std::exit(a == "-h" || a == "--help" ? 0 : 1);
        }
    }
    return o;
}

struct Timing {
    double warm, cold;
};

Timing time_graph(const std::function<void(cudaStream_t)>& body, bench::L2FlushBuffer& flush,
                  cudaStream_t stream, const Options& o) {
    bench::TimedGraph graph;
    body(stream); // eager warm-up configures kernel attributes outside capture
    CUDA_CHECK(cudaStreamSynchronize(stream));
    graph.capture(stream, body);
    const double warm = bench::measure_graph(graph, stream, o.warmup, o.repeat).median_us;
    std::vector<double> samples;
    for (int i = 0; i < o.warmup + o.repeat; ++i) {
        bench::flush_l2(flush, stream);
        const double us = graph.launch_timed(stream);
        if (i >= o.warmup) { samples.push_back(us); }
    }
    return {warm, bench::summarize_timings(samples).median_us};
}

int run(const Options& o) {
    const ops::QsaGeometry base = ops::qwen3_8_flash_next_qsa_geometry();
    ops::QsaGeometry g          = base;
    g.kv_heads                  = o.kv_heads;
    g.query_heads               = 12 * o.kv_heads;
    const int max_sel           = g.max_selected();
    cudaStream_t stream         = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    bench::L2FlushBuffer flush(256ULL << 20);
    std::printf("op,storage,kv_heads,depth,W,warm_us,cold_us\n");

    for (const KvCacheStorage storage : o.storages) {
        const PagedKVStorageLayout layout = paged_kv_storage_layout(storage, kHeadDim);
        for (const int depth : o.depths) {
            const int capacity =
                depth + std::max(64, *std::max_element(o.widths.begin(), o.widths.end()));
            const int pages = (capacity + kPage - 1) / kPage;
            std::vector<int> table(pages);
            for (int p = 0; p < pages; ++p) { table[p] = pages - 1 - p; } // reversed physical pages
            DeviceBuffer d_table(pages * 4), d_rows(4);
            d_table.copy_from_host(table.data(), table.size() * 4);
            const int zero = 0;
            d_rows.copy_from_host(&zero, 4);
            const auto plane = [&](DType dtype, int extent) {
                auto buffer = std::make_unique<DeviceBuffer>(
                    std::max<std::size_t>(static_cast<std::size_t>(extent) * kPage * g.kv_heads *
                                              pages * dtype_size(dtype),
                                          256));
                buffer->fill(0);
                return buffer;
            };
            auto kp = plane(layout.key.data_dtype, layout.key.data_leading_extent);
            auto vp = plane(layout.value.data_dtype, layout.value.data_leading_extent);
            std::unique_ptr<DeviceBuffer> ks, vs;
            PagedKVBatchLayerView kv{};
            kv.k_pages = Tensor(kp->p, layout.key.data_dtype,
                                {layout.key.data_leading_extent, kPage, g.kv_heads, pages});
            kv.v_pages = Tensor(vp->p, layout.value.data_dtype,
                                {layout.value.data_leading_extent, kPage, g.kv_heads, pages});
            if (layout.key.has_scale()) {
                ks = plane(layout.key.scale_dtype, layout.key.scale_leading_extent);
                vs = plane(layout.value.scale_dtype, layout.value.scale_leading_extent);
                kv.k_scale_pages =
                    Tensor(ks->p, layout.key.scale_dtype,
                           {layout.key.scale_leading_extent, kPage, g.kv_heads, pages});
                kv.v_scale_pages =
                    Tensor(vs->p, layout.value.scale_dtype,
                           {layout.value.scale_leading_extent, kPage, g.kv_heads, pages});
            }
            kv.block_tables = Tensor(d_table.p, DType::I32, {pages, 1});
            kv.head_dim     = kHeadDim;
            kv.num_kv_heads = g.kv_heads;
            kv.storage      = storage;
            DeviceBuffer records(static_cast<std::size_t>(ops::kQsaIndexRecordWords) * kPage *
                                 pages * 2);
            DeviceBuffer blocks(static_cast<std::size_t>(ops::kQsaBlockKeyWords) * kPage * pages *
                                2);
            records.fill(0);
            blocks.fill(0);
            const ops::QsaIndexPlanes index{
                Tensor(records.p, DType::BF16, {ops::kQsaIndexRecordWords, kPage, 1, pages}),
                Tensor(blocks.p, DType::BF16, {ops::kQsaBlockKeyWords, kPage, 1, pages}),
                kv.block_tables};
            DeviceBuffer key_norm   = bench::make_bf16(kIndexDim, 3, -0.3F, 0.3F);
            DeviceBuffer query_norm = bench::make_bf16(kIndexDim, 4, -0.3F, 0.3F);
            const Tensor tkn(key_norm.p, DType::BF16, {kIndexDim});
            const Tensor tqn(query_norm.p, DType::BF16, {kIndexDim});
            const Tensor rows(d_rows.p, DType::I32, {1});

            // Fill the history [0, depth) in chunks.
            constexpr int kChunk = 4096;
            {
                DeviceBuffer k = bench::make_bf16(
                    static_cast<std::size_t>(kHeadDim) * g.kv_heads * kChunk, 11, -2.0F, 2.0F);
                DeviceBuffer v = bench::make_bf16(
                    static_cast<std::size_t>(kHeadDim) * g.kv_heads * kChunk, 12, -2.0F, 2.0F);
                DeviceBuffer pos(kChunk * 4), rope(kChunk * 3 * 4);
                for (int first = 0; first < depth; first += kChunk) {
                    // Distinct raw index keys per chunk (K/V contents do not affect timing).
                    DeviceBuffer ik =
                        bench::make_bf16(static_cast<std::size_t>(kIndexDim) * kChunk,
                                         13 + static_cast<std::uint64_t>(first), -2.0F, 2.0F);
                    const int n = std::min(kChunk, depth - first);
                    std::vector<int> p(n), r(static_cast<std::size_t>(n) * 3);
                    for (int t = 0; t < n; ++t) {
                        p[t] = first + t;
                        for (int a = 0; a < 3; ++a) {
                            r[static_cast<std::size_t>(a) * n + t] = first + t;
                        }
                    }
                    pos.copy_from_host(p.data(), p.size() * 4);
                    rope.copy_from_host(r.data(), r.size() * 4);
                    ops::qsa_append(Tensor(k.p, DType::BF16, {kHeadDim, g.kv_heads, n, 1}),
                                    Tensor(v.p, DType::BF16, {kHeadDim, g.kv_heads, n, 1}),
                                    Tensor(ik.p, DType::BF16, {kIndexDim, n, 1}),
                                    Tensor(rope.p, DType::I32, {n, 3}),
                                    Tensor(pos.p, DType::I32, {n, 1}), Tensor{}, rows, tkn, g, kv,
                                    index, stream);
                }
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            for (const int W : o.widths) {
                const std::uint32_t visible = static_cast<std::uint32_t>(depth + W);
                DeviceBuffer k              = bench::make_bf16(
                    static_cast<std::size_t>(kHeadDim) * g.kv_heads * W, 21, -2.0F, 2.0F);
                DeviceBuffer v = bench::make_bf16(
                    static_cast<std::size_t>(kHeadDim) * g.kv_heads * W, 22, -2.0F, 2.0F);
                DeviceBuffer ik =
                    bench::make_bf16(static_cast<std::size_t>(kIndexDim) * W, 23, -2.0F, 2.0F);
                DeviceBuffer iq =
                    bench::make_bf16(static_cast<std::size_t>(kIndexDim) * 4 * W, 24, -2.0F, 2.0F);
                DeviceBuffer q = bench::make_bf16(
                    static_cast<std::size_t>(kHeadDim) * g.query_heads * W, 25, -1.0F, 1.0F);
                DeviceBuffer out(static_cast<std::size_t>(kHeadDim) * g.query_heads * W * 2);
                DeviceBuffer sel(static_cast<std::size_t>(max_sel) * W * 4), cnt(W * 4);
                DeviceBuffer pos(W * 4), rope(W * 3 * 4);
                std::vector<int> p(W), r(static_cast<std::size_t>(W) * 3);
                for (int t = 0; t < W; ++t) {
                    p[t] = depth + t;
                    for (int a = 0; a < 3; ++a) {
                        r[static_cast<std::size_t>(a) * W + t] = depth + t;
                    }
                }
                pos.copy_from_host(p.data(), p.size() * 4);
                rope.copy_from_host(r.data(), r.size() * 4);
                const Tensor tpos(pos.p, DType::I32, {W, 1}), trope(rope.p, DType::I32, {W, 3});
                Tensor tsel(sel.p, DType::I32, {max_sel, W, 1}), tcnt(cnt.p, DType::I32, {W, 1});
                Tensor tout(out.p, DType::BF16, {kHeadDim, g.query_heads, W, 1});
                WorkspaceArena workspace(ops::qsa_select_workspace_bytes(g, visible, W, 1) +
                                         ops::qsa_attention_workspace_bytes(g, W) + (1 << 20));
                const auto append = [&](cudaStream_t s) {
                    ops::qsa_append(Tensor(k.p, DType::BF16, {kHeadDim, g.kv_heads, W, 1}),
                                    Tensor(v.p, DType::BF16, {kHeadDim, g.kv_heads, W, 1}),
                                    Tensor(ik.p, DType::BF16, {kIndexDim, W, 1}), trope, tpos,
                                    Tensor{}, rows, tkn, g, kv, index, s);
                };
                const auto select = [&](cudaStream_t s) {
                    ops::qsa_select(Tensor(iq.p, DType::BF16, {kIndexDim * 4, W, 1}), trope, tpos,
                                    Tensor{}, rows, tqn, index, g, visible, workspace, tsel, tcnt,
                                    s);
                };
                const auto attend = [&](cudaStream_t s) {
                    ops::qsa_attention(Tensor(q.p, DType::BF16, {kHeadDim, g.query_heads, W, 1}),
                                       tsel, tcnt, rows, kv, 1.0F / 16.0F, workspace, tout, s);
                };
                append(stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                const char* sname = storage == KvCacheStorage::BFloat16      ? "bf16"
                                    : storage == KvCacheStorage::Int8Group64 ? "int8"
                                                                             : "fp8";
                const auto report = [&](const char* op, Timing t) {
                    std::printf("%s,%s,%d,%d,%d,%.2f,%.2f\n", op, sname, g.kv_heads, depth, W,
                                t.warm, t.cold);
                    std::fflush(stdout);
                };
                report("append", time_graph(append, flush, stream, o));
                report("select", time_graph(select, flush, stream, o));
                report("attention", time_graph(attend, flush, stream, o));
                report("chain", time_graph(
                                    [&](cudaStream_t s) {
                                        append(s);
                                        select(s);
                                        attend(s);
                                    },
                                    flush, stream, o));
                if (storage == KvCacheStorage::BFloat16) {
                    // ninfer-ext's raw-record selection on the same records and queries.
                    ext::SelectArgs args{};
                    args.index_query     = static_cast<const __nv_bfloat16*>(iq.p);
                    args.query_positions = static_cast<const std::int32_t*>(rope.p);
                    args.query_axes      = 3;
                    args.cache           = static_cast<const std::int32_t*>(pos.p);
                    args.table_rows      = static_cast<const std::int32_t*>(d_rows.p);
                    args.tables          = static_cast<const std::int32_t*>(d_table.p);
                    args.table_stride    = pages;
                    args.records         = static_cast<const __nv_bfloat16*>(records.p);
                    args.query_norm      = static_cast<const __nv_bfloat16*>(query_norm.p);
                    args.key_norm        = static_cast<const __nv_bfloat16*>(key_norm.p);
                    args.max_blocks      = static_cast<std::int32_t>(visible / 4 + 1);
                    DeviceBuffer scores(static_cast<std::size_t>(args.max_blocks) * W * 4);
                    args.scores = static_cast<float*>(scores.p);
                    DeviceBuffer esel(static_cast<std::size_t>(max_sel) * W * 4), ecnt(W * 4);
                    args.selected     = static_cast<std::int32_t*>(esel.p);
                    args.counts       = static_cast<std::int32_t*>(ecnt.p);
                    args.width        = W;
                    args.lanes        = 1;
                    args.max_selected = max_sel;
                    args.geometry     = {g.index_heads, g.index_dim,  g.budget,       g.ratio,
                                         g.rotary_dim,  g.rope_theta, g.rms_norm_eps, {}};
                    for (int i = 0; i < 64; ++i) { args.geometry.pair_axes[i] = g.pair_axes[i]; }
                    report("ext-select",
                           time_graph([&](cudaStream_t s) { ext::launch(args, W, s); }, flush,
                                      stream, o));
                    // Same selected sets? (Both are exact top-k; report agreement, not a gate.)
                    select(stream);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    const auto a = std::vector<int>(static_cast<std::size_t>(max_sel) * W);
                    std::vector<int> mine(a.size()), theirs(a.size()), mc(W), tc(W);
                    CUDA_CHECK(
                        cudaMemcpy(mine.data(), sel.p, mine.size() * 4, cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(theirs.data(), esel.p, theirs.size() * 4,
                                          cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(mc.data(), cnt.p, W * 4, cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(tc.data(), ecnt.p, W * 4, cudaMemcpyDeviceToHost));
                    // FP64 oracle of the selection from the stored raw records (contract
                    // semantics), to tell which route departs from it and by how much.
                    const auto rec = [&] {
                        std::vector<std::uint16_t> h(
                            static_cast<std::size_t>(ops::kQsaIndexRecordWords) * kPage * pages);
                        CUDA_CHECK(
                            cudaMemcpy(h.data(), records.p, h.size() * 2, cudaMemcpyDeviceToHost));
                        return h;
                    }();
                    const auto host_bf16 = [](const DeviceBuffer& b, std::size_t n) {
                        std::vector<std::uint16_t> h(n);
                        CUDA_CHECK(cudaMemcpy(h.data(), b.p, n * 2, cudaMemcpyDeviceToHost));
                        std::vector<double> out(n);
                        for (std::size_t i = 0; i < n; ++i) {
                            std::uint32_t u = static_cast<std::uint32_t>(h[i]) << 16;
                            float f;
                            std::memcpy(&f, &u, 4);
                            out[i] = f;
                        }
                        return out;
                    };
                    const auto kn = host_bf16(key_norm, kIndexDim),
                               qn = host_bf16(query_norm, kIndexDim);
                    const auto qh = host_bf16(iq, static_cast<std::size_t>(kIndexDim) * 4 * W);
                    const auto bf = [](double x) {
                        float f = static_cast<float>(x);
                        std::uint32_t u;
                        std::memcpy(&u, &f, 4);
                        u += 0x7fffu + ((u >> 16) & 1u);
                        u &= 0xffff0000u;
                        std::memcpy(&f, &u, 4);
                        return static_cast<double>(f);
                    };
                    const auto rope = [&](std::vector<double>& x, double pos) {
                        for (int i = 0; i < 32; ++i) {
                            const float inv =
                                1.0F / std::pow(1.0e7F, static_cast<float>(2 * i) / 64.0F);
                            const double angle = static_cast<double>(static_cast<float>(pos) * inv);
                            const double a = x[i], b = x[i + 32];
                            x[i]      = a * std::cos(angle) - b * std::sin(angle);
                            x[i + 32] = b * std::cos(angle) + a * std::sin(angle);
                        }
                    };
                    const auto norm = [&](std::vector<double>& x, const std::vector<double>& w) {
                        double ss = 0.0;
                        for (double v : x) ss += v * v;
                        const double inv = 1.0 / std::sqrt(ss / 128.0 + 1e-6);
                        for (int d = 0; d < 128; ++d) x[d] *= inv * (1.0 + w[d]);
                    };
                    for (int c = 0; c < W; ++c) {
                        const int n = depth + c + 1, blocks_n = n / 4;
                        if (blocks_n <= 512) continue;
                        std::vector<std::vector<double>> qv(4);
                        for (int h = 0; h < 4; ++h) {
                            qv[h].assign(qh.begin() + (c * 4 + h) * 128,
                                         qh.begin() + (c * 4 + h + 1) * 128);
                            norm(qv[h], qn);
                            rope(qv[h], depth + c);
                        }
                        std::vector<double> score(blocks_n);
                        for (int b = 0; b < blocks_n; ++b) {
                            std::vector<double> m(128, 0.0);
                            for (int t = 0; t < 4; ++t) {
                                const int tok = 4 * b + t;
                                const std::size_t base =
                                    (static_cast<std::size_t>(table[tok / kPage]) * kPage +
                                     tok % kPage) *
                                    ops::kQsaIndexRecordWords;
                                for (int d = 0; d < 128; ++d) {
                                    std::uint32_t u = static_cast<std::uint32_t>(rec[base + d])
                                                      << 16;
                                    float f;
                                    std::memcpy(&f, &u, 4);
                                    m[d] += f;
                                }
                            }
                            for (double& x : m) x = bf(x / 4.0);
                            norm(m, kn);
                            rope(m, 4 * b);
                            double sc = 0.0;
                            for (int h = 0; h < 4; ++h) {
                                double dot = 0.0;
                                for (int d = 0; d < 128; ++d) dot += qv[h][d] * m[d];
                                sc += std::max(dot, 0.0);
                            }
                            score[b] = sc / std::sqrt(128.0);
                        }
                        std::vector<int> order(blocks_n);
                        std::iota(order.begin(), order.end(), 0);
                        std::stable_sort(order.begin(), order.end(),
                                         [&](int x, int y) { return score[x] > score[y]; });
                        const double cut = score[order[511]], top = score[order[0]];
                        std::vector<char> want(blocks_n, 0);
                        for (int i = 0; i < 512; ++i) want[order[i]] = 1;
                        const auto compare = [&](const std::vector<int>& sel_host, int count,
                                                 const char* who) {
                            std::vector<char> got(blocks_n, 0);
                            for (int i = 0; i < count; ++i) {
                                const int tok = sel_host[static_cast<std::size_t>(c) * max_sel + i];
                                if (tok / 4 < blocks_n) got[tok / 4] = 1;
                            }
                            int differ   = 0;
                            double worst = 0.0;
                            for (int b = 0; b < blocks_n; ++b) {
                                if (got[b] != want[b]) {
                                    ++differ;
                                    worst = std::max(worst, std::abs(score[b] - cut) / top);
                                }
                            }
                            std::printf("# depth=%d W=%d col=%d %s vs FP64 oracle: %d blocks "
                                        "differ, worst |score-cut|/max=%.2e\n",
                                        depth, W, c, who, differ, worst);
                        };
                        compare(mine, mc[c], "tp2");
                        compare(theirs, tc[c], "ext");
                    }
                }
            }
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse(argc, argv));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "qsa_bench: %s\n", error.what());
        return 1;
    }
}
