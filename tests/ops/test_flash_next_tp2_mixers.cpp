// Two-device parity of the Qwen3.8-Flash-Next mixers at tp 2: each rank runs its head shard
// of the same logical computation, and every rank output is compared with the single-device
// computation of the same global heads.
//
// QSA (12 attention layers + MTP): rank r holds KV head r and its 12 query heads (12/1); the
// indexer is replicated. Per KV profile allowed at tp 2 (BF16, INT8-G64, NVFP4-G16):
//  * append: each rank appends its KV head and the same indexer keys into its own planes;
//  * select: the replicated indexer on each device returns exactly the selection of the
//    single-device 24/2 layer (indices and counts), so the ranks need no exchange;
//  * attention: rank r's 12/1 output equals heads [12r, 12r+12) of the 24/2 output within the
//    27B head-local parity bound, twice the storage's registered attention criterion (the two
//    geometries partition the keys of a split-K pass differently; test_attention_headlocal.cpp
//    explains the bound). The number of bit-identical elements is reported.
// GDN (36 layers): rank r holds key heads [8r, 8r+8) with value heads [24r, 24r+24) (8/24); the
// composed mixer after the projections (projected conv, gating, recurrence with ReplaySSM record,
// sigmoid gated norm, then the 36-layer fold) is per head, so rank outputs, q/k/v convolution
// outputs, folded states and conv histories are expected bit for bit equal to the single-device
// 16/48 run's slices (as the 27B's test_gdn_headsplit.cpp), over one ordinary decode round and one
// MTP verify (W = 4, valid 4/3, commits 3/2).
//
// QSA layer fixture after tests/ops/test_sparse_attention.cpp, GDN fixture after
// tests/ops/test_flash_next_gdn.cpp. Needs two CUDA devices; 77 with fewer.
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/sparse_attention.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_storage.h"
#include "ops/attention_criteria.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

void set_device(int device) { cuda_check(cudaSetDevice(device), "cudaSetDevice"); }

float hashed(std::uint64_t key, float lo, float hi) {
    key += 0x9e3779b97f4a7c15ULL;
    key = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9ULL;
    key = (key ^ (key >> 27)) * 0x94d049bb133111ebULL;
    key ^= key >> 31;
    const double unit = static_cast<double>(key >> 11) * 0x1.0p-53;
    return bf16_to_f32(f32_to_bf16(static_cast<float>(lo + (hi - lo) * unit)));
}

std::uint64_t mix(std::uint64_t a, std::uint64_t b, std::uint64_t c = 0, std::uint64_t d = 0) {
    return (((a * 1000003ULL + b) * 1000033ULL + c) * 1000037ULL) + d;
}

// ==== QSA
// =========================================================================================

constexpr std::int32_t kHeadDim    = 256;
constexpr std::int32_t kIndexHeads = 4;
constexpr std::int32_t kIndexDim   = 128;
constexpr std::int32_t kPage       = kPagedKVPageSize;
constexpr std::int32_t kQueryHeads = 24;
constexpr std::int32_t kKvHeads    = 2;

const ops::QsaGeometry kGeometry = ops::qwen3_8_flash_next_qsa_geometry();

std::int32_t axis_position(std::int32_t token, int axis) { return token / (axis + 1) + 3 * axis; }

// A paged QSA layer of `lanes` sequences on the current device, holding `kv_heads` KV heads.
struct Layer {
    std::int32_t lanes, capacity, kv_heads, pages_per_lane, pages;
    KvCacheStorage storage;
    PagedKVStorageLayout layout;
    std::vector<int> table_row, tables;
    DeviceBuffer d_tables, d_rows;
    DeviceBuffer k_pages, v_pages, k_scales, v_scales, records, block_keys;
    PagedKVBatchLayerView kv{};
    ops::QsaIndexPlanes index{};

    Layer(std::int32_t lanes_, std::int32_t capacity_, KvCacheStorage storage_,
          std::int32_t kv_heads_)
        : lanes(lanes_), capacity(capacity_), kv_heads(kv_heads_),
          pages_per_lane((capacity_ + kPage - 1) / kPage), pages(pages_per_lane * lanes_),
          storage(storage_), layout(paged_kv_storage_layout(storage_, kHeadDim)) {
        table_row.resize(lanes);
        tables.assign(static_cast<std::size_t>(pages_per_lane) * lanes, -1);
        for (int b = 0; b < lanes; ++b) {
            table_row[b] = lanes - 1 - b;
            for (int p = 0; p < pages_per_lane; ++p) {
                tables[static_cast<std::size_t>(table_row[b]) * pages_per_lane + p] =
                    (pages_per_lane - 1 - p) * lanes + b;
            }
        }
        d_tables         = to_device_i32(tables);
        d_rows           = to_device_i32(table_row);
        const auto plane = [&](DeviceBuffer& buffer, DType dtype, std::int32_t extent) {
            const std::size_t bytes =
                static_cast<std::size_t>(extent) * kPage * kv_heads * pages * dtype_size(dtype);
            buffer = DeviceBuffer(std::max<std::size_t>(bytes, 256));
            buffer.fill(0);
            return Tensor(buffer.p, dtype, {extent, kPage, kv_heads, pages});
        };
        kv.k_pages = plane(k_pages, layout.key.data_dtype, layout.key.data_leading_extent);
        kv.v_pages = plane(v_pages, layout.value.data_dtype, layout.value.data_leading_extent);
        if (layout.key.has_scale()) {
            kv.k_scale_pages =
                plane(k_scales, layout.key.scale_dtype, layout.key.scale_leading_extent);
        }
        if (layout.value.has_scale()) {
            kv.v_scale_pages =
                plane(v_scales, layout.value.scale_dtype, layout.value.scale_leading_extent);
        }
        kv.block_tables = Tensor(d_tables.p, DType::I32, {pages_per_lane, lanes});
        kv.head_dim     = kHeadDim;
        kv.num_kv_heads = kv_heads;
        kv.storage      = storage;
        records =
            DeviceBuffer(static_cast<std::size_t>(ops::kQsaIndexRecordWords) * kPage * pages * 2);
        records.fill(0);
        block_keys =
            DeviceBuffer(static_cast<std::size_t>(ops::kQsaBlockKeyWords) * kPage * pages * 2);
        block_keys.fill(0);
        index.records =
            Tensor(records.p, DType::BF16, {ops::kQsaIndexRecordWords, kPage, 1, pages});
        index.block_keys =
            Tensor(block_keys.p, DType::BF16, {ops::kQsaBlockKeyWords, kPage, 1, pages});
        index.block_tables = kv.block_tables;
    }

    Tensor rows() const { return Tensor(d_rows.p, DType::I32, {lanes}); }
};

float k_value(int lane, int token, int head, int d) {
    return hashed(mix(1, lane, token, head * 1000 + d), -2.0F, 2.0F);
}

float v_value(int lane, int token, int head, int d) {
    return hashed(mix(2, lane, token, head * 1000 + d), -2.0F, 2.0F);
}

float index_key_value(int lane, int token, int d) {
    return hashed(mix(3, lane, token, d), -2.0F, 2.0F);
}

std::vector<std::uint16_t> norm_bits(std::uint64_t seed) {
    std::vector<std::uint16_t> w(kIndexDim);
    for (int d = 0; d < kIndexDim; ++d) { w[d] = f32_to_bf16(hashed(mix(seed, d), -0.3F, 0.3F)); }
    return w;
}

// Appends tokens [0, tokens) of every lane, KV heads [first_head, first_head + layer.kv_heads).
void append_all(Layer& layer, int first_head, int tokens, const DeviceBuffer& key_norm) {
    constexpr int kChunk = 400;
    const int B          = layer.lanes;
    for (int first = 0; first < tokens; first += kChunk) {
        const int W               = std::min(kChunk, tokens - first);
        const std::size_t columns = static_cast<std::size_t>(W) * B;
        std::vector<std::uint16_t> k(columns * layer.kv_heads * kHeadDim),
            v(columns * layer.kv_heads * kHeadDim), idx(columns * kIndexDim);
        std::vector<int> cache(columns), rope(columns * 3), valid(B, W);
        for (int b = 0; b < B; ++b) {
            for (int w = 0; w < W; ++w) {
                const std::size_t c = static_cast<std::size_t>(b) * W + w;
                const int t         = first + w;
                cache[c]            = t;
                for (int a = 0; a < 3; ++a) { rope[a * columns + c] = axis_position(t, a); }
                for (int h = 0; h < layer.kv_heads; ++h) {
                    for (int d = 0; d < kHeadDim; ++d) {
                        const std::size_t at = (c * layer.kv_heads + h) * kHeadDim + d;
                        k[at]                = f32_to_bf16(k_value(b, t, first_head + h, d));
                        v[at]                = f32_to_bf16(v_value(b, t, first_head + h, d));
                    }
                }
                for (int d = 0; d < kIndexDim; ++d) {
                    idx[c * kIndexDim + d] = f32_to_bf16(index_key_value(b, t, d));
                }
            }
        }
        DeviceBuffer dk = to_device(k), dv = to_device(v), di = to_device(idx);
        DeviceBuffer dc = to_device_i32(cache), dr = to_device_i32(rope),
                     dvalid = to_device_i32(valid);
        ops::qsa_append(Tensor(dk.p, DType::BF16, {kHeadDim, layer.kv_heads, W, B}),
                        Tensor(dv.p, DType::BF16, {kHeadDim, layer.kv_heads, W, B}),
                        Tensor(di.p, DType::BF16, {kIndexDim, W, B}),
                        Tensor(dr.p, DType::I32, {static_cast<std::int32_t>(columns), 3}),
                        Tensor(dc.p, DType::I32, {W, B}), Tensor(dvalid.p, DType::I32, {B}),
                        layer.rows(), Tensor(key_norm.p, DType::BF16, {kIndexDim}), kGeometry,
                        layer.kv, layer.index, nullptr);
        cuda_check(cudaDeviceSynchronize(), "qsa_append");
    }
}

struct Query {
    std::int32_t width;
    std::vector<std::vector<int>> positions; // per lane
    std::vector<int> valid;
};

struct Selection {
    std::vector<int> selected, counts;
};

Selection select(const Layer& layer, const Query& query, std::uint32_t max_visible,
                 const DeviceBuffer& query_norm) {
    const int W = query.width, B = layer.lanes;
    const std::size_t columns = static_cast<std::size_t>(W) * B;
    std::vector<std::uint16_t> q(static_cast<std::size_t>(kIndexHeads) * kIndexDim * columns);
    for (std::size_t i = 0; i < q.size(); ++i) {
        q[i] = f32_to_bf16(hashed(mix(0x51, i), -2.0F, 2.0F));
    }
    std::vector<int> cache(columns), rope(columns * 3);
    for (int b = 0; b < B; ++b) {
        for (int w = 0; w < W; ++w) {
            const std::size_t c = static_cast<std::size_t>(b) * W + w;
            cache[c]            = query.positions[b][w];
            for (int a = 0; a < 3; ++a) { rope[a * columns + c] = axis_position(cache[c], a); }
        }
    }
    DeviceBuffer dq = to_device(q), dc = to_device_i32(cache), dr = to_device_i32(rope),
                 dvalid = to_device_i32(query.valid);
    const int max_sel   = kGeometry.max_selected();
    GuardedDeviceBuffer selected(static_cast<std::size_t>(max_sel) * columns * 4);
    GuardedDeviceBuffer counts(columns * 4);
    selected.fill(0xff);
    counts.fill(0xff);
    Tensor tsel(selected.data(), DType::I32, {max_sel, W, B});
    Tensor tcnt(counts.data(), DType::I32, {W, B});
    WorkspaceArena workspace(ops::qsa_select_workspace_bytes(kGeometry, max_visible, W, B));
    cuda_check(cudaDeviceSynchronize(), "select inputs");
    ops::qsa_select(Tensor(dq.p, DType::BF16, {kIndexHeads * kIndexDim, W, B}),
                    Tensor(dr.p, DType::I32, {static_cast<std::int32_t>(columns), 3}),
                    Tensor(dc.p, DType::I32, {W, B}), Tensor(dvalid.p, DType::I32, {B}),
                    layer.rows(), Tensor(query_norm.p, DType::BF16, {kIndexDim}), layer.index,
                    kGeometry, max_visible, workspace, tsel, tcnt, nullptr);
    cuda_check(cudaDeviceSynchronize(), "qsa_select");
    if (selected.verify_guards("qsa_select selected") + counts.verify_guards("qsa_select counts") !=
        0) {
        throw std::runtime_error("qsa_select wrote outside its outputs");
    }
    Selection out{from_device<int>(selected.data(), static_cast<std::size_t>(max_sel) * columns),
                  from_device<int>(counts.data(), columns)};
    // Entries past a count are unspecified: normalize them before comparing selections.
    for (std::size_t c = 0; c < columns; ++c) {
        for (int i = std::max(out.counts[c], 0); i < max_sel; ++i) {
            out.selected[c * max_sel + static_cast<std::size_t>(i)] = -1;
        }
    }
    return out;
}

// qsa_attention of `heads` query heads (global heads [first_head, first_head + heads)) over the
// layer's KV heads, with the given selection. Returns the BF16 output bits [256, heads, W, B].
std::vector<std::uint16_t> attend(const Layer& layer, int first_head, int heads, int width,
                                  const Selection& selection) {
    const int B = layer.lanes;
    std::vector<std::uint16_t> q(static_cast<std::size_t>(kHeadDim) * heads * width * B);
    for (int c = 0; c < width * B; ++c) {
        for (int h = 0; h < heads; ++h) {
            for (int d = 0; d < kHeadDim; ++d) {
                q[(static_cast<std::size_t>(c) * heads + h) * kHeadDim + d] =
                    f32_to_bf16(hashed(mix(0x61, c, first_head + h, d), -1.0F, 1.0F));
            }
        }
    }
    const int max_sel = kGeometry.max_selected();
    DeviceBuffer dq = to_device(q), dsel = to_device_i32(selection.selected),
                 dcnt = to_device_i32(selection.counts);
    GuardedDeviceBuffer out(q.size() * 2);
    out.fill(0xff);
    Tensor tq(dq.p, DType::BF16, {kHeadDim, heads, width, B});
    Tensor ts(dsel.p, DType::I32, {max_sel, width, B});
    Tensor tc(dcnt.p, DType::I32, {width, B});
    Tensor tout(out.data(), DType::BF16, {kHeadDim, heads, width, B});
    ops::QsaGeometry g = kGeometry;
    g.query_heads      = heads;
    g.kv_heads         = layer.kv_heads;
    WorkspaceArena workspace(ops::qsa_attention_workspace_bytes(g, width * B));
    cuda_check(cudaDeviceSynchronize(), "attention inputs");
    ops::qsa_attention(tq, ts, tc, layer.rows(), layer.kv, 1.0F / 16.0F, workspace, tout, nullptr);
    cuda_check(cudaDeviceSynchronize(), "qsa_attention");
    if (out.verify_guards("qsa_attention out") != 0) {
        throw std::runtime_error("qsa_attention wrote outside its output");
    }
    return from_device<std::uint16_t>(out.data(), q.size());
}

const char* profile_name(KvCacheStorage s) {
    switch (s) {
    case KvCacheStorage::BFloat16:
        return "BF16";
    case KvCacheStorage::Int8Group64:
        return "INT8";
    case KvCacheStorage::Nvfp4Group16:
        return "NVFP4";
    default:
        return "?";
    }
}

int test_qsa() {
    int failures          = 0;
    constexpr int kLanes  = 2;
    constexpr int kTokens = 2600; // 650 blocks > 512: the indexer selects
    constexpr int kWidth  = 4;
    const Query query{
        kWidth,
        {{kTokens - 4, kTokens - 3, kTokens - 2, kTokens - 1}, {1997, 1998, 2412, 2413}},
        {4, 3}};
    for (const KvCacheStorage storage :
         {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Nvfp4Group16}) {
        const std::string label = std::string("qsa tp2 ") + profile_name(storage);
        // Single device: 24/2.
        set_device(0);
        const DeviceBuffer key_norm0   = to_device(norm_bits(0x23)),
                           query_norm0 = to_device(norm_bits(0x22));
        Layer full(kLanes, kTokens, storage, kKvHeads);
        append_all(full, 0, kTokens, key_norm0);
        const Selection full_selection = select(full, query, kTokens, query_norm0);
        const auto full_out            = attend(full, 0, kQueryHeads, kWidth, full_selection);
        const int group                = kQueryHeads / kKvHeads; // 12
        for (int rank = 0; rank < 2; ++rank) {
            set_device(rank);
            const DeviceBuffer key_norm   = to_device(norm_bits(0x23)),
                               query_norm = to_device(norm_bits(0x22));
            Layer shard(kLanes, kTokens, storage, 1);
            append_all(shard, rank, kTokens, key_norm);
            const Selection shard_selection = select(shard, query, kTokens, query_norm);
            const std::string rank_label    = label + " rank " + std::to_string(rank);
            failures += verify_exact((rank_label + " selection == 1 GPU").c_str(),
                                     shard_selection.selected, full_selection.selected);
            failures += verify_exact((rank_label + " counts == 1 GPU").c_str(),
                                     shard_selection.counts, full_selection.counts);
            const auto out = attend(shard, rank * group, group, kWidth, shard_selection);
            // Heads [12r, 12r+12) of every column of the 24/2 output.
            std::vector<double> got, want;
            std::size_t identical = 0;
            for (int c = 0; c < kWidth * kLanes; ++c) {
                for (int h = 0; h < group; ++h) {
                    for (int d = 0; d < kHeadDim; ++d) {
                        const std::uint16_t a =
                            out[(static_cast<std::size_t>(c) * group + h) * kHeadDim + d];
                        const std::uint16_t b =
                            full_out[(static_cast<std::size_t>(c) * kQueryHeads + rank * group +
                                      h) *
                                         kHeadDim +
                                     d];
                        identical += a == b;
                        got.push_back(bf16_to_f32(a));
                        want.push_back(bf16_to_f32(b));
                    }
                }
            }
            std::cout << rank_label << ": attention " << identical << " of " << got.size()
                      << " elements bit-identical to the 24/2 heads\n";
            failures += verify_reduction(rank_label + " attention parity", got, want,
                                         attention_parity_criterion(storage));
        }
    }
    return failures;
}

// ==== GDN
// =========================================================================================

constexpr int kDim        = 128;
constexpr int kLayers     = 36;
constexpr int kLayer      = 7;
constexpr int kRows       = 2;
constexpr int kSlots      = 4;
constexpr double kNormEps = 1.0e-6;
constexpr float kScale    = 0.08838834764831845F; // 1/sqrt(128)

struct Geometry {
    int qk_heads, value_heads;

    int query_rows() const { return qk_heads * kDim; }

    int value_rows() const { return value_heads * kDim; }

    int channels() const { return 2 * query_rows() + value_rows(); }
};

constexpr Geometry kFull{16, 48};
constexpr Geometry kShard{8, 24};

// Global channel of shard channel `c` of rank `rank` (Q|K|V sections, each rank its heads).
int global_channel(int rank, int c) {
    const int q = kShard.query_rows(), v = kShard.value_rows();
    if (c < q) { return rank * q + c; }
    if (c < 2 * q) { return kFull.query_rows() + rank * q + (c - q); }
    return 2 * kFull.query_rows() + rank * v + (c - 2 * q);
}

// The whole logical layer state and parameters, generated once over the 16/48 geometry.
struct Logical {
    std::vector<float> conv_weight;          // [4][C] tap-major
    std::vector<float> a_log, dt_bias, norm; // [48], [48], [128]
    std::vector<std::vector<float>> history; // per slot [3][C]
    std::vector<std::vector<float>> state;   // per slot [48][128][128]
};

Logical make_logical() {
    Logical l;
    const int C = kFull.channels(), Hv = kFull.value_heads;
    l.conv_weight.resize(static_cast<std::size_t>(C) * 4);
    for (std::size_t i = 0; i < l.conv_weight.size(); ++i) {
        l.conv_weight[i] = hashed(mix(11, i), -0.6F, 0.6F);
    }
    for (int h = 0; h < Hv; ++h) {
        l.a_log.push_back(hashed(mix(12, h), -1.0F, 1.5F));
        l.dt_bias.push_back(hashed(mix(13, h), -1.0F, 1.0F));
    }
    for (int d = 0; d < kDim; ++d) { l.norm.push_back(hashed(mix(14, d), 0.5F, 1.5F)); }
    for (int slot = 0; slot < kSlots; ++slot) {
        std::vector<float> history(static_cast<std::size_t>(C) * 3);
        for (std::size_t i = 0; i < history.size(); ++i) {
            history[i] = hashed(mix(15, slot, i), -1.0F, 1.0F);
        }
        std::vector<float> state(static_cast<std::size_t>(kDim) * kDim * Hv);
        for (std::size_t i = 0; i < state.size(); ++i) {
            // FP32 state, not BF16-rounded: use the raw hash range.
            state[i] = 0.05F * hashed(mix(16, slot, i), -1.0F, 1.0F) +
                       1.0e-4F * hashed(mix(17, slot, i), -1.0F, 1.0F);
        }
        l.history.push_back(std::move(history));
        l.state.push_back(std::move(state));
    }
    return l;
}

std::vector<std::uint16_t> bits(const std::vector<float>& v) {
    std::vector<std::uint16_t> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) { out[i] = f32_to_bf16(v[i]); }
    return out;
}

// One device's GDN layer of geometry g (the full one, or rank `rank`'s shard).
struct GdnDevice {
    Geometry g;
    int rank; // -1: the single-device 16/48 layer
    DeviceBuffer conv_weight, a_log, dt_bias, norm, state_storage, record_storage;
    LinearAttentionStatePoolLayout state_layout;
    std::unique_ptr<LinearAttentionStatePool> pool;
    GdnReplayRecordLayout record_layout;
    GdnReplayRecords records;
    std::unique_ptr<ops::GdnReplayFoldPlan> fold;

    int channel(int c) const { return rank < 0 ? c : global_channel(rank, c); }

    int head(int h) const { return rank < 0 ? h : rank * kShard.value_heads + h; }

    GdnDevice(Geometry geometry, int r, const Logical& l, int verify_width) : g(geometry), rank(r) {
        const int C = g.channels(), Hv = g.value_heads, CF = kFull.channels();
        std::vector<float> conv(static_cast<std::size_t>(C) * 4);
        for (int j = 0; j < 4; ++j) {
            for (int c = 0; c < C; ++c) {
                conv[static_cast<std::size_t>(j) * C + c] =
                    l.conv_weight[static_cast<std::size_t>(j) * CF + channel(c)];
            }
        }
        std::vector<float> a_log_v, dt_v;
        for (int h = 0; h < Hv; ++h) {
            a_log_v.push_back(l.a_log[head(h)]);
            dt_v.push_back(l.dt_bias[head(h)]);
        }
        conv_weight = to_device(bits(conv));
        a_log       = to_device(a_log_v);
        dt_bias     = to_device(dt_v);
        norm        = to_device(bits(l.norm));

        LayoutBuilder state_builder;
        state_layout = plan_linear_attention_state_pool(
            state_builder, {.layers         = static_cast<std::uint32_t>(kLayers),
                            .conv_channels  = C,
                            .conv_width     = 3,
                            .value_heads    = Hv,
                            .value_head_dim = kDim,
                            .key_head_dim   = kDim,
                            .slot_count     = kSlots,
                            .conv_dtype     = DType::BF16});
        const std::size_t state_bytes = state_builder.finish(256);
        state_storage                 = DeviceBuffer(state_bytes);
        state_storage.fill(0);
        pool = std::make_unique<LinearAttentionStatePool>(DeviceSpan{state_storage.p, state_bytes},
                                                          state_layout);
        for (int slot = 0; slot < kSlots; ++slot) {
            std::vector<float> history(static_cast<std::size_t>(C) * 3);
            for (int j = 0; j < 3; ++j) {
                for (int c = 0; c < C; ++c) {
                    history[static_cast<std::size_t>(j) * C + c] =
                        l.history[slot][static_cast<std::size_t>(j) * CF + channel(c)];
                }
            }
            const auto h_bits      = bits(history);
            const Tensor conv_slot = pool->conv_slot(kLayer, slot);
            cuda_check(cudaMemcpy(conv_slot.data, h_bits.data(), h_bits.size() * 2,
                                  cudaMemcpyHostToDevice),
                       "upload history");
            const std::size_t per_head = static_cast<std::size_t>(kDim) * kDim;
            const Tensor rec           = pool->recurrent_slot(kLayer, slot);
            cuda_check(cudaMemcpy(rec.data, l.state[slot].data() + head(0) * per_head,
                                  per_head * Hv * 4, cudaMemcpyHostToDevice),
                       "upload state");
        }
        LayoutBuilder builder;
        record_layout           = plan_gdn_replay_records(builder, {.layers          = kLayers,
                                                                    .record_capacity = kRows,
                                                                    .width           = verify_width,
                                                                    .conv_channels   = C,
                                                                    .qk_heads        = g.qk_heads,
                                                                    .value_heads     = Hv,
                                                                    .key_dim         = kDim,
                                                                    .value_dim       = kDim});
        const std::size_t bytes = builder.finish(256);
        record_storage          = DeviceBuffer(bytes);
        record_storage.fill(0);
        records = GdnReplayRecords({record_storage.p, bytes}, record_layout);
        fold    = std::make_unique<ops::GdnReplayFoldPlan>(records, pool->all_layers_view());
    }

    std::vector<std::uint16_t> history(int slot) const {
        const Tensor conv = pool->conv_slot(kLayer, slot);
        return from_device<std::uint16_t>(conv.data, conv.numel());
    }

    std::vector<std::uint32_t> state(int slot) const {
        const Tensor rec = pool->recurrent_slot(kLayer, slot);
        return from_device<std::uint32_t>(rec.data, rec.numel());
    }
};

// Logical round inputs over the 16/48 geometry: projected [C, W] per row, z [Hv*128, W],
// a, b [Hv, W].
struct LogicalInputs {
    std::vector<std::vector<float>> projected, z, a, b; // per row
};

LogicalInputs make_inputs(int width, std::uint64_t seed) {
    LogicalInputs in;
    const int C = kFull.channels(), V = kFull.value_rows(), Hv = kFull.value_heads;
    for (int row = 0; row < kRows; ++row) {
        std::vector<float> p(static_cast<std::size_t>(C) * width),
            z(static_cast<std::size_t>(V) * width), a(static_cast<std::size_t>(Hv) * width),
            b(static_cast<std::size_t>(Hv) * width);
        for (std::size_t i = 0; i < p.size(); ++i) {
            p[i] = hashed(mix(seed, row, 1, i), -1.5F, 1.5F);
        }
        for (std::size_t i = 0; i < z.size(); ++i) {
            z[i] = hashed(mix(seed, row, 2, i), -3.0F, 3.0F);
        }
        for (std::size_t i = 0; i < a.size(); ++i) {
            a[i] = hashed(mix(seed, row, 3, i), -3.0F, 3.0F);
        }
        for (std::size_t i = 0; i < b.size(); ++i) {
            b[i] = hashed(mix(seed, row, 4, i), -3.0F, 3.0F);
        }
        in.projected.push_back(std::move(p));
        in.z.push_back(std::move(z));
        in.a.push_back(std::move(a));
        in.b.push_back(std::move(b));
    }
    return in;
}

// Device buffers of one round, the device's slice of the logical inputs, [.., W, rows].
struct Round {
    int width;
    DeviceBuffer projected, z, a, b, q, k, v, gg, beta, o, y, valid, src, dst;

    Round(const GdnDevice& d, const LogicalInputs& in, int w, const std::vector<int>& valid_rows,
          const std::vector<int>& src_slots, const std::vector<int>& dst_slots)
        : width(w) {
        const Geometry& g = d.g;
        const int C = g.channels(), V = g.value_rows(), Hv = g.value_heads;
        const int CF = kFull.channels(), VF = kFull.value_rows(), HF = kFull.value_heads;
        std::vector<float> p, zz, aa, bb;
        for (int row = 0; row < kRows; ++row) {
            for (int t = 0; t < w; ++t) {
                for (int c = 0; c < C; ++c) {
                    p.push_back(in.projected[row][static_cast<std::size_t>(t) * CF + d.channel(c)]);
                }
                for (int r = 0; r < V; ++r) {
                    zz.push_back(
                        in.z[row][static_cast<std::size_t>(t) * VF + d.head(0) * kDim + r]);
                }
                for (int h = 0; h < Hv; ++h) {
                    aa.push_back(in.a[row][static_cast<std::size_t>(t) * HF + d.head(h)]);
                    bb.push_back(in.b[row][static_cast<std::size_t>(t) * HF + d.head(h)]);
                }
            }
        }
        projected                 = to_device(bits(p));
        z                         = to_device(bits(zz));
        a                         = to_device(bits(aa));
        b                         = to_device(bits(bb));
        const std::size_t columns = static_cast<std::size_t>(w) * kRows;
        q       = DeviceBuffer(static_cast<std::size_t>(g.query_rows()) * columns * 2);
        k       = DeviceBuffer(static_cast<std::size_t>(g.query_rows()) * columns * 2);
        this->v = DeviceBuffer(static_cast<std::size_t>(V) * columns * 2);
        gg      = DeviceBuffer(static_cast<std::size_t>(Hv) * columns * 4);
        beta    = DeviceBuffer(static_cast<std::size_t>(Hv) * columns * 4);
        o       = DeviceBuffer(static_cast<std::size_t>(V) * columns * 2);
        y       = DeviceBuffer(static_cast<std::size_t>(V) * columns * 2);
        y.fill(0);
        this->valid = to_device_i32(valid_rows);
        src         = to_device_i32(src_slots);
        dst         = to_device_i32(dst_slots);
    }
};

void gate_and_norm(const GdnDevice& d, Round& r, bool norm_only, cudaStream_t stream) {
    const Geometry& g = d.g;
    const int columns = r.width * kRows;
    if (!norm_only) {
        Tensor gg(r.gg.p, DType::FP32, {g.value_heads, columns});
        Tensor beta(r.beta.p, DType::FP32, {g.value_heads, columns});
        ops::gdn_gating(Tensor(r.a.p, DType::BF16, {g.value_heads, columns}),
                        Tensor(r.b.p, DType::BF16, {g.value_heads, columns}),
                        Tensor(d.a_log.p, DType::FP32, {g.value_heads}),
                        Tensor(d.dt_bias.p, DType::FP32, {g.value_heads}), gg, beta, stream);
        return;
    }
    Tensor y(r.y.p, DType::BF16, {kDim, g.value_heads, columns});
    ops::gated_rmsnorm_sigmoid(Tensor(r.o.p, DType::BF16, {kDim, g.value_heads, columns}),
                               Tensor(d.norm.p, DType::BF16, {kDim}),
                               Tensor(r.z.p, DType::BF16, {kDim, g.value_heads, columns}),
                               static_cast<float>(kNormEps), y, stream);
}

void decode_round(GdnDevice& d, Round& r, cudaStream_t stream) {
    const Geometry& g = d.g;
    const auto layer  = d.pool->layer_view(kLayer);
    Tensor conv       = layer.conv;
    Tensor q(r.q.p, DType::BF16, {g.query_rows(), 1, kRows});
    Tensor k(r.k.p, DType::BF16, {g.query_rows(), 1, kRows});
    Tensor v(r.v.p, DType::BF16, {g.value_rows(), 1, kRows});
    const Tensor src(r.src.p, DType::I32, {kRows}), dst(r.dst.p, DType::I32, {kRows});
    ops::gdn_projected_conv_snapshot(Tensor(r.projected.p, DType::BF16, {g.channels(), 1, kRows}),
                                     Tensor(d.conv_weight.p, DType::BF16, {g.channels(), 4}), conv,
                                     Tensor{}, src, dst, q, k, v, stream);
    gate_and_norm(d, r, false, stream);
    Tensor recurrent = layer.recurrent;
    Tensor o(r.o.p, DType::BF16, {kDim, g.value_heads, 1, kRows});
    ops::gated_delta_net_batch_update(Tensor(r.q.p, DType::BF16, {kDim, g.qk_heads, 1, kRows}),
                                      Tensor(r.k.p, DType::BF16, {kDim, g.qk_heads, 1, kRows}),
                                      Tensor(r.v.p, DType::BF16, {kDim, g.value_heads, 1, kRows}),
                                      Tensor(r.gg.p, DType::FP32, {g.value_heads, 1, kRows}),
                                      Tensor(r.beta.p, DType::FP32, {g.value_heads, 1, kRows}),
                                      kScale, true, recurrent, src, dst, o, stream);
    gate_and_norm(d, r, true, stream);
}

void verify_round(GdnDevice& d, Round& r, cudaStream_t stream) {
    const Geometry& g            = d.g;
    const int W                  = r.width;
    const auto layer             = d.pool->layer_view(kLayer);
    GdnReplayRecordLayer records = d.records.layer(kLayer, kRows);
    Tensor record                = records.conv;
    cuda_check(cudaMemcpyAsync(record.data, r.projected.p, record.bytes(), cudaMemcpyDeviceToDevice,
                               stream),
               "write conv record");
    Tensor q(r.q.p, DType::BF16, {g.query_rows(), W, kRows});
    Tensor k(r.k.p, DType::BF16, {g.query_rows(), W, kRows});
    Tensor v(r.v.p, DType::BF16, {g.value_rows(), W, kRows});
    const Tensor valid(r.valid.p, DType::I32, {kRows}), src(r.src.p, DType::I32, {kRows});
    ops::gdn_projected_conv_record(record, Tensor(d.conv_weight.p, DType::BF16, {g.channels(), 4}),
                                   layer.conv, valid, src, q, k, v, stream);
    gate_and_norm(d, r, false, stream);
    Tensor o(r.o.p, DType::BF16, {kDim, g.value_heads, W, kRows});
    Tensor key_record   = records.key;
    Tensor value_record = records.value;
    Tensor gate_record  = records.gate;
    ops::gated_delta_net_replay_record(Tensor(r.q.p, DType::BF16, {kDim, g.qk_heads, W, kRows}),
                                       Tensor(r.k.p, DType::BF16, {kDim, g.qk_heads, W, kRows}),
                                       Tensor(r.v.p, DType::BF16, {kDim, g.value_heads, W, kRows}),
                                       Tensor(r.gg.p, DType::FP32, {g.value_heads, W, kRows}),
                                       Tensor(r.beta.p, DType::FP32, {g.value_heads, W, kRows}),
                                       kScale, layer.recurrent, valid, src, key_record,
                                       value_record, gate_record, o, stream);
    gate_and_norm(d, r, true, stream);
}

// Compares rank `rank`'s round outputs with the 16/48 run's slices of the same heads/channels.
int compare_round(const GdnDevice& full, const Round& rf, const GdnDevice& shard, const Round& rs,
                  int rank, const std::string& label) {
    int failures     = 0;
    const int W      = rf.width;
    const auto cols  = static_cast<std::size_t>(W) * kRows;
    const auto slice = [&](const DeviceBuffer& whole_buffer, int whole_rows, int rows, int first,
                           int device_whole) {
        set_device(device_whole);
        const auto all =
            from_device<std::uint16_t>(whole_buffer.p, static_cast<std::size_t>(whole_rows) * cols);
        std::vector<std::uint16_t> out;
        for (std::size_t c = 0; c < cols; ++c) {
            out.insert(out.end(), all.begin() + static_cast<std::ptrdiff_t>(c * whole_rows + first),
                       all.begin() + static_cast<std::ptrdiff_t>(c * whole_rows + first + rows));
        }
        return out;
    };
    const auto own = [&](const DeviceBuffer& buffer, int rows) {
        set_device(rank);
        return from_device<std::uint16_t>(buffer.p, static_cast<std::size_t>(rows) * cols);
    };
    const int qs = kShard.query_rows(), vs = kShard.value_rows();
    failures += verify_exact((label + " y").c_str(), own(rs.y, vs),
                             slice(rf.y, kFull.value_rows(), vs, rank * vs, 0));
    failures += verify_exact((label + " q").c_str(), own(rs.q, qs),
                             slice(rf.q, kFull.query_rows(), qs, rank * qs, 0));
    failures += verify_exact((label + " k").c_str(), own(rs.k, qs),
                             slice(rf.k, kFull.query_rows(), qs, rank * qs, 0));
    failures += verify_exact((label + " v").c_str(), own(rs.v, vs),
                             slice(rf.v, kFull.value_rows(), vs, rank * vs, 0));
    (void)full;
    (void)shard;
    return failures;
}

int compare_slot(const GdnDevice& full, const GdnDevice& shard, int rank, int slot,
                 const std::string& label) {
    int failures = 0;
    set_device(0);
    const auto full_state   = full.state(slot);
    const auto full_history = full.history(slot);
    set_device(rank);
    const auto state           = shard.state(slot);
    const auto history         = shard.history(slot);
    const std::size_t per_head = static_cast<std::size_t>(kDim) * kDim;
    const std::vector<std::uint32_t> want_state(
        full_state.begin() + static_cast<std::ptrdiff_t>(rank * kShard.value_heads * per_head),
        full_state.begin() +
            static_cast<std::ptrdiff_t>((rank + 1) * kShard.value_heads * per_head));
    failures += verify_exact((label + " state").c_str(), state, want_state);
    const int C = kShard.channels(), CF = kFull.channels();
    std::vector<std::uint16_t> want_history(static_cast<std::size_t>(C) * 3);
    for (int j = 0; j < 3; ++j) {
        for (int c = 0; c < C; ++c) {
            want_history[static_cast<std::size_t>(j) * C + c] =
                full_history[static_cast<std::size_t>(j) * CF + global_channel(rank, c)];
        }
    }
    failures += verify_exact((label + " conv history").c_str(), history, want_history);
    return failures;
}

int test_gdn() {
    constexpr int kVerify = 4;
    const Logical logical = make_logical();
    set_device(0);
    GdnDevice full(kFull, -1, logical, kVerify);
    std::array<std::unique_ptr<GdnDevice>, 2> shards;
    for (int rank = 0; rank < 2; ++rank) {
        set_device(rank);
        shards[rank] = std::make_unique<GdnDevice>(kShard, rank, logical, kVerify);
    }
    int failures = 0;
    // Round 1: ordinary decode, slots {0,2} -> {1,3}.
    {
        const auto in = make_inputs(1, 900);
        const std::vector<int> valid{1, 1}, src{0, 2}, dst{1, 3};
        set_device(0);
        Round rf(full, in, 1, valid, src, dst);
        decode_round(full, rf, nullptr);
        cuda_check(cudaDeviceSynchronize(), "full decode");
        for (int rank = 0; rank < 2; ++rank) {
            set_device(rank);
            Round rs(*shards[rank], in, 1, valid, src, dst);
            decode_round(*shards[rank], rs, nullptr);
            cuda_check(cudaDeviceSynchronize(), "shard decode");
            const std::string label = "gdn tp2 decode rank " + std::to_string(rank);
            failures += compare_round(full, rf, *shards[rank], rs, rank, label);
            for (const int slot : dst) {
                failures += compare_slot(full, *shards[rank], rank, slot,
                                         label + " slot " + std::to_string(slot));
            }
        }
    }
    // Round 2: MTP verify W=4 from slots {1,3}, valid {4,3}, then fold commits {3,2} into {0,2}.
    {
        const auto in = make_inputs(kVerify, 950);
        const std::vector<int> valid{4, 3}, src{1, 3}, dst{0, 2};
        const std::vector<ops::GdnReplayFoldRow> commit{{1, 0, 3}, {3, 2, 2}};
        set_device(0);
        Round rf(full, in, kVerify, valid, src, dst);
        verify_round(full, rf, nullptr);
        full.fold->execute(commit, nullptr);
        cuda_check(cudaDeviceSynchronize(), "full verify");
        for (int rank = 0; rank < 2; ++rank) {
            set_device(rank);
            Round rs(*shards[rank], in, kVerify, valid, src, dst);
            verify_round(*shards[rank], rs, nullptr);
            shards[rank]->fold->execute(commit, nullptr);
            cuda_check(cudaDeviceSynchronize(), "shard verify");
            const std::string label = "gdn tp2 verify W=4 rank " + std::to_string(rank);
            failures += compare_round(full, rf, *shards[rank], rs, rank, label);
            for (const int slot : {0, 1, 2, 3}) {
                failures += compare_slot(full, *shards[rank], rank, slot,
                                         label + " slot " + std::to_string(slot));
            }
        }
    }
    return failures;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        int devices = 0;
        cuda_check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
        if (devices < 2) {
            std::cout << "SKIP: the tp2 mixer parity needs two CUDA devices\n";
            return 77;
        }
        int failures = 0;
        failures += test_gdn();
        failures += test_qsa();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " flash_next_tp2_mixers\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "flash_next_tp2_mixers: " << error.what() << '\n';
        return 1;
    }
}
