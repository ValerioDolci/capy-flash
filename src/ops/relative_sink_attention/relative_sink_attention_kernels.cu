#include "ops/relative_sink_attention/launch.h"

#include "core/device.h"
#include "ops/common/pdl_launch.cuh"
#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/launcher/kernel_attr_once.h"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

constexpr int kWarp          = 32;
constexpr int kMaxHeadDim    = 256;
constexpr int kMaxPerLane    = kMaxHeadDim / kWarp;
constexpr int kPrologueWarps = 8;
constexpr int kKeyWarps      = 4;

__device__ __forceinline__ float load(const __nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float round_bf16(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

// The output element at `at` from its attention value o: bf16(o), or with the gate
// bf16(bf16(o) * sigmoid(gate)) exactly as ops::sigmoid_mul computes it.
__device__ __forceinline__ __nv_bfloat16 gated_value(float o, bool gated, float gate) {
    if (gated) o = round_bf16(o) * sigmoid(gate);
    return __float2bfloat16_rn(o);
}

__device__ __forceinline__ __nv_bfloat16 output_value(const RelativeSinkOperands& op,
                                                      std::int64_t at, float o) {
    return gated_value(o, op.gate != nullptr, op.gate != nullptr ? load(op.gate[at]) : 0.0f);
}

// Butterfly sum: every lane adds the same pairs, so all lanes hold bit-identical results.
__device__ __forceinline__ float warp_sum(float x) {
    for (int offset = kWarp / 2; offset > 0; offset >>= 1)
        x += __shfl_xor_sync(0xffffffffU, x, offset);
    return x;
}

struct RingAccess {
    RelativeSinkRing ring;

    __device__ int first_visible(int position) const { return max(0, position - ring.window + 1); }

    __device__ bool retains(int position, int end) const { return position >= end - ring.capacity; }

    __device__ __nv_bfloat16* key(int row, int head, int position) const {
        const std::int64_t slot = position % ring.capacity;
        return ring.k + slot * ring.k_stride[0] + head * ring.k_stride[1] + row * ring.k_stride[2];
    }

    __device__ __half* value(int row, int head, int position) const {
        const std::int64_t slot = position % ring.capacity;
        return ring.v + slot * ring.v_stride[0] + head * ring.v_stride[1] + row * ring.v_stride[2];
    }
};

struct PageAccess {
    RelativeSinkPages pages;

    __device__ int first_visible(int) const { return 0; }

    __device__ bool retains(int, int) const { return true; }

    __device__ std::int64_t page(int row, int position) const {
        return pages
            .block_table[(position >> 6) * pages.table_stride[0] + row * pages.table_stride[1]];
    }

    __device__ __nv_bfloat16* key(int row, int head, int position) const {
        return pages.k + (position & 63) * pages.k_stride[0] + head * pages.k_stride[1] +
               page(row, position) * pages.k_stride[2];
    }

    __device__ __half* value(int row, int head, int position) const {
        return pages.v + (position & 63) * pages.v_stride[0] + head * pages.v_stride[1] +
               page(row, position) * pages.v_stride[2];
    }
};

// tau(p) of the contract.
__device__ __forceinline__ float position_scale(const RelativeSinkAttentionParams& params, int p) {
    return __fadd_rn(
        1.0f,
        __fmul_rn(params.log_scale_alpha,
                  logf(fmaxf(__fdiv_rn(static_cast<float>(p + 1), params.log_scale_floor), 1.0f))));
}

// One head vector of the prologue: q (kind 0), r (1), k (2) or v (3) of one head and token.
struct PrologueItem {
    int kind;
    int n;               // elements: dr for r, D otherwise
    std::int64_t offset; // of the vector in its input and in its prepared copy
    const __nv_bfloat16* source;
    const __nv_bfloat16* scale; // learned scale of q and r, nullptr for k and v
};

__device__ __forceinline__ PrologueItem prologue_item(const RelativeSinkOperands& op, int kind,
                                                      int head, int token) {
    const int D  = op.head_dim;
    const int dr = op.params.relative_dim;
    PrologueItem item;
    item.kind = kind;
    item.n    = kind == 1 ? dr : D;
    item.offset =
        static_cast<std::int64_t>(item.n) *
        (head + static_cast<std::int64_t>(kind <= 1 ? op.query_heads : op.kv_heads) * token);
    item.source = kind == 0   ? op.q + item.offset
                  : kind == 1 ? op.r + item.offset
                  : kind == 2 ? op.k + item.offset
                              : op.v + item.offset;
    item.scale  = kind == 0   ? op.q_scale + static_cast<std::int64_t>(D) * head
                  : kind == 1 ? op.r_scale + static_cast<std::int64_t>(dr) * head
                              : nullptr;
    return item;
}

// The lane's elements d = lane + 32 i of the item, times its learned scale; zero beyond n. Only
// loads, so a warp can request several items before using any.
__device__ __forceinline__ void prologue_load(const PrologueItem& item, int lane,
                                              float (&x)[kMaxPerLane]) {
#pragma unroll
    for (int i = 0; i < kMaxPerLane; ++i) {
        const int d = lane + kWarp * i;
        x[i]        = 0.0f;
        if (d < item.n) {
            float value = load(item.source[d]);
            if (item.scale != nullptr) value = __fmul_rn(value, load(item.scale[d]));
            x[i] = value;
        }
    }
}

// Turns the loaded elements into the prepared ones: the BF16 values of q', r', k', and for v the
// BF16 value that its FP16 copy holds.
__device__ __forceinline__ void prologue_finish(const RelativeSinkAttentionParams& params,
                                                const PrologueItem& item, float tau, int lane,
                                                float (&x)[kMaxPerLane]) {
    float squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kMaxPerLane; ++i)
        if (lane + kWarp * i < item.n) squares += x[i] * x[i];
    squares = warp_sum(squares);
    if (item.kind == 3) {
        const float inverse =
            rsqrtf(__fdiv_rn(squares, static_cast<float>(item.n)) + params.value_epsilon);
#pragma unroll
        for (int i = 0; i < kMaxPerLane; ++i) x[i] = round_bf16(x[i] * inverse);
        return;
    }
    const float threshold = item.kind == 1 ? params.relative_clip : params.qk_clip;
    const float factor    = fminf(__fdiv_rn(threshold, sqrtf(squares) + params.clip_epsilon), 1.0f);
#pragma unroll
    for (int i = 0; i < kMaxPerLane; ++i) {
        float y = round_bf16(x[i] * factor);
        if (item.kind <= 1) y = round_bf16(y * tau);
        x[i] = y;
    }
}

// Blocks (token, slice); each warp prepares one whole head vector of q, r, k or v.
__global__ void prologue_kernel(RelativeSinkOperands op) {
    pdl_enter();
    const int token = blockIdx.x;
    const int lane  = threadIdx.x % kWarp;
    const int warp  = threadIdx.x / kWarp;
    const int Hq    = op.query_heads;
    const int Hkv   = op.kv_heads;
    const float tau = position_scale(op.params, op.positions[token]);
    const int items = 2 * Hq + 2 * Hkv;
    for (int item = warp + kPrologueWarps * blockIdx.y; item < items;
         item += kPrologueWarps * gridDim.y) {
        const int kind = item < Hq ? 0 : item < 2 * Hq ? 1 : item < 2 * Hq + Hkv ? 2 : 3;
        const int head = kind == 0   ? item
                         : kind == 1 ? item - Hq
                         : kind == 2 ? item - 2 * Hq
                                     : item - 2 * Hq - Hkv;
        const PrologueItem prepared = prologue_item(op, kind, head, token);
        float x[kMaxPerLane];
        prologue_load(prepared, lane, x);
        prologue_finish(op.params, prepared, tau, lane, x);
#pragma unroll
        for (int i = 0; i < kMaxPerLane; ++i) {
            const int d = lane + kWarp * i;
            if (d >= prepared.n) continue;
            const std::int64_t at = prepared.offset + d;
            if (kind == 3)
                op.v_prepared[at] = __float2half_rn(x[i]);
            else
                (kind == 0   ? op.q_prepared
                 : kind == 1 ? op.r_prepared
                             : op.k_prepared)[at] = __float2bfloat16_rn(x[i]);
        }
    }
}

// Canonical decomposition of one query's visible keys [lo, p]: chunks of C keys starting at lo,
// C a multiple of 128 chosen from the visible count alone so that there are at most
// kRelativeSinkMaxChunks chunks. Inside a chunk, key begin + i belongs to warp (i / 32) % 4 and
// lane i % 32, in steps of 128 keys. Every route evaluates the same chunks with the same device
// functions and explicit FP32 operations, so a query's result depends only on the query, never
// on the route, the call's T or how a prefill is split.
constexpr int kChunkUnit    = kKeyWarps * kWarp;
constexpr int kKeysInFlight = 8; // value rows one warp loads before using them (D > 128)
// Decode route: blocks per cluster; block b owns chunk b, so a window of at most kDecodeMaxKeys.
constexpr int kDecodeBlocks  = 8;
constexpr int kDecodeMaxKeys = kDecodeBlocks * kChunkUnit;

struct Query {
    int token; // t + T * b
    int b;
    int p;      // absolute position
    int L;      // first position of the call's block in this row
    int lo;     // first visible position
    int chunk;  // keys per chunk
    int chunks; // number of chunks
    int row;    // cache lane or table row
    int g;      // KV head
    int h0;     // first query head of this block
    int heads;  // query heads of this block (<= HB)
};

template <int HB, class Cache>
__device__ Query make_query(const RelativeSinkOperands& op, const Cache& cache, int token,
                            int slice) {
    Query q;
    const int group   = op.query_heads / op.kv_heads;
    const int slices  = (group + HB - 1) / HB;
    q.token           = token;
    q.b               = token / op.tokens;
    q.p               = op.positions[token];
    q.L               = op.positions[op.tokens * q.b];
    q.lo              = cache.first_visible(q.p);
    const int visible = q.p - q.lo + 1;
    const int span    = kChunkUnit * kRelativeSinkMaxChunks;
    q.chunk           = kChunkUnit * ((visible + span - 1) / span);
    q.chunks          = (visible + q.chunk - 1) / q.chunk;
    q.row             = op.rows[q.b];
    q.g               = slice / slices;
    const int first   = (slice % slices) * HB;
    q.h0              = q.g * group + first;
    q.heads           = min(HB, group - first);
    return q;
}

// The block's prepared queries and relative states, read by every lane (broadcast).
template <int HB, int E>
struct QueryState {
    float q[HB][kWarp * E];
    float rel[HB][kWarp];
};

template <int HB, int E>
__device__ void load_query_state(const RelativeSinkOperands& op, const Query& query,
                                 QueryState<HB, E>& state) {
    const int D  = op.head_dim;
    const int dr = op.params.relative_dim;
    for (int i = threadIdx.x; i < HB * kWarp * E; i += blockDim.x) {
        const int hh = i / (kWarp * E), d = i % (kWarp * E);
        const int h    = query.h0 + min(hh, query.heads - 1);
        state.q[hh][d] = d < D ? load(op.q_prepared[static_cast<std::int64_t>(D) *
                                                        (h + op.query_heads * query.token) +
                                                    d])
                               : 0.0f;
    }
    for (int i = threadIdx.x; i < HB * kWarp; i += blockDim.x) {
        const int hh = i / kWarp, j = i % kWarp;
        const int h      = query.h0 + min(hh, query.heads - 1);
        state.rel[hh][j] = j < dr ? load(op.r_prepared[static_cast<std::int64_t>(dr) *
                                                           (h + op.query_heads * query.token) +
                                                       j])
                                  : 0.0f;
    }
    __syncthreads();
}

template <class Cache>
__device__ const __nv_bfloat16* key_row(const RelativeSinkOperands& op, const Cache& cache,
                                        const Query& query, int s) {
    return s < query.L ? cache.key(query.row, query.g, s)
                       : op.k_prepared + static_cast<std::int64_t>(op.head_dim) *
                                             (query.g + static_cast<std::int64_t>(op.kv_heads) *
                                                            (s - query.L + op.tokens * query.b));
}

template <class Cache>
__device__ const __half* value_row(const RelativeSinkOperands& op, const Cache& cache,
                                   const Query& query, int s) {
    return s < query.L ? cache.value(query.row, query.g, s)
                       : op.v_prepared + static_cast<std::int64_t>(op.head_dim) *
                                             (query.g + static_cast<std::int64_t>(op.kv_heads) *
                                                            (s - query.L + op.tokens * query.b));
}

// dot[hh] += q'[hh][d0 + d] k[d0 + d] in element order over the vectors of one K row held in raw
// (8 elements each), up to D.
template <int HB, int E>
__device__ __forceinline__ void accumulate_dot(const QueryState<HB, E>& state,
                                               const uint4 (&raw)[4 * E], int d0, int D,
                                               float (&dot)[HB]) {
#pragma unroll
    for (int u = 0; u < 4 * E; ++u) {
        if (d0 + u * 8 >= D) break;
        const auto* pair = reinterpret_cast<const __nv_bfloat162*>(&raw[u]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 k = __bfloat1622float2(pair[j]);
            const int d    = d0 + u * 8 + 2 * j;
#pragma unroll
            for (int hh = 0; hh < HB; ++hh) {
                dot[hh] = __fmaf_rn(state.q[hh][d], k.x, dot[hh]);
                dot[hh] = __fmaf_rn(state.q[hh][d + 1], k.y, dot[hh]);
            }
        }
    }
}

// The bias column of one distance (dr elements of rel_proj, zero beyond).
__device__ __forceinline__ void load_column(const RelativeSinkOperands& op, int distance,
                                            float (&column)[kWarp]) {
#pragma unroll
    for (int j = 0; j < kWarp; ++j)
        column[j] =
            j < op.params.relative_dim
                ? load(op.rel_proj[distance +
                                   static_cast<std::int64_t>(op.params.relative_extent) * j])
                : 0.0f;
}

// Logits of one key for every head of the block from its dot products and, at a distance below
// R, its bias column.
template <int HB, int E>
__device__ __forceinline__ void finish_logits(const RelativeSinkOperands& op,
                                              const QueryState<HB, E>& state,
                                              const float (&dot)[HB], int distance,
                                              const float (&column)[kWarp], float (&logit)[HB]) {
    const bool biased = distance < op.params.relative_extent;
    float bias[HB]    = {};
    if (biased) {
#pragma unroll
        for (int j = 0; j < kWarp; ++j)
            if (j < op.params.relative_dim)
#pragma unroll
                for (int hh = 0; hh < HB; ++hh)
                    bias[hh] = __fmaf_rn(state.rel[hh][j], column[j], bias[hh]);
    }
#pragma unroll
    for (int hh = 0; hh < HB; ++hh) {
        float score = round_bf16(__fmul_rn(round_bf16(dot[hh]), op.params.scale));
        if (biased) score = round_bf16(__fadd_rn(score, round_bf16(bias[hh])));
        logit[hh] = score;
    }
}

// Logits of key s for every head of the block, computed by one lane alone.
template <int HB, int E, class Cache>
__device__ void lane_logits(const RelativeSinkOperands& op, const Cache& cache, const Query& query,
                            const QueryState<HB, E>& state, int s, float (&logit)[HB]) {
    const auto* key = reinterpret_cast<const uint4*>(key_row(op, cache, query, s));
    float dot[HB]   = {};
    // The whole row (up to 32 * E elements) is requested before any of it is used.
    constexpr int kVectors = 4 * E;
    for (int d0 = 0; d0 < op.head_dim; d0 += kVectors * 8) {
        uint4 raw[kVectors];
#pragma unroll
        for (int u = 0; u < kVectors; ++u)
            if (d0 + u * 8 < op.head_dim) raw[u] = key[d0 / 8 + u];
        accumulate_dot<HB, E>(state, raw, d0, op.head_dim, dot);
    }
    const int distance = query.p - s;
    // Every column element is requested before any is used.
    float column[kWarp] = {};
    if (distance < op.params.relative_extent) load_column(op, distance, column);
    finish_logits<HB, E>(op, state, dot, distance, column, logit);
}

// One key's logit x in a lane's running (max, sum of exponentials).
__device__ __forceinline__ void online_update(float& m, float& z, float x) {
    if (x > m) {
        z = __fadd_rn(__fmul_rn(z, expf(m - x)), 1.0f);
        m = x;
    } else {
        z = __fadd_rn(z, expf(x - m));
    }
}

// (max, sum) merge; symmetric in its two arguments, so butterfly partners agree bit for bit.
__device__ __forceinline__ void merge_stats(float& m, float& z, float other_m, float other_z) {
    const float top = fmaxf(m, other_m);
    if (top == -INFINITY) return;
    z = __fadd_rn(__fmul_rn(z, expf(m - top)), __fmul_rn(other_z, expf(other_m - top)));
    m = top;
}

template <int HB>
struct ChunkStats {
    float m[HB];
    float z[HB];
};

// Max and sum of exponentials of one chunk for every head; every thread receives the result.
template <int HB, int E, class Cache>
__device__ ChunkStats<HB> chunk_stats(const RelativeSinkOperands& op, const Cache& cache,
                                      const Query& query, const QueryState<HB, E>& state, int c) {
    __shared__ float warp_m[kKeyWarps][HB];
    __shared__ float warp_z[kKeyWarps][HB];
    const int lane  = threadIdx.x % kWarp;
    const int warp  = threadIdx.x / kWarp;
    const int begin = query.lo + c * query.chunk;
    const int end   = min(begin + query.chunk, query.p + 1);
    float m[HB], z[HB];
#pragma unroll
    for (int hh = 0; hh < HB; ++hh) {
        m[hh] = -INFINITY;
        z[hh] = 0.0f;
    }
    for (int s = begin + threadIdx.x; s < end; s += kChunkUnit) {
        float x[HB];
        lane_logits<HB, E>(op, cache, query, state, s, x);
#pragma unroll
        for (int hh = 0; hh < HB; ++hh) online_update(m[hh], z[hh], x[hh]);
    }
#pragma unroll
    for (int hh = 0; hh < HB; ++hh)
        for (int offset = kWarp / 2; offset > 0; offset >>= 1)
            merge_stats(m[hh], z[hh], __shfl_xor_sync(0xffffffffU, m[hh], offset),
                        __shfl_xor_sync(0xffffffffU, z[hh], offset));
    if (lane == 0) {
#pragma unroll
        for (int hh = 0; hh < HB; ++hh) {
            warp_m[warp][hh] = m[hh];
            warp_z[warp][hh] = z[hh];
        }
    }
    __syncthreads();
    ChunkStats<HB> stats;
#pragma unroll
    for (int hh = 0; hh < HB; ++hh) {
        float chunk_m = warp_m[0][hh], chunk_z = warp_z[0][hh];
        for (int w = 1; w < kKeyWarps; ++w)
            merge_stats(chunk_m, chunk_z, warp_m[w][hh], warp_z[w][hh]);
        stats.m[hh] = chunk_m;
        stats.z[hh] = chunk_z;
    }
    __syncthreads();
    return stats;
}

// Global max and normalizer of one head from its chunk statistics and its sink, in chunk order.
__device__ void combine_stats(float sink, const float* m, const float* z, int chunks, int stride,
                              float& max_out, float& norm_out) {
    float top = sink;
    for (int c = 0; c < chunks; ++c) top = fmaxf(top, m[c * stride]);
    float norm = expf(sink - top);
    for (int c = 0; c < chunks; ++c)
        if (z[c * stride] > 0.0f)
            norm = __fadd_rn(norm, __fmul_rn(z[c * stride], expf(m[c * stride] - top)));
    max_out  = top;
    norm_out = norm;
}

// sum_s bf16(p_s) v_s over one chunk for every head of the block, written to out[hh * D + d]
// (warp partials added in warp order). Each lane computes its key's probabilities; the warp then
// walks its 32 keys in lane order with lanes spread over the value dimensions.
template <int HB, int E, class Cache>
__device__ void chunk_values(const RelativeSinkOperands& op, const Cache& cache, const Query& query,
                             const QueryState<HB, E>& state, int c, const float (&top)[HB],
                             const float (&norm)[HB], float* out) {
    __shared__ float partial[kKeyWarps][HB][kWarp * E];
    // D <= 128: each lane copies its key's value row to shared memory asynchronously, together
    // with its key row load, so a step costs one memory round trip.
    constexpr bool kStaged = E <= 4;
    __shared__ __align__(16) __half staged[kStaged ? kKeyWarps : 1][kWarp][kStaged ? kWarp * E : 8];
    const int lane   = threadIdx.x % kWarp;
    const int warp   = threadIdx.x / kWarp;
    const int D      = op.head_dim;
    const int begin  = query.lo + c * query.chunk;
    const int end    = min(begin + query.chunk, query.p + 1);
    float acc[HB][E] = {};
    for (int base = begin + warp * kWarp; base < end; base += kChunkUnit) {
        if constexpr (kStaged) {
            if (base + lane < end) {
                const __half* value = value_row(op, cache, query, base + lane);
                for (int d = 0; d < D; d += 8)
                    __pipeline_memcpy_async(&staged[warp][lane][d], value + d, 16);
            }
            __pipeline_commit();
        }
        float probability[HB] = {};
        if (base + lane < end) {
            float x[HB];
            lane_logits<HB, E>(op, cache, query, state, base + lane, x);
#pragma unroll
            for (int hh = 0; hh < HB; ++hh)
                probability[hh] = round_bf16(__fdiv_rn(expf(x[hh] - top[hh]), norm[hh]));
        }
        const int count = min(kWarp, end - base);
        if constexpr (kStaged) {
            __pipeline_wait_prior(0);
            __syncwarp();
            for (int source = 0; source < kWarp; ++source) {
#pragma unroll
                for (int hh = 0; hh < HB; ++hh) {
                    const float pk = __shfl_sync(0xffffffffU, probability[hh], source);
                    if (source < count)
#pragma unroll
                        for (int e = 0; e < E; ++e) {
                            const int d = lane + kWarp * e;
                            if (d < D)
                                acc[hh][e] = __fmaf_rn(pk, __half2float(staged[warp][source][d]),
                                                       acc[hh][e]);
                        }
                }
            }
            __syncwarp();
            continue;
        }
        for (int first = 0; first < count; first += kKeysInFlight) {
            float v[kKeysInFlight][E];
#pragma unroll
            for (int u = 0; u < kKeysInFlight; ++u) {
                const __half* value = value_row(op, cache, query, base + min(first + u, count - 1));
#pragma unroll
                for (int e = 0; e < E; ++e) {
                    const int d = lane + kWarp * e;
                    v[u][e]     = d < D ? __half2float(value[d]) : 0.0f;
                }
            }
#pragma unroll
            for (int u = 0; u < kKeysInFlight; ++u) {
                const int source = first + u;
#pragma unroll
                for (int hh = 0; hh < HB; ++hh) {
                    const float pk =
                        __shfl_sync(0xffffffffU, probability[hh], source & (kWarp - 1));
                    if (source < count)
#pragma unroll
                        for (int e = 0; e < E; ++e) acc[hh][e] = __fmaf_rn(pk, v[u][e], acc[hh][e]);
                }
            }
        }
    }
#pragma unroll
    for (int hh = 0; hh < HB; ++hh)
#pragma unroll
        for (int e = 0; e < E; ++e) {
            const int d = lane + kWarp * e;
            if (d < D) partial[warp][hh][d] = acc[hh][e];
        }
    __syncthreads();
    for (int i = threadIdx.x; i < query.heads * D; i += blockDim.x) {
        const int hh = i / D, d = i % D;
        float sum = partial[0][hh][d];
        for (int w = 1; w < kKeyWarps; ++w) sum = __fadd_rn(sum, partial[w][hh][d]);
        out[i] = sum;
    }
    __syncthreads();
}

__device__ std::int64_t stats_index(const RelativeSinkOperands& op, int token, int h, int c) {
    return (static_cast<std::int64_t>(token) * op.query_heads + h) * kRelativeSinkMaxChunks + c;
}

// Threads of a wide block: warp (g, hh) = (w % 4, w / 4) takes head hh of the chunk's keys whose
// index i satisfies (i / 32) % 4 = g, lane i % 32 (the canonical decomposition), so a thread runs
// one head's share of a key instead of every head's.
template <int HB>
constexpr int kWideThreads = kKeyWarps * HB * kWarp;

// lane_logits for one head hh of the block: the same operations in the same order.
template <int HB, int E, class Cache>
__device__ float head_logit(const RelativeSinkOperands& op, const Cache& cache, const Query& query,
                            const QueryState<HB, E>& state, int hh, int s) {
    const auto* key        = reinterpret_cast<const uint4*>(key_row(op, cache, query, s));
    const float* q         = state.q[hh];
    float dot              = 0.0f;
    constexpr int kVectors = 4 * E;
    for (int d0 = 0; d0 < op.head_dim; d0 += kVectors * 8) {
        uint4 raw[kVectors];
#pragma unroll
        for (int u = 0; u < kVectors; ++u)
            if (d0 + u * 8 < op.head_dim) raw[u] = key[d0 / 8 + u];
#pragma unroll
        for (int u = 0; u < kVectors; ++u) {
            if (d0 + u * 8 >= op.head_dim) break;
            const auto* pair = reinterpret_cast<const __nv_bfloat162*>(&raw[u]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 k = __bfloat1622float2(pair[j]);
                const int d    = d0 + u * 8 + 2 * j;
                dot            = __fmaf_rn(q[d], k.x, dot);
                dot            = __fmaf_rn(q[d + 1], k.y, dot);
            }
        }
    }
    const int distance = query.p - s;
    const bool biased  = distance < op.params.relative_extent;
    float bias         = 0.0f;
    if (biased) {
        float column[kWarp];
        load_column(op, distance, column);
#pragma unroll
        for (int j = 0; j < kWarp; ++j)
            if (j < op.params.relative_dim) bias = __fmaf_rn(state.rel[hh][j], column[j], bias);
    }
    float score = round_bf16(__fmul_rn(round_bf16(dot), op.params.scale));
    if (biased) score = round_bf16(__fadd_rn(score, round_bf16(bias)));
    return score;
}

// One chunk's statistics for every head of a wide block, written to the workspace.
template <int HB, int E, class Cache>
__device__ void chunk_stats_wide(const RelativeSinkOperands& op, const Cache& cache,
                                 const Query& query, const QueryState<HB, E>& state, int c,
                                 int lane, int group, int hh, float (&warp_m)[kKeyWarps][HB],
                                 float (&warp_z)[kKeyWarps][HB]) {
    const int begin = query.lo + c * query.chunk;
    const int end   = min(begin + query.chunk, query.p + 1);
    float m = -INFINITY, z = 0.0f;
    for (int s = begin + group * kWarp + lane; s < end; s += kChunkUnit)
        online_update(m, z, head_logit<HB, E>(op, cache, query, state, hh, s));
    for (int offset = kWarp / 2; offset > 0; offset >>= 1)
        merge_stats(m, z, __shfl_xor_sync(0xffffffffU, m, offset),
                    __shfl_xor_sync(0xffffffffU, z, offset));
    if (lane == 0) {
        warp_m[group][hh] = m;
        warp_z[group][hh] = z;
    }
    __syncthreads();
    if (threadIdx.x < query.heads) {
        const int h   = threadIdx.x;
        float chunk_m = warp_m[0][h], chunk_z = warp_z[0][h];
        for (int w = 1; w < kKeyWarps; ++w)
            merge_stats(chunk_m, chunk_z, warp_m[w][h], warp_z[w][h]);
        const std::int64_t at = stats_index(op, query.token, query.h0 + h, c);
        op.chunk_max[at]      = chunk_m;
        op.chunk_sum[at]      = chunk_z;
    }
    __syncthreads();
}

// Split route, stage 1: chunk statistics, one wide block per (chunk slot, KV head slice, token);
// slot x takes the chunks x, x + slots, ...
template <int HB, int E, class Cache>
__global__ void __launch_bounds__(kWideThreads<HB>)
    split_stats_kernel(RelativeSinkOperands op, Cache cache) {
    pdl_enter();
    const Query query = make_query<HB>(op, cache, blockIdx.z, blockIdx.y);
    if (static_cast<int>(blockIdx.x) >= query.chunks) return;
    __shared__ QueryState<HB, E> state;
    __shared__ float warp_m[kKeyWarps][HB];
    __shared__ float warp_z[kKeyWarps][HB];
    load_query_state(op, query, state);
    const int lane  = threadIdx.x % kWarp;
    const int warp  = threadIdx.x / kWarp;
    const int group = warp % kKeyWarps;
    const int hh    = warp / kKeyWarps;
    for (int c = blockIdx.x; c < query.chunks; c += gridDim.x)
        chunk_stats_wide<HB, E>(op, cache, query, state, c, lane, group, hh, warp_m, warp_z);
}

// Split route, stage 2: chunk value sums under the global normalization, one wide block per
// (chunk slot, KV head slice, token) as in stage 1. Steps of 128 keys; D <= 128: the step's value
// rows are copied to shared memory by the first head's warps while every warp computes its
// probabilities.
template <int HB, int E, class Cache>
__global__ void __launch_bounds__(kWideThreads<HB>)
    split_values_kernel(RelativeSinkOperands op, Cache cache) {
    pdl_enter();
    const Query query = make_query<HB>(op, cache, blockIdx.z, blockIdx.y);
    if (static_cast<int>(blockIdx.x) >= query.chunks) return;
    constexpr bool kStaged = E <= 4;
    __shared__ QueryState<HB, E> state;
    __shared__ __align__(16) __half staged[kStaged ? kChunkUnit : 1][kStaged ? kWarp * E : 8];
    __shared__ float partial[kKeyWarps][HB][kWarp * E];
    load_query_state(op, query, state);
    const int lane  = threadIdx.x % kWarp;
    const int warp  = threadIdx.x / kWarp;
    const int group = warp % kKeyWarps;
    const int hh    = warp / kKeyWarps;
    const int D     = op.head_dim;
    const int h     = query.h0 + min(hh, query.heads - 1);
    float top, norm;
    {
        const std::int64_t first = stats_index(op, query.token, h, 0);
        combine_stats(load(op.sink[h + op.query_heads * query.token]), op.chunk_max + first,
                      op.chunk_sum + first, query.chunks, 1, top, norm);
    }
    for (int c = blockIdx.x; c < query.chunks; c += gridDim.x) {
        const int begin = query.lo + c * query.chunk;
        const int end   = min(begin + query.chunk, query.p + 1);
        const int steps = (end - begin + kChunkUnit - 1) / kChunkUnit;
        float acc[E]    = {};
        for (int step = 0; step < steps; ++step) {
            const int base  = begin + step * kChunkUnit + group * kWarp;
            const int s     = base + lane;
            const int count = max(0, min(kWarp, end - base));
            if constexpr (kStaged) {
                if (hh == 0 && s < end) {
                    const __half* value = value_row(op, cache, query, s);
                    for (int d = 0; d < D; d += 8)
                        __pipeline_memcpy_async(&staged[group * kWarp + lane][d], value + d, 16);
                }
                __pipeline_commit();
            }
            const float probability =
                s < end ? round_bf16(__fdiv_rn(
                              expf(head_logit<HB, E>(op, cache, query, state, hh, s) - top), norm))
                        : 0.0f;
            if constexpr (kStaged) {
                __pipeline_wait_prior(0);
                __syncthreads();
                // Lane l takes the elements El .. El + E - 1.
                const int d0 = E * lane;
#pragma unroll
                for (int source = 0; source < kWarp; ++source) {
                    const float pk = __shfl_sync(0xffffffffU, probability, source);
                    if (source < count && d0 < D) {
                        const __half* row = &staged[group * kWarp + source][d0];
#pragma unroll
                        for (int e = 0; e < E; e += 2) {
                            const float2 value =
                                __half22float2(*reinterpret_cast<const __half2*>(row + e));
                            acc[e]     = __fmaf_rn(pk, value.x, acc[e]);
                            acc[e + 1] = __fmaf_rn(pk, value.y, acc[e + 1]);
                        }
                    }
                }
                __syncthreads();
            } else {
                for (int first = 0; first < count; first += kKeysInFlight) {
                    float v[kKeysInFlight][E];
#pragma unroll
                    for (int u = 0; u < kKeysInFlight; ++u) {
                        const __half* value =
                            value_row(op, cache, query, base + min(first + u, count - 1));
#pragma unroll
                        for (int e = 0; e < E; ++e) {
                            const int d = lane + kWarp * e;
                            v[u][e]     = d < D ? __half2float(value[d]) : 0.0f;
                        }
                    }
#pragma unroll
                    for (int u = 0; u < kKeysInFlight; ++u) {
                        const int source = first + u;
                        const float pk =
                            __shfl_sync(0xffffffffU, probability, source & (kWarp - 1));
                        if (source < count)
#pragma unroll
                            for (int e = 0; e < E; ++e) acc[e] = __fmaf_rn(pk, v[u][e], acc[e]);
                    }
                }
            }
        }
#pragma unroll
        for (int e = 0; e < E; ++e) {
            const int d = kStaged ? E * lane + e : lane + kWarp * e;
            if (d < D) partial[group][hh][d] = acc[e];
        }
        __syncthreads();
        for (int i = threadIdx.x; i < query.heads * D; i += blockDim.x) {
            const int oh = i / D, d = i % D;
            float sum = partial[0][oh][d];
            for (int w = 1; w < kKeyWarps; ++w) sum = __fadd_rn(sum, partial[w][oh][d]);
            op.chunk_out[stats_index(op, query.token, query.h0 + oh, c) * D + d] = sum;
        }
        __syncthreads();
    }
}

// Paged verification calls whose row's last token sees more than kColumnsMinKeys keys go to
// split_columns_kernel (shared K/V rows for all tokens); the others to split_online_kernel.
constexpr int kColumnsMaxTokens = 8;
constexpr int kColumnsMinKeys   = 4096;

// Whether the columns kernel takes row b of a paged call (see split_columns_kernel).
__device__ __forceinline__ bool columns_row(const RelativeSinkOperands& op, int b) {
    return op.tokens >= 2 && op.tokens <= kColumnsMaxTokens && op.head_dim <= kWarp * 4 &&
           op.positions[op.tokens * b + op.tokens - 1] + 1 > kColumnsMinKeys;
}

// Paged split route (see launch_routes): one pass over the keys. Per chunk and head, each warp
// keeps the running max m of its keys' logits, the sum z of exp(logit - m) and the value sum
// sum_s exp(logit_s - m) v_s, rescaled by exp(m_old - m_new) when a step raises m; the chunk
// merges its warps in warp order. Probabilities are not rounded to BF16 (the finish divides the
// combined value sum by the normalizer), so K is read once. A query with a single chunk (at most
// 128 visible keys: its block holds every key) keeps the two-pass arithmetic instead, BF16
// probabilities included, with the split route's bits. Grid and steps as the two-pass stages.
template <int HB, int E, class Cache>
__global__ void __launch_bounds__(kWideThreads<HB>)
    split_online_kernel(RelativeSinkOperands op, Cache cache) {
    pdl_enter();
    const Query query = make_query<HB>(op, cache, blockIdx.z, blockIdx.y);
    if (static_cast<int>(blockIdx.x) >= query.chunks || (E == 4 && columns_row(op, query.b)))
        return;
    constexpr bool kStaged = E <= 4;
    __shared__ QueryState<HB, E> state;
    __shared__ __align__(16) __half staged[kStaged ? kChunkUnit : 1][kStaged ? kWarp * E : 8];
    __shared__ float partial[kKeyWarps][HB][kWarp * E];
    __shared__ float warp_m[kKeyWarps][HB];
    __shared__ float warp_z[kKeyWarps][HB];
    load_query_state(op, query, state);
    const int lane   = threadIdx.x % kWarp;
    const int warp   = threadIdx.x / kWarp;
    const int group  = warp % kKeyWarps;
    const int hh     = warp / kKeyWarps;
    const int D      = op.head_dim;
    const bool exact = query.chunks == 1;
    for (int c = blockIdx.x; c < query.chunks; c += gridDim.x) {
        const int begin = query.lo + c * query.chunk;
        const int end   = min(begin + query.chunk, query.p + 1);
        const int steps = (end - begin + kChunkUnit - 1) / kChunkUnit;
        float m = -INFINITY, z = 0.0f, acc[E] = {};
        for (int step = 0; step < steps; ++step) {
            const int base  = begin + step * kChunkUnit + group * kWarp;
            const int s     = base + lane;
            const int count = max(0, min(kWarp, end - base));
            if constexpr (kStaged) {
                if (hh == 0 && s < end) {
                    const __half* value = value_row(op, cache, query, s);
                    for (int d = 0; d < D; d += 8)
                        __pipeline_memcpy_async(&staged[group * kWarp + lane][d], value + d, 16);
                }
                __pipeline_commit();
            }
            const float x = s < end ? head_logit<HB, E>(op, cache, query, state, hh, s) : -INFINITY;
            float top     = x;
            for (int offset = kWarp / 2; offset > 0; offset >>= 1)
                top = fmaxf(top, __shfl_xor_sync(0xffffffffU, top, offset));
            // A step with keys of this warp has a finite max.
            float p = 0.0f, scale = 1.0f;
            if (exact) {
                // The split route's statistics, normalization and BF16 probability (one step).
                float lane_m = -INFINITY, lane_z = 0.0f;
                if (s < end) online_update(lane_m, lane_z, x);
                for (int offset = kWarp / 2; offset > 0; offset >>= 1)
                    merge_stats(lane_m, lane_z, __shfl_xor_sync(0xffffffffU, lane_m, offset),
                                __shfl_xor_sync(0xffffffffU, lane_z, offset));
                if (lane == 0) {
                    warp_m[group][hh] = lane_m;
                    warp_z[group][hh] = lane_z;
                }
                __syncthreads();
                float chunk_m = warp_m[0][hh], chunk_z = warp_z[0][hh];
                for (int w = 1; w < kKeyWarps; ++w)
                    merge_stats(chunk_m, chunk_z, warp_m[w][hh], warp_z[w][hh]);
                const float sink = load(
                    op.sink[query.h0 + min(hh, query.heads - 1) + op.query_heads * query.token]);
                const float best = fmaxf(sink, chunk_m);
                float norm       = expf(sink - best);
                if (chunk_z > 0.0f)
                    norm = __fadd_rn(norm, __fmul_rn(chunk_z, expf(chunk_m - best)));
                p = s < end ? round_bf16(__fdiv_rn(expf(x - best), norm)) : 0.0f;
            } else if (count > 0) {
                const float raised = fmaxf(m, top);
                scale              = expf(m - raised);
                p                  = s < end ? expf(x - raised) : 0.0f;
                z                  = __fadd_rn(__fmul_rn(z, scale), warp_sum(p));
                m                  = raised;
#pragma unroll
                for (int e = 0; e < E; ++e) acc[e] = __fmul_rn(acc[e], scale);
            }
            if constexpr (kStaged) {
                __pipeline_wait_prior(0);
                __syncthreads();
                const int d0 = E * lane;
#pragma unroll
                for (int source = 0; source < kWarp; ++source) {
                    const float pk = __shfl_sync(0xffffffffU, p, source);
                    if (source < count && d0 < D) {
                        const __half* row = &staged[group * kWarp + source][d0];
#pragma unroll
                        for (int e = 0; e < E; e += 2) {
                            const float2 value =
                                __half22float2(*reinterpret_cast<const __half2*>(row + e));
                            acc[e]     = __fmaf_rn(pk, value.x, acc[e]);
                            acc[e + 1] = __fmaf_rn(pk, value.y, acc[e + 1]);
                        }
                    }
                }
                __syncthreads();
            } else {
                for (int first = 0; first < count; first += kKeysInFlight) {
                    float v[kKeysInFlight][E];
#pragma unroll
                    for (int u = 0; u < kKeysInFlight; ++u) {
                        const __half* value =
                            value_row(op, cache, query, base + min(first + u, count - 1));
#pragma unroll
                        for (int e = 0; e < E; ++e) {
                            const int d = lane + kWarp * e;
                            v[u][e]     = d < D ? __half2float(value[d]) : 0.0f;
                        }
                    }
#pragma unroll
                    for (int u = 0; u < kKeysInFlight; ++u) {
                        const int source = first + u;
                        const float pk   = __shfl_sync(0xffffffffU, p, source & (kWarp - 1));
                        if (source < count)
#pragma unroll
                            for (int e = 0; e < E; ++e) acc[e] = __fmaf_rn(pk, v[u][e], acc[e]);
                    }
                }
            }
        }
        if (lane == 0) {
            warp_m[group][hh] = m;
            warp_z[group][hh] = z;
        }
#pragma unroll
        for (int e = 0; e < E; ++e) {
            const int d = kStaged ? E * lane + e : lane + kWarp * e;
            if (d < D) partial[group][hh][d] = acc[e];
        }
        __syncthreads();
        if (exact) {
            // The chunk's value sum, warps added in warp order (the output before rounding).
            for (int i = threadIdx.x; i < query.heads * D; i += blockDim.x) {
                const int oh = i / D, d = i % D;
                float sum = partial[0][oh][d];
                for (int w = 1; w < kKeyWarps; ++w) sum = __fadd_rn(sum, partial[w][oh][d]);
                op.chunk_out[stats_index(op, query.token, query.h0 + oh, c) * D + d] = sum;
            }
            __syncthreads();
            continue;
        }
        // The chunk: warps rescaled to the chunk's max and added in warp order.
        for (int i = threadIdx.x; i < query.heads * D; i += blockDim.x) {
            const int oh = i / D, d = i % D;
            float top = warp_m[0][oh];
            for (int w = 1; w < kKeyWarps; ++w) top = fmaxf(top, warp_m[w][oh]);
            float sum = __fmul_rn(partial[0][oh][d], expf(warp_m[0][oh] - top));
            for (int w = 1; w < kKeyWarps; ++w)
                sum = __fadd_rn(sum, __fmul_rn(partial[w][oh][d], expf(warp_m[w][oh] - top)));
            op.chunk_out[stats_index(op, query.token, query.h0 + oh, c) * D + d] = sum;
        }
        if (threadIdx.x < query.heads) {
            const int h = threadIdx.x;
            float top   = warp_m[0][h];
            for (int w = 1; w < kKeyWarps; ++w) top = fmaxf(top, warp_m[w][h]);
            float sum = __fmul_rn(warp_z[0][h], expf(warp_m[0][h] - top));
            for (int w = 1; w < kKeyWarps; ++w)
                sum = __fadd_rn(sum, __fmul_rn(warp_z[w][h], expf(warp_m[w][h] - top)));
            const std::int64_t at = stats_index(op, query.token, query.h0 + h, c);
            op.chunk_max[at]      = top;
            op.chunk_sum[at]      = sum;
        }
        __syncthreads();
    }
}

// Writes the call's K/V row of one token into the cache when the cache retains its position.
// Callers run it after every read of the cache by this call.
template <class Cache>
__device__ void append_token(const RelativeSinkOperands& op, const Cache& cache, int token,
                             int first_head, int end_head) {
    const int D   = op.head_dim;
    const int Hkv = op.kv_heads;
    const int b   = token / op.tokens;
    const int p   = op.positions[token];
    const int L   = op.positions[op.tokens * b];
    const int row = op.rows[b];
    if (!cache.retains(p, L + op.tokens)) return;
    for (int i = threadIdx.x + D * first_head; i < D * end_head; i += blockDim.x) {
        const int d = i % D;
        const int g = i / D;
        const std::int64_t source =
            static_cast<std::int64_t>(D) * (g + static_cast<std::int64_t>(Hkv) * token) + d;
        cache.key(row, g, p)[d]   = op.k_prepared[source];
        cache.value(row, g, p)[d] = op.v_prepared[source];
    }
}

// Split route, stage 3: one block per (token, query head) adds the chunk sums in chunk order and
// rounds to BF16; the blocks of the first Hkv heads then append one KV head each.
template <class Cache>
__global__ void split_finish_kernel(RelativeSinkOperands op, Cache cache) {
    pdl_enter();
    constexpr int kBatch = 16;
    const int token      = blockIdx.x;
    const int h          = blockIdx.y;
    const int chunks     = make_query<1>(op, cache, token, 0).chunks;
    const int D          = op.head_dim;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float* parts = op.chunk_out + stats_index(op, token, h, 0) * D + d;
        float sum          = parts[0];
        int c              = 1;
        for (; c + kBatch <= chunks; c += kBatch) {
            float part[kBatch];
#pragma unroll
            for (int u = 0; u < kBatch; ++u) part[u] = parts[static_cast<std::int64_t>(c + u) * D];
#pragma unroll
            for (int u = 0; u < kBatch; ++u) sum = __fadd_rn(sum, part[u]);
        }
        for (; c < chunks; ++c) sum = __fadd_rn(sum, parts[static_cast<std::int64_t>(c) * D]);
        const std::int64_t at = static_cast<std::int64_t>(D) * (h + op.query_heads * token) + d;
        op.out[at]            = output_value(op, at, sum);
    }
    if (h < op.kv_heads) append_token(op, cache, token, h, h + 1);
}

// Paged split route for 2 <= T <= kColumnsMaxTokens (verification rounds) on a row whose last
// token sees more than kColumnsMinKeys keys (shorter rows: split_online_kernel, launched with it;
// each kernel leaves the other's rows): the one pass of split_online_kernel for every token of
// the row in one block per (chunk slot, KV head slice, row), so each step's K and V rows are
// copied to shared memory once for all the tokens instead of read once per token. Per token and
// head the operations and their order are split_online's, so a token's bits do not depend on
// the call's T or on the kernel. Tokens are walked in groups of equal chunk size (the visible
// count of a row's tokens straddles at most one multiple of 8192).


template <int HB>
struct ColumnsShared {
    __align__(16) __half value[kChunkUnit][kWarp * 4];
    __align__(16) __nv_bfloat16 keys[kChunkUnit][kWarp * 4 + 8];
    QueryState<HB, 4> state[kColumnsMaxTokens];
    float partial[kKeyWarps][HB][kWarp * 4];
    float warp_m[kKeyWarps][HB];
    float warp_z[kKeyWarps][HB];
};

template <int HB>
__global__ void __launch_bounds__(kWideThreads<HB>)
    split_columns_kernel(RelativeSinkOperands op, PageAccess cache) {
    pdl_enter();
    extern __shared__ __align__(16) unsigned char columns_shared[];
    auto& sh        = *reinterpret_cast<ColumnsShared<HB>*>(columns_shared);
    const int lane  = threadIdx.x % kWarp;
    const int warp  = threadIdx.x / kWarp;
    const int group = warp % kKeyWarps;
    const int hh    = warp / kKeyWarps;
    const int D     = op.head_dim;
    const int T     = op.tokens;
    const int R     = op.params.relative_extent;
    const int row_b = blockIdx.z;
    if (!columns_row(op, row_b)) return;
    Query query[kColumnsMaxTokens];
#pragma unroll
    for (int t = 0; t < kColumnsMaxTokens; ++t)
        if (t < T) query[t] = make_query<HB>(op, cache, t + T * row_b, blockIdx.y);
    int most = 0; // chunks of the token with the most
#pragma unroll
    for (int t = 0; t < kColumnsMaxTokens; ++t)
        if (t < T) most = max(most, query[t].chunks);
    if (static_cast<int>(blockIdx.x) >= most) return;
    for (int t = 0; t < T; ++t) load_query_state(op, query[t], sh.state[t]);
    const Query& last = query[T - 1];
    for (int part = 0; part < 2; ++part) {
        // Tokens of this part: chunk size equal to the first token's (part 0) or larger (part 1).
        int chunk = 0, top_p = -1, chunks = 0;
#pragma unroll
        for (int t = 0; t < kColumnsMaxTokens; ++t)
            if (t < T && (query[t].chunk == query[0].chunk) == (part == 0)) {
                chunk  = query[t].chunk;
                top_p  = query[t].p;
                chunks = query[t].chunks;
            }
        if (top_p < 0) continue;
        for (int c = blockIdx.x; c < chunks; c += gridDim.x) {
            const int begin = last.lo + c * chunk;
            const int end   = min(begin + chunk, top_p + 1);
            const int steps = (end - begin + kChunkUnit - 1) / kChunkUnit;
            float m[kColumnsMaxTokens], z[kColumnsMaxTokens], acc[kColumnsMaxTokens][4];
#pragma unroll
            for (int t = 0; t < kColumnsMaxTokens; ++t) {
                m[t] = -INFINITY;
                z[t] = 0.0f;
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[t][e] = 0.0f;
            }
            for (int step = 0; step < steps; ++step) {
                const int first = begin + step * kChunkUnit;
                // The step's rows: committed ones from the cache, the call's from the workspace.
                const int vectors = D / 8;
                for (int i = threadIdx.x; i < 2 * kChunkUnit * vectors; i += blockDim.x) {
                    const int key = i % (kChunkUnit * vectors) / vectors, u = i % vectors;
                    const int s = first + key;
                    if (s > top_p) continue;
                    if (i < kChunkUnit * vectors)
                        __pipeline_memcpy_async(&sh.keys[key][u * 8],
                                                key_row(op, cache, last, s) + u * 8, 16);
                    else
                        __pipeline_memcpy_async(&sh.value[key][u * 8],
                                                value_row(op, cache, last, s) + u * 8, 16);
                }
                __pipeline_commit();
                __pipeline_wait_prior(0);
                __syncthreads();
                const int base = first + group * kWarp;
                const int s    = base + lane;
#pragma unroll
                for (int t = 0; t < kColumnsMaxTokens; ++t) {
                    if (t >= T || (query[t].chunk == query[0].chunk) != (part == 0)) continue;
                    const int end_t = min(begin + chunk, query[t].p + 1);
                    const int count = max(0, min(kWarp, end_t - base));
                    if (count == 0) continue; // warp-uniform: nothing of this token here
                    float x = -INFINITY;
                    if (s < end_t) {
                        const int distance  = query[t].p - s;
                        float column[kWarp] = {};
                        if (distance < R) load_column(op, distance, column);
                        float dot      = 0.0f;
                        const float* q = sh.state[t].q[hh];
                        const auto* row =
                            reinterpret_cast<const uint4*>(sh.keys[group * kWarp + lane]);
                        for (int u = 0; u < 16; ++u) {
                            if (u * 8 >= D) break;
                            const uint4 key  = row[u];
                            const auto* pair = reinterpret_cast<const __nv_bfloat162*>(&key);
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const float2 k = __bfloat1622float2(pair[j]);
                                dot            = __fmaf_rn(q[u * 8 + 2 * j], k.x, dot);
                                dot            = __fmaf_rn(q[u * 8 + 2 * j + 1], k.y, dot);
                            }
                        }
                        float bias = 0.0f;
                        if (distance < R)
#pragma unroll
                            for (int j = 0; j < kWarp; ++j)
                                if (j < op.params.relative_dim)
                                    bias = __fmaf_rn(sh.state[t].rel[hh][j], column[j], bias);
                        x = round_bf16(__fmul_rn(round_bf16(dot), op.params.scale));
                        if (distance < R) x = round_bf16(__fadd_rn(x, round_bf16(bias)));
                    }
                    float top = x;
                    for (int offset = kWarp / 2; offset > 0; offset >>= 1)
                        top = fmaxf(top, __shfl_xor_sync(0xffffffffU, top, offset));
                    const float raised = fmaxf(m[t], top);
                    const float scale  = expf(m[t] - raised);
                    const float p      = s < end_t ? expf(x - raised) : 0.0f;
                    z[t]               = __fadd_rn(__fmul_rn(z[t], scale), warp_sum(p));
                    m[t]               = raised;
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc[t][e] = __fmul_rn(acc[t][e], scale);
                    const int d0 = 4 * lane;
                    for (int source = 0; source < kWarp; ++source) {
                        const float pk = __shfl_sync(0xffffffffU, p, source);
                        if (source < count && d0 < D) {
                            const __half* vrow = &sh.value[group * kWarp + source][d0];
#pragma unroll
                            for (int e = 0; e < 4; e += 2) {
                                const float2 value =
                                    __half22float2(*reinterpret_cast<const __half2*>(vrow + e));
                                acc[t][e]     = __fmaf_rn(pk, value.x, acc[t][e]);
                                acc[t][e + 1] = __fmaf_rn(pk, value.y, acc[t][e + 1]);
                            }
                        }
                    }
                }
                __syncthreads();
            }
            // Each token's chunk: warps rescaled to the chunk's max and added in warp order.
#pragma unroll
            for (int t = 0; t < kColumnsMaxTokens; ++t) {
                if (t >= T || (query[t].chunk == query[0].chunk) != (part == 0) ||
                    c >= query[t].chunks)
                    continue;
                if (lane == 0) {
                    sh.warp_m[group][hh] = m[t];
                    sh.warp_z[group][hh] = z[t];
                }
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    if (4 * lane + e < D) sh.partial[group][hh][4 * lane + e] = acc[t][e];
                __syncthreads();
                const Query& qt = query[t];
                for (int i = threadIdx.x; i < qt.heads * D; i += blockDim.x) {
                    const int oh = i / D, d = i % D;
                    float top = sh.warp_m[0][oh];
                    for (int w = 1; w < kKeyWarps; ++w) top = fmaxf(top, sh.warp_m[w][oh]);
                    float sum = __fmul_rn(sh.partial[0][oh][d], expf(sh.warp_m[0][oh] - top));
                    for (int w = 1; w < kKeyWarps; ++w)
                        sum = __fadd_rn(
                            sum, __fmul_rn(sh.partial[w][oh][d], expf(sh.warp_m[w][oh] - top)));
                    op.chunk_out[stats_index(op, qt.token, qt.h0 + oh, c) * D + d] = sum;
                }
                if (threadIdx.x < qt.heads) {
                    const int h = threadIdx.x;
                    float top   = sh.warp_m[0][h];
                    for (int w = 1; w < kKeyWarps; ++w) top = fmaxf(top, sh.warp_m[w][h]);
                    float sum = __fmul_rn(sh.warp_z[0][h], expf(sh.warp_m[0][h] - top));
                    for (int w = 1; w < kKeyWarps; ++w)
                        sum =
                            __fadd_rn(sum, __fmul_rn(sh.warp_z[w][h], expf(sh.warp_m[w][h] - top)));
                    const std::int64_t at = stats_index(op, qt.token, qt.h0 + h, c);
                    op.chunk_max[at]      = top;
                    op.chunk_sum[at]      = sum;
                }
                __syncthreads();
            }
        }
    }
}

// Paged split route, last stage: one block per (token, query head) combines the chunks of the
// one pass: M = max(sink, chunk maxima), Z = exp(sink - M) + sum_c z_c exp(m_c - M) and
// o = sum_c o_c exp(m_c - M) / Z, chunks in chunk order; the blocks of the first Hkv heads then
// append one KV head each.
template <class Cache>
__global__ void split_online_finish_kernel(RelativeSinkOperands op, Cache cache) {
    pdl_enter();
    __shared__ float weight[kRelativeSinkMaxChunks];
    __shared__ float normalizer;
    const int token          = blockIdx.x;
    const int h              = blockIdx.y;
    const int chunks         = make_query<1>(op, cache, token, 0).chunks;
    const int D              = op.head_dim;
    const std::int64_t first = stats_index(op, token, h, 0);
    if (chunks == 1) {
        // A single chunk kept the two-pass arithmetic: its value sum is the output.
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            const std::int64_t at = static_cast<std::int64_t>(D) * (h + op.query_heads * token) + d;
            op.out[at]            = output_value(op, at, op.chunk_out[first * D + d]);
        }
        if (h < op.kv_heads) append_token(op, cache, token, h, h + 1);
        return;
    }
    if (threadIdx.x < kWarp) {
        const int lane   = threadIdx.x;
        float top        = load(op.sink[h + op.query_heads * token]);
        const float sink = top;
        for (int c = lane; c < chunks; c += kWarp) top = fmaxf(top, op.chunk_max[first + c]);
        for (int offset = kWarp / 2; offset > 0; offset >>= 1)
            top = fmaxf(top, __shfl_xor_sync(0xffffffffU, top, offset));
        for (int c = lane; c < chunks; c += kWarp) weight[c] = expf(op.chunk_max[first + c] - top);
        __syncwarp();
        if (lane == 0) {
            float norm = expf(sink - top);
            for (int c = 0; c < chunks; ++c)
                norm = __fadd_rn(norm, __fmul_rn(op.chunk_sum[first + c], weight[c]));
            normalizer = norm;
        }
    }
    __syncthreads();
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float* parts = op.chunk_out + first * D + d;
        float sum          = 0.0f;
        for (int c = 0; c < chunks; ++c)
            sum = __fadd_rn(sum, __fmul_rn(parts[static_cast<std::int64_t>(c) * D], weight[c]));
        const std::int64_t at = static_cast<std::int64_t>(D) * (h + op.query_heads * token) + d;
        op.out[at]            = output_value(op, at, __fdiv_rn(sum, normalizer));
    }
    if (h < op.kv_heads) append_token(op, cache, token, h, h + 1);
}

// Decode route (see decode_route; D <= 128, GQA group <= HB): the whole call, prologue included,
// in one launch of one cluster of kDecodeBlocks blocks per (row, KV head). Block b owns chunk b;
// warp (g, hh) = (w % 4, w / 4) takes head hh of the chunk's keys begin + 32g + lane, so each
// warp runs one head's share and the work of a key is spread over HB warps. Prepared vectors,
// logits, statistics and value sums are the split route's, element by element in the same order,
// so every route gives the same bits. Before waiting on the kernel before it, a block reads
// weights only (the bias columns of its keys) and asks L2 for its cache rows, both addressed from
// a position and row read around L1 (a stale guess costs a reload, never a wrong value); right
// after the wait it copies the rows to shared memory with the same addresses, all threads on
// consecutive 16 bytes. With a single chunk its block finishes alone; otherwise statistics and
// chunk sums travel through distributed shared memory and block 0 writes the output.
template <int HB>
struct DecodeShared {
    QueryState<HB, 4> state;
    __align__(16) __half value[kChunkUnit][kWarp * 4]; // V rows of the block's keys
    // K rows of the block's keys (k' of the call's token at its own row); 16 bytes of padding
    // keep a quarter warp reading eight rows on distinct banks.
    __align__(16) __nv_bfloat16 keys[kChunkUnit][kWarp * 4 + 8];
    float warp_m[kKeyWarps][HB];
    float warp_z[kKeyWarps][HB];
    float chunk_m[HB]; // statistics of this block's chunk, read by the cluster
    float chunk_z[HB];
    float all_m[kDecodeBlocks][HB]; // every chunk's statistics, gathered from the cluster
    float all_z[kDecodeBlocks][HB];
    float partial[kKeyWarps][HB][kWarp * 4];
    float chunk_out[HB][kWarp * 4]; // value sum of this block's chunk, read by the cluster
};

template <int HB>
constexpr int kDecodeThreads = kKeyWarps * HB * kWarp;

__device__ __forceinline__ void prefetch_line(const void* line) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(line));
}

// Copies the K and V rows of the keys first .. first + 127 committed below `committed` to shared
// memory: consecutive threads take consecutive 16 bytes of a row.
template <int HB, class Cache>
__device__ __forceinline__ void stage_rows(const Cache& cache, int row, int g, int first,
                                           int committed, int D, DecodeShared<HB>& sh) {
    const int vectors = D / 8;
    const int all     = kChunkUnit * vectors;
    for (int i = threadIdx.x; i < 2 * all; i += blockDim.x) {
        const int key = i % all / vectors, u = i % vectors;
        const int s = first + key;
        if (s >= committed) continue;
        if (i < all)
            __pipeline_memcpy_async(&sh.keys[key][u * 8], cache.key(row, g, s) + u * 8, 16);
        else
            __pipeline_memcpy_async(&sh.value[key][u * 8], cache.value(row, g, s) + u * 8, 16);
    }
    __pipeline_commit();
}

// Prologue item j of a decode block: q' then r' of its HB heads, then k' and v' of its KV head.
template <int HB>
__device__ __forceinline__ PrologueItem decode_item(const RelativeSinkOperands& op,
                                                    const Query& query, int j) {
    if (j < 2 * HB) {
        const int hh = j < HB ? j : j - HB;
        return prologue_item(op, j < HB ? 0 : 1, query.h0 + min(hh, query.heads - 1), query.token);
    }
    return prologue_item(op, j == 2 * HB ? 2 : 3, query.g, query.token);
}

template <int HB, class Cache>
__global__ void __cluster_dims__(kDecodeBlocks, 1, 1) __launch_bounds__(kDecodeThreads<HB>)
    decode_kernel(RelativeSinkOperands op, Cache cache) {
    namespace cg                    = cooperative_groups;
    const cg::cluster_group cluster = cg::this_cluster();
    extern __shared__ __align__(16) unsigned char decode_shared[];
    auto& sh        = *reinterpret_cast<DecodeShared<HB>*>(decode_shared);
    const int lane  = threadIdx.x % kWarp;
    const int warp  = threadIdx.x / kWarp;
    const int group = warp % kKeyWarps;     // the warp's 32 keys
    const int hh    = warp / kKeyWarps;     // the warp's head
    const int index = group * kWarp + lane; // the lane's key in the chunk
    const int rank  = static_cast<int>(cluster.block_rank());
    const int token = blockIdx.y;
    const int slice = blockIdx.x / kDecodeBlocks;
    const int D     = op.head_dim;
    const int Hq    = op.query_heads;
    const int R     = op.params.relative_extent;

    // Before the wait: the guessed position and row give the bias column (a weight) of the lane's
    // key and the L2 prefetch of the cache rows of the block's keys.
    const int guess       = __ldcg(op.positions + token);
    const int guess_row   = __ldcg(op.rows + token / op.tokens);
    const int guess_first = cache.first_visible(guess) + rank * kChunkUnit;
    float column[kWarp]   = {};
    {
        const int s = guess_first + index;
        if (s <= guess && guess - s < R) load_column(op, guess - s, column);
        const int lines = (D * 2 + 127) / 128;
        for (int i = threadIdx.x; i < 2 * kChunkUnit * lines; i += blockDim.x) {
            const int key = i / lines % kChunkUnit, line = i % lines;
            if (guess_first + key >= guess) continue;
            const void* base =
                i < kChunkUnit * lines
                    ? static_cast<const void*>(cache.key(guess_row, slice, guess_first + key))
                    : static_cast<const void*>(cache.value(guess_row, slice, guess_first + key));
            prefetch_line(static_cast<const char*>(base) + 128 * line);
        }
    }
    pdl_enter();
    stage_rows<HB>(cache, guess_row, slice, guess_first, guess, D, sh);

    const Query query = make_query<HB>(op, cache, token, slice);
    const bool alone  = query.chunks == 1;
    if (query.p != guess || query.row != guess_row) {
        __pipeline_wait_prior(0);
        stage_rows<HB>(cache, query.row, query.g, query.lo + rank * kChunkUnit, query.L, D, sh);
    }
    if (alone && rank > 0) {
        __pipeline_wait_prior(0);
        return;
    }
    const int begin    = query.lo + rank * kChunkUnit;
    const int end      = min(begin + kChunkUnit, query.p + 1);
    const int s        = begin + index;
    const bool live    = s < end;
    const int distance = query.p - s;
    const int count    = max(0, min(kWarp, end - begin - group * kWarp));
    const bool owner   = query.p >= begin && query.p < end;

    if (query.p != guess && live && distance < R) load_column(op, distance, column);

    // Sink of the warp's head; gate of the thread's output element (block 0 writes the output).
    const int outs   = query.heads * D;
    const int output = static_cast<int>(threadIdx.x);
    const float sink = load(op.sink[query.h0 + min(hh, query.heads - 1) + Hq * query.token]);
    const float gate = op.gate != nullptr && output < outs
                           ? load(op.gate[static_cast<std::int64_t>(D) *
                                              (query.h0 + output / D + Hq * query.token) +
                                          output % D])
                           : 0.0f;
    // Prologue: warp j prepares item j.
    if (warp < 2 * HB + (owner ? 2 : 0)) {
        const PrologueItem item = decode_item<HB>(op, query, warp);
        float x[kMaxPerLane];
        prologue_load(item, lane, x);
        prologue_finish(op.params, item, position_scale(op.params, query.p), lane, x);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d = lane + kWarp * i;
            if (warp < HB)
                sh.state.q[warp][d] = x[i];
            else if (warp < 2 * HB && i == 0)
                sh.state.rel[warp - HB][d] = x[0];
            else if (warp == 2 * HB && d < D)
                sh.keys[query.p - begin][d] = __float2bfloat16_rn(x[i]);
            else if (warp == 2 * HB + 1 && d < D)
                sh.value[query.p - begin][d] = __float2half_rn(x[i]);
        }
    }
    __pipeline_wait_prior(0);
    __syncthreads();

    // The lane's logit for the warp's head and the chunk's statistics.
    float x = -INFINITY, m = -INFINITY, z = 0.0f;
    if (live) {
        float dot       = 0.0f;
        const float* q  = sh.state.q[hh];
        const auto* row = reinterpret_cast<const uint4*>(sh.keys[index]);
#pragma unroll
        for (int u = 0; u < 16; ++u) {
            if (u * 8 >= D) break;
            const uint4 key  = row[u];
            const auto* pair = reinterpret_cast<const __nv_bfloat162*>(&key);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 k = __bfloat1622float2(pair[j]);
                dot            = __fmaf_rn(q[u * 8 + 2 * j], k.x, dot);
                dot            = __fmaf_rn(q[u * 8 + 2 * j + 1], k.y, dot);
            }
        }
        const bool biased = distance < R;
        float bias        = 0.0f;
        if (biased)
#pragma unroll
            for (int j = 0; j < kWarp; ++j)
                if (j < op.params.relative_dim)
                    bias = __fmaf_rn(sh.state.rel[hh][j], column[j], bias);
        x = round_bf16(__fmul_rn(round_bf16(dot), op.params.scale));
        if (biased) x = round_bf16(__fadd_rn(x, round_bf16(bias)));
        online_update(m, z, x);
    }
    for (int offset = kWarp / 2; offset > 0; offset >>= 1)
        merge_stats(m, z, __shfl_xor_sync(0xffffffffU, m, offset),
                    __shfl_xor_sync(0xffffffffU, z, offset));
    if (lane == 0) {
        sh.warp_m[group][hh] = m;
        sh.warp_z[group][hh] = z;
    }
    __syncthreads();
    float chunk_m = sh.warp_m[0][hh], chunk_z = sh.warp_z[0][hh];
    for (int w = 1; w < kKeyWarps; ++w)
        merge_stats(chunk_m, chunk_z, sh.warp_m[w][hh], sh.warp_z[w][hh]);
    float top, norm;
    if (alone) {
        // combine_stats over the one chunk.
        top  = fmaxf(sink, chunk_m);
        norm = expf(sink - top);
        if (chunk_z > 0.0f) norm = __fadd_rn(norm, __fmul_rn(chunk_z, expf(chunk_m - top)));
    } else {
        if (group == 0 && lane == 0) {
            sh.chunk_m[hh] = chunk_m;
            sh.chunk_z[hh] = chunk_z;
        }
        cluster.sync();
        // One remote read per thread, then combine_stats from this block's copy.
        if (static_cast<int>(threadIdx.x) < query.chunks * HB) {
            const int c = threadIdx.x / HB, k = threadIdx.x % HB;
            const DecodeShared<HB>* other = cluster.map_shared_rank(&sh, c);
            sh.all_m[c][k]                = other->chunk_m[k];
            sh.all_z[c][k]                = other->chunk_z[k];
        }
        __syncthreads();
        combine_stats(sink, &sh.all_m[0][hh], &sh.all_z[0][hh], query.chunks, HB, top, norm);
    }

    // sum_s bf16(p_s) v_s over the warp's keys in lane order for the warp's head; lane l takes
    // the elements 4l .. 4l + 3.
    const float probability = live ? round_bf16(__fdiv_rn(expf(x - top), norm)) : 0.0f;
    float acc[4]            = {};
    const int d0            = 4 * lane;
    const bool on           = d0 < D;
#pragma unroll
    for (int source = 0; source < kWarp; ++source) {
        const float pk = __shfl_sync(0xffffffffU, probability, source);
        if (source < count && on) {
            const uint2 raw =
                *reinterpret_cast<const uint2*>(&sh.value[group * kWarp + source][d0]);
            const float2 low  = __half22float2(*reinterpret_cast<const __half2*>(&raw.x));
            const float2 high = __half22float2(*reinterpret_cast<const __half2*>(&raw.y));
            acc[0]            = __fmaf_rn(pk, low.x, acc[0]);
            acc[1]            = __fmaf_rn(pk, low.y, acc[1]);
            acc[2]            = __fmaf_rn(pk, high.x, acc[2]);
            acc[3]            = __fmaf_rn(pk, high.y, acc[3]);
        }
    }
    if (on)
#pragma unroll
        for (int e = 0; e < 4; ++e) sh.partial[group][hh][d0 + e] = acc[e];
    __syncthreads();
    // Warp partials added in warp order, then chunk sums in chunk order by block 0.
    float sum = 0.0f;
    if (output < outs) {
        const int oh = output / D, od = output % D;
        sum = sh.partial[0][oh][od];
        for (int w = 1; w < kKeyWarps; ++w) sum = __fadd_rn(sum, sh.partial[w][oh][od]);
    }
    if (!alone) {
        if (output < outs) sh.chunk_out[output / D][output % D] = sum;
        cluster.sync();
        if (rank == 0 && output < outs) {
            // Every remote read is issued before the sums, which stay in chunk order.
            float part[kDecodeBlocks];
#pragma unroll
            for (int c = 1; c < kDecodeBlocks; ++c)
                part[c] = c < query.chunks
                              ? cluster.map_shared_rank(&sh, c)->chunk_out[output / D][output % D]
                              : 0.0f;
#pragma unroll
            for (int c = 1; c < kDecodeBlocks; ++c)
                if (c < query.chunks) sum = __fadd_rn(sum, part[c]);
        }
    }
    if (rank == 0 && output < outs)
        op.out[static_cast<std::int64_t>(D) * (query.h0 + output / D + Hq * query.token) +
               output % D] = gated_value(sum, op.gate != nullptr, gate);
    // The call's K/V row; the slot it overwrites (position p - capacity) is outside the window.
    if (owner && cache.retains(query.p, query.L + op.tokens)) {
        const int vectors = D / 8;
        const int t       = static_cast<int>(threadIdx.x);
        if (t < vectors)
            reinterpret_cast<uint4*>(cache.key(query.row, query.g, query.p))[t] =
                reinterpret_cast<const uint4*>(sh.keys[query.p - begin])[t];
        else if (t < 2 * vectors)
            reinterpret_cast<uint4*>(cache.value(query.row, query.g, query.p))[t - vectors] =
                reinterpret_cast<const uint4*>(&sh.value[query.p - begin][0])[t - vectors];
    }
    // No block leaves while another may still read its shared memory.
    if (!alone) cluster.sync();
}

// Where the decode route applies: one token per row on a ring whose window fits the cluster's
// chunks (the model's local layers at T = 1). Longer calls take the split route, paged calls too
// (the host cannot know how many keys a captured call will see).
bool decode_route(const RingAccess& access, int tokens) {
    return tokens == 1 && access.ring.window <= kDecodeMaxKeys;
}

bool decode_route(const PageAccess&, int) { return false; }

// Sequential route: one block per (token, KV head slice) walks the same chunks in order.
template <int HB, int E, class Cache>
__global__ void sequential_kernel(RelativeSinkOperands op, Cache cache) {
    __shared__ float chunk_m[kRelativeSinkMaxChunks][HB];
    __shared__ float chunk_z[kRelativeSinkMaxChunks][HB];
    __shared__ float result[HB * kWarp * E];
    const Query query = make_query<HB>(op, cache, blockIdx.x, blockIdx.y);
    __shared__ QueryState<HB, E> state;
    load_query_state(op, query, state);
    for (int c = 0; c < query.chunks; ++c) {
        const ChunkStats<HB> stats = chunk_stats<HB, E>(op, cache, query, state, c);
        if (threadIdx.x == 0)
#pragma unroll
            for (int hh = 0; hh < HB; ++hh) {
                chunk_m[c][hh] = stats.m[hh];
                chunk_z[c][hh] = stats.z[hh];
            }
    }
    __syncthreads();
    float top[HB], norm[HB];
#pragma unroll
    for (int hh = 0; hh < HB; ++hh) {
        const int h = query.h0 + min(hh, query.heads - 1);
        combine_stats(load(op.sink[h + op.query_heads * query.token]), &chunk_m[0][hh],
                      &chunk_z[0][hh], query.chunks, HB, top[hh], norm[hh]);
    }
    const int D              = op.head_dim;
    constexpr int kPerThread = (HB * kWarp * E + kKeyWarps * kWarp - 1) / (kKeyWarps * kWarp);
    float sum[kPerThread];
    for (int c = 0; c < query.chunks; ++c) {
        chunk_values<HB, E>(op, cache, query, state, c, top, norm, result);
#pragma unroll
        for (int j = 0; j < kPerThread; ++j) {
            const int i = threadIdx.x + j * kKeyWarps * kWarp;
            if (i < query.heads * D) sum[j] = c == 0 ? result[i] : __fadd_rn(sum[j], result[i]);
        }
    }
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        const int i = threadIdx.x + j * kKeyWarps * kWarp;
        if (i < query.heads * D) {
            const std::int64_t at =
                static_cast<std::int64_t>(D) * (query.h0 + i / D + op.query_heads * query.token) +
                i % D;
            op.out[at] = output_value(op, at, sum[j]);
        }
    }
}

// Flash route (T > 16, D <= 128), FlashAttention-2 style. One block per (64 query positions, up
// to 3 query heads of one KV head); warp w takes 16 positions of head w / 4. Key tiles of 64 keys
// aligned to multiples of 64 positions, K (BF16) and V (FP16 as stored) copied with cp.async into
// a double buffer that all warps share; only the tiles holding visible keys are visited (window,
// causal diagonal). q.k and the bias table r'.P run on BF16 tensor cores and the logits keep the
// contract's rounding points; the softmax is online (one pass, running max): exp(x - m) rounded to
// FP16 is the A operand of an FP16 MMA with V that accumulates each tile in FP16 before the FP32
// output (as the Qwen prefill attention with NINFER_ATTN_PV_F16ACC), and the sink and the
// normalizer enter at the end.
// Results agree with the other routes within the oracle criterion, not bit for bit; equal calls
// give equal bits.
constexpr int kFlashPositions = 64;
constexpr int kFlashKeys      = 64;
constexpr int kFlashHeads     = 3;
constexpr int kFlashWarps     = kFlashHeads * kFlashPositions / 16;
constexpr int kFlashThreads   = kFlashWarps * kWarp;
constexpr int kFlashMaxDim    = 128;
constexpr int kFlashRow       = kFlashMaxDim + 8; // padded shared row: conflict-free ldmatrix
constexpr int kFlashBias      = kFlashKeys + 16;  // distances one warp sees in a tile (79 used)
constexpr float kLog2e        = 1.4426950408889634f;

struct FlashShared {
    __nv_bfloat16 k[2][kFlashKeys][kFlashRow];
    __half v[2][kFlashKeys][kFlashRow];
    // Per warp: bias[i][n] = bf16(r'_i . P[:, base - n]) for its 16 rows, base = the distance of
    // row i to key i + n - 15 of the tile.
    __nv_bfloat16 bias[kFlashWarps][16][kFlashBias];
};

__device__ __forceinline__ unsigned pack_bf16(float low, float high) {
    const __nv_bfloat162 pair = __floats2bfloat162_rn(low, high);
    return *reinterpret_cast<const unsigned*>(&pair);
}

__device__ __forceinline__ unsigned load_pair(const __nv_bfloat16* at) {
    return *reinterpret_cast<const unsigned*>(at);
}

__device__ __forceinline__ unsigned shared_address(const void* at) {
    return static_cast<unsigned>(__cvta_generic_to_shared(at));
}

// Copies keys s0 .. s0 + 63 of KV head g into buffer `buffer`: keys before L from the cache, the
// call's own keys from the prepared block; keys outside [first, last] are zero.
template <class Cache>
__device__ __forceinline__ void flash_load_tile(const RelativeSinkOperands& op, const Cache& cache,
                                                int row, int g, int b, int L, int first, int last,
                                                int s0, FlashShared& sh, int buffer) {
    const int D       = op.head_dim;
    const int vectors = D / 8;
    for (int i = threadIdx.x; i < kFlashKeys * vectors; i += blockDim.x) {
        const int key = i / vectors, d = (i % vectors) * 8;
        const int s = s0 + key;
        if (s >= first && s <= last) {
            const std::int64_t block_row =
                static_cast<std::int64_t>(D) *
                (g + static_cast<std::int64_t>(op.kv_heads) * (s - L + op.tokens * b));
            const __nv_bfloat16* k = s < L ? cache.key(row, g, s) : op.k_prepared + block_row;
            const __half* v        = s < L ? cache.value(row, g, s) : op.v_prepared + block_row;
            __pipeline_memcpy_async(&sh.k[buffer][key][d], k + d, 16);
            __pipeline_memcpy_async(&sh.v[buffer][key][d], v + d, 16);
        } else {
            *reinterpret_cast<uint4*>(&sh.k[buffer][key][d]) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(&sh.v[buffer][key][d]) = make_uint4(0, 0, 0, 0);
        }
    }
}

template <class Cache>
__global__ void __launch_bounds__(kFlashThreads, 1)
    flash_kernel(RelativeSinkOperands op, Cache cache) {
    extern __shared__ __align__(16) unsigned char flash_smem[];
    FlashShared& sh = *reinterpret_cast<FlashShared*>(flash_smem);
    const int lane  = threadIdx.x % kWarp;
    const int warp  = threadIdx.x / kWarp;
    const int g4 = lane / 4, q4 = lane % 4;
    const int T     = op.tokens;
    const int D     = op.head_dim;
    const int Hq    = op.query_heads;
    const int R     = op.params.relative_extent;
    const int dr    = op.params.relative_dim;
    const int tiles = (T + kFlashPositions - 1) / kFlashPositions;
    const int b     = blockIdx.x / tiles;
    // Heaviest query tiles (most keys under a causal mask) first.
    const int t0         = (tiles - 1 - static_cast<int>(blockIdx.x % tiles)) * kFlashPositions;
    const int group      = Hq / op.kv_heads;
    const int per_g      = (group + kFlashHeads - 1) / kFlashHeads;
    const int g          = blockIdx.y / per_g;
    const int first_head = (blockIdx.y % per_g) * kFlashHeads;
    const int heads      = min(kFlashHeads, group - first_head);
    const bool own       = warp / 4 < heads; // warps beyond the slice's heads only copy tiles
    const int h          = g * group + first_head + min(warp / 4, heads - 1);
    const int r0         = (warp % 4) * 16; // the warp's first position in the block's tile
    const int rows       = min(kFlashPositions, T - t0);
    const int L          = op.positions[T * b];
    const int row        = op.rows[b];
    const int p0         = L + t0;
    const int last       = p0 + rows - 1;
    const int first      = cache.first_visible(p0);
    const int start      = first - first % kFlashKeys;

    // This thread's two query rows of the MMA fragments: warp rows g4 and g4 + 8.
    int position[2], lo[2];
    bool live[2];
    float sink[2];
    for (int i = 0; i < 2; ++i) {
        const int r = r0 + g4 + 8 * i;
        live[i]     = own && r < rows;
        position[i] = p0 + r;
        lo[i]       = cache.first_visible(position[i]);
        sink[i]     = live[i] ? load(op.sink[h + Hq * (t0 + r + T * b)]) : 0.0f;
    }
    // Keys any row of this warp sees; a tile outside them is skipped by the warp.
    const int warp_first = cache.first_visible(p0 + r0);
    const int warp_last  = min(p0 + r0 + 15, last);
    const bool warp_live = own && r0 < rows;

    const auto query_base = [&](int i) {
        return static_cast<std::int64_t>(D) * (h + Hq * (t0 + r0 + g4 + 8 * i + T * b));
    };
    unsigned qa[kFlashMaxDim / 16][4];
#pragma unroll
    for (int kk = 0; kk < kFlashMaxDim / 16; ++kk) {
        const int d   = kk * 16 + q4 * 2;
        const bool on = d < D;
        qa[kk][0]     = on && live[0] ? load_pair(op.q_prepared + query_base(0) + d) : 0U;
        qa[kk][1]     = on && live[1] ? load_pair(op.q_prepared + query_base(1) + d) : 0U;
        qa[kk][2]     = on && live[0] ? load_pair(op.q_prepared + query_base(0) + d + 8) : 0U;
        qa[kk][3]     = on && live[1] ? load_pair(op.q_prepared + query_base(1) + d + 8) : 0U;
    }
    unsigned ra[2][4];
    {
        const auto rel = [&](int i, int j) {
            return live[i] && j < dr
                       ? load(op.r_prepared[static_cast<std::int64_t>(dr) *
                                                (h + Hq * (t0 + r0 + g4 + 8 * i + T * b)) +
                                            j])
                       : 0.0f;
        };
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            const int j = ks * 16 + q4 * 2;
            ra[ks][0]   = pack_bf16(rel(0, j), rel(0, j + 1));
            ra[ks][1]   = pack_bf16(rel(1, j), rel(1, j + 1));
            ra[ks][2]   = pack_bf16(rel(0, j + 8), rel(0, j + 9));
            ra[ks][3]   = pack_bf16(rel(1, j + 8), rel(1, j + 9));
        }
    }
    // P[j, distance], zero outside [0, R) x [0, dr). The load is unconditional (clamped address)
    // so that a tile's loads issue back to back instead of one guarded branch each.
    const auto P = [&](int distance, int j) {
        const float value = load(op.rel_proj[min(max(distance, 0), R - 1) +
                                             static_cast<std::int64_t>(R) * min(j, dr - 1)]);
        return distance >= 0 && distance < R && j < dr ? value : 0.0f;
    };

    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    float o[kFlashMaxDim / 8][4] = {};
    const int count              = last < start ? 0 : (last - start) / kFlashKeys + 1;
    if (count > 0) flash_load_tile(op, cache, row, g, b, L, first, last, start, sh, 0);
    __pipeline_commit();
    for (int tile = 0; tile < count; ++tile) {
        const int s0     = start + tile * kFlashKeys;
        const int buffer = tile & 1;
        if (tile + 1 < count)
            flash_load_tile(op, cache, row, g, b, L, first, last, s0 + kFlashKeys, sh, buffer ^ 1);
        __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();
        if (warp_live && s0 <= warp_last && s0 + kFlashKeys - 1 >= warp_first) {
            // Bias table of the warp's rows for this tile, when some distance is below R.
            __nv_bfloat16(&bias)[16][kFlashBias] = sh.bias[warp];
            const int base                       = p0 + r0 - s0 + 15;
            const bool with_bias                 = base >= 0 && base - (kFlashBias - 2) < R;
            if (with_bias) {
#pragma unroll
                for (int nt = 0; nt < kFlashBias / 8; ++nt) {
                    float c[4]         = {};
                    const int distance = base - nt * 8 - g4;
#pragma unroll
                    for (int ks = 0; ks < 2; ++ks) {
                        if (ks * 16 >= dr) break;
                        const int j = ks * 16 + q4 * 2;
                        mma_bf16(c[0], c[1], c[2], c[3], ra[ks][0], ra[ks][1], ra[ks][2], ra[ks][3],
                                 pack_bf16(P(distance, j), P(distance, j + 1)),
                                 pack_bf16(P(distance, j + 8), P(distance, j + 9)));
                    }
                    const int col                                    = nt * 8 + q4 * 2;
                    *reinterpret_cast<unsigned*>(&bias[g4][col])     = pack_bf16(c[0], c[1]);
                    *reinterpret_cast<unsigned*>(&bias[g4 + 8][col]) = pack_bf16(c[2], c[3]);
                }
                __syncwarp();
            }

            // q.k for 16 rows x 64 keys.
            float x[kFlashKeys / 8][4] = {};
            const unsigned k_base      = shared_address(&sh.k[buffer][0][0]);
#pragma unroll
            for (int kk = 0; kk < kFlashMaxDim / 16; ++kk) {
                if (kk * 16 >= D) break;
#pragma unroll
                for (int np = 0; np < kFlashKeys / 16; ++np) {
                    const int key = np * 16 + (lane / 16) * 8 + lane % 8;
                    const int d   = kk * 16 + ((lane / 8) % 2) * 8;
                    unsigned b0, b1, b2, b3;
                    ldmatrix_x4(b0, b1, b2, b3, k_base + (key * kFlashRow + d) * 2);
                    mma_bf16(x[2 * np][0], x[2 * np][1], x[2 * np][2], x[2 * np][3], qa[kk][0],
                             qa[kk][1], qa[kk][2], qa[kk][3], b0, b1);
                    mma_bf16(x[2 * np + 1][0], x[2 * np + 1][1], x[2 * np + 1][2], x[2 * np + 1][3],
                             qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b2, b3);
                }
            }

            // Logits with the contract's rounding points; invisible entries are -inf. Branch-free
            // and specialized on the bias, so the loop schedules as straight-line code.
            float top[2]      = {-INFINITY, -INFINITY};
            const auto logits = [&](auto biased) {
#pragma unroll
                for (int nt = 0; nt < kFlashKeys / 8; ++nt)
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int i   = e / 2;
                        const int r   = g4 + 8 * i;
                        const int col = nt * 8 + q4 * 2 + e % 2;
                        const int s   = s0 + col;
                        float logit = round_bf16(__fmul_rn(round_bf16(x[nt][e]), op.params.scale));
                        if constexpr (decltype(biased)::value)
                            logit = round_bf16(
                                __fadd_rn(logit, __bfloat162float(bias[r][col - r + 15])));
                        x[nt][e] = live[i] && s >= lo[i] && s <= position[i] ? logit : -INFINITY;
                        top[i]   = fmaxf(top[i], x[nt][e]);
                    }
            };
            if (with_bias)
                logits(std::true_type{});
            else
                logits(std::false_type{});

            // Online softmax: new running max, rescale factor, exponentials as FP16 MMA operands.
            float shift[2], alpha[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                top[i]            = fmaxf(top[i], __shfl_xor_sync(0xffffffffU, top[i], 1));
                top[i]            = fmaxf(top[i], __shfl_xor_sync(0xffffffffU, top[i], 2));
                const float fresh = fmaxf(m[i], top[i]);
                const float used  = fresh == -INFINITY ? 0.0f : fresh;
                alpha[i]          = exp2_approx((m[i] - used) * kLog2e);
                m[i]              = fresh;
                shift[i]          = used * kLog2e;
                l[i] *= alpha[i];
            }
            unsigned pa[kFlashKeys / 16][4];
#pragma unroll
            for (int kc = 0; kc < kFlashKeys / 16; ++kc) {
                float p[2][4];
#pragma unroll
                for (int half = 0; half < 2; ++half)
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float value =
                            exp2_approx(__fmaf_rn(x[2 * kc + half][e], kLog2e, -shift[e / 2]));
                        p[half][e] = value;
                        l[e / 2] += value;
                    }
                pa[kc][0] = pack_f16x2(p[0][0], p[0][1]);
                pa[kc][1] = pack_f16x2(p[0][2], p[0][3]);
                pa[kc][2] = pack_f16x2(p[1][0], p[1][1]);
                pa[kc][3] = pack_f16x2(p[1][2], p[1][3]);
            }

            // o = alpha o + p . v: the tile's product accumulates in FP16 (full tensor rate on
            // GeForce Blackwell; |sum| <= 64 max|v|), then joins the FP32 o. Half of D at a time.
            const unsigned v_base = shared_address(&sh.v[buffer][0][0]);
#pragma unroll
            for (int part = 0; part < 2; ++part) {
                if (part * kFlashMaxDim / 2 >= D) break;
                unsigned acc[kFlashMaxDim / 16][2] = {};
#pragma unroll
                for (int kc = 0; kc < kFlashKeys / 16; ++kc)
#pragma unroll
                    for (int np = 0; np < kFlashMaxDim / 32; ++np) {
                        const int d0 = part * kFlashMaxDim / 2 + np * 16;
                        if (d0 >= D) break;
                        const int key = kc * 16 + ((lane / 8) % 2) * 8 + lane % 8;
                        const int d   = d0 + (lane / 16) * 8;
                        unsigned b0, b1, b2, b3;
                        ldmatrix_x4_t(b0, b1, b2, b3, v_base + (key * kFlashRow + d) * 2);
                        mma_f16_hacc(acc[2 * np][0], acc[2 * np][1], pa[kc][0], pa[kc][1],
                                     pa[kc][2], pa[kc][3], b0, b1);
                        mma_f16_hacc(acc[2 * np + 1][0], acc[2 * np + 1][1], pa[kc][0], pa[kc][1],
                                     pa[kc][2], pa[kc][3], b2, b3);
                    }
#pragma unroll
                for (int n = 0; n < kFlashMaxDim / 16; ++n) {
                    const int nd    = part * kFlashMaxDim / 16 + n;
                    const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&acc[n][0]));
                    const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&acc[n][1]));
                    o[nd][0]        = __fmaf_rn(o[nd][0], alpha[0], lo.x);
                    o[nd][1]        = __fmaf_rn(o[nd][1], alpha[0], lo.y);
                    o[nd][2]        = __fmaf_rn(o[nd][2], alpha[1], hi.x);
                    o[nd][3]        = __fmaf_rn(o[nd][3], alpha[1], hi.y);
                }
            }
        }
        __syncthreads();
    }
    if (!warp_live) return;

    // Sink and normalizer: o / (sum exp(x - top) + exp(sink - top)).
    float scale[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        l[i] += __shfl_xor_sync(0xffffffffU, l[i], 1);
        l[i] += __shfl_xor_sync(0xffffffffU, l[i], 2);
        const float top   = fmaxf(m[i], sink[i]);
        const float carry = m[i] == -INFINITY ? 0.0f : expf(m[i] - top);
        scale[i]          = __fdiv_rn(carry, __fmaf_rn(l[i], carry, expf(sink[i] - top)));
    }
#pragma unroll
    for (int nd = 0; nd < kFlashMaxDim / 8; ++nd) {
        if (nd * 8 >= D) break;
        for (int i = 0; i < 2; ++i) {
            if (!live[i]) continue;
            const std::int64_t at = query_base(i) + nd * 8 + q4 * 2;
            op.out[at]            = output_value(op, at, o[nd][2 * i] * scale[i]);
            op.out[at + 1]        = output_value(op, at + 1, o[nd][2 * i + 1] * scale[i]);
        }
    }
}

// Sequential route: the append runs after the attention kernel on the same stream.
template <class Cache>
__global__ void append_kernel(RelativeSinkOperands op, Cache cache) {
    append_token(op, cache, blockIdx.x, 0, op.kv_heads);
}

template <int HB, int E, class Cache>
void launch_routes(const RelativeSinkOperands& op, const Cache& cache, cudaStream_t stream) {
    constexpr int kThreads = kKeyWarps * kWarp;
    const int group        = op.query_heads / op.kv_heads;
    const int slices       = op.kv_heads * ((group + HB - 1) / HB);
    const int tokens       = op.tokens * op.batch;
    if constexpr (E == 4) {
        if (group <= HB && decode_route(cache, op.tokens)) {
            constexpr int kBytes = sizeof(DecodeShared<HB>);
            static FuncAttrPerDevice attribute;
            attribute.ensure(decode_kernel<HB, Cache>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kBytes);
            launch_maybe_dependent(true, dim3(kDecodeBlocks * op.kv_heads, tokens),
                                   dim3(kDecodeThreads<HB>), kBytes, stream,
                                   decode_kernel<HB, Cache>, op, cache);
            return;
        }
    }
    const int items = 2 * op.query_heads + 2 * op.kv_heads;
    launch_maybe_dependent(op.tokens <= kDependentLaunchMaxTokens,
                           dim3(tokens, (items + kPrologueWarps - 1) / kPrologueWarps),
                           dim3(kPrologueWarps * kWarp), 0, stream, prologue_kernel, op);
    if (op.tokens <= kRelativeSinkSplitMaxTokens) {
        // Chunk slots: as many blocks per call as at one token, so short contexts do not pay
        // for thousands of blocks that find no chunk.
        const dim3 grid(kRelativeSinkMaxChunks / min(op.tokens, kRelativeSinkSplitMaxTokens),
                        slices, tokens);
        if constexpr (std::is_same_v<Cache, PageAccess>) {
            launch_maybe_dependent(true, grid, dim3(kWideThreads<HB>), 0, stream,
                                   split_online_kernel<HB, E, Cache>, op, cache);
            if (E == 4 && op.tokens >= 2 && op.tokens <= kColumnsMaxTokens) {
                constexpr int kBytes = sizeof(ColumnsShared<HB>);
                static FuncAttrPerDevice attribute;
                attribute.ensure(split_columns_kernel<HB>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, kBytes);
                launch_maybe_dependent(true, dim3(kRelativeSinkMaxChunks, slices, op.batch),
                                       dim3(kWideThreads<HB>), kBytes, stream,
                                       split_columns_kernel<HB>, op, cache);
            }
            launch_maybe_dependent(true, dim3(tokens, op.query_heads),
                                   dim3(static_cast<unsigned>(min(op.head_dim, 256))), 0, stream,
                                   split_online_finish_kernel<Cache>, op, cache);
        } else {
            launch_maybe_dependent(true, grid, dim3(kWideThreads<HB>), 0, stream,
                                   split_stats_kernel<HB, E, Cache>, op, cache);
            launch_maybe_dependent(true, grid, dim3(kWideThreads<HB>), 0, stream,
                                   split_values_kernel<HB, E, Cache>, op, cache);
            launch_maybe_dependent(true, dim3(tokens, op.query_heads),
                                   dim3(static_cast<unsigned>(min(op.head_dim, 256))), 0, stream,
                                   split_finish_kernel<Cache>, op, cache);
        }
    } else {
        if (op.head_dim <= kFlashMaxDim) {
            constexpr int kBytes = sizeof(FlashShared);
            static FuncAttrPerDevice attribute;
            attribute.ensure(flash_kernel<Cache>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kBytes);
            const int per_g = (group + kFlashHeads - 1) / kFlashHeads;
            const int tiles = (op.tokens + kFlashPositions - 1) / kFlashPositions;
            flash_kernel<Cache><<<dim3(tiles * op.batch, op.kv_heads * per_g),
                                  4 * kWarp * min(group, kFlashHeads), kBytes, stream>>>(op, cache);
        } else {
            sequential_kernel<HB, E, Cache>
                <<<dim3(tokens, slices), kThreads, 0, stream>>>(op, cache);
        }
        CUDA_CHECK(cudaGetLastError());
        append_kernel<Cache><<<tokens, 256, 0, stream>>>(op, cache);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Heads per block follow the GQA group (3, else 4); elements per lane follow D (<= 128, <= 256).
// Neither changes any per-head or per-key operation, so every instance gives the same bits.
template <class Cache>
void launch_all(const RelativeSinkOperands& op, const Cache& cache, cudaStream_t stream) {
    const bool triple = op.query_heads / op.kv_heads == 3;
    const bool narrow = op.head_dim <= 128;
    if (triple && narrow)
        launch_routes<3, 4>(op, cache, stream);
    else if (triple)
        launch_routes<3, 8>(op, cache, stream);
    else if (narrow)
        launch_routes<4, 4>(op, cache, stream);
    else
        launch_routes<4, 8>(op, cache, stream);
}

} // namespace

void relative_sink_attention_launch(const RelativeSinkOperands& operands,
                                    const RelativeSinkRing& ring, cudaStream_t stream) {
    launch_all(operands, RingAccess{ring}, stream);
}

void relative_sink_attention_launch(const RelativeSinkOperands& operands,
                                    const RelativeSinkPages& pages, cudaStream_t stream) {
    launch_all(operands, PageAccess{pages}, stream);
}

} // namespace ninfer::ops::detail
