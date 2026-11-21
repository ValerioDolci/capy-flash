#include "ops/relative_sink_attention/launch.h"

#include "core/device.h"
#include "ops/common/pdl_launch.cuh"
#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#include <cmath>
#include <cstdint>

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
__device__ __forceinline__ __nv_bfloat16 output_value(const RelativeSinkOperands& op,
                                                      std::int64_t at, float o) {
    if (op.gate != nullptr) o = round_bf16(o) * sigmoid(load(op.gate[at]));
    return __float2bfloat16_rn(o);
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

// Blocks (token, slice); each warp prepares one whole head vector of q, r, k or v.
__global__ void prologue_kernel(RelativeSinkOperands op) {
    pdl_enter();
    const int token = blockIdx.x;
    const int lane  = threadIdx.x % kWarp;
    const int warp  = threadIdx.x / kWarp;
    const int D     = op.head_dim;
    const int Hq    = op.query_heads;
    const int Hkv   = op.kv_heads;
    const int dr    = op.params.relative_dim;
    const int p     = op.positions[token];
    const float tau = __fadd_rn(
        1.0f, __fmul_rn(op.params.log_scale_alpha,
                        logf(fmaxf(__fdiv_rn(static_cast<float>(p + 1), op.params.log_scale_floor),
                                   1.0f))));
    const int items = 2 * Hq + 2 * Hkv;
    for (int item = warp + kPrologueWarps * blockIdx.y; item < items;
         item += kPrologueWarps * gridDim.y) {
        const int kind = item < Hq ? 0 : item < 2 * Hq ? 1 : item < 2 * Hq + Hkv ? 2 : 3;
        const int head = kind == 0   ? item
                         : kind == 1 ? item - Hq
                         : kind == 2 ? item - 2 * Hq
                                     : item - 2 * Hq - Hkv;
        const int n    = kind == 1 ? dr : D;
        const std::int64_t offset =
            static_cast<std::int64_t>(n) *
            (head + static_cast<std::int64_t>(kind <= 1 ? Hq : Hkv) * token);
        const __nv_bfloat16* source = kind == 0   ? op.q + offset
                                      : kind == 1 ? op.r + offset
                                      : kind == 2 ? op.k + offset
                                                  : op.v + offset;
        const __nv_bfloat16* scale  = kind == 0 ? op.q_scale + static_cast<std::int64_t>(D) * head
                                      : kind == 1 ? op.r_scale + static_cast<std::int64_t>(dr) * head
                                                  : nullptr;
        float x[kMaxPerLane];
        float squares = 0.0f;
#pragma unroll
        for (int i = 0; i < kMaxPerLane; ++i) {
            const int d = lane + kWarp * i;
            x[i]        = 0.0f;
            if (d < n) {
                float value = load(source[d]);
                if (scale != nullptr) value = __fmul_rn(value, load(scale[d]));
                x[i] = value;
                squares += value * value;
            }
        }
        squares = warp_sum(squares);
        if (kind == 3) {
            const float inverse =
                rsqrtf(__fdiv_rn(squares, static_cast<float>(n)) + op.params.value_epsilon);
#pragma unroll
            for (int i = 0; i < kMaxPerLane; ++i) {
                const int d = lane + kWarp * i;
                if (d < n) op.v_prepared[offset + d] = __float2half_rn(round_bf16(x[i] * inverse));
            }
            continue;
        }
        const float threshold = kind == 1 ? op.params.relative_clip : op.params.qk_clip;
        const float factor =
            fminf(__fdiv_rn(threshold, sqrtf(squares) + op.params.clip_epsilon), 1.0f);
        __nv_bfloat16* target = kind == 0   ? op.q_prepared + offset
                                : kind == 1 ? op.r_prepared + offset
                                            : op.k_prepared + offset;
#pragma unroll
        for (int i = 0; i < kMaxPerLane; ++i) {
            const int d = lane + kWarp * i;
            if (d >= n) continue;
            float y = round_bf16(x[i] * factor);
            if (kind <= 1) y = round_bf16(y * tau);
            target[d] = __float2bfloat16_rn(y);
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
// Cluster route: blocks per cluster, chunks per block and logits one block keeps between passes.
constexpr int kClusterBlocks    = 8;
constexpr int kClusterChunks    = kRelativeSinkMaxChunks / kClusterBlocks;
constexpr int kClusterStoreSize = 6144; // logits of all heads of a block

template <int HB>
constexpr int kClusterStoredKeys = kClusterStoreSize / HB;

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
#pragma unroll
        for (int u = 0; u < kVectors; ++u) {
            if (d0 + u * 8 >= op.head_dim) break;
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
    const int distance = query.p - s;
    const int R        = op.params.relative_extent;
    float bias[HB]     = {};
    if (distance < R) {
        // Every column element is requested before any is used.
        float column[kWarp];
#pragma unroll
        for (int j = 0; j < kWarp; ++j)
            column[j] = j < op.params.relative_dim
                            ? load(op.rel_proj[distance + static_cast<std::int64_t>(R) * j])
                            : 0.0f;
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
        if (distance < R) score = round_bf16(__fadd_rn(score, round_bf16(bias[hh])));
        logit[hh] = score;
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
// With `store`, the logit of the chunk's key begin + i for head hh is kept (exactly, it is a BF16
// value) at store[hh * kClusterStoredKeys<HB> + i].
template <int HB, int E, class Cache>
__device__ ChunkStats<HB> chunk_stats(const RelativeSinkOperands& op, const Cache& cache,
                                      const Query& query, const QueryState<HB, E>& state, int c,
                                      __nv_bfloat16* store = nullptr) {
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
        if (store != nullptr)
#pragma unroll
            for (int hh = 0; hh < HB; ++hh)
                store[hh * kClusterStoredKeys<HB> + s - begin] = __float2bfloat16_rn(x[hh]);
#pragma unroll
        for (int hh = 0; hh < HB; ++hh) {
            if (x[hh] > m[hh]) {
                z[hh] = __fadd_rn(__fmul_rn(z[hh], expf(m[hh] - x[hh])), 1.0f);
                m[hh] = x[hh];
            } else {
                z[hh] = __fadd_rn(z[hh], expf(x[hh] - m[hh]));
            }
        }
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

// Split route, stage 1: chunk statistics, one block per (chunk, KV head slice, token).
template <int HB, int E, class Cache>
__global__ void split_stats_kernel(RelativeSinkOperands op, Cache cache) {
    pdl_enter();
    const Query query = make_query<HB>(op, cache, blockIdx.z, blockIdx.y);
    const int c       = blockIdx.x;
    if (c >= query.chunks) return;
    __shared__ QueryState<HB, E> state;
    load_query_state(op, query, state);
    const ChunkStats<HB> stats = chunk_stats<HB, E>(op, cache, query, state, c);
    if (threadIdx.x < query.heads) {
        const int hh          = threadIdx.x;
        const std::int64_t at = stats_index(op, query.token, query.h0 + hh, c);
        op.chunk_max[at]      = stats.m[hh];
        op.chunk_sum[at]      = stats.z[hh];
    }
}

// Split route, stage 2: chunk value sums under the global normalization.
template <int HB, int E, class Cache>
__global__ void split_values_kernel(RelativeSinkOperands op, Cache cache) {
    pdl_enter();
    __shared__ float result[HB * kWarp * E];
    const Query query = make_query<HB>(op, cache, blockIdx.z, blockIdx.y);
    const int c       = blockIdx.x;
    if (c >= query.chunks) return;
    __shared__ QueryState<HB, E> state;
    load_query_state(op, query, state);
    float top[HB], norm[HB];
#pragma unroll
    for (int hh = 0; hh < HB; ++hh) {
        const int h              = query.h0 + min(hh, query.heads - 1);
        const std::int64_t first = stats_index(op, query.token, h, 0);
        combine_stats(load(op.sink[h + op.query_heads * query.token]), op.chunk_max + first,
                      op.chunk_sum + first, query.chunks, 1, top[hh], norm[hh]);
    }
    chunk_values<HB, E>(op, cache, query, state, c, top, norm, result);
    const int D = op.head_dim;
    for (int i = threadIdx.x; i < query.heads * D; i += blockDim.x)
        op.chunk_out[stats_index(op, query.token, query.h0 + i / D, c) * D + i % D] = result[i];
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

// Cluster route (see cluster_route; D <= 128, GQA group <= 4): one cluster of kClusterBlocks
// blocks per (token, KV head) in one launch. Block b takes the chunks b, b + 8, ...
// and evaluates them with the split route's functions in the split route's order, so both routes
// give the same bits; chunk statistics and chunk value sums travel through distributed shared
// memory instead of workspace and kernel boundaries. Logits stay in shared memory between the two
// passes (keys are read once) while a block's keys fit its store, and every block
// prefetches its key and value rows to L2 before the first pass.
template <int HB>
struct ClusterShared {
    float chunk_m[kClusterChunks][HB];
    float chunk_z[kClusterChunks][HB];
    float all_m[kRelativeSinkMaxChunks][HB];
    float all_z[kRelativeSinkMaxChunks][HB];
    float chunk_out[kClusterChunks][HB][kWarp * 4];
    float partial[kKeyWarps][HB][kWarp * 4];
    __nv_bfloat16 logits[HB][kClusterStoredKeys<HB>];
};

__device__ __forceinline__ void prefetch_row(const void* row, int bytes) {
    for (int offset = 0; offset < bytes; offset += 128)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(static_cast<const char*>(row) + offset));
}

// sum_s bf16(p_s) v_s over one chunk, as chunk_values (same products in the same order per output
// element), with value rows read straight into registers: lane l holds the elements 64e + 2l and
// 64e + 2l + 1 of its warp's 32 rows. Logits come from `stored` (see chunk_stats) or are
// recomputed. The chunk sum lands in out[hh * 128 + d].
template <int HB, class Cache>
__device__ void cluster_chunk_values(const RelativeSinkOperands& op, const Cache& cache,
                                     const Query& query, const QueryState<HB, 4>& state, int c,
                                     const float (&top)[HB], const float (&norm)[HB],
                                     const __nv_bfloat16* stored,
                                     float (&partial)[kKeyWarps][HB][kWarp * 4], float* out) {
    constexpr int kPairs      = 2; // pairs of value elements per lane and row (D <= 128)
    const int lane            = threadIdx.x % kWarp;
    const int warp            = threadIdx.x / kWarp;
    const int D               = op.head_dim;
    const int begin           = query.lo + c * query.chunk;
    const int end             = min(begin + query.chunk, query.p + 1);
    float acc[HB][2 * kPairs] = {};
    for (int base = begin + warp * kWarp; base < end; base += kChunkUnit) {
        const int count = min(kWarp, end - base);
        const int s     = base + lane;
        const auto own  = reinterpret_cast<unsigned long long>(
            s < end ? value_row(op, cache, query, s) : nullptr);
        __half2 v[kWarp][kPairs];
#pragma unroll
        for (int source = 0; source < kWarp; ++source) {
            const auto* row =
                reinterpret_cast<const __half*>(__shfl_sync(0xffffffffU, own, source));
#pragma unroll
            for (int e = 0; e < kPairs; ++e) {
                const int d  = 64 * e + 2 * lane;
                v[source][e] = source < count && d < D ? *reinterpret_cast<const __half2*>(row + d)
                                                       : __floats2half2_rn(0.0f, 0.0f);
            }
        }
        float probability[HB] = {};
        if (s < end) {
            float x[HB];
            if (stored != nullptr) {
#pragma unroll
                for (int hh = 0; hh < HB; ++hh)
                    x[hh] = load(stored[hh * kClusterStoredKeys<HB> + s - begin]);
            } else {
                lane_logits<HB, 4>(op, cache, query, state, s, x);
            }
#pragma unroll
            for (int hh = 0; hh < HB; ++hh)
                probability[hh] = round_bf16(__fdiv_rn(expf(x[hh] - top[hh]), norm[hh]));
        }
#pragma unroll
        for (int source = 0; source < kWarp; ++source) {
#pragma unroll
            for (int hh = 0; hh < HB; ++hh) {
                const float pk = __shfl_sync(0xffffffffU, probability[hh], source);
                if (source < count)
#pragma unroll
                    for (int e = 0; e < kPairs; ++e) {
                        const float2 value = __half22float2(v[source][e]);
                        acc[hh][2 * e]     = __fmaf_rn(pk, value.x, acc[hh][2 * e]);
                        acc[hh][2 * e + 1] = __fmaf_rn(pk, value.y, acc[hh][2 * e + 1]);
                    }
            }
        }
    }
#pragma unroll
    for (int hh = 0; hh < HB; ++hh)
#pragma unroll
        for (int e = 0; e < kPairs; ++e) {
            const int d = 64 * e + 2 * lane;
            if (d < D) {
                partial[warp][hh][d]     = acc[hh][2 * e];
                partial[warp][hh][d + 1] = acc[hh][2 * e + 1];
            }
        }
    __syncthreads();
    for (int i = threadIdx.x; i < query.heads * D; i += blockDim.x) {
        const int hh = i / D, d = i % D;
        float sum = partial[0][hh][d];
        for (int w = 1; w < kKeyWarps; ++w) sum = __fadd_rn(sum, partial[w][hh][d]);
        out[hh * kWarp * 4 + d] = sum;
    }
    __syncthreads();
}

template <int HB, class Cache>
__global__ void __cluster_dims__(kClusterBlocks, 1, 1)
    cluster_kernel(RelativeSinkOperands op, Cache cache) {
    pdl_enter();
    namespace cg                    = cooperative_groups;
    const cg::cluster_group cluster = cg::this_cluster();
    __shared__ QueryState<HB, 4> state;
    __shared__ ClusterShared<HB> sh;
    const int rank    = static_cast<int>(cluster.block_rank());
    const Query query = make_query<HB>(op, cache, blockIdx.y, blockIdx.x / kClusterBlocks);
    const int mine =
        query.chunks > rank ? (query.chunks - rank + kClusterBlocks - 1) / kClusterBlocks : 0;
    const bool keep     = mine * query.chunk <= kClusterStoredKeys<HB>;
    const auto chunk_of = [&](int i) { return rank + kClusterBlocks * i; };

    // Prefetch only what the store holds: at most kClusterStoredKeys rows per block go to L2.
    for (int i = 0; keep && i < mine; ++i) {
        const int begin = query.lo + chunk_of(i) * query.chunk;
        const int end   = min(begin + query.chunk, query.p + 1);
        for (int s = begin + threadIdx.x; s < end; s += kChunkUnit) {
            prefetch_row(key_row(op, cache, query, s), op.head_dim * 2);
            prefetch_row(value_row(op, cache, query, s), op.head_dim * 2);
        }
    }
    if (mine > 0) load_query_state(op, query, state);
    for (int i = 0; i < mine; ++i) {
        const ChunkStats<HB> stats = chunk_stats<HB, 4>(
            op, cache, query, state, chunk_of(i), keep ? &sh.logits[0][i * query.chunk] : nullptr);
        if (threadIdx.x == 0)
#pragma unroll
            for (int hh = 0; hh < HB; ++hh) {
                sh.chunk_m[i][hh] = stats.m[hh];
                sh.chunk_z[i][hh] = stats.z[hh];
            }
    }
    cluster.sync();

    if (mine > 0) {
        for (int i = threadIdx.x; i < query.chunks * HB; i += blockDim.x) {
            const int c = i / HB, hh = i % HB;
            const ClusterShared<HB>* owner = cluster.map_shared_rank(&sh, c % kClusterBlocks);
            sh.all_m[c][hh]                = owner->chunk_m[c / kClusterBlocks][hh];
            sh.all_z[c][hh]                = owner->chunk_z[c / kClusterBlocks][hh];
        }
        __syncthreads();
        float top[HB], norm[HB];
#pragma unroll
        for (int hh = 0; hh < HB; ++hh) {
            const int h = query.h0 + min(hh, query.heads - 1);
            combine_stats(load(op.sink[h + op.query_heads * query.token]), &sh.all_m[0][hh],
                          &sh.all_z[0][hh], query.chunks, HB, top[hh], norm[hh]);
        }
        for (int i = 0; i < mine; ++i)
            cluster_chunk_values<HB>(op, cache, query, state, chunk_of(i), top, norm,
                                     keep ? &sh.logits[0][i * query.chunk] : nullptr, sh.partial,
                                     &sh.chunk_out[i][0][0]);
    }
    cluster.sync();

    // Chunk sums added in chunk order; the output elements are spread over the cluster.
    const int D = op.head_dim;
    for (int i = threadIdx.x + rank * blockDim.x; i < query.heads * D;
         i += blockDim.x * kClusterBlocks) {
        const int hh = i / D, d = i % D;
        float sum = cluster.map_shared_rank(&sh, 0)->chunk_out[0][hh][d];
        for (int c = 1; c < query.chunks; ++c)
            sum = __fadd_rn(sum, cluster.map_shared_rank(&sh, c % kClusterBlocks)
                                     ->chunk_out[c / kClusterBlocks][hh][d]);
        const std::int64_t at =
            static_cast<std::int64_t>(D) * (query.h0 + hh + op.query_heads * query.token) + d;
        op.out[at] = output_value(op, at, sum);
    }
    // Every block of this cluster has read the cache (one token per call: no other cluster reads
    // this KV head's rows).
    if (rank == kClusterBlocks - 1 && query.h0 == query.g * (op.query_heads / op.kv_heads))
        append_token(op, cache, query.token, query.g, query.g + 1);
    cluster.sync();
}

// Where the cluster route is faster than the split route (measured, 12/4, cold L2): one token on
// a ring. With more tokens it loses to occupancy (a cluster's eight blocks stay resident through
// both passes), and a paged call may see far more keys than eight blocks walk quickly, which the
// host cannot know when the call is captured.
bool cluster_route(const RingAccess&, int tokens) { return tokens == 1; }

bool cluster_route(const PageAccess&, int) { return false; }

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

// Tiled route (T > 16, D <= 128): tensor-core MMA for q.k, the bias table r'.P and p.v. One block
// per (16 query columns, KV head slice), one warp per query head; the warps share key tiles of
// 32 keys in shared memory. Two passes as everywhere: per-row max and normalizer, then
// bf16(p) . v with p (exactly BF16) as the MMA operand. Value rows are observed as BF16 copies of
// their FP16 values (exact for |v| >= 2^-14). Results agree with the other routes within the
// oracle criterion, not bit for bit; equal calls give equal bits.
constexpr int kTileQueries = 16;
constexpr int kTileKeys    = 32;
constexpr int kTileMaxDim  = 128;
constexpr int kTileWarps   = 4;
constexpr int kBiasColumns = kTileKeys + kTileQueries; // distances seen by one tile, rounded up

struct TileShared {
    __nv_bfloat16 k[kTileKeys][kTileMaxDim + 8];
    __nv_bfloat16 vt[kTileMaxDim][kTileKeys + 8];
    float bias[kTileWarps][kTileQueries][kBiasColumns];
};

__device__ __forceinline__ unsigned pack_bf16(float low, float high) {
    const __nv_bfloat162 pair = __floats2bfloat162_rn(low, high);
    return *reinterpret_cast<const unsigned*>(&pair);
}

__device__ __forceinline__ unsigned load_pair(const __nv_bfloat16* at) {
    return *reinterpret_cast<const unsigned*>(at);
}

template <class Cache>
__device__ void load_key_tile(const RelativeSinkOperands& op, const Cache& cache, int row, int g,
                              int b, int L, int first, int last, int s0, TileShared& sh,
                              bool values) {
    const int D       = op.head_dim;
    const int vectors = D / 8;
    for (int i = threadIdx.x; i < kTileKeys * vectors; i += blockDim.x) {
        const int key = i / vectors, d = (i % vectors) * 8;
        const int s     = s0 + key;
        const bool live = s >= first && s <= last;
        const std::int64_t block_row =
            static_cast<std::int64_t>(D) *
            (g + static_cast<std::int64_t>(op.kv_heads) * (s - L + op.tokens * b));
        if (live) {
            const __nv_bfloat16* k = s < L ? cache.key(row, g, s) : op.k_prepared + block_row;
            __pipeline_memcpy_async(&sh.k[key][d], k + d, 16);
        } else {
            *reinterpret_cast<uint4*>(&sh.k[key][d]) = make_uint4(0, 0, 0, 0);
        }
        if (values) {
            uint4 raw = make_uint4(0, 0, 0, 0);
            if (live) {
                const __half* v = s < L ? cache.value(row, g, s) : op.v_prepared + block_row;
                raw             = *reinterpret_cast<const uint4*>(v + d);
            }
            const auto* halves = reinterpret_cast<const __half*>(&raw);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                sh.vt[d + j][key] = __float2bfloat16_rn(__half2float(halves[j]));
        }
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
}

// bias[r][c] = bf16(r'_r . P[:, dmin + c]) for the tile's distances, zero outside [0, R).
__device__ void bias_table(const RelativeSinkOperands& op, const unsigned (&ra)[2][4], int dmin,
                           float (&bias)[kTileQueries][kBiasColumns]) {
    const int lane = threadIdx.x % kWarp;
    const int g4 = lane / 4, q4 = lane % 4;
    const int R  = op.params.relative_extent;
    const int dr = op.params.relative_dim;
    const auto P = [&](int distance, int j) {
        return distance >= 0 && distance < R && j < dr
                   ? load(op.rel_proj[distance + static_cast<std::int64_t>(R) * j])
                   : 0.0f;
    };
#pragma unroll
    for (int nt = 0; nt < kBiasColumns / 8; ++nt) {
        float c[4]         = {};
        const int distance = dmin + nt * 8 + g4;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            if (ks * 16 >= dr) break;
            const int j = ks * 16 + q4 * 2;
            mma_bf16(c[0], c[1], c[2], c[3], ra[ks][0], ra[ks][1], ra[ks][2], ra[ks][3],
                     pack_bf16(P(distance, j), P(distance, j + 1)),
                     pack_bf16(P(distance, j + 8), P(distance, j + 9)));
        }
        const int col         = nt * 8 + q4 * 2;
        bias[g4][col]         = round_bf16(c[0]);
        bias[g4][col + 1]     = round_bf16(c[1]);
        bias[g4 + 8][col]     = round_bf16(c[2]);
        bias[g4 + 8][col + 1] = round_bf16(c[3]);
    }
    __syncwarp();
}

template <class Cache>
__global__ void tiled_kernel(RelativeSinkOperands op, Cache cache) {
    __shared__ __align__(16) TileShared sh;
    const int lane = threadIdx.x % kWarp;
    const int warp = threadIdx.x / kWarp;
    const int g4 = lane / 4, q4 = lane % 4;
    const int T     = op.tokens;
    const int D     = op.head_dim;
    const int Hq    = op.query_heads;
    const int tiles = (T + kTileQueries - 1) / kTileQueries;
    const int b     = blockIdx.x / tiles;
    const int t0    = (blockIdx.x % tiles) * kTileQueries;
    const int group = Hq / op.kv_heads;
    const int per_g = (group + kTileWarps - 1) / kTileWarps;
    const int g     = blockIdx.y / per_g;
    const int first = (blockIdx.y % per_g) * kTileWarps;
    const int heads = min(kTileWarps, group - first);
    const int h     = g * group + first + min(warp, heads - 1);
    const int rows  = min(kTileQueries, T - t0);
    const int L     = op.positions[T * b];
    const int row   = op.rows[b];
    const int p0    = L + t0;
    const int last  = p0 + rows - 1;
    const int lo0   = cache.first_visible(p0);

    // This thread's two query rows of the MMA fragments.
    int position[2], lo[2];
    bool live[2];
    float sink[2];
    for (int i = 0; i < 2; ++i) {
        const int r = g4 + 8 * i;
        live[i]     = r < rows;
        position[i] = p0 + r;
        lo[i]       = cache.first_visible(position[i]);
        sink[i]     = live[i] ? load(op.sink[h + Hq * (t0 + r + T * b)]) : 0.0f;
    }
    const auto query_base = [&](int i) {
        return static_cast<std::int64_t>(D) * (h + Hq * (t0 + g4 + 8 * i + T * b));
    };
    unsigned qa[kTileMaxDim / 16][4];
#pragma unroll
    for (int kk = 0; kk < kTileMaxDim / 16; ++kk) {
        const int d   = kk * 16 + q4 * 2;
        const bool on = d < D;
        qa[kk][0]     = on && live[0] ? load_pair(op.q_prepared + query_base(0) + d) : 0U;
        qa[kk][1]     = on && live[1] ? load_pair(op.q_prepared + query_base(1) + d) : 0U;
        qa[kk][2]     = on && live[0] ? load_pair(op.q_prepared + query_base(0) + d + 8) : 0U;
        qa[kk][3]     = on && live[1] ? load_pair(op.q_prepared + query_base(1) + d + 8) : 0U;
    }
    unsigned ra[2][4];
    {
        const int dr   = op.params.relative_dim;
        const auto rel = [&](int i, int j) {
            return live[i] && j < dr ? load(op.r_prepared[static_cast<std::int64_t>(dr) *
                                                              (h + Hq * (t0 + g4 + 8 * i + T * b)) +
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

    // Logits of one key tile in MMA accumulator layout; invisible entries are -inf.
    const auto tile_logits = [&](int s0, float (&x)[kTileKeys / 8][4]) {
        const int dmin = p0 - (s0 + kTileKeys - 1);
        bias_table(op, ra, dmin, sh.bias[warp]);
#pragma unroll
        for (int nt = 0; nt < kTileKeys / 8; ++nt) {
            float c[4] = {};
#pragma unroll
            for (int kk = 0; kk < kTileMaxDim / 16; ++kk) {
                if (kk * 16 >= D) break;
                const int d = kk * 16 + q4 * 2;
                mma_bf16(c[0], c[1], c[2], c[3], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3],
                         load_pair(&sh.k[nt * 8 + g4][d]), load_pair(&sh.k[nt * 8 + g4][d + 8]));
            }
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i   = e / 2;
                const int col = nt * 8 + q4 * 2 + e % 2;
                const int s   = s0 + col;
                const int r   = g4 + 8 * i;
                float logit   = -INFINITY;
                if (live[i] && s >= lo[i] && s <= position[i]) {
                    logit = round_bf16(__fmul_rn(round_bf16(c[e]), op.params.scale));
                    if (position[i] - s < op.params.relative_extent)
                        logit =
                            round_bf16(__fadd_rn(logit, sh.bias[warp][r][r - col + kTileKeys - 1]));
                }
                x[nt][e] = logit;
            }
        }
    };

    // Pass 1: per-row max and sum of exponentials.
    float m[2] = {-INFINITY, -INFINITY}, z[2] = {0.0f, 0.0f};
    for (int s0 = lo0; s0 <= last; s0 += kTileKeys) {
        load_key_tile(op, cache, row, g, b, L, lo0, last, s0, sh, false);
        float x[kTileKeys / 8][4];
        tile_logits(s0, x);
#pragma unroll
        for (int nt = 0; nt < kTileKeys / 8; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = e / 2;
                if (x[nt][e] == -INFINITY) continue;
                if (x[nt][e] > m[i]) {
                    z[i] = __fadd_rn(__fmul_rn(z[i], expf(m[i] - x[nt][e])), 1.0f);
                    m[i] = x[nt][e];
                } else {
                    z[i] = __fadd_rn(z[i], expf(x[nt][e] - m[i]));
                }
            }
        __syncthreads();
    }
    float top[2], norm[2];
    for (int i = 0; i < 2; ++i) {
        for (int offset = 1; offset < 4; offset <<= 1)
            merge_stats(m[i], z[i], __shfl_xor_sync(0xffffffffU, m[i], offset),
                        __shfl_xor_sync(0xffffffffU, z[i], offset));
        combine_stats(sink[i], &m[i], &z[i], 1, 1, top[i], norm[i]);
    }

    // Pass 2: o = sum bf16(p) v.
    float o[kTileMaxDim / 8][4] = {};
    for (int s0 = lo0; s0 <= last; s0 += kTileKeys) {
        load_key_tile(op, cache, row, g, b, L, lo0, last, s0, sh, true);
        float x[kTileKeys / 8][4];
        tile_logits(s0, x);
        unsigned pa[kTileKeys / 16][4];
#pragma unroll
        for (int kc = 0; kc < kTileKeys / 16; ++kc) {
            float p[2][4];
#pragma unroll
            for (int half = 0; half < 2; ++half)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float logit = x[2 * kc + half][e];
                    p[half][e]        = logit == -INFINITY
                                            ? 0.0f
                                            : round_bf16(__fdiv_rn(expf(logit - top[e / 2]), norm[e / 2]));
                }
            pa[kc][0] = pack_bf16(p[0][0], p[0][1]);
            pa[kc][1] = pack_bf16(p[0][2], p[0][3]);
            pa[kc][2] = pack_bf16(p[1][0], p[1][1]);
            pa[kc][3] = pack_bf16(p[1][2], p[1][3]);
        }
#pragma unroll
        for (int nd = 0; nd < kTileMaxDim / 8; ++nd) {
            if (nd * 8 >= D) break;
#pragma unroll
            for (int kc = 0; kc < kTileKeys / 16; ++kc)
                mma_bf16(o[nd][0], o[nd][1], o[nd][2], o[nd][3], pa[kc][0], pa[kc][1], pa[kc][2],
                         pa[kc][3], load_pair(&sh.vt[nd * 8 + g4][kc * 16 + q4 * 2]),
                         load_pair(&sh.vt[nd * 8 + g4][kc * 16 + 8 + q4 * 2]));
        }
        __syncthreads();
    }
    if (warp >= heads) return;
#pragma unroll
    for (int nd = 0; nd < kTileMaxDim / 8; ++nd) {
        if (nd * 8 >= D) break;
        for (int i = 0; i < 2; ++i) {
            if (!live[i]) continue;
            const std::int64_t at = query_base(i) + nd * 8 + q4 * 2;
            op.out[at]            = output_value(op, at, o[nd][2 * i]);
            op.out[at + 1]        = output_value(op, at + 1, o[nd][2 * i + 1]);
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
        if (group <= HB && cluster_route(cache, op.tokens)) {
            launch_maybe_dependent(true, dim3(kClusterBlocks * op.kv_heads, tokens), dim3(kThreads),
                                   0, stream, cluster_kernel<HB, Cache>, op, cache);
            return;
        }
    }
    if (op.tokens <= kRelativeSinkSplitMaxTokens) {
        const dim3 grid(kRelativeSinkMaxChunks, slices, tokens);
        launch_maybe_dependent(true, grid, dim3(kThreads), 0, stream,
                               split_stats_kernel<HB, E, Cache>, op, cache);
        launch_maybe_dependent(true, grid, dim3(kThreads), 0, stream,
                               split_values_kernel<HB, E, Cache>, op, cache);
        launch_maybe_dependent(true, dim3(tokens, op.query_heads),
                               dim3(static_cast<unsigned>(min(op.head_dim, 256))), 0, stream,
                               split_finish_kernel<Cache>, op, cache);
    } else {
        if (op.head_dim <= kTileMaxDim) {
            const int per_g = (group + kTileWarps - 1) / kTileWarps;
            const int tiles = (op.tokens + kTileQueries - 1) / kTileQueries;
            tiled_kernel<Cache><<<dim3(tiles * op.batch, op.kv_heads * per_g),
                                  kWarp * min(group, kTileWarps), 0, stream>>>(op, cache);
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
    const int items = 2 * op.query_heads + 2 * op.kv_heads;
    launch_maybe_dependent(
        op.tokens <= kDependentLaunchMaxTokens,
        dim3(op.tokens * op.batch, (items + kPrologueWarps - 1) / kPrologueWarps),
        dim3(kPrologueWarps * kWarp), 0, stream, prologue_kernel, op);
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
