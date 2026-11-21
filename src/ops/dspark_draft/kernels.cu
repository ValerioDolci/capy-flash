// Kernels of the DSpark drafter Ops (ninfer/ops/dspark_draft.h).
#include "core/device.h"
#include "ops/dspark_draft/launch.h"

#include <cuda_bf16.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kD    = 128;
constexpr int kHalf = kD / 2;

__device__ __forceinline__ float round_bf16(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ __forceinline__ float load(const __nv_bfloat16* p) { return __bfloat162float(*p); }

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) x += __shfl_xor_sync(0xFFFFFFFFU, x, offset);
    return x;
}

// Sum over the block (blockDim.x a multiple of 32, at most 1024); every thread gets the total.
__device__ float block_sum(float x, float* scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    x = warp_sum(x);
    __syncthreads();
    if (lane == 0) scratch[warp] = x;
    __syncthreads();
    float total = 0.0F;
    for (int w = 0; w < int(blockDim.x >> 5); ++w) total += scratch[w];
    return total;
}

// ---- feature_mix_rmsnorm ------------------------------------------------------------------------
__global__ void feature_mix_rmsnorm_kernel(const __nv_bfloat16* __restrict__ features,
                                           const __nv_bfloat16* __restrict__ logits, MixNorms norms,
                                           int hidden, int width, std::size_t block_stride,
                                           int states, int rows, float eps,
                                           const int* __restrict__ count,
                                           __nv_bfloat16* __restrict__ out,
                                           __nv_bfloat16* __restrict__ last) {
    extern __shared__ float mixed[];
    __shared__ float scratch[32];
    __shared__ float alpha[8];
    const int t = blockIdx.x, r = blockIdx.y;
    const int c          = count != nullptr ? *count : width;
    const bool final_row = r == rows - 1;
    if (final_row && (c <= 0 || t != c - 1)) return;
    if (threadIdx.x == 0) {
        float m = -FLT_MAX;
        for (int s = 0; s < states; ++s) m = fmaxf(m, load(logits + r * states + s));
        float e[8], total = 0.0F;
        for (int s = 0; s < states; ++s) {
            e[s] = expf(load(logits + r * states + s) - m);
            total += e[s];
        }
        for (int s = 0; s < states; ++s) alpha[s] = round_bf16(e[s] / total);
    }
    __syncthreads();
    float squares = 0.0F;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        float m = 0.0F;
        for (int s = 0; s < states; ++s) {
            m += load(features + std::size_t(s) * block_stride + std::size_t(t) * hidden + h) *
                 alpha[s];
        }
        mixed[h] = m;
        squares += m * m;
    }
    const float inv    = rsqrtf(block_sum(squares, scratch) / float(hidden) + eps);
    const auto* w      = static_cast<const __nv_bfloat16*>(norms.p[r]);
    __nv_bfloat16* dst = final_row ? last : out + (std::size_t(r) * width + t) * hidden;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        dst[h] = __float2bfloat16_rn(mixed[h] * inv * load(w + h));
    }
}

__global__ void consecutive_positions_kernel(const int* __restrict__ start,
                                             const int* __restrict__ minus, int* __restrict__ out,
                                             int tokens) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < tokens) out[t] = *start - (minus != nullptr ? *minus : 0) + t;
}

// ---- per-head norm + RoPE (64 threads: pair i, i + 64) ------------------------------------------
__device__ void norm_rope_head(const __nv_bfloat16* src, __nv_bfloat16* dst,
                               const __nv_bfloat16* weight, int position, const InvFrequencies& inv,
                               float eps, float* scratch) {
    const int i   = threadIdx.x;
    const float a = load(src + i), b = load(src + i + kHalf);
    const float r     = 1.0F / sqrtf(block_sum(a * a + b * b, scratch) / float(kD) + eps);
    const float n1    = round_bf16(a * r * load(weight + i));
    const float n2    = round_bf16(b * r * load(weight + i + kHalf));
    const float angle = __fmul_rn(float(position), inv.v[i]);
    float s = 0.0F, c = 0.0F;
    sincosf(angle, &s, &c);
    dst[i]         = __float2bfloat16_rn(__fsub_rn(__fmul_rn(n1, c), __fmul_rn(n2, s)));
    dst[i + kHalf] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(n2, c), __fmul_rn(n1, s)));
}

__global__ void head_rmsnorm_rope_kernel(__nv_bfloat16* x, int rows, int query_heads,
                                         const __nv_bfloat16* __restrict__ q_norm,
                                         const __nv_bfloat16* __restrict__ k_norm,
                                         const int* __restrict__ positions, InvFrequencies inv,
                                         float eps) {
    __shared__ float scratch[32];
    const int t = blockIdx.x, h = blockIdx.y;
    __nv_bfloat16* head = x + std::size_t(t) * rows + std::size_t(h) * kD;
    norm_rope_head(head, head, h < query_heads ? q_norm : k_norm, positions[t], inv, eps, scratch);
}

// ---- context K/V store
// ---------------------------------------------------------------------------
__global__ void context_kv_store_kernel(StoreLayers layers, int kv_heads, int width,
                                        const int* __restrict__ positions,
                                        const int* __restrict__ count, InvFrequencies inv,
                                        float eps) {
    __shared__ float scratch[32];
    const int j = blockIdx.x, l = blockIdx.y, head = blockIdx.z;
    const int c = *count;
    if (j >= c) return;
    const StoreLayer& layer = layers.l[l];
    const int p             = positions[j];
    if (p < positions[c - 1] + 1 - layer.capacity) return;
    const std::size_t slot   = std::size_t(p % layer.capacity);
    const int h              = head % kv_heads;
    const __nv_bfloat16* src = static_cast<const __nv_bfloat16*>(layer.raw) +
                               std::size_t(j) * 2 * kv_heads * kD + std::size_t(head) * kD;
    if (head < kv_heads) {
        norm_rope_head(src, static_cast<__nv_bfloat16*>(layer.k) + (slot * kv_heads + h) * kD,
                       static_cast<const __nv_bfloat16*>(layer.k_norm), p, inv, eps, scratch);
    } else {
        __nv_bfloat16* dst = static_cast<__nv_bfloat16*>(layer.v) + (slot * kv_heads + h) * kD;
        dst[threadIdx.x]   = src[threadIdx.x];
        dst[threadIdx.x + kHalf] = src[threadIdx.x + kHalf];
    }
}

// ---- block attention
// ----------------------------------------------------------------------------- Grid (Hkv,
// kChunks), 4 warps. Each lane owns four of the 128 dimensions; every warp walks its keys of the
// chunk with an online softmax for the Q = G * B queries of the KV head.
template <int G>
__global__ void __launch_bounds__(128)
    block_attention_partial_kernel(const __nv_bfloat16* __restrict__ qkv, int rows, int query_heads,
                                   int kv_heads, int block,
                                   const __nv_bfloat16* __restrict__ k_cache,
                                   const __nv_bfloat16* __restrict__ v_cache, int capacity,
                                   const int* __restrict__ frontier, int window, float scale,
                                   float* __restrict__ partial) {
    constexpr int Q = G * kBlockAttentionMaxBlock;
    __shared__ float q_s[Q][kD];
    __shared__ float m_s[4][Q], l_s[4][Q];
    __shared__ float acc_s[Q][kD];
    const int g = blockIdx.x, chunk = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int queries = G * block;
    for (int e = threadIdx.x; e < queries * kD; e += blockDim.x) {
        const int q = e / kD, d = e % kD;
        const int b = q / G, h = g * G + q % G;
        q_s[q][d] = load(qkv + std::size_t(b) * rows + std::size_t(h) * kD + d);
    }
    __syncthreads();
    const int P     = *frontier;
    const int lo    = window > 0 ? max(0, P - window) : 0;
    const int n_ctx = P - lo;
    const int total = n_ctx + block;
    const int len   = (total + gridDim.y - 1) / gridDim.y;
    const int first = chunk * len, end = min(total, first + len);
    float m[Q], l[Q], acc[Q][4];
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        m[q] = -INFINITY;
        l[q] = 0.0F;
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[q][i] = 0.0F;
    }
    for (int j = first + warp; j < end; j += 4) {
        const __nv_bfloat16 *kp, *vp;
        if (j < n_ctx) {
            const std::size_t slot = std::size_t((lo + j) % capacity);
            kp                     = k_cache + (slot * kv_heads + g) * kD;
            vp                     = v_cache + (slot * kv_heads + g) * kD;
        } else {
            const std::size_t column = std::size_t(j - n_ctx) * rows;
            kp                       = qkv + column + std::size_t(query_heads + g) * kD;
            vp                       = qkv + column + std::size_t(query_heads + kv_heads + g) * kD;
        }
        float k4[4], v4[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            k4[i] = load(kp + lane * 4 + i);
            v4[i] = load(vp + lane * 4 + i);
        }
#pragma unroll
        for (int q = 0; q < Q; ++q) {
            if (q >= queries) break;
            float dot = 0.0F;
#pragma unroll
            for (int i = 0; i < 4; ++i) dot = fmaf(q_s[q][lane * 4 + i], k4[i], dot);
            const float s      = warp_sum(dot) * scale;
            const float m_new  = fmaxf(m[q], s);
            const float corr   = expf(m[q] - m_new);
            const float weight = expf(s - m_new);
            l[q]               = l[q] * corr + weight;
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[q][i] = fmaf(acc[q][i], corr, weight * v4[i]);
            m[q] = m_new;
        }
    }
    if (lane == 0) {
        for (int q = 0; q < queries; ++q) {
            m_s[warp][q] = m[q];
            l_s[warp][q] = l[q];
        }
    }
    __syncthreads();
    // Warp 0's lanes rescale their accumulators first, then the others add theirs.
    for (int w = 0; w < 4; ++w) {
        if (warp == w) {
#pragma unroll
            for (int q = 0; q < Q; ++q) {
                if (q >= queries) break;
                float M = -INFINITY;
                for (int x = 0; x < 4; ++x) M = fmaxf(M, m_s[x][q]);
                const float f = M == -INFINITY ? 0.0F : expf(m[q] - M);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float value      = acc[q][i] * f;
                    acc_s[q][lane * 4 + i] = w == 0 ? value : acc_s[q][lane * 4 + i] + value;
                }
            }
        }
        __syncthreads();
    }
    const std::size_t base = (std::size_t(g) * gridDim.y + chunk) * Q;
    for (int q = warp; q < queries; q += 4) {
        float M = -INFINITY;
        for (int x = 0; x < 4; ++x) M = fmaxf(M, m_s[x][q]);
        float L = 0.0F;
        for (int x = 0; x < 4; ++x) L += M == -INFINITY ? 0.0F : l_s[x][q] * expf(m_s[x][q] - M);
        float* dst = partial + (base + q) * (kD + 2);
        if (lane == 0) {
            dst[0] = M;
            dst[1] = L;
        }
        for (int d = lane; d < kD; d += 32) dst[2 + d] = acc_s[q][d];
    }
}

template <int G>
__global__ void block_attention_combine_kernel(const float* __restrict__ partial, int chunks,
                                               int query_heads, int block,
                                               __nv_bfloat16* __restrict__ out) {
    constexpr int Q = G * kBlockAttentionMaxBlock;
    const int g = blockIdx.x, q = blockIdx.y, d = threadIdx.x;
    if (q >= G * block) return;
    float M = -INFINITY;
    for (int c = 0; c < chunks; ++c) {
        const float* p = partial + ((std::size_t(g) * chunks + c) * Q + q) * (kD + 2);
        if (p[1] > 0.0F) M = fmaxf(M, p[0]);
    }
    float L = 0.0F, A = 0.0F;
    for (int c = 0; c < chunks; ++c) {
        const float* p = partial + ((std::size_t(g) * chunks + c) * Q + q) * (kD + 2);
        if (p[1] > 0.0F) {
            const float f = expf(p[0] - M);
            L += p[1] * f;
            A += p[2 + d] * f;
        }
    }
    const int b = q / G, h = g * G + q % G;
    out[(std::size_t(b) * query_heads + h) * kD + d] = __float2bfloat16_rn(A / L);
}

// ---- add + rmsnorm
// --------------------------------------------------------------------------------
__global__ void add_rmsnorm_kernel(__nv_bfloat16* residual, const __nv_bfloat16* __restrict__ y,
                                   const __nv_bfloat16* __restrict__ weight, int hidden, float eps,
                                   __nv_bfloat16* __restrict__ out) {
    extern __shared__ float sum[];
    __shared__ float scratch[32];
    const std::size_t column = std::size_t(blockIdx.x) * hidden;
    float squares            = 0.0F;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        const float s = load(residual + column + h) + load(y + column + h);
        sum[h]        = s;
        squares += s * s;
    }
    const float inv = rsqrtf(block_sum(squares, scratch) / float(hidden) + eps);
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        residual[column + h] = __float2bfloat16_rn(sum[h]);
        out[column + h]      = __float2bfloat16_rn(sum[h] * inv * load(weight + h));
    }
}

// ---- recurrent head
// -------------------------------------------------------------------------------- One warp per
// output row o: the [hidden, context] columns of W against the G hidden columns and the context
// vector.
__global__ void recurrent_head_context_kernel(const __nv_bfloat16* __restrict__ weight, int in,
                                              int rank, const __nv_bfloat16* __restrict__ bias,
                                              const __nv_bfloat16* __restrict__ hidden, int width,
                                              int steps, const __nv_bfloat16* __restrict__ context,
                                              int outputs, float* __restrict__ out) {
    const int lane = threadIdx.x & 31;
    const int o    = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (o >= outputs) return;
    const __nv_bfloat16* w            = weight + std::size_t(o) * in + 2 * rank;
    float acc[kRecurrentMaxSteps + 1] = {};
    for (int k = lane; k < width; k += 32) {
        const float wh = load(w + k), wc = load(w + width + k);
        for (int s = 0; s < steps; ++s)
            acc[s] = fmaf(wh, load(hidden + std::size_t(s) * width + k), acc[s]);
        acc[kRecurrentMaxSteps] = fmaf(wc, load(context + k), acc[kRecurrentMaxSteps]);
    }
    const float shared = warp_sum(acc[kRecurrentMaxSteps]);
    for (int s = 0; s < steps; ++s) {
        const float v = warp_sum(acc[s]);
        if (lane == 0) out[std::size_t(s) * outputs + o] = load(bias + o) + v + shared;
    }
}

__device__ __forceinline__ int choose(const float* choice) {
    return choice[2] > choice[0] ? int(choice[3]) : int(choice[1]);
}

// raw[o] = context[o, step] + W[o, 0:2r] . [state_in, embedding[prev]]; one warp per row.
__global__ void recurrent_head_raw_kernel(const __nv_bfloat16* __restrict__ weight, int in,
                                          int rank, const __nv_bfloat16* __restrict__ embedding,
                                          const float* __restrict__ context, int step,
                                          const int* __restrict__ anchor,
                                          const __nv_bfloat16* __restrict__ state,
                                          const float* __restrict__ choice, int* drafts,
                                          float* __restrict__ raw) {
    const int prev    = step == 0 ? *anchor : choose(choice);
    const int outputs = 3 * rank;
    if (blockIdx.x == 0 && threadIdx.x == 0 && step > 0) drafts[step - 1] = prev;
    const int lane = threadIdx.x & 31;
    const int o    = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (o >= outputs) return;
    const __nv_bfloat16* w = weight + std::size_t(o) * in;
    const __nv_bfloat16* e = embedding + std::size_t(prev) * rank;
    float acc              = 0.0F;
    for (int k = lane; k < rank; k += 32) {
        acc = fmaf(load(w + k), load(state + k), acc);
        acc = fmaf(load(w + rank + k), load(e + k), acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) raw[o] = context[std::size_t(step) * outputs + o] + acc;
}

__device__ __forceinline__ bool better(float v, int id, float bv, int bid) {
    return v > bv || (v == bv && id < bid);
}

// Gates (every block), state (block 0), then a warp per vocabulary row; the last block to finish
// reduces the blocks' top-1 into the choice slots.
__global__ void __launch_bounds__(256) recurrent_head_logits_kernel(
    const float* __restrict__ raw, const __nv_bfloat16* __restrict__ state_in,
    __nv_bfloat16* __restrict__ state_out, const __nv_bfloat16* __restrict__ folded,
    const __nv_bfloat16* __restrict__ folded_bias, const __nv_bfloat16* __restrict__ base, int rows,
    int step, float cap, int first, int rank_index, int ranks, float* __restrict__ best_value,
    int* __restrict__ best_id, unsigned* __restrict__ done, float* __restrict__ choice) {
    constexpr int R = kRecurrentRank;
    __shared__ float lat[R];
    __shared__ float values[8];
    __shared__ int ids[8];
    __shared__ bool last;
    for (int i = threadIdx.x; i < R; i += blockDim.x) {
        const float g = round_bf16(__fdiv_rn(1.0F, 1.0F + expf(-round_bf16(raw[i]))));
        const float c = round_bf16(tanhf(round_bf16(raw[R + i])));
        if (blockIdx.x == 0) {
            const float keep   = round_bf16(g * load(state_in + i));
            const float update = round_bf16(round_bf16(1.0F - g) * c);
            state_out[i]       = __float2bfloat16_rn(keep + update);
        }
        lat[i] = round_bf16(tanhf(round_bf16(raw[2 * R + i])));
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float l8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) l8[i] = lat[lane * 8 + i];
    float bv        = -INFINITY;
    int bid         = 0x7FFFFFFF;
    const int warps = gridDim.x * (blockDim.x >> 5);
    for (int v = blockIdx.x * (blockDim.x >> 5) + warp; v < rows; v += warps) {
        const uint4 packed =
            __ldg(reinterpret_cast<const uint4*>(folded + std::size_t(v) * R) + lane);
        const auto* h = reinterpret_cast<const __nv_bfloat162*>(&packed);
        float dot     = 0.0F;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __bfloat1622float2(h[i]);
            dot            = fmaf(l8[2 * i], f.x, dot);
            dot            = fmaf(l8[2 * i + 1], f.y, dot);
        }
        dot               = warp_sum(dot);
        const float bias  = round_bf16(load(folded_bias + v) + dot);
        const float x     = round_bf16(load(base + std::size_t(step) * rows + v) + bias);
        const float logit = round_bf16(__fmul_rn(cap, tanhf(__fdiv_rn(x, cap))));
        if (better(logit, first + v, bv, bid)) {
            bv  = logit;
            bid = first + v;
        }
    }
    if (lane == 0) {
        values[warp] = bv;
        ids[warp]    = bid;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < int(blockDim.x >> 5); ++w) {
            if (better(values[w], ids[w], bv, bid)) {
                bv  = values[w];
                bid = ids[w];
            }
        }
        best_value[blockIdx.x] = bv;
        best_id[blockIdx.x]    = bid;
        __threadfence();
        last = atomicAdd(done, 1U) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last || threadIdx.x != 0) return;
    __threadfence();
    for (unsigned b = 0; b < gridDim.x; ++b) {
        const float v = static_cast<volatile float*>(best_value)[b];
        const int id  = static_cast<volatile int*>(best_id)[b];
        if (better(v, id, bv, bid)) {
            bv  = v;
            bid = id;
        }
    }
    const int other            = 1 - rank_index;
    choice[2 * rank_index]     = bv;
    choice[2 * rank_index + 1] = float(bid);
    choice[2 * other]          = ranks == 2 ? 0.0F : -INFINITY;
    choice[2 * other + 1]      = 0.0F;
    *done                      = 0U;
}

__global__ void recurrent_head_select_kernel(const float* __restrict__ choice, int* drafts,
                                             int last) {
    drafts[last] = choose(choice);
}

} // namespace

void feature_mix_rmsnorm_launch(const void* features, const void* logits, const MixNorms& norms,
                                int hidden, int width, std::size_t block_stride, int states,
                                int rows, float eps, const int* count, void* out, void* last,
                                cudaStream_t stream) {
    feature_mix_rmsnorm_kernel<<<dim3(unsigned(width), unsigned(rows)), 256,
                                 std::size_t(hidden) * sizeof(float), stream>>>(
        static_cast<const __nv_bfloat16*>(features), static_cast<const __nv_bfloat16*>(logits),
        norms, hidden, width, block_stride, states, rows, eps, count,
        static_cast<__nv_bfloat16*>(out), static_cast<__nv_bfloat16*>(last));
    CUDA_CHECK(cudaGetLastError());
}

void consecutive_positions_launch(const int* start, const int* minus, int* out, int tokens,
                                  cudaStream_t stream) {
    consecutive_positions_kernel<<<unsigned((tokens + 255) / 256), 256, 0, stream>>>(start, minus,
                                                                                     out, tokens);
    CUDA_CHECK(cudaGetLastError());
}

void head_rmsnorm_rope_launch(void* x, int rows, int tokens, int query_heads, int key_heads,
                              const void* q_norm, const void* k_norm, const int* positions,
                              const InvFrequencies& inv, float eps, cudaStream_t stream) {
    head_rmsnorm_rope_kernel<<<dim3(unsigned(tokens), unsigned(query_heads + key_heads)), kHalf, 0,
                               stream>>>(static_cast<__nv_bfloat16*>(x), rows, query_heads,
                                         static_cast<const __nv_bfloat16*>(q_norm),
                                         static_cast<const __nv_bfloat16*>(k_norm), positions, inv,
                                         eps);
    CUDA_CHECK(cudaGetLastError());
}

void context_kv_store_launch(const StoreLayers& layers, int count_layers, int kv_heads, int width,
                             const int* positions, const int* count, const InvFrequencies& inv,
                             float eps, cudaStream_t stream) {
    context_kv_store_kernel<<<dim3(unsigned(width), unsigned(count_layers), unsigned(2 * kv_heads)),
                              kHalf, 0, stream>>>(layers, kv_heads, width, positions, count, inv,
                                                  eps);
    CUDA_CHECK(cudaGetLastError());
}

void block_attention_launch(const void* qkv, int rows, int query_heads, int kv_heads, int block,
                            const void* k_cache, const void* v_cache, int capacity,
                            const int* frontier, int window, float scale, float* partial, void* out,
                            cudaStream_t stream) {
    const int group = query_heads / kv_heads;
    const dim3 grid{unsigned(kv_heads), unsigned(kBlockAttentionChunks)};
    const auto* q  = static_cast<const __nv_bfloat16*>(qkv);
    const auto* kc = static_cast<const __nv_bfloat16*>(k_cache);
    const auto* vc = static_cast<const __nv_bfloat16*>(v_cache);
    auto* o        = static_cast<__nv_bfloat16*>(out);
    const dim3 combine{unsigned(kv_heads), unsigned(group * block)};
    switch (group) {
#define NINFER_BLOCK_ATTENTION(G)                                                                  \
    case G:                                                                                        \
        block_attention_partial_kernel<G>                                                          \
            <<<grid, 128, 0, stream>>>(q, rows, query_heads, kv_heads, block, kc, vc, capacity,    \
                                       frontier, window, scale, partial);                          \
        block_attention_combine_kernel<G>                                                          \
            <<<combine, kD, 0, stream>>>(partial, kBlockAttentionChunks, query_heads, block, o);   \
        break;
        NINFER_BLOCK_ATTENTION(1)
        NINFER_BLOCK_ATTENTION(2)
        NINFER_BLOCK_ATTENTION(3)
        NINFER_BLOCK_ATTENTION(4)
#undef NINFER_BLOCK_ATTENTION
    default:
        break;
    }
    CUDA_CHECK(cudaGetLastError());
}

void add_rmsnorm_launch(void* residual, const void* y, const void* weight, int hidden, int tokens,
                        float eps, void* out, cudaStream_t stream) {
    add_rmsnorm_kernel<<<unsigned(tokens), 256, std::size_t(hidden) * sizeof(float), stream>>>(
        static_cast<__nv_bfloat16*>(residual), static_cast<const __nv_bfloat16*>(y),
        static_cast<const __nv_bfloat16*>(weight), hidden, eps, static_cast<__nv_bfloat16*>(out));
    CUDA_CHECK(cudaGetLastError());
}

void recurrent_head_context_launch(const void* weight, int in, int rank, const void* bias,
                                   const void* hidden, int width, int steps, const void* context,
                                   float* out, cudaStream_t stream) {
    const int outputs = 3 * rank;
    recurrent_head_context_kernel<<<unsigned((outputs + 7) / 8), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(weight), in, rank,
        static_cast<const __nv_bfloat16*>(bias), static_cast<const __nv_bfloat16*>(hidden), width,
        steps, static_cast<const __nv_bfloat16*>(context), outputs, out);
    CUDA_CHECK(cudaGetLastError());
}

void recurrent_head_step_launch(const RecurrentLaunch& a, cudaStream_t stream) {
    const int outputs = 3 * kRecurrentRank;
    recurrent_head_raw_kernel<<<unsigned((outputs + 7) / 8), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(a.weight), a.in, kRecurrentRank,
        static_cast<const __nv_bfloat16*>(a.embedding), a.context, a.step, a.anchor,
        static_cast<const __nv_bfloat16*>(a.state_in), a.choice, a.drafts, a.raw);
    CUDA_CHECK(cudaMemsetAsync(a.done, 0, sizeof(unsigned), stream));
    recurrent_head_logits_kernel<<<unsigned(kRecurrentBlocks), 256, 0, stream>>>(
        a.raw, static_cast<const __nv_bfloat16*>(a.state_in),
        static_cast<__nv_bfloat16*>(a.state_out), static_cast<const __nv_bfloat16*>(a.folded),
        static_cast<const __nv_bfloat16*>(a.folded_bias), static_cast<const __nv_bfloat16*>(a.base),
        a.rows, a.step, a.cap, a.first, a.rank, a.ranks, a.best_value, a.best_id, a.done, a.choice);
    CUDA_CHECK(cudaGetLastError());
}

void recurrent_head_select_launch(const float* choice, int* drafts, int last, cudaStream_t stream) {
    recurrent_head_select_kernel<<<1, 1, 0, stream>>>(choice, drafts, last);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
