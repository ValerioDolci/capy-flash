// Implements: include/ninfer/ops/message_gate.h
// message_gate: two routes with one arithmetic. Every reduction keeps the same order in both, so
// they give the same bits: each norm is a fixed 256-thread block sum, each bottleneck row is one
// warp whose lane l sums the elements i = l mod 32 in increasing i and then a warp tree, and each
// gate argument sums r = 0..R-1 in order.
//   cluster route: a cluster of kCluster CTAs per tile of up to kTile tokens (at most kMaxClusters
//     clusters while the tiles allow). CTA c stages its R / kCluster rows of `down` in shared
//     memory with asynchronous copies and prefetches its
//     forget/gain rows into L2, computes its bottleneck values for the tile (each token's hidden
//     and message held in registers), the cluster exchanges them through distributed shared memory,
//     and CTA c then gates its slice of H, reading each forget/gain row once for the whole tile.
//   token route: one CTA per token reading every weight (shapes outside the cluster route).
// message_gates_rmsnorm: the batch route below, with its own reduction order (see there).
#include "ops/message_gate/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ninfer/ops/message_gate.h"
#include "ops/common/pdl_launch.cuh"
#include "ops/common/warp.cuh"

#include "ops/launcher/kernel_attr_once.h"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock   = 256;
constexpr int kWarps   = kBlock / 32;
constexpr int kCluster = 8;  // CTAs per token tile
constexpr int kTile    = 16; // tokens per cluster
// Rows of `down` staged per CTA: one warp per row.
constexpr int kMaxClusterRank = kCluster * kWarps;
// Dynamic shared memory opt-in of the cluster route (sm_120 allows 99 KiB per block, static
// shared memory included).
constexpr int kMaxClusterSmem = 96 * 1024;
// Elements of hidden and of message one thread holds per token (H <= kBlock * kMaxPerThread).
constexpr int kMaxPerThread = 12;
// Clusters per launch before a cluster takes more tokens: a cluster of eight 85 KiB CTAs fills
// eight SMs of one GPC, so about one cluster per GPC runs at a time (sm_120 boards with 6 GPCs);
// more clusters run in a second wave.
constexpr int kMaxClusters = 6;

__device__ __forceinline__ float block_sum(float value, float* scratch) {
    value          = warp_sum(value);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) { scratch[warp] = value; }
    __syncthreads();
    float total = 0.0F;
    if (warp == 0) {
        total = lane < kWarps ? scratch[lane] : 0.0F;
        total = warp_sum(total);
        if (lane == 0) { scratch[0] = total; }
    }
    __syncthreads();
    total = scratch[0];
    __syncthreads();
    return total;
}

__global__ void __launch_bounds__(kBlock)
    message_gate_kernel(const float* hidden, const float* message,
                        const __nv_bfloat16* __restrict__ down, const float* __restrict__ forget,
                        const float* __restrict__ gain, float eps, float* out, int width,
                        int rank) {
    __shared__ float scratch[32];
    __shared__ float bottleneck[kMessageGateMaxRank];
    const std::int64_t base = static_cast<std::int64_t>(blockIdx.x) * width;
    const float* h          = hidden + base;
    const float* m          = message + base;

    float hh = 0.0F;
    float mm = 0.0F;
    for (int i = static_cast<int>(threadIdx.x); i < width; i += kBlock) {
        hh += h[i] * h[i];
        mm += m[i] * m[i];
    }
    const float inv_h = rsqrtf(block_sum(hh, scratch) / static_cast<float>(width) + eps);
    const float inv_m = rsqrtf(block_sum(mm, scratch) / static_cast<float>(width) + eps);

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    for (int r = warp; r < rank; r += kWarps) {
        const __nv_bfloat16* row = down + static_cast<std::int64_t>(r) * 2 * width;
        float dot                = 0.0F;
        for (int i = lane; i < width; i += 32) {
            const float zh = __bfloat162float(__float2bfloat16_rn(h[i] * inv_h));
            const float zm = __bfloat162float(__float2bfloat16_rn(m[i] * inv_m));
            dot += __bfloat162float(row[i]) * zh + __bfloat162float(row[width + i]) * zm;
        }
        dot = warp_sum(dot);
        if (lane == 0) { bottleneck[r] = __bfloat162float(__float2bfloat16_rn(dot)); }
    }
    __syncthreads();

    for (int i = static_cast<int>(threadIdx.x); i < width; i += kBlock) {
        const float* f_row = forget + static_cast<std::int64_t>(i) * rank;
        const float* g_row = gain + static_cast<std::int64_t>(i) * rank;
        float f            = 0.0F;
        float g            = 0.0F;
        for (int r = 0; r < rank; ++r) {
            f += f_row[r] * bottleneck[r];
            g += g_row[r] * bottleneck[r];
        }
        const float hv = h[i];
        const float mv = m[i];
        out[base + i]  = (1.0F - tanhf(f)) * hv + tanhf(g) * mv;
    }
}

// Dynamic shared memory of one cluster CTA, in this order:
//   rows    [rows][2H] BF16   this CTA's rows of down
//   z       [2H]       BF16   [bf16(nh) ; bf16(nm)] of the current token
//   d_local [kTile][rows] FP32  this CTA's bottleneck values
//   d_all   [kTile][R] FP32   the whole bottleneck, gathered from the cluster
struct ClusterLayout {
    std::size_t z, d_local, d_all, bytes;

    __host__ __device__ ClusterLayout(int width, int rank) {
        const int rows = rank / kCluster;
        z              = static_cast<std::size_t>(rows) * 2 * width * sizeof(__nv_bfloat16);
        d_local        = z + static_cast<std::size_t>(2) * width * sizeof(__nv_bfloat16);
        d_all          = d_local + static_cast<std::size_t>(kTile) * rows * sizeof(float);
        bytes          = d_all + static_cast<std::size_t>(kTile) * rank * sizeof(float);
    }
};

__device__ __forceinline__ void prefetch_l2(const void* address) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(address));
}

// One cluster's tile of one gate: CTA c of the cluster, the tile's first token t0.
template <int kTileT>
__device__ __forceinline__ void
gate_cluster_tile(const float* hidden, const float* message, const __nv_bfloat16* __restrict__ down,
                  const float* __restrict__ forget, const float* __restrict__ gain, float eps,
                  float* out, int width, int rank, int tokens, int tile, int c, int t0,
                  float* scratch) {
    extern __shared__ __align__(16) unsigned char smem[];
    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    const ClusterLayout layout(width, rank);
    const int rows  = rank / kCluster;
    const int count = min(tile, tokens - t0);
    auto* staged    = reinterpret_cast<__nv_bfloat16*>(smem);
    auto* z         = reinterpret_cast<__nv_bfloat16*>(smem + layout.z);
    auto* d_local   = reinterpret_cast<float*>(smem + layout.d_local);
    auto* d_all     = reinterpret_cast<float*>(smem + layout.d_all);
    const int tid   = static_cast<int>(threadIdx.x);
    const int lane  = tid & 31;
    const int warp  = tid >> 5;
    const int slice = (width + kCluster - 1) / kCluster;
    const int end   = min(width, (c + 1) * slice);

    // Stage this CTA's rows of down (contiguous) and pull its forget/gain rows into L2 while the
    // bottleneck of the tile is computed.
    {
        const auto* source =
            reinterpret_cast<const uint4*>(down + static_cast<std::int64_t>(c) * rows * 2 * width);
        auto* destination = reinterpret_cast<uint4*>(staged);
        const int vectors = rows * 2 * width / 8;
        for (int v = tid; v < vectors; v += kBlock) {
            __pipeline_memcpy_async(destination + v, source + v, sizeof(uint4));
        }
        __pipeline_commit();
        for (int i = c * slice + tid; i < end; i += kBlock) {
            for (int f = 0; f < rank; f += 32) {
                prefetch_l2(forget + static_cast<std::int64_t>(i) * rank + f);
                prefetch_l2(gain + static_cast<std::int64_t>(i) * rank + f);
            }
        }
    }

    // The weights above are constant: the kernel before this one may still be running until here.
    pdl_enter();
    for (int t = 0; t < count; ++t) {
        const float* h = hidden + static_cast<std::int64_t>(t0 + t) * width;
        const float* m = message + static_cast<std::int64_t>(t0 + t) * width;
        // Element i = tid + kBlock k: the order of the token route's strided loop.
        float hv[kMaxPerThread];
        float mv[kMaxPerThread];
#pragma unroll
        for (int k = 0; k < kMaxPerThread; ++k) {
            const int i = tid + kBlock * k;
            hv[k]       = i < width ? h[i] : 0.0F;
            mv[k]       = i < width ? m[i] : 0.0F;
        }
        float hh = 0.0F;
        float mm = 0.0F;
#pragma unroll
        for (int k = 0; k < kMaxPerThread; ++k) {
            hh += hv[k] * hv[k];
            mm += mv[k] * mv[k];
        }
        const float inv_h = rsqrtf(block_sum(hh, scratch) / static_cast<float>(width) + eps);
        const float inv_m = rsqrtf(block_sum(mm, scratch) / static_cast<float>(width) + eps);
#pragma unroll
        for (int k = 0; k < kMaxPerThread; ++k) {
            const int i = tid + kBlock * k;
            if (i < width) {
                z[i]         = __float2bfloat16_rn(hv[k] * inv_h);
                z[width + i] = __float2bfloat16_rn(mv[k] * inv_m);
            }
        }
        if (t == 0) { __pipeline_wait_prior(0); }
        __syncthreads();
        if (warp < rows) {
            const __nv_bfloat16* row = staged + static_cast<std::int64_t>(warp) * 2 * width;
            float dot                = 0.0F;
            for (int i = lane; i < width; i += 32) {
                const float zh = __bfloat162float(z[i]);
                const float zm = __bfloat162float(z[width + i]);
                dot += __bfloat162float(row[i]) * zh + __bfloat162float(row[width + i]) * zm;
            }
            dot = warp_sum(dot);
            if (lane == 0) {
                d_local[t * rows + warp] = __bfloat162float(__float2bfloat16_rn(dot));
            }
        }
        __syncthreads();
    }

    // Gate this CTA's slice of H in chunks staged in the shared memory of rows and z (free since
    // the bottleneck is complete): the slice of forget and of gain is contiguous, so the copies
    // coalesce; a padded row stride keeps the per-thread float4 reads free of bank conflicts. The
    // first chunk is in flight while the cluster exchanges the bottleneck.
    const int stride = rank + 4;
    const int capacity =
        min(kBlock, static_cast<int>(layout.d_local / (2 * sizeof(float) * stride)));
    auto* f_stage     = reinterpret_cast<float*>(smem);
    auto* g_stage     = f_stage + static_cast<std::size_t>(capacity) * stride;
    const int vectors = rank / 4;
    const auto stage  = [&](int i0, int n) {
        for (int v = tid; v < n * vectors; v += kBlock) {
            const int row             = v / vectors;
            const int q               = v - row * vectors;
            const std::int64_t source = static_cast<std::int64_t>(i0 + row) * rank + 4 * q;
            __pipeline_memcpy_async(f_stage + row * stride + 4 * q, forget + source,
                                     sizeof(float4));
            __pipeline_memcpy_async(g_stage + row * stride + 4 * q, gain + source, sizeof(float4));
        }
        __pipeline_commit();
    };
    if (c * slice < end) { stage(c * slice, min(capacity, end - c * slice)); }

    // Gather the whole bottleneck of the tile; the second barrier keeps every CTA's d_local alive
    // until all peers have read it.
    cluster.sync();
    for (int k = tid; k < count * rank; k += kBlock) {
        const int t       = k / rank;
        const int r       = k - t * rank;
        const float* peer = cluster.map_shared_rank(d_local, r / rows);
        d_all[k]          = peer[t * rows + r % rows];
    }
    cluster.sync();

    for (int i0 = c * slice; i0 < end; i0 += capacity) {
        const int n = min(capacity, end - i0);
        if (i0 != c * slice) {
            __syncthreads(); // the previous chunk is no longer read
            stage(i0, n);
        }
        __pipeline_wait_prior(0);
        __syncthreads();
        if (tid < n) {
            const auto* f_row = reinterpret_cast<const float4*>(f_stage + tid * stride);
            const auto* g_row = reinterpret_cast<const float4*>(g_stage + tid * stride);
            float f[kTileT];
            float g[kTileT];
#pragma unroll
            for (int t = 0; t < kTileT; ++t) {
                f[t] = 0.0F;
                g[t] = 0.0F;
            }
            for (int q = 0; q < vectors; ++q) {
                const float4 fv   = f_row[q];
                const float4 gv   = g_row[q];
                const float fr[4] = {fv.x, fv.y, fv.z, fv.w};
                const float gr[4] = {gv.x, gv.y, gv.z, gv.w};
#pragma unroll
                for (int k = 0; k < 4; ++k) {
#pragma unroll
                    for (int t = 0; t < kTileT; ++t) {
                        if (t < count) {
                            const float d = d_all[t * rank + 4 * q + k];
                            f[t] += fr[k] * d;
                            g[t] += gr[k] * d;
                        }
                    }
                }
            }
            const int i = i0 + tid;
#pragma unroll
            for (int t = 0; t < kTileT; ++t) {
                if (t < count) {
                    const std::int64_t index = static_cast<std::int64_t>(t0 + t) * width + i;
                    const float hv           = hidden[index];
                    const float mv           = message[index];
                    out[index]               = (1.0F - tanhf(f[t])) * hv + tanhf(g[t]) * mv;
                }
            }
        }
    }
}

template <int kTileT>
__global__ void __cluster_dims__(kCluster, 1, 1) __launch_bounds__(kBlock)
    message_gate_cluster_kernel(const float* hidden, const float* message,
                                const __nv_bfloat16* __restrict__ down,
                                const float* __restrict__ forget, const float* __restrict__ gain,
                                float eps, float* out, int width, int rank, int tokens, int tile) {
    __shared__ float scratch[32];
    gate_cluster_tile<kTileT>(hidden, message, down, forget, gain, eps, out, width, rank, tokens,
                              tile, static_cast<int>(blockIdx.x) % kCluster,
                              static_cast<int>(blockIdx.x) / kCluster * tile, scratch);
}

// Batch route (message_gates_rmsnorm at decode and verification widths): a cluster of kCluster
// CTAs per gate and token tile, H / 8 threads per CTA. Thread j owns elements [8j, 8j + 8) of
// hidden and message (norms and bottleneck) and row i = c H / 8 + j of the gate and of its norm.
// Before the wait for the previous kernel, CTA c holds its R / kCluster rows of down in registers
// and its H / kCluster rows of forget and gain as BF16 in shared memory (exact: they are BF16
// values widened to FP32). Per token, every reduction is fixed: a thread's 8 (bottleneck: 16)
// terms in order, a warp xor tree, the warps in order, then the CTAs of the cluster in order; a
// gate argument is a thread's R terms in four interleaved partial sums. So a column gets the same
// bits at every width, tile and batch size. After the wait: one block sum for the
// two input norms, one for the CTA's bottleneck rows, the rows pushed to every CTA of the cluster
// (one cluster barrier), the gate, the CTA's output squares pushed likewise (one barrier), the
// norm.
constexpr int kBatchThreads  = 288; // H <= 8 * kBatchThreads on this route
constexpr int kBatchWarps    = kBatchThreads / 32;
constexpr int kBatchMaxRows  = 8; // rows of down per CTA: R <= kCluster * kBatchMaxRows
constexpr int kBatchMaxRank  = kCluster * kBatchMaxRows;
constexpr int kBatchVectorsR = kBatchMaxRank / 8; // 16-byte chunks of a forget or gain row

std::size_t batch_shared_bytes(int width, int rank) {
    return static_cast<std::size_t>(2) * (rank / 8) * (width / 8) * sizeof(uint4);
}

// The dynamic shared memory opt-in of the batch kernel: the largest shape of the route.
constexpr int kBatchMaxShared = 2 * kBatchVectorsR * kBatchThreads * 16;

__device__ __forceinline__ void cluster_arrive_relaxed() {
    asm volatile("barrier.cluster.arrive.relaxed;" ::: "memory");
}

// A full cluster barrier: the stores to peers' shared memory before it are visible after it.
__device__ __forceinline__ void cluster_barrier() {
    asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory");
}

__device__ __forceinline__ void cluster_wait() {
    asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
}

// The CTA's threads issue bulk L2 prefetches of [base, base + bytes) in 4 KiB chunks (the copy
// engine fetches them: the loads that follow hit L2 instead of waiting on DRAM). Only a lone
// gate's launch uses them: its 8 SMs are limited by their own outstanding loads, while a batch of
// gates already keeps DRAM busy and the prefetches then slow it down (measured at T = 1).
__device__ __forceinline__ void bulk_prefetch_l2(const void* base, std::uint32_t bytes) {
    constexpr std::uint32_t kChunk = 4096U;
    const auto* bytes_base         = static_cast<const unsigned char*>(base);
    for (std::uint32_t offset = threadIdx.x * kChunk; offset < bytes;
         offset += blockDim.x * kChunk) {
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(bytes_base + offset),
                     "r"(min(kChunk, bytes - offset))
                     : "memory");
    }
}

__device__ __forceinline__ void bf16x8_to_float(const uint4& v, float (&out)[8]) {
    const auto* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float2 f = __bfloat1622float2(p[k]);
        out[2 * k]     = f.x;
        out[2 * k + 1] = f.y;
    }
}

__device__ __forceinline__ float sum_warps(const float* partial, int warps) {
    float total = 0.0F;
    for (int w = 0; w < warps; ++w) { total += partial[w]; }
    return total;
}

template <int kTileT>
__global__ void __cluster_dims__(kCluster, 1, 1) __launch_bounds__(kBatchThreads, 1)
    message_gate_batch_kernel(MessageGateBatch batch, float eps, float norm_eps, int width,
                              int rank, int tokens, int tile, bool prefetch, bool bf16_weights) {
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ float input_partial[kTileT][2][kBatchWarps];
    __shared__ float row_partial[kTileT][kBatchMaxRows][kBatchWarps];
    __shared__ float out_partial[kTileT][kBatchWarps];
    __shared__ __align__(16) float bottleneck[kTileT][kBatchMaxRank]; // written by every CTA
    __shared__ float out_squares[kTileT][kCluster];                   // written by every CTA

    cluster_arrive_relaxed(); // waited for before the first store to a peer
    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    const MessageGateBatchItem& g             = batch.gates[blockIdx.y];
    const int c                               = static_cast<int>(blockIdx.x) % kCluster;
    const int t0                              = static_cast<int>(blockIdx.x) / kCluster * tile;
    const int count                           = min(tile, tokens - t0);
    const int j                               = static_cast<int>(threadIdx.x);
    const int lane                            = j & 31;
    const int warp                            = j >> 5;
    const int threads                         = width / 8;
    const int warps                           = threads / 32;
    const int rows                            = rank / kCluster;
    const int vectors                         = rank / 8;
    const int i                               = c * threads + j; // this thread's row of the gate
    auto* staged = reinterpret_cast<uint4*>(smem);               // [2][vectors][threads]

    // Weights: rows c * rows + r of down (this thread's 8 hidden and 8 message columns), and the
    // CTA's forget and gain rows narrowed to BF16 in chunks of 8 ranks, chunk q of row j at
    // q * threads + j (a warp reads 512 consecutive bytes). Non-coherent global loads, all of a
    // matrix issued before its first use (they do not wait on the shared stores).
    if (prefetch) {
        const auto slice_bytes = static_cast<std::uint32_t>(threads) * rank * sizeof(float);
        bulk_prefetch_l2(g.down + static_cast<std::int64_t>(c) * rows * 2 * width,
                         static_cast<std::uint32_t>(rows) * 2 * width * sizeof(__nv_bfloat16));
        bulk_prefetch_l2(g.forget + static_cast<std::int64_t>(c) * threads * rank, slice_bytes);
        bulk_prefetch_l2(g.gain + static_cast<std::int64_t>(c) * threads * rank, slice_bytes);
    }
    uint4 down_h[kBatchMaxRows];
    uint4 down_m[kBatchMaxRows];
#pragma unroll
    for (int r = 0; r < kBatchMaxRows; ++r) {
        if (r < rows) {
            const auto* row = reinterpret_cast<const uint4*>(
                g.down + static_cast<std::int64_t>(c * rows + r) * 2 * width);
            down_h[r] = __ldg(row + j);
            down_m[r] = __ldg(row + threads + j);
        }
    }
    const __nv_bfloat16 gamma = g.norm[i];
    const int quads           = rank / 4; // float4 per row
    if (bf16_weights) {
        // forget/gain stored as BF16: chunk q of row j copied as is.
#pragma unroll
        for (int which = 0; which < 2; ++which) {
            const auto* source = reinterpret_cast<const uint4*>(
                reinterpret_cast<const __nv_bfloat16*>(which == 0 ? g.forget : g.gain) +
                static_cast<std::int64_t>(c) * threads * rank);
            uint4 x[kBatchVectorsR];
#pragma unroll
            for (int k = 0; k < kBatchVectorsR; ++k) {
                if (k < vectors) { x[k] = __ldg(source + j + k * threads); }
            }
#pragma unroll
            for (int k = 0; k < kBatchVectorsR; ++k) {
                if (k < vectors) {
                    const int v   = j + k * threads;
                    const int row = v / vectors;
                    staged[(which * vectors + (v - row * vectors)) * threads + row] = x[k];
                }
            }
        }
    }
#pragma unroll
    for (int which = 0; which < 2 && !bf16_weights; ++which) {
        const auto* source = reinterpret_cast<const float4*>(
            (which == 0 ? g.forget : g.gain) + static_cast<std::int64_t>(c) * threads * rank);
        auto* target = reinterpret_cast<uint2*>(staged + which * vectors * threads);
        float4 x[2 * kBatchVectorsR];
#pragma unroll
        for (int k = 0; k < 2 * kBatchVectorsR; ++k) {
            if (k < quads) { x[k] = __ldg(source + j + k * threads); }
        }
#pragma unroll
        for (int k = 0; k < 2 * kBatchVectorsR; ++k) {
            if (k < quads) {
                const int v             = j + k * threads;
                const int row           = v / quads;
                const int quad          = v - row * quads;
                const __nv_bfloat162 lo = __floats2bfloat162_rn(x[k].x, x[k].y);
                const __nv_bfloat162 hi = __floats2bfloat162_rn(x[k].z, x[k].w);
                uint2 packed;
                packed.x = *reinterpret_cast<const unsigned*>(&lo);
                packed.y = *reinterpret_cast<const unsigned*>(&hi);
                target[2 * ((quad >> 1) * threads + row) + (quad & 1)] = packed;
            }
        }
    }

    pdl_enter();
    const float inv_width = 1.0F / static_cast<float>(width);

    // The two input norms of every token of the tile.
#pragma unroll
    for (int t = 0; t < kTileT; ++t) {
        if (t < count) {
            const std::int64_t base = static_cast<std::int64_t>(t0 + t) * width + 8 * j;
            const float4 h0         = *reinterpret_cast<const float4*>(g.hidden + base);
            const float4 h1         = *reinterpret_cast<const float4*>(g.hidden + base + 4);
            const float4 m0         = *reinterpret_cast<const float4*>(g.message + base);
            const float4 m1         = *reinterpret_cast<const float4*>(g.message + base + 4);
            const float hh = h0.x * h0.x + h0.y * h0.y + h0.z * h0.z + h0.w * h0.w + h1.x * h1.x +
                             h1.y * h1.y + h1.z * h1.z + h1.w * h1.w;
            const float mm = m0.x * m0.x + m0.y * m0.y + m0.z * m0.z + m0.w * m0.w + m1.x * m1.x +
                             m1.y * m1.y + m1.z * m1.z + m1.w * m1.w;
            const float hs = warp_sum(hh);
            const float ms = warp_sum(mm);
            if (lane == 0) {
                input_partial[t][0][warp] = hs;
                input_partial[t][1][warp] = ms;
            }
        }
    }
    __syncthreads();

    // The CTA's bottleneck rows: this thread's 16 terms, then the warps.
#pragma unroll
    for (int t = 0; t < kTileT; ++t) {
        if (t < count) {
            const float inv_h = rsqrtf(sum_warps(input_partial[t][0], warps) * inv_width + eps);
            const float inv_m = rsqrtf(sum_warps(input_partial[t][1], warps) * inv_width + eps);
            const std::int64_t base = static_cast<std::int64_t>(t0 + t) * width + 8 * j;
            const float4 h0         = *reinterpret_cast<const float4*>(g.hidden + base);
            const float4 h1         = *reinterpret_cast<const float4*>(g.hidden + base + 4);
            const float4 m0         = *reinterpret_cast<const float4*>(g.message + base);
            const float4 m1         = *reinterpret_cast<const float4*>(g.message + base + 4);
            const float hv[8]       = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
            const float mv[8]       = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
            float zh[8], zm[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                zh[k] = __bfloat162float(__float2bfloat16_rn(hv[k] * inv_h));
                zm[k] = __bfloat162float(__float2bfloat16_rn(mv[k] * inv_m));
            }
#pragma unroll
            for (int r = 0; r < kBatchMaxRows; ++r) {
                if (r < rows) {
                    float wh[8], wm[8];
                    bf16x8_to_float(down_h[r], wh);
                    bf16x8_to_float(down_m[r], wm);
                    float dot = 0.0F;
#pragma unroll
                    for (int k = 0; k < 8; ++k) { dot = fmaf(wh[k], zh[k], dot); }
#pragma unroll
                    for (int k = 0; k < 8; ++k) { dot = fmaf(wm[k], zm[k], dot); }
                    dot = warp_sum(dot);
                    if (lane == 0) { row_partial[t][r][warp] = dot; }
                }
            }
        }
    }
    __syncthreads();
    cluster_wait(); // every CTA of the cluster runs: its shared memory may be written
    for (int k = j; k < count * rows; k += threads) {
        const int t   = k / rows;
        const int r   = k - t * rows;
        const float d = __bfloat162float(__float2bfloat16_rn(sum_warps(row_partial[t][r], warps)));
        const int place = c * rows + r;
#pragma unroll
        for (int p = 0; p < kCluster; ++p) {
            cluster.map_shared_rank(&bottleneck[t][0], p)[place] = d;
        }
    }
    cluster_barrier();

    // The gate of row i, then the CTA's squares of the output.
    float out[kTileT];
#pragma unroll
    for (int t = 0; t < kTileT; ++t) {
        out[t] = 0.0F;
        if (t < count) {
            const std::int64_t index = static_cast<std::int64_t>(t0 + t) * width + i;
            const float hv           = g.hidden[index];
            const float mv           = g.message[index];
            float f[4]               = {0.0F, 0.0F, 0.0F, 0.0F};
            float a[4]               = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
            for (int q = 0; q < kBatchVectorsR; ++q) {
                if (q < vectors) {
                    float fw[8], gw[8];
                    bf16x8_to_float(staged[q * threads + j], fw);
                    bf16x8_to_float(staged[(vectors + q) * threads + j], gw);
                    const float4 d0   = *reinterpret_cast<const float4*>(&bottleneck[t][8 * q]);
                    const float4 d1   = *reinterpret_cast<const float4*>(&bottleneck[t][8 * q + 4]);
                    const float dv[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        f[k & 3] = fmaf(fw[k], dv[k], f[k & 3]);
                        a[k & 3] = fmaf(gw[k], dv[k], a[k & 3]);
                    }
                }
            }
            const float fs = (f[0] + f[1]) + (f[2] + f[3]);
            const float gs = (a[0] + a[1]) + (a[2] + a[3]);
            out[t]         = (1.0F - tanhf(fs)) * hv + tanhf(gs) * mv;
            g.out[index]   = out[t];
            const float sq = warp_sum(out[t] * out[t]);
            if (lane == 0) { out_partial[t][warp] = sq; }
        }
    }
    __syncthreads();
    for (int t = j; t < count; t += threads) {
        const float s = sum_warps(out_partial[t], warps);
#pragma unroll
        for (int p = 0; p < kCluster; ++p) {
            cluster.map_shared_rank(&out_squares[t][0], p)[c] = s;
        }
    }
    cluster_barrier();

    const float weight = __bfloat162float(gamma);
#pragma unroll
    for (int t = 0; t < kTileT; ++t) {
        if (t < count) {
            float total = 0.0F;
#pragma unroll
            for (int p = 0; p < kCluster; ++p) { total += out_squares[t][p]; }
            const float inv = rsqrtf(total * inv_width + norm_eps);
            g.normed[static_cast<std::int64_t>(t0 + t) * width + i] =
                __float2bfloat16_rn(out[t] * inv * weight);
        }
    }
}

bool batch_route(const Tensor& hidden, const Tensor& message, const Tensor& down,
                 const Tensor& forget, const Tensor& gain, const Tensor& out) {
    const int width    = hidden.ne[0];
    const int rank     = down.ne[1];
    const auto aligned = [](const Tensor& t) {
        return (reinterpret_cast<std::uintptr_t>(t.data) & 15U) == 0;
    };
    return rank % kCluster == 0 && rank % 8 == 0 && rank <= kBatchMaxRank && width % 256 == 0 &&
           width <= 8 * kBatchThreads &&
           batch_shared_bytes(width, rank) <= static_cast<std::size_t>(kBatchMaxShared) &&
           aligned(down) && aligned(forget) && aligned(gain) && aligned(hidden) &&
           aligned(message) && aligned(out);
}

bool cluster_route(const Tensor& hidden, const Tensor& message, const Tensor& down,
                   const Tensor& forget, const Tensor& gain, const Tensor& out) {
    const int width    = hidden.ne[0];
    const int rank     = down.ne[1];
    const auto aligned = [](const Tensor& t) {
        return (reinterpret_cast<std::uintptr_t>(t.data) & 15U) == 0;
    };
    return rank % kCluster == 0 && rank <= kMaxClusterRank && rank % 4 == 0 && width % 8 == 0 &&
           width <= kBlock * kMaxPerThread &&
           ClusterLayout(width, rank).bytes <= static_cast<std::size_t>(kMaxClusterSmem) &&
           aligned(down) && aligned(forget) && aligned(gain) && aligned(hidden) &&
           aligned(message) && aligned(out);
}

} // namespace

void message_gate_launch(const Tensor& hidden, const Tensor& message, const Tensor& down,
                         const Tensor& forget, const Tensor& gain, float eps, Tensor& out,
                         cudaStream_t stream) {
    const int width  = hidden.ne[0];
    const int rank   = down.ne[1];
    const int tokens = hidden.ne[1];
    if (cluster_route(hidden, message, down, forget, gain, out)) {
        const auto bytes = static_cast<int>(ClusterLayout(width, rank).bytes);
        // As few tokens per cluster as keeps the clusters at most kMaxClusters (each one reads the
        // weights once for its tile), at most kTile; the kernel instance is the smallest tile size
        // that holds the tile.
        const int tile    = std::clamp((tokens + kMaxClusters - 1) / kMaxClusters, 1, kTile);
        const auto grid   = static_cast<unsigned>((tokens + tile - 1) / tile * kCluster);
        const auto launch = [&](auto kernel, FuncAttrPerDevice& attr) {
            attr.ensure(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxClusterSmem);
            launch_maybe_dependent(
                tokens <= kDependentLaunchMaxTokens, dim3(grid), dim3(kBlock),
                static_cast<std::size_t>(bytes), stream, kernel,
                static_cast<const float*>(hidden.data), static_cast<const float*>(message.data),
                static_cast<const __nv_bfloat16*>(down.data),
                static_cast<const float*>(forget.data), static_cast<const float*>(gain.data), eps,
                static_cast<float*>(out.data), width, rank, tokens, tile);
        };
        if (tile == 1) {
            static FuncAttrPerDevice attr;
            launch(message_gate_cluster_kernel<1>, attr);
        } else if (tile <= 2) {
            static FuncAttrPerDevice attr;
            launch(message_gate_cluster_kernel<2>, attr);
        } else if (tile <= 4) {
            static FuncAttrPerDevice attr;
            launch(message_gate_cluster_kernel<4>, attr);
        } else if (tile <= 8) {
            static FuncAttrPerDevice attr;
            launch(message_gate_cluster_kernel<8>, attr);
        } else {
            static FuncAttrPerDevice attr;
            launch(message_gate_cluster_kernel<kTile>, attr);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const auto grid = static_cast<unsigned>(tokens);
    message_gate_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const float*>(hidden.data), static_cast<const float*>(message.data),
        static_cast<const __nv_bfloat16*>(down.data), static_cast<const float*>(forget.data),
        static_cast<const float*>(gain.data), eps, static_cast<float*>(out.data), width, rank);
    CUDA_CHECK(cudaGetLastError());
}

bool message_gate_batch_launch(const MessageGateBatch& batch, int gates, const Tensor& hidden,
                               const Tensor& message, const Tensor& down, const Tensor& forget,
                               const Tensor& gain, const Tensor& out, float eps, float norm_eps,
                               cudaStream_t stream) {
    if (!batch_route(hidden, message, down, forget, gain, out)) { return false; }
    const int width  = hidden.ne[0];
    const int rank   = down.ne[1];
    const int tokens = hidden.ne[1];
    // The gates share the clusters a single gate would take.
    const int clusters = std::max(1, kMaxClusters / gates);
    const int tile     = std::clamp((tokens + clusters - 1) / clusters, 1, kTile);
    const dim3 grid(static_cast<unsigned>((tokens + tile - 1) / tile * kCluster),
                    static_cast<unsigned>(gates));
    const auto launch = [&](auto kernel, FuncAttrPerDevice& attr) {
        attr.ensure(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kBatchMaxShared);
        launch_maybe_dependent(
            tokens <= kDependentLaunchMaxTokens, grid, dim3(static_cast<unsigned>(width / 8)),
            batch_shared_bytes(width, rank), stream, kernel, batch, eps, norm_eps, width, rank,
            tokens, tile, gates == 1, forget.dtype == DType::BF16);
    };
    if (tile == 1) {
        static FuncAttrPerDevice attr;
        launch(message_gate_batch_kernel<1>, attr);
    } else if (tile <= 2) {
        static FuncAttrPerDevice attr;
        launch(message_gate_batch_kernel<2>, attr);
    } else if (tile <= 4) {
        static FuncAttrPerDevice attr;
        launch(message_gate_batch_kernel<4>, attr);
    } else if (tile <= 8) {
        static FuncAttrPerDevice attr;
        launch(message_gate_batch_kernel<8>, attr);
    } else {
        static FuncAttrPerDevice attr;
        launch(message_gate_batch_kernel<kTile>, attr);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

} // namespace ninfer::ops::detail
