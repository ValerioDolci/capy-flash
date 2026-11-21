// Implements: include/ninfer/ops/message_gate.h
// Two routes with one arithmetic. Every reduction keeps the same order in both, so they give the
// same bits: each norm is a fixed 256-thread block sum, each bottleneck row is one warp whose lane
// l sums the elements i = l mod 32 in increasing i and then a warp tree, and each gate argument
// sums r = 0..R-1 in order.
//   cluster route: a cluster of kCluster CTAs per tile of up to kTile tokens (at most kMaxClusters
//     clusters while the tiles allow). CTA c stages its R / kCluster rows of `down` in shared
//     memory with asynchronous copies and prefetches its
//     forget/gain rows into L2, computes its bottleneck values for the tile (each token's hidden
//     and message held in registers), the cluster exchanges them through distributed shared memory,
//     and CTA c then gates its slice of H, reading each forget/gain row once for the whole tile.
//   token route: one CTA per token reading every weight (shapes outside the cluster route).
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

// rmsnorm_f32_kernel's arithmetic on one row x (written by this cluster in this kernel: plain
// loads).
__device__ __forceinline__ void norm_row(const float* x, const __nv_bfloat16* __restrict__ weight,
                                         float eps, __nv_bfloat16* out, int width, float* scratch) {
    float squares = 0.0F;
    for (int i = static_cast<int>(threadIdx.x); i < width; i += kBlock) { squares += x[i] * x[i]; }
    squares        = warp_sum(squares);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) { scratch[warp] = squares; }
    __syncthreads();
    float total = 0.0F;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) { total += scratch[w]; }
    const float inv = rsqrtf(total / static_cast<float>(width) + eps);
    for (int i = static_cast<int>(threadIdx.x); i < width; i += kBlock) {
        out[i] = __float2bfloat16_rn(x[i] * inv * __bfloat162float(weight[i]));
    }
    __syncthreads(); // scratch is reused by the next row
}

// Gate blockIdx.y of the batch on the cluster route, then the norms of the tile's rows: after
// the cluster barrier every slice of the tile is written, and CTA c norms rows c, c + 8, ...
template <int kTileT>
__global__ void __cluster_dims__(kCluster, 1, 1) __launch_bounds__(kBlock)
    message_gate_batch_kernel(MessageGateBatch batch, float eps, float norm_eps, int width,
                              int rank, int tokens, int tile) {
    __shared__ float scratch[32];
    const MessageGateBatchItem& g = batch.gates[blockIdx.y];
    const int c                   = static_cast<int>(blockIdx.x) % kCluster;
    const int t0                  = static_cast<int>(blockIdx.x) / kCluster * tile;
    gate_cluster_tile<kTileT>(g.hidden, g.message, g.down, g.forget, g.gain, eps, g.out, width,
                              rank, tokens, tile, c, t0, scratch);
    __threadfence();
    cooperative_groups::this_cluster().sync();
    const int count = min(tile, tokens - t0);
    for (int t = c; t < count; t += kCluster) {
        const std::int64_t row = static_cast<std::int64_t>(t0 + t) * width;
        norm_row(g.out + row, g.norm, norm_eps, g.normed + row, width, scratch);
    }
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
    if (!cluster_route(hidden, message, down, forget, gain, out)) { return false; }
    const int width  = hidden.ne[0];
    const int rank   = down.ne[1];
    const int tokens = hidden.ne[1];
    const auto bytes = static_cast<int>(ClusterLayout(width, rank).bytes);
    // The gates share the clusters a single gate would take.
    const int clusters = std::max(1, kMaxClusters / gates);
    const int tile     = std::clamp((tokens + clusters - 1) / clusters, 1, kTile);
    const dim3 grid(static_cast<unsigned>((tokens + tile - 1) / tile * kCluster),
                    static_cast<unsigned>(gates));
    const auto launch = [&](auto kernel, FuncAttrPerDevice& attr) {
        attr.ensure(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxClusterSmem);
        launch_maybe_dependent(tokens <= kDependentLaunchMaxTokens, grid, dim3(kBlock),
                               static_cast<std::size_t>(bytes), stream, kernel, batch, eps,
                               norm_eps, width, rank, tokens, tile);
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
