// Implements: include/ninfer/ops/moe_sigmoid_route.h
// One cluster of blocks per column: each warp of the cluster evaluates the logits of a
// strided set of experts (lane-strided FP32 dot, then a butterfly sum) and writes score and key
// into the shared memory of the cluster's first block, which then selects the k largest keys
// (lower id on an exact tie: one pass by place in that order, or k block-wide arg-max rounds when
// the experts outnumber the threads) and the weights. A logit's reduction order
// depends only on H, so every device of one architecture gives the same bits for the same column.
#include "ops/moe_sigmoid_route/launch.h"

#include "core/device.h"
#include "ops/common/pdl_launch.cuh" // CUDA_CHECK
#include "ninfer/ops/moe_sigmoid_route.h"
#include "ops/common/warp.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

namespace cg = cooperative_groups;

// Columns of a narrow call get a cluster of 8 blocks of 16 warps each; from kClusterMaxColumns
// columns on, one block of 8 warps per column, and from kTileMinColumns on a block per
// kTileColumns columns (moe_sigmoid_route_tile_kernel) where the tile fits (same bits everywhere).
constexpr int kClusterMaxColumns = 64;

// (key, id) order of the selection: larger key first, lower id on an exact tie. id < 0 = empty.
__device__ __forceinline__ bool better(float key, int id, float best_key, int best_id) {
    if (id < 0) { return false; }
    if (best_id < 0) { return true; }
    return key > best_key || (key == best_key && id < best_id);
}

// k block-wide arg-max rounds (larger key first, lower id on an exact tie) into chosen, for
// geometries with more experts than threads.
template <int kWarps>
__device__ void select_rounds(const float* key, int experts, int top_k, int* chosen,
                              float* best_key_of, int* best_id_of) {
    constexpr int kThreads = kWarps * 32;
    const int warp         = static_cast<int>(threadIdx.x) >> 5;
    const int lane         = static_cast<int>(threadIdx.x) & 31;
    for (int j = 0; j < top_k; ++j) {
        float best_key = 0.0F;
        int best_id    = -1;
        for (int e = static_cast<int>(threadIdx.x); e < experts; e += kThreads) {
            bool taken = false;
            for (int i = 0; i < j; ++i) { taken = taken || chosen[i] == e; }
            if (!taken && better(key[e], e, best_key, best_id)) {
                best_key = key[e];
                best_id  = e;
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_key = __shfl_xor_sync(kFullWarpMask, best_key, offset);
            const int other_id    = __shfl_xor_sync(kFullWarpMask, best_id, offset);
            if (better(other_key, other_id, best_key, best_id)) {
                best_key = other_key;
                best_id  = other_id;
            }
        }
        if (lane == 0) {
            best_key_of[warp] = best_key;
            best_id_of[warp]  = best_id;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float k_best = 0.0F;
            int id_best  = -1;
            for (int w = 0; w < kWarps; ++w) {
                if (better(best_key_of[w], best_id_of[w], k_best, id_best)) {
                    k_best  = best_key_of[w];
                    id_best = best_id_of[w];
                }
            }
            chosen[j] = id_best;
        }
        __syncthreads();
    }
}

template <int kCluster, int kWarps>
__global__ void __cluster_dims__(kCluster, 1, 1) __launch_bounds__(kWarps * 32)
    moe_sigmoid_route_kernel(const __nv_bfloat16* __restrict__ x, const float* __restrict__ router,
                             const float* __restrict__ bias, std::int32_t hidden,
                             std::int32_t experts, std::int32_t top_k, float route_scale,
                             std::int32_t* __restrict__ ids, float* __restrict__ weights,
                             const std::int32_t* __restrict__ valid_columns,
                             std::int32_t lane_width) {
    pdl_enter();
    extern __shared__ float shared[];
    float* score = shared;           // [E]
    float* key   = shared + experts; // [E]
    __shared__ float best_key_of[kWarps];
    __shared__ int best_id_of[kWarps];
    __shared__ int chosen[kMoeSigmoidRouteMaxTopK];

    constexpr int kThreads    = kWarps * 32;
    cg::cluster_group cluster = cg::this_cluster();
    const int rank            = static_cast<int>(cluster.block_rank());
    const int column          = static_cast<int>(blockIdx.x) / kCluster;
    const int warp            = static_cast<int>(threadIdx.x) >> 5;
    const int lane            = static_cast<int>(threadIdx.x) & 31;
    std::int32_t* out_ids     = ids + static_cast<std::int64_t>(column) * top_k;
    float* out_weights        = weights + static_cast<std::int64_t>(column) * top_k;

    if (valid_columns != nullptr && column % lane_width >= valid_columns[column / lane_width]) {
        if (rank != 0) { return; }
        for (int j = static_cast<int>(threadIdx.x); j < top_k; j += kThreads) {
            out_ids[j]     = -1;
            out_weights[j] = 0.0F;
        }
        return;
    }

    cluster.sync(); // every block of the cluster runs before its shared memory is written
    float* leader_score    = cluster.map_shared_rank(score, 0);
    float* leader_key      = cluster.map_shared_rank(key, 0);
    const __nv_bfloat16* u = x + static_cast<std::int64_t>(column) * hidden;
    for (int e = rank * kWarps + warp; e < experts; e += kCluster * kWarps) {
        const float* row = router + static_cast<std::int64_t>(e) * hidden;
        float sum        = 0.0F;
        for (int h = lane; h < hidden; h += 32) { sum = fmaf(row[h], __bfloat162float(u[h]), sum); }
        sum = warp_sum(sum);
        if (lane == 0) {
            const float s   = 1.0F / (1.0F + expf(-sum));
            leader_score[e] = s;
            leader_key[e]   = s + bias[e];
        }
    }
    cluster.sync();
    if (rank != 0) { return; }

    if (experts <= kThreads) {
        // One pass: an expert's place in the order (key, id) is the count of experts before it;
        // the order is total, so the places are distinct and the first k are the k rounds' picks.
        const int e = static_cast<int>(threadIdx.x);
        if (e < experts) {
            const float mine = key[e];
            int place        = 0;
#pragma unroll 4
            for (int f = 0; f < experts; ++f) { place += better(key[f], f, mine, e) ? 1 : 0; }
            if (place < top_k) { chosen[place] = e; }
        }
        __syncthreads();
    } else {
        select_rounds<kWarps>(key, experts, top_k, chosen, best_key_of, best_id_of);
    }

    if (threadIdx.x == 0) {
        float total = 0.0F;
        for (int j = 0; j < top_k; ++j) { total += score[chosen[j]]; }
        const float denominator = total + 1e-20F;
        for (int j = 0; j < top_k; ++j) {
            out_ids[j]     = chosen[j];
            out_weights[j] = score[chosen[j]] / denominator * route_scale;
        }
    }
}

// Wide calls: one block per kTileColumns columns. Warp w evaluates experts kTileExperts at a time
// for all the block's columns, the columns' inputs staged in shared memory: a router row is read
// once per block instead of once per column, and every logit keeps the narrow kernel's lane
// chains (lane l sums h = l, l + 32, ... in order) and butterfly sum, so the same bits. Each
// column then selects by place in the (key, id) order.
constexpr int kTileColumns = 8;
// Below this a block per column is faster: the tile's time is a block's latency, not its reads.
constexpr int kTileMinColumns = 128;
constexpr int kTileExperts    = 4;
constexpr int kTileWarps      = 16;
constexpr int kTileSteps      = 8; // h steps of 32 whose router loads are issued together
constexpr int kTileThreads    = kTileWarps * 32;
// Dynamic shared memory of the tile kernel without opting in (its static part is ~2 KiB).
constexpr std::size_t kTileMaxSharedBytes = 46 * 1024;

std::size_t tile_input_bytes(std::int32_t hidden) {
    return (static_cast<std::size_t>(kTileColumns) * hidden * 2 + 15) / 16 * 16;
}

std::size_t tile_shared_bytes(std::int32_t hidden, std::int32_t experts) {
    return tile_input_bytes(hidden) + 2 * static_cast<std::size_t>(kTileColumns) * experts * 4;
}

__global__ void __launch_bounds__(kTileThreads) moe_sigmoid_route_tile_kernel(
    const __nv_bfloat16* __restrict__ x, const float* __restrict__ router,
    const float* __restrict__ bias, std::int32_t hidden, std::int32_t experts, std::int32_t top_k,
    float route_scale, std::int32_t* __restrict__ ids, float* __restrict__ weights,
    const std::int32_t* __restrict__ valid_columns, std::int32_t lane_width, std::int32_t columns,
    std::uint32_t input_bytes) {
    extern __shared__ unsigned char tile_shared[];
    auto* s_u      = reinterpret_cast<__nv_bfloat16*>(tile_shared);       // [kTileColumns][H]
    float* s_score = reinterpret_cast<float*>(tile_shared + input_bytes); // [kTileColumns][E]
    float* s_key   = s_score + kTileColumns * experts;                    // [kTileColumns][E]
    __shared__ int chosen[kTileColumns][kMoeSigmoidRouteMaxTopK];
    __shared__ bool live[kTileColumns];

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int c0   = static_cast<int>(blockIdx.x) * kTileColumns;
    const int n    = min(kTileColumns, columns - c0);
    if (threadIdx.x < kTileColumns) {
        const int t = c0 + static_cast<int>(threadIdx.x);
        live[threadIdx.x] =
            static_cast<int>(threadIdx.x) < n &&
            (valid_columns == nullptr || t % lane_width < valid_columns[t / lane_width]);
    }
    const __nv_bfloat16 zero = __float2bfloat16_rn(0.0F);
    const __nv_bfloat16* xs  = x + static_cast<std::int64_t>(c0) * hidden;
    if (hidden % 8 == 0) {
        // 16-byte copies (x is 16-byte aligned and every column starts on a multiple of 8).
        const int vectors = kTileColumns * hidden / 8, live_vectors = n * hidden / 8;
        auto* dst       = reinterpret_cast<uint4*>(s_u);
        const auto* src = reinterpret_cast<const uint4*>(xs);
        for (int i = static_cast<int>(threadIdx.x); i < vectors; i += kTileThreads) {
            dst[i] = i < live_vectors ? src[i] : make_uint4(0U, 0U, 0U, 0U);
        }
    } else {
        for (int i = static_cast<int>(threadIdx.x); i < kTileColumns * hidden; i += kTileThreads) {
            s_u[i] = i < n * hidden ? xs[i] : zero;
        }
    }
    __syncthreads();

    for (int e0 = warp * kTileExperts; e0 < experts; e0 += kTileWarps * kTileExperts) {
        float acc[kTileExperts][kTileColumns];
#pragma unroll
        for (int j = 0; j < kTileExperts; ++j) {
#pragma unroll
            for (int c = 0; c < kTileColumns; ++c) { acc[j][c] = 0.0F; }
        }
        const auto step = [&](int h, const float (&w)[kTileExperts]) {
#pragma unroll
            for (int c = 0; c < kTileColumns; ++c) {
                const float a = __bfloat162float(s_u[c * hidden + h]);
#pragma unroll
                for (int j = 0; j < kTileExperts; ++j) { acc[j][c] = fmaf(w[j], a, acc[j][c]); }
            }
        };
        const auto load = [&](int h, float (&w)[kTileExperts]) {
#pragma unroll
            for (int j = 0; j < kTileExperts; ++j) {
                w[j] = e0 + j < experts ? router[static_cast<std::int64_t>(e0 + j) * hidden + h]
                                        : 0.0F;
            }
        };
        int h = lane;
        for (; h + 32 * (kTileSteps - 1) < hidden; h += 32 * kTileSteps) {
            float w[kTileSteps][kTileExperts];
#pragma unroll
            for (int i = 0; i < kTileSteps; ++i) { load(h + 32 * i, w[i]); }
#pragma unroll
            for (int i = 0; i < kTileSteps; ++i) { step(h + 32 * i, w[i]); }
        }
        for (; h < hidden; h += 32) {
            float w[kTileExperts];
            load(h, w);
            step(h, w);
        }
#pragma unroll
        for (int j = 0; j < kTileExperts; ++j) {
#pragma unroll
            for (int c = 0; c < kTileColumns; ++c) {
                const float sum = warp_sum(acc[j][c]);
                if (lane == 0 && e0 + j < experts) {
                    const float s                 = 1.0F / (1.0F + expf(-sum));
                    s_score[c * experts + e0 + j] = s;
                    s_key[c * experts + e0 + j]   = s + bias[e0 + j];
                }
            }
        }
    }
    __syncthreads();

    for (int i = static_cast<int>(threadIdx.x); i < n * experts; i += kTileThreads) {
        const int c = i / experts, e = i - c * experts;
        if (!live[c]) { continue; }
        const float* key = s_key + c * experts;
        const float mine = key[e];
        int place        = 0;
#pragma unroll 4
        for (int f = 0; f < experts; ++f) { place += better(key[f], f, mine, e) ? 1 : 0; }
        if (place < top_k) { chosen[c][place] = e; }
    }
    __syncthreads();

    if (static_cast<int>(threadIdx.x) < n) {
        const int c           = static_cast<int>(threadIdx.x);
        std::int32_t* out_ids = ids + static_cast<std::int64_t>(c0 + c) * top_k;
        float* out_weights    = weights + static_cast<std::int64_t>(c0 + c) * top_k;
        const float* score    = s_score + c * experts;
        if (!live[c]) {
            for (int j = 0; j < top_k; ++j) {
                out_ids[j]     = -1;
                out_weights[j] = 0.0F;
            }
            return;
        }
        float total = 0.0F;
        for (int j = 0; j < top_k; ++j) { total += score[chosen[c][j]]; }
        const float denominator = total + 1e-20F;
        for (int j = 0; j < top_k; ++j) {
            out_ids[j]     = chosen[c][j];
            out_weights[j] = score[chosen[c][j]] / denominator * route_scale;
        }
    }
}

} // namespace

void moe_sigmoid_route_launch(const Tensor& x, const Tensor& router, const Tensor& bias,
                              std::int32_t top_k, float route_scale, Tensor& ids, Tensor& weights,
                              const Tensor* valid_columns, cudaStream_t stream) {
    const std::int32_t hidden  = x.ne[0];
    const std::int32_t columns = x.ne[1];
    const std::int32_t experts = router.ne[1];
    const std::int32_t lane_width =
        valid_columns != nullptr ? columns / valid_columns->ne[0] : columns;
    const std::size_t shared_bytes = 2 * static_cast<std::size_t>(experts) * sizeof(float);
    const auto* valid =
        valid_columns != nullptr ? static_cast<const std::int32_t*>(valid_columns->data) : nullptr;
    const auto launch = [&](auto kernel, int cluster, int warps) {
        launch_maybe_dependent(
            columns <= kDependentLaunchMaxTokens, dim3(static_cast<unsigned>(columns * cluster)),
            dim3(static_cast<unsigned>(warps * 32)), shared_bytes, stream, kernel,
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const float*>(router.data),
            static_cast<const float*>(bias.data), hidden, experts, top_k, route_scale,
            static_cast<std::int32_t*>(ids.data), static_cast<float*>(weights.data), valid,
            lane_width);
    };
    const std::size_t tile_bytes = tile_shared_bytes(hidden, experts);
    if (columns < kClusterMaxColumns) {
        launch(moe_sigmoid_route_kernel<8, 16>, 8, 16);
    } else if (columns >= kTileMinColumns && experts <= kTileThreads &&
               tile_bytes <= kTileMaxSharedBytes) {
        const auto blocks = static_cast<unsigned>((columns + kTileColumns - 1) / kTileColumns);
        moe_sigmoid_route_tile_kernel<<<blocks, kTileThreads, tile_bytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const float*>(router.data),
            static_cast<const float*>(bias.data), hidden, experts, top_k, route_scale,
            static_cast<std::int32_t*>(ids.data), static_cast<float*>(weights.data), valid,
            lane_width, columns, static_cast<std::uint32_t>(tile_input_bytes(hidden)));
    } else {
        launch(moe_sigmoid_route_kernel<1, 8>, 1, 8);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
