#pragma once

#include "ops/common/memory.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Grouping steps shared by the grouped expert Ops (moe_grouped, moe_grouped_gguf). Assignment
// a = column * TopK + k; rank[a] orders a among the assignments of its expert (-1: no slot) and
// offsets[e] is the first grouped row of expert e.

// packed_index[a] = the grouped row of assignment a (-1 without a slot); packed_token[row] = its
// column.
template <int TopK>
__global__ void moe_grouping_place_kernel(const std::int32_t* __restrict__ ids,
                                          const std::int32_t* __restrict__ rank,
                                          std::int32_t assignments,
                                          const std::int32_t* __restrict__ offsets,
                                          std::int32_t* __restrict__ packed_token,
                                          std::int32_t* __restrict__ packed_index) {
    const int a = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (a >= assignments) { return; }
    const int r = rank[a];
    if (r < 0) {
        packed_index[a] = -1;
        return;
    }
    const int position     = offsets[ids[a]] + r;
    packed_token[position] = a / TopK;
    packed_index[a]        = position;
}

// out[column] = sum over k in top-k order of weight * grouped[packed_index], FP32 fmaf. One CTA
// per column; thread i owns features [8i, 8i+8). Routes without a slot add nothing.
template <int Hidden, int TopK>
__global__ void __launch_bounds__(Hidden / 8)
    moe_grouping_merge_kernel(const float* __restrict__ grouped,
                              const std::int32_t* __restrict__ packed_index,
                              const float* __restrict__ weights, float* __restrict__ out) {
    const int column  = static_cast<int>(blockIdx.x);
    const int feature = static_cast<int>(threadIdx.x) * 8;
    float sum[8]      = {};
#pragma unroll
    for (int k = 0; k < TopK; ++k) {
        const int row = packed_index[column * TopK + k];
        if (row < 0) { continue; }
        const float weight  = weights[column * TopK + k];
        const float* source = grouped + static_cast<std::int64_t>(row) * Hidden + feature;
        const float4 a      = load_vec<float4>(source);
        const float4 b      = load_vec<float4>(source + 4);
        const float v[8]    = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) { sum[i] = fmaf(weight, v[i], sum[i]); }
    }
    float* destination = out + static_cast<std::int64_t>(column) * Hidden + feature;
    store_vec(destination, make_float4(sum[0], sum[1], sum[2], sum[3]));
    store_vec(destination + 4, make_float4(sum[4], sum[5], sum[6], sum[7]));
}

} // namespace ninfer::ops::detail
