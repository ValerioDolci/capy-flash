#pragma once
#include "ops/softmax_attention/common/head_mapping.cuh"

namespace ninfer::ops::detail {
inline constexpr int kCausalHeadDim = 256;

template <int HeadDim, int QueryHeads, int KVHeads>
struct CausalGeometry : AttentionHeadMapping<QueryHeads, KVHeads> {
    static_assert(HeadDim > 0 && HeadDim % 64 == 0);
    static constexpr int kHeadDim = HeadDim;
};

using CausalD256H24Kv4 = CausalGeometry<256, 24, 4>;
// One device's half of [256,24,4] under two-device tensor parallelism (BF16, INT8 and NVFP4
// caches).
using CausalD256H12Kv2 = CausalGeometry<256, 12, 2>;
} // namespace ninfer::ops::detail
