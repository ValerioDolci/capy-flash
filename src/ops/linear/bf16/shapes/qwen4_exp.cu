#include "ops/linear/bf16/bf16_launch.cuh"
#include "ops/linear/bf16/bf16_shapes.h"

namespace ninfer::ops::detail {
// Qwen3.8-Flash-Next projections kept in BF16 by the FP4-Local checkpoint, at prefill widths:
// hyper-connection Down(+inject) and Up, the MTP dense layer and the PLE key|value parent.
// Dynamic K: one instance serves every K of a family. RowFast rasters keep the activation tile
// in L2 across the row tiles (TokenFast re-streams it per row tile once T*K outgrows L2). Row
// tiles of 48 and 16 do not divide every N (324, 6656, 12800...): the last tile reads zero rows
// and stores only the real ones. Decode widths (T <= 8 for the mixers, 16 for the dense family)
// keep the generic GEMV. Schedules and cutoffs: warm-L2 sweep of 30 schedules on RTX 5070 Ti,
// T 8..16384.
namespace {
using SlicedR16T32W8  = Bf16A16SlicedKMmaSchedule<16, 32, 8>;
using SlicedR32T32W4  = Bf16A16SlicedKMmaSchedule<32, 32, 4>;
using SlicedR48T32W4  = Bf16A16SlicedKMmaSchedule<48, 32, 4>;
using TmaR48T128K64S2 = Bf16A16TmaMmaSchedule<48, 128, 64, 48, 32, 2, 1, Bf16MmaRaster::RowFast>;
using TmaR64T128K64S2 = Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2, 1, Bf16MmaRaster::RowFast>;
} // namespace

// [320|324, 10240]: few rows over a wide K. K-sliced warps fill the GPU until T ~ 512; beyond,
// 48-row tiles (7 of them for 324 rows instead of 6 x 64).
Bf16Launch select_bf16_qwen4_hc_down(std::int32_t tokens) {
    if (tokens <= 8) return launch_bf16_generic;
    if (tokens <= 64) return launch_bf16_sliced_k_mma<SlicedR16T32W8>;
    if (tokens <= 128) return launch_bf16_sliced_k_mma<SlicedR32T32W4>;
    if (tokens <= 512) return launch_bf16_sliced_k_mma<SlicedR48T32W4>;
    return launch_bf16_tma_mma<TmaR48T128K64S2>;
}

// [10240, 320]: K = 320 is five 64-wide stages. The generic tiled kernel is
// still faster for 17 <= T <= 64.
Bf16Launch select_bf16_qwen4_hc_up(std::int32_t tokens) {
    if (tokens <= 8 || (tokens > 16 && tokens <= 64)) return launch_bf16_generic;
    return launch_bf16_tma_mma<TmaR64T128K64S2>;
}

// MTP q|k|gate|v [13312|6656, 2560], indexer [640, 2560], output [2560, 6144|3072], shared
// expert gate|up [1280, 2560] and down [2560, 640], embedding/hidden projections [2560, 2560],
// and the PLE key|value parent [12800, 2560].
template <int Rows, int K>
Bf16Launch select_bf16_qwen4_dense(std::int32_t tokens) {
    if (tokens <= 16) return launch_bf16_generic;
    // K-sliced warps until the row tiles alone fill the GPU (the 12800/13312-row parents at T >
    // 32).
    if constexpr (K % SlicedR48T32W4::kBlockK == 0) {
        if (tokens <= (Rows >= 12800 ? 32 : 64)) return launch_bf16_sliced_k_mma<SlicedR48T32W4>;
    }
    if (tokens <= 256 || (Rows <= 1280 && tokens <= 1024))
        return launch_bf16_tma_mma<TmaR48T128K64S2>;
    return launch_bf16_tma_mma<TmaR64T128K64S2>;
}

template Bf16Launch select_bf16_qwen4_dense<13312, 2560>(std::int32_t);
template Bf16Launch select_bf16_qwen4_dense<12800, 2560>(std::int32_t);
template Bf16Launch select_bf16_qwen4_dense<6656, 2560>(std::int32_t);
template Bf16Launch select_bf16_qwen4_dense<2560, 6144>(std::int32_t);
template Bf16Launch select_bf16_qwen4_dense<2560, 3072>(std::int32_t);
template Bf16Launch select_bf16_qwen4_dense<2560, 2560>(std::int32_t);
template Bf16Launch select_bf16_qwen4_dense<2560, 640>(std::int32_t);
template Bf16Launch select_bf16_qwen4_dense<1280, 2560>(std::int32_t);
template Bf16Launch select_bf16_qwen4_dense<640, 2560>(std::int32_t);
} // namespace ninfer::ops::detail
