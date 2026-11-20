#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// Qwen4Exp (Qwen3.8-Flash-Next) dense Q8 projections at decode widths: T <= 8 on a sliced-K MMA
// schedule (8 K-warps, BlockK 512; its 8-token tile serves both capacities, so 5..8 tokens cost
// about what 4 cost), T >= 9 on the generic routes.
using Access = Q8ScaleAccess;
using Stage  = Q8ActivationStage;
using Direct =
    Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Direct, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using Shared =
    Q8A16SlicedKMmaSchedule<8, 8, 2, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;

template <class Geometry, class Schedule>
Q8Launch select(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_sliced<Geometry, 4, Schedule>;
    if (tokens <= 8) return launch_q8_a16_sliced<Geometry, 8, Schedule>;
    return select_q8_generic(tokens);
}

// The K = 2560 row projections: the register-resident warp-row pipeline at T <= 4 (at most four
// tokens), the dense projections' MMA schedule at T 5..8.
template <int N>
Q8Launch select_rows_k2560(std::int32_t tokens) {
    if (tokens <= 4) return select_q8_qwen4_rows_k2560(tokens);
    if (tokens <= 8) return launch_q8_a16_sliced<Q8LinearGeometry<N, 2560>, 8, Direct>;
    return select_q8_generic(tokens);
}
} // namespace

Q8Launch select_q8_qwen4_n16384_k2560(std::int32_t tokens) {
    return select_rows_k2560<16384>(tokens);
}

Q8Launch select_q8_qwen4_n12800_k2560(std::int32_t tokens) {
    return select_rows_k2560<12800>(tokens);
}

Q8Launch select_q8_qwen4_n8192_k2560(std::int32_t tokens) {
    return select_rows_k2560<8192>(tokens);
}

Q8Launch select_q8_qwen4_n2560_k2560(std::int32_t tokens) {
    return select_rows_k2560<2560>(tokens);
}

Q8Launch select_q8_qwen4_n1280_k2560(std::int32_t tokens) {
    return select_rows_k2560<1280>(tokens);
}

Q8Launch select_q8_qwen4_n13312_k2560(std::int32_t tokens) {
    return select<Q8LinearGeometry<13312, 2560>, Direct>(tokens);
}

Q8Launch select_q8_qwen4_n6656_k2560(std::int32_t tokens) {
    return select<Q8LinearGeometry<6656, 2560>, Direct>(tokens);
}

Q8Launch select_q8_qwen4_n2560_k6144(std::int32_t tokens) {
    return select<Q8LinearGeometry<2560, 6144>, Direct>(tokens);
}

Q8Launch select_q8_qwen4_n2560_k3072(std::int32_t tokens) {
    return select<Q8LinearGeometry<2560, 3072>, Direct>(tokens);
}

Q8Launch select_q8_qwen4_n640_k2560(std::int32_t tokens) {
    return select<Q8LinearGeometry<640, 2560>, Shared>(tokens);
}

Q8Launch select_q8_qwen4_n96_k2560(std::int32_t tokens) {
    return select<Q8LinearGeometry<96, 2560>, Direct>(tokens);
}

} // namespace ninfer::ops::detail
