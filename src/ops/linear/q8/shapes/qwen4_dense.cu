#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_instance_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// Qwen4Exp (Qwen3.8-Flash-Next) dense Q8 projections at decode widths: T <= 16 on a sliced-K MMA
// schedule (8 K-warps, BlockK 512), T >= 17 on the generic routes. T <= 4 runs 4 token columns per
// CTA, T = 5..8 the same schedule with 8 columns per CTA (one pass over the weights), T = 9..16
// the 4-column instance in groups of 4 (grid y). A token column gets the bits of T = 4 at every
// decoding and verification width from 1 to 16 (MTP K = 3 on up to four lanes included).
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
    if (tokens <= 16) return launch_q8_a16_sliced<Geometry, 4, Schedule>;
    return select_q8_generic(tokens);
}
} // namespace

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
