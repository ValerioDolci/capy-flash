#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"

namespace ninfer::ops::detail {
// Qwen3.8-Flash-Next: GDN in_proj qkv+z, one TP2 rank (10240/2 + 6144/2). A16 only (decode);
// schedules inherited from nvfp4_attn_qkgv (nvfp4_shape_families.cuh), not re-measured at this
// shape. No A4 route: uses_a4 is always false.
namespace {
using Geometry = Nvfp4Geometry<8192, 2560>;
using Gemv =
    Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
using C2 = Nvfp4A16SimtSchedule<8, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::SharedPhase,
                                Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                Nvfp4SimtBlockOrder::RowsContiguous, 1>;
// 32-row CTAs at T <= 8: 22.5 us instead of 24.6 us with 16-row CTAs at T = 4.
using C8 = Nvfp4A16SlicedKMmaSchedule<8, 8, 2, Cache::ca, Cache::cg,
                                      Nvfp4ActivationStage::PaddedZero, 2, 2>;

Nvfp4Launch select_a16(int tokens) {
    if (tokens == 1) return nvfp4_linear_a16_gemv<Geometry, Gemv>;
    if (tokens <= 2) return nvfp4_linear_a16_simt<Geometry, 2, C2, true>;
    if (tokens <= 8) return nvfp4_linear_a16_sliced_k<Geometry, C8>;
    if (tokens <= 16) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<16, 8, 2>>;
    if (tokens <= 64)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>;
    return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}

bool uses_a4(std::int32_t, std::int32_t) { return false; }
} // namespace

const Nvfp4LinearShape kNvfp4N8192K2560{8192, 2560, select_a16, nullptr, uses_a4};
} // namespace ninfer::ops::detail
