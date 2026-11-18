#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_tiles.h"

namespace ninfer::ops::detail {
// Two-device input-column half of [5120,6144]. It inherits the parent's measured schedules and TMA
// threshold (n5120_k6144.cu); they have not been re-measured at this shape. Its A4 floor is the
// attention/GDN output family's constant, which the fused residual projection
// (nvfp4_linear_add_plan.cpp) reads too: at tp 2 rank 1 runs this linear() and rank 0 runs
// linear_add over the other half of the same row-parallel projection, so the two ranks take the
// same activation route at every width, as tp 1's single linear_add [5120,6144] does.
// 3072 = 6 * 512 = 12 * 256, so every parent instance keeps whole K tiles.
namespace {
using Geometry = Nvfp4Geometry<5120, 3072>;
using Gemv =
    Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
using C2 = Nvfp4A16SimtSchedule<4, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                Nvfp4SimtBlockOrder::RowsContiguous, 1>;

Nvfp4Launch select_a16(int tokens) {
    if (tokens == 1) return nvfp4_linear_a16_gemv<Geometry, Gemv>;
    if (tokens <= 2) return nvfp4_linear_a16_simt<Geometry, 2, C2, true>;
    if (tokens <= 8) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<8, 8, 2>>;
    if (tokens <= 16) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<16, 8, 2>>;
    if (tokens <= 24) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 8, 1>>;
    if (tokens <= 32) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 4, 2>>;
    if (tokens <= 48) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 2, 2>>;
    if (tokens <= 64) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<64, 2, 2>>;
    if (tokens <= 128)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 2, 1>>;
    return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}

// AllowA4Decode at T=1..7: the eight-K-warp sliced-K schedule of T=3..8 at every width, instead of
// the GEMV at T=1 and the SIMT kernel at T=2. Its reduction depends on the K-warp count only
// (nvfp4_a16_sliced_k_mma.cuh), so every width gives a column the bits of T=4.
// linear_add over the other half does the same (nvfp4_linear_add_a16.cu).
Nvfp4Launch select_a16_decode(std::int32_t) {
    return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<8, 8, 2>>;
}

Nvfp4A4Route select_a4(std::int32_t tokens) {
    if (tokens >= kNvfp4OutputDownFamilyFirstTmaTokens)
        return nvfp4_a4_tma_route<Nvfp4GeometryId::N5120K3072>();
    if (tokens <= 128) return nvfp4_a4_mma_route<Geometry, Nvfp4A4T32R64>();
    if (tokens <= 192) return nvfp4_a4_mma_route<Geometry, Nvfp4A4T64R128>();
    if (tokens <= 384) return nvfp4_a4_mma_route<Geometry, Nvfp4A4T128R128Resident>();
    if (tokens <= 512) return nvfp4_a4_mma_route<Geometry, Nvfp4A4T128R128Pipelined>();
    return nvfp4_a4_mma_route<Geometry, Nvfp4A4T128R128Resident>();
}

bool uses_a4(std::int32_t, std::int32_t max_tokens) {
    return max_tokens >= kNvfp4OutputFamilyFirstA4Tokens;
}
} // namespace

const Nvfp4LinearShape kNvfp4N5120K3072{5120,      3072,    select_a16,
                                        select_a4, uses_a4, select_a16_decode};
} // namespace ninfer::ops::detail
