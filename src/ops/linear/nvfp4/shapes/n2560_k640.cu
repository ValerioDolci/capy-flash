#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"

namespace ninfer::ops::detail {
// Qwen3.8-Flash-Next at one device (NVFP4-dense recipe): shared expert down projection. A16 only.
// K = 640 is not a multiple of the 512-value phase of the GEMV and SIMT schedules (32 lanes x 16
// values), so every token count up to 16 takes the sliced-K MMA with two K warps (block K 128,
// five groups); larger counts take the tiled MMA (block K 128 / 64). Not re-measured at this shape.
// No A4 route: uses_a4 is always false.
namespace {
using Geometry = Nvfp4Geometry<2560, 640>;

Nvfp4Launch select_a16(int tokens) {
    if (tokens <= 8) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<8, 2, 2>>;
    if (tokens <= 16) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<16, 2, 2>>;
    if (tokens <= 64)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>;
    return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}

bool uses_a4(std::int32_t, std::int32_t) { return false; }
} // namespace

const Nvfp4LinearShape kNvfp4N2560K640{2560, 640, select_a16, nullptr, uses_a4};
} // namespace ninfer::ops::detail
