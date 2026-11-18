#include "ops/linear/fp8/fp8_geometry.h"
#include "ops/linear/fp8/fp8_shape_families.cuh"

namespace ninfer::ops::detail {
// Two-device input-column half of [5120,17408]: the fp8_down selection, not re-measured at this
// shape. Its A8 floor is the fused residual projection's (fp8_linear_add_plan.cpp), not the parent
// linear()'s: at tp 2 rank 1 runs this linear() and rank 0 runs linear_add over the other half of
// the same row-parallel projection, so the two ranks take the same activation route at every width.
namespace {
using Geometry = Fp8Geometry<5120, 8704>;
using namespace fp8_down;

bool uses_a8(std::int32_t, std::int32_t max_tokens) {
    return max_tokens >= kFp8DownFamilyFirstA8Tokens;
}

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    if (max_tokens > 384) return Bulk::kPartialBytes;
    if (max_tokens > 256) return Wide::kPartialBytes;
    if (max_tokens > 128) return Mid::kPartialBytes;
    return max_tokens > 64 ? Small::kPartialBytes : 0;
}
} // namespace

const Fp8LinearShape kFp8N5120K8704{5120,
                                    8704,
                                    fp8_down::launch_a16<Geometry>,
                                    fp8_down::launch_a8<Geometry>,
                                    uses_a8,
                                    partial_capacity_bytes};
} // namespace ninfer::ops::detail
