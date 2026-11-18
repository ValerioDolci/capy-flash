#include "ops/linear/fp8/fp8_shape_families.cuh"

namespace ninfer::ops::detail {
// Two-device output-row half of [14336,5120]: the fp8_attn_qkgv selection, not re-measured at this
// shape.
namespace {
using Geometry = Fp8Geometry<7168, 5120>;
using namespace fp8_attn_qkgv;

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    // Unlike the parent's 56 row tiles, the half's 28 leave Bulk a split-K tail wave from its
    // first width on (T=289: 84 tiles).
    return max_tokens > 288 ? Bulk::kPartialBytes : 0;
}
} // namespace

const Fp8LinearShape kFp8N7168K5120{7168,
                                    5120,
                                    fp8_attn_qkgv::launch_a16<Geometry>,
                                    fp8_attn_qkgv::launch_a8<Geometry>,
                                    uses_a8,
                                    partial_capacity_bytes};
} // namespace ninfer::ops::detail
