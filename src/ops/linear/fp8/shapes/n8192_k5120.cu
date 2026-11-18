#include "ops/linear/fp8/fp8_shape_families.cuh"

namespace ninfer::ops::detail {
// Two-device output-row half of [16384,5120]: the fp8_gdn_qkvz selection, not re-measured at this
// shape.
namespace {
using Geometry = Fp8Geometry<8192, 5120>;
using namespace fp8_gdn_qkvz;

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    // Unlike the parent's 64 row tiles, the half's 32 leave Bulk a split-K tail wave from its
    // first width on (T=193: 64 tiles); MidBulk needs less.
    return max_tokens > 192 ? Bulk::kPartialBytes : 0;
}
} // namespace

const Fp8LinearShape kFp8N8192K5120{8192,
                                    5120,
                                    fp8_gdn_qkvz::launch_a16<Geometry>,
                                    fp8_gdn_qkvz::launch_a8<Geometry>,
                                    uses_a8,
                                    partial_capacity_bytes};
} // namespace ninfer::ops::detail
