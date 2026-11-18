#include "ops/linear/fp8/fp8_shape_families.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Fp8Geometry<16384, 5120>;
using namespace fp8_gdn_qkvz;

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    return max_tokens > 256 ? Bulk::kPartialBytes : 0;
}
} // namespace

const Fp8LinearShape kFp8N16384K5120{16384,
                                     5120,
                                     fp8_gdn_qkvz::launch_a16<Geometry>,
                                     fp8_gdn_qkvz::launch_a8<Geometry>,
                                     uses_a8,
                                     partial_capacity_bytes};
} // namespace ninfer::ops::detail
