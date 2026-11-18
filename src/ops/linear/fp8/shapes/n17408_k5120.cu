#include "ops/linear/fp8/fp8_shape_families.cuh"

namespace ninfer::ops::detail {
// Two-device output-row half of [34816,5120]: the fp8_gate_up selection, not re-measured at this
// shape.
namespace {
using Geometry = Fp8Geometry<17408, 5120>;
using namespace fp8_gate_up;

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 5; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    // Reserve one wave for any interval using the bulk route. The allocation is
    // independent of which calls in that interval have an underfilled last wave.
    return max_tokens > 256 ? Bulk::kPartialBytes : 0;
}
} // namespace

const Fp8LinearShape kFp8N17408K5120{17408,
                                     5120,
                                     fp8_gate_up::launch_a16<Geometry>,
                                     fp8_gate_up::launch_a8<Geometry>,
                                     uses_a8,
                                     partial_capacity_bytes};
} // namespace ninfer::ops::detail
