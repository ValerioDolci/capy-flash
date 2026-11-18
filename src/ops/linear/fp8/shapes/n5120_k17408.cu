#include "ops/linear/fp8/fp8_shape_families.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Fp8Geometry<5120, 17408>;
using namespace fp8_down;

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    if (max_tokens > 384) return Bulk::kPartialBytes;
    if (max_tokens > 256) return Wide::kPartialBytes;
    if (max_tokens > 128) return Mid::kPartialBytes;
    return max_tokens > 64 ? Small::kPartialBytes : 0;
}
} // namespace

const Fp8LinearShape kFp8N5120K17408{5120,
                                     17408,
                                     fp8_down::launch_a16<Geometry>,
                                     fp8_down::launch_a8<Geometry>,
                                     uses_a8,
                                     partial_capacity_bytes};
} // namespace ninfer::ops::detail
