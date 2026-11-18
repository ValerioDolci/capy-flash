#include "ops/linear/nvfp4/nvfp4_shape_families.cuh"

namespace ninfer::ops::detail {
// Two-device output-row half of [34816,5120]: the nvfp4_gate_up selection and its TMA GEMM, not
// re-measured at this shape.
const Nvfp4LinearShape kNvfp4N17408K5120 =
    nvfp4_gate_up::shape_with_tma<Nvfp4N17408K5120, Nvfp4GeometryId::N17408K5120>;
} // namespace ninfer::ops::detail
