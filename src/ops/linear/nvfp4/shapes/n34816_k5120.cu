#include "ops/linear/nvfp4/nvfp4_shape_families.cuh"

namespace ninfer::ops::detail {
// The MLP gate|up family's measured shape (nvfp4_gate_up), with its TMA GEMM.
const Nvfp4LinearShape kNvfp4N34816K5120 =
    nvfp4_gate_up::shape_with_tma<Nvfp4N34816K5120, Nvfp4GeometryId::N34816K5120>;
} // namespace ninfer::ops::detail
