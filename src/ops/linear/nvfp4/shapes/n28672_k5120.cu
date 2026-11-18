#include "ops/linear/nvfp4/nvfp4_shape_families.cuh"

namespace ninfer::ops::detail {
// The gate|up parent of a dense FFN of intermediate width 14336 [28672,5120] as a plain Linear
// problem (LinearSwiGLU has its own route for it): the nvfp4_gate_up selection, not re-measured at
// this shape; the widest prompts take the resident A4 MMA tile (no TMA geometry is registered for
// it).
const Nvfp4LinearShape kNvfp4N28672K5120 = nvfp4_gate_up::shape<Nvfp4N28672K5120>;
} // namespace ninfer::ops::detail
