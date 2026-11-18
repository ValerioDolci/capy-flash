#include "ops/linear/nvfp4/nvfp4_shape_families.cuh"

namespace ninfer::ops::detail {
// Down projection of a dense FFN of intermediate width 14336 over hidden 5120: the nvfp4_down
// selection, not re-measured at this shape; the largest prompt widths take the resident A4 MMA tile
// instead of a TMA route (no TMA geometry is registered for it).
const Nvfp4LinearShape kNvfp4N5120K14336 = nvfp4_down::shape<Nvfp4Geometry<5120, 14336>>;
} // namespace ninfer::ops::detail
