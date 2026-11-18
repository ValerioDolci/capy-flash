#include "ops/linear/nvfp4/nvfp4_shape_families.cuh"

namespace ninfer::ops::detail {
// Two-device input-column half of the dense FFN-14336 down projection [5120,14336]: the
// nvfp4_down_half selection, not re-measured at this shape; the largest prompt widths take the
// resident A4 MMA tile (no TMA geometry is registered for it).
const Nvfp4LinearShape kNvfp4N5120K7168 = nvfp4_down_half::shape<Nvfp4Geometry<5120, 7168>>;
} // namespace ninfer::ops::detail
