#include "ops/linear/nvfp4/nvfp4_shape_families.cuh"

namespace ninfer::ops::detail {
// The two-device row half of the GDN q|k|v|z parent at hidden 5120 [8192,5120] as a plain Linear
// problem (the fused GDN input projection has its own kernels for it): the nvfp4_gdn_qkvz
// selection, not re-measured at this shape; the widest prompts take the resident A4 MMA tile (no
// TMA geometry is registered).
const Nvfp4LinearShape kNvfp4N8192K5120 = nvfp4_gdn_qkvz::shape<Nvfp4N8192K5120>;
} // namespace ninfer::ops::detail
