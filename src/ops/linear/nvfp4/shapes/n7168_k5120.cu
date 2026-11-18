#include "ops/linear/nvfp4/nvfp4_shape_families.cuh"

namespace ninfer::ops::detail {
// The two-device row half of the attention q|k|gate|v parent at hidden 5120 with 24 query and 4
// key/value heads of 256 [7168,5120] as a plain Linear problem (the fused attention input
// projection has its own kernels for it): the nvfp4_attn_qkgv selection, not re-measured at this
// shape; the widest prompts take the resident A4 MMA tile (no TMA geometry is registered).
const Nvfp4LinearShape kNvfp4N7168K5120 = nvfp4_attn_qkgv::shape<Nvfp4N7168K5120>;
} // namespace ninfer::ops::detail
