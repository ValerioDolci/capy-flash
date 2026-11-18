#include "ops/linear/nvfp4/nvfp4_shape_families.cuh"

namespace ninfer::ops::detail {
// The GDN q|k|v|z family's measured shape (nvfp4_gdn_qkvz), with its TMA GEMM.
const Nvfp4LinearShape kNvfp4N16384K5120 =
    nvfp4_gdn_qkvz::shape_with_tma<Nvfp4N16384K5120, Nvfp4GeometryId::N16384K5120>;
} // namespace ninfer::ops::detail
