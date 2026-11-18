#include "ops/linear/nvfp4/nvfp4_shape_families.cuh"

namespace ninfer::ops::detail {
// The MLP down family's measured shape (nvfp4_down), with its TMA GEMM and its AllowA4Decode
// selection.
const Nvfp4LinearShape kNvfp4N5120K17408 =
    nvfp4_down::shape_with_tma_decode<Nvfp4N5120K17408, Nvfp4GeometryId::N5120K17408>;
} // namespace ninfer::ops::detail
