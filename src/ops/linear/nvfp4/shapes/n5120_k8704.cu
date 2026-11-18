#include "ops/linear/nvfp4/nvfp4_shape_families.cuh"

namespace ninfer::ops::detail {
// Two-device input-column half of [5120,17408]: the measured shape of nvfp4_down_half, with its
// TMA GEMM and its AllowA4Decode selection. 8704 = 17 * 512 = 34 * 256 = 68 K128 tiles, so every
// instance of the family keeps whole K tiles.
const Nvfp4LinearShape kNvfp4N5120K8704 =
    nvfp4_down_half::shape_with_tma_decode<Nvfp4N5120K8704, Nvfp4GeometryId::N5120K8704>;
} // namespace ninfer::ops::detail
