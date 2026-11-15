#include "ops/linear/nvfp4/nvfp4_untuned_shape.cuh"

namespace ninfer::ops::detail {
// Qwen3.8-Flash-Next at one device (NVFP4-dense recipe): QSA and GDN output projections.
const Nvfp4LinearShape kNvfp4N2560K6144 =
    nvfp4_untuned::shape<2560, 6144, nvfp4_untuned::NarrowRows>;
} // namespace ninfer::ops::detail
