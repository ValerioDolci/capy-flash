#include "ops/linear/nvfp4/nvfp4_untuned_shape.cuh"

namespace ninfer::ops::detail {
// Qwen3.8-Flash-Next: GDN/QSA output, one TP2 rank (6144/2 input columns).
const Nvfp4LinearShape kNvfp4N2560K3072 =
    nvfp4_untuned::shape<2560, 3072, nvfp4_untuned::NarrowRows>;
} // namespace ninfer::ops::detail
