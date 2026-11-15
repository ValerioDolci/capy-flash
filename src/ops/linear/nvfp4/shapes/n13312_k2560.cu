#include "ops/linear/nvfp4/nvfp4_untuned_shape.cuh"

namespace ninfer::ops::detail {
// Qwen3.8-Flash-Next at one device (NVFP4-dense recipe): QSA attention input, query|key|gate|value.
const Nvfp4LinearShape kNvfp4N13312K2560 =
    nvfp4_untuned::shape<13312, 2560, nvfp4_untuned::WideRows>;
} // namespace ninfer::ops::detail
