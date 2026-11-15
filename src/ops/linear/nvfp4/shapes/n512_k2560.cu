#include "ops/linear/nvfp4/nvfp4_untuned_shape.cuh"

namespace ninfer::ops::detail {
// Qwen3.8-Flash-Next: router (replicated).
const Nvfp4LinearShape kNvfp4N512K2560 = nvfp4_untuned::shape<512, 2560, nvfp4_untuned::WideRows>;
} // namespace ninfer::ops::detail
