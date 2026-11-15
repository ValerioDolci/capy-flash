#include "ops/linear/nvfp4/nvfp4_untuned_shape.cuh"

namespace ninfer::ops::detail {
// Qwen3.8-Flash-Next: shared expert gate_up, one TP2 rank (1280/2).
const Nvfp4LinearShape kNvfp4N640K2560 = nvfp4_untuned::shape<640, 2560, nvfp4_untuned::WideRows>;
} // namespace ninfer::ops::detail
