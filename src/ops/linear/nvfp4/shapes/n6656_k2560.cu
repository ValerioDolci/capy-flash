#include "ops/linear/nvfp4/nvfp4_untuned_shape.cuh"

namespace ninfer::ops::detail {
// Qwen3.8-Flash-Next: QSA q(+gate)+kv, one TP2 rank ((12288 + 1024)/2).
const Nvfp4LinearShape kNvfp4N6656K2560 = nvfp4_untuned::shape<6656, 2560, nvfp4_untuned::WideRows>;
} // namespace ninfer::ops::detail
