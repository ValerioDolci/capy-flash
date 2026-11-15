#include "ops/linear/nvfp4/nvfp4_untuned_shape.cuh"

namespace ninfer::ops::detail {
// Qwen3.8-Flash-Next: LM head, one TP2 rank (248320/2).
const Nvfp4LinearShape kNvfp4N124160K2560 =
    nvfp4_untuned::shape<124160, 2560, nvfp4_untuned::WideRows>;
} // namespace ninfer::ops::detail
