#include "ops/linear/nvfp4/nvfp4_untuned_shape.cuh"

namespace ninfer::ops::detail {
// Qwen3.8-Flash-Next at one device (NVFP4-dense recipe): GDN input, query|key|value|z.
const Nvfp4LinearShape kNvfp4N16384K2560 =
    nvfp4_untuned::shape<16384, 2560, nvfp4_untuned::WideRows>;
} // namespace ninfer::ops::detail
