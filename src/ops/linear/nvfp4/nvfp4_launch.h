#pragma once
#include "core/weight.h"
#include "core/tensor.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear/nvfp4/nvfp4_layout.h"
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {
using Nvfp4Launch   = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);
using Nvfp4A4Launch = void (*)(const Weight&, Tensor&, Nvfp4A4Workspace, std::int32_t,
                               cudaStream_t);

// A selected A4 route: the GEMM to run, and the activation-scale layout that GEMM reads. They
// travel together because the quantizer writes the plane before the GEMM runs and the two must
// agree; a shape selects a route rather than selecting the two halves separately.
struct Nvfp4A4Route {
    Nvfp4A4Launch launch;
    Nvfp4ScaleLayout scales;
};
} // namespace ninfer::ops::detail
