#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// One plan and one set of launchers for every NVFP4 gate|up problem the wrapper registers
// (src/ops/wrapper/linear_swiglu.cpp): [34816,5120] and its two-device half [17408,5120], and the
// dense FFN-14336 gate|up [28672,5120] and its half [14336,5120]. They share K; every launcher
// reads the output rows from the weight. Every route, schedule and token cutoff is the
// [34816,5120] problem's; the FFN-14336 problems inherit them, not re-measured.

enum class Nvfp4LinearSwiGluRoute : std::uint8_t {
    DecodeFusedA16,
    SmallTFusedA16,
    // AllowA4Decode at T=1..7 (nvfp4_linear_swiglu_sliced_k_launch).
    SlicedKFusedA16,
    FusedA4,
    TmaFusedA4,
};

[[nodiscard]] Nvfp4LinearSwiGluRoute nvfp4_linear_swiglu_route(LinearPolicy policy,
                                                               std::int32_t tokens);

[[nodiscard]] std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                                       std::int32_t min_tokens,
                                                                       std::int32_t max_tokens);

void nvfp4_linear_swiglu_decode_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream);
void nvfp4_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
// T=1..8 on the eight-K-warp sliced-K schedules of the small-T launcher's T=3..8, whose reduction
// is the same at every T.
void nvfp4_linear_swiglu_sliced_k_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                         cudaStream_t stream);
using Nvfp4LinearSwiGluA4Launch = void (*)(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                                           cudaStream_t);
// The MMA schedule nvfp4_linear_swiglu_a4_launch() runs at a width, with its quantizer (row-major
// activation scales at every width).
[[nodiscard]] Nvfp4LinearSwiGluA4Launch nvfp4_linear_swiglu_select_a4(std::int32_t tokens);
void nvfp4_linear_swiglu_a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                   WorkspaceArena& workspace, cudaStream_t stream);

// `workspace` may be null when the resolved route needs none; a quantized-activation route then
// throws.
void nvfp4_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                  LinearPolicy policy, WorkspaceArena* workspace,
                                  cudaStream_t stream);

} // namespace ninfer::ops::detail
