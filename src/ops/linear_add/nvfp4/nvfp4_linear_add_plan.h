#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// A16: nvfp4_linear_add_a16_launch; A16Decode: AllowA4Decode below the A4 crossover
// (nvfp4_linear_add_a16_decode_launch); A4: nvfp4_linear_add_a4_launch.
enum class Nvfp4LinearAddRoute : std::uint8_t {
    A16,
    A16Decode,
    A4,
};

[[nodiscard]] Nvfp4LinearAddRoute nvfp4_linear_add_route(std::int32_t output_rows,
                                                         std::int32_t input_rows,
                                                         LinearPolicy policy, std::int32_t tokens);

[[nodiscard]] std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                                    std::int32_t input_rows,
                                                                    LinearPolicy policy,
                                                                    std::int32_t min_tokens,
                                                                    std::int32_t max_tokens);

void nvfp4_linear_add_a16_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                 cudaStream_t stream);

// The A16 route of AllowA4Decode below the A4 crossover: a column gets the bits of width 4 at every
// width. The output family ([5120,6144] and its half) runs the matrix schedules at every T (the
// eight-K-warp sliced-K schedule of T=3..8 at T=1 and 2 too), the MLP down family ([5120,17408]
// and its half) the SIMT small-T kernel at every T (the decode kernel at T=1 and the sliced-K
// schedule at T=6..7 otherwise). linear() over the other half of a row-parallel projection does
// the same (Nvfp4LinearShape::a16_decode).
void nvfp4_linear_add_a16_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                        cudaStream_t stream);

void nvfp4_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                    cudaStream_t stream);

void nvfp4_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                     cudaStream_t stream);

using Nvfp4LinearAddA4Launch = void (*)(const Weight&, Tensor&, Nvfp4A4Workspace, std::int32_t,
                                        cudaStream_t);

// The A4 route's GEMM at a width and the activation-scale layout it reads; the quantizer writes
// that layout before the GEMM runs, so the two are selected together.
struct Nvfp4LinearAddA4Route {
    Nvfp4LinearAddA4Launch launch;
    Nvfp4ScaleLayout scales;
};

// The selection of nvfp4_linear_add_a4_launch() for the problem [output_rows,input_rows]; throws
// for a problem without A4 schedules.
[[nodiscard]] Nvfp4LinearAddA4Route
nvfp4_linear_add_select_a4(std::int32_t output_rows, std::int32_t input_rows, std::int32_t tokens);

void nvfp4_linear_add_a4_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                Nvfp4A4Workspace workspace, cudaStream_t stream);

// `workspace` may be null when the resolved route needs none; the A4 route then throws.
void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                               LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
