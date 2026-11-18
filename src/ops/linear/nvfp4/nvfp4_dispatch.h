#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_shapes.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops::detail {

// The launch nvfp4_dispatch() takes over [n,k] at `tokens` columns: the shape's A16 launch, its
// AllowA4Decode A16 launch (Nvfp4LinearShape::a16_decode, below the A4 crossover and at most
// kAllowA4DecodeMaxWidth columns) or A4.
enum class Nvfp4LinearRoute : std::uint8_t {
    A16,
    A16Decode,
    A4,
};

[[nodiscard]] Nvfp4LinearRoute nvfp4_linear_route(std::int32_t output_rows, std::int32_t input_rows,
                                                  LinearPolicy policy, std::int32_t tokens);

// The launch nvfp4_dispatch() runs over [n,k] at `tokens` columns: `a16` on the A16 and
// A16Decode routes, `a4` (the GEMM and the scale layout the quantizer writes for it) on A4.
struct Nvfp4LinearSelection {
    Nvfp4LinearRoute route;
    Nvfp4Launch a16 = nullptr;
    Nvfp4A4Route a4{};
};

[[nodiscard]] Nvfp4LinearSelection nvfp4_linear_selection(std::int32_t output_rows,
                                                          std::int32_t input_rows,
                                                          LinearPolicy policy, std::int32_t tokens);

// Every registered [n,k] problem, in registration order.
[[nodiscard]] std::span<const Nvfp4LinearShape* const> nvfp4_linear_shapes();

[[nodiscard]] std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t output_rows,
                                                                std::int32_t input_rows,
                                                                LinearPolicy policy,
                                                                std::int32_t min_tokens,
                                                                std::int32_t max_tokens);

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
