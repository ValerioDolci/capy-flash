#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using Q4LinearAddLaunch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

// The registered Q4 residual problem ([5120,6144]); select_q4_linear_add() admits it at every T.
bool q4_linear_add_admits(std::int32_t rows, std::int32_t k) noexcept;

Q4LinearAddLaunch select_q4_linear_add(std::int32_t rows, std::int32_t k, std::int32_t tokens);

} // namespace ninfer::ops::detail
