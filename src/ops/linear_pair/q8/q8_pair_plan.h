#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Q8PairScheduleId {
    TwoSimtR8C4,
    DualMmaR32C64,
    DualMmaR32C128,
};

struct Q8PairProblem {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q8PairPlan {
    Q8PairScheduleId schedule;
};

Q8PairProblem q8_pair_problem(const Tensor& x, const Weight& first_weight,
                              const Tensor& first_out) noexcept;
bool q8_pair_admits(const Q8PairProblem& problem) noexcept;
Q8PairPlan q8_pair_resolve_plan(const Q8PairProblem& problem);

void q8_pair_execute_plan(Q8PairPlan plan, const Tensor& x, const Weight& first_weight,
                          const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                          cudaStream_t stream);
void q8_pair_dispatch(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                      Tensor& first_out, Tensor& second_out, cudaStream_t stream);

} // namespace ninfer::ops::detail
