#pragma once

#include "core/weight.h"
#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class Q8LinearAddScheduleId {
    SplitKMmaCapacity,
    GroupedSplitK,
};

struct Q8LinearAddPlan {
    Q8LinearAddScheduleId schedule;
};

struct Q8LinearAddProblem {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
    std::int32_t cols;
};

bool q8_linear_add_admits(const Q8LinearAddProblem& problem) noexcept;
Q8LinearAddPlan q8_linear_add_resolve_plan(const Q8LinearAddProblem& problem);

void q8_linear_add_execute_plan(const Q8LinearAddPlan& plan, const Tensor& x, const Weight& w,
                                Tensor& residual_out, cudaStream_t stream);
void q8_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            cudaStream_t stream);

} // namespace ninfer::ops::detail
