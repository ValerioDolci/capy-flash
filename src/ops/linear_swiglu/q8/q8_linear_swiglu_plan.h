#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Q8LinearSwiGluScheduleId {
    DFlash2SmallT,
    DFlash2MmaR32C64K128,
    DFlash2MmaR64C64K128,
    DFlash2MmaR64C80K128,
    DFlash2MmaR64C96K128,
    DFlash2MmaR64C128,
};

struct Q8LinearSwiGluProblem {
    std::int32_t gate_up_rows;
    std::int32_t output_rows;
    std::int32_t k;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q8LinearSwiGluPlan {
    Q8LinearSwiGluScheduleId schedule;
};

const char* q8_linear_swiglu_schedule_name(Q8LinearSwiGluScheduleId schedule) noexcept;
bool q8_linear_swiglu_admits(const Q8LinearSwiGluProblem& problem) noexcept;
Q8LinearSwiGluPlan q8_linear_swiglu_resolve_plan(const Q8LinearSwiGluProblem& problem);

void q8_linear_swiglu_execute_plan(const Q8LinearSwiGluPlan& plan, const Tensor& x, const Weight& w,
                                   Tensor& out, cudaStream_t stream);
void q8_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
