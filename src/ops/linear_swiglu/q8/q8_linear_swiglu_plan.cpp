#include "core/weight.h"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_plan.h"

#include "ops/linear_swiglu/q8/q8_linear_swiglu_kernels.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {

namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    Q8LinearSwiGluScheduleId schedule;
};

// Small-T K-split followed by row-tiled MMA; live columns do not select exact-T kernels.
constexpr std::array<RouteSpec, 6> kDFlash2Routes{{
    {1, 40, Q8LinearSwiGluScheduleId::DFlash2SmallT},
    {41, 63, Q8LinearSwiGluScheduleId::DFlash2MmaR32C64K128},
    {64, 64, Q8LinearSwiGluScheduleId::DFlash2MmaR64C64K128},
    {65, 80, Q8LinearSwiGluScheduleId::DFlash2MmaR64C80K128},
    {81, 96, Q8LinearSwiGluScheduleId::DFlash2MmaR64C96K128},
    {97, kAnyCols, Q8LinearSwiGluScheduleId::DFlash2MmaR64C128},
}};

template <std::size_t N>
constexpr bool catalog_is_closed(const std::array<RouteSpec, N>& routes) {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.first != expected || route.first > route.last) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(kDFlash2Routes),
              "Q8 DFlash2 LinearSwiGLU routes must be exact and closed");

bool is_dflash2_shape(const Q8LinearSwiGluProblem& problem) noexcept {
    return problem.gate_up_rows == 34816 && problem.output_rows == 17408 && problem.k == 5120 &&
           problem.padded_k == 5120;
}

} // namespace

const char* q8_linear_swiglu_schedule_name(Q8LinearSwiGluScheduleId schedule) noexcept {
    switch (schedule) {
    case Q8LinearSwiGluScheduleId::DFlash2SmallT:
        return "linear_swiglu.q8.dflash2.small_t";
    case Q8LinearSwiGluScheduleId::DFlash2MmaR32C64K128:
        return "linear_swiglu.q8.dflash2.mma.r32.c64.k128";
    case Q8LinearSwiGluScheduleId::DFlash2MmaR64C64K128:
        return "linear_swiglu.q8.dflash2.mma.r64.c64.k128";
    case Q8LinearSwiGluScheduleId::DFlash2MmaR64C80K128:
        return "linear_swiglu.q8.dflash2.mma.r64.c80.k128";
    case Q8LinearSwiGluScheduleId::DFlash2MmaR64C96K128:
        return "linear_swiglu.q8.dflash2.mma.r64.c96.k128";
    case Q8LinearSwiGluScheduleId::DFlash2MmaR64C128:
        return "linear_swiglu.q8.dflash2.mma.pair.r32.c128";
    }
    return "linear_swiglu.q8.unknown";
}

bool q8_linear_swiglu_admits(const Q8LinearSwiGluProblem& problem) noexcept {
    return is_dflash2_shape(problem) && problem.cols > 0;
}

Q8LinearSwiGluPlan q8_linear_swiglu_resolve_plan(const Q8LinearSwiGluProblem& problem) {
    if (!q8_linear_swiglu_admits(problem)) {
        throw std::invalid_argument(
            "Q8 LinearSwiGLU: exact problem or column count is not admitted");
    }
    for (const RouteSpec& route : kDFlash2Routes) {
        if (problem.cols >= route.first && problem.cols <= route.last) { return {route.schedule}; }
    }
    throw std::logic_error("Q8 LinearSwiGLU: admitted problem has no route");
}

void q8_linear_swiglu_execute_plan(const Q8LinearSwiGluPlan& plan, const Tensor& x, const Weight& w,
                                   Tensor& out, cudaStream_t stream) {
    const Q8LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q8LinearSwiGluPlan resolved = q8_linear_swiglu_resolve_plan(problem);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("Q8 LinearSwiGLU: plan does not match exact problem");
    }
    switch (plan.schedule) {
    case Q8LinearSwiGluScheduleId::DFlash2SmallT:
        q8_dflash2_linear_swiglu_small_t_launch(x, w, out, stream);
        return;
    case Q8LinearSwiGluScheduleId::DFlash2MmaR32C64K128:
        q8_dflash2_linear_swiglu_mma_r32_c64_k128_launch(x, w, out, stream);
        return;
    case Q8LinearSwiGluScheduleId::DFlash2MmaR64C64K128:
        q8_dflash2_linear_swiglu_mma_r64_c64_k128_launch(x, w, out, stream);
        return;
    case Q8LinearSwiGluScheduleId::DFlash2MmaR64C80K128:
        q8_dflash2_linear_swiglu_mma_r64_c80_k128_launch(x, w, out, stream);
        return;
    case Q8LinearSwiGluScheduleId::DFlash2MmaR64C96K128:
        q8_dflash2_linear_swiglu_mma_r64_c96_k128_launch(x, w, out, stream);
        return;
    case Q8LinearSwiGluScheduleId::DFlash2MmaR64C128:
        q8_linear_swiglu_mma_r64_c128_launch(x, w, out, stream);
        return;
    }
    throw std::logic_error("Q8 LinearSwiGLU: unknown schedule");
}

void q8_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const Q8LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    q8_linear_swiglu_execute_plan(q8_linear_swiglu_resolve_plan(problem), x, w, out, stream);
}

} // namespace ninfer::ops::detail
