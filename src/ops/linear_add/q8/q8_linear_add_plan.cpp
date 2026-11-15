#include "core/weight.h"
#include "ops/linear_add/q8/q8_linear_add_plan.h"

#include "ops/linear_add/q8/q8_linear_add_kernels.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    Q8LinearAddScheduleId schedule;
};

constexpr std::array<RouteSpec, 2> kN5120Routes{{
    {1, 64, Q8LinearAddScheduleId::SplitKMmaCapacity},
    {65, kAnyCols, Q8LinearAddScheduleId::GroupedSplitK},
}};

template <std::size_t N>
constexpr bool routes_are_closed(const std::array<RouteSpec, N>& routes) {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.first != expected || route.last < route.first) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return routes.back().last == kAnyCols && expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(routes_are_closed(kN5120Routes),
              "Q8 LinearAdd routes must be exact, contiguous, and closed");

} // namespace

bool q8_linear_add_admits(const Q8LinearAddProblem& problem) noexcept {
    const bool shape = problem.rows == 5120 && (problem.k == 6144 || problem.k == 17408);
    return shape && problem.padded_k == problem.k && problem.cols >= 1;
}

Q8LinearAddPlan q8_linear_add_resolve_plan(const Q8LinearAddProblem& problem) {
    if (!q8_linear_add_admits(problem)) {
        throw std::invalid_argument("q8 linear_add: exact problem or column count is not admitted");
    }
    for (const RouteSpec& route : kN5120Routes) {
        if (problem.cols >= route.first && problem.cols <= route.last) { return {route.schedule}; }
    }
    throw std::logic_error("q8 linear_add: admitted problem has no covering route");
}

void q8_linear_add_execute_plan(const Q8LinearAddPlan& plan, const Tensor& x, const Weight& w,
                                Tensor& residual_out, cudaStream_t stream) {
    const Q8LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q8LinearAddPlan resolved = q8_linear_add_resolve_plan(problem);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("q8 linear_add: plan does not match the exact problem");
    }
    switch (plan.schedule) {
    case Q8LinearAddScheduleId::SplitKMmaCapacity:
        q8_linear_add_splitk_capacity_launch(x, w, residual_out, stream);
        return;
    case Q8LinearAddScheduleId::GroupedSplitK:
        q8_linear_add_grouped_launch(x, w, residual_out, stream);
        return;
    }
    throw std::logic_error("q8 linear_add: unknown schedule");
}

void q8_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            cudaStream_t stream) {
    const Q8LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    q8_linear_add_execute_plan(q8_linear_add_resolve_plan(problem), x, w, residual_out, stream);
}

} // namespace ninfer::ops::detail
