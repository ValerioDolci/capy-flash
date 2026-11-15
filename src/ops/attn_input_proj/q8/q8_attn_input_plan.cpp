#include "core/weight.h"
#include "ops/attn_input_proj/q8/q8_attn_input_plan.h"

#include "ops/attn_input_proj/q8/q8_attn_input_kernels.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {

namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    Q8AttnInputScheduleId schedule;
};

constexpr std::array<RouteSpec, 6> kDFlash2Routes{{
    {1, 48, Q8AttnInputScheduleId::DFlash2SmallT},
    {49, 63, Q8AttnInputScheduleId::DFlash2MmaR16C64K128},
    {64, 96, Q8AttnInputScheduleId::DFlash2MmaR32C32K128},
    {97, 128, Q8AttnInputScheduleId::DFlash2MmaR32C64K128},
    {129, 192, Q8AttnInputScheduleId::DFlash2MmaR32C64},
    {193, kAnyCols, Q8AttnInputScheduleId::DFlash2MmaR64C128},
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
              "Q8 DFlash2 attention input routes must be exact and closed");

bool is_dflash2_shape(const Q8AttnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.query_rows == 4096 && problem.kv_rows == 1024 &&
           problem.parent_rows == 6144 && problem.padded_k == 5120;
}

} // namespace

const char* q8_attn_input_schedule_name(Q8AttnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q8AttnInputScheduleId::DFlash2SmallT:
        return "attn_input_proj.q8.dflash2.small_t";
    case Q8AttnInputScheduleId::DFlash2MmaR16C64K128:
        return "attn_input_proj.q8.dflash2.mma.r16.c64.k128";
    case Q8AttnInputScheduleId::DFlash2MmaR32C32K128:
        return "attn_input_proj.q8.dflash2.mma.r32.c32.k128";
    case Q8AttnInputScheduleId::DFlash2MmaR32C64K128:
        return "attn_input_proj.q8.dflash2.mma.r32.c64.k128";
    case Q8AttnInputScheduleId::DFlash2MmaR32C64:
        return "attn_input_proj.q8.dflash2.mma.r32.c64";
    case Q8AttnInputScheduleId::DFlash2MmaR64C128:
        return "attn_input_proj.q8.dflash2.mma.r64.c128";
    }
    return "attn_input_proj.q8.unknown";
}

bool q8_attn_input_admits(const Q8AttnInputProblem& problem) noexcept {
    return is_dflash2_shape(problem) && problem.cols > 0;
}

Q8AttnInputPlan q8_attn_input_resolve_plan(const Q8AttnInputProblem& problem) {
    if (!q8_attn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q8 attention input: exact problem or column count is not admitted");
    }
    for (const RouteSpec& route : kDFlash2Routes) {
        if (problem.cols >= route.first && problem.cols <= route.last) { return {route.schedule}; }
    }
    throw std::logic_error("Q8 attention input: admitted problem has no covering route");
}

void q8_attn_input_execute_plan(const Q8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& k, Tensor& v, cudaStream_t stream) {
    const Q8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    const Q8AttnInputPlan resolved = q8_attn_input_resolve_plan(problem);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument(
            "Q8 attention input: plan does not match exact three-output problem");
    }
    switch (plan.schedule) {
    case Q8AttnInputScheduleId::DFlash2SmallT:
        q8_dflash2_attn_input_small_t_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::DFlash2MmaR16C64K128:
        q8_dflash2_attn_input_mma_r16_c64_k128_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::DFlash2MmaR32C32K128:
        q8_dflash2_attn_input_mma_r32_c32_k128_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::DFlash2MmaR32C64K128:
        q8_dflash2_attn_input_mma_r32_c64_k128_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::DFlash2MmaR32C64:
        q8_dflash2_attn_input_mma_r32_c64_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::DFlash2MmaR64C128:
        q8_dflash2_attn_input_mma_r64_c128_launch(x, weight, q, k, v, stream);
        return;
    }
    throw std::logic_error("Q8 attention input: unknown schedule");
}

void q8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                            cudaStream_t stream) {
    const Q8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    q8_attn_input_execute_plan(q8_attn_input_resolve_plan(problem), x, weight, q, k, v, stream);
}

} // namespace ninfer::ops::detail
