#include "core/weight.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/vector_output.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear_swiglu/token_major_mma_epilogue.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear/nvfp4/nvfp4_a4_tiles.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// Every registered gate|up problem shares K (nvfp4_linear_swiglu_plan.h); the output rows come
// from the weight, so these instances serve all of them.
using Geometry = Nvfp4N34816K5120;
// Column tiles amortize gate/up decode over the complete speculative block.
using M96N128 = Nvfp4A4MmaSchedule<96, 128, 256, 3, 4, 2, 1>;

template <class Schedule>
void launch_gemm(const Weight& weight, Tensor& out, Nvfp4A4Workspace workspace, int tokens,
                 cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>;
    launch_nvfp4_a4_mma<S>(nvfp4_a4_operands(weight, workspace, tokens, Nvfp4ScaleLayout::RowMajor),
                           LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n / 2},
                           SwiGluTokenMajorMmaEpilogue{}, stream, SwiGluTokenMajorMmaRows<S>{});
}

template <class Schedule>
void launch(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace,
            cudaStream_t stream) {
    auto scope = workspace.scope();
    const Nvfp4A4Workspace scratch =
        allocate_nvfp4_a4_workspace(workspace, x.ne[1], Geometry::kInputRows);
    launch_nvfp4_a4_quantize(x, weight, scratch, Nvfp4ScaleLayout::RowMajor, stream);
    launch_gemm<Schedule>(weight, out, scratch, x.ne[1], stream);
}

} // namespace

Nvfp4LinearSwiGluA4Launch nvfp4_linear_swiglu_select_a4(std::int32_t tokens) {
    if (tokens <= 32) return launch<Nvfp4A4T32R128>;
    if (tokens <= Nvfp4A4T64R128W16::kBlockTokens) return launch<Nvfp4A4T64R128W16>;
    if (tokens <= M96N128::kBlockTokens) return launch<M96N128>;
    return launch<Nvfp4A4T128R128PipelinedW16>;
}

void nvfp4_linear_swiglu_a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    nvfp4_linear_swiglu_select_a4(x.ne[1])(x, weight, out, workspace, stream);
}

} // namespace ninfer::ops::detail
