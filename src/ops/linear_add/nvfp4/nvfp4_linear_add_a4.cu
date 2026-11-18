#include "core/weight.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_tiles.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma_launch.h"
#include "ops/linear/common/epilogue.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using A4Route = Nvfp4LinearAddA4Route;

template <class Geometry, class Schedule>
void launch_gemm(const Weight& weight, Tensor& residual, Nvfp4A4Workspace workspace,
                 std::int32_t tokens, cudaStream_t stream) {
    launch_nvfp4_a4_mma<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a4_operands(weight, workspace, tokens, Nvfp4ScaleLayout::RowMajor),
        LinearBf16Output{static_cast<__nv_bfloat16*>(residual.data), weight.n},
        LinearResidualAddEpilogue{{static_cast<__nv_bfloat16*>(residual.data), weight.n}}, stream);
}

void launch_tma(const Weight& weight, Tensor& residual, Nvfp4A4Workspace workspace,
                std::int32_t tokens, cudaStream_t stream) {
    launch_nvfp4_a4_tma_linear_add(
        nvfp4_a4_operands(weight, workspace, tokens, Nvfp4ScaleLayout::Tiled256),
        static_cast<__nv_bfloat16*>(residual.data), stream);
}

template <class Geometry, class Schedule>
constexpr A4Route mma_route() {
    return {launch_gemm<Geometry, Schedule>, Nvfp4ScaleLayout::RowMajor};
}

template <class Geometry>
A4Route select_mma(std::int32_t tokens) {
    if (tokens <= 64) return mma_route<Geometry, Nvfp4A4T32R64>();
    if (tokens <= 128) return mma_route<Geometry, Nvfp4A4T32R128>();
    if (tokens <= 192) return mma_route<Geometry, Nvfp4A4T64R128>();
    if (tokens <= 384) return mma_route<Geometry, Nvfp4A4T128R128Resident>();
    if (tokens <= 512) return mma_route<Geometry, Nvfp4A4T128R128Pipelined>();
    return mma_route<Geometry, Nvfp4A4T128R128Resident>();
}

} // namespace

// The two-device halves [5120,8704] and [5120,3072] take the TMA route at the same width as
// linear() over the same shard, so both ranks of a row-parallel projection run one schedule.
Nvfp4LinearAddA4Route nvfp4_linear_add_select_a4(std::int32_t output_rows, std::int32_t input_rows,
                                                 std::int32_t tokens) {
    if (tokens >= kNvfp4OutputDownFamilyFirstTmaTokens) {
        return {launch_tma, Nvfp4ScaleLayout::Tiled256};
    }
    switch (resolve_nvfp4_geometry(output_rows, input_rows)) {
    case Nvfp4GeometryId::N5120K6144:
        return select_mma<Nvfp4N5120K6144>(tokens);
    case Nvfp4GeometryId::N5120K17408:
        return select_mma<Nvfp4N5120K17408>(tokens);
    // The two-device halves inherit the MMA bands of the problem they halve (not re-measured).
    case Nvfp4GeometryId::N5120K8704:
        return select_mma<Nvfp4N5120K8704>(tokens);
    case Nvfp4GeometryId::N5120K3072:
        return select_mma<Nvfp4N5120K3072>(tokens);
    case Nvfp4GeometryId::N14336K5120:
    case Nvfp4GeometryId::N16384K5120:
    case Nvfp4GeometryId::N34816K5120:
    case Nvfp4GeometryId::N17408K5120:
        break;
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported problem");
}

void nvfp4_linear_add_a4_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                Nvfp4A4Workspace workspace, cudaStream_t stream) {
    const std::int32_t tokens         = x.ne[1];
    const Nvfp4LinearAddA4Route route = nvfp4_linear_add_select_a4(weight.n, weight.k, tokens);
    launch_nvfp4_a4_quantize(x, weight, workspace, route.scales, stream);
    route.launch(weight, residual, workspace, tokens, stream);
}

} // namespace ninfer::ops::detail
