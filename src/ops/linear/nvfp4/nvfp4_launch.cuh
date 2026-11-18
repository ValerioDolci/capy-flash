#pragma once
#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_a16_simt.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_tma_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

namespace ninfer::ops::detail {
template <class Geometry, class Schedule>
void nvfp4_linear_a16_gemv(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>;
    launch_nvfp4_a16_gemv<S>(nvfp4_a16_operands(x, w),
                             LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
                             LinearIdentityEpilogue{}, stream);
}

template <class Geometry, int Capacity, class Schedule, bool Full = false>
void nvfp4_linear_a16_simt(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows, Capacity, Full>;
    launch_nvfp4_a16_simt<S>(nvfp4_a16_operands(x, w),
                             LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
                             LinearIdentityEpilogue{}, stream);
}

template <class Geometry, class Schedule>
void nvfp4_linear_a16_mma(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_nvfp4_a16_mma<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

template <class Geometry, class Schedule>
void nvfp4_linear_a16_sliced_k(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
        LinearIdentityEpilogue{}, stream);
}

template <class Geometry, class Schedule>
void nvfp4_linear_a4_mma(const Weight& w, Tensor& y, Nvfp4A4Workspace workspace, int tokens,
                         cudaStream_t stream) {
    launch_nvfp4_a4_mma<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a4_operands(w, workspace, tokens, Nvfp4ScaleLayout::RowMajor),
        LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n}, LinearIdentityEpilogue{},
        stream);
}

template <Nvfp4GeometryId Geometry, Nvfp4ScaleLayout Layout>
void nvfp4_linear_a4_tma(const Weight& weight, Tensor& out, Nvfp4A4Workspace scratch, int tokens,
                         cudaStream_t stream) {
    launch_nvfp4_a4_tma_linear(Geometry, nvfp4_a4_operands(weight, scratch, tokens, Layout),
                               static_cast<__nv_bfloat16*>(out.data), stream);
}

template <Nvfp4GeometryId Geometry, Nvfp4ScaleLayout Layout = Nvfp4ScaleLayout::Tiled256>
constexpr Nvfp4A4Route nvfp4_a4_tma_route() {
    return {nvfp4_linear_a4_tma<Geometry, Layout>, Layout};
}

template <class Geometry, class Schedule>
constexpr Nvfp4A4Route nvfp4_a4_mma_route() {
    return {nvfp4_linear_a4_mma<Geometry, Schedule>, Nvfp4ScaleLayout::RowMajor};
}
} // namespace ninfer::ops::detail
