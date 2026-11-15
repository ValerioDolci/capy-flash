#include "ops/linear/q8/q8_geometry.h"
#include "core/weight.h"
#include "ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_kernels.h"
#include "core/device.h"
#include "ops/linear/q8/q8_schedule.cuh"
#include "ops/linear/q8/q8_launch.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include "ninfer/ops/tp2/dynamic_grouped_conv.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <array>
#include <algorithm>
#include <string>
#include <utility>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kRows = 5120, kGroups = 320;

__device__ __forceinline__ void finish_value(int row, int col, int width, float current,
                                             float previous, const __nv_bfloat16* base,
                                             const __nv_bfloat16* delta, __nv_bfloat16* residual) {
    const int index = col * kRows + row, di = col * 2 * kGroups + row / 16;
    float value = fmaf(__bfloat162float(base[2 * kRows + row]) + __bfloat162float(delta[di]),
                       current, __bfloat162float(residual[index]));
    if (col % width != 0)
        value =
            fmaf(__bfloat162float(base[3 * kRows + row]) + __bfloat162float(delta[di + kGroups]),
                 previous, value);
    residual[index] = __float2bfloat16_rn(value);
}

using Launch = Q8Launch;

template <int InputRows, int TileColumns>
void tiled_projection(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int Warps =
        InputRows == 4096 ? (TileColumns <= 40 ? 8 : 4) : (TileColumns <= 32 ? 8 : 4);
    constexpr Cache Activation =
        InputRows == 4096 && ((TileColumns > 24 && TileColumns <= 40) || TileColumns > 48)
            ? Cache::cg
            : Cache::ca;
    using Geometry    = Q8LinearGeometry<kRows, InputRows>;
    using Schedule    = Q8A16SlicedKMmaSchedule<TileColumns, Warps, 1, Warps == 8 ? 2 : 3,
                                                Q8ScaleAccess::Shared, Activation>;
    const int columns = x.ne[1];
    LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), kRows};
    const dim3 grid(kRows / 16, (columns + TileColumns - 1) / TileColumns);
    launch_q8_a16_sliced_k_mma<
        typename Schedule::template with_problem<Geometry::kInputRows, TileColumns, false>,
        Q8SlicedKIdentityRows>(q8_linear_operands(x, weight), output, LinearIdentityEpilogue{},
                               stream);
    CUDA_CHECK(cudaGetLastError());
}

// Live columns stay dynamic; only the eight-column MMA accumulator layout is specialized.
template <int C, std::size_t... I>
constexpr auto make_launchers(std::index_sequence<I...>) {
    return std::array<Launch, sizeof...(I)>{&tiled_projection<C, 8 * (1 + static_cast<int>(I))>...};
}

constexpr auto attention = make_launchers<4096>(std::make_index_sequence<11>{});
constexpr auto mlp       = make_launchers<17408>(std::make_index_sequence<11>{});

__global__ void finish_kernel(const __nv_bfloat16* projected, const __nv_bfloat16* base,
                              const __nv_bfloat16* delta, __nv_bfloat16* residual, int width) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x, col = blockIdx.y;
    if (row >= kRows) return;
    const int index = col * kRows + row;
    finish_value(row, col, width, __bfloat162float(projected[index]),
                 col % width ? __bfloat162float(projected[index - kRows]) : 0.0f, base, delta,
                 residual);
}

void materialized(Q8DynamicConvAddSchedule schedule, const Tensor& x, const Weight& weight,
                  const Tensor& base, const Tensor& delta, Tensor& residual, Tensor& projected,
                  cudaStream_t stream) {
    const int tokens  = x.ne[1] * x.ne[2];
    const Tensor flat = x.view({x.ne[0], tokens});
    Tensor result     = projected.view({kRows, tokens});
    switch (schedule) {
    case Q8DynamicConvAddSchedule::TiledMma: {
        const auto& launchers = x.ne[0] == 4096 ? attention : mlp;
        launchers[(tokens - 1) / 8](flat, weight, result, stream);
        break;
    }
    case Q8DynamicConvAddSchedule::MmaK128:
        launch_q8_a16_mma_r64x32_t64_k128_a1(flat, weight, result, stream);
        break;
    }
    const dim3 grid((kRows + 255) / 256, tokens);
    finish_kernel<<<grid, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(projected.data),
                                            static_cast<const __nv_bfloat16*>(base.data),
                                            static_cast<const __nv_bfloat16*>(delta.data),
                                            static_cast<__nv_bfloat16*>(residual.data), x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void q8_dynamic_grouped_conv_add_materialized_launch(Q8DynamicConvAddSchedule schedule,
                                                     const Tensor& x, const Weight& weight,
                                                     const Tensor& base, const Tensor& delta,
                                                     Tensor& residual, Tensor& projected,
                                                     cudaStream_t stream) {
    materialized(schedule, x, weight, base, delta, residual, projected, stream);
}
} // namespace ninfer::ops::detail

// The finish half of the materialized route as its own Op (ninfer/ops/tp2/dynamic_grouped_conv.h),
// launching finish_kernel above.

namespace ninfer::ops {
namespace {

void require_conv_tensor(const Tensor& t, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                         std::int32_t n3, const char* label) {
    if (t.dtype != DType::BF16 || t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != n2 ||
        t.ne[3] != n3 || !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string("dynamic_grouped_conv_finish_add: invalid ") +
                                    label);
    }
}

} // namespace

void dynamic_grouped_conv_finish_add(const Tensor& projected, const Tensor& base_kernel,
                                     const Tensor& finish_delta, Tensor& residual,
                                     cudaStream_t stream) {
    const std::int32_t width = projected.ne[1];
    const std::int32_t batch = projected.ne[2];
    if (width < 2 || width > 16 || batch < 1 || batch > 8) {
        throw std::invalid_argument("dynamic_grouped_conv_finish_add: W must be 2..16, B 1..8");
    }
    require_conv_tensor(projected, detail::kRows, width, batch, 1, "projected");
    require_conv_tensor(residual, detail::kRows, width, batch, 1, "residual");
    require_conv_tensor(base_kernel, detail::kRows, 2, 2, 1, "base kernel");
    require_conv_tensor(finish_delta, detail::kGroups, 2, width, batch, "finish delta");
    const auto begin = [](const Tensor& t) { return reinterpret_cast<std::uintptr_t>(t.data); };
    const auto end   = [&](const Tensor& t) { return begin(t) + t.bytes(); };
    for (const Tensor* other : {&projected, &base_kernel, &finish_delta}) {
        if (begin(residual) < end(*other) && begin(*other) < end(residual)) {
            throw std::invalid_argument("dynamic_grouped_conv_finish_add: residual overlaps");
        }
    }
    const dim3 grid((detail::kRows + 255) / 256, width * batch);
    detail::finish_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected.data),
        static_cast<const __nv_bfloat16*>(base_kernel.data),
        static_cast<const __nv_bfloat16*>(finish_delta.data),
        static_cast<__nv_bfloat16*>(residual.data), width);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
