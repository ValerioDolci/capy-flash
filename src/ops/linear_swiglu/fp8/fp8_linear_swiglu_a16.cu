#include "ops/linear/fp8/fp8_instances.cuh"
#include "ops/linear_swiglu/row_major_mma_epilogue.cuh"
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/fp8/fp8_a16_simt.cuh"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_output.cuh"
#include "ops/linear_swiglu/tp2/linear_swiglu_half.h"
#include "ops/linear/fp8/fp8_geometry.h"

#include <stdexcept>

// The [34816,5120] parent's A16 launchers and the small-T one of its two-device [17408,5120] half:
// the small-T kernel takes the intermediate rows as a template argument (one instance per problem);
// the matrix launcher reads them from the weight and serves both (tp2/linear_swiglu_half.h).

namespace ninfer::ops::detail {

static_assert(Fp8N17408K5120::kOutputRows == kLinearSwiGluHalfRows &&
              Fp8N17408K5120::kInputRows == Fp8N34816K5120::kInputRows);

namespace {

template <class Geometry>
void launch_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int kIntermediate = Geometry::kOutputRows / 2;
    using Schedule =
        Fp8A16SimtSchedule<4, 2, 16, 4, 1, Fp8SimtActivationAccess::TokenPacked,
                           Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
    using Rows = Fp8SwiGluRows<Schedule::kRowsPerWarp / 2, kIntermediate>;
    launch_fp8_a16_simt<Fp8ScheduleInstance<Schedule, Geometry::kInputRows, 4>>(
        fp8_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), kIntermediate}, Fp8SwiGluEpilogue{},
        stream, Rows{});
}

} // namespace

void fp8_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream) {
    launch_small_t<Fp8N34816K5120>(x, weight, out, stream);
}

void fp8_linear_swiglu_half_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                           cudaStream_t stream) {
    launch_small_t<Fp8N17408K5120>(x, weight, out, stream);
}

void fp8_linear_swiglu_matrix_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream) {
    const auto p = fp8_a16_operands(x, weight);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), weight.n / 2};
    const auto sliced = [&]<int T, int W, int Stages>() {
        using S = Fp8ScheduleInstance<Fp8SlicedInstance<T, W, Stages>, 5120>;
        launch_fp8_a16_sliced_k_mma<S>(p, output, Fp8SwiGluEpilogue{}, stream,
                                       SwiGluRowMajorMmaRows<S>{});
    };
    const auto mma = [&]<class Schedule>() {
        using S = Fp8ScheduleInstance<Schedule, 5120>;
        launch_fp8_a16_mma<S>(p, output, SwiGluRowMajorMmaEpilogue{}, stream,
                              SwiGluRowMajorMmaRows<S>{});
    };
    if (x.ne[1] <= 8) return sliced.template operator()<8, 4, 1>();
    if (x.ne[1] <= 16) return sliced.template operator()<16, 4, 1>();
    if (x.ne[1] <= 24) return sliced.template operator()<32, 8, 1>();
    if (x.ne[1] <= 32) return sliced.template operator()<32, 4, 1>();
    if (x.ne[1] <= 64)
        return mma.template operator()<Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>();
    if (x.ne[1] <= 96)
        return mma.template operator()<Fp8A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>();
    mma.template operator()<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>();
}
} // namespace ninfer::ops::detail
