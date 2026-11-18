#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"

namespace ninfer::ops::detail {
namespace {
// The two-device input-column halves [5120,3072] and [5120,8704] inherit the schedules of the
// [5120,6144] and [5120,17408] problems they halve; they were not re-measured at the half K. Every
// schedule below fits both halves: the sliced-K blocks (512, 256 or 128 columns) and the MMA
// blocks (128 or 64 columns) divide 3072 and 8704.
template <int K>
void launch_matrix(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    const int tokens  = x.ne[1];
    const auto p      = nvfp4_a16_operands(x, weight);
    const auto output = LinearBf16Output{static_cast<__nv_bfloat16*>(residual.data), weight.n};
    const auto epilogue =
        LinearResidualAddEpilogue{{static_cast<__nv_bfloat16*>(residual.data), weight.n}};
    if constexpr (K == Nvfp4N5120K6144::kInputRows || K == Nvfp4N5120K3072::kInputRows) {
        if (tokens <= 8) {
            launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 8, 2>, K>>(
                p, output, epilogue, stream);
            return;
        }
        if (tokens <= 16) {
            launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2>, K>>(
                p, output, epilogue, stream);
            return;
        }
        if (tokens <= 24) {
            launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 8, 1>, K>>(
                p, output, epilogue, stream);
            return;
        }
        if (tokens <= 32) {
            launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 2>, K>>(
                p, output, epilogue, stream);
            return;
        }
        if (tokens <= 48) {
            launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 2, 2>, K>>(
                p, output, epilogue, stream);
            return;
        }
        if (tokens <= 64) {
            launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<64, 2, 2>, K>>(
                p, output, epilogue, stream);
            return;
        }
        if (tokens <= 128) {
            launch_nvfp4_a16_mma<
                Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 2, 1>, K>>(
                p, output, epilogue, stream);
            return;
        }
        {
            launch_nvfp4_a16_mma<
                Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, K>>(
                p, output, epilogue, stream);
            return;
        }
    } else {
        if (tokens <= 8) {
            launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 4, 2>, K>>(
                p, output, epilogue, stream);
            return;
        }
        if (tokens <= 16) {
            launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 4, 2>, K>>(
                p, output, epilogue, stream);
            return;
        }
        if (tokens <= 24) {
            launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 2>, K>>(
                p, output, epilogue, stream);
            return;
        }
        if (tokens <= 32) {
            launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 1>, K>>(
                p, output, epilogue, stream);
            return;
        }
        if (tokens <= 48) {
            launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 1>, K>>(
                p, output, epilogue, stream);
            return;
        }
        if (tokens <= 64) {
            launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<64, 2, 2>, K>>(
                p, output, epilogue, stream);
            return;
        }
        if (tokens <= 128) {
            launch_nvfp4_a16_mma<
                Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>, K>>(
                p, output, epilogue, stream);
            return;
        }
        {
            launch_nvfp4_a16_mma<
                Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, K>>(
                p, output, epilogue, stream);
            return;
        }
    }
}

void launch_matrix_for(const Tensor& x, const Weight& weight, Tensor& residual,
                       cudaStream_t stream) {
    switch (resolve_nvfp4_geometry(weight.n, weight.k)) {
    case Nvfp4GeometryId::N5120K6144:
        launch_matrix<Nvfp4N5120K6144::kInputRows>(x, weight, residual, stream);
        return;
    case Nvfp4GeometryId::N5120K17408:
        launch_matrix<Nvfp4N5120K17408::kInputRows>(x, weight, residual, stream);
        return;
    case Nvfp4GeometryId::N5120K3072:
        launch_matrix<Nvfp4N5120K3072::kInputRows>(x, weight, residual, stream);
        return;
    case Nvfp4GeometryId::N5120K8704:
        launch_matrix<Nvfp4N5120K8704::kInputRows>(x, weight, residual, stream);
        return;
    case Nvfp4GeometryId::N14336K5120:
    case Nvfp4GeometryId::N16384K5120:
    case Nvfp4GeometryId::N34816K5120:
    case Nvfp4GeometryId::N17408K5120:
        break;
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported problem");
}

bool output_family(const Weight& weight) {
    return weight.k == Nvfp4N5120K6144::kInputRows || weight.k == Nvfp4N5120K3072::kInputRows;
}
} // namespace

void nvfp4_linear_add_a16_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                 cudaStream_t stream) {
    if (x.ne[1] == 1) {
        nvfp4_linear_add_decode_launch(x, weight, residual, stream);
        return;
    }
    if (x.ne[1] <= (output_family(weight) ? 2 : 5)) {
        nvfp4_linear_add_small_t_launch(x, weight, residual, stream);
        return;
    }
    launch_matrix_for(x, weight, residual, stream);
}

void nvfp4_linear_add_a16_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                        cudaStream_t stream) {
    if (output_family(weight)) {
        launch_matrix_for(x, weight, residual, stream);
        return;
    }
    nvfp4_linear_add_small_t_launch(x, weight, residual, stream);
}
} // namespace ninfer::ops::detail
