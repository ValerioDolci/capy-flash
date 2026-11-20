#include "ops/linear/q8/q8_instances.cuh"
#include "ops/linear/q8/q8_launch.h"
#include "ops/linear/q8/q8_warp_row_launch.cuh"

#include "ops/linear/q8/q8_output.cuh"

namespace ninfer::ops::detail {
void launch_q8_a16_warp_row_r4_p10(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    q8_with_output(out, out.ne[0], [&](auto output) {
        launch_q8_a16_warp_row<q8_instances::WarpRowR4P10>(q8_linear_operands(x, w), output,
                                                           LinearIdentityEpilogue{}, stream);
    });
}

void launch_q8_a16_warp_row_r4_p3(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream) {
    q8_with_output(out, out.ne[0], [&](auto output) {
        launch_q8_a16_warp_row<q8_instances::WarpRowR4P3>(q8_linear_operands(x, w), output,
                                                          LinearIdentityEpilogue{}, stream);
    });
}

void launch_q8_a16_warp_row_r2_w2_p5(const Tensor& x, const Weight& w, Tensor& out,
                                     cudaStream_t stream) {
    q8_with_output(out, out.ne[0], [&](auto output) {
        launch_q8_a16_warp_row<q8_instances::WarpRowR2W2P5>(q8_linear_operands(x, w), output,
                                                            LinearIdentityEpilogue{}, stream);
    });
}

void launch_q8_a16_warp_row_r4_p10_t8(const Tensor& x, const Weight& w, Tensor& out,
                                      cudaStream_t stream) {
    q8_with_output(out, out.ne[0], [&](auto output) {
        launch_q8_a16_warp_row<q8_instances::WarpRowR4P10T8>(q8_linear_operands(x, w), output,
                                                             LinearIdentityEpilogue{}, stream);
    });
}

void launch_q8_a16_warp_row_r4_p3_t8(const Tensor& x, const Weight& w, Tensor& out,
                                     cudaStream_t stream) {
    q8_with_output(out, out.ne[0], [&](auto output) {
        launch_q8_a16_warp_row<q8_instances::WarpRowR4P3T8>(q8_linear_operands(x, w), output,
                                                            LinearIdentityEpilogue{}, stream);
    });
}
} // namespace ninfer::ops::detail
