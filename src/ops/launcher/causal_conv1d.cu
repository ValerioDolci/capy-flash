// ninfer::ops - causal_conv1d launcher: grid/block/stream configuration + kernel launch.
#include "ops/launcher/causal_conv1d.h"

#include "ops/common/math.h"
#include "ops/kernel/causal_conv1d.cuh"
#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

// The small-T routes launch one thread per channel and token in one block, so the bound and the
// channel tile together may not exceed the hardware block ceiling.
static_assert(kCausalConvChannelTile * kCausalConvParallelMaxTokens <= 1024,
              "small-T block would exceed 1024 threads");

namespace {

int grid_for(std::int64_t n, int block, const char* label) {
    const std::int64_t grid = div_up(n, static_cast<std::int64_t>(block));
    if (grid > std::numeric_limits<int>::max()) {
        throw std::overflow_error(std::string("causal_conv1d: ") + label +
                                  " grid exceeds CUDA limit");
    }
    return static_cast<int>(std::max<std::int64_t>(1, grid));
}

int prefill_output_grid_for(std::int32_t C, std::int32_t T, int block) {
    const std::int64_t c_blocks =
        div_up(static_cast<std::int64_t>(C), static_cast<std::int64_t>(block));
    const std::int64_t grid = c_blocks * static_cast<std::int64_t>(T);
    if (grid > std::numeric_limits<int>::max()) {
        throw std::overflow_error("causal_conv1d: prefill output grid exceeds CUDA limit");
    }
    return static_cast<int>(std::max<std::int64_t>(1, grid));
}

} // namespace

template <std::int32_t Rows0, std::int32_t Rows1, std::int32_t Rows2>
void smallt_split_launch(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                         Tensor& conv_state_out, Tensor& out0, Tensor& out1, Tensor& out2,
                         cudaStream_t stream) {
    const std::int32_t C = x.ne[0];
    const std::int32_t T = x.ne[1];
    if (Rows0 + Rows1 + Rows2 != C) {
        throw std::invalid_argument("causal_conv1d: split geometry does not cover the channels");
    }
    const dim3 block(kCausalConvChannelTile, static_cast<unsigned int>(T));
    const int grid = grid_for(C, kCausalConvChannelTile, "small-T");

    const CausalConvSplitOutput3<__nv_bfloat16, Rows0, Rows1, Rows2> out_map{
        static_cast<__nv_bfloat16*>(out0.data), static_cast<__nv_bfloat16*>(out1.data),
        static_cast<__nv_bfloat16*>(out2.data)};
    causal_conv1d_smallt_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<const __nv_bfloat16*>(conv_state_in.data),
        static_cast<__nv_bfloat16*>(conv_state_out.data), out_map, C, T);
    CUDA_CHECK(cudaGetLastError());
}

void causal_conv1d_smallt_split_launch(const Tensor& x, const Tensor& weight,
                                       const Tensor& conv_state_in, Tensor& conv_state_out,
                                       Tensor& out0, Tensor& out1, Tensor& out2,
                                       CausalConvSplitGeometry geometry, cudaStream_t stream) {
    switch (geometry) {
    case CausalConvSplitGeometry::Rows1024x1024x3072:
        smallt_split_launch<1024, 1024, 3072>(x, weight, conv_state_in, conv_state_out, out0, out1,
                                              out2, stream);
        return;
    case CausalConvSplitGeometry::Rows2048x2048x6144:
        smallt_split_launch<2048, 2048, 6144>(x, weight, conv_state_in, conv_state_out, out0, out1,
                                              out2, stream);
        return;
    }
    throw std::logic_error("causal_conv1d: unhandled split geometry");
}

template <std::int32_t Rows0, std::int32_t Rows1, std::int32_t Rows2>
void prefill_split_launch(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                          Tensor& conv_state_out, Tensor& out0, Tensor& out1, Tensor& out2,
                          cudaStream_t stream) {
    static_assert((Rows0 % 2) == 0 && (Rows1 % 2) == 0 && (Rows2 % 2) == 0,
                  "a pair must not straddle two destinations");
    constexpr int kChannelBlock = 256;
    constexpr int kPairBlock    = 256;
    const std::int32_t C        = x.ne[0];
    const std::int32_t T        = x.ne[1];
    if (Rows0 + Rows1 + Rows2 != C) {
        throw std::invalid_argument("causal_conv1d: split geometry does not cover the channels");
    }

    const CausalConvSplitOutput3<__nv_bfloat162, Rows0 / 2, Rows1 / 2, Rows2 / 2> out_map{
        reinterpret_cast<__nv_bfloat162*>(out0.data), reinterpret_cast<__nv_bfloat162*>(out1.data),
        reinterpret_cast<__nv_bfloat162*>(out2.data)};
    causal_conv1d_prefill_pairs_kernel<<<prefill_output_grid_for(C / 2, T, kPairBlock), kPairBlock,
                                         0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<const __nv_bfloat16*>(conv_state_in.data), out_map, C, T);
    CUDA_CHECK(cudaGetLastError());

    causal_conv1d_prefill_state_kernel<<<grid_for(C, kChannelBlock, "prefill state"), kChannelBlock,
                                         0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(conv_state_in.data),
        static_cast<__nv_bfloat16*>(conv_state_out.data), C, T);
    CUDA_CHECK(cudaGetLastError());
}

void causal_conv1d_prefill_split_launch(const Tensor& x, const Tensor& weight,
                                        const Tensor& conv_state_in, Tensor& conv_state_out,
                                        Tensor& out0, Tensor& out1, Tensor& out2,
                                        CausalConvSplitGeometry geometry, cudaStream_t stream) {
    switch (geometry) {
    case CausalConvSplitGeometry::Rows1024x1024x3072:
        prefill_split_launch<1024, 1024, 3072>(x, weight, conv_state_in, conv_state_out, out0, out1,
                                               out2, stream);
        return;
    case CausalConvSplitGeometry::Rows2048x2048x6144:
        prefill_split_launch<2048, 2048, 6144>(x, weight, conv_state_in, conv_state_out, out0, out1,
                                               out2, stream);
        return;
    }
    throw std::logic_error("causal_conv1d: unhandled split geometry");
}

} // namespace ninfer::ops::detail
