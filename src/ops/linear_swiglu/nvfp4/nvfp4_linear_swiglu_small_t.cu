#include "core/weight.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_epilogue.cuh"

#include <cuda_bf16.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

// Nvfp4SwiGluRows pairs gate row i with up row i + N/2 from the runtime row count, so each instance
// serves [34816,5120] and its two-device output-row half [17408,5120]. The half inherits the
// parent's token cutoffs; the sliced-K schedules below were measured at both on an RTX 5070 Ti.
using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    static constexpr auto kActivationAccess = ActiveTokens <= 4
                                                  ? Nvfp4SimtActivationAccess::SharedPhase
                                                  : Nvfp4SimtActivationAccess::TokenPacked;
    static constexpr int kWarpsPerCta       = ActiveTokens >= 13 ? 16 : (ActiveTokens >= 5 ? 4 : 8);
    using Schedule =
        Nvfp4A16SimtSchedule<kWarpsPerCta, 1, 2, 16, ActiveTokens, 1, kActivationAccess,
                             Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                             Nvfp4SimtBlockOrder::RowsContiguous, 1>;
    launch_nvfp4_a16_simt<Nvfp4ScheduleInstance<Schedule, 5120, ActiveTokens, true>>(
        nvfp4_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n / 2},
        Nvfp4SwiGluEpilogue{}, stream, Nvfp4SwiGluRows<1>{});
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{&launch_exact<2 + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers = make_launchers(std::make_index_sequence<1>{});

// Sliced-K over 8 K warps with 2 stages, the reduction of the generic <8,8,2> and <16,8,2>
// instances, so every T keeps their results bit for bit. What changes is how many CTAs an SM
// holds; the generic instances hold 3 (T<=8) and 2 (T<=16) of 8 warps, and on the 70-SM
// RTX 5070 Ti that leaves the shared-memory pipe throttled and DRAM at ~64 %:
// - T<=4 stages 4 activation rows instead of 8 and caps registers at 64: 4 CTAs per SM,
//   [17408,5120] 85.0 -> 78.2 us at T=4 (A4 route 78.9), [34816,5120] 166.2 -> 149.7 (A4 151.4);
// - T<=8 and T<=16 give each CTA two 16-row tiles that share one activation stage:
//   [17408,5120] 88.3 -> 83.2 us at T=8, 124.4 -> 114.7 us at T=16.
template <int Tokens, int MinBlocksPerSm, int RowTiles, int StageTokens>
using SlicedK =
    Nvfp4A16SlicedKMmaSchedule<8, Tokens, MinBlocksPerSm, Cache::ca, Cache::cg,
                               Nvfp4ActivationStage::PaddedZero, 2, RowTiles, StageTokens>;

template <class Schedule, int Capacity = 0>
void launch_sliced_k(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Schedule, 5120, Capacity>>(
        nvfp4_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n / 2},
        Nvfp4SwiGluEpilogue{}, stream, Nvfp4SwiGluRows<8>{});
}

// Above 16 tokens A16 has no measured 5070 Ti schedule; these are the routes of
// Wallawalla47/ninfer-custom 1f2f5e97, which serve the DFlash2 drafter's NVFP4 gate/up (A16Only)
// at batched widths: one sliced-K tile through 32 tokens, then 64-token Tensor Core tiles whose
// m16 fragments pair 8 gate rows over their 8 up rows (Nvfp4SwiGluMmaRows). The target's gate/up
// is AllowA4 and never reaches them.
void launch_a16_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_nvfp4_a16_mma<
        Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>, 5120>>(
        nvfp4_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n / 2},
        Nvfp4SwiGluEpilogue{}, stream, Nvfp4SwiGluMmaRows{});
}

} // namespace

void nvfp4_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream) {
    if (x.ne[1] <= 2)
        kLaunchers[0](x, weight, out, stream);
    else if (x.ne[1] <= 4)
        launch_sliced_k<SlicedK<8, 4, 1, 4>, 4>(x, weight, out, stream);
    else if (x.ne[1] <= 8)
        launch_sliced_k<SlicedK<8, 2, 2, 8>>(x, weight, out, stream);
    else if (x.ne[1] <= 16)
        launch_sliced_k<SlicedK<16, 1, 2, 16>>(x, weight, out, stream);
    else if (x.ne[1] <= 32)
        launch_sliced_k<Nvfp4SlicedInstance<32, 4, 2>>(x, weight, out, stream);
    else
        launch_a16_mma(x, weight, out, stream);
}

void nvfp4_linear_swiglu_sliced_k_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                         cudaStream_t stream) {
    if (x.ne[1] <= 4)
        launch_sliced_k<SlicedK<8, 4, 1, 4>, 4>(x, weight, out, stream);
    else if (x.ne[1] <= 8)
        launch_sliced_k<SlicedK<8, 2, 2, 8>>(x, weight, out, stream);
    else
        throw std::invalid_argument("nvfp4 linear_swiglu sliced-K: T must be at most 8");
}

} // namespace ninfer::ops::detail
