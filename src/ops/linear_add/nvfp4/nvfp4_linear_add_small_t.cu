#include "core/weight.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/common/epilogue.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    // The [5120,3072] half takes the warp crossover of the [5120,6144] problem it halves.
    constexpr bool kOutputFamily = Geometry::kInputRows == Nvfp4N5120K6144::kInputRows ||
                                   Geometry::kInputRows == Nvfp4N5120K3072::kInputRows;
    // One warp owns a row, so residency changes speed only. At T=4..5 the MLP down family asks for
    // 6 CTAs per SM (<= 80 registers instead of 96): on an RTX 5070 Ti [5120,8704] 47.1 -> 46.5 us
    // at T=4 and 57.1 -> 51.3 at T=5. T=3 keeps 1: there it cost +9 % (43.3 -> 47.1 us).
    constexpr int kMinBlocks = !kOutputFamily && ActiveTokens >= 4 ? 6 : 1;
    // Four rows per warp for the MLP down family at T=3..5 (RTX 5070 Ti, ninfer_linear_bench on the
    // plain linear() of [5120,8704]: T=3/4 46.8 -> 44.7 us, T=5 53.0 -> 46.8 us; the per-row K loop
    // is unchanged, so outputs are bit-identical). Two rows elsewhere, as before. On the MTP3
    // decode round at tp 2 (rank 0 runs this fused half): 17.137 -> 17.017 ms (-0.7 %), 60/60 texts
    // identical.
    constexpr int kRowsPerWarp = !kOutputFamily && ActiveTokens >= 3 && ActiveTokens <= 5 ? 4 : 2;
    using Schedule             = Nvfp4A16SimtSchedule<
                    (ActiveTokens <= 16 && ActiveTokens >= (kOutputFamily ? 14 : 8)) ? 16 : 4, 1, kRowsPerWarp,
        (ActiveTokens >= 17 && ActiveTokens <= 20) ? 8 : 16, ActiveTokens, 1,
        Nvfp4SimtActivationAccess::TokenPacked, Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default,
        1, Nvfp4SimtBlockOrder::RowsContiguous, kMinBlocks>;
    launch_nvfp4_a16_simt<
        Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows, ActiveTokens, true>>(
        nvfp4_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(residual.data), weight.n},
        LinearResidualAddEpilogue{{static_cast<__nv_bfloat16*>(residual.data), weight.n}}, stream);
}

template <class Geometry, int First, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, First + static_cast<int>(Offsets)>...};
}

// Widths 2..5 for the output family. The MLP down family runs this kernel at T=2..5 under every
// policy and at T=1..7 under AllowA4Decode (nvfp4_linear_add_a16_decode_launch), whose per-row K
// loop gives a column the same bits at every width.
constexpr int kOutputFirst = 2;
constexpr int kOutputLast  = 5;
constexpr int kDownFirst   = 1;
constexpr int kDownLast    = kAllowA4DecodeMaxWidth;
static_assert(kDownLast < kNvfp4DownFamilyFirstA4Tokens);

template <class Geometry, int First, int Last>
constexpr auto make_launchers() {
    return make_launchers<Geometry, First>(std::make_index_sequence<Last - First + 1>{});
}

constexpr auto kResidual6144Launchers =
    make_launchers<Nvfp4N5120K6144, kOutputFirst, kOutputLast>();
constexpr auto kResidual17408Launchers = make_launchers<Nvfp4N5120K17408, kDownFirst, kDownLast>();
constexpr auto kResidual8704Launchers  = make_launchers<Nvfp4N5120K8704, kDownFirst, kDownLast>();
constexpr auto kResidual3072Launchers =
    make_launchers<Nvfp4N5120K3072, kOutputFirst, kOutputLast>();

template <class Launchers>
void launch_width(const Launchers& launchers, int first, const Tensor& x, const Weight& weight,
                  Tensor& residual, cudaStream_t stream) {
    const int index = x.ne[1] - first;
    if (index < 0 || index >= static_cast<int>(launchers.size())) {
        throw std::invalid_argument("nvfp4 linear_add small-T: unsupported T");
    }
    launchers[static_cast<std::size_t>(index)](x, weight, residual, stream);
}

} // namespace

void nvfp4_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                     cudaStream_t stream) {
    switch (resolve_nvfp4_geometry(weight.n, weight.k)) {
    case Nvfp4GeometryId::N5120K6144:
        launch_width(kResidual6144Launchers, kOutputFirst, x, weight, residual, stream);
        return;
    case Nvfp4GeometryId::N5120K17408:
        launch_width(kResidual17408Launchers, kDownFirst, x, weight, residual, stream);
        return;
    case Nvfp4GeometryId::N5120K8704:
        launch_width(kResidual8704Launchers, kDownFirst, x, weight, residual, stream);
        return;
    case Nvfp4GeometryId::N5120K3072:
        launch_width(kResidual3072Launchers, kOutputFirst, x, weight, residual, stream);
        return;
    case Nvfp4GeometryId::N14336K5120:
    case Nvfp4GeometryId::N16384K5120:
    case Nvfp4GeometryId::N34816K5120:
    case Nvfp4GeometryId::N17408K5120:
        break;
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported problem");
}

} // namespace ninfer::ops::detail
