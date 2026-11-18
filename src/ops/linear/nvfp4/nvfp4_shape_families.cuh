#pragma once
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_tiles.h"

// Families of NVFP4 linear() shapes that share one selection: each family's schedules and token
// cutoffs were measured on the shape it names, and a shape of the family is the family's selection
// instantiated at its own Nvfp4Geometry (`shape<Geometry>`). A TMA GEMM exists only for the
// geometries Nvfp4GeometryId registers; a shape with one puts the TMA rows in front of the
// family's MMA band (`with_tma`), a shape without one runs the band at every A4 width.
namespace ninfer::ops::detail::nvfp4_family {
using Gemv =
    Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;

template <Nvfp4A4Route (*Mma)(std::int32_t), std::int32_t FirstTma, Nvfp4GeometryId Tma>
Nvfp4A4Route with_tma(std::int32_t tokens) {
    if (tokens >= FirstTma) return nvfp4_a4_tma_route<Tma>();
    return Mma(tokens);
}

inline bool always_a4(std::int32_t, std::int32_t) { return true; }
} // namespace ninfer::ops::detail::nvfp4_family

// The MLP gate|up, measured on [34816,5120]: its two-device half [17408,5120] and the FFN-14336
// gate|up [28672,5120] inherit it. The TMA GEMM from 256 columns.
namespace ninfer::ops::detail::nvfp4_gate_up {
using C2 = Nvfp4A16SimtSchedule<8, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::SharedPhase,
                                Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                Nvfp4SimtBlockOrder::RowsContiguous, 1>;

template <class Geometry>
Nvfp4Launch a16(std::int32_t tokens) {
    if (tokens == 1) return nvfp4_linear_a16_gemv<Geometry, nvfp4_family::Gemv>;
    if (tokens <= 2) return nvfp4_linear_a16_simt<Geometry, 2, C2, true>;
    if (tokens <= 8) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<8, 8, 2>>;
    if (tokens <= 16) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<16, 8, 2>>;
    if (tokens <= 24) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 4, 1>>;
    if (tokens <= 32) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 4, 2>>;
    if (tokens <= 64)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>;
    if (tokens <= 96)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>;
    return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}

template <class Geometry>
Nvfp4A4Route a4(std::int32_t tokens) {
    if (tokens <= 32) return nvfp4_a4_mma_route<Geometry, Nvfp4A4T32R128>();
    if (tokens <= 64) return nvfp4_a4_mma_route<Geometry, Nvfp4A4T64R128W16>();
    if (tokens <= 128) return nvfp4_a4_mma_route<Geometry, Nvfp4A4T128R128Pipelined>();
    return nvfp4_a4_mma_route<Geometry, Nvfp4A4T128R128Resident>();
}

template <class Geometry>
constexpr Nvfp4LinearShape shape{Geometry::kOutputRows, Geometry::kInputRows, a16<Geometry>,
                                 a4<Geometry>, nvfp4_family::always_a4};

template <class Geometry, Nvfp4GeometryId Tma>
constexpr Nvfp4LinearShape shape_with_tma{
    Geometry::kOutputRows, Geometry::kInputRows, a16<Geometry>,
    nvfp4_family::with_tma<a4<Geometry>, 256, Tma>, nvfp4_family::always_a4};
} // namespace ninfer::ops::detail::nvfp4_gate_up

// The attention q|k|gate|v, measured on [14336,5120]: its two-device row half [7168,5120]
// inherits it. A4 from kNvfp4AttnInputFamilyFirstA4Tokens, as the fused attention input
// projection over the same problems. [14336,5120] has two TMA rows of its own (n14336_k5120.cu).
namespace ninfer::ops::detail::nvfp4_attn_qkgv {
using C2 = Nvfp4A16SimtSchedule<8, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::SharedPhase,
                                Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                Nvfp4SimtBlockOrder::RowsContiguous, 1>;

template <class Geometry>
Nvfp4Launch a16(std::int32_t tokens) {
    if (tokens == 1) return nvfp4_linear_a16_gemv<Geometry, nvfp4_family::Gemv>;
    if (tokens <= 2) return nvfp4_linear_a16_simt<Geometry, 2, C2, true>;
    if (tokens <= 8) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<8, 8, 2>>;
    if (tokens <= 16) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<16, 8, 2>>;
    if (tokens <= 24) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 4, 1>>;
    if (tokens <= 32)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>>;
    if (tokens <= 64)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>;
    if (tokens <= 128)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>>;
    return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}

template <class Geometry>
Nvfp4A4Route a4(std::int32_t tokens) {
    if (tokens <= 64) return nvfp4_a4_mma_route<Geometry, Nvfp4A4T32R64>();
    if (tokens <= 96) return nvfp4_a4_mma_route<Geometry, Nvfp4A4T32R128>();
    if (tokens <= 128) return nvfp4_a4_mma_route<Geometry, Nvfp4A4T128R128PipelinedW16>();
    if (tokens <= 192) return nvfp4_a4_mma_route<Geometry, Nvfp4A4T64R128>();
    if (tokens <= 384) return nvfp4_a4_mma_route<Geometry, Nvfp4A4T128R128Resident>();
    if (tokens <= 512) return nvfp4_a4_mma_route<Geometry, Nvfp4A4T128R128PipelinedW16>();
    return nvfp4_a4_mma_route<Geometry, Nvfp4A4T128R128Resident>();
}

inline bool uses_a4(std::int32_t, std::int32_t max_tokens) {
    return max_tokens >= kNvfp4AttnInputFamilyFirstA4Tokens;
}

template <class Geometry>
constexpr Nvfp4LinearShape shape{Geometry::kOutputRows, Geometry::kInputRows, a16<Geometry>,
                                 a4<Geometry>, uses_a4};
} // namespace ninfer::ops::detail::nvfp4_attn_qkgv

// The GDN q|k|v|z, measured on [16384,5120]: its two-device row half [8192,5120] inherits it. The
// TMA GEMM from 512 columns.
namespace ninfer::ops::detail::nvfp4_gdn_qkvz {
using C2 = Nvfp4A16SimtSchedule<4, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::SharedPhase,
                                Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                Nvfp4SimtBlockOrder::RowsContiguous, 1>;

template <class Geometry>
Nvfp4Launch a16(std::int32_t tokens) {
    if (tokens == 1) return nvfp4_linear_a16_gemv<Geometry, nvfp4_family::Gemv>;
    if (tokens <= 2) return nvfp4_linear_a16_simt<Geometry, 2, C2, true>;
    if (tokens <= 8) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<8, 8, 2>>;
    if (tokens <= 16) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<16, 8, 2>>;
    if (tokens <= 24) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 8, 1>>;
    if (tokens <= 32) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 8, 2>>;
    if (tokens <= 64)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>>;
    if (tokens <= 96)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>;
    return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}

template <class Geometry>
Nvfp4A4Route a4(std::int32_t tokens) {
    if (tokens <= 32) return nvfp4_a4_mma_route<Geometry, Nvfp4A4T32R64>();
    if (tokens <= 64) return nvfp4_a4_mma_route<Geometry, Nvfp4A4T32R128>();
    if (tokens <= 96) return nvfp4_a4_mma_route<Geometry, Nvfp4A4T32R128>();
    if (tokens <= 128) return nvfp4_a4_mma_route<Geometry, Nvfp4A4T128R128Pipelined>();
    if (tokens <= 192) return nvfp4_a4_mma_route<Geometry, Nvfp4A4T64R128>();
    return nvfp4_a4_mma_route<Geometry, Nvfp4A4T128R128Resident>();
}

template <class Geometry>
constexpr Nvfp4LinearShape shape{Geometry::kOutputRows, Geometry::kInputRows, a16<Geometry>,
                                 a4<Geometry>, nvfp4_family::always_a4};

template <class Geometry, Nvfp4GeometryId Tma>
constexpr Nvfp4LinearShape shape_with_tma{
    Geometry::kOutputRows, Geometry::kInputRows, a16<Geometry>,
    nvfp4_family::with_tma<a4<Geometry>, 512, Tma>, nvfp4_family::always_a4};
} // namespace ninfer::ops::detail::nvfp4_gdn_qkvz

// The MLP down projection, measured on [5120,17408] (nvfp4_down) and on its two-device input-column
// half [5120,8704] (nvfp4_down_half, its own SIMT tile at T=3..5 and sliced-K partition at
// T=6..8). The FFN-14336 down [5120,14336] inherits the first, its half [5120,7168] the second.
// Both share the A4 band, the A4 floor kNvfp4DownFamilyFirstA4Tokens (which linear_add over the
// other half of a row-parallel pair reads too, so both ranks take the same activation route at
// every width), the TMA GEMM from kNvfp4OutputDownFamilyFirstTmaTokens, and the AllowA4Decode
// selection.
namespace ninfer::ops::detail::nvfp4_down {
using C2 = Nvfp4A16SimtSchedule<4, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                Nvfp4SimtBlockOrder::RowsContiguous, 1>;
// C4 (T=3..4) and C5 ask for 6 CTAs per SM (<= 80 registers instead of 112); rows stay one per
// warp, so only speed changes. On an RTX 5070 Ti [5120,8704] T=3/4/5: 50.1/50.4/53.3 ->
// 44.7/44.8/49.0 us; [5120,17408]: 94.2/94.7/99.3 -> 85.5/85.9/97.0 us.
using C4 = Nvfp4A16SimtSchedule<4, 1, 2, 16, 4, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                Nvfp4SimtBlockOrder::RowsContiguous, 6>;
using C5 = Nvfp4A16SimtSchedule<4, 1, 2, 16, 5, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                Nvfp4SimtBlockOrder::RowsContiguous, 6>;

template <class Geometry>
Nvfp4Launch a16(std::int32_t tokens) {
    if (tokens == 1) return nvfp4_linear_a16_gemv<Geometry, nvfp4_family::Gemv>;
    if (tokens <= 2) return nvfp4_linear_a16_simt<Geometry, 2, C2, true>;
    if (tokens <= 4) return nvfp4_linear_a16_simt<Geometry, 4, C4, false>;
    if (tokens == 5) return nvfp4_linear_a16_simt<Geometry, 5, C5, true>;
    if (tokens <= 8) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<8, 4, 2>>;
    if (tokens <= 16) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<16, 4, 2>>;
    if (tokens <= 24) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 4, 2>>;
    if (tokens <= 32) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 4, 1>>;
    if (tokens <= 48) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 4, 1>>;
    if (tokens <= 64) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<64, 2, 2>>;
    if (tokens <= 128)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>>;
    return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}

// AllowA4Decode at T=1..7: the SIMT kernel at every width, instead of the GEMV at T=1 and the
// sliced-K schedule at T=6..7. Its per-row K loop depends on the lanes per row, the values per
// lane and the accumulator chains only, which C4, C5 and these share, so every width gives a
// column the bits of T=4. linear_add over the other half does the same
// (nvfp4_linear_add_small_t.cu).
template <int Tokens>
using CDecode = Nvfp4A16SimtSchedule<4, 1, 2, 16, Tokens, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                     Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                     Nvfp4SimtBlockOrder::RowsContiguous, Tokens == 1 ? 1 : 6>;

template <class Geometry, Nvfp4Launch (*A16)(std::int32_t)>
Nvfp4Launch a16_decode(std::int32_t tokens) {
    switch (tokens) {
    case 1:
        return nvfp4_linear_a16_simt<Geometry, 1, CDecode<1>, true>;
    case 6:
        return nvfp4_linear_a16_simt<Geometry, 6, CDecode<6>, true>;
    case 7:
        return nvfp4_linear_a16_simt<Geometry, 7, CDecode<7>, true>;
    default:
        return A16(tokens);
    }
}

template <class Geometry>
Nvfp4A4Route a4(std::int32_t tokens) {
    if (tokens <= 64) return nvfp4_a4_mma_route<Geometry, Nvfp4A4T32R64>();
    if (tokens <= 128) return nvfp4_a4_mma_route<Geometry, Nvfp4A4T32R128>();
    if (tokens <= 192) return nvfp4_a4_mma_route<Geometry, Nvfp4A4T64R128>();
    if (tokens <= 384) return nvfp4_a4_mma_route<Geometry, Nvfp4A4T128R128Resident>();
    if (tokens <= 512) return nvfp4_a4_mma_route<Geometry, Nvfp4A4T128R128Pipelined>();
    return nvfp4_a4_mma_route<Geometry, Nvfp4A4T128R128Resident>();
}

inline bool uses_a4(std::int32_t, std::int32_t max_tokens) {
    return max_tokens >= kNvfp4DownFamilyFirstA4Tokens;
}

template <class Geometry, Nvfp4GeometryId Tma>
constexpr auto a4_with_tma =
    nvfp4_family::with_tma<a4<Geometry>, kNvfp4OutputDownFamilyFirstTmaTokens, Tma>;

template <class Geometry>
constexpr Nvfp4LinearShape shape{Geometry::kOutputRows, Geometry::kInputRows, a16<Geometry>,
                                 a4<Geometry>, uses_a4};

template <class Geometry, Nvfp4GeometryId Tma>
constexpr Nvfp4LinearShape shape_with_tma_decode{Geometry::kOutputRows,
                                                 Geometry::kInputRows,
                                                 a16<Geometry>,
                                                 a4_with_tma<Geometry, Tma>,
                                                 uses_a4,
                                                 a16_decode<Geometry, a16<Geometry>>};
} // namespace ninfer::ops::detail::nvfp4_down

namespace ninfer::ops::detail::nvfp4_down_half {
// C4 (T=3..4) and C5 ask for 6 CTAs per SM (<= 80 registers instead of 112). Four rows per warp
// (16 per CTA, 320 CTAs) instead of two: each warp amortizes its activation reads over twice the
// weight rows, and the per-row K loop is unchanged, so outputs stay bit-identical. Measured with
// ninfer_linear_bench on an RTX 5070 Ti (70 SMs), medians of 3x60: T=3/4 46.8 -> 44.7 us
// (-4.5 %), T=5 53.0 -> 46.8 us (-11.6 %). Two rows per warp was the previous choice (and
// nvfp4_down's); one row per warp costs +30 %, eight rows spill registers (+120 %).
using C4 = Nvfp4A16SimtSchedule<4, 1, 4, 16, 4, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                Nvfp4SimtBlockOrder::RowsContiguous, 6>;
using C5 = Nvfp4A16SimtSchedule<4, 1, 4, 16, 5, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                Nvfp4SimtBlockOrder::RowsContiguous, 6>;

template <class Geometry>
Nvfp4Launch a16(std::int32_t tokens) {
    if (tokens == 1) return nvfp4_linear_a16_gemv<Geometry, nvfp4_family::Gemv>;
    if (tokens <= 2) return nvfp4_linear_a16_simt<Geometry, 2, nvfp4_down::C2, true>;
    if (tokens <= 4) return nvfp4_linear_a16_simt<Geometry, 4, C4, false>;
    if (tokens == 5) return nvfp4_linear_a16_simt<Geometry, 5, C5, true>;
    // Eight K-warps (256 threads) instead of four at T=6..8: 53.0 -> 50.9 us (-3.9 %) on the
    // RTX 5070 Ti; the K partition changes, so T=6..8 outputs are not bit-identical to nvfp4_down.
    if (tokens <= 8) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<8, 8, 2>>;
    if (tokens <= 16) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<16, 4, 2>>;
    if (tokens <= 24) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 4, 2>>;
    if (tokens <= 32) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 4, 1>>;
    if (tokens <= 48) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 4, 1>>;
    if (tokens <= 64) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<64, 2, 2>>;
    if (tokens <= 128)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>>;
    return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}

template <class Geometry>
constexpr Nvfp4LinearShape shape{Geometry::kOutputRows, Geometry::kInputRows, a16<Geometry>,
                                 nvfp4_down::a4<Geometry>, nvfp4_down::uses_a4};

template <class Geometry, Nvfp4GeometryId Tma>
constexpr Nvfp4LinearShape shape_with_tma_decode{
    Geometry::kOutputRows, Geometry::kInputRows,
    a16<Geometry>,         nvfp4_down::a4_with_tma<Geometry, Tma>,
    nvfp4_down::uses_a4,   nvfp4_down::a16_decode<Geometry, a16<Geometry>>};
} // namespace ninfer::ops::detail::nvfp4_down_half
