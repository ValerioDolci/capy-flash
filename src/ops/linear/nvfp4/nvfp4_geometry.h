#pragma once
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
template <std::int32_t OutputRows, std::int32_t InputRows>
struct Nvfp4Geometry {
    static_assert(OutputRows > 0 && InputRows > 0);
    static_assert((OutputRows % 128) == 0);
    static_assert((InputRows % 64) == 0);

    static constexpr std::int32_t kOutputRows       = OutputRows;
    static constexpr std::int32_t kInputRows        = InputRows;
    static constexpr std::int32_t kGroupsPerRow     = InputRows / 16;
    static constexpr std::int32_t kScaleTilesPerRow = InputRows / 64;
    static constexpr std::int32_t kCodeBytesPerRow  = InputRows / 2;
};

template <std::int32_t InputRows>
struct Nvfp4ActivationGeometry {
    static_assert(InputRows > 0);
    static_assert((InputRows % 64) == 0);

    static constexpr std::int32_t kInputRows       = InputRows;
    static constexpr std::int32_t kGroupsPerRow    = InputRows / 16;
    static constexpr std::int32_t kCodeBytesPerRow = InputRows / 2;
};

using Nvfp4N14336K5120 = Nvfp4Geometry<14336, 5120>;
using Nvfp4N16384K5120 = Nvfp4Geometry<16384, 5120>;
using Nvfp4N34816K5120 = Nvfp4Geometry<34816, 5120>;
using Nvfp4N5120K6144  = Nvfp4Geometry<5120, 6144>;
using Nvfp4N5120K17408 = Nvfp4Geometry<5120, 17408>;
using Nvfp4N17408K5120 = Nvfp4Geometry<17408, 5120>;
using Nvfp4N5120K8704  = Nvfp4Geometry<5120, 8704>;
using Nvfp4N5120K3072  = Nvfp4Geometry<5120, 3072>;
// Two-device head-local shards of the fused attention [14336,5120] and GDN [16384,5120] input
// parents: problems of those projections' fused kernels and Linear registrations.
using Nvfp4N7168K5120 = Nvfp4Geometry<7168, 5120>;
using Nvfp4N8192K5120 = Nvfp4Geometry<8192, 5120>;
// The gate|up parent of a dense FFN of intermediate width 14336 (one device; its two-device half
// is [14336,5120], Nvfp4N14336K5120).
using Nvfp4N28672K5120 = Nvfp4Geometry<28672, 5120>;

// First token count at which nvfp4 linear_add over the attention output ([5120,6144] and its
// two-device half [5120,3072]) and over the MLP down projection ([5120,17408] and [5120,8704])
// takes the A4 route (nvfp4_linear_add_plan.cpp: 17 and 8). linear() over each half uses the same
// crossover, so rank 0's linear_add and rank 1's linear() in one row-parallel pair always take the
// same route.
inline constexpr std::int32_t kNvfp4OutputFamilyFirstA4Tokens = 17;
inline constexpr std::int32_t kNvfp4DownFamilyFirstA4Tokens   = 8;
// First token count at which the same row-parallel pairs take the A4 TMA route: linear_add over
// every problem above (nvfp4_linear_add_a4.cu) and linear() over [5120,6144], [5120,17408] and
// their halves, so both ranks of a pair also run the same schedule past the A4 crossover.
inline constexpr std::int32_t kNvfp4OutputDownFamilyFirstTmaTokens = 1024;
// First token count at which nvfp4 attn_input_proj over [14336,5120] and its two-device shard
// [7168,5120] takes the A4 route (nvfp4_attn_input_plan.cpp); linear() over the same two problems
// uses the same crossover. [14336,5120] is also the two-device half of the dense FFN-14336 gate|up:
// linear() there (the unfused chain) keeps this crossover, while the fused linear_swiglu over that
// half has its own (nvfp4_linear_swiglu_plan.cpp).
inline constexpr std::int32_t kNvfp4AttnInputFamilyFirstA4Tokens = 4;
// First token count at which the fused A4 launchers switch to their TMA GEMM:
// attn_input_proj over [14336,5120] and its shard (Tiled128 scales, Tiled256 from 1024),
// gdn_input_proj over [16384,5120] and its shard, and linear_swiglu over its gate|up problems.
inline constexpr std::int32_t kNvfp4AttnInputFamilyFirstTmaTokens = 512;
inline constexpr std::int32_t kNvfp4GdnInputFamilyFirstTmaTokens  = 512;
inline constexpr std::int32_t kNvfp4SwiGluFamilyFirstTmaTokens    = 256;

using Nvfp4Activation3072Geometry  = Nvfp4ActivationGeometry<3072>;
using Nvfp4Activation5120Geometry  = Nvfp4ActivationGeometry<5120>;
using Nvfp4Activation6144Geometry  = Nvfp4ActivationGeometry<6144>;
using Nvfp4Activation8704Geometry  = Nvfp4ActivationGeometry<8704>;
using Nvfp4Activation17408Geometry = Nvfp4ActivationGeometry<17408>;
// The dense FFN-14336 down projection's activation and its two-device half.
using Nvfp4Activation14336Geometry = Nvfp4ActivationGeometry<14336>;
using Nvfp4Activation7168Geometry  = Nvfp4ActivationGeometry<7168>;

enum class Nvfp4GeometryId : std::uint8_t {
    N14336K5120,
    N16384K5120,
    N34816K5120,
    N5120K6144,
    N5120K17408,
    // Two-device shards: the output-row half of the MLP gate/up projection and the input-column
    // halves of the MLP down projection and of the attention and GDN output projections.
    N17408K5120,
    N5120K8704,
    N5120K3072,
};

inline Nvfp4GeometryId resolve_nvfp4_geometry(std::int32_t output_rows, std::int32_t input_rows) {
    if (output_rows == Nvfp4N14336K5120::kOutputRows &&
        input_rows == Nvfp4N14336K5120::kInputRows) {
        return Nvfp4GeometryId::N14336K5120;
    }
    if (output_rows == Nvfp4N16384K5120::kOutputRows &&
        input_rows == Nvfp4N16384K5120::kInputRows) {
        return Nvfp4GeometryId::N16384K5120;
    }
    if (output_rows == Nvfp4N34816K5120::kOutputRows &&
        input_rows == Nvfp4N34816K5120::kInputRows) {
        return Nvfp4GeometryId::N34816K5120;
    }
    if (output_rows == Nvfp4N5120K6144::kOutputRows && input_rows == Nvfp4N5120K6144::kInputRows) {
        return Nvfp4GeometryId::N5120K6144;
    }
    if (output_rows == Nvfp4N5120K17408::kOutputRows &&
        input_rows == Nvfp4N5120K17408::kInputRows) {
        return Nvfp4GeometryId::N5120K17408;
    }
    if (output_rows == Nvfp4N17408K5120::kOutputRows &&
        input_rows == Nvfp4N17408K5120::kInputRows) {
        return Nvfp4GeometryId::N17408K5120;
    }
    if (output_rows == Nvfp4N5120K8704::kOutputRows && input_rows == Nvfp4N5120K8704::kInputRows) {
        return Nvfp4GeometryId::N5120K8704;
    }
    if (output_rows == Nvfp4N5120K3072::kOutputRows && input_rows == Nvfp4N5120K3072::kInputRows) {
        return Nvfp4GeometryId::N5120K3072;
    }
    throw std::invalid_argument("unsupported NVFP4 problem");
}

} // namespace ninfer::ops::detail
