#include "ops/linear/nvfp4/nvfp4_shape_families.cuh"

namespace ninfer::ops::detail {
// The attention q|k|gate|v family's measured shape (nvfp4_attn_qkgv), with two TMA rows of its own
// in front of the family's MMA band: Tiled128 scales from 512 columns, Tiled256 from 1024.
namespace {
using Geometry = Nvfp4N14336K5120;

Nvfp4A4Route select_a4(std::int32_t tokens) {
    if (tokens >= 1024) return nvfp4_a4_tma_route<Nvfp4GeometryId::N14336K5120>();
    if (tokens >= 512)
        return nvfp4_a4_tma_route<Nvfp4GeometryId::N14336K5120, Nvfp4ScaleLayout::Tiled128>();
    return nvfp4_attn_qkgv::a4<Geometry>(tokens);
}
} // namespace

const Nvfp4LinearShape kNvfp4N14336K5120{14336, 5120, nvfp4_attn_qkgv::a16<Geometry>, select_a4,
                                         nvfp4_attn_qkgv::uses_a4};
} // namespace ninfer::ops::detail
