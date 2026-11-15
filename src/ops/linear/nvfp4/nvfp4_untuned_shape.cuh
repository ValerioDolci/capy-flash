#pragma once
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::nvfp4_untuned {

// A16-only selection of the Qwen3.8-Flash-Next NVFP4 shapes without a measurement of their own:
// the Gemv and SIMT tile of n14336_k5120.cu (WideRows) or n5120_k3072.cu (NarrowRows), no A4 route.
using Gemv =
    Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
using WideRows   = Nvfp4A16SimtSchedule<8, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::SharedPhase,
                                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                        Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using NarrowRows = Nvfp4A16SimtSchedule<4, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                        Nvfp4SimtBlockOrder::RowsContiguous, 1>;

template <int N, int K, class Simt>
Nvfp4Launch select_a16(int tokens) {
    using Geometry = Nvfp4Geometry<N, K>;
    if (tokens == 1) return nvfp4_linear_a16_gemv<Geometry, Gemv>;
    if (tokens <= 2) return nvfp4_linear_a16_simt<Geometry, 2, Simt, true>;
    if (tokens <= 8) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<8, 8, 2>>;
    if (tokens <= 16) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<16, 8, 2>>;
    if (tokens <= 64)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>;
    return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}

template <int N, int K, class Simt>
void launch_a16(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    select_a16<N, K, Simt>(x.ne[1])(x, w, y, stream);
}

template <int N, int K>
void no_a4(const Tensor&, const Weight&, Tensor&, Nvfp4A4Workspace, cudaStream_t) {
    throw std::invalid_argument("nvfp4 linear [" + std::to_string(N) + "," + std::to_string(K) +
                                "]: no A4 route");
}

inline bool uses_a4(std::int32_t, std::int32_t) { return false; }

template <int N, int K, class Simt>
constexpr Nvfp4LinearShape shape{N, K, launch_a16<N, K, Simt>, no_a4<N, K>, uses_a4};

} // namespace ninfer::ops::detail::nvfp4_untuned
