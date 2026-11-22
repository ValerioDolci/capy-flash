#pragma once

// TMA descriptors passed BY VALUE as kernel parameters (__grid_constant__).
//
// CUtensorMap is alignas(64) and the descriptor blocks are alignas(128). MSVC cannot pass an
// over-aligned type by value (C2719), and nvcc's host stub does exactly that, so on MSVC the host
// compilation pass sees a same-size type without the alignment. That is safe because the bytes a
// kernel receives are packed by the CUDA runtime from the DEVICE-side parameter layout: the host
// stub only hands over a pointer to each argument and the sizes match (static_assert below). The
// device pass, and every non-MSVC build, keeps CUtensorMap and the alignment unchanged.

#include <cuda.h>

#include <cstring>

#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)

namespace ninfer::ops {

struct KernelTensorMap {
    cuuint64_t opaque[16];

    KernelTensorMap() = default;

    KernelTensorMap(const CUtensorMap& map) noexcept { std::memcpy(opaque, &map, sizeof(opaque)); }

    // nvcc's host pass also parses kernel bodies, where `&descriptors.weight` is handed to a TMA
    // load that takes `const CUtensorMap*`. Those bodies never run on the host; this only lets them
    // type-check.
    const CUtensorMap* operator&() const noexcept {
        return reinterpret_cast<const CUtensorMap*>(opaque);
    }
};

static_assert(sizeof(KernelTensorMap) == sizeof(CUtensorMap));

} // namespace ninfer::ops

#    define NINFER_KERNEL_PARAM_ALIGNAS(bytes)

#else

namespace ninfer::ops {
using KernelTensorMap = CUtensorMap;
} // namespace ninfer::ops

#    define NINFER_KERNEL_PARAM_ALIGNAS(bytes) alignas(bytes)

#endif
