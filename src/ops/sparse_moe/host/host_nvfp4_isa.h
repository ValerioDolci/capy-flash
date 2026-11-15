#pragma once

// Internal to host_nvfp4*.cpp: the per-ISA entry points behind HostNvfp4Kernels and the scalar
// helpers both ISAs share. Every function of detail::avx512 / detail::avx2 carries its target
// attribute and runs only after host_isa_supported() (the files hold no static initializer that
// executes wide instructions; tools/check_host_isa.py verifies it on the built objects).

#include "ops/sparse_moe/host/host_nvfp4.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::host::detail {

// E4M3FN value table (256 entries, 64-byte aligned; NaN code 0x7F/0xFF read as 0).
[[nodiscard]] const float* e4m3_table() noexcept;
// e4m3_rn of the A4 codec (scalar; the same function for both ISAs).
[[nodiscard]] float a4_e4m3_rn(float v) noexcept;

#define NINFER_HOST_ISA_ENTRY_POINTS                                                               \
    void quantize_blocks(const float* x, std::int32_t block0, std::int32_t blocks,                 \
                         W4A8Activation a);                                                        \
    void quantize_bf16(const std::uint16_t* x, std::int32_t k, W4A8Activation a);                  \
    void quantize_blocks_a4(const float* x, std::int32_t block0, std::int32_t blocks, float gs,    \
                            W4A8Activation a);                                                     \
    void quantize_bf16_a4(const std::uint16_t* x, std::int32_t k, float gs, W4A8Activation a);     \
    void rows_dot(const Nvfp4Plane& plane, std::int64_t r0, std::int32_t n,                        \
                  const W4A8Activation* acts, std::int32_t tokens, float* out, std::size_t ldo);

namespace avx512 {
NINFER_HOST_ISA_ENTRY_POINTS
} // namespace avx512

namespace avx2 {
NINFER_HOST_ISA_ENTRY_POINTS
} // namespace avx2

#undef NINFER_HOST_ISA_ENTRY_POINTS

} // namespace ninfer::ops::host::detail
