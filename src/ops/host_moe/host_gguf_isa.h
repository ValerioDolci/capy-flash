#pragma once

// Internal to host_gguf*.cpp / host_q8*.cpp: the per-ISA entry points behind HostGgufKernels and
// the geometry both ISAs share. Every function of gguf::detail::avx512 / gguf::detail::avx2 is
// compiled for its instruction set (#pragma GCC target after the includes) and runs only after
// host_gguf_isa_supported() (the files hold no static initializer that executes wide instructions;
// tools/check_host_isa.py verifies it on the built objects). The entry points do the same argument
// checks as the public functions of host_gguf.h / host_q8.h.

#include "ops/host_moe/host_gguf.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::host::gguf::detail {

// ------------------------------------------------------------------ geometry (both ISAs)
inline constexpr int kChunk = 64;

constexpr int chunks_of(ActFormat a) { return a == ActFormat::Q8_K ? 4 : 1; }

constexpr int group_elems_of(ActFormat a) { return a == ActFormat::Q8_K ? QK_K : 64; }

constexpr int wgroup_bytes(int C) { return C * kChunk + 64; }

constexpr int agroup_bytes(int C) { return C * kChunk + 128; }

constexpr int groups_of(ActFormat a, std::int32_t k) {
    return (k + group_elems_of(a) - 1) / group_elems_of(a);
}

// Throws std::invalid_argument unless k is a positive multiple of fmt's block size (baseline code).
void check_k(WeightFormat fmt, std::int32_t k);

#define NINFER_GGUF_ISA_ENTRY_POINTS                                                               \
    void decode_rows(WeightFormat fmt, const void* rows, std::size_t row_bytes,                    \
                     std::int32_t nrows, std::int32_t k, void* dst);                               \
    void decode_rows_big(WeightFormat fmt, const void* rows, std::size_t row_bytes,                \
                         std::int32_t nrows, std::int32_t k, void* dst);                           \
    void pack_act_q8_K(const block_q8_K* x, std::int32_t k, void* dst);                            \
    void pack_act_q8_0(const block_q8_0* x, std::int32_t k, void* dst);                            \
    void quantize_pack_bf16(ActFormat act, const std::uint16_t* x, std::int32_t k, void* dst);     \
    void quantize_pack_f32(ActFormat act, const float* x, std::int32_t k, void* dst);              \
    void quantize_pack_bf16_q8k_g32(const std::uint16_t* x, std::int32_t k, void* dst);            \
    void quantize_pack_bf16_q8f32(ActFormat act, const std::uint16_t* x, std::int32_t k,           \
                                  void* dst);                                                      \
    void quantize_pack_f32_q8f32(ActFormat act, const float* x, std::int32_t k, void* dst);        \
    void quantize_pack_q8_0_half_q8f32(const float* x32, std::int32_t group, std::int32_t half,    \
                                       void* dst);                                                 \
    void quantize_pack_q8_0_half(const float* x32, std::int32_t group, std::int32_t half,          \
                                 void* dst);                                                       \
    void gemm_decoded(WeightFormat fmt, const void* wdec, std::int32_t nrows, std::int32_t k,      \
                      const void* const* acts, std::int32_t T, float* out, std::size_t ldo);       \
    void decode_rows_bf16(WeightFormat fmt, const void* rows, std::size_t row_bytes,               \
                          std::int32_t nrows, std::int32_t k, std::uint16_t* dst);                 \
    void gemm_bf16(const std::uint16_t* weights, std::int32_t nrows, std::int32_t k,               \
                   const void* const* acts, std::int32_t T, float* out, std::size_t ldo);          \
    void silu_mul(const float* g, const float* u, float* h, std::int32_t n);                       \
    /* host_q8.h */                                                                                \
    void quantize_row_q8_K_bf16(const std::uint16_t* x, block_q8_K* y, std::int64_t k);            \
    void quantize_row_q8_K_f32(const float* x, block_q8_K* y, std::int64_t k);                     \
    void quantize_row_q8_0_bf16(const std::uint16_t* x, block_q8_0* y, std::int64_t k);            \
    void quantize_row_q8_0_f32(const float* x, block_q8_0* y, std::int64_t k);

namespace avx512 {
NINFER_GGUF_ISA_ENTRY_POINTS
} // namespace avx512

namespace avx2 {
NINFER_GGUF_ISA_ENTRY_POINTS
} // namespace avx2

#undef NINFER_GGUF_ISA_ENTRY_POINTS

} // namespace ninfer::ops::host::gguf::detail
