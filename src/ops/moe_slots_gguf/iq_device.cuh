// Modified for capy-flash from fngine:src/kernels/moe/iq_device.cuh @ c20a885
// (satellitedown/fngine). See NOTICE.
// Modified for fngine from llama.cpp/ggml:ggml/src/ggml-cuda/vecdotq.cuh,
// ggml/src/ggml-cuda/quantize.cu, ggml/src/ggml-quants.c, ggml/src/ggml-common.h @
// 3cf03257f219afbe7334045ff7c6a06ac68c627d. See NOTICE.
// SPDX-License-Identifier: MIT (ggml portions)
//
// Device-side GGUF expert formats used by the moe_slots_gguf / moe_grouped_gguf kernels, over the
// block structs of ggml-iq.h (ggml-common.h layout):
//   * Sub32 integer decode of one 32-element sub-block for dp4a dot products (ported from the
//     vec_dot_*_q8_1 functions of vecdotq.cuh; the IQ2_S / IQ3_XXS "+0.5" scale terms are carried
//     as exact odd integer multipliers instead of ggml's truncating integer division),
//   * dequant8: exact FP32 values of 8 consecutive elements (ported from the dequantize_row_*
//     reference functions of ggml-quants.c; identical products),
//   * quantize_q8_lane: ggml quantize_q8_1 for one 32-value block with an FP32 scale.
// Changes in this tree: the format enum is ninfer::ops::MoeGgufFormat; quantize_q8_lane pins both
// divisions to IEEE round-to-nearest (__fdiv_rn), the rule of the host quantizers, whatever the
// compile flags.
#pragma once

#include "ninfer/ops/moe_slots_gguf.h"

#include "ggml-iq/ggml-iq.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdint.h>

namespace ninfer::ops::gguf_device {

// Device copies of the ggml tables (kmask_iq2xs, ksigns_iq2xs, iq2s_grid, iq3xxs_grid, iq3s_grid,
// kvalues_iq4nl). The shared header declares them `static const`; give them device storage here.
#define static static __device__
#include "ggml-iq/ggml-iq-tables.h"
#undef static

using Format = MoeGgufFormat;

static_assert(sizeof(block_iq2_s) == 82);
static_assert(sizeof(block_iq3_xxs) == 98);
static_assert(sizeof(block_iq3_s) == 110);
static_assert(sizeof(block_iq4_xs) == 136);
static_assert(sizeof(block_iq4_nl) == 18);
static_assert(sizeof(block_q2_0) == 18);

template <Format F>
struct FormatInfo;

template <>
struct FormatInfo<Format::IQ2_S> {
    using Block                      = block_iq2_s;
    static constexpr int kBlockElems = 256;
};

template <>
struct FormatInfo<Format::IQ3_XXS> {
    using Block                      = block_iq3_xxs;
    static constexpr int kBlockElems = 256;
};

template <>
struct FormatInfo<Format::IQ3_S> {
    using Block                      = block_iq3_s;
    static constexpr int kBlockElems = 256;
};

template <>
struct FormatInfo<Format::IQ4_XS> {
    using Block                      = block_iq4_xs;
    static constexpr int kBlockElems = 256;
};

template <>
struct FormatInfo<Format::IQ4_NL> {
    using Block                      = block_iq4_nl;
    static constexpr int kBlockElems = 32;
};

template <>
struct FormatInfo<Format::Q2_0> {
    using Block                      = block_q2_0;
    static constexpr int kBlockElems = 64;
};

template <Format F, int K>
inline constexpr int kRowBytes =
    K / FormatInfo<F>::kBlockElems * static_cast<int>(sizeof(typename FormatInfo<F>::Block));

__device__ __forceinline__ float fp16_bits_to_float(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

// Little-endian 32-bit word from 2-byte aligned storage (ggml get_int_b2).
__device__ __forceinline__ int get_int_b2(const void* x, int i32) {
    const uint16_t* x16 = static_cast<const uint16_t*>(x);
    return int(uint32_t(x16[2 * i32]) | (uint32_t(x16[2 * i32 + 1]) << 16));
}

__device__ __forceinline__ int get_int_b4(const void* x, int i32) {
    return static_cast<const int*>(x)[i32];
}

// kvalues_iq4nl packed as 4 little-endian words for __byte_perm lookups.
__device__ __forceinline__ uint32_t iq4nl_word(int i) {
    constexpr int8_t v[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                              1,    13,   25,  38,  53,  69,  89,  113};
    return uint32_t(uint8_t(v[4 * i])) | (uint32_t(uint8_t(v[4 * i + 1])) << 8) |
           (uint32_t(uint8_t(v[4 * i + 2])) << 16) | (uint32_t(uint8_t(v[4 * i + 3])) << 24);
}

// ggml get_int_from_table_16 (CUDA path) over kvalues_iq4nl: q4 holds 8 nibble indices; .x gets
// the values of the low nibbles, .y of the high nibbles.
__device__ __forceinline__ int2 iq4nl_lookup(int q4) {
    const uint32_t t0 = iq4nl_word(0), t1 = iq4nl_word(1), t2 = iq4nl_word(2), t3 = iq4nl_word(3);
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210 | ((uint32_t(q4) & 0x88888888) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low   = __byte_perm(t0, t1, uint32_t(q4) >> shift);
        const uint32_t high  = __byte_perm(t2, t3, uint32_t(q4) >> shift);
        tmp[i]               = __byte_perm(low, high, sel >> shift);
    }
    return make_int2(int(__byte_perm(tmp[0], tmp[1], 0x6420)),
                     int(__byte_perm(tmp[0], tmp[1], 0x7531)));
}

__device__ __forceinline__ uint32_t unpack_ksigns(uint8_t v) {
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ p << 7;
    return s * 0x01010101;
}

// ------------------------------------------------------------------ integer sub-block decode
// value[i] = d * m(i) * int8(q[i/4] byte i%4) for i in 0..31, with m(i) = m0 for i < 16 and m1
// otherwise. dot(sub, x_q8) = d * dx * (m0 * sum_{j<4} dp4a(q[j], u[j]) + m1 * sum_{j>=4} ...).
struct Sub32 {
    int q[8];
    int m0, m1;
    float d;
};

// Codebook bytes of a format (0: no table) and its global device copy. Kernels stage the table
// into shared memory and pass that copy to decode_sub32: per-lane random lookups into a global
// table cost one L1 wavefront per distinct line, into shared memory only bank conflicts.
template <Format F>
inline constexpr int kGridBytes = 0;
template <>
inline constexpr int kGridBytes<Format::IQ2_S> = 1024 * 8;
template <>
inline constexpr int kGridBytes<Format::IQ3_XXS> = 256 * 4;
template <>
inline constexpr int kGridBytes<Format::IQ3_S> = 512 * 4;

template <Format F>
__device__ __forceinline__ const void* grid_table() {
    if constexpr (F == Format::IQ2_S)
        return iq2s_grid;
    else if constexpr (F == Format::IQ3_XXS)
        return iq3xxs_grid;
    else if constexpr (F == Format::IQ3_S)
        return iq3s_grid;
    else
        return nullptr;
}

// `grid`: the format's codebook (grid_table<F>() or a shared-memory copy; unused without one).
template <Format F>
__device__ __forceinline__ void decode_sub32(const uint8_t* row, int sub, Sub32& s,
                                             const void* grid);

template <>
__device__ __forceinline__ void decode_sub32<Format::IQ2_S>(const uint8_t* row, int sub, Sub32& s,
                                                            const void* grid_base) {
    const block_iq2_s* b  = reinterpret_cast<const block_iq2_s*>(row) + (sub >> 3);
    const int ib32        = sub & 7;
    const int qs_packed   = get_int_b2(b->qs, ib32);
    const uint8_t* qs     = reinterpret_cast<const uint8_t*>(&qs_packed);
    const int qh          = b->qh[ib32];
    const int signs_32    = get_int_b2(b->qs, 8 + ib32);
    const uint8_t* signs8 = reinterpret_cast<const uint8_t*>(&signs_32);
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid =
            static_cast<const uint2*>(grid_base)[qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)];
        const int signs0 =
            __vcmpne4(((signs8[l0 / 2] & 0x03) << 7) | ((signs8[l0 / 2] & 0x0C) << 21), 0);
        const int signs1 =
            __vcmpne4(((signs8[l0 / 2] & 0x30) << 3) | ((signs8[l0 / 2] & 0xC0) << 17), 0);
        s.q[l0 + 0] = __vsub4(int(grid.x) ^ signs0, signs0);
        s.q[l0 + 1] = __vsub4(int(grid.y) ^ signs1, signs1);
    }
    const int sc = b->scales[ib32];
    s.m0         = 2 * (sc & 0x0F) + 1;
    s.m1         = 2 * (sc >> 4) + 1;
    s.d          = fp16_bits_to_float(b->d) * 0.125f;
}

template <>
__device__ __forceinline__ void decode_sub32<Format::IQ3_XXS>(const uint8_t* row, int sub, Sub32& s,
                                                              const void* grid) {
    const block_iq3_xxs* b = reinterpret_cast<const block_iq3_xxs*>(row) + (sub >> 3);
    const int ib32         = sub & 7;
    const int2 q3_packed = make_int2(get_int_b2(b->qs, 2 * ib32), get_int_b2(b->qs, 2 * ib32 + 1));
    const uint8_t* q3    = reinterpret_cast<const uint8_t*>(&q3_packed);
    const uint32_t aux32 = uint32_t(get_int_b2(b->qs, 16 + ib32));
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint32_t g0    = static_cast<const uint32_t*>(grid)[q3[l0 + 0]];
        const uint32_t g1    = static_cast<const uint32_t*>(grid)[q3[l0 + 1]];
        const uint32_t signs = unpack_ksigns(uint8_t(aux32 >> (7 * l0 / 2)));
        const int signs0     = __vcmpne4(signs & 0x08040201, 0);
        const int signs1     = __vcmpne4(signs & 0x80402010, 0);
        s.q[l0 + 0]          = __vsub4(int(g0) ^ signs0, signs0);
        s.q[l0 + 1]          = __vsub4(int(g1) ^ signs1, signs1);
    }
    s.m0 = s.m1 = 2 * int(aux32 >> 28) + 1;
    s.d         = fp16_bits_to_float(b->d) * 0.25f;
}

template <>
__device__ __forceinline__ void decode_sub32<Format::IQ3_S>(const uint8_t* row, int sub, Sub32& s,
                                                            const void* grid) {
    const block_iq3_s* b  = reinterpret_cast<const block_iq3_s*>(row) + (sub >> 3);
    const int ib32        = sub & 7;
    const int2 qs_packed  = make_int2(get_int_b2(b->qs, 2 * ib32), get_int_b2(b->qs, 2 * ib32 + 1));
    const uint8_t* qs     = reinterpret_cast<const uint8_t*>(&qs_packed);
    const int qh          = b->qh[ib32];
    const int signs_32    = get_int_b2(b->signs, ib32);
    const uint8_t* signs8 = reinterpret_cast<const uint8_t*>(&signs_32);
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint32_t g0 =
            static_cast<const uint32_t*>(grid)[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)];
        const uint32_t g1 =
            static_cast<const uint32_t*>(grid)[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)];
        const int signs0 =
            __vcmpne4(((signs8[l0 / 2] & 0x03) << 7) | ((signs8[l0 / 2] & 0x0C) << 21), 0);
        const int signs1 =
            __vcmpne4(((signs8[l0 / 2] & 0x30) << 3) | ((signs8[l0 / 2] & 0xC0) << 17), 0);
        s.q[l0 + 0] = __vsub4(int(g0) ^ signs0, signs0);
        s.q[l0 + 1] = __vsub4(int(g1) ^ signs1, signs1);
    }
    s.m0 = s.m1 = 1 + 2 * ((b->scales[ib32 >> 1] >> ((ib32 & 1) * 4)) & 0x0F);
    s.d         = fp16_bits_to_float(b->d);
}

template <>
__device__ __forceinline__ void decode_sub32<Format::IQ4_XS>(const uint8_t* row, int sub, Sub32& s,
                                                             const void*) {
    const block_iq4_xs* b = reinterpret_cast<const block_iq4_xs*>(row) + (sub >> 3);
    const int ib32        = sub & 7;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int2 v = iq4nl_lookup(get_int_b4(b->qs, 4 * ib32 + j));
        s.q[j]       = v.x;
        s.q[j + 4]   = v.y;
    }
    const int ls = ((b->scales_l[ib32 >> 1] >> ((ib32 & 1) * 4)) & 0x0F) |
                   (((b->scales_h >> (2 * ib32)) & 0x03) << 4);
    s.m0 = s.m1 = ls - 32;
    s.d         = fp16_bits_to_float(b->d);
}

template <>
__device__ __forceinline__ void decode_sub32<Format::IQ4_NL>(const uint8_t* row, int sub, Sub32& s,
                                                             const void*) {
    const block_iq4_nl* b = reinterpret_cast<const block_iq4_nl*>(row) + sub;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int2 v = iq4nl_lookup(get_int_b2(b->qs, l));
        s.q[l]       = v.x;
        s.q[l + 4]   = v.y;
    }
    s.m0 = s.m1 = 1;
    s.d         = fp16_bits_to_float(b->d);
}

// Q2_0: 2-bit codes, LSB first, value (q - 1) * d; one 64-element block = two sub-blocks.
template <>
__device__ __forceinline__ void decode_sub32<Format::Q2_0>(const uint8_t* row, int sub, Sub32& s,
                                                           const void*) {
    const block_q2_0* b = reinterpret_cast<const block_q2_0*>(row) + (sub >> 1);
    const uint16_t* qs  = reinterpret_cast<const uint16_t*>(b->qs) + (sub & 1) * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t q  = qs[j];
        const uint32_t lo = q & 0xFF, hi = q >> 8;
        const uint32_t ex =
            (lo & 0x03) | ((lo & 0x0C) << 6) | ((lo & 0x30) << 12) | ((lo & 0xC0) << 18);
        const uint32_t ey =
            (hi & 0x03) | ((hi & 0x0C) << 6) | ((hi & 0x30) << 12) | ((hi & 0xC0) << 18);
        s.q[2 * j + 0] = int(__vsub4(ex, 0x01010101u));
        s.q[2 * j + 1] = int(__vsub4(ey, 0x01010101u));
    }
    s.m0 = s.m1 = 1;
    s.d         = fp16_bits_to_float(b->d);
}

// Integer dot of a decoded sub-block with 32 int8 activations (8 words).
__device__ __forceinline__ int sub32_dot(const Sub32& s, const int* u) {
    int s0 = 0, s1 = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) s0 = __dp4a(s.q[j], u[j], s0);
#pragma unroll
    for (int j = 4; j < 8; ++j) s1 = __dp4a(s.q[j], u[j], s1);
    return s.m0 * s0 + s.m1 * s1;
}

// ------------------------------------------------------------------- exact 8-element dequant
// out[j] = value of element 8*g + j of the row (ggml dequantize_row_* reference arithmetic).
template <Format F>
__device__ __forceinline__ void dequant8(const uint8_t* row, int g, float out[8]);

template <>
__device__ __forceinline__ void dequant8<Format::IQ2_S>(const uint8_t* row, int g, float out[8]) {
    const block_iq2_s* b = reinterpret_cast<const block_iq2_s*>(row) + (g >> 5);
    const int ib32 = (g >> 2) & 7, l = g & 3;
    const float d    = fp16_bits_to_float(b->d);
    const int ls     = (b->scales[ib32] >> (4 * (l >> 1))) & 0x0F;
    const float dl   = d * (0.5f + float(ls)) * 0.25f;
    const uint2 grid = reinterpret_cast<const uint2*>(
        iq2s_grid)[b->qs[4 * ib32 + l] | ((b->qh[ib32] << (8 - 2 * l)) & 0x300)];
    const uint8_t signs = b->qs[32 + 4 * ib32 + l];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t gv = ((j < 4 ? grid.x : grid.y) >> (8 * (j & 3))) & 0xFF;
        out[j]            = dl * float(gv) * ((signs >> j) & 1 ? -1.f : 1.f);
    }
}

template <>
__device__ __forceinline__ void dequant8<Format::IQ3_XXS>(const uint8_t* row, int g, float out[8]) {
    const block_iq3_xxs* b = reinterpret_cast<const block_iq3_xxs*>(row) + (g >> 5);
    const int ib32 = (g >> 2) & 7, l = g & 3;
    const float d        = fp16_bits_to_float(b->d);
    const uint32_t aux32 = uint32_t(get_int_b2(b->qs, 16 + ib32));
    const float db       = d * (0.5f + float(aux32 >> 28)) * 0.5f;
    const uint32_t sv    = (aux32 >> (7 * l)) & 127; // ksigns_iq2xs[sv] = sv | parity(sv) << 7
    const uint32_t signs = sv | ((__popc(sv) & 1u) << 7);
    const uint32_t g1    = iq3xxs_grid[b->qs[8 * ib32 + 2 * l + 0]];
    const uint32_t g2    = iq3xxs_grid[b->qs[8 * ib32 + 2 * l + 1]];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t gv = ((j < 4 ? g1 : g2) >> (8 * (j & 3))) & 0xFF;
        out[j]            = db * float(gv) * ((signs >> j) & 1 ? -1.f : 1.f);
    }
}

template <>
__device__ __forceinline__ void dequant8<Format::IQ3_S>(const uint8_t* row, int g, float out[8]) {
    const block_iq3_s* b = reinterpret_cast<const block_iq3_s*>(row) + (g >> 5);
    const int ib32 = (g >> 2) & 7, l = g & 3;
    const float d       = fp16_bits_to_float(b->d);
    const float db      = d * float(1 + 2 * ((b->scales[ib32 >> 1] >> ((ib32 & 1) * 4)) & 0x0F));
    const int qh        = b->qh[ib32];
    const uint32_t g1   = iq3s_grid[b->qs[8 * ib32 + 2 * l + 0] | ((qh << (8 - 2 * l)) & 256)];
    const uint32_t g2   = iq3s_grid[b->qs[8 * ib32 + 2 * l + 1] | ((qh << (7 - 2 * l)) & 256)];
    const uint8_t signs = b->signs[4 * ib32 + l];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t gv = ((j < 4 ? g1 : g2) >> (8 * (j & 3))) & 0xFF;
        out[j]            = db * float(gv) * ((signs >> j) & 1 ? -1.f : 1.f);
    }
}

template <>
__device__ __forceinline__ void dequant8<Format::IQ4_XS>(const uint8_t* row, int g, float out[8]) {
    const block_iq4_xs* b = reinterpret_cast<const block_iq4_xs*>(row) + (g >> 5);
    const int ib = (g >> 2) & 7, part = g & 3;
    const float d = fp16_bits_to_float(b->d);
    const int ls =
        ((b->scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F) | (((b->scales_h >> (2 * ib)) & 3) << 4);
    const float dl      = d * float(ls - 32);
    const uint32_t* q32 = reinterpret_cast<const uint32_t*>(b->qs + 16 * ib + 8 * (part & 1));
    const uint2 q = make_uint2(q32[0], q32[1]); // 4-byte aligned (staged rows may not be 8-aligned)
    const int2 v0 = iq4nl_lookup(int(q.x)), v1 = iq4nl_lookup(int(q.y));
    const uint32_t w0 = uint32_t(part >> 1 ? v0.y : v0.x), w1 = uint32_t(part >> 1 ? v1.y : v1.x);
#pragma unroll
    for (int j = 0; j < 8; ++j)
        out[j] = dl * float(int(int8_t(((j < 4 ? w0 : w1) >> (8 * (j & 3))) & 0xFF)));
}

template <>
__device__ __forceinline__ void dequant8<Format::IQ4_NL>(const uint8_t* row, int g, float out[8]) {
    const block_iq4_nl* b = reinterpret_cast<const block_iq4_nl*>(row) + (g >> 2);
    const int part        = g & 3;
    const float d         = fp16_bits_to_float(b->d);
    const uint32_t q0     = uint32_t(get_int_b2(b->qs, 2 * (part & 1)));
    const uint32_t q1     = uint32_t(get_int_b2(b->qs, 2 * (part & 1) + 1));
    const int2 v0 = iq4nl_lookup(int(q0)), v1 = iq4nl_lookup(int(q1));
    const uint32_t w0 = uint32_t(part >> 1 ? v0.y : v0.x), w1 = uint32_t(part >> 1 ? v1.y : v1.x);
#pragma unroll
    for (int j = 0; j < 8; ++j)
        out[j] = d * float(int(int8_t(((j < 4 ? w0 : w1) >> (8 * (j & 3))) & 0xFF)));
}

template <>
__device__ __forceinline__ void dequant8<Format::Q2_0>(const uint8_t* row, int g, float out[8]) {
    const block_q2_0* b = reinterpret_cast<const block_q2_0*>(row) + (g >> 3);
    const int part      = g & 7;
    const float d       = fp16_bits_to_float(b->d);
    const uint32_t q    = reinterpret_cast<const uint16_t*>(b->qs)[part];
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = float(int((q >> (2 * j)) & 3) - 1) * d;
}

// ------------------------------------------------------------------------- q8 activations
// ggml quantize_q8_1 for one 32-value block held one value per lane of a full warp:
// d = amax / 127, q = roundf(v / d) (0 when amax == 0). Returns q; *d_out gets d (FP32).
__device__ __forceinline__ int8_t quantize_q8_lane(float v, float* d_out) {
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = __fdiv_rn(amax, 127.0f);
    *d_out        = d;
    return amax == 0.0f ? int8_t(0) : int8_t(roundf(__fdiv_rn(v, d)));
}

} // namespace ninfer::ops::gguf_device
