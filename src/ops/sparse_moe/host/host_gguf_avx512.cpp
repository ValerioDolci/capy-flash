// Modified for fngine from llama.cpp/ggml:ggml/src/ggml-cpu/arch/x86/quants.c @
// 3cf03257f219afbe7334045ff7c6a06ac68c627d. See NOTICE.
// SPDX-License-Identifier: MIT
//
// Modified for capy-flash from fngine:src/cpu/iq_cpu.cpp @ c20a885 (satellitedown/fngine; this
// file is SPDX MIT because it derives from ggml). See NOTICE. capy-flash changes: namespace
// ninfer::ops::host::gguf::detail::avx512 (dispatched by host_gguf.cpp, AVX2 forms in
// host_gguf_avx2.cpp), include paths, lazily built constant tables, ISA by #pragma GCC target after
// the includes (no inline or template function of a header is emitted with AVX-512 instructions).
//
// Block semantics follow ggml's ggml_vec_dot_{iq2_s,iq3_xxs,iq3_s,iq4_xs}_q8_K and
// ggml_vec_dot_iq4_nl_q8_0 (grid tables, sign bits, sub-block scales, IQ4 nibble table, final
// 1/8 and 1/4 factors). fngine restructures them into a decode-once GEMM (see iq_cpu.h):
//  * each GGUF block is expanded once into biased uint8 weights in a lane-permuted order that
//    makes every int32 lane belong to one sub-block, so the sub-block scale is a per-lane float;
//  * a register-blocked vpdpbusd micro-kernel runs over (rows x tokens) tiles with exact int32
//    sums per block and one FP32 scale step per block.
// Q2_0 (value (q - 1) * d) is new: it uses the same biased form, w + 128 = q + 127.
#include "ops/sparse_moe/host/host_gguf_isa.h"

#include "ggml-iq/ggml-iq-tables.h"

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <type_traits>

namespace ninfer::ops::host::gguf::detail::avx512 {
namespace {

// ------------------------------------------------------------------ constant tables
// Built on first use from the AVX-512 kernels; the constructor is baseline code (it precedes the
// target pragma).
struct Tables {
    alignas(64) int32_t perm_q8k[4][16];      // (4j + c) & 31
    alignas(64) uint8_t sign_ctrl[4][64];     // byte 4j+b -> bit 16(j%2) + 4c + b
    alignas(64) uint8_t iq3_perm[64];         // byte 16c + j <- 4j + c
    alignas(64) uint8_t iq2_perm[64];         // byte 16p + j <- 2j + p (p < 2)
    alignas(64) uint8_t iq4xs_perm[4][64];    // byte 4j+b <- 16(j/2) + 4c + b
    alignas(64) int32_t half_idx[16];         // j / 2
    alignas(64) int32_t quarter_idx[16];      // j / 4
    alignas(64) int32_t odd4[16];             // 4 (j % 2)
    alignas(64) int32_t iq2_qh_shift[2][16];  // 4 (j % 2) + 2p
    alignas(64) int32_t iq3s_qh_shift[4][16]; // 4 (j % 2) + c
    alignas(64) int32_t pair4[16];            // 4 ((j / 2) % 2)
    alignas(64) int32_t pair2[16];            // 2 (j / 2)
    alignas(64) int32_t sel_even[16];         // 2j
    alignas(64) int32_t sel_odd[16];          // 2j + 1
    alignas(64) int32_t sign_k4[2][16];       // (16h + j) / 4
    alignas(64) int32_t shift7[16];           // 7 (j % 4)
    alignas(16) uint8_t iq4_biased[16];       // kvalues_iq4nl + 128
    // The IQ3 grids as byte planes for vpermi2b lookups (plane b = byte b of every grid word):
    // iq3xxs 256 entries, iq3s 512.
    alignas(64) uint8_t iq3xxs_plane[4][256];
    alignas(64) uint8_t iq3s_plane[4][512];
    alignas(64) uint8_t iq3_pos_bit[64]; // byte 16c + j -> 4j + c (bit of a 64-bit qh word)

    Tables() {
        std::memset(this, 0, sizeof(*this));
        for (int j = 0; j < 16; ++j) {
            for (int c = 0; c < 4; ++c) {
                perm_q8k[c][j]      = (4 * j + c) & 31;
                iq3s_qh_shift[c][j] = 4 * (j % 2) + c;
                for (int b = 0; b < 4; ++b) {
                    sign_ctrl[c][4 * j + b]  = uint8_t(16 * (j % 2) + 4 * c + b);
                    iq4xs_perm[c][4 * j + b] = uint8_t(16 * (j / 2) + 4 * c + b);
                }
                iq3_perm[16 * c + j] = uint8_t(4 * j + c);
            }
            for (int p = 0; p < 2; ++p) {
                iq2_perm[16 * p + j] = uint8_t(2 * j + p);
                iq2_qh_shift[p][j]   = 4 * (j % 2) + 2 * p;
                sign_k4[p][j]        = (16 * p + j) / 4;
            }
            half_idx[j]    = j / 2;
            quarter_idx[j] = j / 4;
            odd4[j]        = 4 * (j % 2);
            pair4[j]       = 4 * ((j / 2) % 2);
            pair2[j]       = 2 * (j / 2);
            sel_even[j]    = 2 * j;
            sel_odd[j]     = 2 * j + 1;
            shift7[j]      = 7 * (j % 4);
            iq4_biased[j]  = uint8_t(int(kvalues_iq4nl[j]) + 128);
            for (int c = 0; c < 4; ++c) iq3_pos_bit[16 * c + j] = uint8_t(4 * j + c);
        }
    }
};

void fill_planes(Tables& t) {
    for (int i = 0; i < 256; ++i)
        for (int b = 0; b < 4; ++b) t.iq3xxs_plane[b][i] = uint8_t(iq3xxs_grid[i] >> (8 * b));
    for (int i = 0; i < 512; ++i)
        for (int b = 0; b < 4; ++b) t.iq3s_plane[b][i] = uint8_t(iq3s_grid[i] >> (8 * b));
}

struct TablesWithPlanes : Tables {
    TablesWithPlanes() { fill_planes(*this); }
};

const Tables& tables() {
    static const TablesWithPlanes t;
    return t;
}


} // namespace
} // namespace ninfer::ops::host::gguf::detail::avx512

#pragma GCC push_options
#pragma GCC target(                                                                                \
    "avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,avx512vbmi,avx512bitalg,avx512vpopcntdq,avx512bf16,avx2,fma,f16c,bmi2")

namespace ninfer::ops::host::gguf::detail::avx512 {
namespace {

inline float fp16(uint16_t h) { return _cvtsh_ss(h); }

inline uint64_t load_u64(const void* p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}

inline uint32_t load_u32(const void* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

inline uint16_t load_u16(const void* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}

inline __m512i ld(const void* p) { return _mm512_loadu_si512(p); }

inline __m512 ldf(const void* p) { return _mm512_loadu_ps(p); }

inline void st(void* p, __m512i v) { _mm512_storeu_si512(p, v); }

inline void stf(void* p, __m512 v) { _mm512_storeu_ps(p, v); }

inline __m512i vi(const int32_t* p) { return _mm512_load_si512(p); }

inline __m512i vb(const uint8_t* p) { return _mm512_load_si512(p); }

inline __m512i k128() { return _mm512_set1_epi8(char(0x80)); }

template <int I>
inline __m128i lane128(__m512i v) {
    return _mm512_extracti32x4_epi32(v, I);
}

// Grid magnitudes g (< 128) with sign mask m (bit set -> negative) -> w + 128.
inline __m512i bias_sign(__m512i g, __mmask64 m) {
    return _mm512_mask_sub_epi8(_mm512_add_epi8(g, k128()), m, k128(), g);
}

// 8 bytes -> dword lane j = byte j / 2.
inline __m512i bytes8_halves(uint64_t v) {
    return _mm512_permutexvar_epi32(
        vi(tables().half_idx), _mm512_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<long long>(v))));
}

// 4 bytes -> dword lane j = byte j / 4.
inline __m512i bytes4_quarters(uint32_t v) {
    return _mm512_permutexvar_epi32(vi(tables().quarter_idx),
                                    _mm512_cvtepu8_epi32(_mm_cvtsi32_si128(static_cast<int>(v))));
}

inline __m512 lane_scale(__m512i ints, float d) {
    return _mm512_mul_ps(_mm512_cvtepi32_ps(ints), _mm512_set1_ps(d));
}

inline __m512i odd_of(__m512i s) { // 2s + 1
    return _mm512_add_epi32(_mm512_slli_epi32(s, 1), _mm512_set1_epi32(1));
}

// Weight rows stream from DRAM (cold experts); rows are decoded in address order, so a fixed
// prefetch distance also covers the start of the next row.
constexpr int kPrefetchBytes = 1024;

inline void prefetch_ahead(const void* p) {
    const char* c = static_cast<const char*>(p) + kPrefetchBytes;
    _mm_prefetch(c, _MM_HINT_T0);
    _mm_prefetch(c + 64, _MM_HINT_T0);
}

// ------------------------------------------------------------------ weight decoders
// Q8_K formats: one 256-element block -> [4 x 64 B w+128 (lane-permuted)][64 B F].
void dec_iq2_s(const block_iq2_s& x, uint8_t* dst) {
    const float d     = fp16(x.d) * 0.125f;
    const __m512i low = _mm512_permutexvar_epi8(
        vb(tables().iq2_perm),
        _mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.qs))));
    const __m512i qh    = bytes8_halves(load_u64(x.qh));
    const __m512i signs = _mm512_cvtepu32_epi64(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.qs + QK_K / 8)));
    const auto* grid = reinterpret_cast<const long long*>(iq2s_grid);
    for (int p = 0; p < 2; ++p) {
        // Grid qword 2j + p holds the element dwords 4j + 2p (low half) and 4j + 2p + 1 (high).
        const __m512i hi = _mm512_and_si512(_mm512_srlv_epi32(qh, vi(tables().iq2_qh_shift[p])),
                                            _mm512_set1_epi32(3));
        const __m512i lo = _mm512_cvtepu8_epi32(p == 0 ? lane128<0>(low) : lane128<1>(low));
        const __m512i gi = _mm512_or_si512(lo, _mm512_slli_epi32(hi, 8));
        const __m512i g0 = _mm512_i32gather_epi64(_mm512_castsi512_si256(gi), grid, 8);
        const __m512i g1 = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(gi, 1), grid, 8);
        for (int h = 0; h < 2; ++h) {
            const int c     = 2 * p + h;
            const __m512i g = _mm512_permutex2var_epi32(
                g0, vi(h == 0 ? tables().sel_even : tables().sel_odd), g1);
            st(dst + kChunk * c,
               bias_sign(g, _mm512_bitshuffle_epi64_mask(signs, vb(tables().sign_ctrl[c]))));
        }
    }
    const __m512i s =
        _mm512_and_si512(_mm512_srlv_epi32(bytes8_halves(load_u64(x.scales)), vi(tables().odd4)),
                         _mm512_set1_epi32(15));
    stf(dst + 4 * kChunk, lane_scale(odd_of(s), d));
}

// IQ3 decoders: the grid is looked up by vpermi2b on byte planes instead of gathered (vpgatherdd).
// perm byte 16c + j holds the index of chunk c, lane j; plane b of the lookup gives byte b of that
// grid word; a byte interleave and a 128-bit lane transpose put the words where a gather puts them,
// so the decoded bytes are those of the AVX2 decoders (ninfer_host_gguf_isa_test).
inline void interleave_planes(const __m512i (&P)[4], __m512i (&out)[4]) {
    const __m512i a_lo = _mm512_unpacklo_epi8(P[0], P[1]), a_hi = _mm512_unpackhi_epi8(P[0], P[1]);
    const __m512i b_lo = _mm512_unpacklo_epi8(P[2], P[3]), b_hi = _mm512_unpackhi_epi8(P[2], P[3]);
    const __m512i r0 = _mm512_unpacklo_epi16(a_lo, b_lo), r1 = _mm512_unpackhi_epi16(a_lo, b_lo);
    const __m512i r2 = _mm512_unpacklo_epi16(a_hi, b_hi), r3 = _mm512_unpackhi_epi16(a_hi, b_hi);
    const __m512i v0 = _mm512_shuffle_i32x4(r0, r1, 0x44), v1 = _mm512_shuffle_i32x4(r0, r1, 0xEE);
    const __m512i v2 = _mm512_shuffle_i32x4(r2, r3, 0x44), v3 = _mm512_shuffle_i32x4(r2, r3, 0xEE);
    out[0] = _mm512_shuffle_i32x4(v0, v2, 0x88);
    out[1] = _mm512_shuffle_i32x4(v0, v2, 0xDD);
    out[2] = _mm512_shuffle_i32x4(v1, v3, 0x88);
    out[3] = _mm512_shuffle_i32x4(v1, v3, 0xDD);
}

void dec_iq3_xxs(const block_iq3_xxs& x, uint8_t* dst) {
    const float d      = fp16(x.d) * 0.25f;
    const __m512i perm = _mm512_permutexvar_epi8(vb(tables().iq3_perm), ld(x.qs));
    const __m512i aux  = _mm512_castsi256_si512(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.qs + QK_K / 4)));
    // Sign byte k = 4 ib + l: 7-bit index (aux[ib] >> 7l) & 127 plus parity bit 7 (ksigns_iq2xs).
    __m128i sb[2];
    for (int h = 0; h < 2; ++h) {
        __m512i v = _mm512_and_si512(
            _mm512_srlv_epi32(_mm512_permutexvar_epi32(vi(tables().sign_k4[h]), aux),
                              vi(tables().shift7)),
            _mm512_set1_epi32(127));
        const __m512i par = _mm512_and_si512(_mm512_popcnt_epi32(v), _mm512_set1_epi32(1));
        sb[h]             = _mm512_cvtepi32_epi8(_mm512_or_si512(v, _mm512_slli_epi32(par, 7)));
    }
    const __m512i signs = _mm512_cvtepu32_epi64(_mm256_set_m128i(sb[1], sb[0]));
    const __mmask64 hi7 = _mm512_movepi8_mask(perm);
    __m512i P[4], g[4];
    for (int b = 0; b < 4; ++b) {
        const uint8_t* t = tables().iq3xxs_plane[b];
        const __m512i lo = _mm512_permutex2var_epi8(ld(t), perm, ld(t + 64));
        const __m512i hi = _mm512_permutex2var_epi8(ld(t + 128), perm, ld(t + 192));
        P[b]             = _mm512_mask_blend_epi8(hi7, lo, hi);
    }
    interleave_planes(P, g);
    for (int c = 0; c < 4; ++c)
        st(dst + kChunk * c,
           bias_sign(g[c], _mm512_bitshuffle_epi64_mask(signs, vb(tables().sign_ctrl[c]))));
    const __m512i ls = _mm512_srli_epi32(_mm512_permutexvar_epi32(vi(tables().half_idx), aux), 28);
    stf(dst + 4 * kChunk, lane_scale(odd_of(ls), d));
}

void dec_iq3_s(const block_iq3_s& x, uint8_t* dst) {
    const float d      = fp16(x.d);
    const __m512i perm = _mm512_permutexvar_epi8(vb(tables().iq3_perm), ld(x.qs));
    const __m512i signs =
        _mm512_cvtepu32_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.signs)));
    // Bit 8 of index 4j + c (qh bit 4j + c of the 64-bit word) for byte 16c + j.
    const __mmask64 hi8 = _mm512_bitshuffle_epi64_mask(
        _mm512_set1_epi64(static_cast<long long>(load_u64(x.qh))), vb(tables().iq3_pos_bit));
    const __mmask64 hi7 = _mm512_movepi8_mask(perm);
    __m512i P[4], g[4];
    for (int b = 0; b < 4; ++b) {
        const uint8_t* t = tables().iq3s_plane[b];
        const __m512i q0 = _mm512_permutex2var_epi8(ld(t), perm, ld(t + 64));
        const __m512i q1 = _mm512_permutex2var_epi8(ld(t + 128), perm, ld(t + 192));
        const __m512i q2 = _mm512_permutex2var_epi8(ld(t + 256), perm, ld(t + 320));
        const __m512i q3 = _mm512_permutex2var_epi8(ld(t + 384), perm, ld(t + 448));
        P[b]             = _mm512_mask_blend_epi8(hi8, _mm512_mask_blend_epi8(hi7, q0, q1),
                                                  _mm512_mask_blend_epi8(hi7, q2, q3));
    }
    interleave_planes(P, g);
    for (int c = 0; c < 4; ++c)
        st(dst + kChunk * c,
           bias_sign(g[c], _mm512_bitshuffle_epi64_mask(signs, vb(tables().sign_ctrl[c]))));
    const __m512i s =
        _mm512_and_si512(_mm512_srlv_epi32(bytes4_quarters(load_u32(x.scales)), vi(tables().pair4)),
                         _mm512_set1_epi32(15));
    stf(dst + 4 * kChunk, lane_scale(odd_of(s), d));
}

void dec_iq4_xs(const block_iq4_xs& x, uint8_t* dst) {
    const float d    = fp16(x.d);
    const __m512i q0 = ld(x.qs), q1 = ld(x.qs + 64);
    const __m512i tbl = _mm512_broadcast_i32x4(
        _mm_load_si128(reinterpret_cast<const __m128i*>(tables().iq4_biased)));
    for (int c = 0; c < 4; ++c) {
        __m512i b = _mm512_permutex2var_epi8(q0, vb(tables().iq4xs_perm[c]), q1);
        b         = _mm512_mask_srli_epi16(b, 0xCCCCCCCCu, b, 4); // odd lanes: high nibbles
        b         = _mm512_and_si512(b, _mm512_set1_epi8(0x0f));
        st(dst + kChunk * c, _mm512_shuffle_epi8(tbl, b));
    }
    const __m512i sl = _mm512_and_si512(
        _mm512_srlv_epi32(bytes4_quarters(load_u32(x.scales_l)), vi(tables().pair4)),
        _mm512_set1_epi32(15));
    const __m512i sh = _mm512_and_si512(
        _mm512_srlv_epi32(_mm512_set1_epi32(load_u16(&x.scales_h)), vi(tables().pair2)),
        _mm512_set1_epi32(3));
    const __m512i ls =
        _mm512_sub_epi32(_mm512_or_si512(sl, _mm512_slli_epi32(sh, 4)), _mm512_set1_epi32(32));
    stf(dst + 4 * kChunk, lane_scale(ls, d));
}

// Q8_0 formats: one 64-element group -> [64 B w+128 (natural order)][64 B F].
// Bytes [A(16) | A(16) | B(16) | B(16)] -> nibble indices [A lo | A hi | B lo | B hi].
inline __m512i nibbles_lo_hi(__m512i q) {
    q = _mm512_mask_blend_epi64(0xCC, q, _mm512_srli_epi16(q, 4));
    return _mm512_and_si512(q, _mm512_set1_epi8(0x0f));
}

void dec_iq4_nl(const block_iq4_nl* xa, const block_iq4_nl* xb, uint8_t* dst) {
    const __m512i tbl = _mm512_broadcast_i32x4(
        _mm_load_si128(reinterpret_cast<const __m128i*>(tables().iq4_biased)));
    const __m128i qa = _mm_loadu_si128(reinterpret_cast<const __m128i*>(xa->qs));
    const __m128i qb = xb ? _mm_loadu_si128(reinterpret_cast<const __m128i*>(xb->qs)) : qa;
    __m512i q        = _mm512_castsi128_si512(qa);
    q                = _mm512_inserti32x4(q, qa, 1);
    q                = _mm512_inserti32x4(q, qb, 2);
    q                = _mm512_inserti32x4(q, qb, 3);
    __m512i w        = _mm512_shuffle_epi8(tbl, nibbles_lo_hi(q));
    if (!xb) w = _mm512_mask_mov_epi8(k128(), 0xFFFFFFFFull, w); // padded half: w = 0
    st(dst, w);
    stf(dst + kChunk, _mm512_mask_blend_ps(0xFF00, _mm512_set1_ps(fp16(xa->d)),
                                           _mm512_set1_ps(xb ? fp16(xb->d) : 0.0f)));
}

void dec_q2_0(const block_q2_0& x, uint8_t* dst) {
    // Bit offsets of the 8 codes held by each qword (2 packed bytes): 0, 2, ..., 14.
    const __m512i ctrl = _mm512_set1_epi64(0x0E0C0A0806040200ll);
    const __m128i qb   = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x.qs));
    const __m512i q    = _mm512_and_si512(
        _mm512_multishift_epi64_epi8(ctrl, _mm512_cvtepu16_epi64(qb)), _mm512_set1_epi8(3));
    st(dst, _mm512_add_epi8(q, _mm512_set1_epi8(127))); // (q - 1) + 128
    stf(dst + kChunk, _mm512_set1_ps(fp16(x.d)));
}

template <class Block, void (*Dec)(const Block&, uint8_t*)>
void decode_rows_q8k(const void* rows, std::size_t row_bytes, int32_t nrows, int nblk,
                     uint8_t* dst) {
    constexpr int WG = wgroup_bytes(4);
    for (int r = 0; r < nrows; ++r) {
        const auto* x = reinterpret_cast<const Block*>(static_cast<const uint8_t*>(rows) +
                                                       std::size_t(r) * row_bytes);
        for (int b = 0; b < nblk; ++b, dst += WG) {
            prefetch_ahead(x + b);
            Dec(x[b], dst);
        }
    }
}

// ------------------------------------------------------------------ activation packing
inline __m512i lane_sums128(__m512i q) { // 128 * (sum of the 4 int8 of each lane)
    return _mm512_dpbusd_epi32(_mm512_setzero_si512(), k128(), q);
}

void pack_group_q8K(const block_q8_K& x, uint8_t* dst) {
    const __m512i q0 = ld(x.qs), q1 = ld(x.qs + 64), q2 = ld(x.qs + 128), q3 = ld(x.qs + 192);
    __m512i sum = _mm512_setzero_si512();
    for (int c = 0; c < 4; ++c) {
        const __m512i p = vi(tables().perm_q8k[c]);
        const __m512i v = _mm512_mask_blend_epi32(0xFF00, _mm512_permutex2var_epi32(q0, p, q1),
                                                  _mm512_permutex2var_epi32(q2, p, q3));
        st(dst + kChunk * c, v);
        sum = _mm512_add_epi32(sum, lane_sums128(v));
    }
    stf(dst + 4 * kChunk, _mm512_set1_ps(x.d));
    st(dst + 4 * kChunk + 64, _mm512_sub_epi32(_mm512_setzero_si512(), sum));
}

// A Q8_K-format group whose activation scale is per 32 values (eight Q8_0 blocks,
// ggml's quantize_row_q8_0_ref codes and binary16 scales) instead of one per 256: lane j
// (elements 16j..16j+15) takes the scale of block j / 2. Same micro-kernel, finer activations.
void pack_group_q8K_g32(const block_q8_0* x, uint8_t* dst) {
    alignas(64) int8_t qs[QK_K];
    for (int b = 0; b < 8; ++b) std::memcpy(qs + 32 * b, x[b].qs, 32);
    const __m512i q0 = ld(qs), q1 = ld(qs + 64), q2 = ld(qs + 128), q3 = ld(qs + 192);
    __m512i sum = _mm512_setzero_si512();
    for (int c = 0; c < 4; ++c) {
        const __m512i p = vi(tables().perm_q8k[c]);
        const __m512i v = _mm512_mask_blend_epi32(0xFF00, _mm512_permutex2var_epi32(q0, p, q1),
                                                  _mm512_permutex2var_epi32(q2, p, q3));
        st(dst + kChunk * c, v);
        sum = _mm512_add_epi32(sum, lane_sums128(v));
    }
    alignas(64) float da[16];
    for (int j = 0; j < 16; ++j) da[j] = fp16(x[j / 2].d);
    stf(dst + 4 * kChunk, ldf(da));
    st(dst + 4 * kChunk + 64, _mm512_sub_epi32(_mm512_setzero_si512(), sum));
}

// Activation codec of the GPU GGUF Ops (moe_slots_gguf: ggml quantize_q8_1 with an
// FP32 scale): per 32 values d = amax / 127, q = roundf(v / d) (halves away from zero; 0 when
// amax == 0), scale used = d in FP32. CPU and GPU then dot the same integer codes.
inline __m512 abs_ps(__m512 v) {
    return _mm512_castsi512_ps(
        _mm512_and_si512(_mm512_castps_si512(v), _mm512_set1_epi32(0x7fffffff)));
}

inline __m512i round_away(__m512 x) {
    __m512 t = _mm512_roundscale_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    const __mmask16 up =
        _mm512_cmp_ps_mask(abs_ps(_mm512_sub_ps(x, t)), _mm512_set1_ps(0.5f), _CMP_GE_OQ);
    const __m512 one = _mm512_castsi512_ps(_mm512_or_si512(
        _mm512_castps_si512(_mm512_set1_ps(1.0f)),
        _mm512_and_si512(_mm512_castps_si512(x), _mm512_set1_epi32(int(0x80000000u)))));
    return _mm512_cvttps_epi32(_mm512_mask_add_ps(t, up, t, one));
}

// 32 values -> 32 codes; returns d.
inline float q8f32_block(const float* v, int8_t* q) {
    const __m512 v0 = ldf(v), v1 = ldf(v + 16);
    const float amax = _mm512_reduce_max_ps(_mm512_max_ps(abs_ps(v0), abs_ps(v1)));
    if (amax == 0.0f) {
        std::memset(q, 0, 32);
        return 0.0f;
    }
    const float d   = amax / 127.0f;
    const __m512 dv = _mm512_set1_ps(d);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(q),
                     _mm512_cvtepi32_epi8(round_away(_mm512_div_ps(v0, dv))));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(q + 16),
                     _mm512_cvtepi32_epi8(round_away(_mm512_div_ps(v1, dv))));
    return d;
}

// Q8_K-format group (256 values) from eight 32-value blocks: lane j takes the scale of block j / 2.
void pack_group_q8K_d32(const int8_t* qs, const float* d8, uint8_t* dst) {
    const __m512i q0 = ld(qs), q1 = ld(qs + 64), q2 = ld(qs + 128), q3 = ld(qs + 192);
    __m512i sum = _mm512_setzero_si512();
    for (int c = 0; c < 4; ++c) {
        const __m512i p = vi(tables().perm_q8k[c]);
        const __m512i v = _mm512_mask_blend_epi32(0xFF00, _mm512_permutex2var_epi32(q0, p, q1),
                                                  _mm512_permutex2var_epi32(q2, p, q3));
        st(dst + kChunk * c, v);
        sum = _mm512_add_epi32(sum, lane_sums128(v));
    }
    alignas(64) float da[16];
    for (int j = 0; j < 16; ++j) da[j] = d8[j / 2];
    stf(dst + 4 * kChunk, ldf(da));
    st(dst + 4 * kChunk + 64, _mm512_sub_epi32(_mm512_setzero_si512(), sum));
}

template <class In>
void quantize_pack_q8f32_impl(ActFormat act, const In* x, int32_t k, uint8_t* d) {
    alignas(64) float v[QK_K];
    alignas(64) int8_t qs[QK_K];
    float dd[8];
    const auto widen = [&](const In* src, int n) {
        if constexpr (std::is_same_v<In, float>) {
            std::memcpy(v, src, sizeof(float) * n);
        } else {
            for (int i = 0; i < n; i += 16) {
                const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                stf(v + i, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16)));
            }
        }
    };
    if (act == ActFormat::Q8_K) {
        for (int b = 0; b < k / QK_K; ++b) {
            widen(x + std::size_t(b) * QK_K, QK_K);
            for (int s = 0; s < 8; ++s) dd[s] = q8f32_block(v + 32 * s, qs + 32 * s);
            pack_group_q8K_d32(qs, dd, d + std::size_t(b) * agroup_bytes(4));
        }
        return;
    }
    const int groups = (k + 63) / 64;
    for (int g = 0; g < groups; ++g) {
        const int n = std::min(64, k - 64 * g);
        widen(x + std::size_t(g) * 64, n);
        uint8_t* dst               = d + std::size_t(g) * agroup_bytes(1);
        alignas(64) int8_t q64[64] = {};
        dd[0]                      = q8f32_block(v, q64);
        dd[1]                      = n == 64 ? q8f32_block(v + 32, q64 + 32) : 0.0f;
        const __m512i qv           = ld(q64);
        st(dst, qv);
        stf(dst + kChunk,
            _mm512_mask_blend_ps(0xFF00, _mm512_set1_ps(dd[0]), _mm512_set1_ps(dd[1])));
        st(dst + kChunk + 64, _mm512_sub_epi32(_mm512_setzero_si512(), lane_sums128(qv)));
    }
}

void pack_group_q8_0(const block_q8_0* a, const block_q8_0* b, uint8_t* dst) {
    const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a->qs));
    const __m256i hi =
        b ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b->qs)) : _mm256_setzero_si256();
    const __m512i v = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
    st(dst, v);
    stf(dst + kChunk, _mm512_mask_blend_ps(0xFF00, _mm512_set1_ps(fp16(a->d)),
                                           _mm512_set1_ps(b ? fp16(b->d) : 0.0f)));
    st(dst + kChunk + 64, _mm512_sub_epi32(_mm512_setzero_si512(), lane_sums128(v)));
}

// ------------------------------------------------------------------ micro-kernel
// R decoded rows (stride ws) x N packed activation rows over G groups of C chunks.
template <int C, int R, int N>
void micro(const uint8_t* w, std::size_t ws, const uint8_t* const* a, int G, float* out,
           std::size_t ldo) {
    constexpr int WG = wgroup_bytes(C), AG = agroup_bytes(C);
    __m512 acc[R][N];
#pragma GCC unroll 16
    for (int r = 0; r < R; ++r)
#pragma GCC unroll 16
        for (int t = 0; t < N; ++t) acc[r][t] = _mm512_setzero_ps();
    const uint8_t* ap[N];
#pragma GCC unroll 16
    for (int t = 0; t < N; ++t) ap[t] = a[t];
    for (int g = 0; g < G; ++g) {
        const std::size_t wo = std::size_t(g) * WG, ao = std::size_t(g) * AG;
        __m512i is[R][N];
#pragma GCC unroll 16
        for (int t = 0; t < N; ++t) {
            const __m512i corr = ld(ap[t] + ao + C * kChunk + 64);
#pragma GCC unroll 16
            for (int r = 0; r < R; ++r) is[r][t] = corr;
        }
#pragma GCC unroll 4
        for (int c = 0; c < C; ++c) {
            __m512i av[N];
#pragma GCC unroll 16
            for (int t = 0; t < N; ++t) av[t] = ld(ap[t] + ao + c * kChunk);
#pragma GCC unroll 16
            for (int r = 0; r < R; ++r) {
                const __m512i wv = ld(w + r * ws + wo + c * kChunk);
#pragma GCC unroll 16
                for (int t = 0; t < N; ++t) is[r][t] = _mm512_dpbusd_epi32(is[r][t], wv, av[t]);
            }
        }
#pragma GCC unroll 16
        for (int r = 0; r < R; ++r) {
            const __m512 f = ldf(w + r * ws + wo + C * kChunk);
#pragma GCC unroll 16
            for (int t = 0; t < N; ++t) {
                const __m512 s = _mm512_mul_ps(f, ldf(ap[t] + ao + C * kChunk));
                acc[r][t]      = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is[r][t]), s, acc[r][t]);
            }
        }
    }
#pragma GCC unroll 16
    for (int r = 0; r < R; ++r)
#pragma GCC unroll 16
        for (int t = 0; t < N; ++t) out[r * ldo + t] = _mm512_reduce_add_ps(acc[r][t]);
}

using MicroFn = void (*)(const uint8_t*, std::size_t, const uint8_t* const*, int, float*,
                         std::size_t);

constexpr int kRows = 4;
template <int C>
constexpr int kTokens = C == 4 ? 3 : 4;

template <int C, int R, int... Ns>
constexpr std::array<MicroFn, sizeof...(Ns)> micro_row(std::integer_sequence<int, Ns...>) {
    return {&micro<C, R, Ns + 1>...};
}

template <int C>
struct MicroTable {
    static constexpr auto seq = std::make_integer_sequence<int, kTokens<C>>{};
    static constexpr std::array<std::array<MicroFn, kTokens<C>>, kRows> fns = {
        micro_row<C, 1>(seq), micro_row<C, 2>(seq), micro_row<C, 3>(seq), micro_row<C, 4>(seq)};
};

template <int C>
void gemm_impl(const uint8_t* w, std::size_t ws, int nrows, const uint8_t* const* a, int T, int G,
               float* out, std::size_t ldo) {
    constexpr int N = kTokens<C>;
    for (int t0 = 0; t0 < T; t0 += N) {
        const int nt = std::min(N, T - t0);
        for (int r0 = 0; r0 < nrows; r0 += kRows) {
            const int nr = std::min(kRows, nrows - r0);
            MicroTable<C>::fns[nr - 1][nt - 1](w + std::size_t(r0) * ws, ws, a + t0, G,
                                               out + std::size_t(r0) * ldo + t0, ldo);
        }
    }
}

// ------------------------------------------------------------------ silu
inline __m512 exp512(__m512 x) {
    // Cephes expf: exp(x) = 2^n * p(r), n = round(x / ln2), r = x - n ln2.
    x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-103.0f)), _mm512_set1_ps(88.5f));
    const __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f)),
                                          _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r       = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693359375f), x);
    r              = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.12194440e-4f), r);
    __m512 y       = _mm512_set1_ps(1.9875691500e-4f);
    y              = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.3981999507e-3f));
    y              = _mm512_fmadd_ps(y, r, _mm512_set1_ps(8.3334519073e-3f));
    y              = _mm512_fmadd_ps(y, r, _mm512_set1_ps(4.1665795894e-2f));
    y              = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.6666665459e-1f));
    y              = _mm512_fmadd_ps(y, r, _mm512_set1_ps(5.0000001201e-1f));
    y = _mm512_fmadd_ps(y, _mm512_mul_ps(r, r), _mm512_add_ps(r, _mm512_set1_ps(1.0f)));
    return _mm512_scalef_ps(y, n);
}

} // namespace

void decode_rows(WeightFormat fmt, const void* rows, std::size_t row_bytes, int32_t nrows,
                 int32_t k, void* dst_v) {
    check_k(fmt, k);
    auto* dst      = static_cast<uint8_t*>(dst_v);
    const int nblk = k / format_info(fmt).block_elems;
    switch (fmt) {
    case WeightFormat::IQ2_S:
        decode_rows_q8k<block_iq2_s, dec_iq2_s>(rows, row_bytes, nrows, nblk, dst);
        return;
    case WeightFormat::IQ3_XXS:
        decode_rows_q8k<block_iq3_xxs, dec_iq3_xxs>(rows, row_bytes, nrows, nblk, dst);
        return;
    case WeightFormat::IQ3_S:
        decode_rows_q8k<block_iq3_s, dec_iq3_s>(rows, row_bytes, nrows, nblk, dst);
        return;
    case WeightFormat::IQ4_XS:
        decode_rows_q8k<block_iq4_xs, dec_iq4_xs>(rows, row_bytes, nrows, nblk, dst);
        return;
    case WeightFormat::IQ4_NL: {
        constexpr int WG = wgroup_bytes(1);
        for (int r = 0; r < nrows; ++r) {
            const auto* x = reinterpret_cast<const block_iq4_nl*>(
                static_cast<const uint8_t*>(rows) + std::size_t(r) * row_bytes);
            for (int b = 0; b < nblk; b += 2, dst += WG) {
                prefetch_ahead(x + b);
                dec_iq4_nl(x + b, b + 1 < nblk ? x + b + 1 : nullptr, dst);
            }
        }
        return;
    }
    case WeightFormat::Q2_0: {
        constexpr int WG = wgroup_bytes(1);
        for (int r = 0; r < nrows; ++r) {
            const auto* x = reinterpret_cast<const block_q2_0*>(static_cast<const uint8_t*>(rows) +
                                                                std::size_t(r) * row_bytes);
            for (int b = 0; b < nblk; ++b, dst += WG) {
                prefetch_ahead(x + b);
                dec_q2_0(x[b], dst);
            }
        }
        return;
    }
    }
    throw std::invalid_argument("unknown weight format");
}

void pack_act_q8_K(const block_q8_K* x, int32_t k, void* dst) {
    act_row_bytes(ActFormat::Q8_K, k);
    auto* d = static_cast<uint8_t*>(dst);
    for (int b = 0; b < k / QK_K; ++b) pack_group_q8K(x[b], d + std::size_t(b) * agroup_bytes(4));
}

void pack_act_q8_0(const block_q8_0* x, int32_t k, void* dst) {
    act_row_bytes(ActFormat::Q8_0, k);
    auto* d      = static_cast<uint8_t*>(dst);
    const int nb = k / QK8_0;
    for (int b = 0, g = 0; b < nb; b += 2, ++g)
        pack_group_q8_0(x + b, b + 1 < nb ? x + b + 1 : nullptr,
                        d + std::size_t(g) * agroup_bytes(1));
}

void quantize_pack_bf16_q8f32(ActFormat act, const uint16_t* x, int32_t k, void* dst) {
    act_row_bytes(act, k);
    quantize_pack_q8f32_impl(act, x, k, static_cast<uint8_t*>(dst));
}

void quantize_pack_f32_q8f32(ActFormat act, const float* x, int32_t k, void* dst) {
    act_row_bytes(act, k);
    quantize_pack_q8f32_impl(act, x, k, static_cast<uint8_t*>(dst));
}

void quantize_pack_q8_0_half_q8f32(const float* x32, int32_t group, int32_t half, void* dst) {
    alignas(64) int8_t q[32];
    const float d   = q8f32_block(x32, q);
    auto* g         = static_cast<uint8_t*>(dst) + std::size_t(group) * agroup_bytes(1);
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(g + 32 * half), v);
    const __mmask16 m = half ? 0xFF00 : 0x00FF;
    _mm512_mask_storeu_ps(g + kChunk, m, _mm512_set1_ps(d));
    const __m512i lanes =
        half ? _mm512_inserti64x4(_mm512_setzero_si512(), v, 1) : _mm512_castsi256_si512(v);
    _mm512_mask_storeu_epi32(g + kChunk + 64, m,
                             _mm512_sub_epi32(_mm512_setzero_si512(), lane_sums128(lanes)));
}

void quantize_pack_bf16_q8k_g32(const uint16_t* x, int32_t k, void* dst) {
    act_row_bytes(ActFormat::Q8_K, k);
    auto* d = static_cast<uint8_t*>(dst);
    block_q8_0 q[8];
    for (int b = 0; b < k / QK_K; ++b) {
        quantize_row_q8_0_bf16(x + std::size_t(b) * QK_K, q, QK_K);
        pack_group_q8K_g32(q, d + std::size_t(b) * agroup_bytes(4));
    }
}

void quantize_pack_bf16(ActFormat act, const uint16_t* x, int32_t k, void* dst) {
    act_row_bytes(act, k);
    auto* d = static_cast<uint8_t*>(dst);
    if (act == ActFormat::Q8_K) {
        block_q8_K q;
        for (int b = 0; b < k / QK_K; ++b) {
            quantize_row_q8_K_bf16(x + std::size_t(b) * QK_K, &q, QK_K);
            pack_group_q8K(q, d + std::size_t(b) * agroup_bytes(4));
        }
    } else {
        block_q8_0 q[2];
        const int nb = k / QK8_0;
        for (int b = 0, g = 0; b < nb; b += 2, ++g) {
            const int n = std::min(2, nb - b);
            quantize_row_q8_0_bf16(x + std::size_t(b) * QK8_0, q, n * QK8_0);
            pack_group_q8_0(&q[0], n == 2 ? &q[1] : nullptr, d + std::size_t(g) * agroup_bytes(1));
        }
    }
}

void quantize_pack_f32(ActFormat act, const float* x, int32_t k, void* dst) {
    act_row_bytes(act, k);
    auto* d = static_cast<uint8_t*>(dst);
    if (act == ActFormat::Q8_K) {
        block_q8_K q;
        for (int b = 0; b < k / QK_K; ++b) {
            quantize_row_q8_K_f32(x + std::size_t(b) * QK_K, &q, QK_K);
            pack_group_q8K(q, d + std::size_t(b) * agroup_bytes(4));
        }
    } else {
        block_q8_0 q[2];
        const int nb = k / QK8_0;
        for (int b = 0, g = 0; b < nb; b += 2, ++g) {
            const int n = std::min(2, nb - b);
            quantize_row_q8_0_f32(x + std::size_t(b) * QK8_0, q, n * QK8_0);
            pack_group_q8_0(&q[0], n == 2 ? &q[1] : nullptr, d + std::size_t(g) * agroup_bytes(1));
        }
    }
}

void quantize_pack_q8_0_half(const float* x32, int32_t group, int32_t half, void* dst) {
    block_q8_0 q;
    quantize_row_q8_0_f32(x32, &q, QK8_0);
    auto* d         = static_cast<uint8_t*>(dst) + std::size_t(group) * agroup_bytes(1);
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q.qs));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + 32 * half), v);
    const __mmask16 m = half ? 0xFF00 : 0x00FF;
    _mm512_mask_storeu_ps(d + kChunk, m, _mm512_set1_ps(fp16(q.d)));
    const __m512i lanes =
        half ? _mm512_inserti64x4(_mm512_setzero_si512(), v, 1) : _mm512_castsi256_si512(v);
    _mm512_mask_storeu_epi32(d + kChunk + 64, m,
                             _mm512_sub_epi32(_mm512_setzero_si512(), lane_sums128(lanes)));
}

void gemm_decoded(WeightFormat fmt, const void* wdec, int32_t nrows, int32_t k,
                  const void* const* acts, int32_t T, float* out, std::size_t ldo) {
    if (nrows <= 0 || T <= 0) return;
    const ActFormat a    = format_info(fmt).act;
    const int G          = groups_of(a, k);
    const std::size_t ws = decoded_row_bytes(fmt, k);
    const auto* w        = static_cast<const uint8_t*>(wdec);
    const auto* ap       = reinterpret_cast<const uint8_t* const*>(acts);
    if (a == ActFormat::Q8_K)
        gemm_impl<4>(w, ws, nrows, ap, T, G, out, ldo);
    else
        gemm_impl<1>(w, ws, nrows, ap, T, G, out, ldo);
}

namespace {

// 16 two's-complement int8 weights times their scale -> 16 floats (one rounding, as
// float(w) * F in the decode path's semantics).
inline __m512 scaled16(__m128i q, float s) {
    return _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q)), _mm512_set1_ps(s));
}

// 64 natural-order biased weights (w + 128); quarter p (elements 16p..16p+15) has scale s[p].
// Stores the first `quarters` quarters (2 or 4) as BF16, round to nearest even.
inline void store_bf16_quarters(__m512i biased, const float (&s)[4], int quarters, uint16_t* dst) {
    const __m512i q = _mm512_xor_si512(biased, k128()); // w + 128 -> w as int8
    _mm512_storeu_si512(dst, (__m512i)_mm512_cvtne2ps_pbh(scaled16(lane128<1>(q), s[1]),
                                                          scaled16(lane128<0>(q), s[0])));
    if (quarters == 4)
        _mm512_storeu_si512(dst + 32, (__m512i)_mm512_cvtne2ps_pbh(scaled16(lane128<3>(q), s[3]),
                                                                   scaled16(lane128<2>(q), s[2])));
}

// Q8_K decoded group (chunk c, lane j, byte b <- element 16j + 4c + b) -> natural order:
// out[m] = elements 64m..64m+63, whose quarters are the lanes j = 4m..4m+3.
inline void unpermute_q8k(const uint8_t* g, __m512i (&out)[4]) {
    const __m512i z0 = _mm512_loadu_si512(g), z1 = _mm512_loadu_si512(g + kChunk);
    const __m512i z2 = _mm512_loadu_si512(g + 2 * kChunk), z3 = _mm512_loadu_si512(g + 3 * kChunk);
    // 4x4 dword transpose inside each 128-bit lane L: u_i lane L = the 16 bytes of j = 4L + i.
    const __m512i t0 = _mm512_unpacklo_epi32(z0, z1), t1 = _mm512_unpackhi_epi32(z0, z1);
    const __m512i t2 = _mm512_unpacklo_epi32(z2, z3), t3 = _mm512_unpackhi_epi32(z2, z3);
    const __m512i u0 = _mm512_unpacklo_epi64(t0, t2), u1 = _mm512_unpackhi_epi64(t0, t2);
    const __m512i u2 = _mm512_unpacklo_epi64(t1, t3), u3 = _mm512_unpackhi_epi64(t1, t3);
    // 4x4 transpose of 128-bit lanes: out[m] = [u0 lane m, u1 lane m, u2 lane m, u3 lane m].
    const __m512i v0 = _mm512_shuffle_i32x4(u0, u1, 0x44), v1 = _mm512_shuffle_i32x4(u0, u1, 0xEE);
    const __m512i v2 = _mm512_shuffle_i32x4(u2, u3, 0x44), v3 = _mm512_shuffle_i32x4(u2, u3, 0xEE);
    out[0] = _mm512_shuffle_i32x4(v0, v2, 0x88);
    out[1] = _mm512_shuffle_i32x4(v0, v2, 0xDD);
    out[2] = _mm512_shuffle_i32x4(v1, v3, 0x88);
    out[3] = _mm512_shuffle_i32x4(v1, v3, 0xDD);
}

} // namespace

// BF16 = RNE(float(w) * F) of the decoded layout, in natural element order. Each row segment is
// expanded by one decode_rows call into an L1-resident scratch, then converted in registers
// (Q8_K groups are transposed back from the VNNI lane order with shuffles).
// Big cores: the same decoder (AVX-512 CPUs are not hybrid).
void decode_rows_big(WeightFormat fmt, const void* rows, std::size_t row_bytes, int32_t nrows,
                     int32_t k, void* dst) {
    ::ninfer::ops::host::gguf::detail::avx512::decode_rows(fmt, rows, row_bytes, nrows, k, dst);
}

void decode_rows_bf16(WeightFormat fmt, const void* rows, std::size_t row_bytes, int32_t nrows,
                      int32_t k, uint16_t* dst) {
    check_k(fmt, k);
    const FormatInfo fi         = format_info(fmt);
    const int C                 = chunks_of(fi.act);
    const int elems             = group_elems_of(fi.act);
    const int WG                = wgroup_bytes(C);
    const int G                 = groups_of(fi.act, k);
    constexpr int kScratchBytes = 16384;
    const int seg_groups        = kScratchBytes / WG; // Q8_K: 51 groups, Q8_0: 128 groups
    // A Q8_0 segment boundary must fall on a whole GGUF block (32 elements, always true at 64).
    alignas(64) uint8_t scratch[kScratchBytes];
    for (int r = 0; r < nrows; ++r) {
        const auto* row = static_cast<const uint8_t*>(rows) + std::size_t(r) * row_bytes;
        uint16_t* out   = dst + std::size_t(r) * k;
        for (int g0 = 0; g0 < G; g0 += seg_groups) {
            const int g1    = std::min(G, g0 + seg_groups);
            const int base  = g0 * elems;
            const int seg_k = std::min(g1 * elems, k) - base;
            avx512::decode_rows(fmt, row + std::size_t(base / fi.block_elems) * fi.block_bytes,
                                row_bytes, 1, seg_k, scratch);
            for (int g = g0; g < g1; ++g) {
                const uint8_t* grp = scratch + std::size_t(g - g0) * WG;
                const auto* F      = reinterpret_cast<const float*>(grp + C * kChunk);
                uint16_t* o        = out + std::size_t(g) * elems;
                if (C == 4) {
                    __m512i nat[4];
                    unpermute_q8k(grp, nat);
                    for (int m = 0; m < 4; ++m) {
                        const float s[4] = {F[4 * m], F[4 * m + 1], F[4 * m + 2], F[4 * m + 3]};
                        store_bf16_quarters(nat[m], s, 4, o + 64 * m);
                    }
                } else {
                    // Lanes 0-7 carry the first 32-element block's scale, lanes 8-15 the second's.
                    const float s[4] = {F[0], F[4], F[8], F[12]};
                    const int live   = std::min(elems, k - g * elems);
                    store_bf16_quarters(_mm512_loadu_si512(grp), s, live == 64 ? 4 : 2, o);
                }
            }
        }
    }
}

namespace {
template <int R, int T>
void bf16_micro(const uint16_t* weights, int k, const void* const* acts, float* out,
                std::size_t ldo) {
    __m512 acc[R][T];
    for (int r = 0; r < R; ++r)
        for (int t = 0; t < T; ++t) acc[r][t] = _mm512_setzero_ps();
    for (int i = 0; i < k; i += 32) {
        __m512bh x[T];
        for (int t = 0; t < T; ++t)
            x[t] = (__m512bh)_mm512_loadu_si512(static_cast<const uint16_t*>(acts[t]) + i);
        for (int r = 0; r < R; ++r) {
            const __m512bh w = (__m512bh)_mm512_loadu_si512(weights + std::size_t(r) * k + i);
            for (int t = 0; t < T; ++t) acc[r][t] = _mm512_dpbf16_ps(acc[r][t], w, x[t]);
        }
    }
    for (int r = 0; r < R; ++r)
        for (int t = 0; t < T; ++t) out[std::size_t(r) * ldo + t] = _mm512_reduce_add_ps(acc[r][t]);
}

template <int R>
void bf16_rows(const uint16_t* weights, int k, const void* const* acts, int T, float* out,
               std::size_t ldo) {
    int t = 0;
    for (; t + 4 <= T; t += 4) bf16_micro<R, 4>(weights, k, acts + t, out + t, ldo);
    if (t + 2 <= T) {
        bf16_micro<R, 2>(weights, k, acts + t, out + t, ldo);
        t += 2;
    }
    if (t < T) bf16_micro<R, 1>(weights, k, acts + t, out + t, ldo);
}
} // namespace

void gemm_bf16(const uint16_t* weights, int32_t nrows, int32_t k, const void* const* acts,
               int32_t T, float* out, std::size_t ldo) {
    int r = 0;
    for (; r + 4 <= nrows; r += 4)
        bf16_rows<4>(weights + std::size_t(r) * k, k, acts, T, out + std::size_t(r) * ldo, ldo);
    for (; r < nrows; ++r)
        bf16_rows<1>(weights + std::size_t(r) * k, k, acts, T, out + std::size_t(r) * ldo, ldo);
}

void silu_mul(const float* g, const float* u, float* h, int32_t n) {
    for (int32_t i = 0; i < n; i += 16) {
        const __mmask16 m = n - i >= 16 ? __mmask16(0xFFFF) : __mmask16((1u << (n - i)) - 1u);
        const __m512 gv   = _mm512_maskz_loadu_ps(m, g + i);
        const __m512 uv   = _mm512_maskz_loadu_ps(m, u + i);
        const __m512 e    = exp512(_mm512_sub_ps(_mm512_setzero_ps(), gv));
        const __m512 s    = _mm512_div_ps(gv, _mm512_add_ps(_mm512_set1_ps(1.0f), e));
        _mm512_mask_storeu_ps(h + i, m, _mm512_mul_ps(s, uv));
    }
}

} // namespace ninfer::ops::host::gguf::detail::avx512

#pragma GCC pop_options
