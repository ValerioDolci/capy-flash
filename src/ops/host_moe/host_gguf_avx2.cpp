// Modified for fngine from llama.cpp/ggml:ggml/src/ggml-cpu/arch/x86/quants.c @
// 3cf03257f219afbe7334045ff7c6a06ac68c627d. See NOTICE.
// SPDX-License-Identifier: MIT
//
// The AVX2 + FMA + F16C form of host_gguf_avx512.cpp (namespace
// ninfer::ops::host::gguf::detail::avx2, dispatched by host_gguf.cpp) for CPUs without the AVX-512
// GGUF set (Zen 2/3, Intel client cores, Ice Lake). Same layouts and the same results bit for bit:
//  - decoders: the same biased bytes w + 128 in the same lane-permuted order and the same per-lane
//    scales (the same int -> float conversion and product). Grid words are looked up with scalar
//    loads assembled by _mm256_set_epi* as ggml's AVX2 kernels do (no vpgatherdd, slow on Zen 2);
//    sign bits are expanded with vpshufb + compare; the IQ3_XXS sign bytes (7 bits + parity) are
//    computed with shifts and an xor fold;
//  - packed activations: the same bytes (q, DA, ACORR = -128 * lane sum);
//  - micro-kernel: the exact integer lane sums sum(w * q) = ACORR + sum vpdpbusd(w + 128, q) with
//    vpmaddubsw(|q|, sign(w, q)) + vpmaddwd (no saturation: |w| <= 127 in every format, |q| <= 128,
//    so a pair sum is at most 2 * 128 * 127 = 32512), then per lane the same float product
//    F * DA and FMA into the accumulator; the 16 lanes are kept as two 256-bit halves and reduced
//    in _mm512_reduce_add_ps's order. Rows x tokens tiles differ from the AVX-512 ones; results do
//    not depend on the tile (each output is the same sequence of operations);
//  - silu_mul: the same exp polynomial with the same explicit FMAs; vscalefps(y, n) is y * 2^n1 *
//    2^n2 (n1 = floor(n / 2), n2 = n - n1): the first product is exact, the second rounds once,
//    as vscalefps does (also in the subnormal range);
//  - prefill: decode_rows_bf16 converts with vcvtne2ps2bf16's rounding (RNE, subnormals to signed
//    zero, quiet NaN); gemm_bf16 runs vdpbf16ps's two FMAs per lane (odd pair element, then even)
//    under MXCSR DAZ + FTZ + round-to-nearest.
// The file has no dynamic initializer (constexpr tables only) and is compiled with
// -ffp-contract=off (sources.cmake), like the AVX-512 one: every FMA is explicit.
#include "ops/host_moe/host_gguf_isa.h"

#include "ggml-iq/ggml-iq-tables.h"

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#pragma GCC push_options
#pragma GCC target("avx2,fma,f16c")

namespace ninfer::ops::host::gguf::detail::avx2 {
namespace {

// ------------------------------------------------------------------ constant tables
template <class T, std::size_t N>
struct alignas(32) Table {
    T v[N];
};

// Sign bit e of a 256-bit sign field (bit e % 8 of byte e / 8) for the element of chunk c, lane j,
// byte b (e = 16j + 4c + b) of the half of lanes 8h..8h+7, from the 16 field bytes 16h..16h+15
// broadcast to both 128-bit lanes: byte 16L + 4jj + b (j = 8h + 4L + jj) reads field byte
// 8L + 2jj + c/2 and tests bit 4(c % 2) + b.
constexpr auto kSignShuf = [] {
    std::array<Table<uint8_t, 32>, 4> t{};
    for (int c = 0; c < 4; ++c)
        for (int L = 0; L < 2; ++L)
            for (int jj = 0; jj < 4; ++jj)
                for (int b = 0; b < 4; ++b)
                    t[c].v[16 * L + 4 * jj + b] = uint8_t(8 * L + 2 * jj + c / 2);
    return t;
}();
constexpr auto kSignBit = [] {
    std::array<Table<uint8_t, 32>, 2> t{};
    for (int p = 0; p < 2; ++p)
        for (int i = 0; i < 32; ++i) t[p].v[i] = uint8_t(1u << (4 * p + i % 4));
    return t;
}();
constexpr auto kIq4Biased = [] {
    Table<uint8_t, 32> t{};
    for (int i = 0; i < 32; ++i) t.v[i] = uint8_t(int(kvalues_iq4nl[i % 16]) + 128);
    return t;
}();

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

inline __m256i ld(const void* p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); }

inline __m256 ldf(const void* p) { return _mm256_loadu_ps(reinterpret_cast<const float*>(p)); }

inline void st(void* p, __m256i v) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v); }

inline void stf(void* p, __m256 v) { _mm256_storeu_ps(reinterpret_cast<float*>(p), v); }

inline __m256i k128() { return _mm256_set1_epi8(char(0x80)); }

inline __m256i bcast16(const void* p) {
    return _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

// _mm512_reduce_add_ps of lanes lo (0-7), hi (8-15) in GCC's __MM512_REDUCE_OP order.
inline float reduce_add16(__m256 lo, __m256 hi) {
    const __m256 s8 = _mm256_add_ps(hi, lo);
    const __m128 s4 = _mm_add_ps(_mm256_extractf128_ps(s8, 1), _mm256_castps256_ps128(s8));
    const __m128 s2 = _mm_add_ps(s4, _mm_shuffle_ps(s4, s4, _MM_SHUFFLE(1, 0, 3, 2)));
    const __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(s1);
}

// _mm512_reduce_max_ps, same order and operand order.
inline float reduce_max16(__m256 lo, __m256 hi) {
    const __m256 m8 = _mm256_max_ps(hi, lo);
    const __m128 m4 = _mm_max_ps(_mm256_extractf128_ps(m8, 1), _mm256_castps256_ps128(m8));
    const __m128 m2 = _mm_max_ps(m4, _mm_shuffle_ps(m4, m4, _MM_SHUFFLE(1, 0, 3, 2)));
    const __m128 m1 = _mm_max_ps(m2, _mm_shuffle_ps(m2, m2, _MM_SHUFFLE(0, 1, 0, 1)));
    return _mm_cvtss_f32(m1);
}

// 0xFF where the element is negative (see kSignShuf), for chunk c of the half whose 16 sign bytes
// are broadcast in `field`.
inline __m256i sign_mask(__m256i field, int c) {
    const __m256i bit = _mm256_load_si256(reinterpret_cast<const __m256i*>(kSignBit[c & 1].v));
    const __m256i s   = _mm256_shuffle_epi8(
        field, _mm256_load_si256(reinterpret_cast<const __m256i*>(kSignShuf[c].v)));
    return _mm256_cmpeq_epi8(_mm256_and_si256(s, bit), bit);
}

// Grid magnitudes g (< 128) with sign mask m (0xFF -> negative) -> w + 128.
inline __m256i bias_sign(__m256i g, __m256i m) {
    return _mm256_add_epi8(_mm256_sub_epi8(_mm256_xor_si256(g, m), m), k128());
}

inline __m256 lane_scale(__m256i ints, float d) {
    return _mm256_mul_ps(_mm256_cvtepi32_ps(ints), _mm256_set1_ps(d));
}

inline __m256i odd_of(__m256i s) { // 2s + 1
    return _mm256_add_epi32(_mm256_slli_epi32(s, 1), _mm256_set1_epi32(1));
}

// 4 bytes -> 8 dwords, lane i = byte sel[i].
inline __m256i bytes4_lanes(uint32_t v, __m256i sel) {
    return _mm256_permutevar8x32_epi32(_mm256_cvtepu8_epi32(_mm_cvtsi32_si128(static_cast<int>(v))),
                                       sel);
}

constexpr int kPrefetchBytes = 1024;

inline void prefetch_ahead(const void* p) {
    const char* c = static_cast<const char*>(p) + kPrefetchBytes;
    _mm_prefetch(c, _MM_HINT_T0);
    _mm_prefetch(c + 64, _MM_HINT_T0);
}

// ------------------------------------------------------------------ weight decoders
// Q8_K formats: one 256-element block -> [4 x 64 B w+128 (chunk c, lane j, byte b <- element
// 16j + 4c + b)][64 B F]; chunk c half h = bytes 64c + 32h .. +31 (lanes 8h..8h+7).
void dec_iq2_s(const block_iq2_s& x, uint8_t* dst) {
    const float d = fp16(x.d) * 0.125f;
    // Grid index i (elements 8i..8i+7): qs[i] | ((qh >> 2i) & 3) << 8 (qh as one 64-bit word).
    alignas(32) uint32_t idx[32];
    const uint64_t qh = load_u64(x.qh);
    const __m256i sh2 = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    for (int m = 0; m < 4; ++m) {
        const __m256i lo =
            _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(x.qs + 8 * m)));
        const __m256i hb = _mm256_and_si256(
            _mm256_srlv_epi32(_mm256_set1_epi32(static_cast<int>(uint32_t(qh >> (16 * m)))), sh2),
            _mm256_set1_epi32(3));
        _mm256_store_si256(reinterpret_cast<__m256i*>(idx + 8 * m),
                           _mm256_or_si256(lo, _mm256_slli_epi32(hb, 8)));
    }
    const auto* grid = reinterpret_cast<const long long*>(iq2s_grid);
    const auto q4    = [&](int a, int b, int c, int e) {
        return _mm256_castsi256_ps(
            _mm256_set_epi64x(grid[idx[e]], grid[idx[c]], grid[idx[b]], grid[idx[a]]));
    };
    const uint8_t* signs = x.qs + QK_K / 8;
    for (int h = 0; h < 2; ++h) {
        const int o         = 16 * h;
        const __m256i field = bcast16(signs + 16 * h);
        // Lane j: chunks 0/1 = low/high dword of grid word 2j, chunks 2/3 = of word 2j + 1.
        for (int p = 0; p < 2; ++p) {
            const __m256 X   = q4(o + p, o + 2 + p, o + 8 + p, o + 10 + p);
            const __m256 Y   = q4(o + 4 + p, o + 6 + p, o + 12 + p, o + 14 + p);
            const __m256i g0 = _mm256_castps_si256(_mm256_shuffle_ps(X, Y, 0x88));
            const __m256i g1 = _mm256_castps_si256(_mm256_shuffle_ps(X, Y, 0xDD));
            st(dst + kChunk * (2 * p) + 32 * h, bias_sign(g0, sign_mask(field, 2 * p)));
            st(dst + kChunk * (2 * p + 1) + 32 * h, bias_sign(g1, sign_mask(field, 2 * p + 1)));
        }
        // Lane j: (scales[j / 2] >> 4 (j % 2)) & 15.
        const __m256i s = _mm256_and_si256(
            _mm256_srlv_epi32(
                bytes4_lanes(load_u32(x.scales + 4 * h), _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3)),
                _mm256_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4)),
            _mm256_set1_epi32(15));
        stf(dst + 4 * kChunk + 32 * h, lane_scale(odd_of(s), d));
    }
}

// The 32 IQ3_XXS sign bytes k = 4 ib + l: ksigns_iq2xs[(aux[ib] >> 7l) & 127] = the 7 bits plus
// their parity in bit 7.
inline __m256i iq3xxs_signs(__m256i aux) {
    const __m256i m7 = _mm256_set1_epi32(127);
    __m256i r        = _mm256_and_si256(aux, m7);
    r = _mm256_or_si256(r, _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(aux, 7), m7), 8));
    r = _mm256_or_si256(r, _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(aux, 14), m7), 16));
    r = _mm256_or_si256(r, _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(aux, 21), m7), 24));
    __m256i p = _mm256_xor_si256(r, _mm256_srli_epi32(r, 4));
    p         = _mm256_xor_si256(p, _mm256_srli_epi32(p, 2));
    p         = _mm256_xor_si256(p, _mm256_srli_epi32(p, 1)); // bit 8i = parity of byte i
    return _mm256_or_si256(
        r, _mm256_slli_epi32(_mm256_and_si256(p, _mm256_set1_epi32(0x01010101)), 7));
}

void dec_iq3_xxs(const block_iq3_xxs& x, uint8_t* dst) {
    const float d     = fp16(x.d) * 0.25f;
    const uint8_t* q  = x.qs;
    const __m256i aux = ld(x.qs + QK_K / 4);
    alignas(32) uint8_t signs[32];
    _mm256_store_si256(reinterpret_cast<__m256i*>(signs), iq3xxs_signs(aux));
    const auto* grid = reinterpret_cast<const int*>(iq3xxs_grid);
    for (int h = 0; h < 2; ++h) {
        const __m256i field = bcast16(signs + 16 * h);
        const uint8_t* qh   = q + 32 * h; // lane j = 8h + i reads index 4j + c = 32h + 4i + c
        for (int c = 0; c < 4; ++c) {
            const __m256i g = _mm256_setr_epi32(
                grid[qh[c]], grid[qh[4 + c]], grid[qh[8 + c]], grid[qh[12 + c]], grid[qh[16 + c]],
                grid[qh[20 + c]], grid[qh[24 + c]], grid[qh[28 + c]]);
            st(dst + kChunk * c + 32 * h, bias_sign(g, sign_mask(field, c)));
        }
        // Lane j: aux[j / 2] >> 28.
        const __m256i ls =
            _mm256_srli_epi32(_mm256_permutevar8x32_epi32(
                                  aux, _mm256_add_epi32(_mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3),
                                                        _mm256_set1_epi32(4 * h))),
                              28);
        stf(dst + 4 * kChunk + 32 * h, lane_scale(odd_of(ls), d));
    }
}

void dec_iq3_s(const block_iq3_s& x, uint8_t* dst) {
    const float d = fp16(x.d);
    // Grid index i (elements 4i..4i+3): qs[i] | ((qh >> i) & 1) << 8 (qh as one 64-bit word).
    alignas(32) uint32_t idx[64];
    const uint64_t qh = load_u64(x.qh);
    const __m256i sh  = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    for (int m = 0; m < 8; ++m) {
        const __m256i lo =
            _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(x.qs + 8 * m)));
        const __m256i hb = _mm256_and_si256(
            _mm256_srlv_epi32(_mm256_set1_epi32(static_cast<int>((qh >> (8 * m)) & 0xFF)), sh),
            _mm256_set1_epi32(1));
        _mm256_store_si256(reinterpret_cast<__m256i*>(idx + 8 * m),
                           _mm256_or_si256(lo, _mm256_slli_epi32(hb, 8)));
    }
    const auto* grid = reinterpret_cast<const int*>(iq3s_grid);
    for (int h = 0; h < 2; ++h) {
        const __m256i field = bcast16(x.signs + 16 * h);
        const uint32_t* ih  = idx + 32 * h;
        for (int c = 0; c < 4; ++c) {
            const __m256i g = _mm256_setr_epi32(
                grid[ih[c]], grid[ih[4 + c]], grid[ih[8 + c]], grid[ih[12 + c]], grid[ih[16 + c]],
                grid[ih[20 + c]], grid[ih[24 + c]], grid[ih[28 + c]]);
            st(dst + kChunk * c + 32 * h, bias_sign(g, sign_mask(field, c)));
        }
        // Lane j: (scales[j / 4] >> 4 ((j / 2) % 2)) & 15.
        const __m256i s = _mm256_and_si256(
            _mm256_srlv_epi32(
                bytes4_lanes(load_u32(x.scales),
                             _mm256_add_epi32(_mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1),
                                              _mm256_set1_epi32(2 * h))),
                _mm256_setr_epi32(0, 0, 4, 4, 0, 0, 4, 4)),
            _mm256_set1_epi32(15));
        stf(dst + 4 * kChunk + 32 * h, lane_scale(odd_of(s), d));
    }
}

// ---- big-core decoders (decode_rows_big): the IQ3 grid lookups by vpgatherdd, same bytes as the
// decoders above (P-core decode-only, 15. 11. 2026: IQ3_XXS 151 -> 129, IQ3_S 200 -> 135 us per
// expert; on an E-core ~2x slower; IQ2_S by vpgatherqq was 10-21 % slower and keeps dec_iq2_s).

// Byte C of every dword of v, as dwords (the grid index of lane i when v holds bytes 4i..4i+3).
template <int C>
inline __m256i byte_of_dwords(__m256i v) {
    return _mm256_and_si256(_mm256_srli_epi32(v, 8 * C), _mm256_set1_epi32(0xFF));
}

void dec_iq3_xxs_g(const block_iq3_xxs& x, uint8_t* dst) {
    const float d     = fp16(x.d) * 0.25f;
    const __m256i aux = ld(x.qs + QK_K / 4);
    alignas(32) uint8_t signs[32];
    _mm256_store_si256(reinterpret_cast<__m256i*>(signs), iq3xxs_signs(aux));
    const auto* grid = reinterpret_cast<const int*>(iq3xxs_grid);
    for (int h = 0; h < 2; ++h) {
        const __m256i field = bcast16(signs + 16 * h);
        // Lane i of chunk c reads index 32h + 4i + c: byte c of dword i of the half's indices.
        const __m256i qv    = ld(x.qs + 32 * h);
        const __m256i ix[4] = {byte_of_dwords<0>(qv), byte_of_dwords<1>(qv), byte_of_dwords<2>(qv),
                               byte_of_dwords<3>(qv)};
        for (int c = 0; c < 4; ++c) {
            const __m256i g = _mm256_i32gather_epi32(grid, ix[c], 4);
            st(dst + kChunk * c + 32 * h, bias_sign(g, sign_mask(field, c)));
        }
        const __m256i ls =
            _mm256_srli_epi32(_mm256_permutevar8x32_epi32(
                                  aux, _mm256_add_epi32(_mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3),
                                                        _mm256_set1_epi32(4 * h))),
                              28);
        stf(dst + 4 * kChunk + 32 * h, lane_scale(odd_of(ls), d));
    }
}

void dec_iq3_s_g(const block_iq3_s& x, uint8_t* dst) {
    const float d     = fp16(x.d);
    const uint64_t qh = load_u64(x.qh);
    const auto* grid  = reinterpret_cast<const int*>(iq3s_grid);
    const __m256i sh4 = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    const __m256i one = _mm256_set1_epi32(1);
    for (int h = 0; h < 2; ++h) {
        const __m256i field = bcast16(x.signs + 16 * h);
        // Lane i of chunk c reads index 32h + 4i + c: byte c of dword i of the half's qs, bit
        // 4i + c of the half's 32 qh bits.
        const __m256i qv = ld(x.qs + 32 * h);
        const __m256i hv =
            _mm256_set1_epi32(static_cast<int>(static_cast<uint32_t>(qh >> (32 * h))));
        const __m256i lo[4] = {byte_of_dwords<0>(qv), byte_of_dwords<1>(qv), byte_of_dwords<2>(qv),
                               byte_of_dwords<3>(qv)};
        for (int c = 0; c < 4; ++c) {
            const __m256i hb = _mm256_and_si256(
                _mm256_srlv_epi32(hv, _mm256_add_epi32(sh4, _mm256_set1_epi32(c))), one);
            const __m256i g =
                _mm256_i32gather_epi32(grid, _mm256_or_si256(lo[c], _mm256_slli_epi32(hb, 8)), 4);
            st(dst + kChunk * c + 32 * h, bias_sign(g, sign_mask(field, c)));
        }
        const __m256i s = _mm256_and_si256(
            _mm256_srlv_epi32(
                bytes4_lanes(load_u32(x.scales),
                             _mm256_add_epi32(_mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1),
                                              _mm256_set1_epi32(2 * h))),
                _mm256_setr_epi32(0, 0, 4, 4, 0, 0, 4, 4)),
            _mm256_set1_epi32(15));
        stf(dst + 4 * kChunk + 32 * h, lane_scale(odd_of(s), d));
    }
}

void dec_iq4_xs(const block_iq4_xs& x, uint8_t* dst) {
    const float d     = fp16(x.d);
    const __m256i tbl = _mm256_load_si256(reinterpret_cast<const __m256i*>(kIq4Biased.v));
    const __m256i m4  = _mm256_set1_epi8(0x0f);
    for (int h = 0; h < 2; ++h) {
        // Lane j = 8h + 4L + jj: rows (16 B of qs, one 32-element sub-block each) R0 = 4h + 2L (jj
        // 0, 1: low, high nibbles) and R0 + 1 (jj 2, 3); chunk c = bytes 4c..4c+3 of those rows.
        const __m256i X = ld(x.qs + 64 * h), Y = ld(x.qs + 64 * h + 32);
        const __m256i A  = _mm256_permute2x128_si256(X, Y, 0x20),
                      B  = _mm256_permute2x128_si256(X, Y, 0x31);
        const __m256i Al = _mm256_shuffle_epi8(tbl, _mm256_and_si256(A, m4));
        const __m256i Ah = _mm256_shuffle_epi8(tbl, _mm256_and_si256(_mm256_srli_epi16(A, 4), m4));
        const __m256i Bl = _mm256_shuffle_epi8(tbl, _mm256_and_si256(B, m4));
        const __m256i Bh = _mm256_shuffle_epi8(tbl, _mm256_and_si256(_mm256_srli_epi16(B, 4), m4));
        const __m256i t0 = _mm256_unpacklo_epi32(Al, Ah), t1 = _mm256_unpackhi_epi32(Al, Ah);
        const __m256i u0 = _mm256_unpacklo_epi32(Bl, Bh), u1 = _mm256_unpackhi_epi32(Bl, Bh);
        st(dst + 32 * h, _mm256_unpacklo_epi64(t0, u0));
        st(dst + kChunk + 32 * h, _mm256_unpackhi_epi64(t0, u0));
        st(dst + 2 * kChunk + 32 * h, _mm256_unpacklo_epi64(t1, u1));
        st(dst + 3 * kChunk + 32 * h, _mm256_unpackhi_epi64(t1, u1));
        // Lane j: ((scales_l[j / 4] >> 4 ((j / 2) % 2)) & 15 | ((scales_h >> 2 (j / 2)) & 3) << 4)
        // - 32.
        const __m256i sl = _mm256_and_si256(
            _mm256_srlv_epi32(
                bytes4_lanes(load_u32(x.scales_l),
                             _mm256_add_epi32(_mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1),
                                              _mm256_set1_epi32(2 * h))),
                _mm256_setr_epi32(0, 0, 4, 4, 0, 0, 4, 4)),
            _mm256_set1_epi32(15));
        const __m256i shv = _mm256_and_si256(
            _mm256_srlv_epi32(_mm256_set1_epi32(load_u16(&x.scales_h)),
                              _mm256_add_epi32(_mm256_setr_epi32(0, 0, 2, 2, 4, 4, 6, 6),
                                               _mm256_set1_epi32(8 * h))),
            _mm256_set1_epi32(3));
        const __m256i ls =
            _mm256_sub_epi32(_mm256_or_si256(sl, _mm256_slli_epi32(shv, 4)), _mm256_set1_epi32(32));
        stf(dst + 4 * kChunk + 32 * h, lane_scale(ls, d));
    }
}

// Q8_0 formats: one 64-element group -> [64 B w+128 (natural order)][64 B F].
// 32 IQ4_NL elements (low nibbles, then high nibbles of the 16 code bytes) -> w + 128.
inline __m256i iq4nl32(const block_iq4_nl* x, __m256i tbl) {
    const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x->qs));
    const __m256i n =
        _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(q, 4), q), _mm256_set1_epi8(0x0f));
    return _mm256_shuffle_epi8(tbl, n);
}

void dec_iq4_nl(const block_iq4_nl* xa, const block_iq4_nl* xb, uint8_t* dst) {
    const __m256i tbl = _mm256_load_si256(reinterpret_cast<const __m256i*>(kIq4Biased.v));
    st(dst, iq4nl32(xa, tbl));
    st(dst + 32, xb ? iq4nl32(xb, tbl) : k128()); // padded half: w = 0
    stf(dst + kChunk, _mm256_set1_ps(fp16(xa->d)));
    stf(dst + kChunk + 32, _mm256_set1_ps(xb ? fp16(xb->d) : 0.0f));
}

void dec_q2_0(const block_q2_0& x, uint8_t* dst) {
    // Element i = (qs[i / 4] >> 2 (i % 4)) & 3: dword k of a half holds the 4 elements of code byte
    // 8h + k, q | q << 6 | q << 12 | q << 18 puts bits 2b..2b+1 of q at the bottom of byte b.
    for (int h = 0; h < 2; ++h) {
        const __m256i q =
            _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(x.qs + 8 * h)));
        __m256i t =
            _mm256_or_si256(_mm256_or_si256(q, _mm256_slli_epi32(q, 6)),
                            _mm256_or_si256(_mm256_slli_epi32(q, 12), _mm256_slli_epi32(q, 18)));
        t = _mm256_and_si256(t, _mm256_set1_epi32(0x03030303));
        st(dst + 32 * h, _mm256_add_epi8(t, _mm256_set1_epi8(127))); // (q - 1) + 128
    }
    const __m256 f = _mm256_set1_ps(fp16(x.d));
    stf(dst + kChunk, f);
    stf(dst + kChunk + 32, f);
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
// -128 * (sum of the 4 int8 of each lane of v), as 0 - vpdpbusd(0, 0x80, v).
inline __m256i lane_sums4(__m256i v) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi8(1), v), _mm256_set1_epi16(1));
}

inline __m256i acorr(__m256i sums) {
    return _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_slli_epi32(sums, 7));
}

// 256 int8 codes in natural order -> the four permuted chunks (chunk c, lane j <- dword 4j + c)
// and ACORR of a Q8_K-format group.
void pack_q8k_codes(const int8_t* qs, uint8_t* dst) {
    const __m256i pair = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int h = 0; h < 2; ++h) {
        // Y'_m qword c = (dword c, dword 4 + c) of the m-th 32 bytes of this half.
        __m256i y[4];
        for (int m = 0; m < 4; ++m)
            y[m] = _mm256_permutevar8x32_epi32(ld(qs + 128 * h + 32 * m), pair);
        const __m256i lo01 = _mm256_unpacklo_epi64(y[0], y[1]),
                      hi01 = _mm256_unpackhi_epi64(y[0], y[1]);
        const __m256i lo23 = _mm256_unpacklo_epi64(y[2], y[3]),
                      hi23 = _mm256_unpackhi_epi64(y[2], y[3]);
        const __m256i c[4] = {_mm256_permute2x128_si256(lo01, lo23, 0x20),
                              _mm256_permute2x128_si256(hi01, hi23, 0x20),
                              _mm256_permute2x128_si256(lo01, lo23, 0x31),
                              _mm256_permute2x128_si256(hi01, hi23, 0x31)};
        __m256i sum        = _mm256_setzero_si256();
        for (int k = 0; k < 4; ++k) {
            st(dst + kChunk * k + 32 * h, c[k]);
            sum = _mm256_add_epi32(sum, lane_sums4(c[k]));
        }
        st(dst + 4 * kChunk + 64 + 32 * h, acorr(sum));
    }
}

void pack_group_q8K(const block_q8_K& x, uint8_t* dst) {
    pack_q8k_codes(x.qs, dst);
    stf(dst + 4 * kChunk, _mm256_set1_ps(x.d));
    stf(dst + 4 * kChunk + 32, _mm256_set1_ps(x.d));
}

// Q8_K-format group whose lane j takes the scale da8[j / 2] of 32-value block j / 2.
inline void store_da32(const float* da8, uint8_t* dst) {
    stf(dst, _mm256_setr_ps(da8[0], da8[0], da8[1], da8[1], da8[2], da8[2], da8[3], da8[3]));
    stf(dst + 32, _mm256_setr_ps(da8[4], da8[4], da8[5], da8[5], da8[6], da8[6], da8[7], da8[7]));
}

void pack_group_q8K_g32(const block_q8_0* x, uint8_t* dst) {
    alignas(32) int8_t qs[QK_K];
    for (int b = 0; b < 8; ++b) std::memcpy(qs + 32 * b, x[b].qs, 32);
    pack_q8k_codes(qs, dst);
    float da[8];
    for (int b = 0; b < 8; ++b) da[b] = fp16(x[b].d);
    store_da32(da, dst + 4 * kChunk);
}

inline __m256 abs_ps(__m256 v) {
    return _mm256_and_ps(v, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
}

inline __m256i round_away(__m256 x) {
    __m256 t        = _mm256_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    const __m256 up = _mm256_cmp_ps(abs_ps(_mm256_sub_ps(x, t)), _mm256_set1_ps(0.5f), _CMP_GE_OQ);
    const __m256 one =
        _mm256_or_ps(_mm256_set1_ps(1.0f),
                     _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(int(0x80000000u)))));
    return _mm256_cvttps_epi32(_mm256_blendv_ps(t, _mm256_add_ps(t, one), up));
}

// Low byte of every int32 of a (lanes 0-7) then b (8-15): vpmovdb of the 16 lanes.
inline __m128i narrow16(__m256i a, __m256i b) {
    const __m256i m     = _mm256_set1_epi32(0xFF);
    const __m256i w     = _mm256_packus_epi32(_mm256_and_si256(a, m), _mm256_and_si256(b, m));
    const __m256i bytes = _mm256_packus_epi16(w, w);
    return _mm256_castsi256_si128(
        _mm256_permutevar8x32_epi32(bytes, _mm256_setr_epi32(0, 4, 1, 5, 0, 0, 0, 0)));
}

// 32 values -> 32 codes; returns d (the GPU GGUF Ops' codec, as host_gguf_avx512.cpp).
inline float q8f32_block(const float* v, int8_t* q) {
    const __m256 a0 = ldf(v), a1 = ldf(v + 8), b0 = ldf(v + 16), b1 = ldf(v + 24);
    const float amax =
        reduce_max16(_mm256_max_ps(abs_ps(a0), abs_ps(b0)), _mm256_max_ps(abs_ps(a1), abs_ps(b1)));
    if (amax == 0.0f) {
        std::memset(q, 0, 32);
        return 0.0f;
    }
    const float d   = amax / 127.0f;
    const __m256 dv = _mm256_set1_ps(d);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(q), narrow16(round_away(_mm256_div_ps(a0, dv)),
                                                             round_away(_mm256_div_ps(a1, dv))));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(q + 16),
        narrow16(round_away(_mm256_div_ps(b0, dv)), round_away(_mm256_div_ps(b1, dv))));
    return d;
}

// The 64-byte q of a Q8_0-format group: q, DA halves, ACORR.
inline void store_q8_0_group(const int8_t* q64, float d0, float d1, uint8_t* dst) {
    const __m256i lo = ld(q64), hi = ld(q64 + 32);
    st(dst, lo);
    st(dst + 32, hi);
    stf(dst + kChunk, _mm256_set1_ps(d0));
    stf(dst + kChunk + 32, _mm256_set1_ps(d1));
    st(dst + kChunk + 64, acorr(lane_sums4(lo)));
    st(dst + kChunk + 96, acorr(lane_sums4(hi)));
}

template <class In>
void quantize_pack_q8f32_impl(ActFormat act, const In* x, int32_t k, uint8_t* d) {
    alignas(32) float v[QK_K];
    alignas(32) int8_t qs[QK_K];
    float dd[8];
    const auto widen = [&](const In* src, int n) {
        if constexpr (std::is_same_v<In, float>) {
            std::memcpy(v, src, sizeof(float) * n);
        } else {
            for (int i = 0; i < n; i += 8) {
                const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
                stf(v + i, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16)));
            }
        }
    };
    if (act == ActFormat::Q8_K) {
        for (int b = 0; b < k / QK_K; ++b) {
            widen(x + std::size_t(b) * QK_K, QK_K);
            for (int s = 0; s < 8; ++s) dd[s] = q8f32_block(v + 32 * s, qs + 32 * s);
            uint8_t* dst = d + std::size_t(b) * agroup_bytes(4);
            pack_q8k_codes(qs, dst);
            store_da32(dd, dst + 4 * kChunk);
        }
        return;
    }
    const int groups = (k + 63) / 64;
    for (int g = 0; g < groups; ++g) {
        const int n = std::min(64, k - 64 * g);
        widen(x + std::size_t(g) * 64, n);
        alignas(32) int8_t q64[64] = {};
        dd[0]                      = q8f32_block(v, q64);
        dd[1]                      = n == 64 ? q8f32_block(v + 32, q64 + 32) : 0.0f;
        store_q8_0_group(q64, dd[0], dd[1], d + std::size_t(g) * agroup_bytes(1));
    }
}

void pack_group_q8_0(const block_q8_0* a, const block_q8_0* b, uint8_t* dst) {
    alignas(32) int8_t q64[64] = {};
    std::memcpy(q64, a->qs, 32);
    if (b) std::memcpy(q64 + 32, b->qs, 32);
    store_q8_0_group(q64, fp16(a->d), b ? fp16(b->d) : 0.0f, dst);
}

// Half `half` of packed Q8_0 group `group`: 32 codes, their DA lanes and ACORR lanes.
inline void store_q8_0_half(const int8_t* q32, float d, int32_t group, int32_t half, void* dst) {
    auto* g         = static_cast<uint8_t*>(dst) + std::size_t(group) * agroup_bytes(1);
    const __m256i v = ld(q32);
    st(g + 32 * half, v);
    stf(g + kChunk + 32 * half, _mm256_set1_ps(d));
    st(g + kChunk + 64 + 32 * half, acorr(lane_sums4(v)));
}

// ------------------------------------------------------------------ micro-kernel
// R decoded rows (stride ws) x N packed activation rows over G groups of C chunks. Lanes 0-7 over
// all groups first, then lanes 8-15 (two independent passes of the 512-bit kernel's lane set).
template <int C, int R, int N>
void micro(const uint8_t* w, std::size_t ws, const uint8_t* const* a, int G, float* out,
           std::size_t ldo) {
    constexpr int WG = wgroup_bytes(C), AG = agroup_bytes(C);
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i k80  = k128();
    __m256 lo[R][N];
    for (int h = 0; h < 2; ++h) {
        __m256 acc[R][N];
#pragma GCC unroll 16
        for (int r = 0; r < R; ++r)
#pragma GCC unroll 16
            for (int t = 0; t < N; ++t) acc[r][t] = _mm256_setzero_ps();
        for (int g = 0; g < G; ++g) {
            const std::size_t wo = std::size_t(g) * WG + 32 * h, ao = std::size_t(g) * AG + 32 * h;
            __m256i is[R][N];
#pragma GCC unroll 16
            for (int r = 0; r < R; ++r)
#pragma GCC unroll 16
                for (int t = 0; t < N; ++t) is[r][t] = _mm256_setzero_si256();
#pragma GCC unroll 4
            for (int c = 0; c < C; ++c) {
                __m256i q[N], aq[N];
#pragma GCC unroll 16
                for (int t = 0; t < N; ++t) {
                    q[t]  = ld(a[t] + ao + c * kChunk);
                    aq[t] = _mm256_sign_epi8(q[t], q[t]); // |q| as u8 (128 for -128)
                }
#pragma GCC unroll 16
                for (int r = 0; r < R; ++r) {
                    const __m256i wv =
                        _mm256_xor_si256(ld(w + r * ws + wo + c * kChunk), k80); // w as int8
#pragma GCC unroll 16
                    for (int t = 0; t < N; ++t)
                        is[r][t] = _mm256_add_epi32(
                            is[r][t],
                            _mm256_madd_epi16(
                                _mm256_maddubs_epi16(aq[t], _mm256_sign_epi8(wv, q[t])), ones));
                }
            }
#pragma GCC unroll 16
            for (int r = 0; r < R; ++r) {
                const __m256 f = ldf(w + r * ws + wo + C * kChunk);
#pragma GCC unroll 16
                for (int t = 0; t < N; ++t) {
                    const __m256 s = _mm256_mul_ps(f, ldf(a[t] + ao + C * kChunk));
                    acc[r][t]      = _mm256_fmadd_ps(_mm256_cvtepi32_ps(is[r][t]), s, acc[r][t]);
                }
            }
        }
#pragma GCC unroll 16
        for (int r = 0; r < R; ++r)
#pragma GCC unroll 16
            for (int t = 0; t < N; ++t) {
                if (h == 0)
                    lo[r][t] = acc[r][t];
                else
                    out[r * ldo + t] = reduce_add16(lo[r][t], acc[r][t]);
            }
    }
}

using MicroFn = void (*)(const uint8_t*, std::size_t, const uint8_t* const*, int, float*,
                         std::size_t);

// Tiles: one token x up to 4 rows, or up to 2 tokens x up to 2 rows (16 ymm registers).
template <int C>
constexpr MicroFn kMicro[5][3] = {
    {nullptr, nullptr, nullptr},
    {nullptr, &micro<C, 1, 1>, &micro<C, 1, 2>},
    {nullptr, &micro<C, 2, 1>, &micro<C, 2, 2>},
    {nullptr, &micro<C, 3, 1>, nullptr},
    {nullptr, &micro<C, 4, 1>, nullptr},
};

template <int C>
void gemm_impl(const uint8_t* w, std::size_t ws, int nrows, const uint8_t* const* a, int T, int G,
               float* out, std::size_t ldo) {
    for (int t0 = 0; t0 < T;) {
        const int nt   = T - t0 >= 2 ? 2 : 1;
        const int rows = nt == 1 ? 4 : 2;
        for (int r0 = 0; r0 < nrows; r0 += rows) {
            const int nr = std::min(rows, nrows - r0);
            kMicro<C>[nr][nt](w + std::size_t(r0) * ws, ws, a + t0, G,
                              out + std::size_t(r0) * ldo + t0, ldo);
        }
        t0 += nt;
    }
}

// ------------------------------------------------------------------ silu
// y * 2^n for integral n in [-149, 128] and y in [0.5, 2): vscalefps's single rounding.
inline __m256 scalef(__m256 y, __m256 n) {
    const __m256i ni = _mm256_cvtps_epi32(n);
    const __m256i n1 = _mm256_srai_epi32(ni, 1);
    const __m256i n2 = _mm256_sub_epi32(ni, n1);
    const __m256 p1 =
        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n1, _mm256_set1_epi32(127)), 23));
    const __m256 p2 =
        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n2, _mm256_set1_epi32(127)), 23));
    return _mm256_mul_ps(_mm256_mul_ps(y, p1), p2);
}

inline __m256 exp8(__m256 x) {
    // Cephes expf: exp(x) = 2^n * p(r), n = round(x / ln2), r = x - n ln2.
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-103.0f)), _mm256_set1_ps(88.5f));
    const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r       = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
    r              = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), r);
    __m256 y       = _mm256_set1_ps(1.9875691500e-4f);
    y              = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.3981999507e-3f));
    y              = _mm256_fmadd_ps(y, r, _mm256_set1_ps(8.3334519073e-3f));
    y              = _mm256_fmadd_ps(y, r, _mm256_set1_ps(4.1665795894e-2f));
    y              = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.6666665459e-1f));
    y              = _mm256_fmadd_ps(y, r, _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_fmadd_ps(y, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.0f)));
    return scalef(y, n);
}

inline __m256 silu_mul8(__m256 gv, __m256 uv) {
    const __m256 e = exp8(_mm256_sub_ps(_mm256_setzero_ps(), gv));
    const __m256 s = _mm256_div_ps(gv, _mm256_add_ps(_mm256_set1_ps(1.0f), e));
    return _mm256_mul_ps(s, uv);
}

// ------------------------------------------------------------------ prefill (BF16)
// vcvtne2ps2bf16 of one value set: RNE, zero/subnormal -> signed zero, NaN -> quiet.
inline __m256i bf16_bits(__m256 x) {
    const __m256i u   = _mm256_castps_si256(x);
    const __m256i rne = _mm256_srli_epi32(
        _mm256_add_epi32(_mm256_add_epi32(u, _mm256_set1_epi32(0x7FFF)),
                         _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(1))),
        16);
    const __m256i mag     = _mm256_and_si256(u, _mm256_set1_epi32(0x7FFFFFFF));
    const __m256i sign16  = _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(0x8000));
    const __m256i nan16   = _mm256_or_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(0x40));
    const __m256i is_tiny = _mm256_cmpeq_epi32(_mm256_and_si256(u, _mm256_set1_epi32(0x7F800000)),
                                               _mm256_setzero_si256());
    const __m256i is_nan  = _mm256_cmpgt_epi32(mag, _mm256_set1_epi32(0x7F800000));
    __m256i r             = _mm256_blendv_epi8(rne, sign16, is_tiny);
    return _mm256_and_si256(_mm256_blendv_epi8(r, nan16, is_nan), _mm256_set1_epi32(0xFFFF));
}

// 16 int8 weights (natural order) times scale s -> 16 BF16 at dst.
inline void store_bf16_16(__m128i q, float s, uint16_t* dst) {
    const __m256 sv = _mm256_set1_ps(s);
    const __m256 a  = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q)), sv);
    const __m256 b =
        _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q, 8))), sv);
    const __m256i p =
        _mm256_permute4x64_epi64(_mm256_packus_epi32(bf16_bits(a), bf16_bits(b)), 0xD8);
    st(dst, p);
}

template <int R, int T>
void bf16_micro(const uint16_t* weights, int k, const void* const* acts, float* out,
                std::size_t ldo) {
    const __m256i hi16 = _mm256_set1_epi32(int(0xFFFF0000u));
    __m256 lo[R][T];
    for (int h = 0; h < 2; ++h) {
        __m256 acc[R][T];
        for (int r = 0; r < R; ++r)
            for (int t = 0; t < T; ++t) acc[r][t] = _mm256_setzero_ps();
        const unsigned csr = _mm_getcsr();
        _mm_setcsr((csr & ~0x6000u) | 0x8040u | 0x1F80u); // FTZ, DAZ, round to nearest, masked
        for (int i = 16 * h; i < k; i += 32) {
            __m256 xo[T], xe[T];
            for (int t = 0; t < T; ++t) {
                const __m256i x = ld(static_cast<const uint16_t*>(acts[t]) + i);
                xo[t]           = _mm256_castsi256_ps(_mm256_and_si256(x, hi16));
                xe[t]           = _mm256_castsi256_ps(_mm256_slli_epi32(x, 16));
            }
            for (int r = 0; r < R; ++r) {
                const __m256i wv = ld(weights + std::size_t(r) * k + i);
                const __m256 wo  = _mm256_castsi256_ps(_mm256_and_si256(wv, hi16));
                const __m256 we  = _mm256_castsi256_ps(_mm256_slli_epi32(wv, 16));
                for (int t = 0; t < T; ++t) {
                    acc[r][t] = _mm256_fmadd_ps(wo, xo[t], acc[r][t]);
                    acc[r][t] = _mm256_fmadd_ps(we, xe[t], acc[r][t]);
                }
            }
        }
        _mm_setcsr(csr);
        for (int r = 0; r < R; ++r)
            for (int t = 0; t < T; ++t) {
                if (h == 0)
                    lo[r][t] = acc[r][t];
                else
                    out[std::size_t(r) * ldo + t] = reduce_add16(lo[r][t], acc[r][t]);
            }
    }
}

template <int R>
void bf16_rows(const uint16_t* weights, int k, const void* const* acts, int T, float* out,
               std::size_t ldo) {
    int t = 0;
    for (; t + 2 <= T; t += 2) bf16_micro<R, 2>(weights, k, acts + t, out + t, ldo);
    if (t < T) bf16_micro<R, 1>(weights, k, acts + t, out + t, ldo);
}

} // namespace

// ------------------------------------------------------------------ entry points
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
    alignas(32) int8_t q[32];
    const float d = q8f32_block(x32, q);
    store_q8_0_half(q, d, group, half, dst);
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
    store_q8_0_half(q.qs, fp16(q.d), group, half, dst);
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

void decode_rows_big(WeightFormat fmt, const void* rows, std::size_t row_bytes, int32_t nrows,
                     int32_t k, void* dst_v) {
    check_k(fmt, k);
    auto* dst      = static_cast<uint8_t*>(dst_v);
    const int nblk = k / format_info(fmt).block_elems;
    switch (fmt) {
    case WeightFormat::IQ3_XXS:
        decode_rows_q8k<block_iq3_xxs, dec_iq3_xxs_g>(rows, row_bytes, nrows, nblk, dst);
        return;
    case WeightFormat::IQ3_S:
        decode_rows_q8k<block_iq3_s, dec_iq3_s_g>(rows, row_bytes, nrows, nblk, dst);
        return;
    default:
        ::ninfer::ops::host::gguf::detail::avx2::decode_rows(fmt, rows, row_bytes, nrows, k, dst_v);
        return;
    }
}

// BF16 = RNE(float(w) * F) of the decoded layout, in natural element order (as the AVX-512 form).
void decode_rows_bf16(WeightFormat fmt, const void* rows, std::size_t row_bytes, int32_t nrows,
                      int32_t k, uint16_t* dst) {
    check_k(fmt, k);
    const FormatInfo fi         = format_info(fmt);
    const int C                 = chunks_of(fi.act);
    const int elems             = group_elems_of(fi.act);
    const int WG                = wgroup_bytes(C);
    const int G                 = groups_of(fi.act, k);
    constexpr int kScratchBytes = 16384;
    const int seg_groups        = kScratchBytes / WG;
    alignas(64) uint8_t scratch[kScratchBytes];
    for (int r = 0; r < nrows; ++r) {
        const auto* row = static_cast<const uint8_t*>(rows) + std::size_t(r) * row_bytes;
        uint16_t* out   = dst + std::size_t(r) * k;
        for (int g0 = 0; g0 < G; g0 += seg_groups) {
            const int g1    = std::min(G, g0 + seg_groups);
            const int base  = g0 * elems;
            const int seg_k = std::min(g1 * elems, k) - base;
            avx2::decode_rows(fmt, row + std::size_t(base / fi.block_elems) * fi.block_bytes,
                              row_bytes, 1, seg_k, scratch);
            for (int g = g0; g < g1; ++g) {
                const uint8_t* grp = scratch + std::size_t(g - g0) * WG;
                const auto* F      = reinterpret_cast<const float*>(grp + C * kChunk);
                uint16_t* o        = out + std::size_t(g) * elems;
                if (C == 4) {
                    // Lane j = elements 16j..16j+15: dword j of chunks 0..3.
                    for (int j = 0; j < 16; ++j) {
                        alignas(16) uint8_t nat[16];
                        for (int c = 0; c < 4; ++c)
                            std::memcpy(nat + 4 * c, grp + kChunk * c + 4 * j, 4);
                        const __m128i q =
                            _mm_xor_si128(_mm_load_si128(reinterpret_cast<const __m128i*>(nat)),
                                          _mm_set1_epi8(char(0x80)));
                        store_bf16_16(q, F[j], o + 16 * j);
                    }
                } else {
                    // Lanes 0-7 carry the first 32-element block's scale, lanes 8-15 the second's.
                    const int live = std::min(elems, k - g * elems);
                    for (int p = 0; p < live / 16; ++p) {
                        const __m128i q = _mm_xor_si128(
                            _mm_loadu_si128(reinterpret_cast<const __m128i*>(grp + 16 * p)),
                            _mm_set1_epi8(char(0x80)));
                        store_bf16_16(q, F[4 * p], o + 16 * p);
                    }
                }
            }
        }
    }
}

void gemm_bf16(const uint16_t* weights, int32_t nrows, int32_t k, const void* const* acts,
               int32_t T, float* out, std::size_t ldo) {
    int r = 0;
    for (; r + 2 <= nrows; r += 2)
        bf16_rows<2>(weights + std::size_t(r) * k, k, acts, T, out + std::size_t(r) * ldo, ldo);
    for (; r < nrows; ++r)
        bf16_rows<1>(weights + std::size_t(r) * k, k, acts, T, out + std::size_t(r) * ldo, ldo);
}

void silu_mul(const float* g, const float* u, float* h, int32_t n) {
    int32_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(h + i, silu_mul8(_mm256_loadu_ps(g + i), _mm256_loadu_ps(u + i)));
    if (i < n) {
        const __m256i m =
            _mm256_cmpgt_epi32(_mm256_set1_epi32(n - i), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        _mm256_maskstore_ps(h + i, m,
                            silu_mul8(_mm256_maskload_ps(g + i, m), _mm256_maskload_ps(u + i, m)));
    }
}

} // namespace ninfer::ops::host::gguf::detail::avx2

#pragma GCC pop_options

// gemm_decoded with AVX-VNNI (vpdpbusd on 256-bit registers), for CPUs that have it
// (host_gguf_avxvnni_supported): the lane sums start at the activation's ACORR and add
// vpdpbusd(w + 128, q) per chunk, the AVX-512 kernel's integer sequence. They are the exact sums
// sum(w * q) the vpmaddubsw form computes, and the float steps are the same: the same bits.
#pragma GCC push_options
#pragma GCC target("avx2,fma,f16c,avxvnni")

namespace ninfer::ops::host::gguf::detail::avx2 {
namespace {

template <int C, int R, int N>
void micro_vnni(const uint8_t* w, std::size_t ws, const uint8_t* const* a, int G, float* out,
                std::size_t ldo) {
    constexpr int WG = wgroup_bytes(C), AG = agroup_bytes(C);
    __m256 lo[R][N];
    for (int h = 0; h < 2; ++h) {
        __m256 acc[R][N];
#pragma GCC unroll 16
        for (int r = 0; r < R; ++r)
#pragma GCC unroll 16
            for (int t = 0; t < N; ++t) acc[r][t] = _mm256_setzero_ps();
        for (int g = 0; g < G; ++g) {
            const std::size_t wo = std::size_t(g) * WG + 32 * h, ao = std::size_t(g) * AG + 32 * h;
            __m256i is[R][N];
#pragma GCC unroll 16
            for (int t = 0; t < N; ++t) {
                const __m256i corr = ld(a[t] + ao + C * kChunk + 64);
#pragma GCC unroll 16
                for (int r = 0; r < R; ++r) is[r][t] = corr;
            }
#pragma GCC unroll 4
            for (int c = 0; c < C; ++c) {
                __m256i q[N];
#pragma GCC unroll 16
                for (int t = 0; t < N; ++t) q[t] = ld(a[t] + ao + c * kChunk);
#pragma GCC unroll 16
                for (int r = 0; r < R; ++r) {
                    const __m256i wv = ld(w + r * ws + wo + c * kChunk); // w + 128 as u8
#pragma GCC unroll 16
                    for (int t = 0; t < N; ++t)
                        is[r][t] = _mm256_dpbusd_avx_epi32(is[r][t], wv, q[t]);
                }
            }
#pragma GCC unroll 16
            for (int r = 0; r < R; ++r) {
                const __m256 f = ldf(w + r * ws + wo + C * kChunk);
#pragma GCC unroll 16
                for (int t = 0; t < N; ++t) {
                    const __m256 s = _mm256_mul_ps(f, ldf(a[t] + ao + C * kChunk));
                    acc[r][t]      = _mm256_fmadd_ps(_mm256_cvtepi32_ps(is[r][t]), s, acc[r][t]);
                }
            }
        }
#pragma GCC unroll 16
        for (int r = 0; r < R; ++r)
#pragma GCC unroll 16
            for (int t = 0; t < N; ++t) {
                if (h == 0)
                    lo[r][t] = acc[r][t];
                else
                    out[r * ldo + t] = reduce_add16(lo[r][t], acc[r][t]);
            }
    }
}

// The tiles of kMicro.
template <int C>
constexpr MicroFn kMicroVnni[5][3] = {
    {nullptr, nullptr, nullptr},
    {nullptr, &micro_vnni<C, 1, 1>, &micro_vnni<C, 1, 2>},
    {nullptr, &micro_vnni<C, 2, 1>, &micro_vnni<C, 2, 2>},
    {nullptr, &micro_vnni<C, 3, 1>, nullptr},
    {nullptr, &micro_vnni<C, 4, 1>, nullptr},
};

template <int C>
void gemm_impl_vnni(const uint8_t* w, std::size_t ws, int nrows, const uint8_t* const* a, int T,
                    int G, float* out, std::size_t ldo) {
    for (int t0 = 0; t0 < T;) {
        const int nt   = T - t0 >= 2 ? 2 : 1;
        const int rows = nt == 1 ? 4 : 2;
        for (int r0 = 0; r0 < nrows; r0 += rows) {
            const int nr = std::min(rows, nrows - r0);
            kMicroVnni<C>[nr][nt](w + std::size_t(r0) * ws, ws, a + t0, G,
                                  out + std::size_t(r0) * ldo + t0, ldo);
        }
        t0 += nt;
    }
}

} // namespace

void gemm_decoded_vnni(WeightFormat fmt, const void* wdec, int32_t nrows, int32_t k,
                       const void* const* acts, int32_t T, float* out, std::size_t ldo) {
    if (nrows <= 0 || T <= 0) return;
    const ActFormat a    = format_info(fmt).act;
    const int G          = groups_of(a, k);
    const std::size_t ws = decoded_row_bytes(fmt, k);
    const auto* w        = static_cast<const uint8_t*>(wdec);
    const auto* ap       = reinterpret_cast<const uint8_t* const*>(acts);
    if (a == ActFormat::Q8_K)
        gemm_impl_vnni<4>(w, ws, nrows, ap, T, G, out, ldo);
    else
        gemm_impl_vnni<1>(w, ws, nrows, ap, T, G, out, ldo);
}

} // namespace ninfer::ops::host::gguf::detail::avx2

#pragma GCC pop_options
