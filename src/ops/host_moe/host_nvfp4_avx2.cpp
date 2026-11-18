// NVFP4 W4A8 host dots and activation codecs in AVX2 + FMA (HostIsa::Avx2): the kernels of
// host_nvfp4.cpp (the multi-token dot, the int8 and A4 codecs) for CPUs without AVX-512 (Zen 2/3,
// Intel client cores). Same results bit for bit:
//  - integer block sums: vpmaddubsw + vpmaddwd instead of vpdpbusd; exact, the int16 pair sums
//    cannot saturate (|w| <= 12 or w + 12 <= 24 against |q| <= 127, q + 128 <= 255: at most
//    255 * 12 * 2 = 6120);
//  - every 512-bit FP32 lane set of the AVX-512 kernels is kept as two 256-bit halves (lanes 0-7,
//    8-15) with the same per-lane operations (one product per 64-weight group, one FMA into the
//    accumulator), and reduce_add16 sums them in _mm512_reduce_add_ps's order;
//  - codecs: the same products and roundings per element; vpmovdb's truncation emulated by a byte
//    shuffle.
// Weight layout and loading after Strata's AVX2 NVFP4 rows (src/kernels/cpu/nvfp4_avx2.cpp
// @f57c0ad, MIT, Niko1221 and the Strata contributors): two nibble lookups per 16 code bytes,
// element order rebuilt with 64-bit unpacks, sub-block scales broadcast per 128-bit half (no
// gathers). Every function carries the target attribute and runs only after
// host_isa_supported(Avx2); the file has no dynamic initializer (constexpr tables only).
#include "ops/host_moe/host_nvfp4_isa.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#define NINFER_HOST_AVX2 __attribute__((target("avx2,fma,f16c")))

namespace ninfer::ops::host::detail::avx2 {
namespace {

alignas(16) constexpr std::int8_t kE2M1x2[16]   = {0, 1,  2,  3,  4,  6,  8,  12,
                                                   0, -1, -2, -3, -4, -6, -8, -12};
alignas(16) constexpr std::uint8_t kE2M1x2u[16] = {12, 13, 14, 15, 16, 18, 20, 24,
                                                   12, 11, 10, 9,  8,  6,  4,  0};

// _mm512_reduce_add_ps of the 16 lanes (lo = 0-7, hi = 8-15): halves of 256, of 128, then lanes
// {0,1} + {2,3}, then 0 + 1 (GCC's __MM512_REDUCE_OP; float addition is commutative).
NINFER_HOST_AVX2 inline float reduce_add16(__m256 lo, __m256 hi) {
    const __m256 s8 = _mm256_add_ps(hi, lo);
    const __m128 s4 = _mm_add_ps(_mm256_extractf128_ps(s8, 1), _mm256_castps256_ps128(s8));
    const __m128 s2 = _mm_add_ps(s4, _mm_permute_ps(s4, 0x4E));
    const __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0x55));
    return _mm_cvtss_f32(s1);
}

NINFER_HOST_AVX2 inline float reduce_max16(__m256 lo, __m256 hi) {
    const __m256 m8 = _mm256_max_ps(lo, hi);
    const __m128 m4 = _mm_max_ps(_mm256_extractf128_ps(m8, 1), _mm256_castps256_ps128(m8));
    const __m128 m2 = _mm_max_ps(m4, _mm_permute_ps(m4, 0x4E));
    return _mm_cvtss_f32(_mm_max_ss(m2, _mm_shuffle_ps(m2, m2, 0x55)));
}

NINFER_HOST_AVX2 inline __m256 abs_ps(__m256 v) {
    return _mm256_and_ps(v, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
}

// Lanes 0-3 = a, 4-7 = b.
NINFER_HOST_AVX2 inline __m256 halves_ps(float a, float b) {
    return _mm256_set_m128(_mm_set1_ps(b), _mm_set1_ps(a));
}

// The 64 weights of blocks b..b+3 (32 code bytes) through `lut`, laid out as the 512-bit kernels
// do: block j in bytes 16j..16j+15, its 8 even elements (low nibbles) then its 8 odd ones;
// lo = blocks 0-1, hi = blocks 2-3.
NINFER_HOST_AVX2 inline void decode64(const std::uint8_t* codes, __m256i lut, __m256i& lo,
                                      __m256i& hi) {
    const __m256i m4  = _mm256_set1_epi8(0x0F);
    const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes));
    const __m256i nl  = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw, m4));
    const __m256i nh  = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw, 4), m4));
    const __m256i a   = _mm256_unpacklo_epi64(nl, nh); // [block 0 | block 2]
    const __m256i c   = _mm256_unpackhi_epi64(nl, nh); // [block 1 | block 3]
    lo                = _mm256_permute2x128_si256(a, c, 0x20);
    hi                = _mm256_permute2x128_si256(a, c, 0x31);
}

// sum of each 4 consecutive u8 x s8 products as int32 (vpdpbusd from zero; exact here).
NINFER_HOST_AVX2 inline __m256i dot4(__m256i u8, __m256i s8) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(u8, s8), _mm256_set1_epi16(1));
}

// Low byte of every int32 (vpmovdb): lanes of e then lanes of o.
NINFER_HOST_AVX2 inline __m128i narrow16(__m256i e, __m256i o) {
    const __m256i pick =
        _mm256_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12,
                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i gather = _mm256_setr_epi32(0, 4, 0, 0, 0, 0, 0, 0);
    const __m128i e8 =
        _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(e, pick), gather));
    const __m128i o8 =
        _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(o, pick), gather));
    return _mm_unpacklo_epi64(e8, o8);
}

// 16 lanes (lo = 0-7, hi = 8-15) in stored order: even elements, then odd ones.
NINFER_HOST_AVX2 inline void even_odd_ps(__m256 lo, __m256 hi, __m256& even, __m256& odd) {
    const __m256i eo = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    const __m256 p0  = _mm256_permutevar8x32_ps(lo, eo);
    const __m256 p1  = _mm256_permutevar8x32_ps(hi, eo);
    even             = _mm256_permute2f128_ps(p0, p1, 0x20);
    odd              = _mm256_permute2f128_ps(p0, p1, 0x31);
}

// ---------------------------------------------------------------- dots
// Multi-token dot (V1: (w + 12) u8 x q s8 from corr; V4: (q + 128) u8 x w s8 - 128 sum(w));
// the two give the same integers, V as in host_nvfp4.cpp.
template <int T, int V>
NINFER_HOST_AVX2 inline void dot_multi_token(const std::uint8_t* packed, const std::uint8_t* sc,
                                             int k, const W4A8Activation* act, float* s,
                                             const float* e4m3) {
    static_assert(T >= 1 && T <= kHostMaxTokens && (V == 1 || V == 4));
    const __m256i lut = _mm256_broadcastsi128_si256(
        _mm_load_si128(V == 1 ? reinterpret_cast<const __m128i*>(kE2M1x2u)
                              : reinterpret_cast<const __m128i*>(kE2M1x2)));
    const __m256i c128 = _mm256_set1_epi8(static_cast<char>(0x80));
    __m256 lo[T], hi[T];
#pragma GCC unroll 8
    for (int t = 0; t < T; ++t) lo[t] = hi[t] = _mm256_setzero_ps();
    const int nb = k / kNvfp4Group;
    for (int b = 0; b < nb; b += 4) {
        __m256i wl, wh;
        decode64(packed + static_cast<std::size_t>(b) * 8, lut, wl, wh);
        const std::uint8_t* q4 = sc + static_cast<std::size_t>(b / 4) * kScaleQuadStride;
        const __m256 wsl       = halves_ps(e4m3[q4[0]], e4m3[q4[1]]);
        const __m256 wsh       = halves_ps(e4m3[q4[2]], e4m3[q4[3]]);
        __m256i pl = _mm256_setzero_si256(), ph = _mm256_setzero_si256();
        if constexpr (V == 4) {
            pl = dot4(c128, wl);
            ph = dot4(c128, wh);
        }
        const std::size_t o16 = static_cast<std::size_t>(b) * 16,
                          o4  = static_cast<std::size_t>(b) * 4;
#pragma GCC unroll 8
        for (int t = 0; t < T; ++t) {
            __m256i dl, dh;
            if constexpr (V == 1) {
                const auto* q = reinterpret_cast<const __m256i*>(act[t].q + o16);
                const auto* c = reinterpret_cast<const __m256i*>(act[t].corr + o4);
                dl = _mm256_add_epi32(_mm256_loadu_si256(c), dot4(wl, _mm256_loadu_si256(q)));
                dh = _mm256_add_epi32(_mm256_loadu_si256(c + 1),
                                      dot4(wh, _mm256_loadu_si256(q + 1)));
            } else {
                const auto* qu = reinterpret_cast<const __m256i*>(act[t].qu + o16);
                dl             = _mm256_sub_epi32(dot4(_mm256_loadu_si256(qu), wl), pl);
                dh             = _mm256_sub_epi32(dot4(_mm256_loadu_si256(qu + 1), wh), ph);
            }
            const __m256 sl = _mm256_mul_ps(wsl, _mm256_loadu_ps(act[t].sb16 + o4));
            const __m256 sh = _mm256_mul_ps(wsh, _mm256_loadu_ps(act[t].sb16 + o4 + 8));
            lo[t]           = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dl), sl, lo[t]);
            hi[t]           = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dh), sh, hi[t]);
        }
    }
#pragma GCC unroll 8
    for (int t = 0; t < T; ++t) s[t] = reduce_add16(lo[t], hi[t]);
}

template <int T>
NINFER_HOST_AVX2 void rows_multi_token(const Nvfp4Plane& p, std::int64_t r0, int n,
                                       const W4A8Activation* act, float* out, std::size_t ldo) {
    constexpr int V   = T <= 5 ? 1 : 4;
    const float* e4m3 = e4m3_table();
    for (int i = 0; i < n; ++i) {
        const std::int64_t r = r0 + i;
        float s[T];
        dot_multi_token<T, V>(p.row_codes(r), p.row_scales(r), p.k, act, s, e4m3);
        const float d = p.divisor(r);
#pragma GCC unroll 8
        for (int t = 0; t < T; ++t) out[static_cast<std::size_t>(t) * ldo + i] = s[t] * 0.5F / d;
    }
}

using RowsFn = void (*)(const Nvfp4Plane&, std::int64_t, int, const W4A8Activation*, float*,
                        std::size_t);
constexpr RowsFn kRowsMultiToken[kHostMaxTokens + 1] = {nullptr,
                                                        rows_multi_token<1>,
                                                        rows_multi_token<2>,
                                                        rows_multi_token<3>,
                                                        rows_multi_token<4>,
                                                        rows_multi_token<5>,
                                                        rows_multi_token<6>,
                                                        rows_multi_token<7>,
                                                        rows_multi_token<8>};

// ---------------------------------------------------------------- activation codecs
// Stores q (stored order, int32 lanes of even and odd elements), qu = q + 128, corr and the scale.
NINFER_HOST_AVX2 inline void store_block(__m256i qe, __m256i qo, float s, int b, W4A8Activation a) {
    const std::size_t o = static_cast<std::size_t>(b) * 16;
    const __m128i q8    = narrow16(qe, qo);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(a.q + o), q8);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(a.qu + o),
                     _mm_xor_si128(q8, _mm_set1_epi8(static_cast<char>(0x80))));
    const __m128i sum4 = _mm_madd_epi16(_mm_maddubs_epi16(_mm_set1_epi8(1), q8), _mm_set1_epi16(1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(a.corr + static_cast<std::size_t>(b) * 4),
                     _mm_mullo_epi32(sum4, _mm_set1_epi32(-12)));
    _mm_storeu_ps(a.sb16 + static_cast<std::size_t>(b) * 4, _mm_set1_ps(s));
    a.asb[b] = s;
}

// q = rint(x / s) with s = absmax / 127 (x times fl(1 / s), as host_nvfp4.cpp).
NINFER_HOST_AVX2 inline void quantize_block(__m256 lo, __m256 hi, int b, W4A8Activation a) {
    const float amax = reduce_max16(abs_ps(lo), abs_ps(hi));
    const float s    = amax > 0.0F ? amax / 127.0F : 1.0F;
    const __m256 inv = _mm256_set1_ps(1.0F / s);
    __m256 even, odd;
    even_odd_ps(lo, hi, even, odd);
    store_block(_mm256_cvtps_epi32(_mm256_mul_ps(even, inv)),
                _mm256_cvtps_epi32(_mm256_mul_ps(odd, inv)), s, b, a);
}

// 2 * e2m1_rn(y) of 8 lanes, signed (host_nvfp4.cpp's a4_block_vec).
NINFER_HOST_AVX2 inline __m256i a4_codes(__m256 y) {
    constexpr int kRne = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    const __m256 ya    = abs_ps(y);
    const __m256 r_lo  = _mm256_round_ps(_mm256_mul_ps(ya, _mm256_set1_ps(2.0F)), kRne);
    const __m256 r_mid = _mm256_mul_ps(_mm256_round_ps(ya, kRne), _mm256_set1_ps(2.0F));
    const __m256 r_hi =
        _mm256_min_ps(_mm256_mul_ps(_mm256_round_ps(_mm256_mul_ps(ya, _mm256_set1_ps(0.5F)), kRne),
                                    _mm256_set1_ps(4.0F)),
                      _mm256_set1_ps(12.0F));
    __m256 r = _mm256_blendv_ps(r_lo, r_mid, _mm256_cmp_ps(ya, _mm256_set1_ps(2.0F), _CMP_GT_OQ));
    r        = _mm256_blendv_ps(r, r_hi, _mm256_cmp_ps(ya, _mm256_set1_ps(4.0F), _CMP_GT_OQ));
    const __m256i q32 = _mm256_cvtps_epi32(r);
    const __m256i neg = _mm256_castps_si256(_mm256_cmp_ps(y, _mm256_setzero_ps(), _CMP_LT_OQ));
    return _mm256_blendv_epi8(q32, _mm256_sub_epi32(_mm256_setzero_si256(), q32), neg);
}

NINFER_HOST_AVX2 inline void a4_block(__m256 lo, __m256 hi, int b, float gs, W4A8Activation a) {
    const float amax      = reduce_max16(abs_ps(lo), abs_ps(hi));
    const float sf        = a4_e4m3_rn(gs * (amax * (1.0F / 6.0F)));
    const float out_scale = (amax != 0.F && sf != 0.F) ? 1.0F / (sf * (1.0F / gs)) : 0.F;
    const __m256 os       = _mm256_set1_ps(out_scale);
    __m256 even, odd;
    even_odd_ps(_mm256_mul_ps(lo, os), _mm256_mul_ps(hi, os), even, odd);
    store_block(a4_codes(even), a4_codes(odd), sf / (2.0F * gs), b, a);
}

NINFER_HOST_AVX2 inline void load_bf16x16(const std::uint16_t* x, __m256& lo, __m256& hi) {
    const auto* p = reinterpret_cast<const __m128i*>(x);
    lo = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(p)), 16));
    hi = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(p + 1)), 16));
}

NINFER_HOST_AVX2 void quantize_blocks_impl(const float* x, int block0, int blocks,
                                           W4A8Activation a) {
    for (int i = 0; i < blocks; ++i) {
        const float* v = x + static_cast<std::size_t>(i) * 16;
        quantize_block(_mm256_loadu_ps(v), _mm256_loadu_ps(v + 8), block0 + i, a);
    }
}

NINFER_HOST_AVX2 void quantize_bf16_impl(const std::uint16_t* x, int k, W4A8Activation a) {
    for (int b = 0; b < k / kNvfp4Group; ++b) {
        __m256 lo, hi;
        load_bf16x16(x + static_cast<std::size_t>(b) * 16, lo, hi);
        quantize_block(lo, hi, b, a);
    }
}

NINFER_HOST_AVX2 void quantize_blocks_a4_impl(const float* x, int block0, int blocks, float gs,
                                              W4A8Activation a) {
    for (int i = 0; i < blocks; ++i) {
        const float* v = x + static_cast<std::size_t>(i) * 16;
        a4_block(_mm256_loadu_ps(v), _mm256_loadu_ps(v + 8), block0 + i, gs, a);
    }
}

NINFER_HOST_AVX2 void quantize_bf16_a4_impl(const std::uint16_t* x, int k, float gs,
                                            W4A8Activation a) {
    for (int b = 0; b < k / kNvfp4Group; ++b) {
        __m256 lo, hi;
        load_bf16x16(x + static_cast<std::size_t>(b) * 16, lo, hi);
        a4_block(lo, hi, b, gs, a);
    }
}

} // namespace

void quantize_blocks(const float* x, std::int32_t block0, std::int32_t blocks, W4A8Activation a) {
    quantize_blocks_impl(x, block0, blocks, a);
}

void quantize_bf16(const std::uint16_t* x, std::int32_t k, W4A8Activation a) {
    quantize_bf16_impl(x, k, a);
}

void quantize_blocks_a4(const float* x, std::int32_t block0, std::int32_t blocks, float gs,
                        W4A8Activation a) {
    quantize_blocks_a4_impl(x, block0, blocks, gs, a);
}

void quantize_bf16_a4(const std::uint16_t* x, std::int32_t k, float gs, W4A8Activation a) {
    quantize_bf16_a4_impl(x, k, gs, a);
}

void rows_dot(const Nvfp4Plane& plane, std::int64_t r0, std::int32_t n, const W4A8Activation* acts,
              std::int32_t tokens, float* out, std::size_t ldo) {
    kRowsMultiToken[tokens](plane, r0, n, acts, out, ldo);
}

} // namespace ninfer::ops::host::detail::avx2
