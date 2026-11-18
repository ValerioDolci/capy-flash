// NVFP4 W4A8 host dots. dot_multi_token decodes the weights with FreeToken's nvfp4_i8_grp4
// (python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp @555efd8, Apache-2.0, Copyright
// FlashML-org / FreeToken authors), changed to read the scale quad of a row from the swizzled
// plane and to divide by the plane's FP32 divisor. The AVX2 forms of the same kernels are in
// host_nvfp4_avx2.cpp; this file also selects the ISA (host_isa) and dispatches the public
// functions.
#include "ops/host_moe/host_nvfp4.h"

#include "ops/host_moe/host_nvfp4_isa.h"

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#define NINFER_HOST_AVX512 __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")))

namespace ninfer::ops::host {
namespace {

std::size_t align_up(std::size_t v, std::size_t a) { return (v + a - 1) / a * a; }

float decode_e4m3(std::uint8_t v) {
    const int s = v >> 7, e = (v >> 3) & 0xF, m = v & 7;
    if (e == 0xF && m == 7) return 0.0F; // NaN code: never produced by the converter
    const float f = e == 0 ? std::ldexp(static_cast<float>(m) / 8.0F, -6)
                           : std::ldexp(1.0F + static_cast<float>(m) / 8.0F, e - 7);
    return s ? -f : f;
}

struct E4m3Table {
    alignas(64) std::array<float, 256> value{};

    E4m3Table() {
        for (int i = 0; i < 256; ++i) value[i] = decode_e4m3(static_cast<std::uint8_t>(i));
    }
};

const E4m3Table kE4m3;

alignas(16) const std::int8_t kE2M1x2[16]   = {0, 1,  2,  3,  4,  6,  8,  12,
                                               0, -1, -2, -3, -4, -6, -8, -12};
alignas(16) const std::uint8_t kE2M1x2u[16] = {12, 13, 14, 15, 16, 18, 20, 24,
                                               12, 11, 10, 9,  8,  6,  4,  0};
} // namespace

namespace detail::avx512 {
namespace {

// Multi-token dot. V1: (w + 12) u8 x q s8 from corr; V4: (q + 128) u8 x w s8 from zero, then
// - 128 * sum(w). Same integer sums, same float products, same accumulation: bit-identical.
// V1 keeps 3k bytes of activation per token (spills the 48 KB L1D above ~5 tokens at k 2560),
// V4 2k: V1 up to 5 tokens, V4 above.
template <int T, int V>
NINFER_HOST_AVX512 inline void dot_multi_token(const std::uint8_t* packed, const std::uint8_t* sc,
                                               int k, const W4A8Activation* act, float* s) {
    static_assert(T >= 1 && T <= kHostMaxTokens && (V == 1 || V == 4));
    const __m512i lut =
        V == 1 ? _mm512_broadcast_i32x4(_mm_load_si128(reinterpret_cast<const __m128i*>(kE2M1x2u)))
               : _mm512_broadcast_i32x4(_mm_load_si128(reinterpret_cast<const __m128i*>(kE2M1x2)));
    const __m512i idx       = _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0);
    const __m512i mask0F    = _mm512_set1_epi8(0x0F);
    const __m512i idxsc     = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
    const __m512i c128      = _mm512_set1_epi8(static_cast<char>(0x80));
    const __mmask64 hi_half = 0xFF00FF00FF00FF00ULL;
    __m512 acc[T];
#pragma GCC unroll 8
    for (int t = 0; t < T; ++t) acc[t] = _mm512_setzero_ps();
    const int nb = k / kNvfp4Group;
    for (int b = 0; b < nb; b += 4) {
        const __m256i raw = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(packed + static_cast<std::size_t>(b) * 8));
        const __m512i src = _mm512_permutexvar_epi64(idx, _mm512_castsi256_si512(raw));
        const __m512i lo  = _mm512_and_si512(src, mask0F);
        const __m512i hi  = _mm512_and_si512(_mm512_srli_epi16(src, 4), mask0F);
        const __m512i w   = _mm512_shuffle_epi8(lut, _mm512_mask_blend_epi8(hi_half, lo, hi));
        int sc_raw;
        std::memcpy(&sc_raw, sc + static_cast<std::size_t>(b / 4) * kScaleQuadStride, 4);
        const __m128 ws4 =
            _mm_i32gather_ps(kE4m3.value.data(), _mm_cvtepu8_epi32(_mm_cvtsi32_si128(sc_raw)), 4);
        const __m512 ws = _mm512_permutexvar_ps(idxsc, _mm512_castps128_ps512(ws4));
        __m512i posc    = _mm512_setzero_si512();
        if constexpr (V == 4) posc = _mm512_dpbusd_epi32(posc, c128, w); // +128 * sum(w)
#pragma GCC unroll 8
        for (int t = 0; t < T; ++t) {
            __m512i di;
            if constexpr (V == 1) {
                di = _mm512_dpbusd_epi32(
                    _mm512_loadu_si512(act[t].corr + static_cast<std::size_t>(b) * 4), w,
                    _mm512_loadu_si512(act[t].q + static_cast<std::size_t>(b) * 16));
            } else {
                di = _mm512_sub_epi32(
                    _mm512_dpbusd_epi32(
                        _mm512_setzero_si512(),
                        _mm512_loadu_si512(act[t].qu + static_cast<std::size_t>(b) * 16), w),
                    posc);
            }
            const __m512 scv =
                _mm512_mul_ps(ws, _mm512_loadu_ps(act[t].sb16 + static_cast<std::size_t>(b) * 4));
            acc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(di), scv, acc[t]);
        }
    }
#pragma GCC unroll 8
    for (int t = 0; t < T; ++t) s[t] = _mm512_reduce_add_ps(acc[t]);
}

template <int T>
NINFER_HOST_AVX512 void rows_multi_token(const Nvfp4Plane& p, std::int64_t r0, int n,
                                         const W4A8Activation* act, float* out, std::size_t ldo) {
    constexpr int V = T <= 5 ? 1 : 4;
    for (int i = 0; i < n; ++i) {
        const std::int64_t r = r0 + i;
        float s[T];
        dot_multi_token<T, V>(p.row_codes(r), p.row_scales(r), p.k, act, s);
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

// q = rint(x / s) in [even 8, odd 8] order, s = absmax / 127; corr, sb16, qu of the block.
NINFER_HOST_AVX512 inline void quantize_block(__m512 v, int b, W4A8Activation a) {
    const __m512i even_odd = _mm512_set_epi32(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);
    const float amax       = _mm512_reduce_max_ps(_mm512_abs_ps(v));
    const float s          = amax > 0.0F ? amax / 127.0F : 1.0F;
    const float inv        = 1.0F / s;
    const __m512i qi =
        _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_permutexvar_ps(even_odd, v), _mm512_set1_ps(inv)));
    const __m128i q8 = _mm512_cvtepi32_epi8(qi);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(a.q + static_cast<std::size_t>(b) * 16), q8);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(a.qu + static_cast<std::size_t>(b) * 16),
                     _mm_xor_si128(q8, _mm_set1_epi8(static_cast<char>(0x80))));
    const __m128i sum4 =
        _mm_dpbusd_epi32(_mm_setzero_si128(), _mm_set1_epi8(1), q8); // u8 ones x s8 q
    _mm_storeu_si128(reinterpret_cast<__m128i*>(a.corr + static_cast<std::size_t>(b) * 4),
                     _mm_mullo_epi32(sum4, _mm_set1_epi32(-12)));
    _mm_storeu_ps(a.sb16 + static_cast<std::size_t>(b) * 4, _mm_set1_ps(s));
    a.asb[b] = s;
}

NINFER_HOST_AVX512 void quantize_blocks_impl(const float* x, int block0, int blocks,
                                             W4A8Activation a) {
    for (int i = 0; i < blocks; ++i)
        quantize_block(_mm512_loadu_ps(x + static_cast<std::size_t>(i) * 16), block0 + i, a);
}

NINFER_HOST_AVX512 void quantize_bf16_impl(const std::uint16_t* x, int k, W4A8Activation a) {
    for (int b = 0; b < k / kNvfp4Group; ++b) {
        const __m256i h = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(x + static_cast<std::size_t>(b) * 16));
        const __m512 v = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
        quantize_block(v, b, a);
    }
}

} // namespace
} // namespace detail::avx512

float e4m3_value(std::uint8_t code) noexcept { return kE4m3.value[code]; }

const float* detail::e4m3_table() noexcept { return kE4m3.value.data(); }

// ---------------------------------------------------------------- ISA selection
// __builtin_cpu_supports reads cpuid and, for the AVX/AVX-512 features, the OS-enabled state
// (xgetbv): a feature the kernel does not save is reported absent.
bool host_isa_supported(HostIsa isa) noexcept {
    const bool avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
                      __builtin_cpu_supports("f16c");
    switch (isa) {
    case HostIsa::Avx2:
        return avx2;
    case HostIsa::Avx512Vnni:
        return avx2 && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
               __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vnni");
    }
    return false;
}

bool host_experts_supported() noexcept { return host_isa_supported(HostIsa::Avx2); }

bool host_avx512_vnni_supported() noexcept { return host_isa_supported(HostIsa::Avx512Vnni); }

const char* host_isa_name(HostIsa isa) noexcept {
    switch (isa) {
    case HostIsa::Avx2:
        return "avx2";
    case HostIsa::Avx512Vnni:
        return "avx512";
    }
    return "?";
}

namespace {
HostIsa select_host_isa() {
    const char* env        = std::getenv("NINFER_HOST_ISA");
    const std::string want = env != nullptr ? env : "auto";
    HostIsa isa;
    if (want.empty() || want == "auto") {
        if (host_isa_supported(HostIsa::Avx512Vnni)) return HostIsa::Avx512Vnni;
        if (host_isa_supported(HostIsa::Avx2)) return HostIsa::Avx2;
        throw std::runtime_error(
            "host expert kernels: this CPU runs neither AVX-512 VNNI nor AVX2 + FMA + F16C");
    }
    if (want == "avx2") {
        isa = HostIsa::Avx2;
    } else if (want == "avx512") {
        isa = HostIsa::Avx512Vnni;
    } else {
        throw std::runtime_error("NINFER_HOST_ISA=" + want + ": expected avx2, avx512 or auto");
    }
    if (!host_isa_supported(isa)) {
        throw std::runtime_error("NINFER_HOST_ISA=" + want + ": this CPU does not run it");
    }
    return isa;
}

constexpr HostNvfp4Kernels kAvx512Kernels{HostIsa::Avx512Vnni,
                                          detail::avx512::quantize_blocks,
                                          detail::avx512::quantize_bf16,
                                          detail::avx512::quantize_blocks_a4,
                                          detail::avx512::quantize_bf16_a4,
                                          detail::avx512::rows_dot};
constexpr HostNvfp4Kernels kAvx2Kernels{HostIsa::Avx2,
                                        detail::avx2::quantize_blocks,
                                        detail::avx2::quantize_bf16,
                                        detail::avx2::quantize_blocks_a4,
                                        detail::avx2::quantize_bf16_a4,
                                        detail::avx2::rows_dot};

const HostNvfp4Kernels& active_kernels() {
    static const HostNvfp4Kernels& k = host_nvfp4_kernels(host_isa());
    return k;
}
} // namespace

HostIsa host_isa() {
    static const HostIsa isa = select_host_isa();
    return isa;
}

const HostNvfp4Kernels& host_nvfp4_kernels(HostIsa isa) {
    if (!host_isa_supported(isa)) {
        throw std::runtime_error(std::string("host expert kernels: this CPU does not run ") +
                                 host_isa_name(isa));
    }
    return isa == HostIsa::Avx512Vnni ? kAvx512Kernels : kAvx2Kernels;
}

const HostNvfp4Kernels& host_nvfp4_kernels(std::optional<HostIsa> isa) {
    return isa ? host_nvfp4_kernels(*isa) : active_kernels();
}

std::size_t Nvfp4Plane::payload_bytes(std::int64_t rows, std::int32_t k,
                                      std::int64_t divisor_count) {
    const std::size_t codes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(k) / 2;
    return align_up(codes, 256) +
           static_cast<std::size_t>(rows) * static_cast<std::size_t>(k) / 16 +
           4 * static_cast<std::size_t>(divisor_count);
}

Nvfp4Plane Nvfp4Plane::from_payload(const void* payload, std::int64_t rows, std::int32_t k,
                                    std::int64_t divisor_count) {
    if (rows <= 0 || rows % kNvfp4RowTile || k <= 0 || k % kNvfp4Quad || divisor_count <= 0 ||
        rows % divisor_count || (divisor_count > 1 && (rows / divisor_count) % kNvfp4RowTile)) {
        throw std::invalid_argument("Nvfp4Plane: not a block_scale_k16_m128x4_v1 geometry");
    }
    const auto* base         = static_cast<const std::uint8_t*>(payload);
    const std::size_t codes  = static_cast<std::size_t>(rows) * static_cast<std::size_t>(k) / 2;
    const std::size_t scales = align_up(codes, 256);
    Nvfp4Plane p;
    p.codes            = base;
    p.scales           = base + scales;
    p.divisors         = reinterpret_cast<const float*>(base + scales + codes / 8);
    p.rows             = rows;
    p.k                = k;
    p.rows_per_divisor = rows / divisor_count;
    return p;
}

void W4A8ActivationBuffer::Free::operator()(void* p) const noexcept { std::free(p); }

W4A8ActivationBuffer::W4A8ActivationBuffer(std::int32_t k, std::int32_t count)
    : k_(k), count_(count) {
    if (k <= 0 || k % kNvfp4Quad || count <= 0)
        throw std::invalid_argument("W4A8ActivationBuffer: bad shape");
    const auto K          = static_cast<std::size_t>(k);
    q_stride_             = align_up(K, 64);
    asb_stride_           = align_up(K / 16 * sizeof(float), 64);
    d_stride_             = align_up(K / 4 * sizeof(float), 64);
    const std::size_t per = 2 * q_stride_ + asb_stride_ + 2 * d_stride_;
    void* raw             = std::aligned_alloc(64, per * static_cast<std::size_t>(count));
    if (!raw) throw std::bad_alloc();
    std::memset(raw, 0, per * static_cast<std::size_t>(count));
    storage_.reset(static_cast<std::byte*>(raw));
    const auto n = static_cast<std::size_t>(count);
    q_           = storage_.get();
    qu_          = q_ + q_stride_ * n;
    asb_         = qu_ + q_stride_ * n;
    corr_        = asb_ + asb_stride_ * n;
    sb16_        = corr_ + d_stride_ * n;
}

W4A8Activation W4A8ActivationBuffer::at(std::int32_t i) const noexcept {
    const auto n = static_cast<std::size_t>(i);
    return {reinterpret_cast<std::int8_t*>(q_ + q_stride_ * n),
            reinterpret_cast<float*>(asb_ + asb_stride_ * n),
            reinterpret_cast<std::int32_t*>(corr_ + d_stride_ * n),
            reinterpret_cast<float*>(sb16_ + d_stride_ * n),
            reinterpret_cast<std::uint8_t*>(qu_ + q_stride_ * n)};
}

void detail::avx512::quantize_blocks(const float* x, std::int32_t block0, std::int32_t blocks,
                                     W4A8Activation a) {
    quantize_blocks_impl(x, block0, blocks, a);
}

void detail::avx512::quantize_bf16(const std::uint16_t* x, std::int32_t k, W4A8Activation a) {
    quantize_bf16_impl(x, k, a);
}

float detail::a4_e4m3_rn(float v) noexcept {
    if (!(v > 0.0F)) return 0.0F;
    if (v >= 448.0F) return 448.0F;
    int ex = 0;
    std::frexp(v, &ex);
    const int exponent = std::max(ex - 1, -6);
    const float step   = std::ldexp(1.0F, exponent - 3);
    return std::min(std::nearbyint(v / step) * step, 448.0F);
}

namespace detail::avx512 {
namespace {
// One block of the A4 codec (host_nvfp4.h quantize_blocks_a4): per 16 values sf = e4m3_rn(gs *
// amax / 6), q = 2 * e2m1_rn(x * gs / sf) with ties to the even code, satfinite.
NINFER_HOST_AVX512 void a4_block_vec(__m512 v, int b, float gs, W4A8Activation a) {
    const __m512 av       = _mm512_abs_ps(v);
    const float amax      = _mm512_reduce_max_ps(av);
    const float sf        = a4_e4m3_rn(gs * (amax * (1.0F / 6.0F)));
    const float out_scale = (amax != 0.F && sf != 0.F) ? 1.0F / (sf * (1.0F / gs)) : 0.F;
    const __m512 y        = _mm512_mul_ps(v, _mm512_set1_ps(out_scale));
    const __m512 ya       = _mm512_abs_ps(y);
    constexpr int kRne    = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    // r = 2 * e2m1_rn(|y|) as an integer-valued float: a <= 2 -> rint(2a); 2 < a <= 4 -> 2 rint(a);
    // a > 4 -> 4 rint(a/2) clamped to 12.
    const __m512 r_lo  = _mm512_roundscale_ps(_mm512_mul_ps(ya, _mm512_set1_ps(2.0F)), kRne);
    const __m512 r_mid = _mm512_mul_ps(_mm512_roundscale_ps(ya, kRne), _mm512_set1_ps(2.0F));
    const __m512 r_hi  = _mm512_min_ps(
        _mm512_mul_ps(_mm512_roundscale_ps(_mm512_mul_ps(ya, _mm512_set1_ps(0.5F)), kRne),
                       _mm512_set1_ps(4.0F)),
        _mm512_set1_ps(12.0F));
    const __mmask16 m_mid = _mm512_cmp_ps_mask(ya, _mm512_set1_ps(2.0F), _CMP_GT_OQ);
    const __mmask16 m_hi  = _mm512_cmp_ps_mask(ya, _mm512_set1_ps(4.0F), _CMP_GT_OQ);
    __m512 r              = _mm512_mask_mov_ps(r_lo, m_mid, r_mid);
    r                     = _mm512_mask_mov_ps(r, m_hi, r_hi);
    __m512i q32           = _mm512_cvtps_epi32(r);
    const __mmask16 neg   = _mm512_cmp_ps_mask(y, _mm512_setzero_ps(), _CMP_LT_OQ);
    q32                   = _mm512_mask_sub_epi32(q32, neg, _mm512_setzero_si512(), q32);
    // stored order [even 8, odd 8]
    const __m512i perm  = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    const __m512i qs    = _mm512_permutexvar_epi32(perm, q32);
    const std::size_t o = static_cast<std::size_t>(b) * 16;
    _mm_storeu_si128(reinterpret_cast<__m128i*>(a.q + o), _mm512_cvtepi32_epi8(qs));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(a.qu + o),
                     _mm512_cvtepi32_epi8(_mm512_add_epi32(qs, _mm512_set1_epi32(128))));
    alignas(64) std::int32_t lanes[16];
    _mm512_store_si512(lanes, qs);
    for (int d = 0; d < 4; ++d) {
        a.corr[static_cast<std::size_t>(b) * 4 + d] =
            -12 * (lanes[4 * d] + lanes[4 * d + 1] + lanes[4 * d + 2] + lanes[4 * d + 3]);
    }
    const float sc = sf / (2.0F * gs);
    for (int d = 0; d < 4; ++d) a.sb16[static_cast<std::size_t>(b) * 4 + d] = sc;
    a.asb[b] = sc;
}

NINFER_HOST_AVX512 void quantize_blocks_a4_impl(const float* x, int block0, int blocks, float gs,
                                                W4A8Activation a) {
    for (int i = 0; i < blocks; ++i)
        a4_block_vec(_mm512_loadu_ps(x + static_cast<std::size_t>(i) * 16), block0 + i, gs, a);
}

NINFER_HOST_AVX512 void quantize_bf16_a4_impl(const std::uint16_t* x, int k, float gs,
                                              W4A8Activation a) {
    for (int b = 0; b < k / kNvfp4Group; ++b) {
        const __m256i h = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(x + static_cast<std::size_t>(b) * 16));
        a4_block_vec(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16)), b, gs,
                     a);
    }
}
} // namespace

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
} // namespace detail::avx512

// ---------------------------------------------------------------- public (host_isa()) entry points
void quantize_blocks(const float* x, std::int32_t block0, std::int32_t blocks, W4A8Activation a) {
    active_kernels().quantize_blocks(x, block0, blocks, a);
}

void quantize_bf16(const std::uint16_t* x, std::int32_t k, W4A8Activation a) {
    active_kernels().quantize_bf16(x, k, a);
}

void quantize_blocks_a4(const float* x, std::int32_t block0, std::int32_t blocks, float gs,
                        W4A8Activation a) {
    active_kernels().quantize_blocks_a4(x, block0, blocks, gs, a);
}

void quantize_bf16_a4(const std::uint16_t* x, std::int32_t k, float gs, W4A8Activation a) {
    active_kernels().quantize_bf16_a4(x, k, gs, a);
}

void rows_dot(const Nvfp4Plane& plane, std::int64_t r0, std::int32_t n, const W4A8Activation* acts,
              std::int32_t tokens, float* out, std::size_t ldo) {
    active_kernels().rows_dot(plane, r0, n, acts, tokens, out, ldo);
}

} // namespace ninfer::ops::host
