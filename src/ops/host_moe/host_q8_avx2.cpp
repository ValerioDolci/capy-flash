// Modified for capy-flash from fngine:src/cpu/q8_cpu.cpp @ c20a885 (satellitedown/fngine, Apache-2.0).
// See NOTICE. capy-flash changes: the AVX2 + FMA + F16C form of host_q8.cpp's quantizers
// (namespace ninfer::ops::host::gguf::detail::avx2, dispatched by host_gguf.cpp), the same bytes:
// every 512-bit lane set is kept as two 256-bit halves (lanes 0-7, 8-15) with the same per-lane
// operations, the maxima are reduced in _mm512_reduce_max_ps's order and vpmovdb's truncation to
// bytes is emulated exactly (masked low bytes, unsigned packs).
#include "ops/host_moe/host_gguf_isa.h"

#include <immintrin.h>

#include <cstring>

#pragma GCC push_options
#pragma GCC target("avx2,fma,f16c")

// Bit-exactness notes (this file is compiled with -ffp-contract=off, see sources.cmake):
//  * Q8_K: the reference rounds with nearest_int() (add 1.5*2^23, round-to-nearest-even), which
//    equals vcvtps2dq under the default MXCSR rounding mode; the signed max is the FIRST element
//    of largest magnitude.
//  * Q8_0: the reference uses id = 1/(amax/127) and roundf() (half away from zero).
namespace ninfer::ops::host::gguf::detail::avx2 {
namespace {

// 16 values as two halves (lo = 0-7, hi = 8-15).
struct V16 {
  __m256 lo, hi;
};

inline V16 load16(const float* p) { return {_mm256_loadu_ps(p), _mm256_loadu_ps(p + 8)}; }
inline V16 load16(const uint16_t* p) {
  const auto* q = reinterpret_cast<const __m128i*>(p);
  return {_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(q)), 16)),
          _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(q + 1)), 16))};
}
inline float scalar_at(const float* p, int i) { return p[i]; }
inline float scalar_at(const uint16_t* p, int i) {
  const uint32_t w = uint32_t(p[i]) << 16;
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}

inline __m256 abs8(__m256 v) {
  return _mm256_and_ps(v, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
}

// _mm512_reduce_max_ps (GCC's __MM512_REDUCE_OP order, operands as there).
inline float reduce_max16(__m256 lo, __m256 hi) {
  const __m256 m8 = _mm256_max_ps(hi, lo);
  const __m128 m4 = _mm_max_ps(_mm256_extractf128_ps(m8, 1), _mm256_castps256_ps128(m8));
  const __m128 m2 = _mm_max_ps(m4, _mm_shuffle_ps(m4, m4, _MM_SHUFFLE(1, 0, 3, 2)));
  const __m128 m1 = _mm_max_ps(m2, _mm_shuffle_ps(m2, m2, _MM_SHUFFLE(0, 1, 0, 1)));
  return _mm_cvtss_f32(m1);
}

// Low byte of every int32 of a (lanes 0-7) then b (8-15): vpmovdb of the 16 lanes.
inline __m128i narrow16(__m256i a, __m256i b) {
  const __m256i m = _mm256_set1_epi32(0xFF);
  const __m256i w = _mm256_packus_epi32(_mm256_and_si256(a, m), _mm256_and_si256(b, m));
  // w (u16) = [a0-3 b0-3 | a4-7 b4-7]; bytes after packus = [a0-3 b0-3 a0-3 b0-3 | ...].
  const __m256i bytes = _mm256_packus_epi16(w, w);
  const __m256i order = _mm256_permutevar8x32_epi32(bytes, _mm256_setr_epi32(0, 4, 1, 5, 0, 0, 0, 0));
  return _mm256_castsi256_si128(order);
}

inline int32_t hsum8(__m256i v) {
  __m128i s = _mm_add_epi32(_mm256_extracti128_si256(v, 1), _mm256_castsi256_si128(v));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
  return _mm_cvtsi128_si32(s);
}

template <class T>
void q8K_block(const T* x, block_q8_K* y) {
  __m256 alo = _mm256_setzero_ps(), ahi = _mm256_setzero_ps();
  for (int i = 0; i < 16; ++i) {
    const V16 v = load16(x + 16 * i);
    alo = _mm256_max_ps(alo, abs8(v.lo));
    ahi = _mm256_max_ps(ahi, abs8(v.hi));
  }
  const float amax = reduce_max16(alo, ahi);
  if (!amax) {
    y->d = 0;
    std::memset(y->qs, 0, sizeof(y->qs));
    std::memset(y->bsums, 0, sizeof(y->bsums));
    return;
  }
  const __m256 amax8 = _mm256_set1_ps(amax);
  int idx = 0;
  for (int i = 0; i < 16; ++i) {
    const V16 v = load16(x + 16 * i);
    const unsigned m = unsigned(_mm256_movemask_ps(_mm256_cmp_ps(abs8(v.lo), amax8, _CMP_EQ_OQ))) |
                       unsigned(_mm256_movemask_ps(_mm256_cmp_ps(abs8(v.hi), amax8, _CMP_EQ_OQ))) << 8;
    if (m) {
      idx = 16 * i + __builtin_ctz(m);
      break;
    }
  }
  const float max = scalar_at(x, idx);
  const float iscale = -127.f / max;
  const __m256 is = _mm256_set1_ps(iscale);
  const __m256i lim = _mm256_set1_epi32(127);
  for (int i = 0; i < 16; ++i) {
    const V16 v = load16(x + 16 * i);
    const __m256i qa = _mm256_min_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(v.lo, is)), lim);
    const __m256i qb = _mm256_min_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(v.hi, is)), lim);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(y->qs + 16 * i), narrow16(qa, qb));
    y->bsums[i] = static_cast<int16_t>(hsum8(_mm256_add_epi32(qa, qb)));
  }
  y->d = 1 / iscale;
}

inline __m256i roundf8(__m256 x) {
  __m256 t = _mm256_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
  const __m256 diff = _mm256_sub_ps(x, t);  // exact
  const __m256 up = _mm256_cmp_ps(abs8(diff), _mm256_set1_ps(0.5f), _CMP_GE_OQ);
  const __m256 one = _mm256_or_ps(_mm256_set1_ps(1.0f),
                                  _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(int(0x80000000u)))));
  t = _mm256_blendv_ps(t, _mm256_add_ps(t, one), up);
  return _mm256_cvttps_epi32(t);
}

template <class T>
void q8_0_block(const T* x, block_q8_0* y) {
  const V16 v0 = load16(x), v1 = load16(x + 16);
  const float amax = reduce_max16(_mm256_max_ps(abs8(v0.lo), abs8(v1.lo)), _mm256_max_ps(abs8(v0.hi), abs8(v1.hi)));
  const float d = amax / ((1 << 7) - 1);
  const float id = d ? 1.0f / d : 0.0f;
  y->d = GGML_FP32_TO_FP16(d);
  const __m256 m = _mm256_set1_ps(id);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(y->qs),
                   narrow16(roundf8(_mm256_mul_ps(v0.lo, m)), roundf8(_mm256_mul_ps(v0.hi, m))));
  _mm_storeu_si128(reinterpret_cast<__m128i*>(y->qs + 16),
                   narrow16(roundf8(_mm256_mul_ps(v1.lo, m)), roundf8(_mm256_mul_ps(v1.hi, m))));
}

}  // namespace

void quantize_row_q8_K_bf16(const uint16_t* x, block_q8_K* y, int64_t k) {
  for (int64_t i = 0; i < k / QK_K; ++i) q8K_block(x + i * QK_K, y + i);
}
void quantize_row_q8_K_f32(const float* x, block_q8_K* y, int64_t k) {
  for (int64_t i = 0; i < k / QK_K; ++i) q8K_block(x + i * QK_K, y + i);
}
void quantize_row_q8_0_bf16(const uint16_t* x, block_q8_0* y, int64_t k) {
  for (int64_t i = 0; i < k / QK8_0; ++i) q8_0_block(x + i * QK8_0, y + i);
}
void quantize_row_q8_0_f32(const float* x, block_q8_0* y, int64_t k) {
  for (int64_t i = 0; i < k / QK8_0; ++i) q8_0_block(x + i * QK8_0, y + i);
}

}  // namespace ninfer::ops::host::gguf::detail::avx2

#pragma GCC pop_options
