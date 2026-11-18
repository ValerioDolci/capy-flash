// Modified for capy-flash from fngine:src/cpu/q8_cpu.h @ c20a885 (satellitedown/fngine, Apache-2.0).
// See NOTICE. capy-flash changes: namespace ninfer::ops::host::gguf, include paths, run-time ISA
// dispatch (host_gguf_isa(); AVX-512 in host_q8.cpp, AVX2 in host_q8_avx2.cpp).
#pragma once

// Q8_K / Q8_0 activation quantizers for the CPU expert path (AVX-512 or AVX2, same bytes).
//
// Results are bit-identical to ggml's reference quantizers (`quantize_row_q8_K_ref`,
// `quantize_row_q8_0_ref` in third_party/ggml-iq) applied to the same FP32 values. BF16 inputs
// are widened exactly to FP32 first. One deliberate difference: an all-zero Q8_K block gets
// bsums = 0 (the reference leaves bsums untouched for such blocks).
#include "ggml-iq/ggml-iq.h"

#include <cstdint>

namespace ninfer::ops::host::gguf {

// k % 256 == 0. x: raw BF16 bits.
void quantize_row_q8_K_bf16(const uint16_t* x, block_q8_K* y, int64_t k);
void quantize_row_q8_K_f32(const float* x, block_q8_K* y, int64_t k);

// k % 32 == 0.
void quantize_row_q8_0_bf16(const uint16_t* x, block_q8_0* y, int64_t k);
void quantize_row_q8_0_f32(const float* x, block_q8_0* y, int64_t k);

}  // namespace ninfer::ops::host::gguf
