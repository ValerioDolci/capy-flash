#pragma once

#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Append every K/V row to single-sequence paged growing-cache storage.
 *
 * k/v are contiguous BF16 [256,4|2,T] and positions is contiguous sequential device I32 [T].
 * BF16 cache mode copies K bit-for-bit and stores V as FP16_RNE(BF16 input). INT8-G64 cache rows
 * use one scale for each contiguous 64-value group. For codec input values x, the persistent
 * INT8 group encoding is
 *
 *   a          = max_i abs(FP32(x[i]))
 *   scale_bits = FP16_RNE(a / 127)
 *   s          = FP32(scale_bits)
 *   inv        = s == 0 ? 0 : FP32(1 / s)
 *   code[i]    = s == 0 ? 0 : I8(clamp(RNE_even(FP32(x[i]) * inv), -127, 127))
 *   decode[i]  = FP32(code[i]) * s.
 *
 * FP8_E4M3FN cache rows use one FP16 scale for the complete D256 row:
 *
 *   a          = max_i abs(FP32(x[i]))
 *   a == 0: scale_bits=FP16(+0), code[i]=E4M3FN(+0)
 *   a != 0: raw_scale  = a / 448
 *           scale_bits = FP16_RNE(clamp(raw_scale, 0x1p-24, 65504))
 *           s          = FP32(scale_bits)
 *           inv        = FP32(1 / s)
 *           code[i]    = E4M3FN_RNE_SATFINITE(FP32(x[i]) * inv)
 *   decode[i] = FP32(E4M3FN(code[i])) * s.
 *
 * NVFP4-G16 encodes a D256 row as sixteen independent groups. Each group stores sixteen E2M1
 * codes in eight U8 bytes (lower d in the low nibble) plus one UE4M3 scale byte. Its scale is
 * UE4M3_RNE_SATFINITE(clamp(absmax/6, 2^-9, 448)); an all-zero group stores zero scale and codes.
 * Codes are E2M1_RNE_SATFINITE(x/scale). There is no matrix-level divisor. Thus one represented
 * vector occupies 128 code bytes plus 16 scale bytes. K8V4 uses the existing 256-byte row-scaled
 * FP8 K code plus one FP16 scale, and the 144-byte NVFP4 representation for V.
 *
 * INT8 and FP8 apply the fixed normalized D256 Hadamard preparation to K; NVFP4 and K8V4 apply it
 * to both K and V. Every transform is evaluated in FP32 from represented BF16 source values. The
 * paired Q and output interpretation belongs to the causal softmax_attention contract. Transform
 * results and raw code/scale bytes are not standalone mathematical outputs. Standalone and fused
 * append produce byte-identical cache representations. Every addressed code/value and scale is
 * overwritten, and no unrelated cache row is read or written. Inputs and every cache plane/table
 * are pairwise non-overlapping. The Op owns no persistent allocation, frontier, request identity,
 * or commit authority.
 */
void kv_cache_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                     PagedKVLayerView cache, cudaStream_t stream);

} // namespace ninfer::ops
