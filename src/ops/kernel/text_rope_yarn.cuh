#pragma once

// Device side shared by the Text RoPE kernels of ninfer_ops: the position axis of a rotary pair,
// and the YaRN table that rope_install_text_yarn (ninfer/ops/rope.h) puts in each device's
// constant memory. The table and text_yarn_sincos live in launcher/rope.cu; the other kernels call
// it across translation units (relocatable device code).

#include <cstdint>

namespace ninfer::ops {

// Position row of rotary pair `pair` of a Text head with `axes` position rows: interleaved MRoPE
// (pair % 3) with three, the only row otherwise.
__device__ __forceinline__ int text_rope_axis(int axes, int pair) {
    return axes == 3 ? pair % 3 : 0;
}

// cos and sin of rotary pair `pair` at `position`: angle = float(position) * inv_freq[pair] in
// FP32, then both multiplied by the YaRN attention factor (the Hugging Face cos/sin of rope_type
// "yarn").
__device__ void text_yarn_sincos(std::int32_t position, int pair, float* sine, float* cosine);

} // namespace ninfer::ops
