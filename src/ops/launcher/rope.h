#pragma once

// ninfer::ops::detail - private launch prototype for rope. Included by the wrapper
// and defined by the CUDA launcher.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void rope_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                 cudaStream_t stream);

void rope_single_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                        cudaStream_t stream);

// The YaRN table of rope_install_text_yarn: inverse frequency of each rotary pair and the
// factor on cos and sin.
struct TextRopeYarn {
    float inv_freq[32];
    float attention_scale;
};

// The installed YaRN table when Text RoPE of this rotary_dim and theta takes it, else null.
// Throws when it is installed for this geometry but not on the current device.
const TextRopeYarn* text_rope_yarn(int rotary_dim, float theta);

} // namespace ninfer::ops::detail
