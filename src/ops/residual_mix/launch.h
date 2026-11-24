#pragma once

// ninfer::ops::detail - private launch prototypes for residual_mix.h.

#include "ninfer/ops/depth_attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// The short convolution of a residual branch and, for residual_conv_mix, its residual addition.
struct ResidualConvArgs {
    const __nv_bfloat16* x    = nullptr;
    const float* weight       = nullptr;
    float scale               = 1.0F;
    __nv_bfloat16* state      = nullptr; // advanced in place when record is null
    __nv_bfloat16* record     = nullptr;
    const __nv_bfloat16* base = nullptr; // residual_conv_mix: the BF16 state added, or null
    float* residual           = nullptr; // residual_conv_mix: the FP32 sum
};

// The depth attention and the RMS norms of its first `norms` queries.
struct MixArgs {
    const __nv_bfloat16* state[kDepthAttentionMaxStates] = {};
    const __nv_bfloat16* current                         = nullptr; // depth_attention_rmsnorm only
    const float* current_f32                             = nullptr; // or an FP32 current
    const float* query                                   = nullptr;
    int queries                                          = 1;
    float eps                                            = 0.0F;
    float scale                                          = 0.0F;
    float* mix                                           = nullptr; // FP32 {H, T, Q} or null
    int norms                                            = 0;
    const __nv_bfloat16* norm[kDepthAttentionMaxQueries] = {};
    __nv_bfloat16* normed[kDepthAttentionMaxQueries]     = {};
    float norm_eps                                       = 0.0F;
};

void residual_conv_mix_launch(const ResidualConvArgs& c, const MixArgs& m, int sources, int hidden,
                              int tokens, cudaStream_t stream);
void depth_attention_rmsnorm_launch(const MixArgs& m, int sources, int hidden, int tokens,
                                    cudaStream_t stream);
void residual_conv_round_launch(const ResidualConvArgs& c, const float* residual,
                                __nv_bfloat16* out, int hidden, int tokens, cudaStream_t stream);

} // namespace ninfer::ops::detail
