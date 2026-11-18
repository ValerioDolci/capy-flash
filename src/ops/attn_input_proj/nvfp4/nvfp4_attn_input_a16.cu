#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_a16.cuh"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"

namespace ninfer::ops::detail {

void nvfp4_attn_input_a16_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                 Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_nvfp4_attn_input_a16<Nvfp4AttentionInputOutput, nvfp4_attn_input_decode_launch,
                                nvfp4_attn_input_small_t_launch>(x, weight, q, gate, k, v, stream);
}

} // namespace ninfer::ops::detail
