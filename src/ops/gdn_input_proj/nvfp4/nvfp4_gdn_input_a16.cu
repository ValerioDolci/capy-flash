#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_a16.cuh"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"

namespace ninfer::ops::detail {

void nvfp4_gdn_input_a16_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream) {
    launch_nvfp4_gdn_input_a16<Nvfp4GdnInputOutput, nvfp4_gdn_input_decode_launch,
                               nvfp4_gdn_input_small_t_launch>(x, weight, qkv, z, stream);
}

} // namespace ninfer::ops::detail
