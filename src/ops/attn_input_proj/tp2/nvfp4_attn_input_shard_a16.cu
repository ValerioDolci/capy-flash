// The [7168,5120] two-device shard's instance of the parent's A16 launcher
// (nvfp4/nvfp4_attn_input_a16.cuh), at the shard's section output.
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_a16.cuh"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"
#include "ops/attn_input_proj/tp2/nvfp4_attn_input_shard.h"
#include "ops/linear/nvfp4/nvfp4_geometry.h"

namespace ninfer::ops::detail {

static_assert(Nvfp4N7168K5120::kOutputRows == 2 * (3072 + 512));
static_assert(Nvfp4N7168K5120::kInputRows == Nvfp4N14336K5120::kInputRows);
// Every row tile of the parent's routes (at most the 128-row A4 MMA and TMA tiles) divides each
// shard section, so a tile binds to one section and vector stores never straddle two.
static_assert((3072 % 128) == 0 && (512 % 128) == 0);

void nvfp4_attn_input_shard_a16_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                       Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_nvfp4_attn_input_a16<Nvfp4AttnInputShardOutput, nvfp4_attn_input_shard_decode_launch,
                                nvfp4_attn_input_shard_small_t_launch>(x, weight, q, gate, k, v,
                                                                       stream);
}

} // namespace ninfer::ops::detail
