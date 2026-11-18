// The [8192,5120] two-device shard's instance of the parent's A16 launcher
// (nvfp4/nvfp4_gdn_input_a16.cuh), at the shard's section output.
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_a16.cuh"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/gdn_input_proj/tp2/nvfp4_gdn_input_shard.h"
#include "ops/linear/nvfp4/nvfp4_geometry.h"

namespace ninfer::ops::detail {

static_assert(Nvfp4N8192K5120::kOutputRows == 5120 + 3072);
static_assert(Nvfp4N8192K5120::kInputRows == Nvfp4N16384K5120::kInputRows);
static_assert((5120 % 128) == 0 && (3072 % 128) == 0);

void nvfp4_gdn_input_shard_a16_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                      cudaStream_t stream) {
    launch_nvfp4_gdn_input_a16<Nvfp4GdnInputShardOutput, nvfp4_gdn_input_shard_decode_launch,
                               nvfp4_gdn_input_shard_small_t_launch>(x, weight, qkv, z, stream);
}

} // namespace ninfer::ops::detail
