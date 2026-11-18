#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_a4_tma_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/gdn_input_proj/tp2/nvfp4_gdn_input_shard.h"

namespace ninfer::ops::detail {
namespace {
template <class Output>
void launch_tma(const Nvfp4A4Operands& p, __nv_bfloat16* qkv, __nv_bfloat16* z,
                cudaStream_t stream) {
    launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1>, 5120>>(
        p, Output{qkv, z}, LinearIdentityEpilogue{}, stream);
}
} // namespace

void launch_nvfp4_a4_tma_gdn(const Nvfp4A4Operands& p, __nv_bfloat16* qkv, __nv_bfloat16* z,
                             cudaStream_t stream) {
    launch_tma<Nvfp4GdnInputOutput>(p, qkv, z, stream);
}

void launch_nvfp4_a4_tma_gdn_shard(const Nvfp4A4Operands& p, __nv_bfloat16* qkv, __nv_bfloat16* z,
                                   cudaStream_t stream) {
    launch_tma<Nvfp4GdnInputShardOutput>(p, qkv, z, stream);
}
} // namespace ninfer::ops::detail
