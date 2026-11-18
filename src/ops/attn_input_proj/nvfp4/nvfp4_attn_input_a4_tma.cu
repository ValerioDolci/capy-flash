#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_a4_tma_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"
#include "ops/attn_input_proj/tp2/nvfp4_attn_input_shard.h"

namespace ninfer::ops::detail {
namespace {
template <class Output>
void launch_tma(const Nvfp4A4Operands& p, __nv_bfloat16* query, __nv_bfloat16* gate,
                __nv_bfloat16* key, __nv_bfloat16* value, cudaStream_t stream) {
    const Output output{query, key, gate, value};
    if (p.scale_layout == Nvfp4ScaleLayout::Tiled128)
        launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<128, 4, 1>, 5120>>(
            p, output, LinearIdentityEpilogue{}, stream);
    else
        launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1>, 5120>>(
            p, output, LinearIdentityEpilogue{}, stream);
}
} // namespace

void launch_nvfp4_a4_tma_attention(const Nvfp4A4Operands& p, __nv_bfloat16* query,
                                   __nv_bfloat16* gate, __nv_bfloat16* key, __nv_bfloat16* value,
                                   cudaStream_t stream) {
    launch_tma<Nvfp4AttentionInputOutput>(p, query, gate, key, value, stream);
}

void launch_nvfp4_a4_tma_attention_shard(const Nvfp4A4Operands& p, __nv_bfloat16* query,
                                         __nv_bfloat16* gate, __nv_bfloat16* key,
                                         __nv_bfloat16* value, cudaStream_t stream) {
    launch_tma<Nvfp4AttnInputShardOutput>(p, query, gate, key, value, stream);
}
} // namespace ninfer::ops::detail
