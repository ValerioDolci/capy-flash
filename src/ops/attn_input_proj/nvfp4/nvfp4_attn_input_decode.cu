#include "core/weight.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"

#include "ops/attn_input_proj/tp2/nvfp4_attn_input_shard.h"

#include <cuda_bf16.h>

// One decode launcher source for the [14336,5120] parent and its [7168,5120] two-device shard (one
// instance per q|k|gate|v section output).

namespace ninfer::ops::detail {
namespace {

template <class Output>
void launch_decode(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                   Tensor& v, cudaStream_t stream) {
    using Geometry = Nvfp4N14336K5120;
    using Schedule =
        Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
    static_assert((6144 % 128) == 0);
    static_assert((1024 % 128) == 0);

    const Output output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
    launch_nvfp4_a16_gemv<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a16_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
}

} // namespace

void nvfp4_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_decode<Nvfp4AttentionInputOutput>(x, weight, q, gate, k, v, stream);
}

void nvfp4_attn_input_shard_decode_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                          Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_decode<Nvfp4AttnInputShardOutput>(x, weight, q, gate, k, v, stream);
}

} // namespace ninfer::ops::detail
