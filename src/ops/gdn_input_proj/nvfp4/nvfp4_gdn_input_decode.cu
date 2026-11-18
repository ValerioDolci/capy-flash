#include "core/weight.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"

#include "ops/gdn_input_proj/tp2/nvfp4_gdn_input_shard.h"

// One decode launcher source for the [16384,5120] parent and its [8192,5120] two-device shard (one
// instance per qkv|z section output).

namespace ninfer::ops::detail {
namespace {

template <class Output>
void launch_decode(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                   cudaStream_t stream) {
    using Geometry = Nvfp4N16384K5120;
    using Schedule =
        Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;

    launch_nvfp4_a16_gemv<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a16_operands(x, weight),
        Output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)},
        LinearIdentityEpilogue{}, stream);
}

} // namespace

void nvfp4_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   cudaStream_t stream) {
    launch_decode<Nvfp4GdnInputOutput>(x, weight, qkv, z, stream);
}

void nvfp4_gdn_input_shard_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                         Tensor& z, cudaStream_t stream) {
    launch_decode<Nvfp4GdnInputShardOutput>(x, weight, qkv, z, stream);
}

} // namespace ninfer::ops::detail
