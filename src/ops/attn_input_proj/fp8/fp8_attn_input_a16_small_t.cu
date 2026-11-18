#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "ops/linear/fp8/fp8_instances.cuh"

#include "ops/attn_input_proj/tp2/fp8_attn_input_shard.h"

// One small-T MMA launcher source for the [14336,5120] parent and its [7168,5120] two-device shard
// (one instance per q|k|gate|v section output).

namespace ninfer::ops::detail {
namespace {

template <class Output>
void launch_small_mma(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                      Tensor& v, cudaStream_t stream) {
    const Output output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(gate.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    const auto launch = [&]<class Schedule>() {
        launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Schedule, 5120>>(
            fp8_a16_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    };
    if (x.ne[1] <= 16) return launch.template operator()<Fp8SlicedInstance<16, 4, 1>>();
    if (x.ne[1] <= 24) return launch.template operator()<Fp8SlicedInstance<32, 4, 2>>();
    launch.template operator()<Fp8SlicedInstance<32, 4, 1>>();
}

} // namespace

void fp8_attn_input_a16_small_mma_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                         Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_small_mma<Fp8AttentionInputOutput>(x, weight, q, gate, k, v, stream);
}

void fp8_attn_input_shard_a16_small_mma_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                               Tensor& gate, Tensor& k, Tensor& v,
                                               cudaStream_t stream) {
    launch_small_mma<Fp8AttnInputShardOutput>(x, weight, q, gate, k, v, stream);
}

} // namespace ninfer::ops::detail
