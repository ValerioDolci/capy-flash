#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"

#include "core/device.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/fp8/fp8_a16_gemv.cuh"

#include "ops/attn_input_proj/tp2/fp8_attn_input_shard.h"
#include "ops/linear/fp8/fp8_geometry.h"

#include <cuda_bf16.h>

// One decode launcher source for the [14336,5120] parent and its [7168,5120] two-device shard (one
// instance per q|k|gate|v section output).

namespace ninfer::ops::detail {

static_assert(Fp8N7168K5120::kOutputRows == 2 * (3072 + 512));
static_assert(Fp8N7168K5120::kInputRows == Fp8N14336K5120::kInputRows);

namespace {

template <class Output>
void launch_decode(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                   Tensor& v, cudaStream_t stream) {
    using Geometry = Fp8N14336K5120;
    // Keep six CTAs resident with the four-segment output (39 registers, no spills).
    using Schedule = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 6>;
    const Output output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
    launch_fp8_a16_gemv<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a16_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
}

} // namespace

void fp8_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_decode<Fp8AttentionInputOutput>(x, weight, q, gate, k, v, stream);
}

void fp8_attn_input_shard_decode_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                        Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_decode<Fp8AttnInputShardOutput>(x, weight, q, gate, k, v, stream);
}

} // namespace ninfer::ops::detail
