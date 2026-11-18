#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"
#include "ops/linear/fp8/fp8_a16_mma.cuh"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/common/token_slices.h"
#include "core/device.h"
#include "ops/attn_input_proj/tp2/fp8_attn_input_shard.h"

// One A16 GEMM launcher source for the [14336,5120] parent and its [7168,5120] two-device shard:
// they share K, so the column tiles are the same; only the q|k|gate|v section output differs.

namespace ninfer::ops::detail {
namespace {
template <class Output, class S>
void run(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
         cudaStream_t stream) {
    const Output output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(gate.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    launch_fp8_a16_mma<Fp8ScheduleInstance<S, 5120>>(fp8_a16_operands(x, weight), output,
                                                     LinearIdentityEpilogue{}, stream);
}

template <class Output>
void launch_gemm(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                 Tensor& v, cudaStream_t stream) {
    // Column tiles follow the measured whole-Op envelope, including the 129..160 tail.
    if (x.ne[1] <= 64)
        run<Output, Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>(x, weight, q, gate, k, v, stream);
    else if (x.ne[1] <= 80 || (x.ne[1] >= 129 && x.ne[1] <= 160))
        run<Output, Fp8A16MmaSchedule<32, 80, 128, 32, 16, 1, 3>>(x, weight, q, gate, k, v, stream);
    else if (x.ne[1] <= 96)
        run<Output, Fp8A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>(x, weight, q, gate, k, v, stream);
    else
        run<Output, Fp8A16MmaSchedule<64, 128, 64, 32, 16, 2, 2>>(x, weight, q, gate, k, v, stream);
}

} // namespace

void fp8_attn_input_a16_gemm_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_gemm<Fp8AttentionInputOutput>(x, weight, q, gate, k, v, stream);
}

void fp8_attn_input_shard_a16_gemm_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                          Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_gemm<Fp8AttnInputShardOutput>(x, weight, q, gate, k, v, stream);
}
} // namespace ninfer::ops::detail
