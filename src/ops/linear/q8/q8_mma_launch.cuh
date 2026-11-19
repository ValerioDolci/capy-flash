#pragma once
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q8/q8_a16_mma.cuh"

namespace ninfer::ops::detail {
template <class Schedule, class RowPolicy = Q8MmaIdentityRows<Schedule>, class Output,
          class Epilogue>
void launch_q8_a16_mma(const Q8LinearOperands& operands, Output output, Epilogue epilogue,
                       cudaStream_t stream, RowPolicy row_policy = {}, bool f16 = false) {
    validate_q8_operands(operands);
    if (operands.k % 8 != 0 || operands.padded_k % Schedule::kBlockK != 0)
        throw std::invalid_argument("Q8 MMA requires K aligned to 8 and complete padded K tiles");
    static_assert(Schedule::kBlockRows % RowPolicy::kOutputRowsPerCta == 0);
    constexpr int row_ratio = Schedule::kBlockRows / RowPolicy::kOutputRowsPerCta;
    if (operands.rows % row_ratio != 0)
        throw std::invalid_argument("Q8 MMA row mapping requires complete row pairs");
    for_each_token_slice(operands.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(div_up(operands.rows / row_ratio, RowPolicy::kOutputRowsPerCta),
                        div_up(count, Schedule::kBlockTokens));
        const auto launch_k = [&]<bool Full, bool FullK, bool F16>() {
            constexpr auto kernel =
                q8_a16_mma_kernel<Schedule, Full, FullK, Output, Epilogue, RowPolicy, F16>;
            const int shared = q8_prepare_shared<q8_mma_shared_bytes<Schedule, Epilogue>, kernel>();
            kernel<<<grid, Schedule::kThreads, shared, stream>>>(operands, output, epilogue,
                                                                 row_policy, offset, count);
            CUDA_CHECK(cudaGetLastError());
        };
        const auto launch = [&]<bool Full, bool FullK>() {
            if constexpr (Schedule::kActivationStages == Schedule::kStages) {
                if (f16) {
                    launch_k.template operator()<Full, FullK, true>();
                    return;
                }
            }
            if (f16) { throw std::invalid_argument("Q8 MMA F16 needs two activation stages"); }
            launch_k.template operator()<Full, FullK, false>();
        };
        if (!Schedule::kPredicated && operands.rows % Schedule::kBlockRows == 0 &&
            count % Schedule::kBlockTokens == 0 && operands.k == operands.padded_k &&
            operands.padded_k % 256 == 0)
            launch.template operator()<true, true>();
        else if (operands.k == operands.padded_k)
            launch.template operator()<false, true>();
        else
            launch.template operator()<false, false>();
    });
}
} // namespace ninfer::ops::detail
