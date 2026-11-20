#pragma once
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q8/q8_a16_warp_row.cuh"
#include "ops/linear/q8/q8_operands.h"

#include <stdexcept>

namespace ninfer::ops::detail {
template <class Schedule, class Output, class Epilogue>
void launch_q8_a16_warp_row(const Q8LinearOperands& operands, Output output, Epilogue epilogue,
                            cudaStream_t stream) {
    validate_q8_operands(operands);
    constexpr int W = Schedule::kWarpsPerRow;
    const int pairs = operands.k / 64;
    if (operands.k % 64 != 0 || (pairs + W - 1) / W * 2 > Schedule::kPhases * 8)
        throw std::invalid_argument("Q8 warp-row schedule does not cover K");
    if (operands.tokens > Schedule::kTokenCapacity)
        throw std::invalid_argument("Q8 warp-row schedule exceeds its token capacity");
    const bool full_slice = pairs % W == 0 && pairs / W * 2 == Schedule::kPhases * 8;
    const dim3 grid(div_up(operands.rows, Schedule::kBlockRows));
    const auto launch = [&]<int Tokens, bool FullSlice>() {
        q8_a16_warp_row_kernel<Schedule, Tokens, FullSlice, Output, Epilogue>
            <<<grid, Schedule::kThreads, 0, stream>>>(operands.x, operands.codes, operands.scales,
                                                      output, epilogue, operands.rows, operands.k,
                                                      operands.padded_k);
        CUDA_CHECK(cudaGetLastError());
    };
    const auto launch_tokens = [&]<int Tokens>() {
        if constexpr (Tokens <= Schedule::kTokenCapacity) {
            if (full_slice)
                launch.template operator()<Tokens, true>();
            else
                launch.template operator()<Tokens, false>();
        }
    };
    switch (operands.tokens) {
    case 1:
        launch_tokens.template operator()<1>();
        break;
    case 2:
        launch_tokens.template operator()<2>();
        break;
    case 3:
        launch_tokens.template operator()<3>();
        break;
    case 4:
        launch_tokens.template operator()<4>();
        break;
    case 5:
        launch_tokens.template operator()<5>();
        break;
    case 6:
        launch_tokens.template operator()<6>();
        break;
    case 7:
        launch_tokens.template operator()<7>();
        break;
    default:
        launch_tokens.template operator()<8>();
        break;
    }
}
} // namespace ninfer::ops::detail
