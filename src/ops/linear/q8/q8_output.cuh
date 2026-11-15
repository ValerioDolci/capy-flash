#pragma once
// The Q8 launchers write BF16 (default) or FP32 out, from out.dtype.
#include "core/tensor.h"
#include "ops/linear/common/output.cuh"

namespace ninfer::ops::detail {
template <class Launch>
void q8_with_output(Tensor& out, std::int32_t rows, Launch&& launch) {
    if (out.dtype == DType::FP32) {
        launch(LinearF32Output{static_cast<float*>(out.data), rows});
    } else {
        launch(LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), rows});
    }
}
} // namespace ninfer::ops::detail
