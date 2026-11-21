#pragma once

// ninfer::ops::detail - private launch prototype for message_gate.

#include "core/tensor.h"
#include "ninfer/ops/message_gate.h"

#include <cuda_bf16.h>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// One gate of a message_gates_rmsnorm launch.
struct MessageGateBatchItem {
    const float* hidden       = nullptr;
    const float* message      = nullptr;
    const __nv_bfloat16* down = nullptr;
    const float* forget       = nullptr;
    const float* gain         = nullptr;
    float* out                = nullptr;
    const __nv_bfloat16* norm = nullptr;
    __nv_bfloat16* normed     = nullptr;
};

struct MessageGateBatch {
    MessageGateBatchItem gates[kMessageGateMaxBatch];
};

// The batch on the cluster route (the shapes and alignment of gate 0's tensors stand for all);
// false, launching nothing, when the shape is outside it.
bool message_gate_batch_launch(const MessageGateBatch& batch, int gates, const Tensor& hidden,
                               const Tensor& message, const Tensor& down, const Tensor& forget,
                               const Tensor& gain, const Tensor& out, float eps, float norm_eps,
                               cudaStream_t stream);

void message_gate_launch(const Tensor& hidden, const Tensor& message, const Tensor& down,
                         const Tensor& forget, const Tensor& gain, float eps, Tensor& out,
                         cudaStream_t stream);

} // namespace ninfer::ops::detail
