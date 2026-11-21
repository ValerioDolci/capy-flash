#pragma once

// ninfer::ops::detail - the wide (prefill) route of message_gate.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Calls of at least this many tokens take the wide route when their shapes fit it (H 2304, R 64,
// RTX 5070 Ti: the cluster route is faster at T = 64, the wide one from T = 128).
inline constexpr int kMessageGateWideMinTokens = 128;

// Launches the wide route at any T and returns true when the shapes fit it (false: nothing is
// enqueued; message_gate_launch keeps the call). Same bits as the cluster and token routes.
[[nodiscard]] bool message_gate_wide(const Tensor& hidden, const Tensor& message,
                                     const Tensor& down, const Tensor& forget, const Tensor& gain,
                                     float eps, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
