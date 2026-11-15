#pragma once

// Routes for q3_g128_fp16 row-split weights (Qwen3.8-27B 3-bit GSQ).
//
// One kernel family serves every Text site of the Qwen3.5 dense block:
//   T <= 8 : register-direct tensor-core route (A16);
//   T  > 8 : 128-row x 64-token tiles on mma.sync m16n8k16 BF16 tensor cores. The 3-bit codes are
//            expanded to exact BF16 integers in shared memory, one 128-wide K step per group, and
//            each group's binary16 multiplier scales its FP32 partial before accumulation;
//   T >= 64: by default the A8 prefill route (activations quantized per token and 128-value group
//            to int8, int8 tensor cores, exact int32 group sums scaled in FP32); NINFER_Q3_A8=0
//            keeps the A16 tiles for every T > 8.
// Accumulation is FP32; the output is rounded to BF16 once (after the residual add / SwiGLU).
// Epilogues: row segments (plain Linear, the attention Q|K|Gate|V and GDN QKV|Z splits, the GDN
// conv-compose flat planes), residual add, and SwiGLU over a [gate; up] parent.

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Device scratch of the A8 prefill route (int8 activation codes and their scales). With a null or
// too small workspace the A8 route falls back to the in-kernel quantizing tile: same bits, slower.
struct Q3Workspace {
    void* data        = nullptr; // 256-byte aligned
    std::size_t bytes = 0;
};

// Workspace bytes the A8 route needs for up to `max_tokens` tokens of K = `k` (0 when the A8
// route is off or max_tokens < 64). The size grows with T, so the bound for max_tokens covers
// every smaller call.
std::size_t q3_workspace_bytes(std::int32_t k, std::int32_t max_tokens);

// Test hook (not thread-safe): 0 = automatic (environment), 1 = A16 only, 2 = A8 with in-kernel
// quantization, 3 = A8 with the tall workspace route.
void q3_force_route_for_test(int route);

// Up to four destinations for consecutive row ranges of one parent: row r with
// end[s-1] <= r < end[s] is stored at data[s][token * ld[s] + (r - end[s-1])].
struct Q3RowSegments {
    static constexpr int kMax = 4;
    __nv_bfloat16* data[kMax]{};
    std::int64_t ld[kMax]{};
    std::int32_t end[kMax]{};
    std::int32_t count = 0;
};

// Throws unless `w` is a usable q3_g128_fp16 RowSplit weight (K a multiple of 128).
void validate_q3_weight(const Weight& w, const char* label);

// out[N,T] = W x.
void q3_linear_dispatch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream,
                        const Q3Workspace& ws = {});

// Rows of W x scattered into `segments` (its last end must equal N).
void q3_linear_segments_dispatch(const Tensor& x, const Weight& w, const Q3RowSegments& segments,
                                 cudaStream_t stream, const Q3Workspace& ws = {});

// residual[N,T] = bf16(residual + W x), one rounding.
void q3_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream,
                            const Q3Workspace& ws = {});

// out[N/2,T] = bf16(silu(W[0:N/2] x) * (W[N/2:N] x)), one rounding.
void q3_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream,
                               const Q3Workspace& ws = {});

} // namespace ninfer::ops::detail
