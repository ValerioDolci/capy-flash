#pragma once

#include "ninfer/ops/relative_sink_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Upper bound on the chunks of one query's visible keys (see the kernels' decomposition).
inline constexpr int kRelativeSinkMaxChunks = 64;
// Calls with at most this many tokens per row spread a query's chunks over parallel blocks and
// keep per-chunk partials in workspace; longer calls walk the chunks inside one block.
inline constexpr int kRelativeSinkSplitMaxTokens = 16;

/** Raw operands shared by the prologue, attention and append stages of one call. */
struct RelativeSinkOperands {
    const __nv_bfloat16* q        = nullptr;
    const __nv_bfloat16* k        = nullptr;
    const __nv_bfloat16* v        = nullptr;
    const __nv_bfloat16* r        = nullptr;
    const __nv_bfloat16* sink     = nullptr;
    const __nv_bfloat16* gate     = nullptr; // nullptr: no output gate
    const __nv_bfloat16* q_scale  = nullptr;
    const __nv_bfloat16* r_scale  = nullptr;
    const __nv_bfloat16* rel_proj = nullptr;
    const std::int32_t* positions = nullptr;
    const std::int32_t* rows      = nullptr;
    __nv_bfloat16* out            = nullptr;
    __nv_bfloat16* q_prepared     = nullptr;
    __nv_bfloat16* r_prepared     = nullptr;
    __nv_bfloat16* k_prepared     = nullptr;
    __half* v_prepared            = nullptr;
    float* chunk_max              = nullptr; // [T*B, Hq, kRelativeSinkMaxChunks]
    float* chunk_sum              = nullptr; // [T*B, Hq, kRelativeSinkMaxChunks]
    float* chunk_out              = nullptr; // [T*B, Hq, kRelativeSinkMaxChunks, D]
    std::int32_t head_dim         = 0;
    std::int32_t query_heads      = 0;
    std::int32_t kv_heads         = 0;
    std::int32_t tokens           = 0;
    std::int32_t batch            = 0;
    RelativeSinkAttentionParams params;
};

/** Cyclic cache addressing in elements; position x lives at slot x mod capacity. */
struct RelativeSinkRing {
    __nv_bfloat16* k         = nullptr;
    __half* v                = nullptr;
    std::int64_t k_stride[3] = {}; // slot, head, lane
    std::int64_t v_stride[3] = {};
    std::int32_t capacity    = 0;
    std::int32_t window      = 0;
};

/** Page-major paged cache addressing in elements. */
struct RelativeSinkPages {
    __nv_bfloat16* k                = nullptr;
    __half* v                       = nullptr;
    const std::int32_t* block_table = nullptr;
    std::int64_t k_stride[3]        = {}; // page offset, head, page
    std::int64_t v_stride[3]        = {};
    std::int64_t table_stride[2]    = {}; // logical page, row
};

void relative_sink_attention_launch(const RelativeSinkOperands& operands,
                                    const RelativeSinkRing& ring, cudaStream_t stream);
void relative_sink_attention_launch(const RelativeSinkOperands& operands,
                                    const RelativeSinkPages& pages, cudaStream_t stream);

} // namespace ninfer::ops::detail
