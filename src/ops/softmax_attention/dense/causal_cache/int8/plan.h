#pragma once

#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/common/causal_partition.h"

namespace ninfer::ops::detail {

enum class Int8KvFamily { Grouped, ParallelGrouped, Tiled };

struct Int8KvCausalPlan {
    static constexpr int kTokenTile = 8;
    Int8KvFamily family;
    int query_heads, width, batch;
    CausalAttentionExecutionEnvelope envelope;
    CausalKvPartition partition;
};

// sms is the launching device's SM count (causal_attention_sm_count()).
// NINFER_ATTN_DEC_PIPE=1 (grouped_pipe_mma.cuh): the pipelined kernel serves a grouped plan of
// `heads` and `width` (rows filling three 16-row tiles at group size 6).
bool int8_kv_dec_pipe_enabled(int heads, int width);

Int8KvCausalPlan make_int8_kv_causal_plan(int heads, int width, int batch,
                                          CausalAttentionExecutionEnvelope envelope, int sms);
std::size_t int8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                    CausalAttentionExecutionEnvelope envelope);

} // namespace ninfer::ops::detail
