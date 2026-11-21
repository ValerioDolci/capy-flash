#pragma once

// Two-device tensor-parallel (tp == 2) geometry of the Inkling 2 Text backbone.
//
// Rank r holds the head-split attention shards, the vocabulary half of the output head and the
// expert bank of IDs [r E/2, (r+1) E/2) (load/sharding.h); everything on the residual axis is
// replicated. A visit exchanges twice: the attention output projection's partial sums (BF16
// all-reduce) and the two banks' FP32 MoE partials (allreduce_sum_f32), so both ranks keep the
// same residual, block states, messages, router decisions and convolution states.

#include "models/inkling2/config.h"

namespace ninfer::models::inkling2::execution {

inline constexpr int kTensorParallelWidth = 2;

// The logical Text config narrowed to one rank's share: query and KV heads and the output-head
// rows are divided by `width`; hidden size, head dimensions, experts (global IDs, routed on every
// rank), schedule and every constant are unchanged. Throws std::invalid_argument for an extent
// `width` does not divide.
[[nodiscard]] TextConfig shard_text_config(const TextConfig& config, int width);

} // namespace ninfer::models::inkling2::execution
