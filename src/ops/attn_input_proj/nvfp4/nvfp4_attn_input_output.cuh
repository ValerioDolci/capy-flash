#pragma once
#include "ops/linear/common/output.cuh"

namespace ninfer::ops::detail {
using Nvfp4AttentionInputOutput = LinearBf16SegmentedOutput<6144, 1024, 6144, 1024>;
// The [7168,5120] two-device shard's query|key|gate|value section output, in the parent's section
// order (tp2/nvfp4_attn_input_shard.h).
using Nvfp4AttnInputShardOutput = LinearBf16SegmentedOutput<3072, 512, 3072, 512>;
} // namespace ninfer::ops::detail
