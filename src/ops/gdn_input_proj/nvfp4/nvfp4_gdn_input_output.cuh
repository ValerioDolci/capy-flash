#pragma once
#include "ops/linear/common/output.cuh"

namespace ninfer::ops::detail {
using Nvfp4GdnInputOutput = LinearBf16SegmentedOutput<10240, 6144>;
// The [8192,5120] two-device shard's qkv|z section output (tp2/nvfp4_gdn_input_shard.h). Both
// sections are whole 128-row tiles, like the parent's.
using Nvfp4GdnInputShardOutput = LinearBf16SegmentedOutput<5120, 3072>;
} // namespace ninfer::ops::detail
