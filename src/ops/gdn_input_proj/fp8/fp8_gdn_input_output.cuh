#pragma once
#include "ops/linear/common/output.cuh"

namespace ninfer::ops::detail {
using Fp8GdnInputOutput = LinearBf16SegmentedOutput<10240, 6144>;
// The [8192,5120] two-device shard's qkv|z section output (tp2/fp8_gdn_input_shard.h).
using Fp8GdnInputShardOutput = LinearBf16SegmentedOutput<5120, 3072>;
} // namespace ninfer::ops::detail
