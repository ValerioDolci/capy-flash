#pragma once

#include <cuda_runtime.h>

#include <vector>

namespace ninfer::ops::detail {

// RTX 5090 target, still used by the BF16, FP8, NVFP4 and K8V4 plans. Wave budgets remain owned
// by each dtype plan.
inline constexpr int kCausalAttentionSmCount = 170;

// Streaming multiprocessors of the current device (170 on an RTX 5090, 70 on an RTX 5070 Ti),
// queried once per device. At tp 2 every rank plans on its own device.
int causal_attention_sm_count();

// Distinct SM counts of the visible devices, ascending. Workspace capacity is queried before the
// launching device is known, so it takes the largest plan over these. Without a visible device
// the list is {kCausalAttentionSmCount}.
const std::vector<int>& causal_attention_device_sm_counts();

// Capture reserves partials for the largest live row. Producer and merge use
// the same live count; a wider capture never changes a row's work partition.
struct CausalKvPartition {
    static constexpr int kMaxSplits = 256;
    int capacity                    = 1;
    int target                      = 1;
    int key_shift                   = 6; // log2 of the minimum KV keys per split
    // A balanced partition gives a row of more than `target` units the fewest splits that keep
    // its longest split as short as `target` splits would: ceil(units / ceil(units / target)).
    bool balanced = false;

    __host__ __device__ int units(int visible) const {
        return (visible + (1 << key_shift) - 1) >> key_shift;
    }

    // Largest live count over rows of up to `visible` keys: the grid and partials capacity.
    __host__ __device__ int bound(int visible) const {
        const int count = units(visible);
        return count < target ? count : target;
    }

    // Units per split of the longest split, 1 while the row has no more units than `target`.
    __host__ __device__ unsigned longest(int visible) const {
        const unsigned count = static_cast<unsigned>(units(visible));
        const unsigned limit = static_cast<unsigned>(target);
        return count <= limit ? 1u : (count + limit - 1u) / limit;
    }

    __host__ __device__ int active(int visible) const {
        const int count = units(visible);
        if (count <= target) return count;
        if (!balanced) return target;
        const unsigned run = longest(visible);
        return static_cast<int>((static_cast<unsigned>(count) + run - 1u) / run);
    }
};

// Keys [begin, end) of one split of a balanced partition, begin >= end for idle capacity.
// Unsigned arithmetic: the operands are nonnegative and unsigned division needs no sign fix-up,
// which keeps these register-bound kernels free of spills.
struct CausalSplitKeys {
    int begin, end;
};

// Runs: every split owns `longest` consecutive units and the last one takes the remainder. The
// fused append kernels use it because their last split also encodes the new rows.
__host__ __device__ inline CausalSplitKeys causal_split_run(const CausalKvPartition& partition,
                                                            int visible, int split) {
    const unsigned run   = partition.longest(visible) << partition.key_shift;
    const unsigned begin = static_cast<unsigned>(split) * run;
    const unsigned end   = begin + run;
    const unsigned limit = static_cast<unsigned>(visible);
    return {static_cast<int>(begin), static_cast<int>(end < limit ? end : limit)};
}

// Shares: the balanced count of splits divides the units proportionally, so they differ by at
// most one unit. The query-parallel kernels use it: several CTAs share each SM there, and
// even shares keep the per-SM sums even.
__host__ __device__ inline CausalSplitKeys causal_split_share(const CausalKvPartition& partition,
                                                              int visible, int split) {
    const unsigned count  = static_cast<unsigned>(partition.units(visible));
    const unsigned splits = static_cast<unsigned>(partition.active(visible));
    const unsigned index  = static_cast<unsigned>(split);
    if (index >= splits) return {visible, visible};
    const unsigned begin = (index * count / splits) << partition.key_shift;
    const unsigned end   = ((index + 1u) * count / splits) << partition.key_shift;
    const unsigned limit = static_cast<unsigned>(visible);
    return {static_cast<int>(begin), static_cast<int>(end < limit ? end : limit)};
}

} // namespace ninfer::ops::detail
