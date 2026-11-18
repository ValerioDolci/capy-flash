#pragma once

// The column slices of a tp 2 prefill chunk whose row-parallel all-reduces are pipelined slice by
// slice (TextForward::run_plain, execution/text_forward/plain.h), and the split of the half-chunk
// overlap (prefill_overlap_split). Header-only and free of CUDA, so the plan is unit-tested on the
// host (tests/test_tp2_prefill_slices.cpp).
//
// The sliced schedule keeps every mixer core (norms, input projections, attention, GDN) on the
// whole chunk and cuts only the column-local tail of each block: the output projection with its
// all-reduce, the post-attention norm and the FFN. A slice is therefore free of the half-chunk
// overlap's limits (both halves > 256 columns for the attention family, no multimodal RoPE), so
// it serves chunks under 768 columns, multimodal chunks and batched-feature taps. The plan:
//
//   width  128 * max(1, len / 512) (about four slices, as kurcontko/ninfer-tp2 d0925e24), raised
//          by 128 until at most kMaxCount slices;
//   slices [s * width, (s + 1) * width), the last one takes the rest; a rest under kMinTail
//          columns joins the slice before it, so no slice is ever narrow enough to change the
//          linear route family (the A16 routes, whose K partition differs, start under 17);
//   count  >= 2, else the plan is empty (count 0) and the chunk runs as one call.

#include <cstdint>

namespace ninfer::models::qwen3_5::execution {

inline constexpr std::int32_t kPrefillSliceAlign    = 128;
inline constexpr std::int32_t kPrefillSliceMaxCount = 8; // ops::HostExchange::kSumLanes
inline constexpr std::int32_t kPrefillSliceMinTail  = 64;

struct PrefillSlicePlan {
    std::int32_t width = 0;
    std::int32_t count = 0;

    [[nodiscard]] constexpr bool usable() const noexcept { return count >= 2; }

    [[nodiscard]] constexpr std::int32_t first(std::int32_t slice) const noexcept {
        return slice * width;
    }

    [[nodiscard]] constexpr std::int32_t columns(std::int32_t slice,
                                                 std::int32_t len) const noexcept {
        return slice + 1 < count ? width : len - slice * width;
    }
};

[[nodiscard]] constexpr PrefillSlicePlan prefill_slice_plan(std::int32_t len) noexcept {
    PrefillSlicePlan plan;
    if (len < 2 * kPrefillSliceAlign) { return plan; }
    const std::int32_t quarters = len / (4 * kPrefillSliceAlign);
    std::int32_t width          = kPrefillSliceAlign * (quarters > 1 ? quarters : 1);
    std::int32_t count          = 0;
    for (;; width += kPrefillSliceAlign) {
        const std::int32_t whole = len / width;
        const std::int32_t rest  = len - whole * width;
        count                    = whole + (rest >= kPrefillSliceMinTail ? 1 : 0);
        if (count <= kPrefillSliceMaxCount) { break; }
    }
    if (count < 2) { return plan; }
    plan.width = width;
    plan.count = count;
    return plan;
}

// The half-chunk overlap (TextForward::run_plain): the width of the first half of a `len`-column
// text chunk, or 0 to run the chunk without it. A multiple of 128, so the GDN chunk (16 tokens),
// the attention query tiles and the activation-scale tiles fall on the same column boundaries as in
// one call over the chunk; both halves wider than 256 columns, so their causal attention takes the
// tiled family a whole chunk takes (the grouped family serves widths up to 256).
inline int prefill_overlap_split(int len) {
    constexpr int kAlign = 128, kMinHalf = 384;
    const int split = len / 2 / kAlign * kAlign;
    return split >= kMinHalf && len - split >= kMinHalf ? split : 0;
}

} // namespace ninfer::models::qwen3_5::execution
