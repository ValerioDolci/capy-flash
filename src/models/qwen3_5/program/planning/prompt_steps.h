#pragma once

#include <algorithm>
#include <cstdint>

namespace ninfer::models::qwen3_5::runtime_support {

// The tokens of the next prefill step, `segment` tokens before the next capture frontier (or the
// end of the prompt). The scheduler projects one service quantum per `prefill_chunk` tokens of
// each such segment (request planning) and every step consumes one, so a step never covers less
// than its share of that grid: a segment takes at most 1 + (segment - 1) / prefill_chunk steps.
// The forward may still run a step as several narrower chunks (an expert cache that lends less
// than a chunk now caps each one); the step loop goes on until the step's tokens are done.
//
// `equal_widths` (routed experts whose cache lends a whole chunk now): every chunk pays one expert
// pass whatever its width (the streamed experts cross PCIe once per chunk), so a segment of several
// chunks runs them at equal widths (whole 128-column tiles) instead of full chunks and a short
// tail: 53 930 tokens are four chunks of 13 354 to 13 568, not three of 16 384 and one of 4 778. A
// segment of one chunk is unchanged. Without it the step is a full chunk: models without routed
// experts keep the full-chunk grid (their chunk grid restarts at the rewrite frontiers so that
// resumed and root GDN decompositions agree, whatever the prompt length after the frontier), and
// a cache that lends less than a chunk splits each full step into its own widths, which costs
// the expert passes of full chunks rather than one more per step.
[[nodiscard]] inline std::uint32_t
prompt_step_tokens(std::uint32_t segment, std::uint32_t prefill_chunk, bool equal_widths) noexcept {
    if (!equal_widths || segment <= prefill_chunk) { return std::min(prefill_chunk, segment); }
    const std::uint32_t chunks = (segment - 1) / prefill_chunk + 1;
    const std::uint32_t even   = ((segment - 1) / chunks + 1 + 127U) / 128U * 128U;
    return std::min(prefill_chunk, even);
}

} // namespace ninfer::models::qwen3_5::runtime_support
