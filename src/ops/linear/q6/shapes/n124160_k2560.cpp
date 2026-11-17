#include "ops/linear/q6/q6_shapes.h"

namespace ninfer::ops::detail {

// Qwen4Exp at tp 2: one rank's vocabulary half of the Q6 output head of the
// qwen3_8_flash_next_nvfp4 recipe ([248320,2560] split by rows). Routes measured on a
// [248320,2048] head (the nearest hidden width), untuned on the RTX 5070 Ti.
Q6Launch select_q6_n124160_k2560(std::int32_t tokens) {
    if (tokens <= 1) return launch_q6_a16_gemv_r4_w2_g16;
    // An MTP verification (2..8 columns) on an 8-token tile: the 16-token tile's K split and MMA
    // order, so the same bits, without its padding columns.
    if (tokens <= 8) return launch_q6_a16_sliced_r32_t8_w4_s2;
    if (tokens <= 16) return launch_q6_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 32) return launch_q6_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 40) return launch_q6_a16_mma_r64_t40_k128;
    if (tokens <= 48) return launch_q6_a16_mma_r64_t48_k128;
    if (tokens <= 56) return launch_q6_a16_mma_r64_t56_k128;
    if (tokens <= 64) return launch_q6_a16_sliced_r32_t64_w2_s1;
    if (tokens <= 72) return launch_q6_a16_mma_r64_t72_k128;
    if (tokens <= 80) return launch_q6_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q6_a16_mma_r64_t96;
    if (tokens <= 112) return launch_q6_a16_mma_r64_t112;
    return launch_q6_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
