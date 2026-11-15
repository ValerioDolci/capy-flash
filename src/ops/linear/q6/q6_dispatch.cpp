#include "ops/linear/q6/q6_dispatch.h"
#include "ops/linear/q6/q6_shapes.h"
#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
struct ShapeEntry {
    std::int32_t n, k;
    Q6Launch (*select)(std::int32_t);
};

constexpr std::array kShapes{
    ShapeEntry{248320, 5120, select_q6_n248320_k5120},
    ShapeEntry{248320, 2560, select_q6_n248320_k2560},
    ShapeEntry{1152, 1536, select_q6_n1152_k1536},
    // Qwen4Exp at tp 2: one rank's vocabulary half of the Q6 output head.
    ShapeEntry{124160, 2560, select_q6_n124160_k2560},
};
} // namespace

// Qwen4Exp draft proposal heads, rows of the K = 2560 output head (a prefix view or the
// gathered rows of a draft token shortlist, models/qwen3_5/execution/qwen4_draft_head.h): any row
// count below the full head, T = 1 only (the proposal runs one GEMV per column).
constexpr std::int32_t kDraftHeadK       = 2560;
constexpr std::int32_t kDraftHeadMaxRows = 248320;

Q6Launch select_q6_draft_head(std::int32_t t) {
    if (t != 1) throw std::invalid_argument("q6 linear: draft-head row blocks run at T = 1 only");
    // RTX 5070 Ti, cold L2, T = 1, K = 2560: 708 us for 248 072 rows = 701 GB/s, 317 us for
    // 108 856, vs 999 us / 498 GB/s for the full head's T <= 2 route (simt_r8_t4, which the
    // full-head shape keeps: plain decode output is unchanged).
    return launch_q6_a16_gemv_r8_w1_g8_s4_cg;
}

Q6Launch select_q6_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) throw std::invalid_argument("q6 linear: T must be positive");
    for (const auto& entry : kShapes) {
        if (entry.n == n && entry.k == k) return entry.select(t);
    }
    if (k == kDraftHeadK && n > 0 && n < kDraftHeadMaxRows) return select_q6_draft_head(t);
    throw std::invalid_argument("q6 linear: unsupported shape");
}

Q6Launch select_q6_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("q6 linear: unsupported policy");
    return select_q6_a16_launch(n, k, t);
}

void q6_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    select_q6_launch(weight.n, weight.k, x.ne[1], policy)(x, weight, out, stream);
}
} // namespace ninfer::ops::detail
