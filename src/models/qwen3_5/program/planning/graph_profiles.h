#pragma once
#include "models/qwen3_5/program/program.h"

namespace ninfer::models::qwen3_5::detail {

// `qsa_split_topk`: a Qwen4Exp model with QSA, whose rounds past ops::kQsaSplitTopkMinKeys also
// launch the split top-k (GraphExecutionProfile::split_topk).
[[nodiscard]] std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity,
                                                                         bool qsa_split_topk);
[[nodiscard]] std::vector<GraphExecutionProfile>
mtp_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window, bool qsa_split_topk);
// An MTP-backend round that verifies `width` columns and drafts `drafts` (prompt lookup).
[[nodiscard]] std::vector<GraphExecutionProfile> mtp_graph_profiles(std::uint32_t capacity,
                                                                    std::uint32_t width,
                                                                    std::uint32_t drafts,
                                                                    bool qsa_split_topk);
// The DFlash2 round's profiles.
[[nodiscard]] std::vector<GraphExecutionProfile> dflash_graph_profiles(std::uint32_t capacity,
                                                                       std::uint32_t draft_window);
// DFlash2 adaptive draft window: the draft counts of the one-row graph families (bit k), at most
// `maximum` and without `base` (the every-batch family drafts it).
[[nodiscard]] unsigned dflash_adaptive_drafts(std::uint32_t maximum, std::uint32_t base);

} // namespace ninfer::models::qwen3_5::detail
