#include "models/qwen3_5/program/planning/graph_profiles.h"

#include "ninfer/ops/sparse_attention.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {
// Resource tiers bound inactive attention work. They are not kernel/topology boundaries but for
// the last one: on Qwen4Exp the tiers past kQsaSplitTopkMinKeys also launch the QSA split top-k,
// so they are marked (GraphExecutionProfile::split_topk) and a round crossing the boundary
// re-instantiates that graph executable once; no shorter frontier launches the split.
constexpr std::array<std::uint32_t, 8> kCausalVisibleTiers{
    128, 512, 2048, 4096, 8192, 16384, 32768, ops::kQsaSplitTopkMinKeys};

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

std::vector<GraphExecutionProfile> causal_resource_profiles(std::uint32_t capacity,
                                                            std::uint32_t visible_offset,
                                                            bool qsa_split_topk) {
    std::vector<std::uint32_t> ends;
    for (const auto visible : kCausalVisibleTiers)
        if (visible >= visible_offset) ends.push_back(visible - visible_offset);
    auto profiles = graph_profiles_through(capacity - 1, ends);
    for (GraphExecutionProfile& profile : profiles) {
        profile.split_topk =
            qsa_split_topk &&
            static_cast<std::uint64_t>(profile.min) + visible_offset > ops::kQsaSplitTopkMinKeys;
    }
    return profiles;
}

} // namespace

std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity,
                                                           bool qsa_split_topk) {
    // E+1 is the one-token visible window; the tiers share one topology per exact B (but for the
    // QSA split top-k boundary above).
    return causal_resource_profiles(capacity, 1, qsa_split_topk);
}

std::vector<GraphExecutionProfile>
mtp_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window, bool qsa_split_topk) {
    if (draft_window == 0) { return {}; }
    return mtp_graph_profiles(capacity, draft_window + 1U, draft_window, qsa_split_topk);
}

std::vector<GraphExecutionProfile> mtp_graph_profiles(std::uint32_t capacity, std::uint32_t width,
                                                      std::uint32_t drafts, bool qsa_split_topk) {
    if (width < 2 || drafts == 0 || capacity == 0) { return {}; }
    // The final AR call can see E + (width - 1) + drafts (2K for an MTP round of K drafts). Target
    // verify and MTP attention are update-compatible across the resource tiers, independently of
    // the selected KV representation, but for the QSA split top-k boundary
    // (causal_resource_profiles).
    return causal_resource_profiles(capacity, (width - 1U) + drafts, qsa_split_topk);
}

std::vector<GraphExecutionProfile> dflash_graph_profiles(std::uint32_t capacity,
                                                         std::uint32_t draft_window) {
    if (capacity == 0 || draft_window == 0 || draft_window > 15) {
        throw std::invalid_argument("invalid masked draft graph dimensions");
    }
    auto profiles = graph_profiles_through(capacity - 1, {96, 511, 2047, 8191, 32767});
    for (std::size_t i = 0; i < profiles.size(); ++i) {
        profiles[i].topology_class = static_cast<std::uint32_t>(i);
    }
    return profiles;
}

unsigned dflash_adaptive_drafts(std::uint32_t maximum, std::uint32_t base) {
    // Each family is a graph executable per resource tier, so only the counts worth a round of
    // their own. Measured on the dense 27B at tp 2 (fixed --draft-tokens, ms per round): 1..4
    // drafts 12.5-13.1, 5 and 6 drafts 13.9 and 14.7, 7 drafts 13.6 (the verify GEMMs take A4
    // activations from 8 columns), 10 drafts 14.5, 15 drafts 15.7; and the drafter's blocks of 8
    // (7 drafts) and more draft better than the smaller ones (conversation: 3.2 tokens a round at
    // 6 drafts, 4.0 at 7). So 5 and 6 drafts are dearer than 7 and worse. Fewer drafts than the
    // base count are not offered: a smaller block drafts worse, its rounds then keep fewer drafts
    // and the measured rates pull the window smaller still (a conversation run fell from 278 to
    // 221 tok/s), for at most ~1 ms of a 13.6 ms round.
    constexpr std::array<std::uint32_t, 8> kDrafts{1, 2, 3, 4, 7, 10, 12, 15};
    unsigned mask = 0;
    for (const std::uint32_t k : kDrafts) {
        if (k <= maximum && k > base) { mask |= 1U << k; }
    }
    if (maximum > base && ((mask >> maximum) & 1U) == 0) { mask |= 1U << maximum; }
    return mask;
}

} // namespace ninfer::models::qwen3_5::detail
