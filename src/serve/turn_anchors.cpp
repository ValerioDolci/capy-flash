// Turn-anchor marker injection (serve/turn_anchors.h). The protocol markers own the explicit-marker
// budget; engine anchors yield to it. Protocol markers are added to input.context_cache.markers
// before this call (see GenerationService::prepare()).
#include "serve/turn_anchors.h"

namespace ninfer::serve {

void add_turn_anchor_markers(ninfer::PromptInput& input, std::uint32_t turn_anchors) {
    const std::size_t protocol_markers = input.context_cache.markers.size();
    if (protocol_markers >= ninfer::kMaximumExplicitPromptCacheMarkers) { return; }
    const auto budget =
        static_cast<std::uint32_t>(ninfer::kMaximumExplicitPromptCacheMarkers - protocol_markers);
    if (turn_anchors > budget) { turn_anchors = budget; }
    if (turn_anchors == 0 || input.messages.size() < 2) { return; }
    const std::size_t last_boundary = input.messages.size() - 1U;
    for (std::size_t count = 0; count < turn_anchors && last_boundary > count; ++count) {
        input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
            .after_message_count = static_cast<std::uint32_t>(last_boundary - count),
            .kind                = ninfer::PromptCacheMarkerKind::PrivateLongAnchor,
            .evidence            = ninfer::SharedCandidateEvidence::None,
            .location            = ninfer::PromptCacheMarkerLocation::MessageBoundary,
        });
    }
}

} // namespace ninfer::serve
