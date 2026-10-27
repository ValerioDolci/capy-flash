// add_turn_anchor_markers (--turn-anchors): how many engine anchors survive alongside protocol
// markers. The protocol markers own the explicit-marker budget (kMaximumExplicitPromptCacheMarkers,
// the Anthropic block-level breakpoint allowance); engine anchors yield to them so a client that
// fills the allowance is not rejected for the serving layer's own marker (docs/serving.md).
#include "serve/turn_anchors.h"

#include <iostream>

namespace {

using ninfer::kMaximumExplicitPromptCacheMarkers;
using ninfer::PromptCacheMarkerKind;
using ninfer::PromptInput;
using ninfer::serve::add_turn_anchor_markers;

int failures = 0;

void check(bool condition, const char* message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

PromptInput input_with(std::size_t messages, std::size_t protocol_markers) {
    PromptInput input;
    input.messages.resize(messages);
    for (std::size_t index = 0; index < protocol_markers; ++index) {
        input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
            .after_message_count = static_cast<std::uint32_t>(index + 1),
            .kind                = PromptCacheMarkerKind::SharedStablePrefix,
        });
    }
    return input;
}

} // namespace

int main() {
    // No protocol markers: the requested anchors are added at the trailing message boundaries.
    {
        auto input = input_with(4, 0);
        add_turn_anchor_markers(input, 2);
        check(input.context_cache.markers.size() == 2, "two anchors without protocol markers");
        check(input.context_cache.markers[0].after_message_count == 3, "first anchor boundary");
        check(input.context_cache.markers[1].after_message_count == 2, "second anchor boundary");
        for (const auto& marker : input.context_cache.markers) {
            check(marker.kind == PromptCacheMarkerKind::PrivateLongAnchor, "anchor kind");
        }
    }
    // The protocol markers fill the budget: no engine anchor is added on top.
    {
        auto input = input_with(4, kMaximumExplicitPromptCacheMarkers);
        add_turn_anchor_markers(input, 1);
        check(input.context_cache.markers.size() == kMaximumExplicitPromptCacheMarkers,
              "a full protocol budget leaves no anchor slot");
    }
    // Partial budgets clamp: anchors never push the total past the allowance.
    {
        auto input = input_with(6, kMaximumExplicitPromptCacheMarkers - 1);
        add_turn_anchor_markers(input, 2);
        check(input.context_cache.markers.size() == kMaximumExplicitPromptCacheMarkers,
              "one free slot admits exactly one anchor");
        check(input.context_cache.markers.back().kind == PromptCacheMarkerKind::PrivateLongAnchor,
              "the surviving anchor is the engine one");
    }
    // The message-count guard is unchanged.
    {
        auto input = input_with(1, 0);
        add_turn_anchor_markers(input, 1);
        check(input.context_cache.markers.empty(), "a single message has no anchor boundary");
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " turn anchor marker budget\n";
    return failures == 0 ? 0 : 1;
}
