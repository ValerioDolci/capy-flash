#pragma once

// Turn anchors (--turn-anchors N). A same-session follow-up consumes the previous turn's
// continuation and its own turn-closure capture replaces the previous closure, so once turn N+1 is
// running no checkpoint is left at the end of turn N: a cancelled or edited turn re-prefilled the
// whole conversation (issue #3). A private long-anchor opportunity at the message boundary before
// the final message (and, with N > 1, before the messages preceding it) makes the follow-up's
// prefill capture that branch point; the Engine keeps and replaces it like an explicit anchor,
// within --max-long-anchors-per-continuation. Protocol markers stay first: Engine order follows
// the marker order, and the anchors are the first to go when the protocol markers already fill
// the explicit-marker budget: a client that sends kMaximumExplicitPromptCacheMarkers breakpoints
// (Anthropic allows exactly that many) must not be rejected because the serving layer appended
// its own anchor on top.

#include "ninfer/types.h"

#include <cstdint>

namespace ninfer::serve {

void add_turn_anchor_markers(ninfer::PromptInput& input, std::uint32_t turn_anchors);

} // namespace ninfer::serve
