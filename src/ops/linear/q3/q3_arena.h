#pragma once

// The A8 prefill route's activation scratch, taken from a caller workspace.

#include "core/arena.h"
#include "ops/linear/q3/q3_linear.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::ops::detail {

// Holds the A8 scratch of one Q3 call for the lifetime of this object (the arena scope is released
// on destruction, after the launches are enqueued: the next call reuses the bytes in stream order).
// Leaves the workspace empty, so the call takes the in-kernel quantizing tile with the same bits,
// when there is no arena, the A8 route is off or T < 64, or the arena lacks room.
class Q3ArenaWorkspace {
public:
    Q3ArenaWorkspace(WorkspaceArena* arena, std::int32_t k, std::int32_t tokens) {
        if (arena == nullptr) return;
        const std::size_t bytes = q3_workspace_bytes(k, tokens);
        if (bytes == 0) return;
        // Exact fit test (the planner reserves these bytes at the stage's peak, with no slack).
        const auto at  = reinterpret_cast<std::uintptr_t>(arena->base()) + arena->used();
        const auto pad = static_cast<std::size_t>((256 - (at & 255u)) & 255u);
        if (arena->used() + pad + bytes > arena->capacity()) return;
        scope_.emplace(arena->scope());
        const DeviceSpan span = arena->alloc_bytes(bytes, 256);
        ws_                   = {span.data, bytes};
    }

    Q3ArenaWorkspace(const Q3ArenaWorkspace&)            = delete;
    Q3ArenaWorkspace& operator=(const Q3ArenaWorkspace&) = delete;

    [[nodiscard]] const Q3Workspace& get() const noexcept { return ws_; }

private:
    std::optional<WorkspaceArena::Scope> scope_;
    Q3Workspace ws_;
};

} // namespace ninfer::ops::detail
