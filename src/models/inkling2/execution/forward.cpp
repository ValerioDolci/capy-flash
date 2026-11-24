// The Inkling 2 forward's two instantiations: execution (WorkspaceArena) and workspace sizing
// (WorkspaceLayoutBuilder). Its callers include only the declarations (forward.h), so the member
// definitions compile here once.
#include "models/inkling2/execution/forward.h"

#include "models/inkling2/execution/forward/core.h"
#include "models/inkling2/execution/forward/attention.h"
#include "models/inkling2/execution/forward/head.h"
#include "models/inkling2/execution/forward/moe.h"
#include "models/inkling2/execution/forward/overlap.h"
#include "models/inkling2/execution/forward/prefetch.h"
#include "models/inkling2/execution/forward/residual.h"
#include "models/inkling2/execution/forward/visit.h"

#include <algorithm>

namespace ninfer::models::inkling2::execution {

template class Forward<WorkspaceArena>;
template class Forward<WorkspaceLayoutBuilder>;

std::size_t forward_workspace_bytes(const TextConfig& config,
                                    std::array<ForwardRank<WorkspaceArena>, 2> ranks, int width,
                                    std::int32_t max_tokens, std::int32_t max_logit_columns) {
    std::array<WorkspaceLayoutBuilder, 2> builders;
    std::array<ForwardRank<WorkspaceLayoutBuilder>, 2> plan{};
    for (std::size_t r = 0; r < 2; ++r) {
        plan[r].config     = ranks[r].config;
        plan[r].parameters = ranks[r].parameters;
        plan[r].work       = &builders[r];
        plan[r].device     = ranks[r].device;
    }
    Forward<WorkspaceLayoutBuilder> forward(config, plan, width, {}, max_tokens, max_logit_columns);
    forward.run({}, ForwardColumns{max_tokens, 0, max_logit_columns}, {});
    std::size_t bytes = 0;
    for (int r = 0; r < width; ++r) {
        bytes = std::max(bytes, builders[static_cast<std::size_t>(r)].peak_bytes());
    }
    return bytes;
}

} // namespace ninfer::models::inkling2::execution
