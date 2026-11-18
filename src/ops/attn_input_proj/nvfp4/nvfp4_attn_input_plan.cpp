#include "core/weight.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "ops/attn_input_proj/tp2/nvfp4_attn_input_shard.h"
#include "ops/linear/nvfp4/nvfp4_layout.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

Nvfp4AttnInputRoute nvfp4_attn_input_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 attn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) {
        return Nvfp4AttnInputRoute::A16;
    }
    if (!allows_a4(policy)) {
        throw std::invalid_argument("nvfp4 attn_input_proj: unsupported policy");
    }
    // AllowA4Decode runs A4 at every width, as the MTP K=3 verification width 4 does: the A4 MMA
    // tile is the same through T=64, so a token column has the same bits at every width.
    if (policy == LinearPolicy::AllowA4Decode && tokens <= kAllowA4DecodeMaxWidth) {
        return Nvfp4AttnInputRoute::A4;
    }
    return tokens >= kNvfp4AttnInputFamilyFirstA4Tokens ? Nvfp4AttnInputRoute::A4
                                                        : Nvfp4AttnInputRoute::A16;
}

std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 attn_input_proj workspace: invalid token interval");
    }
    (void)nvfp4_attn_input_route(policy, min_tokens);
    return nvfp4_attn_input_route(policy, max_tokens) == Nvfp4AttnInputRoute::A4
               ? nvfp4_a4_workspace_capacity_bytes(max_tokens, Nvfp4N14336K5120::kInputRows)
               : 0;
}

namespace {

// The parent [14336,5120] and its [7168,5120] two-device shard share K, so this one plan serves
// both: the routes, the token cutoffs and the workspace are the parent's; only the launchers,
// which write each problem's section output, differ.
struct Launchers {
    void (*a16)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&, cudaStream_t);
    void (*a4)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&, Nvfp4A4Workspace,
               cudaStream_t);
};

constexpr Launchers kLaunchers{nvfp4_attn_input_a16_launch, nvfp4_attn_input_a4_launch};
constexpr Launchers kShardLaunchers{nvfp4_attn_input_shard_a16_launch,
                                    nvfp4_attn_input_shard_a4_launch};

void dispatch(const Launchers& launch, const Tensor& x, const Weight& weight, Tensor& q,
              Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
              cudaStream_t stream) {
    if (nvfp4_attn_input_route(policy, x.ne[1]) == Nvfp4AttnInputRoute::A16) {
        launch.a16(x, weight, q, gate, k, v, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 A4 attn_input_proj requires caller workspace");
    }
    auto scope                     = workspace->scope();
    const Nvfp4A4Workspace scratch = allocate_nvfp4_a4_workspace(*workspace, x.ne[1], weight.k);
    launch.a4(x, weight, q, gate, k, v, scratch, stream);
}

} // namespace

void nvfp4_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                               cudaStream_t stream) {
    dispatch(kLaunchers, x, weight, q, gate, k, v, policy, workspace, stream);
}

void nvfp4_attn_input_shard_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, LinearPolicy policy,
                                     WorkspaceArena* workspace, cudaStream_t stream) {
    dispatch(kShardLaunchers, x, weight, q, gate, k, v, policy, workspace, stream);
}

std::size_t nvfp4_attn_input_shard_workspace_capacity_bytes(LinearPolicy policy,
                                                            std::int32_t min_tokens,
                                                            std::int32_t max_tokens) {
    return nvfp4_attn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
}

} // namespace ninfer::ops::detail
