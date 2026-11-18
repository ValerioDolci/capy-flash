#include "core/weight.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"

#include "ops/gdn_input_proj/tp2/fp8_gdn_input_shard.h"
#include "ops/linear/fp8/fp8_geometry.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Fp8GdnInputRoute : std::uint8_t {
    A16,
    A8,
};

Fp8GdnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("fp8 gdn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only) { return Fp8GdnInputRoute::A16; }
    if (!allows_a8(policy)) {
        throw std::invalid_argument("fp8 gdn_input_proj: unsupported policy");
    }
    return tokens >= 17 ? Fp8GdnInputRoute::A8 : Fp8GdnInputRoute::A16;
}

// The parent [16384,5120] and its [8192,5120] two-device shard share K, so this one plan serves
// both: the routes and token cutoffs are the parent's; the launchers, which write each problem's
// qkv|z section output, and the split-K partial reservation are each problem's own.
struct Problem {
    void (*decode)(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t);
    void (*matrix)(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t);
    void (*a8)(const Tensor&, const Weight&, Tensor&, Tensor&, Fp8A8Workspace, cudaStream_t);
    std::size_t (*partial_capacity_bytes)(std::int32_t);
};

constexpr Problem kParent{fp8_gdn_input_decode_launch, fp8_gdn_input_matrix_launch,
                          fp8_gdn_input_a8_launch, fp8_gdn_input_partial_capacity_bytes};
constexpr Problem kShard{fp8_gdn_input_shard_decode_launch, fp8_gdn_input_shard_matrix_launch,
                         fp8_gdn_input_shard_a8_launch, fp8_gdn_input_shard_partial_capacity_bytes};

std::size_t workspace_capacity_bytes(const Problem& problem, LinearPolicy policy,
                                     std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 gdn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    return resolve_route(policy, max_tokens) == Fp8GdnInputRoute::A8
               ? fp8_a8_workspace_capacity_bytes(max_tokens, Fp8N16384K5120::kInputRows,
                                                 problem.partial_capacity_bytes(max_tokens))
               : 0;
}

void a16_dispatch(const Problem& problem, const Tensor& x, const Weight& weight, Tensor& qkv,
                  Tensor& z, cudaStream_t stream) {
    if (x.ne[1] == 1) {
        problem.decode(x, weight, qkv, z, stream);
    } else {
        problem.matrix(x, weight, qkv, z, stream);
    }
}

void a8_dispatch(const Problem& problem, const Tensor& x, const Weight& weight, Tensor& qkv,
                 Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                   = workspace.scope();
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(
        workspace, x.ne[1], weight.k, problem.partial_capacity_bytes(x.ne[1]));
    problem.a8(x, weight, qkv, z, scratch, stream);
}

void dispatch(const Problem& problem, const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
              LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Fp8GdnInputRoute::A16) {
        a16_dispatch(problem, x, weight, qkv, z, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("fp8 A8 gdn_input_proj requires caller workspace");
    }
    a8_dispatch(problem, x, weight, qkv, z, *workspace, stream);
}

} // namespace

std::size_t fp8_gdn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    return workspace_capacity_bytes(kParent, policy, min_tokens, max_tokens);
}

void fp8_gdn_input_a16_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream) {
    a16_dispatch(kParent, x, weight, qkv, z, stream);
}

void fp8_gdn_input_a8_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                               WorkspaceArena& workspace, cudaStream_t stream) {
    a8_dispatch(kParent, x, weight, qkv, z, workspace, stream);
}

void fp8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                            LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    dispatch(kParent, x, weight, qkv, z, policy, workspace, stream);
}

std::size_t fp8_gdn_input_shard_workspace_capacity_bytes(LinearPolicy policy,
                                                         std::int32_t min_tokens,
                                                         std::int32_t max_tokens) {
    return workspace_capacity_bytes(kShard, policy, min_tokens, max_tokens);
}

void fp8_gdn_input_shard_a16_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                      cudaStream_t stream) {
    a16_dispatch(kShard, x, weight, qkv, z, stream);
}

void fp8_gdn_input_shard_a8_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                     WorkspaceArena& workspace, cudaStream_t stream) {
    a8_dispatch(kShard, x, weight, qkv, z, workspace, stream);
}

void fp8_gdn_input_shard_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                  LinearPolicy policy, WorkspaceArena* workspace,
                                  cudaStream_t stream) {
    dispatch(kShard, x, weight, qkv, z, policy, workspace, stream);
}

} // namespace ninfer::ops::detail
