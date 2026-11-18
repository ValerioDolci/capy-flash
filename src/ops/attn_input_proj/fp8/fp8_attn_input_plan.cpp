#include "core/weight.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"

#include "ops/attn_input_proj/tp2/fp8_attn_input_shard.h"
#include "ops/linear/fp8/fp8_geometry.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Fp8AttnInputRoute : std::uint8_t {
    A16,
    A8,
};

Fp8AttnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("fp8 attn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only) { return Fp8AttnInputRoute::A16; }
    if (!allows_a8(policy)) {
        throw std::invalid_argument("fp8 attn_input_proj: unsupported policy");
    }
    return tokens >= 17 ? Fp8AttnInputRoute::A8 : Fp8AttnInputRoute::A16;
}

// The parent [14336,5120] and its [7168,5120] two-device shard share K, so this one plan serves
// both: the routes and token cutoffs are the parent's; the launchers, which write each problem's
// q|k|gate|v section output, and the split-K partial reservation are each problem's own.
using Launch = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                        cudaStream_t);

struct Problem {
    Launch decode;
    Launch small_mma;
    Launch gemm;
    void (*a8)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&, Fp8A8Workspace,
               cudaStream_t);
    std::size_t (*partial_capacity_bytes)(std::int32_t);
};

constexpr Problem kParent{fp8_attn_input_decode_launch, fp8_attn_input_a16_small_mma_launch,
                          fp8_attn_input_a16_gemm_launch, fp8_attn_input_a8_launch,
                          fp8_attn_input_partial_capacity_bytes};
constexpr Problem kShard{fp8_attn_input_shard_decode_launch,
                         fp8_attn_input_shard_a16_small_mma_launch,
                         fp8_attn_input_shard_a16_gemm_launch, fp8_attn_input_shard_a8_launch,
                         fp8_attn_input_shard_partial_capacity_bytes};

void launch_a16(const Problem& problem, const Tensor& x, const Weight& weight, Tensor& q,
                Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[1] == 1)
        problem.decode(x, weight, q, gate, k, v, stream);
    else if (x.ne[1] <= kFp8AttnInputLastSmallMmaT)
        problem.small_mma(x, weight, q, gate, k, v, stream);
    else
        problem.gemm(x, weight, q, gate, k, v, stream);
}

std::size_t workspace_capacity_bytes(const Problem& problem, LinearPolicy policy,
                                     std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 attn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    return resolve_route(policy, max_tokens) == Fp8AttnInputRoute::A8
               ? fp8_a8_workspace_capacity_bytes(max_tokens, Fp8N14336K5120::kInputRows,
                                                 problem.partial_capacity_bytes(max_tokens))
               : 0;
}

void dispatch(const Problem& problem, const Tensor& x, const Weight& weight, Tensor& q,
              Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
              cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Fp8AttnInputRoute::A16) {
        launch_a16(problem, x, weight, q, gate, k, v, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("fp8 A8 attn_input_proj requires caller workspace");
    }
    auto scope                   = workspace->scope();
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(
        *workspace, x.ne[1], weight.k, problem.partial_capacity_bytes(x.ne[1]));
    problem.a8(x, weight, q, gate, k, v, scratch, stream);
}

} // namespace

std::size_t fp8_attn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    return workspace_capacity_bytes(kParent, policy, min_tokens, max_tokens);
}

std::size_t fp8_attn_input_shard_workspace_capacity_bytes(LinearPolicy policy,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens) {
    return workspace_capacity_bytes(kShard, policy, min_tokens, max_tokens);
}

void fp8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                             Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                             cudaStream_t stream) {
    dispatch(kParent, x, weight, q, gate, k, v, policy, workspace, stream);
}

void fp8_attn_input_shard_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, LinearPolicy policy,
                                   WorkspaceArena* workspace, cudaStream_t stream) {
    dispatch(kShard, x, weight, q, gate, k, v, policy, workspace, stream);
}

} // namespace ninfer::ops::detail
