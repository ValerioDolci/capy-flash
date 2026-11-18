#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
const std::array kShapes{
    &kNvfp4N14336K5120, &kNvfp4N16384K5120,  &kNvfp4N34816K5120,  &kNvfp4N5120K6144,
    &kNvfp4N5120K17408, &kNvfp4N17408K5120,  &kNvfp4N5120K8704,   &kNvfp4N5120K3072,
    &kNvfp4N8192K2560,  &kNvfp4N6656K2560,   &kNvfp4N124160K2560, &kNvfp4N512K2560,
    &kNvfp4N640K2560,   &kNvfp4N2560K3072,   &kNvfp4N13312K2560,  &kNvfp4N16384K2560,
    &kNvfp4N1280K2560,  &kNvfp4N248320K2560, &kNvfp4N2560K6144,   &kNvfp4N2560K640,
    &kNvfp4N28672K5120, &kNvfp4N5120K14336,  &kNvfp4N5120K7168,   &kNvfp4N8192K5120,
    &kNvfp4N7168K5120};

const Nvfp4LinearShape& resolve_shape(std::int32_t n, std::int32_t k, LinearPolicy policy) {
    if (!valid_linear_policy(policy))
        throw std::invalid_argument("nvfp4 linear: unsupported policy");
    for (const auto* shape : kShapes)
        if (shape->n == n && shape->k == k) return *shape;
    throw std::invalid_argument("nvfp4 linear: unsupported shape");
}
} // namespace

Nvfp4LinearRoute nvfp4_linear_route(std::int32_t n, std::int32_t k, LinearPolicy policy,
                                    std::int32_t tokens) {
    const auto& shape = resolve_shape(n, k, policy);
    if (allows_a4(policy) && shape.uses_a4(tokens, tokens)) return Nvfp4LinearRoute::A4;
    if (policy == LinearPolicy::AllowA4Decode && tokens <= kAllowA4DecodeMaxWidth &&
        shape.a16_decode != nullptr)
        return Nvfp4LinearRoute::A16Decode;
    return Nvfp4LinearRoute::A16;
}

std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k,
                                                  LinearPolicy policy, std::int32_t min_tokens,
                                                  std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("nvfp4 linear workspace: invalid token interval");
    const auto& shape = resolve_shape(n, k, policy);
    return allows_a4(policy) && shape.uses_a4(min_tokens, max_tokens)
               ? nvfp4_a4_workspace_capacity_bytes(max_tokens, k)
               : 0;
}

Nvfp4LinearSelection nvfp4_linear_selection(std::int32_t n, std::int32_t k, LinearPolicy policy,
                                            std::int32_t tokens) {
    const auto& shape = resolve_shape(n, k, policy);
    const auto route  = nvfp4_linear_route(n, k, policy, tokens);
    switch (route) {
    case Nvfp4LinearRoute::A16:
        return {route, shape.a16(tokens)};
    case Nvfp4LinearRoute::A16Decode:
        return {route, shape.a16_decode(tokens)};
    case Nvfp4LinearRoute::A4:
        break;
    }
    return {route, nullptr, shape.a4(tokens)};
}

std::span<const Nvfp4LinearShape* const> nvfp4_linear_shapes() { return kShapes; }

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    WorkspaceArena* workspace, cudaStream_t stream) {
    validate_nvfp4_weight(weight, "nvfp4 linear");
    if (x.ne[1] <= 0) throw std::invalid_argument("nvfp4 linear: T must be positive");
    const auto selection = nvfp4_linear_selection(weight.n, weight.k, policy, x.ne[1]);
    if (selection.route != Nvfp4LinearRoute::A4) return selection.a16(x, weight, out, stream);
    if (workspace == nullptr)
        throw std::invalid_argument("nvfp4 A4 linear requires caller workspace");
    auto scope         = workspace->scope();
    const auto scratch = allocate_nvfp4_a4_workspace(*workspace, x.ne[1], weight.k);
    launch_nvfp4_a4_quantize(x, weight, scratch, selection.a4.scales, stream);
    selection.a4.launch(weight, out, scratch, x.ne[1], stream);
}
} // namespace ninfer::ops::detail
