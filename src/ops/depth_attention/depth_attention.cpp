// ninfer::ops - depth_attention wrapper: validates the public contract, then launches.
#include "ninfer/ops/depth_attention.h"

#include "ops/common/validate.h"
#include "ops/depth_attention/launch.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::invalid_argument("depth_attention: " + message); }
}

bool is_dense(const Tensor& t, DType dtype) {
    return t.dtype == dtype && t.data != nullptr && t.is_contiguous() && t.numel() > 0;
}

} // namespace

void depth_attention(std::span<const Tensor> states, const Tensor& current, const Tensor& query,
                     float eps, float scale, Tensor& out, cudaStream_t stream) {
    require(states.size() <= static_cast<std::size_t>(kDepthAttentionMaxStates),
            "at most kDepthAttentionMaxStates block states");
    require(is_dense(current, DType::BF16) || is_dense(current, DType::FP32),
            "current must be contiguous non-empty BF16 or FP32");
    const std::int32_t hidden = current.ne[0];
    const std::int32_t tokens = current.ne[1];
    require(current.ne[2] == 1 && current.ne[3] == 1, "current must be [H,T]");
    require(tokens <= std::numeric_limits<int>::max(), "too many tokens");
    for (const Tensor& state : states) {
        require(is_dense(state, DType::BF16), "states must be contiguous non-empty BF16");
        require(state.ne[0] == hidden && state.ne[1] == tokens && state.ne[2] == 1 &&
                    state.ne[3] == 1,
                "every state must match current");
    }
    require(is_dense(query, DType::FP32), "query must be contiguous non-empty FP32");
    const std::int32_t queries = query.ne[1];
    require(query.ne[0] == hidden && query.ne[2] == 1 && query.ne[3] == 1 && queries >= 1 &&
                queries <= kDepthAttentionMaxQueries,
            "query must be ne={H,Q} with 1 <= Q <= kDepthAttentionMaxQueries");
    require(is_dense(out, DType::FP32), "out must be contiguous non-empty FP32");
    require(out.ne[0] == hidden && out.ne[1] == tokens && out.ne[2] == queries && out.ne[3] == 1,
            "out must be ne={H,T,Q}");
    require(std::isfinite(eps) && eps > 0.0F, "eps must be finite and positive");
    require(std::isfinite(scale), "scale must be finite");
    require(!detail::overlaps(out, current) && !detail::overlaps(out, query),
            "out must not overlap an input");
    for (const Tensor& state : states) {
        require(!detail::overlaps(out, state), "out must not overlap an input");
    }
    detail::depth_attention_launch(states, current, query, eps, scale, out, stream);
}

} // namespace ninfer::ops
