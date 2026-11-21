// ninfer::ops - residual_mix wrappers: validate the public contracts, then launch.
#include "ninfer/ops/residual_mix.h"

#include "ninfer/ops/depth_attention.h"
#include "ninfer/ops/short_conv.h"
#include "ops/common/validate.h"
#include "ops/residual_mix/launch.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* op, const std::string& message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

bool is_dense(const Tensor& t, DType dtype) {
    return t.dtype == dtype && t.data != nullptr && t.is_contiguous() && t.numel() > 0;
}

bool is_matrix(const Tensor& t, std::int32_t rows, std::int32_t columns) {
    return t.ne[0] == rows && t.ne[1] == columns && t.ne[2] == 1 && t.ne[3] == 1;
}

// The width and token count of a call, from the convolution input or the current state.
void check_extent(const char* op, const Tensor& x) {
    require(is_dense(x, DType::BF16) && x.ne[2] == 1 && x.ne[3] == 1, op,
            "the branch/current must be contiguous BF16 [H,T]");
    require(x.ne[0] >= 1 && x.ne[0] <= kResidualMixMaxHidden, op,
            "H must be 1..kResidualMixMaxHidden");
    require(x.ne[1] >= 1 && x.ne[1] <= kResidualMixMaxTokens, op,
            "T must be 1..kResidualMixMaxTokens");
}

detail::ResidualConvArgs conv_args(const char* op, const ResidualConv& conv) {
    check_extent(op, conv.x);
    const std::int32_t H = conv.x.ne[0], T = conv.x.ne[1];
    require(is_dense(conv.weight, DType::FP32) && is_matrix(conv.weight, kShortConvTaps, H), op,
            "conv weight must be contiguous FP32 ne={4,H}");
    require(std::isfinite(conv.scale), op, "conv scale must be finite");
    require(is_dense(conv.state, DType::BF16) && is_matrix(conv.state, H, kShortConvTaps - 1), op,
            "conv state must be contiguous BF16 ne={H,3}");
    const bool record = conv.record.data != nullptr;
    if (record) {
        require(is_dense(conv.record, DType::BF16) &&
                    is_matrix(conv.record, H, T + kShortConvTaps - 1),
                op, "record must be contiguous BF16 ne={H,T+3}");
        require(!detail::overlaps(conv.record, conv.x) &&
                    !detail::overlaps(conv.record, conv.state),
                op, "record must not overlap an input");
    } else {
        require(T == 1, op, "advancing the state in place takes one token");
        require(!detail::overlaps(conv.state, conv.x), op, "state must not overlap x");
    }
    detail::ResidualConvArgs a;
    a.x      = static_cast<const __nv_bfloat16*>(conv.x.data);
    a.weight = static_cast<const float*>(conv.weight.data);
    a.scale  = conv.scale;
    a.state  = static_cast<__nv_bfloat16*>(conv.state.data);
    a.record = record ? static_cast<__nv_bfloat16*>(conv.record.data) : nullptr;
    return a;
}

detail::MixArgs mix_args(const char* op, std::span<const Tensor> states, const Tensor& query,
                         float eps, float scale, std::span<const Tensor> norms, float norm_eps,
                         std::span<Tensor> normed, std::int32_t H, std::int32_t T) {
    require(states.size() <= static_cast<std::size_t>(kDepthAttentionMaxStates), op,
            "at most kDepthAttentionMaxStates block states");
    detail::MixArgs m;
    for (std::size_t s = 0; s < states.size(); ++s) {
        require(is_dense(states[s], DType::BF16) && is_matrix(states[s], H, T), op,
                "every state must be contiguous BF16 [H,T]");
        m.state[s] = static_cast<const __nv_bfloat16*>(states[s].data);
    }
    require(is_dense(query, DType::FP32) && query.ne[0] == H && query.ne[2] == 1 &&
                query.ne[3] == 1 && query.ne[1] >= 1 && query.ne[1] <= kDepthAttentionMaxQueries,
            op, "query must be FP32 ne={H,Q} with 1 <= Q <= kDepthAttentionMaxQueries");
    require(std::isfinite(eps) && eps > 0.0F && std::isfinite(scale), op,
            "eps must be finite and positive, scale finite");
    require(norms.size() == normed.size() &&
                (norms.empty() || norms.size() == static_cast<std::size_t>(query.ne[1])),
            op, "one output per norm, and none or one norm per query");
    require(norms.empty() || (std::isfinite(norm_eps) && norm_eps > 0.0F), op,
            "norm eps must be finite and positive");
    m.query    = static_cast<const float*>(query.data);
    m.queries  = query.ne[1];
    m.eps      = eps;
    m.scale    = scale;
    m.norms    = static_cast<int>(norms.size());
    m.norm_eps = norm_eps;
    for (std::size_t q = 0; q < norms.size(); ++q) {
        require(is_dense(norms[q], DType::BF16) && norms[q].numel() == H, op,
                "every norm weight must be contiguous BF16 with H elements");
        require(is_dense(normed[q], DType::BF16) && is_matrix(normed[q], H, T), op,
                "every normed output must be contiguous BF16 [H,T]");
        for (const Tensor& state : states) {
            require(!detail::overlaps(normed[q], state), op, "outputs must not overlap an input");
        }
        m.norm[q]   = static_cast<const __nv_bfloat16*>(norms[q].data);
        m.normed[q] = static_cast<__nv_bfloat16*>(normed[q].data);
    }
    return m;
}

} // namespace

void residual_conv_mix(const ResidualConv& conv, const Tensor* base, std::span<const Tensor> states,
                       const Tensor& query, float mix_eps, float mix_scale,
                       std::span<const Tensor> norms, float norm_eps, Tensor& residual, Tensor* mix,
                       std::span<Tensor> normed, cudaStream_t stream) {
    constexpr const char* op   = "residual_conv_mix";
    detail::ResidualConvArgs c = conv_args(op, conv);
    const std::int32_t H = conv.x.ne[0], T = conv.x.ne[1];
    detail::MixArgs m =
        mix_args(op, states, query, mix_eps, mix_scale, norms, norm_eps, normed, H, T);
    if (base != nullptr) {
        require(is_dense(*base, DType::BF16) && is_matrix(*base, H, T), op,
                "base must be contiguous BF16 [H,T]");
        c.base = static_cast<const __nv_bfloat16*>(base->data);
    }
    require(is_dense(residual, DType::FP32) && is_matrix(residual, H, T), op,
            "residual must be contiguous FP32 [H,T]");
    c.residual = static_cast<float*>(residual.data);
    require(mix != nullptr || !norms.empty(), op, "nothing to write: no mix and no norms");
    if (mix != nullptr) {
        require(is_dense(*mix, DType::FP32) && mix->ne[0] == H && mix->ne[1] == T &&
                    mix->ne[2] == m.queries && mix->ne[3] == 1,
                op, "mix must be contiguous FP32 ne={H,T,Q}");
        m.mix = static_cast<float*>(mix->data);
    }
    std::vector<const Tensor*> outputs{&residual};
    if (mix != nullptr) outputs.push_back(mix);
    for (const Tensor& t : normed) outputs.push_back(&t);
    std::vector<const Tensor*> inputs{&conv.x, &conv.weight, &conv.state, &query};
    if (base != nullptr) inputs.push_back(base);
    for (const Tensor& t : states) inputs.push_back(&t);
    for (const Tensor& t : norms) inputs.push_back(&t);
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        for (const Tensor* in : inputs) {
            require(!detail::overlaps(*outputs[i], *in), op, "outputs must not overlap an input");
        }
        for (std::size_t j = i + 1; j < outputs.size(); ++j) {
            require(!detail::overlaps(*outputs[i], *outputs[j]), op, "outputs must be disjoint");
        }
    }
    detail::residual_conv_mix_launch(c, m, static_cast<int>(states.size()) + 1, H, T, stream);
}

void depth_attention_rmsnorm(std::span<const Tensor> states, const Tensor& current,
                             const Tensor& query, float mix_eps, float mix_scale,
                             std::span<const Tensor> norms, float norm_eps,
                             std::span<Tensor> normed, cudaStream_t stream) {
    constexpr const char* op = "depth_attention_rmsnorm";
    check_extent(op, current);
    const std::int32_t H = current.ne[0], T = current.ne[1];
    detail::MixArgs m =
        mix_args(op, states, query, mix_eps, mix_scale, norms, norm_eps, normed, H, T);
    require(!norms.empty(), op, "one norm per query");
    m.current = static_cast<const __nv_bfloat16*>(current.data);
    for (std::size_t q = 0; q < normed.size(); ++q) {
        require(!detail::overlaps(normed[q], current) && !detail::overlaps(normed[q], query), op,
                "outputs must not overlap an input");
        for (std::size_t j = q + 1; j < normed.size(); ++j) {
            require(!detail::overlaps(normed[q], normed[j]), op, "outputs must be disjoint");
        }
    }
    detail::depth_attention_rmsnorm_launch(m, static_cast<int>(states.size()) + 1, H, T, stream);
}

void residual_conv_round(const ResidualConv& conv, const Tensor& residual, Tensor& out,
                         cudaStream_t stream) {
    constexpr const char* op         = "residual_conv_round";
    const detail::ResidualConvArgs c = conv_args(op, conv);
    const std::int32_t H = conv.x.ne[0], T = conv.x.ne[1];
    require(is_dense(residual, DType::FP32) && is_matrix(residual, H, T), op,
            "residual must be contiguous FP32 [H,T]");
    require(is_dense(out, DType::BF16) && is_matrix(out, H, T), op,
            "out must be contiguous BF16 [H,T]");
    for (const Tensor* in : {&conv.x, &conv.weight, &conv.state, &residual}) {
        require(!detail::overlaps(out, *in), op, "out must not overlap an input");
    }
    if (conv.record.data != nullptr) {
        require(!detail::overlaps(out, conv.record), op, "out must not overlap the record");
    }
    detail::residual_conv_round_launch(c, static_cast<const float*>(residual.data),
                                       static_cast<__nv_bfloat16*>(out.data), H, T, stream);
}

} // namespace ninfer::ops
