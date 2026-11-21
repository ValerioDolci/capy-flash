// ninfer::ops - short convolution wrappers: validate the public contract, then launch.
#include "ninfer/ops/short_conv.h"

#include "ops/common/validate.h"
#include "ops/short_conv/launch.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* op, const std::string& message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

void require_dense(const Tensor& t, DType dtype, const char* op, const char* label) {
    require(t.dtype == dtype && t.data != nullptr && t.is_contiguous() && t.numel() > 0, op,
            std::string(label) + " must be contiguous non-empty " +
                (dtype == DType::FP32 ? "FP32" : "BF16"));
}

bool is_matrix(const Tensor& t, std::int32_t rows, std::int32_t columns) {
    return t.ne[0] == rows && t.ne[1] == columns && t.ne[2] == 1 && t.ne[3] == 1;
}

// Checks shared by both tap orders: activation, output, state and their alias rules.
void require_common(const char* op, const Tensor& x, float scale, const Tensor& state_in,
                    const Tensor& state_out, const Tensor& out) {
    require_dense(x, DType::BF16, op, "x");
    const std::int32_t channels = x.ne[0];
    const std::int32_t tokens   = x.ne[1];
    require(is_matrix(x, channels, tokens), op, "x must be [C,T]");
    require_dense(out, DType::BF16, op, "out");
    require(is_matrix(out, channels, tokens), op, "out must match x");
    for (const Tensor* state : {&state_in, &state_out}) {
        require_dense(*state, DType::BF16, op, "state");
        require(is_matrix(*state, channels, kShortConvTaps - 1), op, "states must be ne={C,3}");
    }
    require(state_in.data == state_out.data || !detail::overlaps(state_in, state_out), op,
            "state_out must be disjoint from state_in or exactly the same storage");
    require(std::isfinite(scale) && scale > 0.0F, op, "scale must be finite and positive");
    for (const Tensor* input : {&x, &state_in, &state_out}) {
        require(!detail::overlaps(*input, out), op, "out must not overlap x or a state");
    }
    require(!detail::overlaps(x, state_out), op, "state_out must not overlap x");
}

} // namespace

void short_conv_static(const Tensor& x, const Tensor& weight, float scale, const Tensor& state_in,
                       Tensor& state_out, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "short_conv_static";
    require_common(op, x, scale, state_in, state_out, out);
    require_dense(weight, DType::FP32, op, "weight");
    require(is_matrix(weight, kShortConvTaps, x.ne[0]), op, "weight must be ne={4,C}");
    require(!detail::overlaps(weight, out) && !detail::overlaps(weight, state_out), op,
            "weight must not overlap an output");
    detail::short_conv_launch(x, weight, nullptr, scale, state_in, state_out, out, stream);
}

void short_conv_dynamic(const Tensor& x, const Tensor& static_weight, const Tensor& dyn,
                        float scale, const Tensor& state_in, Tensor& state_out, Tensor& out,
                        cudaStream_t stream) {
    constexpr const char* op = "short_conv_dynamic";
    require_common(op, x, scale, state_in, state_out, out);
    require_dense(static_weight, DType::FP32, op, "static_weight");
    require(is_matrix(static_weight, x.ne[0], kShortConvTaps), op,
            "static_weight must be ne={C,4}");
    require_dense(dyn, DType::BF16, op, "dyn");
    require(is_matrix(dyn, dyn.ne[0], x.ne[1]) && dyn.ne[0] % kShortConvTaps == 0, op,
            "dyn must be ne={4G,T}");
    const std::int32_t groups = dyn.ne[0] / kShortConvTaps;
    require(x.ne[0] % groups == 0, op, "the channel count must be a multiple of the groups");
    for (const Tensor* input : {&static_weight, &dyn}) {
        require(!detail::overlaps(*input, out) && !detail::overlaps(*input, state_out), op,
                "static_weight and dyn must not overlap an output");
    }
    detail::short_conv_launch(x, static_weight, &dyn, scale, state_in, state_out, out, stream);
}

} // namespace ninfer::ops
