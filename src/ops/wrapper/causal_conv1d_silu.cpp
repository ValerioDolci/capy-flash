// ninfer::ops - causal_conv1d wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/causal_conv1d_silu.h"

#include "ops/launcher/causal_conv1d.h" // detail::causal_conv1d_*_launch
#include "ops/common/validate.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::ops {
namespace {

std::int64_t numel_allow_zero(const Tensor& t, const char* label) {
    bool has_zero = false;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument(std::string("causal_conv1d: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { has_zero = true; }
    }
    if (has_zero) { return 0; }

    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("causal_conv1d: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

void require_x_shape(const Tensor& x) {
    if (x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("causal_conv1d: x must have shape [C,T]");
    }
    if (x.ne[0] <= 0) { throw std::invalid_argument("causal_conv1d: C must be positive"); }
}

void require_weight_shape(const Tensor& weight, std::int32_t C) {
    if (weight.ne[0] != C || weight.ne[1] != 4 || weight.ne[2] != 1 || weight.ne[3] != 1) {
        throw std::invalid_argument("causal_conv1d: weight must have shape [C,4]");
    }
}

void require_state_shape(const Tensor& conv_state, std::int32_t C) {
    if (conv_state.ne[0] != C || conv_state.ne[1] != 3 || conv_state.ne[2] != 1 ||
        conv_state.ne[3] != 1) {
        throw std::invalid_argument("causal_conv1d: conv_state must have shape [C,3]");
    }
}

void require_accessible(const Tensor& tensor, const char* label) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("causal_conv1d: ") + label + " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string("causal_conv1d: ") + label +
                                    " data pointer must be non-null");
    }
}

detail::CausalConvSplitGeometry resolve_split_geometry(const Tensor& x, const Tensor& out0,
                                                       const Tensor& out1, const Tensor& out2) {
    if (out0.dtype != DType::BF16 || out1.dtype != DType::BF16 || out2.dtype != DType::BF16) {
        throw std::invalid_argument("causal_conv1d: split destinations must be BF16");
    }
    if (out0.ne[2] != 1 || out0.ne[3] != 1 || out1.ne[2] != 1 || out1.ne[3] != 1 ||
        out2.ne[2] != 1 || out2.ne[3] != 1) {
        throw std::invalid_argument("causal_conv1d: split destinations must be rank-2");
    }
    if (out0.ne[1] != x.ne[1] || out1.ne[1] != x.ne[1] || out2.ne[1] != x.ne[1]) {
        throw std::invalid_argument("causal_conv1d: split destinations must match the x column "
                                    "count");
    }
    if (x.ne[0] == 5120 && out0.ne[0] == 1024 && out1.ne[0] == 1024 && out2.ne[0] == 3072) {
        return detail::CausalConvSplitGeometry::Rows1024x1024x3072;
    }
    if (x.ne[0] == 10240 && out0.ne[0] == 2048 && out1.ne[0] == 2048 && out2.ne[0] == 6144) {
        return detail::CausalConvSplitGeometry::Rows2048x2048x6144;
    }
    throw std::invalid_argument("causal_conv1d: split received an unregistered row profile");
}

// The prefill split route reads and writes __nv_bfloat162. The requirement is stated uniformly for
// the entry rather than per route, so a caller does not have to know which route its column count
// selects.
void require_pair_aligned(const Tensor& tensor, const char* label) {
    constexpr std::uintptr_t kPairBytes = 4; // one BF16 pair
    const auto address                  = reinterpret_cast<std::uintptr_t>(tensor.data);
    if ((address & (kPairBytes - 1)) != 0) {
        throw std::invalid_argument(std::string("causal_conv1d: ") + label +
                                    " must be four-byte aligned");
    }
}

// The contract allows the two state arguments to be disjoint or exactly the same storage, and
// nothing in between: every route either reads a state element before publishing it from another
// thread, or declares both pointers restrict.
void require_state_alias_rule(const Tensor& conv_state_in, const Tensor& conv_state_out) {
    if (conv_state_in.data == conv_state_out.data) { return; }
    if (detail::overlaps(conv_state_in, conv_state_out)) {
        throw std::invalid_argument("causal_conv1d: conv_state_in and conv_state_out must be "
                                    "disjoint or exactly the same storage");
    }
}

void require_split_nonoverlap(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                              const Tensor& conv_state_out, const Tensor& out0, const Tensor& out1,
                              const Tensor& out2) {
    const Tensor* const writes[4] = {&out0, &out1, &out2, &conv_state_out};
    const Tensor* const others[4] = {&x, &weight, &conv_state_in, &conv_state_out};
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            if (detail::overlaps(*writes[i], *writes[j])) {
                throw std::invalid_argument(
                    "causal_conv1d: split destinations must not overlap each other");
            }
        }
        for (const Tensor* other : others) {
            // The state pair is governed by require_state_alias_rule, which admits an exact alias.
            if (writes[i] == &conv_state_out && other == &conv_state_in) { continue; }
            if (writes[i] == other) { continue; }
            if (detail::overlaps(*writes[i], *other)) {
                throw std::invalid_argument(
                    "causal_conv1d: split destinations must not overlap x, weight, or the state");
            }
        }
    }
}

} // namespace

void causal_conv1d_silu_split(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                              Tensor& conv_state_out, Tensor& out0, Tensor& out1, Tensor& out2,
                              cudaStream_t stream) {
    if (x.dtype != DType::BF16 || weight.dtype != DType::BF16 ||
        conv_state_in.dtype != DType::BF16 || conv_state_out.dtype != DType::BF16) {
        throw std::invalid_argument("causal_conv1d: x/weight/conv_state must be BF16");
    }
    require_x_shape(x);
    require_weight_shape(weight, x.ne[0]);
    require_state_shape(conv_state_in, x.ne[0]);
    require_state_shape(conv_state_out, x.ne[0]);
    const detail::CausalConvSplitGeometry geometry = resolve_split_geometry(x, out0, out1, out2);

    const std::int64_t n = numel_allow_zero(x, "x");
    (void)numel_allow_zero(weight, "weight");
    (void)numel_allow_zero(conv_state_in, "conv_state_in");
    (void)numel_allow_zero(conv_state_out, "conv_state_out");
    (void)numel_allow_zero(out0, "out0");
    (void)numel_allow_zero(out1, "out1");
    (void)numel_allow_zero(out2, "out2");
    if (n == 0) { return; }

    const std::pair<const Tensor*, const char*> operands[7] = {{&x, "x"},
                                                               {&weight, "weight"},
                                                               {&conv_state_in, "conv_state_in"},
                                                               {&conv_state_out, "conv_state_out"},
                                                               {&out0, "out0"},
                                                               {&out1, "out1"},
                                                               {&out2, "out2"}};
    for (const auto& operand : operands) {
        require_accessible(*operand.first, operand.second);
        require_pair_aligned(*operand.first, operand.second);
    }
    require_state_alias_rule(conv_state_in, conv_state_out);
    require_split_nonoverlap(x, weight, conv_state_in, conv_state_out, out0, out1, out2);

    // Two routes, chosen by measurement. The small-T kernel launches one block of kChannelTile by T
    // threads, so it holds two CTAs per SM to T = 24 and one above that; even at its ceiling of 32
    // it beats the prefill pair, which is flat above it.
    if (x.ne[1] <= detail::kCausalConvParallelMaxTokens) {
        detail::causal_conv1d_smallt_split_launch(x, weight, conv_state_in, conv_state_out, out0,
                                                  out1, out2, geometry, stream);
    } else {
        detail::causal_conv1d_prefill_split_launch(x, weight, conv_state_in, conv_state_out, out0,
                                                   out1, out2, geometry, stream);
    }
}

} // namespace ninfer::ops
