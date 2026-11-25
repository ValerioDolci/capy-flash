// ninfer::ops - loop_contrast wrappers: validate the public contract, then launch.
#include "ninfer/ops/loop_contrast.h"

#include "ops/common/validate.h"
#include "ops/loop_contrast/launch.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* op, const std::string& message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

bool same_shape(const Tensor& a, const Tensor& b) {
    for (int d = 0; d < 4; ++d) {
        if (a.ne[d] != b.ne[d]) { return false; }
    }
    return true;
}

bool usable(const Tensor& t, DType dtype) {
    return t.dtype == dtype && t.data != nullptr && t.is_contiguous() && t.numel() > 0;
}

} // namespace

void loop_extrapolate(const Tensor& last, const Tensor& reference, float w, Tensor& out,
                      cudaStream_t stream) {
    constexpr const char* op = "loop_extrapolate";
    require(usable(last, DType::BF16) && usable(reference, DType::BF16) && usable(out, DType::BF16),
            op, "last, reference and out must be contiguous non-empty BF16");
    require(same_shape(last, reference) && same_shape(last, out), op,
            "last, reference and out must have the same shape");
    require(std::isfinite(w), op, "the strength must be finite");
    require(!detail::overlaps(out, reference), op, "out must not overlap reference");
    require(out.data == last.data || !detail::overlaps(out, last), op,
            "out must be last itself or not overlap it");
    detail::loop_extrapolate_launch(last, reference, w, out, stream);
}

void loop_contrast_logits(const Tensor& capped, const Tensor& reference, float w, bool adaptive,
                          Tensor& out, Tensor& logits, cudaStream_t stream) {
    constexpr const char* op = "loop_contrast_logits";
    require(usable(capped, DType::FP32) && usable(reference, DType::FP32) &&
                usable(out, DType::FP32) && usable(logits, DType::BF16),
            op, "capped, reference, out (FP32) and logits (BF16) must be contiguous non-empty");
    require(capped.ne[2] == 1 && capped.ne[3] == 1 && same_shape(capped, reference) &&
                same_shape(capped, out) && same_shape(capped, logits),
            op, "every operand must be [V, n] like capped");
    require(std::isfinite(w), op, "the strength must be finite");
    require(!detail::overlaps(out, reference) && !detail::overlaps(logits, capped) &&
                !detail::overlaps(logits, reference) && !detail::overlaps(logits, out),
            op, "out and logits must not overlap the other operands");
    require(out.data == capped.data || !detail::overlaps(out, capped), op,
            "out must be capped itself or not overlap it");
    detail::loop_contrast_logits_launch(capped, reference, w, adaptive, out, logits, stream);
}

} // namespace ninfer::ops
