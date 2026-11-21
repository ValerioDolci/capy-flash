// ninfer::ops - rmsnorm_f32 wrapper: validates the public contract, then launches.
#include "ninfer/ops/rmsnorm_f32.h"

#include "ops/common/validate.h"
#include "ops/rmsnorm_f32/launch.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::invalid_argument("rmsnorm_f32: " + message); }
}

bool is_dense(const Tensor& t, DType dtype) {
    return t.dtype == dtype && t.data != nullptr && t.is_contiguous() && t.numel() > 0;
}

} // namespace

void rmsnorm_f32(const Tensor& x, const Tensor& weight, float eps, Tensor& out,
                 cudaStream_t stream) {
    require(is_dense(x, DType::FP32), "x must be contiguous non-empty FP32");
    require(is_dense(out, DType::BF16), "out must be contiguous non-empty BF16");
    for (int d = 0; d < 4; ++d) { require(out.ne[d] == x.ne[d], "out must match x"); }
    require(is_dense(weight, DType::BF16) && weight.numel() == x.ne[0],
            "weight must be contiguous BF16 [D]");
    require(x.numel() / x.ne[0] <= std::numeric_limits<int>::max(), "too many rows");
    require(std::isfinite(eps) && eps > 0.0F, "eps must be finite and positive");
    require(!detail::overlaps(out, x) && !detail::overlaps(out, weight),
            "out must not overlap an input");
    detail::rmsnorm_f32_launch(x, weight, eps, out, stream);
}

} // namespace ninfer::ops
