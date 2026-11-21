// ninfer::ops - softcap wrapper: validates the public contract, then launches.
#include "ninfer/ops/softcap.h"

#include "ops/common/validate.h"
#include "ops/softcap/launch.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::invalid_argument("softcap: " + message); }
}

} // namespace

void softcap(const Tensor& x, float cap, Tensor& out, cudaStream_t stream) {
    require(x.dtype == DType::BF16 && x.data != nullptr && x.is_contiguous() && x.numel() > 0,
            "x must be contiguous non-empty BF16");
    require(out.dtype == DType::FP32 && out.data != nullptr && out.is_contiguous(),
            "out must be contiguous FP32");
    for (int d = 0; d < 4; ++d) { require(out.ne[d] == x.ne[d], "out must match x"); }
    require(std::isfinite(cap) && cap > 0.0F, "cap must be finite and positive");
    require(!detail::overlaps(out, x), "out must not overlap x");
    detail::softcap_launch(x, cap, out, stream);
}

} // namespace ninfer::ops
