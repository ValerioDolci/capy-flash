// ninfer::ops - FP32 residual wrappers: validate the public contract, then launch.
#include "ninfer/ops/residual_f32.h"

#include "ops/common/validate.h"
#include "ops/residual_f32/launch.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* op, const std::string& message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

void require_like(const Tensor& t, DType dtype, const Tensor& shape, const char* op,
                  const char* label) {
    require(t.dtype == dtype && t.data != nullptr && t.is_contiguous() && t.numel() > 0, op,
            std::string(label) + " must be contiguous non-empty " +
                (dtype == DType::FP32 ? "FP32" : "BF16"));
    for (int d = 0; d < 4; ++d) {
        require(t.ne[d] == shape.ne[d], op, std::string(label) + " must match the other operands");
    }
}

} // namespace

void residual_add_f32(const Tensor* x, const Tensor& y, Tensor& sum, cudaStream_t stream) {
    constexpr const char* op = "residual_add_f32";
    require_like(y, DType::BF16, y, op, "y");
    require_like(sum, DType::FP32, y, op, "sum");
    require(!detail::overlaps(sum, y), op, "sum must not overlap y");
    if (x != nullptr) {
        require_like(*x, DType::BF16, y, op, "x");
        require(!detail::overlaps(sum, *x), op, "sum must not overlap x");
    }
    detail::residual_add_f32_launch(x, y, sum, stream);
}

void residual_add_round_bf16(const Tensor& sum, const Tensor& z, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "residual_add_round_bf16";
    require_like(z, DType::BF16, z, op, "z");
    require_like(sum, DType::FP32, z, op, "sum");
    require_like(out, DType::BF16, z, op, "out");
    require(!detail::overlaps(out, sum), op, "out must not overlap sum");
    require(out.data == z.data || !detail::overlaps(out, z), op,
            "out must be disjoint from z or exactly the same storage");
    detail::residual_add_round_bf16_launch(sum, z, out, stream);
}

} // namespace ninfer::ops
