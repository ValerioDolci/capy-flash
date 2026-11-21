// ninfer::ops - moe_sigmoid_route: validate the public contract, launch.
#include "ninfer/ops/moe_sigmoid_route.h"

#include "ops/moe_sigmoid_route/launch.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr const char* kOp = "moe_sigmoid_route";

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::invalid_argument(std::string(kOp) + ": " + message); }
}

void require_dense(const Tensor& t, DType dtype, const char* label) {
    require(t.dtype == dtype && t.data != nullptr && t.is_contiguous() && t.numel() > 0,
            std::string(label) + " must be contiguous and non-empty with the expected dtype");
}

bool matrix(const Tensor& t, std::int32_t rows, std::int32_t columns) {
    return t.ne[0] == rows && t.ne[1] == columns && t.ne[2] == 1 && t.ne[3] == 1;
}

} // namespace

void moe_sigmoid_route(const Tensor& x, const Tensor& router, const Tensor& bias,
                       std::int32_t top_k, float route_scale, Tensor& ids, Tensor& weights,
                       cudaStream_t stream, const Tensor* valid_columns) {
    require_dense(x, DType::BF16, "x");
    require_dense(router, DType::FP32, "router");
    require_dense(bias, DType::FP32, "bias");
    require_dense(ids, DType::I32, "ids");
    require_dense(weights, DType::FP32, "weights");
    const std::int32_t hidden = x.ne[0], columns = x.ne[1], experts = router.ne[1];
    require(matrix(x, hidden, columns), "x must be [H,T]");
    require(matrix(router, hidden, experts), "router must be FP32 [H,E]");
    require(experts >= 1 && experts <= kMoeSigmoidRouteMaxExperts, "E outside [1, 4096]");
    require(bias.numel() == experts, "bias must be [E]");
    require(top_k >= 1 && top_k <= experts && top_k <= kMoeSigmoidRouteMaxTopK,
            "top_k outside [1, min(E, 64)]");
    require(std::isfinite(route_scale), "route_scale must be finite");
    require(matrix(ids, top_k, columns), "ids must be [k,T]");
    require(matrix(weights, top_k, columns), "weights must be [k,T]");
    if (valid_columns != nullptr) {
        require_dense(*valid_columns, DType::I32, "valid columns");
        require(valid_columns->numel() == valid_columns->ne[0] &&
                    columns % valid_columns->ne[0] == 0,
                "valid columns must be [B] with B dividing T");
    }
    detail::moe_sigmoid_route_launch(x, router, bias, top_k, route_scale, ids, weights,
                                     valid_columns, stream);
}

} // namespace ninfer::ops
