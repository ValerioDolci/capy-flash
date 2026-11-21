#include "ninfer/ops/rmsnorm_f32.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// One FP32 sum of squares over D <= 2304 terms (~1e-6 relative) and one BF16 rounding of each
// element (at most u = 2^-8 relative); the bound is 2u.
constexpr PointwiseCriterion kRmsnormF32Criterion{/*absolute*/ 1.0e-6, /*relative*/ 1.0 / 128.0};

constexpr float kEps = 1.0e-6F;

struct Case {
    std::int32_t width;
    std::int32_t tokens;
    std::int32_t blocks; // ne[2]: the query blocks of a depth_attention output
    float magnitude;
    std::uint32_t seed;
};

std::vector<std::uint16_t> run_device(const Case& c, const std::vector<float>& x,
                                      const std::vector<float>& weight) {
    const std::size_t n = x.size();
    DeviceBuffer dx     = to_device_f32(x);
    DeviceBuffer dw     = to_device_bf16(weight);
    GuardedDeviceBuffer out(n * sizeof(std::uint16_t));
    out.fill(0xff);
    const Tensor tx(dx.p, DType::FP32, {c.width, c.tokens, c.blocks});
    const Tensor tw(dw.p, DType::BF16, {c.width});
    Tensor to(out.data(), DType::BF16, {c.width, c.tokens, c.blocks});
    ops::rmsnorm_f32(tx, tw, kEps, to, nullptr);
    cuda_synchronize();
    if (out.verify_guards("rmsnorm_f32") != 0) { throw std::runtime_error("guard overwritten"); }
    return from_device<std::uint16_t>(out.data(), n);
}

int run_case(const Case& c, int device_count) {
    const std::int64_t D = c.width;
    const std::size_t n  = static_cast<std::size_t>(D) * c.tokens * c.blocks;
    std::vector<float> x(n), weight(static_cast<std::size_t>(D));
    fill_uniform(x, c.seed, -c.magnitude, c.magnitude);
    fill_uniform(weight, c.seed + 1, -2.0F, 2.0F);
    round_to_bf16(weight);
    std::vector<double> expected(n);
    for (std::size_t row = 0; row < n / D; ++row) {
        double squares = 0.0;
        for (std::int64_t d = 0; d < D; ++d) { squares += double(x[row * D + d]) * x[row * D + d]; }
        const double inv = 1.0 / std::sqrt(squares / double(D) + double(kEps));
        for (std::int64_t d = 0; d < D; ++d) {
            expected[row * D + d] = double(x[row * D + d]) * inv * double(weight[d]);
        }
    }
    const std::string id =
        "rmsnorm_f32 D=" + std::to_string(c.width) + " T=" + std::to_string(c.tokens) +
        " blocks=" + std::to_string(c.blocks) + " |x|<=" + std::to_string(c.magnitude);
    cuda_check(cudaSetDevice(0), "cudaSetDevice");
    const std::vector<std::uint16_t> got = run_device(c, x, weight);
    std::vector<double> got64(n);
    for (std::size_t i = 0; i < n; ++i) { got64[i] = bf16_to_f32(got[i]); }
    int failures = verify_pointwise(id, got64, expected, kRmsnormF32Criterion);
    if (device_count > 1) {
        cuda_check(cudaSetDevice(1), "cudaSetDevice");
        failures += verify_exact((id + " device 1").c_str(), run_device(c, x, weight), got);
        cuda_check(cudaSetDevice(0), "cudaSetDevice");
    }
    std::cout << (failures ? "FAIL " : "OK ") << id << '\n';
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    int failures = 0;
    try {
        failures += run_case({2304, 1, 1, 3.0F, 1}, device_count);
        failures += run_case({2304, 16, 1, 3.0F, 2}, device_count);
        failures += run_case({2304, 4, 3, 0.01F, 3}, device_count);
        failures += run_case({2304, 512, 1, 500.0F, 4}, device_count);
        failures += run_case({512, 16, 3, 3.0F, 5}, device_count);
        failures += run_case({100, 3, 1, 3.0F, 6}, device_count);
        DeviceBuffer x(64 * 4), w(64 * 2), o(64 * 2);
        try {
            const Tensor tx(x.p, DType::BF16, {64});
            Tensor to(o.p, DType::BF16, {64});
            ops::rmsnorm_f32(tx, Tensor(w.p, DType::BF16, {64}), kEps, to, nullptr);
            std::cerr << "rmsnorm_f32 accepted a BF16 input\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    } catch (const std::exception& error) {
        std::cerr << "rmsnorm_f32: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " rmsnorm_f32\n";
    return failures ? 1 : 0;
}
