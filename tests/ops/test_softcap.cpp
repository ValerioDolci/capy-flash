#include "ninfer/ops/softcap.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// One FP32 division, tanh (a few ulp) and multiplication: well under 1e-6 relative. The absolute
// floor covers inputs whose quotient by the cap is an FP32 subnormal.
constexpr PointwiseCriterion kSoftcapCriterion{/*absolute*/ 1.0e-37, /*relative*/ 1.0e-6};

constexpr float kCap = 30.0F;

std::vector<float> run_device(const std::vector<float>& x, std::int32_t rows, std::int32_t cols) {
    const std::size_t n = x.size();
    DeviceBuffer dx     = to_device_bf16(x);
    GuardedDeviceBuffer out(n * sizeof(float));
    out.fill(0xff);
    const Tensor tx(dx.p, DType::BF16, {rows, cols});
    Tensor to(out.data(), DType::FP32, {rows, cols});
    ops::softcap(tx, kCap, to, nullptr);
    cuda_synchronize();
    if (out.verify_guards("softcap") != 0) { throw std::runtime_error("guard overwritten"); }
    return from_device<float>(out.data(), n);
}

std::vector<std::uint32_t> bits(const std::vector<float>& v) {
    std::vector<std::uint32_t> b(v.size());
    std::memcpy(b.data(), v.data(), v.size() * sizeof(float));
    return b;
}

int run_case(const std::string& id, std::vector<float> x, std::int32_t rows, int device_count) {
    round_to_bf16(x);
    const std::int32_t cols = static_cast<std::int32_t>(x.size() / rows);
    cuda_check(cudaSetDevice(0), "cudaSetDevice");
    const std::vector<float> got = run_device(x, rows, cols);
    std::vector<double> finite_got, finite_expected;
    int failures = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double xi = x[i];
        if (std::isnan(xi)) {
            if (!std::isnan(got[i])) {
                std::cerr << id << ": NaN input gave " << got[i] << '\n';
                ++failures;
            }
            continue;
        }
        const double expected = std::isinf(xi) ? std::copysign(double(kCap), xi)
                                               : double(kCap) * std::tanh(xi / double(kCap));
        if (xi == 0.0 && std::signbit(got[i]) != std::signbit(xi)) {
            std::cerr << id << ": the sign of zero was not kept\n";
            ++failures;
        }
        if (std::isinf(xi) && got[i] != static_cast<float>(expected)) {
            std::cerr << id << ": infinity gave " << got[i] << '\n';
            ++failures;
        }
        finite_got.push_back(got[i]);
        finite_expected.push_back(expected);
    }
    failures += verify_pointwise(id, finite_got, finite_expected, kSoftcapCriterion);
    if (device_count > 1) {
        cuda_check(cudaSetDevice(1), "cudaSetDevice");
        failures +=
            verify_exact((id + " device 1").c_str(), bits(run_device(x, rows, cols)), bits(got));
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
        const float inf   = std::numeric_limits<float>::infinity();
        const float big   = bf16_to_f32(0x7f7fU); // largest finite BF16
        const float small = bf16_to_f32(0x0001U); // smallest positive BF16 subnormal
        std::vector<float> extremes{
            0.0F,    -0.0F, small,  -small,  1.0e-38F, -1.0e-38F,
            1.0e-3F, 1.0F,  -1.0F,  29.875F, -29.875F, 30.0F,
            -30.0F,  60.0F, -60.0F, 448.0F,  -448.0F,  1.0e4F,
            -1.0e4F, big,   -big,   inf,     -inf,     std::numeric_limits<float>::quiet_NaN()};
        failures += run_case("softcap extremes", extremes,
                             static_cast<std::int32_t>(extremes.size()), device_count);
        // Logits of the vocabulary-split head: one rank's half and the whole vocabulary.
        for (const auto& [rows, cols] : {std::pair{100512, 1}, std::pair{100512, 3},
                                         std::pair{201024, 2}, std::pair{4096, 16}}) {
            std::vector<float> x(static_cast<std::size_t>(rows) * cols);
            fill_uniform(x, static_cast<std::uint32_t>(rows + cols), -120.0F, 120.0F);
            failures +=
                run_case("softcap [" + std::to_string(rows) + "," + std::to_string(cols) + "]", x,
                         rows, device_count);
        }
        DeviceBuffer b(64 * 4);
        try {
            const Tensor tx(b.p, DType::BF16, {64});
            Tensor to(b.p, DType::FP32, {64});
            ops::softcap(tx, kCap, to, nullptr);
            std::cerr << "softcap accepted an output over its input\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    } catch (const std::exception& error) {
        std::cerr << "softcap: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " softcap\n";
    return failures ? 1 : 0;
}
