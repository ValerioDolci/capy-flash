#include "ninfer/ops/residual_f32.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// Exact oracle: the sum of two FP32 values is exact in FP64 whenever it can matter (an exponent gap
// above 29 leaves the smaller operand below half an FP32 ulp of the larger, in FP64 and in FP32),
// so one conversion of the FP64 sum to FP32 is the one IEEE FP32 addition of the contract.
float fl32_add(float a, float b) { return static_cast<float>(double(a) + double(b)); }

std::vector<float> bf16_values(std::size_t n, std::uint32_t seed) {
    // Uniform values times a power of two spread over a wide exponent range.
    std::vector<float> v(n), e(n);
    fill_uniform(v, seed, -1.0F, 1.0F);
    fill_uniform(e, seed + 1, -12.0F, 12.0F);
    for (std::size_t i = 0; i < n; ++i) { v[i] = std::ldexp(v[i], static_cast<int>(e[i])); }
    round_to_bf16(v);
    return v;
}

int run_case(std::int32_t width, std::int32_t tokens, bool with_x, std::uint32_t seed) {
    const std::size_t n  = static_cast<std::size_t>(width) * tokens;
    std::vector<float> x = bf16_values(n, seed);
    std::vector<float> y = bf16_values(n, seed + 2);
    std::vector<float> z = bf16_values(n, seed + 4);
    // Fixed cases at the front: cancellation, signed zeros, infinity, and FP32 sums that sit
    // exactly between two BF16 values (ties to even, both directions).
    const float ties[][3] = {
        {1.0F, -1.0F, 0.0F},
        {-0.0F, -0.0F, -0.0F},
        {std::numeric_limits<float>::infinity(), 1.0F, 1.0F},
        {1.0F, 0.0078125F, 0.0F},        // 1 + 2^-7 is a BF16 value
        {1.0F, 0.00390625F, 0.0F},       // 1 + 2^-8: tie, rounds down to 1
        {1.0F, 0.0078125F, 0.00390625F}, // 1 + 3 * 2^-8: tie, rounds up to 1 + 2^-6
        {256.0F, 1.0F, 0.001953125F},    // a sum needing the FP32 significand
        {65536.0F, 0.0078125F, 0.0F}     // 2^16 + 2^-7: exact in FP32, not in BF16
    };
    for (std::size_t k = 0; k < std::size(ties) && k < n; ++k) {
        x[k] = ties[k][0];
        y[k] = ties[k][1];
        z[k] = ties[k][2];
    }
    std::vector<float> sum(n);
    std::vector<std::uint16_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        sum[i] = with_x ? fl32_add(x[i], y[i]) : y[i];
        out[i] = f32_to_bf16(fl32_add(sum[i], z[i]));
    }

    DeviceBuffer dx = to_device_bf16(x), dy = to_device_bf16(y), dz = to_device_bf16(z);
    GuardedDeviceBuffer dsum(n * sizeof(float)), dout(n * sizeof(std::uint16_t));
    dsum.fill(0xff);
    dout.fill(0xff);
    const Tensor tx(dx.p, DType::BF16, {width, tokens});
    const Tensor ty(dy.p, DType::BF16, {width, tokens});
    const Tensor tz(dz.p, DType::BF16, {width, tokens});
    Tensor ts(dsum.data(), DType::FP32, {width, tokens});
    Tensor to(dout.data(), DType::BF16, {width, tokens});
    ops::residual_add_f32(with_x ? &tx : nullptr, ty, ts, nullptr);
    ops::residual_add_round_bf16(ts, tz, to, nullptr);
    // In place over z.
    Tensor tz_out(dz.p, DType::BF16, {width, tokens});
    ops::residual_add_round_bf16(ts, tz, tz_out, nullptr);
    cuda_synchronize();

    const std::string id = "residual_f32 [" + std::to_string(width) + "," + std::to_string(tokens) +
                           "]" + (with_x ? "" : " block start");
    std::vector<std::uint32_t> sum_bits(n);
    std::memcpy(sum_bits.data(), sum.data(), n * sizeof(float));
    int failures =
        verify_exact((id + " sum").c_str(), from_device<std::uint32_t>(dsum.data(), n), sum_bits);
    failures +=
        verify_exact((id + " out").c_str(), from_device<std::uint16_t>(dout.data(), n), out);
    failures +=
        verify_exact((id + " out in place").c_str(), from_device<std::uint16_t>(dz.p, n), out);
    failures += dsum.verify_guards(id) + dout.verify_guards(id);
    std::cout << (failures ? "FAIL " : "OK ") << id << '\n';
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    try {
        failures += run_case(2304, 1, true, 1);
        failures += run_case(2304, 1, false, 11);
        failures += run_case(2304, 16, true, 21);
        failures += run_case(2304, 512, true, 31);
        failures += run_case(2304, 512, false, 41);
        failures += run_case(512, 7, true, 51);
        DeviceBuffer b(64 * 2);
        try {
            Tensor overlap(b.p, DType::FP32, {32});
            ops::residual_add_f32(nullptr, Tensor(b.p, DType::BF16, {32}), overlap, nullptr);
            std::cerr << "residual_add_f32 accepted a sum over y\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    } catch (const std::exception& error) {
        std::cerr << "residual_f32: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " residual_f32\n";
    return failures ? 1 : 0;
}
