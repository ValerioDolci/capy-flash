#include "ninfer/ops/loop_contrast.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// The extrapolation is one FP32 subtraction and one fused multiply-add: the host's std::fma
// reproduces it bit for bit.
int extrapolate_case(std::int32_t rows, std::int32_t cols, float w, bool in_place,
                     std::uint32_t seed) {
    const std::size_t n = std::size_t(rows) * cols;
    std::vector<float> last(n), reference(n);
    fill_uniform(last, seed, -8.0F, 8.0F);
    fill_uniform(reference, seed + 1, -8.0F, 8.0F);
    // Most elements close to the reference, as the states of two passes are.
    for (std::size_t i = 0; i < n; i += 3) { reference[i] = last[i] * 0.97F; }
    round_to_bf16(last);
    round_to_bf16(reference);
    std::vector<std::uint16_t> expected(n);
    for (std::size_t i = 0; i < n; ++i) {
        expected[i] = f32_to_bf16(std::fma(w, last[i] - reference[i], last[i]));
    }
    DeviceBuffer dl = to_device_bf16(last), dr = to_device_bf16(reference);
    GuardedDeviceBuffer guarded(n * 2);
    const Tensor tl(dl.p, DType::BF16, {rows, cols});
    const Tensor tr(dr.p, DType::BF16, {rows, cols});
    Tensor out = in_place ? Tensor(dl.p, DType::BF16, {rows, cols})
                          : Tensor(guarded.data(), DType::BF16, {rows, cols});
    ops::loop_extrapolate(tl, tr, w, out, nullptr);
    cuda_synchronize();
    const std::string id = "loop_extrapolate [" + std::to_string(rows) + "," +
                           std::to_string(cols) + "] w " + std::to_string(w) +
                           (in_place ? " in place" : "");
    int failures = in_place ? 0 : guarded.verify_guards(id);
    failures += verify_exact(id.c_str(), from_device<std::uint16_t>(out.data, n), expected);
    std::cout << (failures ? "FAIL " : "OK ") << id << '\n';
    return failures;
}

// Soft-capped logits of a peaked distribution: a few large values over a broad body.
std::vector<float> logits_like(std::int32_t rows, std::int32_t cols, std::uint32_t seed, bool tie) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> body(0.0F, 3.0F);
    std::uniform_real_distribution<float> peak(8.0F, 26.0F);
    std::uniform_int_distribution<std::int32_t> row(0, rows - 1);
    std::vector<float> out(std::size_t(rows) * cols);
    for (std::int32_t j = 0; j < cols; ++j) {
        float* column = out.data() + std::size_t(j) * rows;
        for (std::int32_t v = 0; v < rows; ++v) {
            column[v] = 30.0F * std::tanh(body(rng) / 30.0F);
        }
        const std::int32_t peaks = 1 + j % 4;
        for (std::int32_t k = 0; k < peaks; ++k) { column[row(rng)] = peak(rng); }
        if (tie && j % 2 == 0) {
            const float top  = *std::max_element(column, column + rows);
            column[row(rng)] = top;
        }
    }
    return out;
}

// The double-precision oracle of loop_contrast_logits for column `j`.
void oracle(const std::vector<float>& c, const std::vector<float>& r, std::int32_t rows,
            std::int32_t j, float w, bool adaptive, std::vector<double>& out) {
    const float* column    = c.data() + std::size_t(j) * rows;
    const float* reference = r.data() + std::size_t(j) * rows;
    double wj              = w;
    if (adaptive) {
        double m1 = -INFINITY, m2 = -INFINITY;
        for (std::int32_t v = 0; v < rows; ++v) {
            const double x = column[v];
            if (x > m1) {
                m2 = m1;
                m1 = x;
            } else {
                m2 = std::max(m2, x);
            }
        }
        double sum = 0.0;
        for (std::int32_t v = 0; v < rows; ++v) { sum += std::exp(double(column[v]) - m1); }
        wj = double(w) * (1.0 - (1.0 / sum - std::exp(m2 - m1) / sum));
    }
    for (std::int32_t v = 0; v < rows; ++v) {
        out.push_back(double(column[v]) + wj * (double(column[v]) - double(reference[v])));
    }
}

int logits_case(std::int32_t rows, std::int32_t cols, float w, bool adaptive, bool tie,
                bool in_place, std::uint32_t seed) {
    const std::size_t n                = std::size_t(rows) * cols;
    const std::vector<float> capped    = logits_like(rows, cols, seed, tie);
    const std::vector<float> reference = logits_like(rows, cols, seed + 7, false);
    DeviceBuffer dc = to_device_f32(capped), dr = to_device_f32(reference);
    GuardedDeviceBuffer out_buffer(n * 4), logits_buffer(n * 2);
    const Tensor tc(dc.p, DType::FP32, {rows, cols});
    const Tensor tr(dr.p, DType::FP32, {rows, cols});
    Tensor out = in_place ? Tensor(dc.p, DType::FP32, {rows, cols})
                          : Tensor(out_buffer.data(), DType::FP32, {rows, cols});
    Tensor logits(logits_buffer.data(), DType::BF16, {rows, cols});
    ops::loop_contrast_logits(tc, tr, w, adaptive, out, logits, nullptr);
    cuda_synchronize();
    const std::string id = "loop_contrast_logits [" + std::to_string(rows) + "," +
                           std::to_string(cols) + "] w " + std::to_string(w) +
                           (adaptive ? " adaptive" : "") + (tie ? " ties" : "") +
                           (in_place ? " in place" : "");
    int failures = logits_buffer.verify_guards(id);
    if (!in_place) { failures += out_buffer.verify_guards(id); }
    const auto got  = from_device<float>(out.data, n);
    const auto bf16 = from_device<std::uint16_t>(logits.data, n);
    std::vector<std::uint16_t> rounded(n);
    for (std::size_t i = 0; i < n; ++i) { rounded[i] = f32_to_bf16(got[i]); }
    failures += verify_exact((id + " logits = bf16(out)").c_str(), bf16, rounded);
    if (!adaptive) {
        // Fixed strength: one subtraction and one fused multiply-add, bit for bit.
        std::vector<float> expected(n);
        for (std::size_t i = 0; i < n; ++i) {
            expected[i] = std::fma(w, capped[i] - reference[i], capped[i]);
        }
        failures += verify_exact(id.c_str(), got, expected);
    } else {
        // The FP32 softmax denominator over the column: a relative error of a few 1e-6 in the
        // strength, times |c - r| <= 60.
        constexpr PointwiseCriterion kAdaptive{/*absolute*/ 2.0e-4, /*relative*/ 1.0e-6};
        std::vector<double> expected, measured(got.begin(), got.end());
        for (std::int32_t j = 0; j < cols; ++j) {
            oracle(capped, reference, rows, j, w, adaptive, expected);
        }
        failures += verify_pointwise(id, measured, expected, kAdaptive);
    }
    // Every column alone gives the bits it has among the others.
    for (const std::int32_t j : {0, cols - 1}) {
        const std::vector<float> one_c(capped.begin() + std::ptrdiff_t(j) * rows,
                                       capped.begin() + std::ptrdiff_t(j + 1) * rows);
        const std::vector<float> one_r(reference.begin() + std::ptrdiff_t(j) * rows,
                                       reference.begin() + std::ptrdiff_t(j + 1) * rows);
        DeviceBuffer c1 = to_device_f32(one_c), r1 = to_device_f32(one_r);
        DeviceBuffer o1(std::size_t(rows) * 4), l1(std::size_t(rows) * 2);
        Tensor to(o1.p, DType::FP32, {rows, 1});
        Tensor tl(l1.p, DType::BF16, {rows, 1});
        ops::loop_contrast_logits(Tensor(c1.p, DType::FP32, {rows, 1}),
                                  Tensor(r1.p, DType::FP32, {rows, 1}), w, adaptive, to, tl,
                                  nullptr);
        cuda_synchronize();
        const std::vector<float> column(got.begin() + std::ptrdiff_t(j) * rows,
                                        got.begin() + std::ptrdiff_t(j + 1) * rows);
        failures += verify_exact((id + " column " + std::to_string(j) + " alone").c_str(),
                                 from_device<float>(o1.p, std::size_t(rows)), column);
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
    int failures = 0;
    try {
        cuda_check(cudaSetDevice(0), "cudaSetDevice");
        // The Inkling 2 Mini residual: H = 2304, decode, verification and prefill widths.
        for (const std::int32_t cols : {1, 4, 16, 4096}) {
            failures += extrapolate_case(2304, cols, 0.3F, false, 11U + std::uint32_t(cols));
        }
        failures += extrapolate_case(2304, 16, 0.5F, true, 5U);
        failures += extrapolate_case(2304, 16, -0.25F, false, 6U);
        failures += extrapolate_case(2304, 16, 0.0F, false, 7U);
        failures += extrapolate_case(1000, 3, 1.5F, false, 8U); // not a multiple of the block
        // The Inkling 2 Mini vocabulary (200058 rows): decode, a verification round, scoring.
        for (const std::int32_t cols : {1, 4, 64}) {
            failures += logits_case(200058, cols, 0.3F, false, false, false, 21U);
            failures += logits_case(200058, cols, 0.5F, true, true, false, 22U);
        }
        failures += logits_case(200058, 4, 0.5F, true, false, true, 23U);
        failures += logits_case(200058, 4, 0.3F, false, false, true, 24U);
        failures += logits_case(777, 3, 1.0F, true, true, false, 25U); // a short column
        // Contract violations.
        DeviceBuffer b(64 * 4);
        try {
            const Tensor c(b.p, DType::FP32, {16, 1});
            Tensor logits(b.p, DType::BF16, {16, 1});
            Tensor out(b.p, DType::FP32, {16, 1});
            ops::loop_contrast_logits(c, c, 0.5F, false, out, logits, nullptr);
            std::cerr << "loop_contrast_logits accepted logits over its input\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    } catch (const std::exception& error) {
        std::cerr << "loop_contrast: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " loop_contrast\n";
    return failures ? 1 : 0;
}
