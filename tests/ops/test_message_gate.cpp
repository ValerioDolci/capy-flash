#include "ninfer/ops/message_gate.h"
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

// The contract keeps two BF16 roundings (the bottleneck's input z and output d). A value within
// FP32 error of a BF16 rounding boundary may land on the neighbouring BF16 value: one such flip of
// d[r] moves d[r] by one BF16 ulp (2^-8 relative), which moves every gate argument of the token
// by forget[i,r] * ulp(d[r]) <= ~1e-3 here, so |out| moves by <= ~2e-3 of its maximum. Flips are
// rare (well under one per token), so the normwise error stays an order of magnitude below the
// pointwise cap. Everything else is FP32 arithmetic over <= 2H terms.
constexpr ReductionCriterion kMessageGateCriterion{
    /*relative_l2*/ 1.0e-3,
    /*gross_absolute*/ 1.0e-5,
    /*gross_relative_to_max_reference*/ 2.0e-3,
};

constexpr float kEps = 1.0e-6F;

double round_bf16(double value) {
    return double(bf16_to_f32(f32_to_bf16(static_cast<float>(value))));
}

struct Case {
    std::int32_t hidden;
    std::int32_t rank;
    std::int32_t tokens;
    std::uint32_t seed;
};

struct Inputs {
    std::vector<float> hidden, message, down, forget, gain;
};

// Bottleneck and gate arguments of order one, as for trained weights over normalized inputs.
Inputs make_inputs(const Case& c) {
    const std::size_t n = static_cast<std::size_t>(c.hidden) * c.tokens;
    Inputs in;
    in.hidden.resize(n);
    in.message.resize(n);
    fill_uniform(in.hidden, c.seed, -2.0F, 2.0F);
    fill_uniform(in.message, c.seed + 1, -3.0F, 3.0F);
    in.down.resize(static_cast<std::size_t>(2) * c.hidden * c.rank);
    const float a = std::sqrt(3.0F / static_cast<float>(2 * c.hidden));
    fill_uniform(in.down, c.seed + 2, -a, a);
    round_to_bf16(in.down);
    const float b = std::sqrt(3.0F / static_cast<float>(c.rank));
    in.forget.resize(static_cast<std::size_t>(c.hidden) * c.rank);
    in.gain.resize(in.forget.size());
    fill_uniform(in.forget, c.seed + 3, -b, b);
    fill_uniform(in.gain, c.seed + 4, -b, b);
    return in;
}

std::vector<double> oracle(const Case& c, const Inputs& in) {
    const std::int64_t H = c.hidden;
    const std::int64_t R = c.rank;
    std::vector<double> out(static_cast<std::size_t>(H) * c.tokens);
    std::vector<double> z(static_cast<std::size_t>(2 * H));
    std::vector<double> d(static_cast<std::size_t>(R));
    for (std::int64_t t = 0; t < c.tokens; ++t) {
        const float* h = in.hidden.data() + t * H;
        const float* m = in.message.data() + t * H;
        double hh = 0.0, mm = 0.0;
        for (std::int64_t i = 0; i < H; ++i) {
            hh += double(h[i]) * h[i];
            mm += double(m[i]) * m[i];
        }
        const double inv_h = 1.0 / std::sqrt(hh / double(H) + double(kEps));
        const double inv_m = 1.0 / std::sqrt(mm / double(H) + double(kEps));
        for (std::int64_t i = 0; i < H; ++i) {
            z[i]     = round_bf16(h[i] * inv_h);
            z[H + i] = round_bf16(m[i] * inv_m);
        }
        for (std::int64_t r = 0; r < R; ++r) {
            double sum = 0.0;
            for (std::int64_t i = 0; i < 2 * H; ++i) {
                sum += double(in.down[r * 2 * H + i]) * z[i];
            }
            d[r] = round_bf16(sum);
        }
        for (std::int64_t i = 0; i < H; ++i) {
            double f = 0.0, g = 0.0;
            for (std::int64_t r = 0; r < R; ++r) {
                f += double(in.forget[i * R + r]) * d[r];
                g += double(in.gain[i * R + r]) * d[r];
            }
            out[t * H + i] = (1.0 - std::tanh(f)) * h[i] + std::tanh(g) * m[i];
        }
    }
    return out;
}

// Runs the Op on the current device; `in_place` writes the result over `hidden`. `shift` places
// hidden, message and out one FP32 element past a 16-byte boundary: every contiguous placement is
// valid, and the result must not depend on it.
std::vector<float> run_device(const Case& c, const Inputs& in, bool in_place, int shift = 0) {
    const std::int32_t H     = c.hidden;
    const std::int32_t T     = c.tokens;
    const std::size_t n      = static_cast<std::size_t>(H) * T;
    const std::size_t padded = (n + 1) * sizeof(float);
    GuardedDeviceBuffer hidden(padded), message(padded), out(padded);
    float* h_data = static_cast<float*>(hidden.data()) + shift;
    float* m_data = static_cast<float*>(message.data()) + shift;
    float* o_data = static_cast<float*>(out.data()) + shift;
    hidden.copy_from_host(in.hidden.data(), n * sizeof(float), shift * sizeof(float));
    message.copy_from_host(in.message.data(), n * sizeof(float), shift * sizeof(float));
    DeviceBuffer down   = to_device_bf16(in.down);
    DeviceBuffer forget = to_device_f32(in.forget);
    DeviceBuffer gain   = to_device_f32(in.gain);
    out.fill(0xff);
    const Tensor th(h_data, DType::FP32, {H, T});
    const Tensor tm(m_data, DType::FP32, {H, T});
    const Tensor td(down.p, DType::BF16, {2 * H, c.rank});
    const Tensor tf(forget.p, DType::FP32, {c.rank, H});
    const Tensor tg(gain.p, DType::FP32, {c.rank, H});
    Tensor to(in_place ? h_data : o_data, DType::FP32, {H, T});
    ops::message_gate(th, tm, td, tf, tg, kEps, to, nullptr);
    cuda_synchronize();
    if (hidden.verify_guards("message_gate hidden") +
            message.verify_guards("message_gate message") + out.verify_guards("message_gate out") !=
        0) {
        throw std::runtime_error("guard overwritten");
    }
    if (!in_place && from_device<float>(h_data, n) != in.hidden) {
        throw std::runtime_error("hidden changed by an out-of-place call");
    }
    if (from_device<float>(m_data, n) != in.message) {
        throw std::runtime_error("message changed");
    }
    return from_device<float>(to.data, n);
}

int run_case(const Case& c, int device_count) {
    const Inputs in      = make_inputs(c);
    const auto expected  = oracle(c, in);
    const std::string id = "message_gate H=" + std::to_string(c.hidden) +
                           " R=" + std::to_string(c.rank) + " T=" + std::to_string(c.tokens);
    cuda_check(cudaSetDevice(0), "cudaSetDevice");
    const std::vector<float> got = run_device(c, in, false);
    std::vector<double> got64(got.begin(), got.end());
    int failures = verify_reduction(id, got64, expected, kMessageGateCriterion);
    failures += verify_exact((id + " in place").c_str(), run_device(c, in, true), got);
    failures += verify_exact((id + " shifted").c_str(), run_device(c, in, false, 1), got);
    failures += verify_exact((id + " shifted in place").c_str(), run_device(c, in, true, 1), got);
    if (device_count > 1) {
        cuda_check(cudaSetDevice(1), "cudaSetDevice");
        failures += verify_exact((id + " device 1").c_str(), run_device(c, in, false), got);
        cuda_check(cudaSetDevice(0), "cudaSetDevice");
    }
    std::cout << (failures ? "FAIL " : "OK ") << id << '\n';
    return failures;
}

int run_rejections() {
    constexpr std::int32_t H = 64;
    constexpr std::int32_t T = 2;
    DeviceBuffer h(H * T * 4), m(H * T * 4), d(2 * H * 300 * 2), w(H * 300 * 4), o(H * T * 4);
    const Tensor th(h.p, DType::FP32, {H, T});
    const Tensor tm(m.p, DType::FP32, {H, T});
    Tensor to(o.p, DType::FP32, {H, T});
    int failures = 0;
    auto expect  = [&](const char* what, auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) { return; }
        std::cerr << "message_gate accepted " << what << '\n';
        ++failures;
    };
    expect("rank above the maximum", [&] {
        const Tensor tw(w.p, DType::FP32, {300, H});
        ops::message_gate(th, tm, Tensor(d.p, DType::BF16, {2 * H, 300}), tw, tw, kEps, to,
                          nullptr);
    });
    expect("FP32 down", [&] {
        const Tensor tw(w.p, DType::FP32, {16, H});
        ops::message_gate(th, tm, Tensor(d.p, DType::FP32, {2 * H, 16}), tw, tw, kEps, to, nullptr);
    });
    expect("transposed forget", [&] {
        const Tensor tw(w.p, DType::FP32, {H, 16});
        ops::message_gate(th, tm, Tensor(d.p, DType::BF16, {2 * H, 16}), tw, tw, kEps, to, nullptr);
    });
    expect("out shifted over hidden", [&] {
        const Tensor tw(w.p, DType::FP32, {16, H});
        Tensor shifted(static_cast<float*>(h.p) + 1, DType::FP32, {H, T - 1});
        const Tensor th1(h.p, DType::FP32, {H, T - 1});
        const Tensor tm1(m.p, DType::FP32, {H, T - 1});
        ops::message_gate(th1, tm1, Tensor(d.p, DType::BF16, {2 * H, 16}), tw, tw, kEps, shifted,
                          nullptr);
    });
    std::cout << (failures ? "FAIL " : "OK ") << "message_gate rejections\n";
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
        failures += run_rejections();
        std::uint32_t seed = 3;
        for (const std::int32_t T : {1, 2, 4, 6, 8, 15, 16, 17, 40, 64, 256}) {
            failures += run_case({2304, 64, T, seed += 10}, device_count);
        }
        for (const std::int32_t R : {16, 64}) {
            for (const std::int32_t T : {1, 16}) {
                failures += run_case({512, R, T, seed += 10}, device_count);
            }
        }
        failures += run_case({2304, ops::kMessageGateMaxRank, 4, seed += 10}, device_count);
    } catch (const std::exception& error) {
        std::cerr << "message_gate: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " message_gate\n";
    return failures ? 1 : 0;
}
