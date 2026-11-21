#include "ninfer/ops/depth_attention.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// FP32 RMS reductions and dot products over H <= 2304 elements (~1e-7 to 1e-6 relative each), an
// FP32 softmax over <= 9 logits of magnitude <= ~10 and a convex combination of the normalized
// states, all in FP32 with no intermediate rounding to a narrower format.
constexpr ReductionCriterion kDepthAttentionCriterion{
    /*relative_l2*/ 3.0e-6,
    /*gross_absolute*/ 1.0e-6,
    /*gross_relative_to_max_reference*/ 3.0e-6,
};

constexpr float kEps = 1.0e-6F;

struct Case {
    std::int32_t hidden;
    std::int32_t tokens;
    std::int32_t states;
    std::int32_t queries;
    bool current_f32;
    std::uint32_t seed;
};

std::string name(const Case& c) {
    return "depth_attention H=" + std::to_string(c.hidden) + " T=" + std::to_string(c.tokens) +
           " n=" + std::to_string(c.states) + " Q=" + std::to_string(c.queries) +
           (c.current_f32 ? " current FP32" : " current BF16");
}

struct Inputs {
    std::vector<std::vector<float>> states; // BF16 values
    std::vector<float> current;             // BF16 values when !current_f32
    std::vector<float> query;
};

// States of very different magnitudes (the normalization must remove them) and queries large
// enough for a non-uniform softmax.
Inputs make_inputs(const Case& c) {
    static constexpr float kMagnitude[] = {1.0F, 40.0F, 0.02F, 300.0F, 3.0F, 0.5F, 7.0F, 0.1F};
    const std::size_t n                 = static_cast<std::size_t>(c.hidden) * c.tokens;
    Inputs in;
    for (std::int32_t s = 0; s < c.states; ++s) {
        std::vector<float> v(n);
        fill_uniform(v, c.seed + 7U * static_cast<std::uint32_t>(s), -kMagnitude[s], kMagnitude[s]);
        round_to_bf16(v);
        in.states.push_back(std::move(v));
    }
    in.current.resize(n);
    fill_uniform(in.current, c.seed + 101U, -2.0F, 2.0F);
    if (!c.current_f32) { round_to_bf16(in.current); }
    in.query.resize(static_cast<std::size_t>(c.hidden) * c.queries);
    fill_uniform(in.query, c.seed + 202U, -4.0F, 4.0F);
    return in;
}

// FP64 oracle of the contract; out is ne={H,T,Q}.
std::vector<double> oracle(const Case& c, const Inputs& in, double scale) {
    const std::int64_t H = c.hidden;
    const std::int32_t S = c.states + 1;
    std::vector<double> out(static_cast<std::size_t>(H) * c.tokens * c.queries);
    std::vector<double> v(static_cast<std::size_t>(S) * H);
    for (std::int64_t t = 0; t < c.tokens; ++t) {
        for (std::int32_t s = 0; s < S; ++s) {
            const std::vector<float>& src = s < c.states ? in.states[s] : in.current;
            double squares                = 0.0;
            for (std::int64_t h = 0; h < H; ++h) {
                squares += double(src[t * H + h]) * double(src[t * H + h]);
            }
            const double inv = 1.0 / std::sqrt(squares / double(H) + double(kEps));
            for (std::int64_t h = 0; h < H; ++h) { v[s * H + h] = double(src[t * H + h]) * inv; }
        }
        for (std::int32_t q = 0; q < c.queries; ++q) {
            std::vector<double> a(S);
            double maximum = -INFINITY;
            for (std::int32_t s = 0; s < S; ++s) {
                double dot = 0.0;
                for (std::int64_t h = 0; h < H; ++h) {
                    dot += double(in.query[q * H + h]) * v[s * H + h];
                }
                a[s]    = dot * scale;
                maximum = std::max(maximum, a[s]);
            }
            double total = 0.0;
            for (double& x : a) { total += (x = std::exp(x - maximum)); }
            for (std::int64_t h = 0; h < H; ++h) {
                double sum = 0.0;
                for (std::int32_t s = 0; s < S; ++s) { sum += a[s] / total * v[s * H + h]; }
                out[(q * c.tokens + t) * H + h] = sum;
            }
        }
    }
    return out;
}

// Runs the Op on the current device and returns the raw FP32 output.
std::vector<float> run_device(const Case& c, const Inputs& in, float scale) {
    const std::int32_t H = c.hidden;
    const std::int32_t T = c.tokens;
    std::vector<DeviceBuffer> state_buffers;
    std::vector<Tensor> states;
    for (const auto& s : in.states) {
        state_buffers.push_back(to_device_bf16(s));
        states.emplace_back(state_buffers.back().p, DType::BF16, std::initializer_list<int>{H, T});
    }
    DeviceBuffer current = c.current_f32 ? to_device_f32(in.current) : to_device_bf16(in.current);
    DeviceBuffer query   = to_device_f32(in.query);
    const std::size_t n  = static_cast<std::size_t>(H) * T * c.queries;
    GuardedDeviceBuffer out(n * sizeof(float));
    out.fill(0xff);
    const Tensor tc(current.p, c.current_f32 ? DType::FP32 : DType::BF16, {H, T});
    const Tensor tq(query.p, DType::FP32, {H, c.queries});
    Tensor to(out.data(), DType::FP32, {H, T, c.queries});
    ops::depth_attention(states, tc, tq, kEps, scale, to, nullptr);
    cuda_synchronize();
    if (out.verify_guards(name(c)) != 0) { throw std::runtime_error("guard overwritten"); }
    return from_device<float>(out.data(), n);
}

int run_case(const Case& c, int device_count) {
    const Inputs in      = make_inputs(c);
    const float scale    = 1.0F / std::sqrt(static_cast<float>(c.hidden));
    const auto expected  = oracle(c, in, double(scale));
    const std::string id = name(c);
    cuda_check(cudaSetDevice(0), "cudaSetDevice");
    const std::vector<float> got = run_device(c, in, scale);
    std::vector<double> got64(got.begin(), got.end());
    int failures = verify_reduction(id, got64, expected, kDepthAttentionCriterion);
    // Same inputs, same bits: again on this device and on the second device when present.
    failures += verify_exact((id + " repeat").c_str(), run_device(c, in, scale), got);
    if (device_count > 1) {
        cuda_check(cudaSetDevice(1), "cudaSetDevice");
        failures += verify_exact((id + " device 1").c_str(), run_device(c, in, scale), got);
        cuda_check(cudaSetDevice(0), "cudaSetDevice");
    }
    std::cout << (failures ? "FAIL " : "OK ") << id << '\n';
    return failures;
}

int run_rejections() {
    constexpr std::int32_t H = 64;
    constexpr std::int32_t T = 3; // differs from every Q below, so [T,Q,H] is not [Q,T,H]
    DeviceBuffer s(H * T * 2), cur(H * T * 4), q(H * 8 * 4), o(H * T * 8 * 4);
    const Tensor ts(s.p, DType::BF16, {H, T});
    const Tensor tc(cur.p, DType::FP32, {H, T});
    int failures = 0;
    auto expect  = [&](const char* what, auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) { return; }
        std::cerr << "depth_attention accepted " << what << '\n';
        ++failures;
    };
    expect("five queries", [&] {
        Tensor out(o.p, DType::FP32, {H, T, 5});
        ops::depth_attention({&ts, 1}, tc, Tensor(q.p, DType::FP32, {H, 5}), kEps, 0.125F, out,
                             nullptr);
    });
    expect("nine states", [&] {
        const std::vector<Tensor> many(9, ts);
        Tensor out(o.p, DType::FP32, {H, T, 1});
        ops::depth_attention(many, tc, Tensor(q.p, DType::FP32, {H, 1}), kEps, 0.125F, out,
                             nullptr);
    });
    expect("FP32 state", [&] {
        Tensor out(o.p, DType::FP32, {H, T, 1});
        ops::depth_attention({&tc, 1}, tc, Tensor(q.p, DType::FP32, {H, 1}), kEps, 0.125F, out,
                             nullptr);
    });
    expect("[T,Q,H] output", [&] {
        Tensor out(o.p, DType::FP32, {H, 2, T});
        ops::depth_attention({&ts, 1}, tc, Tensor(q.p, DType::FP32, {H, 2}), kEps, 0.125F, out,
                             nullptr);
    });
    std::cout << (failures ? "FAIL " : "OK ") << "depth_attention rejections\n";
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
        std::uint32_t seed               = 5;
        constexpr std::int32_t kTokens[] = {1, 4, 16};
        int index                        = 0;
        // Real hidden size: 0-5 block states (the depth of the model), 1, 3 or 4 queries
        // (attention input of a local / global visit, messages), both current-state dtypes.
        for (std::int32_t n = 0; n <= 5; ++n) {
            for (const std::int32_t q : {1, 3, 4}) {
                for (const bool f32 : {false, true}) {
                    failures +=
                        run_case({2304, kTokens[index++ % 3], n, q, f32, seed += 10}, device_count);
                }
            }
        }
        failures += run_case({2304, 512, 5, 1, true, seed += 10}, device_count);
        failures +=
            run_case({2304, 16, ops::kDepthAttentionMaxStates, 4, true, seed += 10}, device_count);
        // Synthetic checkpoint hidden size.
        for (const std::int32_t n : {1, 3, 5}) {
            failures += run_case({512, 16, n, 3, n != 3, seed += 10}, device_count);
        }
    } catch (const std::exception& error) {
        std::cerr << "depth_attention: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " depth_attention\n";
    return failures ? 1 : 0;
}
