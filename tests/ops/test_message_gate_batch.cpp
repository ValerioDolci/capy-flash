// message_gates_rmsnorm: every gate's output and norm against an FP64 oracle (message_gate's
// criterion; the norm adds one BF16 rounding), and, bit for bit, the invariances decoding relies
// on: a column's result does not depend on T (every column again as a T = 1 call), on the batch
// (every gate again alone) or on writing in place. Operands 4 bytes off a 16-byte boundary take
// the per-gate Ops: message_gate then rmsnorm_f32, bit for bit.
#include "ninfer/ops/message_gate.h"
#include "ninfer/ops/rmsnorm_f32.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int H      = 2304;
constexpr int R      = 64;
constexpr float kEps = 1.0e-6F;
constexpr int kMaxT  = 40;
constexpr int kShift = 1; // FP32 elements
// message_gate's criterion (test_message_gate.cpp).
constexpr ReductionCriterion kGateCriterion{1.0e-3, 1.0e-5, 2.0e-3};
// The norm adds one BF16 rounding (relative RMS ~1.1e-3) to the gate output's error.
constexpr ReductionCriterion kNormCriterion{3.0e-3, 1.0e-5, 8.0e-3};

double round_bf16(double value) {
    return double(bf16_to_f32(f32_to_bf16(static_cast<float>(value))));
}

struct Gate {
    std::vector<float> down, forget, gain, norm, hidden, message; // hidden, message: [kMaxT][H]
    DeviceBuffer d_down, d_forget, d_gain, d_norm, d_hidden, d_message;
};

// Trained-like scales: bottleneck and gate arguments of order one. forget and gain are BF16
// values, as the Op's domain requires.
std::unique_ptr<Gate> make_gate(std::uint32_t seed) {
    auto g = std::make_unique<Gate>();
    g->down.resize(static_cast<std::size_t>(2) * H * R);
    const float a = std::sqrt(3.0F / static_cast<float>(2 * H));
    fill_uniform(g->down, seed, -a, a);
    round_to_bf16(g->down);
    const float b = std::sqrt(3.0F / static_cast<float>(R));
    g->forget.resize(static_cast<std::size_t>(H) * R);
    g->gain.resize(g->forget.size());
    fill_uniform(g->forget, seed + 1, -b, b);
    fill_uniform(g->gain, seed + 2, -b, b);
    round_to_bf16(g->forget);
    round_to_bf16(g->gain);
    g->norm.resize(H);
    fill_uniform(g->norm, seed + 3, 0.5F, 1.5F);
    round_to_bf16(g->norm);
    g->hidden.resize(static_cast<std::size_t>(kMaxT) * H);
    g->message.resize(g->hidden.size());
    fill_uniform(g->hidden, seed + 4, -2.0F, 2.0F);
    fill_uniform(g->message, seed + 5, -3.0F, 3.0F);
    g->d_down   = to_device_bf16(g->down);
    g->d_forget = to_device_f32(g->forget);
    g->d_gain   = to_device_f32(g->gain);
    g->d_norm   = to_device_bf16(g->norm);
    // One spare element: the operands may sit kShift elements in.
    g->d_hidden  = DeviceBuffer((g->hidden.size() + kShift) * sizeof(float));
    g->d_message = DeviceBuffer((g->hidden.size() + kShift) * sizeof(float));
    return g;
}

// Columns [t0, t0 + T): the gate output and its norm in FP64.
void oracle(const Gate& g, int t0, int T, std::vector<double>& out, std::vector<double>& normed) {
    out.assign(static_cast<std::size_t>(H) * T, 0.0);
    normed.assign(out.size(), 0.0);
    std::vector<double> z(2 * H), d(R);
    for (int t = 0; t < T; ++t) {
        const float* h = g.hidden.data() + static_cast<std::size_t>(t0 + t) * H;
        const float* m = g.message.data() + static_cast<std::size_t>(t0 + t) * H;
        double hh = 0.0, mm = 0.0;
        for (int i = 0; i < H; ++i) {
            hh += double(h[i]) * h[i];
            mm += double(m[i]) * m[i];
        }
        const double inv_h = 1.0 / std::sqrt(hh / H + double(kEps));
        const double inv_m = 1.0 / std::sqrt(mm / H + double(kEps));
        for (int i = 0; i < H; ++i) {
            z[i]     = round_bf16(h[i] * inv_h);
            z[H + i] = round_bf16(m[i] * inv_m);
        }
        for (int r = 0; r < R; ++r) {
            double sum = 0.0;
            for (int i = 0; i < 2 * H; ++i) { sum += double(g.down[r * 2 * H + i]) * z[i]; }
            d[r] = round_bf16(sum);
        }
        double squares = 0.0;
        double* o      = out.data() + static_cast<std::size_t>(t) * H;
        for (int i = 0; i < H; ++i) {
            double f = 0.0, a = 0.0;
            for (int r = 0; r < R; ++r) {
                f += double(g.forget[i * R + r]) * d[r];
                a += double(g.gain[i * R + r]) * d[r];
            }
            o[i] = (1.0 - std::tanh(f)) * h[i] + std::tanh(a) * m[i];
            squares += o[i] * o[i];
        }
        const double inv = 1.0 / std::sqrt(squares / H + double(kEps));
        for (int i = 0; i < H; ++i) {
            normed[static_cast<std::size_t>(t) * H + i] = o[i] * inv * g.norm[i];
        }
    }
}

struct Result {
    std::vector<float> out;
    std::vector<std::uint16_t> normed;
};

// One message_gates_rmsnorm call over columns [t0, t0 + T) of the gates (the first gate's hidden
// for all when `shared`), or the per-gate Ops when `separate`.
std::vector<Result> run(const std::vector<Gate*>& gates, int t0, int T, bool shared, bool in_place,
                        int shift, bool separate) {
    const std::size_t n      = static_cast<std::size_t>(H) * T;
    const std::size_t column = static_cast<std::size_t>(t0) * H;
    const int G              = static_cast<int>(gates.size());
    std::vector<DeviceBuffer> outs, normeds;
    for (int k = 0; k < G; ++k) {
        Gate& g = *gates[k];
        g.d_hidden.copy_from_host(g.hidden.data(), g.hidden.size() * 4, shift * 4);
        g.d_message.copy_from_host(g.message.data(), g.message.size() * 4, shift * 4);
        outs.emplace_back((n + kShift) * sizeof(float));
        normeds.emplace_back(n * 2);
        const Gate& source = *gates[shared ? 0 : k];
        if (in_place) {
            outs[k].copy_from_host(source.hidden.data() + column, n * 4, shift * 4);
        } else {
            outs[k].fill(0xff);
        }
    }
    const auto f32 = [&](void* p) {
        return Tensor(static_cast<float*>(p) + shift, DType::FP32, {H, T});
    };
    std::vector<ops::MessageGateNormed> batch;
    for (int k = 0; k < G; ++k) {
        Gate& g      = *gates[k];
        auto* source = static_cast<float*>(gates[shared ? 0 : k]->d_hidden.p) + column;
        batch.push_back(
            {in_place ? f32(outs[k].p) : f32(source),
             f32(static_cast<float*>(g.d_message.p) + column),
             Tensor(g.d_down.p, DType::BF16, {2 * H, R}), Tensor(g.d_forget.p, DType::FP32, {R, H}),
             Tensor(g.d_gain.p, DType::FP32, {R, H}), f32(outs[k].p),
             Tensor(g.d_norm.p, DType::BF16, {H}), Tensor(normeds[k].p, DType::BF16, {H, T})});
    }
    if (separate) {
        for (auto& x : batch) {
            Tensor out = x.out, normed = x.normed;
            ops::message_gate(x.hidden, x.message, x.down, x.forget, x.gain, kEps, out, nullptr);
            ops::rmsnorm_f32(out, x.norm, kEps, normed, nullptr);
        }
    } else {
        ops::message_gates_rmsnorm(batch, kEps, kEps, nullptr);
    }
    cuda_synchronize();
    std::vector<Result> results(G);
    for (int k = 0; k < G; ++k) {
        results[k].out.resize(n);
        outs[k].copy_to_host(results[k].out.data(), n * 4, shift * 4);
        results[k].normed = from_device<std::uint16_t>(normeds[k], n);
    }
    return results;
}

int same(const std::string& label, const Result& got, const Result& expected, int t = -1) {
    if (t < 0) {
        return verify_exact((label + " out").c_str(), got.out, expected.out) +
               verify_exact((label + " norm").c_str(), got.normed, expected.normed);
    }
    const auto first = static_cast<std::ptrdiff_t>(t) * H;
    const std::vector<float> out(expected.out.begin() + first, expected.out.begin() + first + H);
    const std::vector<std::uint16_t> normed(expected.normed.begin() + first,
                                            expected.normed.begin() + first + H);
    return verify_exact((label + " out").c_str(), got.out, out) +
           verify_exact((label + " norm").c_str(), got.normed, normed);
}

// `shared`: the gates read the first gate's hidden (their own hidden holds the same values).
int run_case(const std::vector<Gate*>& gates, int T, bool shared) {
    const std::string id = "message_gates_rmsnorm G=" + std::to_string(gates.size()) +
                           " T=" + std::to_string(T) + (shared ? " shared" : "");
    int failures      = 0;
    const auto result = run(gates, 0, T, shared, false, 0, false);
    const auto placed = run(gates, 0, T, shared, true, 0, false);
    for (std::size_t k = 0; k < gates.size(); ++k) {
        const std::string label = id + " gate " + std::to_string(k);
        std::vector<double> out, normed;
        oracle(*gates[k], 0, T, out, normed);
        const std::vector<double> out64(result[k].out.begin(), result[k].out.end());
        std::vector<double> norm64(result[k].normed.size());
        for (std::size_t x = 0; x < norm64.size(); ++x) {
            norm64[x] = bf16_to_f32(result[k].normed[x]);
        }
        failures += verify_reduction(label + " out", out64, out, kGateCriterion);
        failures += verify_reduction(label + " norm", norm64, normed, kNormCriterion);
        failures += same(label + " in place", placed[k], result[k]);
        if (gates.size() > 1) {
            failures +=
                same(label + " alone", run({gates[k]}, 0, T, false, false, 0, false)[0], result[k]);
        }
        for (int t = 0; t < T && T > 1; ++t) {
            failures += same(label + " column " + std::to_string(t),
                             run(gates, t, 1, shared, false, 0, false)[k], result[k], t);
        }
    }
    std::cout << (failures ? "FAIL " : "OK ") << id << '\n';
    return failures;
}

// forget and gain given as BF16 copies: the bits of the FP32 matrices of the same values.
int run_bf16(const std::vector<Gate*>& gates, int T, bool shared) {
    const std::string id = "message_gates_rmsnorm G=" + std::to_string(gates.size()) +
                           " T=" + std::to_string(T) + " BF16 weights";
    const auto expected = run(gates, 0, T, shared, false, 0, false);
    const std::size_t n = static_cast<std::size_t>(H) * T;
    std::vector<DeviceBuffer> forget, gain, outs, normeds;
    std::vector<ops::MessageGateNormed> batch;
    for (std::size_t k = 0; k < gates.size(); ++k) {
        Gate& g = *gates[k];
        forget.push_back(to_device_bf16(g.forget));
        gain.push_back(to_device_bf16(g.gain));
        outs.emplace_back(n * 4);
        normeds.emplace_back(n * 2);
        g.d_hidden.copy_from_host(g.hidden.data(), g.hidden.size() * 4);
        g.d_message.copy_from_host(g.message.data(), g.message.size() * 4);
    }
    for (std::size_t k = 0; k < gates.size(); ++k) {
        Gate& g = *gates[k];
        batch.push_back(
            {Tensor(gates[shared ? 0 : k]->d_hidden.p, DType::FP32, {H, T}),
             Tensor(g.d_message.p, DType::FP32, {H, T}),
             Tensor(g.d_down.p, DType::BF16, {2 * H, R}), Tensor(forget[k].p, DType::BF16, {R, H}),
             Tensor(gain[k].p, DType::BF16, {R, H}), Tensor(outs[k].p, DType::FP32, {H, T}),
             Tensor(g.d_norm.p, DType::BF16, {H}), Tensor(normeds[k].p, DType::BF16, {H, T})});
    }
    ops::message_gates_rmsnorm(batch, kEps, kEps, nullptr);
    cuda_synchronize();
    int failures = 0;
    for (std::size_t k = 0; k < gates.size(); ++k) {
        Result got;
        got.out.resize(n);
        outs[k].copy_to_host(got.out.data(), n * 4);
        got.normed = from_device<std::uint16_t>(normeds[k], n);
        failures += same(id + " gate " + std::to_string(k), got, expected[k]);
    }
    std::cout << (failures ? "FAIL " : "OK ") << id << '\n';
    return failures;
}

// Operands off a 16-byte boundary: the per-gate Ops, bit for bit.
int run_shifted(const std::vector<Gate*>& gates, int T, bool shared) {
    const std::string id = "message_gates_rmsnorm G=" + std::to_string(gates.size()) +
                           " T=" + std::to_string(T) + " shifted";
    const auto got      = run(gates, 0, T, shared, false, kShift, false);
    const auto expected = run(gates, 0, T, shared, false, kShift, true);
    int failures        = 0;
    for (std::size_t k = 0; k < gates.size(); ++k) {
        failures += same(id + " gate " + std::to_string(k), got[k], expected[k]);
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
        std::vector<std::unique_ptr<Gate>> owned;
        for (int k = 0; k < 6; ++k) { owned.push_back(make_gate(100U + 10U * k)); }
        // Gates 4 and 5 read gate 0's input, as a local visit's three gates do.
        owned[4]->hidden                  = owned[0]->hidden;
        owned[5]->hidden                  = owned[0]->hidden;
        const std::vector<Gate*> local    = {owned[0].get(), owned[4].get(), owned[5].get()};
        const std::vector<Gate*> distinct = {owned[1].get(), owned[2].get(), owned[3].get()};
        const std::vector<Gate*> one      = {owned[1].get()};
        for (const int T : {1, 2, 3, 4, 5, 8, 9, 16, 17, 40}) {
            failures += run_case(local, T, true);
            failures += run_case(distinct, T, false);
            failures += run_case(one, T, false);
        }
        failures +=
            run_case({owned[0].get(), owned[1].get(), owned[2].get(), owned[3].get()}, 8, false);
        failures += run_case({owned[2].get(), owned[3].get()}, 5, false);
        failures += run_bf16(local, 1, true);
        failures += run_bf16(distinct, 4, false);
        failures += run_bf16(one, 1, false);
        failures += run_shifted(local, 4, true);
        failures += run_shifted(one, 1, false);
    } catch (const std::exception& e) {
        std::cout << "FAIL: " << e.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " message_gates_rmsnorm\n";
    return failures == 0 ? 0 : 1;
}
