#include "ninfer/ops/short_conv.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// Both tap orders: a four-term FP32 sum plus the residual, scaled, rounded once to BF16. One BF16
// rounding is at most u = 2^-8 relative (8 significant bits); the bound is 2u. The absolute floor
// covers FP32 cancellation of terms up to |w x| ~ 3 near a zero result.
constexpr PointwiseCriterion kShortConvCriterion{/*absolute*/ 1.0e-5, /*relative*/ 1.0 / 128.0};

constexpr std::int32_t kHistory = ops::kShortConvTaps - 1;

std::vector<float> make_values(std::size_t count, std::uint32_t seed, float low, float high) {
    std::vector<float> values(count);
    fill_uniform(values, seed, low, high);
    round_to_bf16(values);
    return values;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

// The logical operands of one convolution. Static: weight ne={4,C}, dyn empty. Dynamic:
// weight = static_weight ne={C,4}, dyn ne={4G,T}.
struct Problem {
    std::int32_t channels = 0;
    std::int32_t tokens   = 0;
    std::int32_t groups   = 0; // 0 = static tap order
    float scale           = 1.0F;
    std::vector<float> x, weight, dyn, state;
};

struct Expected {
    std::vector<double> out;
    std::vector<float> state;
};

// FP64 oracle of the contract over the represented inputs.
Expected oracle(const Problem& p) {
    const std::int64_t C = p.channels;
    Expected e;
    e.out.resize(static_cast<std::size_t>(C) * p.tokens);
    auto u = [&](std::int64_t c, std::int64_t s) -> double {
        return s >= 0 ? p.x[s * C + c] : p.state[(kHistory + s) * C + c];
    };
    for (std::int64_t t = 0; t < p.tokens; ++t) {
        for (std::int64_t c = 0; c < C; ++c) {
            double acc = 0.0;
            if (p.groups == 0) {
                for (int j = 0; j < 4; ++j) {
                    acc += double(p.weight[c * 4 + j]) * u(c, t - 3 + j);
                }
            } else {
                const std::int64_t g = c / (C / p.groups);
                for (int lag = 0; lag < 4; ++lag) {
                    const double tap = double(p.weight[lag * C + c]) +
                                       double(p.dyn[t * 4 * p.groups + 4 * g + lag]);
                    acc += tap * u(c, t - lag);
                }
            }
            e.out[t * C + c] = (acc + double(p.x[t * C + c])) * double(p.scale);
        }
    }
    e.state.resize(static_cast<std::size_t>(C) * kHistory);
    for (std::int64_t k = 0; k < kHistory; ++k) {
        for (std::int64_t c = 0; c < C; ++c) {
            e.state[k * C + c] = static_cast<float>(u(c, p.tokens - (kHistory - k)));
        }
    }
    return e;
}

// Device copies of the operands; the state buffer is updated in place by every call.
struct DeviceProblem {
    DeviceBuffer x, weight, dyn, state;

    explicit DeviceProblem(const Problem& p)
        : x(to_device_bf16(p.x)), weight(to_device_f32(p.weight)),
          dyn(p.groups == 0 ? DeviceBuffer(2) : to_device_bf16(p.dyn)),
          state(to_device_bf16(p.state)) {}
};

// Runs columns [first, first + count) of the problem against the device state (in place).
void run(const Problem& p, DeviceProblem& d, std::int32_t first, std::int32_t count, void* out) {
    const std::int32_t C = p.channels;
    const Tensor x(static_cast<std::uint16_t*>(d.x.p) + static_cast<std::size_t>(first) * C,
                   DType::BF16, {C, count});
    Tensor state(d.state.p, DType::BF16, {C, kHistory});
    Tensor o(out, DType::BF16, {C, count});
    if (p.groups == 0) {
        const Tensor w(d.weight.p, DType::FP32, {4, C});
        ops::short_conv_static(x, w, p.scale, state, state, o, nullptr);
    } else {
        const Tensor w(d.weight.p, DType::FP32, {C, 4});
        const Tensor dyn(static_cast<std::uint16_t*>(d.dyn.p) +
                             static_cast<std::size_t>(first) * 4 * p.groups,
                         DType::BF16, {4 * p.groups, count});
        ops::short_conv_dynamic(x, w, dyn, p.scale, state, state, o, nullptr);
    }
}

Problem make_problem(std::int32_t C, std::int32_t T, std::int32_t groups, std::uint32_t seed) {
    Problem p;
    p.channels = C;
    p.tokens   = T;
    p.groups   = groups;
    p.scale    = groups == 0 ? 1.0F / std::sqrt(5.0F) : 1.0F / 3.0F;
    p.x        = make_values(static_cast<std::size_t>(C) * T, seed, -3.0F, 3.0F);
    p.weight   = make_values(static_cast<std::size_t>(C) * 4, seed + 1, -1.0F, 1.0F);
    p.state    = make_values(static_cast<std::size_t>(C) * kHistory, seed + 2, -3.0F, 3.0F);
    if (groups != 0) {
        p.dyn = make_values(static_cast<std::size_t>(4) * groups * T, seed + 3, -1.0F, 1.0F);
    }
    return p;
}

std::string name(const Problem& p) {
    return std::string(p.groups == 0 ? "short_conv_static" : "short_conv_dynamic") +
           " C=" + std::to_string(p.channels) + " T=" + std::to_string(p.tokens) +
           (p.groups == 0 ? "" : " G=" + std::to_string(p.groups));
}

// One call over all tokens against the oracle (output and state), then the same sequence split
// into chunks and token by token, which must reproduce the one-call bits and final state.
int run_case(std::int32_t C, std::int32_t T, std::int32_t groups, std::uint32_t seed) {
    const Problem p      = make_problem(C, T, groups, seed);
    const Expected e     = oracle(p);
    const std::string id = name(p);
    const std::size_t n  = static_cast<std::size_t>(C) * T;
    int failures         = 0;

    DeviceProblem whole(p);
    GuardedDeviceBuffer out(n * sizeof(std::uint16_t));
    out.fill(0xff);
    run(p, whole, 0, T, out.data());
    cuda_synchronize();
    const std::vector<std::uint16_t> one_call = from_device<std::uint16_t>(out.data(), n);
    failures += verify_pointwise(id, from_device_bf16(out.data(), n), e.out, kShortConvCriterion);
    failures += out.verify_guards(id);
    failures +=
        verify_exact((id + " state").c_str(),
                     from_device<std::uint16_t>(whole.state.p, e.state.size()), bf16_bits(e.state));

    for (const std::int32_t chunk : {1, 3, 16}) {
        if (chunk >= T) { continue; }
        DeviceProblem split(p);
        DeviceBuffer split_out(n * sizeof(std::uint16_t));
        for (std::int32_t first = 0; first < T; first += chunk) {
            const std::int32_t count = std::min(chunk, T - first);
            run(p, split, first, count,
                static_cast<std::uint16_t*>(split_out.p) + static_cast<std::size_t>(first) * C);
        }
        cuda_synchronize();
        const std::string label = id + " chunks of " + std::to_string(chunk);
        failures +=
            verify_exact(label.c_str(), from_device<std::uint16_t>(split_out.p, n), one_call);
        failures += verify_exact((label + " state").c_str(),
                                 from_device<std::uint16_t>(split.state.p, e.state.size()),
                                 bf16_bits(e.state));
    }
    std::cout << (failures ? "FAIL " : "OK ") << id << '\n';
    return failures;
}

// Impulse at token 0 of every channel, zero state, scale 1 and small integer taps: the outputs are
// exact and show which tap weighs which token.
int run_impulse(std::int32_t groups) {
    constexpr std::int32_t C = 8;
    constexpr std::int32_t T = 6;
    Problem p;
    p.channels = C;
    p.tokens   = T;
    p.groups   = groups;
    p.scale    = 1.0F;
    p.x.assign(static_cast<std::size_t>(C) * T, 0.0F);
    for (std::int32_t c = 0; c < C; ++c) { p.x[c] = 1.0F; }
    p.state.assign(static_cast<std::size_t>(C) * kHistory, 0.0F);
    const float taps[4] = {1.0F, 2.0F, 3.0F, 5.0F}; // index = position in the weight tensor
    p.weight.resize(static_cast<std::size_t>(C) * 4);
    for (std::int32_t c = 0; c < C; ++c) {
        for (int j = 0; j < 4; ++j) {
            if (groups == 0) {
                p.weight[c * 4 + j] = taps[j];
            } else {
                p.weight[j * C + c] = taps[j];
            }
        }
    }
    std::vector<float> expected(static_cast<std::size_t>(C) * T, 0.0F);
    if (groups != 0) {
        // dyn[t, 4g + lag] = 8 (g + 1) only at lag == t, so token t adds 8 (g + 1) to its tap.
        p.dyn.assign(static_cast<std::size_t>(4) * groups * T, 0.0F);
        for (std::int32_t t = 0; t < 4; ++t) {
            for (std::int32_t g = 0; g < groups; ++g) {
                p.dyn[t * 4 * groups + 4 * g + t] = 8.0F * static_cast<float>(g + 1);
            }
        }
    }
    for (std::int32_t c = 0; c < C; ++c) {
        for (std::int32_t t = 0; t < 4; ++t) {
            // Static: tap 3 is the current token, so token t sees tap 3 - t. Dynamic: lag t.
            float value = groups == 0 ? taps[3 - t] : taps[t];
            if (groups != 0) { value += 8.0F * static_cast<float>(c / (C / groups) + 1); }
            expected[t * C + c] = value + (t == 0 ? 1.0F : 0.0F);
        }
    }
    DeviceProblem d(p);
    DeviceBuffer out(expected.size() * sizeof(std::uint16_t));
    run(p, d, 0, T, out.p);
    cuda_synchronize();
    const std::string id = groups == 0 ? "short_conv_static impulse" : "short_conv_dynamic impulse";
    int failures = verify_exact(id.c_str(), from_device<std::uint16_t>(out.p, expected.size()),
                                bf16_bits(expected));
    std::cout << (failures ? "FAIL " : "OK ") << id << '\n';
    return failures;
}

int run_rejections() {
    constexpr std::int32_t C = 64;
    constexpr std::int32_t T = 4;
    DeviceBuffer x(C * T * 2), w(C * 4 * 4), dyn(16 * T * 2), s(C * 3 * 2), o(C * T * 2);
    const Tensor tx(x.p, DType::BF16, {C, T});
    const Tensor tws(w.p, DType::FP32, {4, C});
    const Tensor twd(w.p, DType::FP32, {C, 4});
    const Tensor td(dyn.p, DType::BF16, {16, T});
    Tensor ts(s.p, DType::BF16, {C, 3});
    Tensor to(o.p, DType::BF16, {C, T});
    int failures = 0;
    auto expect  = [&](const char* what, auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) { return; }
        std::cerr << "short_conv accepted " << what << '\n';
        ++failures;
    };
    expect("BF16 weight", [&] {
        ops::short_conv_static(tx, Tensor(w.p, DType::BF16, {4, C}), 1.0F, ts, ts, to, nullptr);
    });
    expect("transposed static weight",
           [&] { ops::short_conv_static(tx, twd, 1.0F, ts, ts, to, nullptr); });
    expect("zero scale", [&] { ops::short_conv_static(tx, tws, 0.0F, ts, ts, to, nullptr); });
    expect("out over x", [&] {
        Tensor alias(x.p, DType::BF16, {C, T});
        ops::short_conv_static(tx, tws, 1.0F, ts, ts, alias, nullptr);
    });
    expect("shifted state alias", [&] {
        Tensor shifted(static_cast<std::uint16_t*>(s.p) + 1, DType::BF16, {C, 3});
        ops::short_conv_static(tx, tws, 1.0F, ts, shifted, to, nullptr);
    });
    expect("groups not dividing C", [&] {
        ops::short_conv_dynamic(tx, twd, Tensor(dyn.p, DType::BF16, {12, T}), 1.0F, ts, ts, to,
                                nullptr);
    });
    expect("dyn token mismatch", [&] {
        ops::short_conv_dynamic(tx, twd, Tensor(dyn.p, DType::BF16, {16, T - 1}), 1.0F, ts, ts, to,
                                nullptr);
    });
    std::cout << (failures ? "FAIL " : "OK ") << "short_conv rejections\n";
    return failures;
}

// short_conv_pair against the two Ops, bit for bit: outputs and states, in place, both kinds.
int run_pair(std::int32_t C, std::int32_t T, std::int32_t G, std::uint32_t seed) {
    const std::string id = "short_conv_pair C=" + std::to_string(C) + " T=" + std::to_string(T) +
                           " G=" + std::to_string(G);
    const std::size_t n = static_cast<std::size_t>(C) * T;

    struct Call {
        DeviceBuffer x, w, dyn, state, out;
    };

    const auto make_call = [&](std::uint32_t s) {
        Call c;
        c.x = to_device_bf16(make_values(n, s, -2.0F, 2.0F));
        c.w = to_device_f32(make_values(static_cast<std::size_t>(C) * 4, s + 1, -0.5F, 0.5F));
        if (G > 0) {
            c.dyn = to_device_bf16(
                make_values(static_cast<std::size_t>(4 * G) * T, s + 2, -0.3F, 0.3F));
        }
        c.state =
            to_device_bf16(make_values(static_cast<std::size_t>(C) * kHistory, s + 3, -2.0F, 2.0F));
        c.out = DeviceBuffer(n * 2);
        return c;
    };
    Call a = make_call(seed), b = make_call(seed + 10);
    const std::vector<std::uint16_t> state_a = from_device<std::uint16_t>(a.state, C * kHistory);
    const std::vector<std::uint16_t> state_b = from_device<std::uint16_t>(b.state, C * kHistory);
    const auto call                          = [&](Call& c) {
        Tensor st(c.state.p, DType::BF16, {C, kHistory});
        return ops::ShortConvCall{Tensor(c.x.p, DType::BF16, {C, T}),
                                  G > 0 ? Tensor(c.w.p, DType::FP32, {C, 4})
                                                                 : Tensor(c.w.p, DType::FP32, {4, C}),
                                  G > 0 ? Tensor(c.dyn.p, DType::BF16, {4 * G, T}) : Tensor{},
                                  0.4472136F,
                                  st,
                                  st,
                                  Tensor(c.out.p, DType::BF16, {C, T})};
    };
    ops::short_conv_pair(call(a), call(b), nullptr);
    cuda_synchronize();
    const auto pair_out_a = from_device<std::uint16_t>(a.out, n);
    const auto pair_out_b = from_device<std::uint16_t>(b.out, n);
    const auto pair_st_a  = from_device<std::uint16_t>(a.state, C * kHistory);
    const auto pair_st_b  = from_device<std::uint16_t>(b.state, C * kHistory);
    a.state.copy_from_host(state_a.data(), state_a.size() * 2);
    b.state.copy_from_host(state_b.data(), state_b.size() * 2);
    for (Call* c : {&a, &b}) {
        ops::ShortConvCall x = call(*c);
        if (G > 0) {
            ops::short_conv_dynamic(x.x, x.weight, x.dyn, x.scale, x.state_in, x.state_out, x.out,
                                    nullptr);
        } else {
            ops::short_conv_static(x.x, x.weight, x.scale, x.state_in, x.state_out, x.out, nullptr);
        }
    }
    cuda_synchronize();
    int failures =
        verify_exact((id + " first out").c_str(), pair_out_a, from_device<std::uint16_t>(a.out, n));
    failures += verify_exact((id + " second out").c_str(), pair_out_b,
                             from_device<std::uint16_t>(b.out, n));
    failures += verify_exact((id + " first state").c_str(), pair_st_a,
                             from_device<std::uint16_t>(a.state, C * kHistory));
    failures += verify_exact((id + " second state").c_str(), pair_st_b,
                             from_device<std::uint16_t>(b.state, C * kHistory));
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
        failures += run_impulse(0);
        failures += run_impulse(2);
        failures += run_rejections();
        // Static: post-attention/post-MoE (2304), K/V of a local visit at tp 1 (1024) and per
        // rank (512), the synthetic checkpoint (512 hidden, 256 K/V channels).
        std::uint32_t seed = 11;
        for (const std::int32_t C : {2304, 1024, 512, 256}) {
            for (const std::int32_t T : {1, 4, 16, 17, 512}) {
                failures += run_case(C, T, 0, seed += 10);
            }
        }
        // Dynamic: 8 KV heads of 128 at tp 1, 4 per rank, 2 in the synthetic checkpoint.
        for (const auto& [C, G] : {std::pair{1024, 8}, std::pair{512, 4}, std::pair{256, 2}}) {
            for (const std::int32_t T : {1, 4, 16, 17, 512}) {
                failures += run_case(C, T, G, seed += 10);
            }
        }
        for (const std::int32_t T : {1, 4}) {
            failures += run_pair(512, T, 0, seed += 20);
            failures += run_pair(512, T, 4, seed += 20);
            failures += run_pair(2304, T, 0, seed += 20);
        }
    } catch (const std::exception& error) {
        std::cerr << "short_conv: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " short_conv\n";
    return failures ? 1 : 0;
}
