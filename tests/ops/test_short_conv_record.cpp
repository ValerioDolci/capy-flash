#include "ninfer/ops/short_conv.h"
#include "ninfer/ops/short_conv_record.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

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

// The criterion of test_short_conv.cpp (one BF16 rounding of a four-term FP32 sum).
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

// One convolution's operands. Static: weight ne={4,C}. Dynamic (groups > 0): weight =
// static_weight ne={C,4}, dyn ne={4G,T}.
struct Problem {
    std::int32_t channels = 0;
    std::int32_t tokens   = 0;
    std::int32_t groups   = 0;
    float scale           = 1.0F;
    std::vector<float> x, weight, dyn, state;
};

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

// FP64 oracle of out (short_conv.h).
std::vector<double> oracle(const Problem& p) {
    const std::int64_t C = p.channels;
    std::vector<double> out(static_cast<std::size_t>(C) * p.tokens);
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
            out[t * C + c] = (acc + double(p.x[t * C + c])) * double(p.scale);
        }
    }
    return out;
}

struct DeviceProblem {
    DeviceBuffer x, weight, dyn;

    explicit DeviceProblem(const Problem& p)
        : x(to_device_bf16(p.x)), weight(to_device_f32(p.weight)),
          dyn(p.groups == 0 ? DeviceBuffer(2) : to_device_bf16(p.dyn)) {}

    [[nodiscard]] Tensor x_columns(const Problem& p, std::int32_t count) const {
        return Tensor(x.p, DType::BF16, {p.channels, count});
    }

    [[nodiscard]] Tensor dyn_columns(const Problem& p, std::int32_t count) const {
        return Tensor(dyn.p, DType::BF16, {4 * p.groups, count});
    }
};

// The existing convolution over the first `count` tokens, its state advanced in place.
void run_plain(const Problem& p, const DeviceProblem& d, std::int32_t count, void* state,
               void* out) {
    const std::int32_t C = p.channels;
    Tensor s(state, DType::BF16, {C, kHistory});
    Tensor o(out, DType::BF16, {C, count});
    if (p.groups == 0) {
        ops::short_conv_static(d.x_columns(p, count), Tensor(d.weight.p, DType::FP32, {4, C}),
                               p.scale, s, s, o, nullptr);
    } else {
        ops::short_conv_dynamic(d.x_columns(p, count), Tensor(d.weight.p, DType::FP32, {C, 4}),
                                d.dyn_columns(p, count), p.scale, s, s, o, nullptr);
    }
}

void run_record(const Problem& p, const DeviceProblem& d, const void* state, void* record,
                void* out) {
    const std::int32_t C = p.channels;
    const std::int32_t T = p.tokens;
    const Tensor s(const_cast<void*>(state), DType::BF16, {C, kHistory});
    Tensor r(record, DType::BF16, {C, T + kHistory});
    Tensor o(out, DType::BF16, {C, T});
    if (p.groups == 0) {
        ops::short_conv_static_record(d.x_columns(p, T), Tensor(d.weight.p, DType::FP32, {4, C}),
                                      p.scale, s, r, o, nullptr);
    } else {
        ops::short_conv_dynamic_record(d.x_columns(p, T), Tensor(d.weight.p, DType::FP32, {C, 4}),
                                       d.dyn_columns(p, T), p.scale, s, r, o, nullptr);
    }
}

std::string name(const Problem& p) {
    return std::string(p.groups == 0 ? "short_conv_static_record" : "short_conv_dynamic_record") +
           " C=" + std::to_string(p.channels) + " T=" + std::to_string(p.tokens) +
           (p.groups == 0 ? "" : " G=" + std::to_string(p.groups));
}

// The recording call against the existing one (out bit for bit, and against the FP64 oracle),
// its record against [state ; x], then for every committed count n = 0..T the fold against the
// state the existing convolution leaves after the first n tokens.
int run_case(std::int32_t C, std::int32_t T, std::int32_t groups, std::uint32_t seed) {
    const Problem p      = make_problem(C, T, groups, seed);
    const std::string id = name(p);
    const std::size_t n  = static_cast<std::size_t>(C) * T;
    const std::size_t sn = static_cast<std::size_t>(C) * kHistory;
    int failures         = 0;
    const DeviceProblem d(p);

    DeviceBuffer plain_state = to_device_bf16(p.state);
    DeviceBuffer plain_out(n * 2);
    run_plain(p, d, T, plain_state.p, plain_out.p);

    const DeviceBuffer state = to_device_bf16(p.state);
    GuardedDeviceBuffer record((n + sn) * 2);
    GuardedDeviceBuffer out(n * 2);
    record.fill(0xff);
    out.fill(0xff);
    run_record(p, d, state.p, record.data(), out.data());
    cuda_synchronize();
    failures +=
        verify_exact((id + " out = short_conv").c_str(), from_device<std::uint16_t>(out.data(), n),
                     from_device<std::uint16_t>(plain_out.p, n));
    failures +=
        verify_pointwise(id, from_device_bf16(out.data(), n), oracle(p), kShortConvCriterion);
    std::vector<float> u = p.state;
    u.insert(u.end(), p.x.begin(), p.x.end());
    failures += verify_exact((id + " record").c_str(),
                             from_device<std::uint16_t>(record.data(), n + sn), bf16_bits(u));
    failures += verify_exact((id + " state unchanged").c_str(),
                             from_device<std::uint16_t>(state.p, sn), bf16_bits(p.state));
    failures += record.verify_guards(id + " record") + out.verify_guards(id + " out");

    DeviceBuffer tail(sn * 2), committed(4), entry(sizeof(ops::ConvTailFoldEntry));
    const ops::ConvTailFoldEntry host_entry{record.data(), tail.p, C};
    cuda_check(cudaMemcpy(entry.p, &host_entry, sizeof(host_entry), cudaMemcpyHostToDevice),
               "cuda copy");
    for (std::int32_t count = 0; count <= T; ++count) {
        DeviceBuffer expected_state = to_device_bf16(p.state);
        if (count > 0) {
            DeviceBuffer scratch(static_cast<std::size_t>(C) * count * 2);
            run_plain(p, d, count, expected_state.p, scratch.p);
        }
        cuda_check(cudaMemcpy(committed.p, &count, 4, cudaMemcpyHostToDevice), "cuda copy");
        cuda_check(cudaMemset(tail.p, 0xff, sn * 2), "cuda copy");
        ops::conv_tail_fold(static_cast<const ops::ConvTailFoldEntry*>(entry.p), 1, T,
                            Tensor(committed.p, DType::I32, {1}), nullptr);
        cuda_synchronize();
        failures += verify_exact((id + " fold n=" + std::to_string(count)).c_str(),
                                 from_device<std::uint16_t>(tail.p, sn),
                                 from_device<std::uint16_t>(expected_state.p, sn));
    }
    std::cout << (failures ? "FAIL " : "OK ") << id << '\n';
    return failures;
}

// One fold over the 4 tails of 48 visit slots of a rank at tp 2 (K/V 512, attention/FFN 2304)
// plus an odd channel count (the element route), each tail checked against its own record.
int run_many(std::int32_t T, std::int32_t committed_count) {
    std::vector<std::int32_t> channels;
    for (int slot = 0; slot < 48; ++slot) {
        for (const std::int32_t C : {512, 512, 2304, 2304}) { channels.push_back(C); }
    }
    channels.push_back(6);
    std::vector<DeviceBuffer> records, tails;
    std::vector<std::vector<float>> values;
    std::vector<ops::ConvTailFoldEntry> host;
    std::uint32_t seed = 900;
    for (const std::int32_t C : channels) {
        values.push_back(make_values(static_cast<std::size_t>(C) * (T + kHistory), seed++, -3, 3));
        records.push_back(to_device_bf16(values.back()));
        tails.emplace_back(static_cast<std::size_t>(C) * kHistory * 2);
        host.push_back({records.back().p, tails.back().p, C});
    }
    DeviceBuffer entries = to_device(host), committed(4);
    cuda_check(cudaMemcpy(committed.p, &committed_count, 4, cudaMemcpyHostToDevice), "cuda copy");
    ops::conv_tail_fold(static_cast<const ops::ConvTailFoldEntry*>(entries.p),
                        static_cast<std::int32_t>(host.size()), T,
                        Tensor(committed.p, DType::I32, {1}), nullptr);
    cuda_synchronize();
    int failures = 0;
    for (std::size_t e = 0; e < channels.size(); ++e) {
        const std::size_t C = static_cast<std::size_t>(channels[e]);
        const auto first    = values[e].begin() + std::ptrdiff_t(C) * committed_count;
        const std::vector<float> want(first, first + std::ptrdiff_t(C) * kHistory);
        failures +=
            verify_exact(("conv_tail_fold entry " + std::to_string(e)).c_str(),
                         from_device<std::uint16_t>(tails[e].p, C * kHistory), bf16_bits(want));
    }
    const std::string id = "conv_tail_fold " + std::to_string(channels.size()) +
                           " entries T=" + std::to_string(T) +
                           " n=" + std::to_string(committed_count);
    std::cout << (failures ? "FAIL " : "OK ") << id << '\n';
    return failures;
}

int run_rejections() {
    constexpr std::int32_t C = 64;
    constexpr std::int32_t T = 4;
    DeviceBuffer x(C * T * 2), w(C * 4 * 4), s(C * 3 * 2), r(C * (T + 3) * 2), o(C * T * 2);
    const Tensor tx(x.p, DType::BF16, {C, T});
    const Tensor tw(w.p, DType::FP32, {4, C});
    const Tensor ts(s.p, DType::BF16, {C, 3});
    Tensor tr(r.p, DType::BF16, {C, T + 3});
    Tensor to(o.p, DType::BF16, {C, T});
    int failures = 0;
    auto expect  = [&](const char* what, auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) { return; }
        std::cerr << "short_conv_record accepted " << what << '\n';
        ++failures;
    };
    expect("short record", [&] {
        Tensor shorter(r.p, DType::BF16, {C, T + 2});
        ops::short_conv_static_record(tx, tw, 1.0F, ts, shorter, to, nullptr);
    });
    expect("record over the state", [&] {
        Tensor alias(s.p, DType::BF16, {C, T + 3});
        ops::short_conv_static_record(tx, tw, 1.0F, ts, alias, to, nullptr);
    });
    expect("record over out", [&] {
        Tensor alias(o.p, DType::BF16, {C, T + 3});
        ops::short_conv_static_record(tx, tw, 1.0F, ts, alias, to, nullptr);
    });
    expect("zero-width fold", [&] {
        ops::conv_tail_fold(static_cast<const ops::ConvTailFoldEntry*>(r.p), 1, 0,
                            Tensor(s.p, DType::I32, {1}), nullptr);
    });
    expect("FP32 committed count", [&] {
        ops::conv_tail_fold(static_cast<const ops::ConvTailFoldEntry*>(r.p), 1, T,
                            Tensor(s.p, DType::FP32, {1}), nullptr);
    });
    std::cout << (failures ? "FAIL " : "OK ") << "short_conv_record rejections\n";
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
        failures += run_rejections();
        // Verification widths 1..8 (and 16): static over post-attention/post-FFN (2304), local
        // K/V at tp 1 (1024) and per rank (512), the synthetic checkpoint (256) and an odd width.
        std::uint32_t seed = 21;
        for (const std::int32_t C : {2304, 1024, 512, 256, 6}) {
            for (const std::int32_t T : {1, 2, 4, 8, 16}) {
                failures += run_case(C, T, 0, seed += 10);
            }
        }
        for (const auto& [C, G] : {std::pair{1024, 8}, std::pair{512, 4}, std::pair{256, 2}}) {
            for (const std::int32_t T : {1, 2, 4, 8, 16}) {
                failures += run_case(C, T, G, seed += 10);
            }
        }
        failures += run_many(8, 0);
        failures += run_many(8, 3);
        failures += run_many(8, 8);
        failures += run_many(2, 1);
    } catch (const std::exception& error) {
        std::cerr << "short_conv_record: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " short_conv_record\n";
    return failures ? 1 : 0;
}
