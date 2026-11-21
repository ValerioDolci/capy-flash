// moe_sigmoid_route against an FP64 oracle (include/ninfer/ops/moe_sigmoid_route.h).
//
// Logits l_e = router[e] . x in FP64 from the represented inputs, s_e = sigmoid(l_e), keys
// s_e + bias[e]. The Op accumulates in FP32: a lane-strided sum then a butterfly, so
// |l32 - l| <= 2 * H * 2^-24 * sum_i |router[e,i] x[i]| (n*u of the absolute sum, doubled for
// the two-level order); sigmoid is 1/4-Lipschitz and FP32 expf, the division and the bias
// addition add a few ulp, so the key bound is
//   key_bound(t) = bound_l(t) / 4 + 8 * 2^-24 * (1 + max |key|).
// Selection: ids distinct and in [0,E); every chosen key >= the FP64 k-th key - key_bound; keys
// non-increasing along the output within key_bound; equal FP64 keys of identical router rows and
// biases (exact FP32 ties) in increasing id order, and the lower id chosen when the cut falls
// between them. Weights: w_j = s_j / (sum s + 1e-20) * scale over the Op's ids, in FP64; the
// score error s_bound = key_bound propagates as scale * (s_bound / S + s_j k s_bound / S^2), plus
// (k + 4) FP32 roundings of the result. Masked columns hold ids -1 and weights 0 exactly.
// Replication: a column's bits do not depend on T (the T=1 call of a column equals that column of
// the T=16 call) and, with two devices, device 1 gives the same bits as device 0.
#include "core/device.h"
#include "ninfer/ops/moe_sigmoid_route.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

struct RouteCase {
    std::string label;
    int hidden         = 0;
    int experts        = 0;
    int top_k          = 0;
    int tokens         = 0;
    float scale        = 8.0F;
    std::uint32_t seed = 0;
    bool ties          = false;
    std::vector<std::int32_t> valid; // empty: every column routes
    bool graph = false;
};

struct RouteInputs {
    std::vector<float> x;      // [T][H], BF16 values
    std::vector<float> router; // [E][H]
    std::vector<float> bias;   // [E]
};

RouteInputs make_inputs(const RouteCase& c) {
    RouteInputs in;
    in.x.resize(static_cast<std::size_t>(c.hidden) * c.tokens);
    in.router.resize(static_cast<std::size_t>(c.experts) * c.hidden);
    in.bias.resize(static_cast<std::size_t>(c.experts));
    fill_uniform(in.x, c.seed, -1.7F, 1.7F);
    round_to_bf16(in.x);
    // Logits of a few units: std(l) ~ sqrt(H) * 1 * a / sqrt(3).
    const float a = 3.5F / std::sqrt(static_cast<float>(c.hidden));
    fill_uniform(in.router, c.seed + 1, -a, a);
    fill_uniform(in.bias, c.seed + 2, -0.5F, 0.5F);
    if (c.ties) {
        // Exact FP32 key ties: identical rows and biases. Pair (1, 2) is chosen whole (large
        // bias); pair (5, 9) straddles the cut (k - 1 experts above it), so only 5 may be chosen.
        const auto copy_row = [&](int from, int to) {
            std::copy_n(in.router.begin() + static_cast<std::ptrdiff_t>(from) * c.hidden, c.hidden,
                        in.router.begin() + static_cast<std::ptrdiff_t>(to) * c.hidden);
        };
        copy_row(1, 2);
        in.bias[1] = in.bias[2] = 20.0F;
        copy_row(5, 9);
        in.bias[5] = in.bias[9] = 10.0F;
        int above               = 2;
        for (int e = 10; e < c.experts && above < c.top_k - 1; e += 1) {
            in.bias[e] = 15.0F;
            ++above;
        }
    }
    return in;
}

struct RouteOutputs {
    std::vector<std::int32_t> ids;
    std::vector<float> weights;
};

RouteOutputs run_op(const RouteCase& c, const RouteInputs& in, int device, int first_column,
                    int columns) {
    CUDA_CHECK(cudaSetDevice(device));
    std::vector<std::uint16_t> x_bits(static_cast<std::size_t>(c.hidden) * columns);
    for (std::size_t i = 0; i < x_bits.size(); ++i) {
        x_bits[i] = f32_to_bf16(in.x[static_cast<std::size_t>(first_column) * c.hidden + i]);
    }
    GuardedDeviceBuffer dx(x_bits.size() * 2), drouter(in.router.size() * 4),
        dbias(in.bias.size() * 4), dids(static_cast<std::size_t>(c.top_k) * columns * 4),
        dweights(static_cast<std::size_t>(c.top_k) * columns * 4),
        dvalid(std::max<std::size_t>(c.valid.size(), 1) * 4);
    dx.copy_from_host(x_bits.data(), dx.bytes());
    drouter.copy_from_host(in.router.data(), drouter.bytes());
    dbias.copy_from_host(in.bias.data(), dbias.bytes());
    dids.fill(0x7f);
    dweights.fill(0x7f);
    if (!c.valid.empty()) { dvalid.copy_from_host(c.valid.data(), c.valid.size() * 4); }
    Tensor tx(dx.data(), DType::BF16, {c.hidden, columns});
    Tensor trouter(drouter.data(), DType::FP32, {c.hidden, c.experts});
    Tensor tbias(dbias.data(), DType::FP32, {c.experts});
    Tensor tids(dids.data(), DType::I32, {c.top_k, columns});
    Tensor tweights(dweights.data(), DType::FP32, {c.top_k, columns});
    Tensor tvalid(dvalid.data(), DType::I32,
                  {std::max<std::int32_t>(static_cast<std::int32_t>(c.valid.size()), 1)});
    const Tensor* valid = c.valid.empty() ? nullptr : &tvalid;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (c.graph) {
        cudaGraph_t captured;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        ops::moe_sigmoid_route(tx, trouter, tbias, c.top_k, c.scale, tids, tweights, stream, valid);
        CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
        CUDA_CHECK(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(captured));
    } else {
        ops::moe_sigmoid_route(tx, trouter, tbias, c.top_k, c.scale, tids, tweights, stream, valid);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    RouteOutputs out;
    out.ids = from_device<std::int32_t>(dids.data(), static_cast<std::size_t>(c.top_k) * columns);
    out.weights = from_device<float>(dweights.data(), static_cast<std::size_t>(c.top_k) * columns);
    const int guards =
        dx.verify_guards(c.label + " x") + drouter.verify_guards(c.label + " router") +
        dbias.verify_guards(c.label + " bias") + dids.verify_guards(c.label + " ids") +
        dweights.verify_guards(c.label + " weights") + dvalid.verify_guards(c.label + " valid");
    if (guards != 0) { throw std::runtime_error(c.label + ": guard overwritten"); }
    CUDA_CHECK(cudaSetDevice(0));
    return out;
}

bool same_bits(const RouteOutputs& a, const RouteOutputs& b, std::size_t offset_a,
               std::size_t count) {
    return std::memcmp(a.ids.data() + offset_a, b.ids.data(), count * 4) == 0 &&
           std::memcmp(a.weights.data() + offset_a, b.weights.data(), count * 4) == 0;
}

int check_case(const RouteCase& c, int devices) {
    const RouteInputs in   = make_inputs(c);
    const RouteOutputs out = run_op(c, in, 0, 0, c.tokens);
    const int H = c.hidden, E = c.experts, K = c.top_k;
    const int width = c.valid.empty() ? c.tokens : c.tokens / static_cast<int>(c.valid.size());
    int failures    = 0;
    double worst_weight_ratio = 0, worst_weight = 0;
    int near_tie_positions = 0;
    for (int t = 0; t < c.tokens; ++t) {
        const std::int32_t* ids = out.ids.data() + static_cast<std::size_t>(t) * K;
        const float* w          = out.weights.data() + static_cast<std::size_t>(t) * K;
        if (!c.valid.empty() && t % width >= c.valid[static_cast<std::size_t>(t / width)]) {
            for (int j = 0; j < K; ++j) {
                if (ids[j] != -1 || w[j] != 0.0F) {
                    std::cerr << c.label << ": masked column " << t << " routes\n";
                    return failures + 1;
                }
            }
            continue;
        }
        std::vector<double> s(E), key(E);
        double max_abs = 0, max_key = 0;
        for (int e = 0; e < E; ++e) {
            double l = 0, abs_sum = 0;
            for (int i = 0; i < H; ++i) {
                const double p =
                    static_cast<double>(in.router[static_cast<std::size_t>(e) * H + i]) *
                    in.x[static_cast<std::size_t>(t) * H + i];
                l += p;
                abs_sum += std::fabs(p);
            }
            s[e]    = 1.0 / (1.0 + std::exp(-l));
            key[e]  = s[e] + in.bias[e];
            max_abs = std::max(max_abs, abs_sum);
            max_key = std::max(max_key, std::fabs(key[e]));
        }
        const double key_bound = 2.0 * H * std::ldexp(1.0, -24) * max_abs / 4.0 +
                                 8.0 * std::ldexp(1.0, -24) * (1 + max_key);
        std::vector<int> order(E);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return key[a] > key[b]; });
        const double kth = key[order[K - 1]];
        std::vector<bool> seen(E, false);
        for (int j = 0; j < K; ++j) {
            const int e = ids[j];
            if (e < 0 || e >= E || seen[e]) {
                std::cerr << c.label << ": column " << t << " invalid or repeated id " << e << '\n';
                return failures + 1;
            }
            seen[e] = true;
            if (key[e] < kth - key_bound) {
                std::cerr << c.label << ": column " << t << " chose " << e << " below the cut\n";
                ++failures;
            }
            if (j > 0 && key[e] > key[ids[j - 1]] + key_bound) {
                std::cerr << c.label << ": column " << t << " order violated at j=" << j << '\n';
                ++failures;
            }
            if (j > 0 && key[e] == key[ids[j - 1]] && e < ids[j - 1]) {
                std::cerr << c.label << ": column " << t << " exact tie not in id order\n";
                ++failures;
            }
            if (e != order[j]) { ++near_tie_positions; }
        }
        if (c.ties) {
            if (!seen[1] || !seen[2] || !seen[5] || seen[9]) {
                std::cerr << c.label << ": column " << t
                          << " exact ties not resolved to the lower id\n";
                ++failures;
            }
        }
        double total = 0;
        for (int j = 0; j < K; ++j) { total += s[ids[j]]; }
        for (int j = 0; j < K; ++j) {
            const double expected = s[ids[j]] / (total + 1e-20) * c.scale;
            const double limit =
                c.scale * (key_bound / total + s[ids[j]] * K * key_bound / (total * total)) +
                (K + 4) * std::ldexp(1.0, -24) * std::fabs(expected) + 1e-30;
            const double error = std::fabs(w[j] - expected);
            worst_weight       = std::max(worst_weight, error);
            worst_weight_ratio = std::max(worst_weight_ratio, error / limit);
        }
    }
    if (worst_weight_ratio > 1.0) {
        std::cerr << c.label << ": weight error " << worst_weight << " above its limit (x"
                  << worst_weight_ratio << ")\n";
        ++failures;
    }
    // A column's bits do not depend on T or on its position.
    if (c.valid.empty() && c.tokens > 1) {
        const int column       = c.tokens / 2;
        const RouteOutputs one = run_op(c, in, 0, column, 1);
        if (!same_bits(out, one, static_cast<std::size_t>(column) * K, K)) {
            std::cerr << c.label << ": column " << column << " differs alone (T=1)\n";
            ++failures;
        }
    }
    if (devices >= 2) {
        const RouteOutputs other = run_op(c, in, 1, 0, c.tokens);
        if (!same_bits(out, other, 0, static_cast<std::size_t>(K) * c.tokens)) {
            std::cerr << c.label << ": device 1 differs from device 0\n";
            ++failures;
        }
    }
    std::cout << "CASE " << c.label << " H=" << H << " E=" << E << " k=" << K << " T=" << c.tokens
              << " max_weight_err=" << worst_weight << " (limit ratio " << worst_weight_ratio
              << ") near_tie_positions=" << near_tie_positions
              << (devices >= 2 ? " device1=same" : "") << (failures ? " FAIL" : " ok") << '\n';
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    int failures = 0;
    try {
        std::vector<RouteCase> cases;
        for (const int tokens : {1, 4, 16, 512}) {
            cases.push_back({"real T=" + std::to_string(tokens), 2304, 128, 8, tokens, 8.0F,
                             100U + static_cast<std::uint32_t>(tokens)});
            cases.push_back({"synthetic T=" + std::to_string(tokens), 512, 16, 8, tokens, 8.0F,
                             200U + static_cast<std::uint32_t>(tokens)});
        }
        cases.push_back({"real ties", 2304, 128, 8, 4, 8.0F, 301U, true});
        cases.push_back({"synthetic ties", 512, 16, 8, 4, 8.0F, 302U, true});
        cases.push_back({"real valid columns", 2304, 128, 8, 12, 8.0F, 303U, false, {4, 0, 2}});
        cases.push_back(
            {"real valid columns wide", 2304, 128, 8, 256, 8.0F, 306U, false, {40, 128}});
        cases.push_back({"real graph", 2304, 128, 8, 4, 8.0F, 304U, false, {}, true});
        cases.push_back({"odd geometry", 100, 7, 7, 3, 2.5F, 305U});
        for (const auto& c : cases) { failures += check_case(c, devices); }
    } catch (const std::exception& error) {
        std::cerr << "moe_sigmoid_route test error: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << "moe_sigmoid_route: " << failures << " failures\n";
        return 1;
    }
    std::cout << "OK moe_sigmoid_route correctness\n";
    return 0;
}
