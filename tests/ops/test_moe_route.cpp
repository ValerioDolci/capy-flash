// moe_route / moe_combine_shared / moe_accumulate_shared against an FP64 oracle
// (include/ninfer/ops/moe_route.h).
//
// Router: logits l_e = router[e] . x evaluated in FP64 from the represented BF16 inputs. The Op
// accumulates in FP32, so its selection may differ from the FP64 selection only between experts
// whose FP64 logits lie within the FP32 accumulation bound of each other:
//   bound(t) = 2 * 2560 * 2^-24 * max_e sum_i |router[e,i] x[i,t]|   (a sum of 2560 products in
//   FP32 has a relative error below n*u of the absolute sum; doubled for the order of the two-level
//   warp reduction), so every chosen expert must have an FP64 logit >= the FP64 10th-largest logit
//   minus bound, the order must be non-increasing within bound, and the ids distinct.
// Weights: softmax renormalized over the Op's own chosen ids from FP64 logits, |err| <= 2e-5 (FP32
// exp of logits shifted by the maximum: a few ulp of values <= 1, plus the logit error times the
// weight). Shared gate: sigmoid of the FP64 score row, |err| <= 2e-6. Columns >= 64 take the
// tensor-core logits, whose FP32 accumulation is not the SIMT kernel's (gate error 2.9e-6 at
// T=300). The contract bound then applies to the gate too: |err| <= max(2e-6,
// bound_E(t) / 4), the FP32 accumulation bound of the score row (as for the logits) times the
// sigmoid's Lipschitz constant. Valid columns: a column past its lane's valid count must hold ids
// -1 and weights 0 exactly (its shared gate is still checked); the other columns are checked as
// above. Combine: exact against the FP32 fma + RNE defined by the contract.
#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/moe_route.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int H = ops::kMoeRouteHidden;
constexpr int E = ops::kMoeRouteExperts;
constexpr int K = ops::kMoeRouteTopK;

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

// `ties`: duplicate some router rows so exact logit ties occur (lower id must win).
// `valid` (empty: no mask): per lane the valid count of its tokens / valid.size() columns;
// `last_only`: only each lane's last valid column routes (MoeRouteColumns::LastValid).
int run_route_case(const std::string& label, int tokens, std::uint32_t seed, bool ties, bool graph,
                   const std::vector<std::int32_t>& valid = {}, bool last_only = false) {
    std::vector<float> x(static_cast<std::size_t>(H) * tokens);
    std::vector<float> router(static_cast<std::size_t>(E + 1) * H);
    fill_uniform(x, seed, -2.0f, 2.0f);
    fill_uniform(router, seed + 7, -0.05f, 0.05f);
    // A spread of row scales makes the softmax non-uniform, as a trained router.
    std::mt19937 rng(seed + 11);
    std::uniform_real_distribution<float> scale(0.2f, 3.0f);
    for (int e = 0; e <= E; ++e) {
        const float s = scale(rng);
        for (int i = 0; i < H; ++i) router[static_cast<std::size_t>(e) * H + i] *= s;
    }
    if (ties) {
        for (int e = 0; e < E; e += 37) {
            const int twin = (e + 101) % E;
            std::copy_n(router.begin() + static_cast<std::ptrdiff_t>(e) * H, H,
                        router.begin() + static_cast<std::ptrdiff_t>(twin) * H);
        }
    }
    round_to_bf16(x);
    round_to_bf16(router);
    const auto x_bits      = encode_bf16(x);
    const auto router_bits = encode_bf16(router);

    GuardedDeviceBuffer dx(x_bits.size() * 2), drouter(router_bits.size() * 2);
    dx.copy_from_host(x_bits.data(), dx.bytes());
    drouter.copy_from_host(router_bits.data(), drouter.bytes());
    GuardedDeviceBuffer dids(static_cast<std::size_t>(K) * tokens * 4),
        dweights(static_cast<std::size_t>(K) * tokens * 4),
        dgate(static_cast<std::size_t>(tokens) * 4);
    dids.fill(0xff);
    dweights.fill(0xff);
    dgate.fill(0xff);
    Tensor tx(dx.data(), DType::BF16, {H, tokens});
    Tensor trouter(drouter.data(), DType::BF16, {H, E + 1});
    Tensor tids(dids.data(), DType::I32, {K, tokens});
    Tensor tweights(dweights.data(), DType::FP32, {K, tokens});
    Tensor tgate(dgate.data(), DType::FP32, {tokens});
    WorkspaceArena work(ops::moe_route_workspace_bytes(tokens));
    const int lanes = valid.empty() ? 1 : static_cast<int>(valid.size());
    const int width = tokens / lanes;
    GuardedDeviceBuffer dvalid(std::max<std::size_t>(valid.size(), 1) * 4);
    if (!valid.empty()) { dvalid.copy_from_host(valid.data(), valid.size() * 4); }
    Tensor tvalid(dvalid.data(), DType::I32, {lanes});
    const Tensor* valid_columns = valid.empty() ? nullptr : &tvalid;
    const auto routed           = [&](int t) {
        if (valid.empty()) { return true; }
        const std::int32_t count = valid[static_cast<std::size_t>(t / width)];
        return last_only ? t % width == count - 1 : t % width < count;
    };
    const ops::MoeRouteColumns mode =
        last_only ? ops::MoeRouteColumns::LastValid : ops::MoeRouteColumns::ValidPrefix;

    if (graph) {
        cudaStream_t stream;
        cudaGraph_t captured;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        ops::moe_route(tx, trouter, work, tids, tweights, tgate, stream, valid_columns, mode);
        CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
        CUDA_CHECK(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(captured));
        CUDA_CHECK(cudaStreamDestroy(stream));
    } else {
        ops::moe_route(tx, trouter, work, tids, tweights, tgate, nullptr, valid_columns, mode);
        cuda_synchronize();
    }

    const auto ids = from_device<std::int32_t>(dids.data(), static_cast<std::size_t>(K) * tokens);
    const auto weights = from_device<float>(dweights.data(), static_cast<std::size_t>(K) * tokens);
    const auto gate    = from_device<float>(dgate.data(), static_cast<std::size_t>(tokens));

    int failures        = 0;
    double worst_weight = 0, worst_gate = 0, worst_gate_ratio = 0, worst_margin_ratio = 0;
    const bool mma           = tokens >= 64;
    int near_tie_differences = 0;
    int masked               = 0;
    for (int t = 0; t < tokens; ++t) {
        std::vector<double> l(E + 1);
        double max_abs_sum = 0, score_abs_sum = 0;
        for (int e = 0; e <= E; ++e) {
            double sum = 0, abs_sum = 0;
            for (int i = 0; i < H; ++i) {
                const double p = static_cast<double>(router[static_cast<std::size_t>(e) * H + i]) *
                                 x[static_cast<std::size_t>(t) * H + i];
                sum += p;
                abs_sum += std::fabs(p);
            }
            l[e]        = sum;
            max_abs_sum = std::max(max_abs_sum, abs_sum);
            if (e == E) { score_abs_sum = abs_sum; }
        }
        const double bound = 2.0 * H * std::ldexp(1.0, -24) * max_abs_sum;
        std::vector<int> order(E);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return l[a] > l[b]; });
        const double tenth         = l[order[K - 1]];
        const int* row             = ids.data() + static_cast<std::size_t>(t) * K;
        const double expected_gate = 1.0 / (1.0 + std::exp(-l[E]));
        const double gate_error    = std::fabs(gate[t] - expected_gate);
        const double gate_limit =
            mma ? std::max(2e-6, 0.25 * 2.0 * H * std::ldexp(1.0, -24) * score_abs_sum) : 2e-6;
        worst_gate       = std::max(worst_gate, gate_error);
        worst_gate_ratio = std::max(worst_gate_ratio, gate_error / gate_limit);
        if (!routed(t)) {
            for (int k = 0; k < K; ++k) {
                if (row[k] != -1 || weights[static_cast<std::size_t>(t) * K + k] != 0.0F) {
                    std::cerr << label << ": token " << t << " past its lane's valid count routes "
                              << row[k] << " with weight "
                              << weights[static_cast<std::size_t>(t) * K + k] << '\n';
                    return failures + 1;
                }
            }
            ++masked;
            continue;
        }
        std::vector<bool> seen(E, false);
        for (int k = 0; k < K; ++k) {
            const int e = row[k];
            if (e < 0 || e >= E || seen[e]) {
                std::cerr << label << ": token " << t << " invalid or repeated id " << e << '\n';
                return failures + 1;
            }
            seen[e] = true;
            if (l[e] < tenth - bound) {
                std::cerr << label << ": token " << t << " chose expert " << e << " logit " << l[e]
                          << " below the 10th " << tenth << " - " << bound << '\n';
                ++failures;
            }
            if (k > 0 && l[e] > l[row[k - 1]] + bound) {
                std::cerr << label << ": token " << t << " order violated at k=" << k << '\n';
                ++failures;
            }
            if (e != order[k]) ++near_tie_differences;
            worst_margin_ratio = std::max(worst_margin_ratio, (tenth - l[e]) / bound);
        }
        // Exact ties: equal FP64 logits and equal rows -> the lower id must come first.
        for (int k = 1; k < K; ++k) {
            if (l[row[k]] == l[row[k - 1]] && row[k] < row[k - 1]) {
                std::cerr << label << ": token " << t
                          << " exact tie not resolved to the lower id\n";
                ++failures;
            }
        }
        double total = 0;
        for (int k = 0; k < K; ++k) total += std::exp(l[row[k]] - l[row[0]]);
        for (int k = 0; k < K; ++k) {
            const double expected = std::exp(l[row[k]] - l[row[0]]) / total;
            worst_weight          = std::max(
                worst_weight, std::fabs(weights[static_cast<std::size_t>(t) * K + k] - expected));
        }
    }
    if (masked != static_cast<int>(std::count_if(ids.begin(), ids.end(),
                                                 [](std::int32_t e) { return e == -1; })) /
                      K) {
        std::cerr << label << ": a valid column routes nothing\n";
        ++failures;
    }
    if (worst_weight > 2e-5) {
        std::cerr << label << ": routing weight error " << worst_weight << " > 2e-5\n";
        ++failures;
    }
    if (worst_gate_ratio > 1.0) {
        std::cerr << label << ": shared gate error " << worst_gate << " above its limit (x"
                  << worst_gate_ratio << ")\n";
        ++failures;
    }
    failures += dx.verify_guards(label + " x") + drouter.verify_guards(label + " router") +
                dids.verify_guards(label + " ids") + dweights.verify_guards(label + " weights") +
                dgate.verify_guards(label + " gate") + dvalid.verify_guards(label + " valid");
    if (error_stats_enabled()) {
        std::cout << "STATS " << label << " tokens=" << tokens << " weight_err=" << worst_weight
                  << " gate_err=" << worst_gate << " near_tie_positions=" << near_tie_differences
                  << " worst_margin/bound=" << worst_margin_ratio << '\n';
    }
    return failures;
}

int run_combine_case(int rows, int tokens, std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(rows) * tokens;
    std::vector<float> partial(count), shared(count), gate(tokens);
    fill_uniform(partial, seed, -30.0f, 30.0f);
    fill_uniform(shared, seed + 1, -8.0f, 8.0f);
    fill_uniform(gate, seed + 2, 0.0f, 1.0f);
    round_to_bf16(shared);
    const auto shared_bits = encode_bf16(shared);
    GuardedDeviceBuffer dp(count * 4), ds(count * 2), dg(static_cast<std::size_t>(tokens) * 4),
        dy(count * 2);
    dp.copy_from_host(partial.data(), dp.bytes());
    ds.copy_from_host(shared_bits.data(), ds.bytes());
    dg.copy_from_host(gate.data(), dg.bytes());
    Tensor tp(dp.data(), DType::FP32, {rows, tokens});
    Tensor ts(ds.data(), DType::BF16, {rows, tokens});
    Tensor tg(dg.data(), DType::FP32, {tokens});
    Tensor ty(dy.data(), DType::BF16, {rows, tokens});
    ops::moe_combine_shared(tp, tg, ts, ty, nullptr);
    cuda_synchronize();
    std::vector<std::uint16_t> expected(count);
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t t = i / static_cast<std::size_t>(rows);
        expected[i]         = f32_to_bf16(std::fma(gate[t], shared[i], partial[i]));
    }
    int failures =
        verify_exact("moe_combine_shared", from_device<std::uint16_t>(dy.data(), count), expected);
    failures += dp.verify_guards("combine partial") + ds.verify_guards("combine shared") +
                dg.verify_guards("combine gate") + dy.verify_guards("combine y");

    // moe_accumulate_shared (the tp 2 form): the same FP32 fma kept in FP32, in place; its BF16
    // rounding is moe_combine_shared's output bit for bit.
    GuardedDeviceBuffer da(count * 4);
    da.copy_from_host(partial.data(), da.bytes());
    Tensor ta(da.data(), DType::FP32, {rows, tokens});
    ops::moe_accumulate_shared(tg, ts, ta, nullptr);
    cuda_synchronize();
    const std::vector<float> accumulated = from_device<float>(da.data(), count);
    std::size_t fp32_mismatches = 0, rounding_mismatches = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t t    = i / static_cast<std::size_t>(rows);
        const float want       = std::fma(gate[t], shared[i], partial[i]);
        std::uint32_t got_bits = 0, want_bits = 0;
        std::memcpy(&got_bits, &accumulated[i], 4);
        std::memcpy(&want_bits, &want, 4);
        fp32_mismatches += got_bits != want_bits;
        rounding_mismatches += f32_to_bf16(accumulated[i]) != expected[i];
    }
    if (fp32_mismatches != 0 || rounding_mismatches != 0) {
        std::cerr << "moe_accumulate_shared: " << fp32_mismatches << " FP32 and "
                  << rounding_mismatches << " BF16-rounding mismatches of " << count << '\n';
        ++failures;
    }
    failures += da.verify_guards("accumulate partial");
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
        for (const int tokens : {1, 3, 8, 65, 300}) {
            failures += run_route_case("moe_route T=" + std::to_string(tokens), tokens,
                                       1000U + static_cast<std::uint32_t>(tokens), false, false);
        }
        failures += run_route_case("moe_route ties", 9, 77U, true, false);
        failures += run_route_case("moe_route graph", 4, 91U, false, true);
        // A verify window W = 4 of 3 lanes: extents 3 (all valid), 0 (anchor only), 1.
        failures += run_route_case("moe_route valid columns", 12, 93U, false, false, {4, 1, 2});
        failures +=
            run_route_case("moe_route valid columns graph", 12, 95U, false, true, {4, 1, 2});
        // MTP AR step: one column per lane, lanes past their extent route nothing.
        failures += run_route_case("moe_route valid columns W=1", 3, 97U, false, false, {1, 0, 1});
        // MTP alignment: W = 4 of 4 lanes, only the accepted (last valid) column routes;
        // counts 4, 1, 0 (nothing), 2.
        failures +=
            run_route_case("moe_route last valid", 16, 99U, false, false, {4, 1, 0, 2}, true);
        failures +=
            run_route_case("moe_route last valid graph", 16, 101U, false, true, {4, 1, 0, 2}, true);
        failures += run_combine_case(2560, 1, 5U);
        failures += run_combine_case(2560, 8, 6U);
        failures += run_combine_case(2560, 129, 7U);
    } catch (const std::exception& error) {
        std::cerr << "moe_route test error: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << "moe_route: " << failures << " failures\n";
        return 1;
    }
    std::cout << "OK moe_route correctness\n";
    return 0;
}
