// FP64 oracles of the DFlash2 drafter Ops (ninfer/ops/dflash2_draft.h) and of the sliding block
// attention (ninfer/ops/dspark_draft.h) at the shapes of Inkling 2 Mini's DFlash2 drafter (H 2304,
// 144 convolution groups, 32/8 and 16/4 heads of 128, window 2048, vocabulary halves of 100029
// rows) and at small ones. The top candidates are exact (an order of stored values and ids).
#include "ninfer/ops/dflash2_draft.h"
#include "ninfer/ops/dspark_draft.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int D = 128;
// One BF16 rounding of an FP32 evaluation (the final store).
constexpr PointwiseCriterion kRounded{1e-5, 1.0 / 128.0};

std::vector<float> values(std::size_t n, std::uint32_t seed, float lo, float hi) {
    std::vector<float> v(n);
    fill_uniform(v, seed, lo, hi);
    round_to_bf16(v);
    return v;
}

int report(const char* name, int failures) {
    std::cout << (failures ? "FAIL " : "OK ") << name << '\n';
    return failures;
}

int test_conv(int H, int W, bool output_side, std::uint32_t seed) {
    const int G = H / 16;
    auto x      = values(std::size_t(H) * W, seed, -3, 3);
    auto c      = values(std::size_t(4) * G * W, seed + 1, -0.5, 0.5);
    auto base   = values(std::size_t(4) * H, seed + 2, -1, 1.5);
    auto r      = values(std::size_t(H) * W, seed + 3, -3, 3);
    auto dx = to_device_bf16(x), dc = to_device_bf16(c), db = to_device_bf16(base);
    auto dout = to_device_bf16(r);
    Tensor tx(dx.p, DType::BF16, {H, W}), tc(dc.p, DType::BF16, {4 * G, W}),
        tb(db.p, DType::BF16, {H, 2, 2}), tout(dout.p, DType::BF16, {H, W});
    if (output_side) {
        ops::dynamic_conv_output_add(tx, tc, tb, tout, nullptr);
    } else {
        ops::dynamic_conv_input(tx, tc, tb, tout, nullptr);
    }
    cuda_synchronize();
    const int side = output_side ? 1 : 0;
    std::vector<double> ref(std::size_t(H) * W);
    for (int i = 0; i < W; ++i) {
        for (int h = 0; h < H; ++h) {
            const int g            = h / 16;
            const auto coefficient = [&](int tap) {
                return double(base[std::size_t(h) + std::size_t(H) * tap +
                                   2 * std::size_t(H) * side]) +
                       double(c[std::size_t(i) * 4 * G + std::size_t(2 * side + tap) * G + g]);
            };
            double v = coefficient(0) * x[std::size_t(i) * H + h];
            if (i > 0) v += coefficient(1) * x[std::size_t(i - 1) * H + h];
            ref[std::size_t(i) * H + h] = output_side ? r[std::size_t(i) * H + h] + v : v;
        }
    }
    const std::string name =
        std::string(output_side ? "dynamic_conv_output_add" : "dynamic_conv_input") +
        " H=" + std::to_string(H) + " W=" + std::to_string(W);
    return report(name.c_str(),
                  verify_pointwise(name, from_device_bf16(dout, ref.size()), ref, kRounded));
}

// slots ranks over vocabulary blocks of `rows` (the last may be shorter); C columns. Values are
// drawn from few levels so that equal values are common and the id tie-break is exercised.
int test_top16(int rows, int C, int slots, std::uint32_t seed) {
    const int V = rows * slots;
    std::vector<float> logits(std::size_t(V) * C);
    fill_uniform(logits, seed, -4, 4);
    for (auto& v : logits) v = std::round(v * 8.0F) / 8.0F;
    round_to_bf16(logits);
    std::vector<DeviceBuffer> pairs;
    for (int s = 0; s < slots; ++s) {
        std::vector<float> block(std::size_t(rows) * C);
        for (int c = 0; c < C; ++c) {
            for (int v = 0; v < rows; ++v) {
                block[std::size_t(c) * rows + v] =
                    logits[std::size_t(c) * V + std::size_t(s) * rows + v];
            }
        }
        auto dl = to_device_bf16(block);
        pairs.emplace_back(std::size_t(2) * 16 * C * slots * 4);
        WorkspaceArena work(ops::vocab_top16_workspace_bytes(C) + 4096);
        Tensor tl(dl.p, DType::BF16, {rows, C}), tp(pairs.back().p, DType::FP32, {2, 16, C, slots});
        ops::vocab_top16_slots(tl, s * rows, s, tp, work, nullptr);
        cuda_synchronize();
    }
    // The exchange: the FP32 sum of every rank's slots.
    std::vector<float> sum(std::size_t(2) * 16 * C * slots, 0.0F);
    for (int s = 0; s < slots; ++s) {
        const auto part = from_device_f32(pairs[std::size_t(s)], sum.size());
        for (std::size_t i = 0; i < sum.size(); ++i) sum[i] += float(part[i]);
    }
    auto dsum = to_device_f32(sum);
    DeviceBuffer ids(std::size_t(16) * C * 4), vals(std::size_t(16) * C * 4);
    Tensor tp(dsum.p, DType::FP32, {2, 16, C, slots}), ti(ids.p, DType::I32, {16, C}),
        tv(vals.p, DType::FP32, {16, C});
    ops::vocab_top16_merge(tp, ti, tv, nullptr);
    cuda_synchronize();
    std::vector<int> got_ids(std::size_t(16) * C);
    cuda_check(cudaMemcpy(got_ids.data(), ids.p, got_ids.size() * 4, cudaMemcpyDeviceToHost),
               "copy ids");
    const auto got_vals = from_device_f32(vals, std::size_t(16) * C);
    int failures        = 0;
    for (int c = 0; c < C; ++c) {
        std::vector<int> order(static_cast<std::size_t>(V));
        std::iota(order.begin(), order.end(), 0);
        const float* col = &logits[std::size_t(c) * V];
        std::partial_sort(order.begin(), order.begin() + 16, order.end(),
                          [&](int a, int b) { return col[a] != col[b] ? col[a] > col[b] : a < b; });
        for (int k = 0; k < 16; ++k) {
            const int id = got_ids[std::size_t(c) * 16 + k];
            if (id != order[std::size_t(k)] ||
                got_vals[std::size_t(c) * 16 + k] != col[order[std::size_t(k)]]) {
                if (failures < 4) {
                    std::cout << "  column " << c << " rank " << k << ": id " << id << " value "
                              << got_vals[std::size_t(c) * 16 + k] << ", expected "
                              << order[std::size_t(k)] << " " << col[order[std::size_t(k)]] << '\n';
                }
                ++failures;
            }
        }
    }
    const std::string name = "vocab_top16 rows=" + std::to_string(rows) +
                             " C=" + std::to_string(C) + " slots=" + std::to_string(slots);
    return report(name.c_str(), failures);
}

// The query of block column b at P + b sees the cached positions p with P + b - p < window.
int test_sliding(int nq, int kv, int B, int P, int window, std::uint32_t seed) {
    const int capacity      = window;
    const int rows          = (nq + 2 * kv) * D;
    auto qkv                = values(std::size_t(rows) * B, seed, -1, 1);
    const std::size_t plane = std::size_t(D) * kv * capacity;
    auto kc = values(plane, seed + 1, -1, 1), vc = values(plane, seed + 2, -1, 1);
    auto dq = to_device_bf16(qkv), dk = to_device_bf16(kc), dv = to_device_bf16(vc);
    auto df = to_device_i32({P});
    DeviceBuffer out(std::size_t(nq) * D * B * 2);
    WorkspaceArena work(ops::block_attention_workspace_bytes(nq, B) + 4096);
    Tensor tq(dq.p, DType::BF16, {rows, B}), tf(df.p, DType::I32, {1}),
        to(out.p, DType::BF16, {nq * D, B});
    ops::DraftKVCacheView cache{Tensor(dk.p, DType::BF16, {D, kv, capacity}),
                                Tensor(dv.p, DType::BF16, {D, kv, capacity}), capacity};
    const float scale = 1.0F / std::sqrt(float(D));
    ops::sliding_block_attention(tq, nq, kv, cache, tf, window, scale, work, to, nullptr);
    cuda_synchronize();
    std::vector<double> ref(std::size_t(nq) * D * B);
    const int G = nq / kv;
    for (int b = 0; b < B; ++b) {
        const int lo = std::max(0, P + b - window + 1);
        for (int h = 0; h < nq; ++h) {
            const int g    = h / G;
            const float* q = &qkv[std::size_t(b) * rows + std::size_t(h) * D];
            std::vector<double> score;
            std::vector<const float*> vals;
            const auto key = [&](const float* k, const float* v) {
                double dot = 0;
                for (int d = 0; d < D; ++d) dot += double(q[d]) * k[d];
                score.push_back(dot * scale);
                vals.push_back(v);
            };
            for (int p = lo; p < P; ++p) {
                const std::size_t o = (std::size_t(p % capacity) * kv + g) * D;
                key(&kc[o], &vc[o]);
            }
            for (int j = 0; j < B; ++j) {
                const std::size_t c = std::size_t(j) * rows;
                key(&qkv[c + std::size_t(nq + g) * D], &qkv[c + std::size_t(nq + kv + g) * D]);
            }
            const double m = *std::max_element(score.begin(), score.end());
            double sum     = 0;
            for (auto& s : score) sum += s = std::exp(s - m);
            for (int d = 0; d < D; ++d) {
                double acc = 0;
                for (std::size_t j = 0; j < score.size(); ++j) acc += score[j] * vals[j][d];
                ref[(std::size_t(b) * nq + h) * D + d] = acc / sum;
            }
        }
    }
    const std::string name = "sliding_block_attention " + std::to_string(nq) + "/" +
                             std::to_string(kv) + " B=" + std::to_string(B) +
                             " P=" + std::to_string(P) + " window=" + std::to_string(window);
    return report(name.c_str(), verify_pointwise(name, from_device_bf16(out, ref.size()), ref,
                                                 PointwiseCriterion{2e-3, 1.0 / 128.0}));
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    try {
        failures += test_conv(2304, 8, false, 11);
        failures += test_conv(2304, 8, true, 12);
        failures += test_conv(64, 3, false, 13);
        failures += test_conv(64, 1, true, 14);
        failures += test_top16(100029, 7, 2, 21);
        failures += test_top16(200058, 3, 1, 22);
        failures += test_top16(40, 1, 2, 23);
        failures += test_sliding(32, 8, 8, 0, 2048, 31);
        failures += test_sliding(32, 8, 8, 700, 2048, 32);
        failures += test_sliding(32, 8, 8, 2044, 2048, 33); // the window starts inside the block
        failures += test_sliding(16, 4, 8, 2048, 2048, 34);
        failures += test_sliding(16, 4, 8, 13000, 2048, 35); // ring wrapped
        failures += test_sliding(8, 2, 5, 100, 64, 36);
    } catch (const std::exception& error) {
        std::cerr << "dflash2_draft: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " dflash2_draft\n";
    return failures ? 1 : 0;
}
