// YaRN text RoPE (ops::rope_install_text_yarn) on every Text route, on each CUDA device with the
// table installed on all of them: ops::rope pairs 24/4 and 16/2 (fixed kernels) and 24/2, 12/2,
// 12/1 (generic kernel), singles 24, 4, 16, 2 (fixed) and 12, 1 (generic), and
// rmsnorm_rope_text_shard 12/2, each with one position axis and with three (interleaved MRoPE).
// The QSA indexer is covered by ninfer_sparse_attention_test.
//
// Factor 4 against the Hugging Face formula (rope_type "yarn"): the table of ops/kernel/yarn.h,
// equal to transformers' reference bits (ninfer_yarn_test), the FP32 angle float(position) *
// inv_freq[i], cos and sin of it times the attention factor, the rotation in FP64. Positions run
// to 1,048,575 across the original window 262,144. The Op rounds an FP32 rotation to BF16 once,
// which is at most half a BF16 ulp, 2^-8 of the value; the FP32 sincosf and rotation add a few
// FP32 ulps of the pair norm, bounded here by 1e-5 of it. Unrotated dimensions stay bit-exact,
// and the installed table equals yarn.h's bit for bit.
//
// Factor 1 (the table removed again): every route's output is bit-identical to its output in this
// binary before any table was installed. A table on device 0 alone: a Text RoPE on device 1 throws
// that the table is not installed there.
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/tp2/rmsnorm_rope.h"
#include "ops/kernel/yarn.h"
#include "ops/launcher/rope.h"
#include "ops/op_tester.h"

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

constexpr int kHeadDim             = 256;
constexpr int kRotary              = 64;
constexpr int kHalf                = kRotary / 2;
constexpr float kTheta             = 1.0e7F;
constexpr float kFactor            = 4.0F;
constexpr std::uint32_t kOriginal  = 262144;
constexpr float kEps               = 1.0e-6F;
constexpr double kStorageRelative  = 0x1.0p-8;
constexpr double kPairNormAbsolute = 1.0e-5;

const std::vector<int> kBasePositions = {0,      1,      4095,   262143,  262144,  262145,
                                         393216, 524289, 786431, 1048000, 1048500, 1048575};

struct Table {
    std::vector<float> inv_freq;
    float attention_scale;
};

Table yarn_table() {
    rope::YarnParams params;
    params.factor                = kFactor;
    params.original_max_position = static_cast<int>(kOriginal);
    params.theta                 = kTheta;
    params.rotary_dim            = kRotary;
    return {rope::yarn_inv_freq(params), rope::yarn_attention_scale(params)};
}

std::uint32_t bits(float value) {
    std::uint32_t out = 0;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

// Axis-major positions [T] or [T,3]; the three axes differ so the pair-to-axis map matters.
std::vector<int> positions_for(int axes, int tokens) {
    std::vector<int> out(static_cast<std::size_t>(axes) * tokens);
    for (int axis = 0; axis < axes; ++axis) {
        for (int t = 0; t < tokens; ++t) {
            const int base = kBasePositions[static_cast<std::size_t>(
                (t + 5 * axis) % static_cast<int>(kBasePositions.size()))];
            out[static_cast<std::size_t>(axis) * tokens + t] = base;
        }
    }
    return out;
}

std::vector<std::uint16_t> bf16_input(std::size_t count, std::uint32_t seed) {
    std::vector<std::uint16_t> out(count);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-4.0F, 4.0F);
    for (auto& value : out) { value = f32_to_bf16(distribution(generator)); }
    return out;
}

// One [256, heads, T] tensor after RoPE against the Hugging Face YaRN formula.
int check_yarn(const std::string& label, const std::vector<std::uint16_t>& input,
               const std::vector<std::uint16_t>& got, const std::vector<int>& positions, int axes,
               int heads, int tokens, const Table& table) {
    int failures = 0;
    double worst = 0.0;
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < heads; ++h) {
            const std::size_t base =
                (static_cast<std::size_t>(t) * heads + static_cast<std::size_t>(h)) * kHeadDim;
            for (int d = kRotary; d < kHeadDim; ++d) {
                if (got[base + d] != input[base + d] && failures++ == 0) {
                    std::cerr << label << ": unrotated dim changed, token " << t << " head " << h
                              << " d " << d << '\n';
                }
            }
            for (int i = 0; i < kHalf; ++i) {
                const int axis = axes == 3 ? i % 3 : 0;
                const float angle =
                    static_cast<float>(positions[static_cast<std::size_t>(axis) * tokens + t]) *
                    table.inv_freq[static_cast<std::size_t>(i)];
                const double c      = std::cos(static_cast<double>(angle)) * table.attention_scale;
                const double s      = std::sin(static_cast<double>(angle)) * table.attention_scale;
                const double first  = bf16_to_f32(input[base + i]);
                const double second = bf16_to_f32(input[base + i + kHalf]);
                const double norm   = std::hypot(first, second) * table.attention_scale;
                const double expected[2] = {first * c - second * s, second * c + first * s};
                const double observed[2] = {bf16_to_f32(got[base + i]),
                                            bf16_to_f32(got[base + i + kHalf])};
                for (int j = 0; j < 2; ++j) {
                    const double error = std::abs(observed[j] - expected[j]);
                    const double bound =
                        kStorageRelative * std::abs(expected[j]) + kPairNormAbsolute * norm;
                    if (norm > 0.0) { worst = std::max(worst, error / norm); }
                    if (!(error <= bound) && failures++ == 0) {
                        std::cerr << label << ": token " << t << " (position "
                                  << positions[static_cast<std::size_t>(axis) * tokens + t]
                                  << ") head " << h << " pair " << i << " got " << observed[j]
                                  << " expected " << expected[j] << '\n';
                    }
                }
            }
        }
    }
    if (error_stats_enabled()) {
        std::cout << label << ": worst error / pair norm " << worst << '\n';
    }
    return failures != 0 ? 1 : 0;
}

struct Route {
    const char* name;
    int q_heads;
    int k_heads; // 0: the single-tensor form
};

const Route kRoutes[] = {
    {"pair 24/4 (fixed)", 24, 4},   {"pair 16/2 (fixed)", 16, 2},   {"pair 24/2 (generic)", 24, 2},
    {"pair 12/2 (generic)", 12, 2}, {"pair 12/1 (generic)", 12, 1}, {"single 24 (fixed)", 24, 0},
    {"single 4 (fixed)", 4, 0},     {"single 16 (fixed)", 16, 0},   {"single 2 (fixed)", 2, 0},
    {"single 12 (generic)", 12, 0}, {"single 1 (generic)", 1, 0},
};

// Output of one case: the rotated tensors of every route, then the text shard's q and k.
struct Outputs {
    std::vector<std::vector<std::uint16_t>> tensors;
};

// Runs every route on the current device; with `table`, checks each against the formula.
int run_routes(int device, int axes, int tokens, const Table* table, Outputs& out) {
    const std::string where = "device " + std::to_string(device) + " axes " + std::to_string(axes) +
                              " T=" + std::to_string(tokens) +
                              (table != nullptr ? " yarn x4: " : " plain: ");
    const std::vector<int> positions = positions_for(axes, tokens);
    const DeviceBuffer dpos          = to_device_i32(positions);
    const Tensor tpos =
        axes == 1 ? Tensor(dpos.p, DType::I32, {tokens}) : Tensor(dpos.p, DType::I32, {tokens, 3});
    int failures       = 0;
    std::uint32_t seed = 0x5a17U + static_cast<std::uint32_t>(axes * 31 + tokens);
    for (const Route& route : kRoutes) {
        const auto q_in =
            bf16_input(static_cast<std::size_t>(kHeadDim) * route.q_heads * tokens, seed++);
        const auto k_in =
            bf16_input(static_cast<std::size_t>(kHeadDim) * route.k_heads * tokens, seed++);
        const DeviceBuffer dq = to_device(q_in);
        Tensor tq(dq.p, DType::BF16, {kHeadDim, route.q_heads, tokens});
        if (route.k_heads == 0) {
            ops::rope(tpos, kRotary, kTheta, tq, nullptr);
        } else {
            const DeviceBuffer dk = to_device(k_in);
            Tensor tk(dk.p, DType::BF16, {kHeadDim, route.k_heads, tokens});
            ops::rope(tpos, kRotary, kTheta, tq, tk, nullptr);
            cuda_synchronize();
            out.tensors.push_back(from_device<std::uint16_t>(dk, k_in.size()));
            if (table != nullptr) {
                failures += check_yarn(where + route.name + " k", k_in, out.tensors.back(),
                                       positions, axes, route.k_heads, tokens, *table);
            }
        }
        cuda_synchronize();
        out.tensors.push_back(from_device<std::uint16_t>(dq, q_in.size()));
        if (table != nullptr) {
            failures += check_yarn(where + route.name + " q", q_in, out.tensors.back(), positions,
                                   axes, route.q_heads, tokens, *table);
        }
    }

    // rmsnorm_rope_text_shard 12/2: equal to rmsnorm + rmsnorm + rope bit for bit, and its
    // rotation of the materialized norm against the formula.
    const auto q_in = bf16_input(static_cast<std::size_t>(kHeadDim) * 12 * tokens, seed++);
    const auto k_in = bf16_input(static_cast<std::size_t>(kHeadDim) * 2 * tokens, seed++);
    std::vector<float> weight(kHeadDim);
    fill_uniform(weight, seed++, -0.9F, 2.0F);
    const DeviceBuffer dq = to_device(q_in), dk = to_device(k_in), dw = to_device_bf16(weight);
    const DeviceBuffer fq(q_in.size() * 2), fk(k_in.size() * 2), rq(q_in.size() * 2),
        rk(k_in.size() * 2);
    const Tensor tq(dq.p, DType::BF16, {kHeadDim, 12, tokens});
    const Tensor tk(dk.p, DType::BF16, {kHeadDim, 2, tokens});
    const Tensor tw(dw.p, DType::BF16, {kHeadDim});
    Tensor tfq(fq.p, DType::BF16, {kHeadDim, 12, tokens});
    Tensor tfk(fk.p, DType::BF16, {kHeadDim, 2, tokens});
    Tensor trq(rq.p, DType::BF16, {kHeadDim, 12, tokens});
    Tensor trk(rk.p, DType::BF16, {kHeadDim, 2, tokens});
    ops::rmsnorm_rope_text_shard(tpos, kRotary, kTheta, tw, tw, kEps, tq, tk, tfq, tfk, nullptr);
    ops::rmsnorm(tq, tw, kEps, true, trq, nullptr);
    ops::rmsnorm(tk, tw, kEps, true, trk, nullptr);
    cuda_synchronize();
    const auto nq = from_device<std::uint16_t>(rq, q_in.size());
    const auto nk = from_device<std::uint16_t>(rk, k_in.size());
    ops::rope(tpos, kRotary, kTheta, trq, trk, nullptr);
    cuda_synchronize();
    out.tensors.push_back(from_device<std::uint16_t>(fq, q_in.size()));
    out.tensors.push_back(from_device<std::uint16_t>(fk, k_in.size()));
    failures += verify_exact((where + "text shard q == rmsnorm + rope").c_str(),
                             out.tensors[out.tensors.size() - 2],
                             from_device<std::uint16_t>(rq, q_in.size()));
    failures += verify_exact((where + "text shard k == rmsnorm + rope").c_str(), out.tensors.back(),
                             from_device<std::uint16_t>(rk, k_in.size()));
    if (table != nullptr) {
        failures += check_yarn(where + "text shard 12/2 q", nq, out.tensors[out.tensors.size() - 2],
                               positions, axes, 12, tokens, *table);
        failures += check_yarn(where + "text shard 12/2 k", nk, out.tensors.back(), positions, axes,
                               2, tokens, *table);
    }
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
    std::vector<int> devices;
    for (int d = 0; d < std::min(device_count, 2); ++d) { devices.push_back(d); }
    const Table table = yarn_table();
    int failures      = 0;
    try {
        struct Case {
            int axes;
            int tokens;
        };

        const Case kCases[] = {{1, 1}, {1, 12}, {3, 12}};
        std::vector<Outputs> plain;
        for (const int device : devices) {
            cuda_check(cudaSetDevice(device), "cudaSetDevice");
            for (const Case& c : kCases) {
                failures += run_routes(device, c.axes, c.tokens, nullptr, plain.emplace_back());
            }
        }

        ops::rope_install_text_yarn(devices, kFactor, kOriginal, kRotary, kTheta);
        const ops::detail::TextRopeYarn* installed = ops::detail::text_rope_yarn(kRotary, kTheta);
        bool same_table =
            installed != nullptr && bits(installed->attention_scale) == bits(table.attention_scale);
        for (int i = 0; same_table && i < kHalf; ++i) {
            same_table =
                bits(installed->inv_freq[i]) == bits(table.inv_freq[static_cast<std::size_t>(i)]);
        }
        std::cout << "installed table == yarn.h: " << (same_table ? "yes" : "no") << '\n';
        failures += same_table ? 0 : 1;
        for (const int device : devices) {
            cuda_check(cudaSetDevice(device), "cudaSetDevice");
            for (const Case& c : kCases) {
                Outputs yarn;
                failures += run_routes(device, c.axes, c.tokens, &table, yarn);
            }
        }

        if (devices.size() == 2) {
            ops::rope_install_text_yarn({0}, kFactor, kOriginal, kRotary, kTheta);
            cuda_check(cudaSetDevice(1), "cudaSetDevice");
            bool threw = false;
            try {
                Outputs ignored;
                run_routes(1, 1, 1, &table, ignored);
            } catch (const std::logic_error& e) {
                threw = std::string(e.what()).find("not installed on CUDA device 1") !=
                        std::string::npos;
            }
            std::cout << "table on device 0 only, Text RoPE on device 1 throws: "
                      << (threw ? "yes" : "no") << '\n';
            failures += threw ? 0 : 1;
        }

        ops::rope_install_text_yarn(devices, 1.0F, kOriginal, kRotary, kTheta);
        std::size_t index = 0;
        for (const int device : devices) {
            cuda_check(cudaSetDevice(device), "cudaSetDevice");
            for (const Case& c : kCases) {
                Outputs again;
                failures += run_routes(device, c.axes, c.tokens, nullptr, again);
                const Outputs& before = plain[index++];
                for (std::size_t t = 0; t < again.tensors.size(); ++t) {
                    if (again.tensors[t] != before.tensors[t]) {
                        std::cerr << "device " << device << " axes " << c.axes << " T=" << c.tokens
                                  << ": output " << t
                                  << " after the table was removed differs from before\n";
                        ++failures;
                    }
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " YaRN text RoPE on " << devices.size()
              << " device(s)\n";
    return failures == 0 ? 0 : 1;
}
