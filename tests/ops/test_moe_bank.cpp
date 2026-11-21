// moe_bank_capped_swiglu / moe_partial_to_bf16 against an FP64 oracle
// (include/ninfer/ops/moe_bank.h).
//
// Fixture: a stacked bank parent built on the host (nvfp4 block_scale_k16_m128x4_v1 with one
// divisor per expert or per gate/up source, or contiguous bf16), uploaded, and bound as a view of
// whole rows (a complete parent or a range of experts inside a larger one). The oracle decodes
// every weight exactly from the natural fields it generated (E2M1 code, E4M3FN scale, FP32
// divisor; docs tensor-formats §3.3) and evaluates the contract in FP64 with its three semantic
// BF16 roundings (g/a, h, o).
//
// Criterion (rigorous, per element): the Op's FP32 dots differ from FP64 by at most
// (2n + 8) * 2^-24 * sum |w x| (n terms, lane-strided then a butterfly, one divisor division).
// Near a BF16 rounding midpoint that can round g or a to the other neighbour, so the oracle
// carries every reachable BF16 value: g, a in {bf16(v - d), bf16(v), bf16(v + d)}; h over those,
// widened by the FP32 activation's relative error 2^-19; o in bf16(o64 -/+ D) with D the sum of
// |W_d| times the reachable h deviations plus the FP32 dot bound. The partial may differ from
// sum_j w_j o_ref_j by sum_j |w_j| * (largest reachable |o - o_ref|) + (k + 2) * 2^-24 * sum |w o|.
// The reported error is max |partial - S_ref| and its ratio to that limit.
// Accumulate: from a random FP32 init A, the result must equal fl32(A + Store result) bit for
// bit (the contract's tp 1 == tp 2 property: Store(bank 0) then Accumulate(bank 1) equals
// allreduce_sum_f32's fl32(p0 + p1)). moe_partial_to_bf16 must equal host RNE exactly.
#include "core/arena.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "ninfer/ops/moe_bank.h"
#include "ops/nvfp4_test_codec.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr double kU32 = 0x1p-24;

std::uint64_t splitmix(std::uint64_t& state) {
    std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z               = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z               = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Round a double to the nearest BF16 value, ties to even (no double rounding through FP32).
double bf16_of(double v) {
    if (v == 0.0 || !std::isfinite(v)) { return v; }
    int exponent = 0;
    std::frexp(v, &exponent); // |v| in [2^(e-1), 2^e)
    const double ulp = std::ldexp(1.0, std::max(exponent - 8, -133));
    return std::nearbyint(v / ulp) * ulp;
}

// One stacked parent on the host and on the device.
struct Parent {
    bool nvfp4   = false;
    bool mxfp8   = false;
    int rows     = 0;
    int cols     = 0;
    int divisors = 1;
    WeightGeometry geometry;
    std::vector<std::uint8_t> payload;       // the layout bytes
    std::vector<std::uint8_t> scale_natural; // nvfp4 [rows][cols/16]
    std::vector<float> divisor;              // nvfp4 [divisors]
    std::unique_ptr<GuardedDeviceBuffer> device;
    WeightParent bound;

    double weight(std::int64_t row, int col) const {
        if (mxfp8) {
            const std::uint8_t code = payload[static_cast<std::size_t>(row * cols + col)];
            const std::uint8_t exponent =
                payload[geometry.scale_offset + static_cast<std::size_t>(row * (cols / 32) + col / 32)];
            return std::ldexp(static_cast<double>(e4m3_decode(code)), exponent - 127);
        }
        if (!nvfp4) {
            std::uint16_t bits = 0;
            std::memcpy(&bits, payload.data() + (row * cols + col) * 2, 2);
            return bf16_to_f32(bits);
        }
        const std::uint8_t byte = payload[static_cast<std::size_t>(row * (cols / 2) + col / 2)];
        const int code          = (col % 2 == 0) ? (byte & 0xF) : (byte >> 4);
        const std::uint8_t sc =
            scale_natural[static_cast<std::size_t>(row * (cols / 16) + col / 16)];
        return static_cast<double>(kE2M1[code]) * e4m3_decode(sc) /
               divisor[static_cast<std::size_t>(row / (rows / divisors))];
    }
};

// Scale words as in the checkpoint (E4M3FN 30..448: exponent 11..15, never NaN); divisor per
// source den / amax with den = 6 * 448 and amax in [0.1, 0.33]. BF16 weights of the same spread.
// MXFP8 (`mxfp8`, then `nvfp4` is ignored): E4M3FN codes with exponent field 0..8 (never NaN)
// and UE8M0 exponents 118..121, a spread close to the BF16 weights.
std::unique_ptr<Parent> make_parent(bool nvfp4, int rows, int cols, int divisors,
                                    std::uint64_t seed, bool mxfp8 = false) {
    nvfp4       = nvfp4 && !mxfp8;
    auto p      = std::make_unique<Parent>();
    p->nvfp4    = nvfp4;
    p->mxfp8    = mxfp8;
    p->rows     = rows;
    p->cols     = cols;
    p->divisors = nvfp4 ? divisors : 1;
    const std::array<std::uint64_t, 2> shape{static_cast<std::uint64_t>(rows),
                                             static_cast<std::uint64_t>(cols)};
    p->geometry =
        mxfp8 ? weight_geometry(QType::MXFP8_E4M3FN_G32_UE8M0, QuantLayout::BlockScaleK32, shape)
              : weight_geometry(nvfp4 ? QType::NVFP4 : QType::BF16,
                                nvfp4 ? QuantLayout::BlockScaleK16M128x4 : QuantLayout::Contiguous,
                                shape, static_cast<std::uint64_t>(p->divisors));
    p->payload.assign(p->geometry.bytes, 0);
    std::uint64_t state = seed;
    if (mxfp8) {
        const std::size_t count = static_cast<std::size_t>(rows) * cols;
        for (std::size_t i = 0; i < count; ++i) {
            const std::uint64_t r = splitmix(state);
            p->payload[i] = static_cast<std::uint8_t>(((r & 1U) << 7) | ((r >> 1) % 9) << 3 | ((r >> 8) & 7U));
        }
        for (std::size_t i = 0; i < count / 32; ++i) {
            p->payload[p->geometry.scale_offset + i] = static_cast<std::uint8_t>(118 + splitmix(state) % 4);
        }
    } else if (nvfp4) {
        const std::size_t code_bytes = static_cast<std::size_t>(rows) * cols / 2;
        for (std::size_t i = 0; i < code_bytes; i += 8) {
            const std::uint64_t r = splitmix(state);
            std::memcpy(p->payload.data() + i, &r, 8);
        }
        const int groups = cols / 16;
        p->scale_natural.resize(static_cast<std::size_t>(rows) * groups);
        for (auto& s : p->scale_natural) {
            const std::uint64_t r = splitmix(state);
            const int e           = 11 + static_cast<int>(r % 5);
            const int m           = static_cast<int>((r >> 8) % (e == 15 ? 7 : 8));
            s                     = static_cast<std::uint8_t>((e << 3) | m);
        }
        const int k_tiles = cols / 64;
        for (int n = 0; n < rows; ++n) {
            const int inner = n % 128;
            for (int g = 0; g < groups; ++g) {
                const std::size_t offset =
                    p->geometry.scale_offset +
                    (static_cast<std::size_t>(n / 128) * k_tiles + g / 4) * 512 +
                    static_cast<std::size_t>(inner % 32) * 16 +
                    static_cast<std::size_t>(inner / 32) * 4 + g % 4;
                p->payload[offset] = p->scale_natural[static_cast<std::size_t>(n) * groups + g];
            }
        }
        p->divisor.resize(static_cast<std::size_t>(p->divisors));
        for (auto& d : p->divisor) {
            const double amax = 0.1 + 0.23 * static_cast<double>(splitmix(state) >> 11) * 0x1p-53;
            d                 = static_cast<float>(2688.0 / amax);
        }
        std::memcpy(p->payload.data() + p->geometry.divisor_offset, p->divisor.data(),
                    p->divisor.size() * 4);
    } else {
        // Uniform with the RMS of the NVFP4 weights (~0.035 * sqrt(2304 / cols)).
        const double a          = 0.06 * std::sqrt(2304.0 / cols);
        const std::size_t count = static_cast<std::size_t>(rows) * cols;
        for (std::size_t i = 0; i < count; ++i) {
            const double u           = static_cast<double>(splitmix(state) >> 11) * 0x1p-53;
            const std::uint16_t bits = f32_to_bf16(static_cast<float>((2 * u - 1) * a));
            std::memcpy(p->payload.data() + i * 2, &bits, 2);
        }
    }
    p->device = std::make_unique<GuardedDeviceBuffer>(p->payload.size());
    p->device->copy_from_host(p->payload.data(), p->payload.size());
    p->bound.geometry = p->geometry;
    p->bound.data     = static_cast<const std::byte*>(p->device->data());
    return p;
}

WeightView rows_of(const Parent& p, std::int64_t row_begin, std::int64_t row_count) {
    WeightView view;
    view.shape = {static_cast<std::uint64_t>(row_count), static_cast<std::uint64_t>(p.cols)};
    view.parts = {{&p.bound, static_cast<std::uint64_t>(row_begin) * p.cols,
                   static_cast<std::uint64_t>(row_begin + row_count) * p.cols}};
    return view;
}

// A bank fixture: parents holding `parent_experts` experts, of which the view binds
// [view_first, view_first + bank_experts) with global ids starting at first_expert.
struct BankConfig {
    std::string label;
    bool nvfp4          = true;
    int hidden          = 0;
    int intermediate    = 0;
    int parent_experts  = 0;
    int view_first      = 0;
    int bank_experts    = 0;
    int first_expert    = 0;
    int total_experts   = 0; // routing id range
    int gate_up_sources = 1; // divisors per expert in gate_up (1, or 2 = gate and up apart)
    bool mxfp8          = false; // MXFP8 planes (nvfp4 ignored)
};

struct Bank {
    BankConfig config;
    std::unique_ptr<Parent> gate_up, down;
    WeightView gate_up_view, down_view;
};

Bank make_bank(const BankConfig& c, std::uint64_t seed) {
    Bank b;
    b.config  = c;
    b.gate_up = make_parent(c.nvfp4, c.parent_experts * 2 * c.intermediate, c.hidden,
                            c.parent_experts * c.gate_up_sources, seed, c.mxfp8);
    b.down    = make_parent(c.nvfp4, c.parent_experts * c.hidden, c.intermediate, c.parent_experts,
                            seed + 1, c.mxfp8);
    b.gate_up_view =
        rows_of(*b.gate_up, static_cast<std::int64_t>(c.view_first) * 2 * c.intermediate,
                static_cast<std::int64_t>(c.bank_experts) * 2 * c.intermediate);
    b.down_view = rows_of(*b.down, static_cast<std::int64_t>(c.view_first) * c.hidden,
                          static_cast<std::int64_t>(c.bank_experts) * c.hidden);
    return b;
}

struct Routing {
    std::vector<float> x; // [T][H], BF16 values
    std::vector<std::int32_t> ids;
    std::vector<float> weights;
};

// Random distinct ids per token over [0, total); token 0 has every position in the bank, token 1
// none (ids outside it, or -1 when the bank covers every id), token 2 half -1; with `repeat`,
// token 3 repeats its first expert.
Routing make_routing(const BankConfig& c, int tokens, int top_k, std::uint32_t seed, bool repeat) {
    Routing r;
    r.x.resize(static_cast<std::size_t>(tokens) * c.hidden);
    fill_uniform(r.x, seed, -1.7F, 1.7F);
    round_to_bf16(r.x);
    r.ids.resize(static_cast<std::size_t>(tokens) * top_k);
    r.weights.resize(r.ids.size());
    std::mt19937 rng(seed + 1);
    std::uniform_real_distribution<float> wdist(0.1F, 1.0F);
    const auto in_bank = [&](int id) {
        return id >= c.first_expert && id < c.first_expert + c.bank_experts;
    };
    for (int t = 0; t < tokens; ++t) {
        std::vector<int> pool;
        for (int e = 0; e < c.total_experts; ++e) {
            if (t == 0 && !in_bank(e)) { continue; }
            if (t == 1 && in_bank(e)) { continue; }
            pool.push_back(e);
        }
        std::shuffle(pool.begin(), pool.end(), rng);
        float total = 0;
        for (int j = 0; j < top_k; ++j) {
            int id = j < static_cast<int>(pool.size()) ? pool[static_cast<std::size_t>(j)]
                                                       : (t == 0 ? pool[j % pool.size()] : -1);
            if (t == 2 && j % 2 == 1) { id = -1; }
            if (repeat && t == 3 && j == 1) { id = r.ids[static_cast<std::size_t>(t) * top_k]; }
            r.ids[static_cast<std::size_t>(t) * top_k + j]     = id;
            const float w                                      = wdist(rng);
            r.weights[static_cast<std::size_t>(t) * top_k + j] = w;
            total += w;
        }
        for (int j = 0; j < top_k; ++j) {
            r.weights[static_cast<std::size_t>(t) * top_k + j] *= 8.0F / total;
        }
    }
    return r;
}

struct Activation {
    double gate_cap = 4.0, up_cap = 25.0;

    double operator()(double g, double a) const {
        return gate_cap * std::tanh(g / gate_cap) * (1.0 / (1.0 + std::exp(-g))) * up_cap *
               std::tanh(a / up_cap);
    }
};

struct OracleToken {
    std::vector<double> s;     // [H] sum_j w o_ref
    std::vector<double> limit; // [H]
    bool any = false;
};

OracleToken oracle_token(const Bank& b, const Routing& r, int t, int top_k, Activation act) {
    const auto& c = b.config;
    const int H = c.hidden, F = c.intermediate;
    OracleToken out;
    out.s.assign(static_cast<std::size_t>(H), 0.0);
    out.limit.assign(static_cast<std::size_t>(H), 0.0);
    std::vector<double> abs_s(static_cast<std::size_t>(H), 0.0);
    const float* x = r.x.data() + static_cast<std::size_t>(t) * H;
    std::vector<double> h_ref(static_cast<std::size_t>(F)), h_dev(static_cast<std::size_t>(F)),
        h_max(static_cast<std::size_t>(F));
    for (int j = 0; j < top_k; ++j) {
        const int id = r.ids[static_cast<std::size_t>(t) * top_k + j];
        if (id < c.first_expert || id >= c.first_expert + c.bank_experts) { continue; }
        out.any                 = true;
        const double w          = r.weights[static_cast<std::size_t>(t) * top_k + j];
        const std::int64_t e    = c.view_first + (id - c.first_expert);
        const std::int64_t base = e * 2 * F;
        const auto dot          = [&](std::int64_t row, double& abs_sum) {
            double sum = 0;
            abs_sum    = 0;
            for (int i = 0; i < H; ++i) {
                const double p = b.gate_up->weight(row, i) * x[i];
                sum += p;
                abs_sum += std::fabs(p);
            }
            return sum;
        };
        for (int i = 0; i < F; ++i) {
            double abs_g = 0, abs_a = 0;
            const double g64 = dot(base + i, abs_g);
            const double a64 = dot(base + F + i, abs_a);
            const double dg = (2.0 * H + 8) * kU32 * abs_g, da = (2.0 * H + 8) * kU32 * abs_a;
            const double g_ref = bf16_of(g64), a_ref = bf16_of(a64);
            const double gs[3] = {bf16_of(g64 - dg), g_ref, bf16_of(g64 + dg)};
            const double as[3] = {bf16_of(a64 - da), a_ref, bf16_of(a64 + da)};
            const double ref   = bf16_of(act(g_ref, a_ref));
            double dev = 0, hmax = std::fabs(ref);
            for (const double gc : gs) {
                for (const double ac : as) {
                    const double v = act(gc, ac);
                    for (const double cand :
                         {bf16_of(v * (1 - 0x1p-19)), bf16_of(v * (1 + 0x1p-19))}) {
                        dev  = std::max(dev, std::fabs(cand - ref));
                        hmax = std::max(hmax, std::fabs(cand));
                    }
                }
            }
            h_ref[static_cast<std::size_t>(i)] = ref;
            h_dev[static_cast<std::size_t>(i)] = dev;
            h_max[static_cast<std::size_t>(i)] = hmax;
        }
        for (int row = 0; row < H; ++row) {
            const std::int64_t prow = e * H + row;
            double o64 = 0, spread = 0, abs_o = 0;
            for (int i = 0; i < F; ++i) {
                const double d = b.down->weight(prow, i);
                o64 += d * h_ref[static_cast<std::size_t>(i)];
                spread += std::fabs(d) * h_dev[static_cast<std::size_t>(i)];
                abs_o += std::fabs(d) * h_max[static_cast<std::size_t>(i)];
            }
            const double delta = spread + (2.0 * F + 8) * kU32 * abs_o;
            const double o_ref = bf16_of(o64);
            const double lo = bf16_of(o64 - delta), hi = bf16_of(o64 + delta);
            const double dev_o = std::max(std::fabs(lo - o_ref), std::fabs(hi - o_ref));
            const auto h       = static_cast<std::size_t>(row);
            out.s[h] += w * o_ref;
            out.limit[h] += std::fabs(w) * dev_o;
            abs_s[h] += std::fabs(w) * std::max({std::fabs(lo), std::fabs(hi), std::fabs(o_ref)});
        }
    }
    for (int h = 0; h < H; ++h) {
        out.limit[static_cast<std::size_t>(h)] +=
            (top_k + 2) * kU32 * abs_s[static_cast<std::size_t>(h)] + 1e-30;
    }
    return out;
}

struct RunResult {
    std::vector<float> store;       // Store partial [T][H]
    std::vector<float> accumulated; // Accumulate from init
    std::vector<float> init;
    std::vector<std::uint16_t> rounded; // moe_partial_to_bf16(accumulated)
};

RunResult run_op(const Bank& b, const Routing& r, int tokens, int top_k, bool graph) {
    const auto& c           = b.config;
    const int H             = c.hidden;
    const std::size_t count = static_cast<std::size_t>(tokens) * H;
    std::vector<std::uint16_t> x_bits(r.x.size());
    for (std::size_t i = 0; i < x_bits.size(); ++i) { x_bits[i] = f32_to_bf16(r.x[i]); }
    GuardedDeviceBuffer dx(x_bits.size() * 2), dids(r.ids.size() * 4), dw(r.weights.size() * 4),
        dstore(count * 4), dacc(count * 4), dy(count * 2);
    dx.copy_from_host(x_bits.data(), dx.bytes());
    dids.copy_from_host(r.ids.data(), dids.bytes());
    dw.copy_from_host(r.weights.data(), dw.bytes());
    dstore.fill(0x7f);
    RunResult out;
    out.init.resize(count);
    fill_uniform(out.init, 4242U + static_cast<std::uint32_t>(tokens), -3.0F, 3.0F);
    dacc.copy_from_host(out.init.data(), dacc.bytes());
    Tensor tx(dx.data(), DType::BF16, {H, tokens});
    Tensor tids(dids.data(), DType::I32, {top_k, tokens});
    Tensor tw(dw.data(), DType::FP32, {top_k, tokens});
    Tensor tstore(dstore.data(), DType::FP32, {H, tokens});
    Tensor tacc(dacc.data(), DType::FP32, {H, tokens});
    Tensor ty(dy.data(), DType::BF16, {H, tokens});
    WorkspaceArena work(ops::moe_bank_capped_swiglu_workspace_bytes(top_k, c.intermediate, tokens));
    const ops::MoeBank bank{b.gate_up_view, b.down_view, c.first_expert};
    const ops::CappedSwiglu act{4.0F, 25.0F};
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const auto issue = [&] {
        ops::moe_bank_capped_swiglu(tx, tids, tw, bank, act, ops::MoeBankOutput::Store, tstore,
                                    work, stream);
        ops::moe_bank_capped_swiglu(tx, tids, tw, bank, act, ops::MoeBankOutput::Accumulate, tacc,
                                    work, stream);
        ops::moe_partial_to_bf16(tacc, ty, stream);
    };
    if (graph) {
        cudaGraph_t captured;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        issue();
        CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
        CUDA_CHECK(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(captured));
    } else {
        issue();
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    out.store        = from_device<float>(dstore.data(), count);
    out.accumulated  = from_device<float>(dacc.data(), count);
    out.rounded      = from_device<std::uint16_t>(dy.data(), count);
    const int guards = dx.verify_guards("x") + dids.verify_guards("ids") + dw.verify_guards("w") +
                       dstore.verify_guards("store") + dacc.verify_guards("acc") +
                       dy.verify_guards("y") + b.gate_up->device->verify_guards("gate_up") +
                       b.down->device->verify_guards("down");
    if (guards != 0) { throw std::runtime_error(c.label + ": guard overwritten"); }
    return out;
}

int check(const Bank& b, int tokens, int top_k, int checked_tokens, bool repeat, bool graph,
          std::uint32_t seed) {
    const auto& c           = b.config;
    const std::string label = c.label + " T=" + std::to_string(tokens) + (graph ? " graph" : "");
    const Routing r         = make_routing(c, tokens, top_k, seed, repeat);
    const RunResult run     = run_op(b, r, tokens, top_k, graph);
    const int H             = c.hidden;
    int failures            = 0;
    // Checked tokens: all, or the special ones, the last, and an even spread.
    std::vector<int> check_list;
    if (checked_tokens <= 0 || checked_tokens >= tokens) {
        for (int t = 0; t < tokens; ++t) { check_list.push_back(t); }
    } else {
        for (int t = 0; t < std::min(tokens, 4); ++t) { check_list.push_back(t); }
        for (int i = 1; i < checked_tokens - 4; ++i) {
            check_list.push_back(4 + (tokens - 5) * i / (checked_tokens - 4));
        }
        check_list.push_back(tokens - 1);
    }
    double worst = 0, worst_ratio = 0, rms_sum = 0;
    std::size_t rms_count = 0;
    int tokens_in_bank    = 0;
    for (const int t : check_list) {
        const OracleToken o = oracle_token(b, r, t, top_k, Activation{});
        tokens_in_bank += o.any ? 1 : 0;
        for (int h = 0; h < H; ++h) {
            const std::size_t i = static_cast<std::size_t>(t) * H + h;
            const double got    = run.store[i];
            const double error  = std::fabs(got - o.s[static_cast<std::size_t>(h)]);
            const double ratio  = error / o.limit[static_cast<std::size_t>(h)];
            if (!o.any && std::signbit(got)) { ++failures; }
            if (!(ratio <= 1.0)) {
                if (failures < 5) {
                    std::cerr << label << ": token " << t << " row " << h << " got " << got
                              << " want " << o.s[static_cast<std::size_t>(h)] << " limit "
                              << o.limit[static_cast<std::size_t>(h)] << '\n';
                }
                ++failures;
            }
            worst       = std::max(worst, error);
            worst_ratio = std::max(worst_ratio, ratio);
            rms_sum += o.s[static_cast<std::size_t>(h)] * o.s[static_cast<std::size_t>(h)];
            ++rms_count;
        }
    }
    // Accumulate adds exactly the Store value; the rounding is exact RNE.
    std::size_t acc_mismatch = 0, round_mismatch = 0;
    for (std::size_t i = 0; i < run.store.size(); ++i) {
        const float want = run.init[i] + run.store[i];
        acc_mismatch += std::memcmp(&want, &run.accumulated[i], 4) != 0;
        round_mismatch += f32_to_bf16(run.accumulated[i]) != run.rounded[i];
    }
    if (acc_mismatch != 0 || round_mismatch != 0) {
        std::cerr << label << ": " << acc_mismatch << " Accumulate and " << round_mismatch
                  << " rounding mismatches\n";
        ++failures;
    }
    const double rms =
        std::sqrt(rms_sum / static_cast<double>(std::max<std::size_t>(rms_count, 1)));
    std::cout << "CASE " << label << " checked_tokens=" << check_list.size()
              << " in_bank=" << tokens_in_bank << " max_abs_err=" << worst << " rms_ref=" << rms
              << " max_err/rms=" << (rms > 0 ? worst / rms : 0.0)
              << " max_err/limit=" << worst_ratio << (failures ? " FAIL" : " ok") << '\n';
    return failures;
}

// tp 1 == tp 2: two banks of one parent; Store(bank 0) + Accumulate(bank 1) into one partial
// equals fl32(Store(bank 0) + Store(bank 1)), the allreduce_sum_f32 sum, bit for bit.
int check_two_banks(bool nvfp4, int hidden, int intermediate, int experts, int tokens,
                    std::uint32_t seed) {
    const int half = experts / 2, top_k = 8;
    BankConfig c0{"two banks", nvfp4, hidden, intermediate, experts, 0, half, 0, experts, 1};
    Bank b0 = make_bank(c0, seed);
    Bank b1;
    b1.config              = c0;
    b1.config.view_first   = half;
    b1.config.first_expert = half;
    b1.gate_up_view = rows_of(*b0.gate_up, static_cast<std::int64_t>(half) * 2 * intermediate,
                              static_cast<std::int64_t>(half) * 2 * intermediate);
    b1.down_view    = rows_of(*b0.down, static_cast<std::int64_t>(half) * hidden,
                              static_cast<std::int64_t>(half) * hidden);
    const Routing r = make_routing(c0, tokens, top_k, seed + 5, false);
    const std::size_t count = static_cast<std::size_t>(tokens) * hidden;
    std::vector<std::uint16_t> x_bits(r.x.size());
    for (std::size_t i = 0; i < x_bits.size(); ++i) { x_bits[i] = f32_to_bf16(r.x[i]); }
    GuardedDeviceBuffer dx(x_bits.size() * 2), dids(r.ids.size() * 4), dw(r.weights.size() * 4),
        dp0(count * 4), dp1(count * 4), done(count * 4);
    dx.copy_from_host(x_bits.data(), dx.bytes());
    dids.copy_from_host(r.ids.data(), dids.bytes());
    dw.copy_from_host(r.weights.data(), dw.bytes());
    Tensor tx(dx.data(), DType::BF16, {hidden, tokens});
    Tensor tids(dids.data(), DType::I32, {top_k, tokens});
    Tensor tw(dw.data(), DType::FP32, {top_k, tokens});
    Tensor tp0(dp0.data(), DType::FP32, {hidden, tokens});
    Tensor tp1(dp1.data(), DType::FP32, {hidden, tokens});
    Tensor tone(done.data(), DType::FP32, {hidden, tokens});
    WorkspaceArena work(ops::moe_bank_capped_swiglu_workspace_bytes(top_k, intermediate, tokens));
    const ops::MoeBank bank0{b0.gate_up_view, b0.down_view, 0};
    const ops::MoeBank bank1{b1.gate_up_view, b1.down_view, half};
    const ops::CappedSwiglu act{4.0F, 25.0F};
    ops::moe_bank_capped_swiglu(tx, tids, tw, bank0, act, ops::MoeBankOutput::Store, tp0, work,
                                nullptr);
    ops::moe_bank_capped_swiglu(tx, tids, tw, bank1, act, ops::MoeBankOutput::Store, tp1, work,
                                nullptr);
    ops::moe_bank_capped_swiglu(tx, tids, tw, bank0, act, ops::MoeBankOutput::Store, tone, work,
                                nullptr);
    ops::moe_bank_capped_swiglu(tx, tids, tw, bank1, act, ops::MoeBankOutput::Accumulate, tone,
                                work, nullptr);
    cuda_synchronize();
    const auto p0          = from_device<float>(dp0.data(), count);
    const auto p1          = from_device<float>(dp1.data(), count);
    const auto one         = from_device<float>(done.data(), count);
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const float sum = p0[i] + p1[i];
        mismatches += std::memcmp(&sum, &one[i], 4) != 0;
    }
    std::cout << "CASE two banks " << (nvfp4 ? "nvfp4" : "bf16") << " H=" << hidden
              << " T=" << tokens << " tp1_vs_tp2_mismatches=" << mismatches
              << (mismatches ? " FAIL" : " ok") << '\n';
    return mismatches != 0 ? 1 : 0;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    try {
        // Synthetic geometry: H 512, F 256, 16 experts, banks of 8.
        {
            Bank b = make_bank({"syn nvfp4 bank0", true, 512, 256, 8, 0, 8, 0, 16, 1}, 11);
            for (const int t : {1, 4, 16, 47, 48, 512}) {
                failures += check(b, t, 8, 0, false, false, 1000U + static_cast<std::uint32_t>(t));
            }
            failures += check(b, 4, 8, 0, false, true, 1100U);
        }
        {
            // Experts 8..15 of a 16-expert parent, gate and up quantized apart (two divisors).
            Bank b = make_bank({"syn nvfp4 bank1 view", true, 512, 256, 16, 8, 8, 8, 16, 2}, 12);
            failures += check(b, 16, 8, 0, true, false, 1200U);
        }
        {
            Bank b = make_bank({"syn bf16 bank1", false, 512, 256, 8, 0, 8, 8, 16, 1}, 13);
            failures += check(b, 16, 8, 0, true, false, 1300U);
            failures += check(b, 512, 8, 0, false, false, 1301U);
        }
        failures += check_two_banks(true, 512, 256, 16, 16, 14);
        failures += check_two_banks(false, 512, 256, 16, 16, 15);
        // Real geometry: H 2304, F 1152, 128 experts, top-8.
        {
            Bank b =
                make_bank({"real nvfp4 E'=64 off 0", true, 2304, 1152, 64, 0, 64, 0, 128, 1}, 21);
            failures += check(b, 1, 8, 0, false, false, 2001U);
            failures += check(b, 16, 8, 0, false, false, 2016U);
        }
        {
            Bank b =
                make_bank({"real nvfp4 E'=64 off 64", true, 2304, 1152, 64, 0, 64, 64, 128, 1}, 22);
            failures += check(b, 4, 8, 0, false, false, 2104U);
            failures += check(b, 512, 8, 12, false, false, 2112U);
        }
        {
            Bank b = make_bank({"real nvfp4 E'=128", true, 2304, 1152, 128, 0, 128, 0, 128, 1}, 23);
            failures += check(b, 16, 8, 0, false, false, 2216U);
        }
        {
            Bank b =
                make_bank({"real bf16 E'=64 off 64", false, 2304, 1152, 64, 0, 64, 64, 128, 1}, 24);
            failures += check(b, 16, 8, 0, false, false, 2316U);
            failures += check(b, 512, 8, 8, false, false, 2312U);
        }
        {
            Bank b = make_bank({"syn mxfp8 bank1", false, 512, 256, 8, 0, 8, 8, 16, 1, true}, 25);
            failures += check(b, 16, 8, 0, true, false, 2401U);
            failures += check(b, 512, 8, 0, false, false, 2402U);
        }
        {
            Bank b = make_bank(
                {"real mxfp8 E'=64 off 64", false, 2304, 1152, 64, 0, 64, 64, 128, 1, true}, 26);
            failures += check(b, 1, 8, 0, false, false, 2501U);
            failures += check(b, 512, 8, 8, false, false, 2502U);
        }
    } catch (const std::exception& error) {
        std::cerr << "moe_bank test error: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << "moe_bank: " << failures << " failures\n";
        return 1;
    }
    std::cout << "OK moe_bank correctness\n";
    return 0;
}
