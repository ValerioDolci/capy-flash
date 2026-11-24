// Measures moe_bank_capped_swiglu (one bank of stacked experts, nvfp4 W4A16, mxfp8 or bf16) and
// moe_sigmoid_route at given geometry (default H 2304, F 1152, a bank of 64 of 128 experts,
// top-8). Each call routes every token to 8 distinct random experts of 128, so on average 4 fall
// in the bank. One CUDA graph holds --calls calls cycling over --banks separately allocated banks
// (the layers), each call with its own routing, so consecutive calls read different experts.
// Reported per call: median and minimum over --repeat graph launches; U = distinct experts of the
// bank read per call (mean); bytes = U * (bytes of one expert's codes and scales); GB/s = bytes /
// median; floor_us = bytes / --peak-gbs. Router: the same per call, over --banks routers.
// One JSON object per line.
#include "ninfer/ops/moe_bank.h"
#include "ninfer/ops/moe_sigmoid_route.h"

#include "core/device.h"
#include "core/weight_view.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;

namespace {

struct Options {
    std::vector<std::string> formats{"nvfp4", "mxfp8", "bf16"};
    std::vector<int> tokens{1, 4, 8, 512};
    int hidden = 2304, intermediate = 1152, bank_experts = 64, experts = 128, top_k = 8;
    int nvfp4_banks = 3, mxfp8_banks = 2, bf16_banks = 2, calls = 24, wide_calls = 4, warmup = 3,
        repeat      = 20;
    int pool        = 0; // experts a call's tokens choose from (0: all of them)
    double peak_gbs = 896.0;
    bool route      = true;
};

std::vector<std::string> split(const std::string& text) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (start < text.size()) {
        const auto end = text.find(',', start);
        out.push_back(text.substr(start, end - start));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return out;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::puts(
                "usage: ninfer_moe_bank_bench [--format nvfp4,mxfp8,bf16] [--tokens 1,4,8,512] "
                "[--hidden 2304] [--intermediate 1152] [--bank-experts 64] [--experts 128] "
                "[--top-k 8] [--nvfp4-banks 3] [--mxfp8-banks 2] [--bf16-banks 2] [--calls 24] "
                "[--wide-calls 4] [--warmup 3] [--repeat 20] [--peak-gbs 896] [--pool N] "
                "[--no-route]");
            std::exit(0);
        }
        if (arg == "--no-route") {
            o.route = false;
            continue;
        }
        if (++i >= argc) throw std::invalid_argument("missing option value");
        const std::string v = argv[i];
        if (arg == "--format") {
            o.formats = split(v);
        } else if (arg == "--tokens") {
            o.tokens.clear();
            for (const auto& s : split(v)) o.tokens.push_back(std::stoi(s));
        } else if (arg == "--hidden") {
            o.hidden = std::stoi(v);
        } else if (arg == "--intermediate") {
            o.intermediate = std::stoi(v);
        } else if (arg == "--bank-experts") {
            o.bank_experts = std::stoi(v);
        } else if (arg == "--experts") {
            o.experts = std::stoi(v);
        } else if (arg == "--top-k") {
            o.top_k = std::stoi(v);
        } else if (arg == "--mxfp8-banks") {
            o.mxfp8_banks = std::stoi(v);
        } else if (arg == "--nvfp4-banks") {
            o.nvfp4_banks = std::stoi(v);
        } else if (arg == "--bf16-banks") {
            o.bf16_banks = std::stoi(v);
        } else if (arg == "--calls") {
            o.calls = std::stoi(v);
        } else if (arg == "--wide-calls") {
            o.wide_calls = std::stoi(v);
        } else if (arg == "--warmup") {
            o.warmup = std::stoi(v);
        } else if (arg == "--repeat") {
            o.repeat = std::stoi(v);
        } else if (arg == "--pool") {
            o.pool = std::stoi(v);
        } else if (arg == "--peak-gbs") {
            o.peak_gbs = std::stod(v);
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    return o;
}

// One stacked parent: random code bytes, scale words 0x38 (1.0), divisors 1024.
struct Parent {
    DeviceBuffer data;
    WeightParent parent;
};

// format: "nvfp4" (scale words 1.0, divisors 1024), "mxfp8" (random codes, exponents 2^-7) or
// "bf16".
std::unique_ptr<Parent> make_parent(const std::string& format, std::uint64_t rows,
                                    std::uint64_t cols, std::uint64_t divisors,
                                    std::uint64_t seed) {
    auto p           = std::make_unique<Parent>();
    const bool nvfp4 = format == "nvfp4", mxfp8 = format == "mxfp8";
    const std::array<std::uint64_t, 2> shape{rows, cols};
    p->parent.geometry =
        mxfp8 ? weight_geometry(QType::MXFP8_E4M3FN_G32_UE8M0, QuantLayout::BlockScaleK32, shape)
              : weight_geometry(nvfp4 ? QType::NVFP4 : QType::BF16,
                                nvfp4 ? QuantLayout::BlockScaleK16M128x4 : QuantLayout::Contiguous,
                                shape, nvfp4 ? divisors : 1);
    const auto& g = p->parent.geometry;
    if (mxfp8) {
        p->data = bench::make_bf16((g.bytes + 1) / 2, seed, -0.5F, 0.5F);
        CUDA_CHECK(
            cudaMemset(static_cast<std::uint8_t*>(p->data.p) + g.scale_offset, 120, g.scale_bytes));
    } else if (nvfp4) {
        p->data = bench::make_bf16((g.bytes + 1) / 2, seed, -3.0F, 3.0F);
        CUDA_CHECK(cudaMemset(static_cast<std::uint8_t*>(p->data.p) + g.scale_offset, 0x38,
                              g.scale_bytes));
        const std::vector<float> words(divisors, 1024.0F);
        CUDA_CHECK(cudaMemcpy(static_cast<std::uint8_t*>(p->data.p) + g.divisor_offset,
                              words.data(), divisors * 4, cudaMemcpyHostToDevice));
    } else {
        p->data = bench::make_bf16(rows * cols, seed, -0.06F, 0.06F);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    p->parent.data = static_cast<const std::byte*>(p->data.p);
    return p;
}

WeightView whole(const Parent& p) {
    WeightView v;
    v.shape = p.parent.geometry.shape;
    v.parts = {{&p.parent, 0, p.parent.geometry.elements}};
    return v;
}

struct Bank {
    std::unique_ptr<Parent> gate_up, down;
    WeightView gate_up_view, down_view;
};

// Routing of one call: 8 distinct experts of `experts` per token, random weights. With --pool N
// the call's tokens choose among the same N random experts (8: every token the same experts), as
// the tokens of a verification round share experts.
std::vector<int> make_ids(const Options& o, int tokens, std::mt19937& rng, int& distinct) {
    std::vector<int> ids(static_cast<std::size_t>(tokens) * o.top_k);
    std::vector<int> all(static_cast<std::size_t>(o.experts));
    std::iota(all.begin(), all.end(), 0);
    const int pool = o.pool >= o.top_k && o.pool < o.experts ? o.pool : o.experts;
    if (pool < o.experts) std::shuffle(all.begin(), all.end(), rng);
    std::set<int> in_bank;
    for (int t = 0; t < tokens; ++t) {
        std::shuffle(all.begin(), all.begin() + pool, rng);
        for (int j = 0; j < o.top_k; ++j) {
            ids[static_cast<std::size_t>(t) * o.top_k + j] = all[static_cast<std::size_t>(j)];
            if (all[static_cast<std::size_t>(j)] < o.bank_experts) in_bank.insert(all[j]);
        }
    }
    distinct = static_cast<int>(in_bank.size());
    return ids;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const int H = o.hidden, F = o.intermediate, K = o.top_k;
        const int max_t = *std::max_element(o.tokens.begin(), o.tokens.end());
        DeviceBuffer x  = bench::make_bf16(static_cast<std::size_t>(max_t) * H, 43U, -1.7F, 1.7F);
        DeviceBuffer w  = bench::make_f32(static_cast<std::size_t>(max_t) * K, 44U, 0.5F, 1.5F);
        DeviceBuffer out(static_cast<std::size_t>(max_t) * H * sizeof(float));
        WorkspaceArena workspace(ops::moe_bank_capped_swiglu_workspace_bytes(K, F, H, max_t));
        std::printf("{\"device\":\"%s\",\"sms\":%d,\"H\":%d,\"F\":%d,\"bank_experts\":%d,"
                    "\"experts\":%d,\"top_k\":%d,\"calls\":%d,\"wide_calls\":%d,\"repeat\":%d,"
                    "\"peak_gbs\":%.0f}\n",
                    prop.name, prop.multiProcessorCount, H, F, o.bank_experts, o.experts, K,
                    o.calls, o.wide_calls, o.repeat, o.peak_gbs);

        for (const auto& format : o.formats) {
            const bool nvfp4 = format == "nvfp4", mxfp8 = format == "mxfp8";
            if (!nvfp4 && !mxfp8 && format != "bf16") throw std::invalid_argument("bad --format");
            const int banks = nvfp4 ? o.nvfp4_banks : mxfp8 ? o.mxfp8_banks : o.bf16_banks;
            const double expert_bytes = (2.0 * F * H + 1.0 * H * F) * (nvfp4   ? 0.5 + 1.0 / 16
                                                                       : mxfp8 ? 1.0 + 1.0 / 32
                                                                               : 2.0);
            std::vector<Bank> bank(static_cast<std::size_t>(banks));
            for (int b = 0; b < banks; ++b) {
                const auto E         = static_cast<std::uint64_t>(o.bank_experts);
                bank[b].gate_up      = make_parent(format, E * 2 * F, H, E, 100U + 2 * b);
                bank[b].down         = make_parent(format, E * H, F, E, 101U + 2 * b);
                bank[b].gate_up_view = whole(*bank[b].gate_up);
                bank[b].down_view    = whole(*bank[b].down);
            }
            for (const int T : o.tokens) {
                const int calls = T > 8 ? o.wide_calls : o.calls;
                // The routing of a point depends only on T: points compare across runs and
                // token lists.
                std::mt19937 rng(7U + static_cast<unsigned>(T));
                std::vector<DeviceBuffer> ids;
                double distinct = 0;
                for (int c = 0; c < calls; ++c) {
                    int u        = 0;
                    const auto h = make_ids(o, T, rng, u);
                    ids.emplace_back(h.size() * sizeof(int));
                    ids.back().copy_from_host(h.data(), h.size() * sizeof(int));
                    distinct += u;
                }
                distinct /= calls;
                bench::TimedGraph graph;
                graph.capture(stream, [&](cudaStream_t s) {
                    for (int c = 0; c < calls; ++c) {
                        const Bank& b = bank[static_cast<std::size_t>(c % banks)];
                        Tensor tx(x.p, DType::BF16, {H, T});
                        Tensor ti(ids[c].p, DType::I32, {K, T});
                        Tensor tw(w.p, DType::FP32, {K, T});
                        Tensor to(out.p, DType::FP32, {H, T});
                        const ops::MoeBank mb{b.gate_up_view, b.down_view, 0};
                        ops::moe_bank_capped_swiglu(tx, ti, tw, mb, {4.0F, 25.0F},
                                                    ops::MoeBankOutput::Store, to, workspace, s);
                    }
                });
                const bench::ColdTiming t = bench::measure_graph(graph, stream, o.warmup, o.repeat);
                const double median = t.median_us / calls, best = t.min_us / calls;
                const double bytes = distinct * expert_bytes;
                std::printf("{\"op\":\"bank\",\"format\":\"%s\",\"T\":%d,\"U\":%.2f,\"MB\":%.2f,"
                            "\"us_median\":%.2f,\"us_min\":%.2f,\"GBps\":%.1f,\"floor_us\":%.2f}\n",
                            format.c_str(), T, distinct, bytes / 1e6, median, best,
                            bytes / median / 1e3, bytes / o.peak_gbs / 1e3);
                std::fflush(stdout);
            }
        }

        if (o.route) {
            const int routers = std::max(o.nvfp4_banks, 1) * 8;
            std::vector<DeviceBuffer> router, bias;
            for (int r = 0; r < routers; ++r) {
                router.push_back(bench::make_f32(static_cast<std::size_t>(o.experts) * H, 200U + r,
                                                 -0.07F, 0.07F));
                bias.push_back(
                    bench::make_f32(static_cast<std::size_t>(o.experts), 300U + r, -0.5F, 0.5F));
            }
            DeviceBuffer rid(static_cast<std::size_t>(max_t) * K * 4);
            DeviceBuffer rw(static_cast<std::size_t>(max_t) * K * 4);
            for (const int T : o.tokens) {
                const int calls = T > 8 ? o.wide_calls : o.calls;
                bench::TimedGraph graph;
                graph.capture(stream, [&](cudaStream_t s) {
                    for (int c = 0; c < calls; ++c) {
                        Tensor tx(x.p, DType::BF16, {H, T});
                        Tensor tr(router[static_cast<std::size_t>(c % routers)].p, DType::FP32,
                                  {H, o.experts});
                        Tensor tb(bias[static_cast<std::size_t>(c % routers)].p, DType::FP32,
                                  {o.experts});
                        Tensor ti(rid.p, DType::I32, {K, T});
                        Tensor tw(rw.p, DType::FP32, {K, T});
                        ops::moe_sigmoid_route(tx, tr, tb, K, 8.0F, ti, tw, s);
                    }
                });
                const bench::ColdTiming t = bench::measure_graph(graph, stream, o.warmup, o.repeat);
                const double bytes        = static_cast<double>(o.experts) * H * 4;
                std::printf("{\"op\":\"route\",\"T\":%d,\"MB\":%.2f,\"us_median\":%.2f,"
                            "\"us_min\":%.2f,\"floor_us\":%.2f}\n",
                            T, bytes / 1e6, t.median_us / calls, t.min_us / calls,
                            bytes / o.peak_gbs / 1e3);
                std::fflush(stdout);
            }
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ninfer_moe_bank_bench: %s\n", e.what());
        return 1;
    }
}
