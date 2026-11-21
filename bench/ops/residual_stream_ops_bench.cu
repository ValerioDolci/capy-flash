// The per-token residual-stream Ops at the model geometry (H = 2304; K/V channels 512 per rank,
// 4 groups): depth_attention, short_conv_static/dynamic, rmsnorm_f32, residual_add_f32 and
// residual_add_round_bf16. One captured graph issues `--calls` back-to-back calls of one Op on
// L2-resident operands (as in a decode step, where they are activations just written); the time
// per call is the graph time divided by the calls, inter-kernel gaps included.
#include "ninfer/ops/depth_attention.h"
#include "ninfer/ops/residual_f32.h"
#include "ninfer/ops/rmsnorm_f32.h"
#include "ninfer/ops/short_conv.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

void report(const std::string& tag, int calls, double bytes,
            const std::function<void(cudaStream_t)>& one_call) {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    TimedGraph graph;
    graph.capture(stream, [&](cudaStream_t s) {
        for (int c = 0; c < calls; ++c) { one_call(s); }
    });
    const ColdTiming t = measure_graph(graph, stream, 10, 50);
    CUDA_CHECK(cudaStreamDestroy(stream));
    const double per_call = t.median_us / calls;
    std::printf("%-44s per call median=%8.2f us  min=%8.2f us  %7.1f GB/s\n", tag.c_str(), per_call,
                t.min_us / calls, bytes / per_call / 1.0e3);
}

void run(int tokens, int calls) {
    constexpr int H  = 2304;
    constexpr int C  = 512; // K/V channels of one rank
    constexpr int G  = 4;
    const auto col   = static_cast<std::size_t>(H) * tokens;
    const auto T     = std::to_string(tokens);
    const auto bf16s = [](std::size_t n) { return static_cast<double>(n) * 2.0; };
    const auto f32s  = [](std::size_t n) { return static_cast<double>(n) * 4.0; };

    std::vector<DeviceBuffer> state_buffers;
    std::vector<Tensor> states;
    for (int s = 0; s < 5; ++s) {
        state_buffers.push_back(make_bf16(col, 31U + s, -2.0F, 2.0F));
        states.emplace_back(state_buffers.back().p, DType::BF16,
                            std::initializer_list<int>{H, tokens});
    }
    DeviceBuffer current_f32  = make_f32(col, 41U, -2.0F, 2.0F);
    DeviceBuffer current_bf16 = make_bf16(col, 43U, -2.0F, 2.0F);
    DeviceBuffer query        = make_f32(static_cast<std::size_t>(H) * 4, 47U, -1.0F, 1.0F);
    DeviceBuffer da_out(col * 4 * sizeof(float));
    for (const int q : {1, 3, 4}) {
        for (const int n : {1, 4}) {
            const bool f32 = q == 1;
            const Tensor cur(f32 ? current_f32.p : current_bf16.p, f32 ? DType::FP32 : DType::BF16,
                             {H, tokens});
            Tensor out(da_out.p, DType::FP32, {H, tokens, q});
            const std::span<const Tensor> used(states.data(), static_cast<std::size_t>(n));
            const double bytes = n * bf16s(col) + (f32 ? f32s(col) : bf16s(col)) + q * f32s(col);
            report("depth_attention n=" + std::to_string(n) + " Q=" + std::to_string(q) +
                       (f32 ? " cur FP32" : " cur BF16") + " T=" + T,
                   calls, bytes, [&, cur, out, used](cudaStream_t s) mutable {
                       ops::depth_attention(used, cur, Tensor(query.p, DType::FP32, {H, q}),
                                            1.0e-6F, 1.0F / std::sqrt(float(H)), out, s);
                   });
        }
    }

    // Static convolution at the hidden width (post-attention / post-MoE) and the K/V width of a
    // rank; dynamic convolution at the K/V width of a rank.
    for (const int c : {H, C}) {
        const auto n    = static_cast<std::size_t>(c) * tokens;
        DeviceBuffer x  = make_bf16(n, 51U, -2.0F, 2.0F);
        DeviceBuffer w  = make_f32(static_cast<std::size_t>(c) * 4, 53U, -1.0F, 1.0F);
        DeviceBuffer st = make_bf16(static_cast<std::size_t>(c) * 3, 55U, -2.0F, 2.0F);
        DeviceBuffer out(n * 2);
        const Tensor tx(x.p, DType::BF16, {c, tokens});
        const Tensor tw(w.p, DType::FP32, {4, c});
        Tensor ts(st.p, DType::BF16, {c, 3});
        Tensor to(out.p, DType::BF16, {c, tokens});
        report("short_conv_static C=" + std::to_string(c) + " T=" + T, calls,
               bf16s(n) * 2.0 + f32s(c * 4U),
               [&](cudaStream_t s) { ops::short_conv_static(tx, tw, 0.4472136F, ts, ts, to, s); });
        if (c == C) {
            DeviceBuffer dyn =
                make_bf16(static_cast<std::size_t>(4 * G) * tokens, 57U, -1.0F, 1.0F);
            const Tensor tw2(w.p, DType::FP32, {c, 4});
            const Tensor td(dyn.p, DType::BF16, {4 * G, tokens});
            report("short_conv_dynamic C=" + std::to_string(c) + " G=4 T=" + T, calls,
                   bf16s(n) * 2.0 + f32s(c * 4U), [&](cudaStream_t s) {
                       ops::short_conv_dynamic(tx, tw2, td, 1.0F / 3.0F, ts, ts, to, s);
                   });
        }
    }

    DeviceBuffer gamma = make_bf16(H, 61U, 0.5F, 1.5F);
    DeviceBuffer norm_out(col * 2);
    report("rmsnorm_f32 D=2304 T=" + T, calls, f32s(col) + bf16s(col), [&](cudaStream_t s) {
        Tensor out(norm_out.p, DType::BF16, {H, tokens});
        ops::rmsnorm_f32(Tensor(current_f32.p, DType::FP32, {H, tokens}),
                         Tensor(gamma.p, DType::BF16, {H}), 1.0e-6F, out, s);
    });

    DeviceBuffer sum(col * 4);
    DeviceBuffer round_out(col * 2);
    report("residual_add_f32 H=2304 T=" + T, calls, 2.0 * bf16s(col) + f32s(col),
           [&](cudaStream_t s) {
               Tensor ts(sum.p, DType::FP32, {H, tokens});
               ops::residual_add_f32(&states[0], states[1], ts, s);
           });
    report("residual_add_round_bf16 H=2304 T=" + T, calls, 2.0 * bf16s(col) + f32s(col),
           [&](cudaStream_t s) {
               Tensor to(round_out.p, DType::BF16, {H, tokens});
               ops::residual_add_round_bf16(Tensor(sum.p, DType::FP32, {H, tokens}), states[2], to,
                                            s);
           });
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    int calls = 20;
    std::vector<int> tokens;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--calls") && i + 1 < argc) {
            calls = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) {
            tokens.push_back(std::atoi(argv[++i]));
        } else {
            std::fprintf(stderr, "usage: %s [--calls N] [--tokens T]...\n", argv[0]);
            return 2;
        }
    }
    if (calls <= 0) {
        std::fprintf(stderr, "calls must be positive\n");
        return 2;
    }
    if (tokens.empty()) { tokens = {1, 256}; }
    for (const int t : tokens) {
        if (t <= 0) {
            std::fprintf(stderr, "tokens must be positive\n");
            return 2;
        }
        run(t, calls);
    }
    return 0;
}
