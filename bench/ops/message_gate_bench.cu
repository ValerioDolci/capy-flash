// message_gate at the model geometry (H = 2304, R = 64). One captured graph issues `--calls`
// gates back to back, each with its own weight set: with the default 32 sets the 57 MB of weights
// exceed L2, as in a decode step where every gate of every visit has distinct weights. `--hot`
// reuses one weight set (L2 resident); `--shift` places hidden, message and out 4 bytes past a
// 16-byte boundary (a valid contiguous placement that the Op serves with another route). The time
// per call is the graph time divided by the calls. `--parallel B` spreads the calls over B parallel
// graph branches (independent gates, such as the q, k and v gates of one visit).
#include "ninfer/ops/message_gate.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

struct WeightSet {
    DeviceBuffer down, forget, gain;
};

void run(int hidden, int rank, int tokens, int calls, bool hot, int shift, int parallel) {
    const std::size_t columns = static_cast<std::size_t>(hidden) * tokens;
    DeviceBuffer h            = make_f32(columns + 1, 11U, -2.0F, 2.0F);
    DeviceBuffer m            = make_f32(columns + 1, 13U, -3.0F, 3.0F);
    std::vector<DeviceBuffer> outs;
    for (int b = 0; b < parallel; ++b) { outs.emplace_back((columns + 1) * sizeof(float)); }
    std::vector<WeightSet> sets(hot ? 1 : static_cast<std::size_t>(calls));
    for (std::size_t s = 0; s < sets.size(); ++s) {
        sets[s].down =
            make_bf16(static_cast<std::size_t>(2) * hidden * rank, 17U + s, -0.03F, 0.03F);
        sets[s].forget = make_f32(static_cast<std::size_t>(hidden) * rank, 19U + s, -0.2F, 0.2F);
        sets[s].gain   = make_f32(static_cast<std::size_t>(hidden) * rank, 23U + s, -0.2F, 0.2F);
    }
    const Tensor th(static_cast<float*>(h.p) + shift, DType::FP32, {hidden, tokens});
    const Tensor tm(static_cast<float*>(m.p) + shift, DType::FP32, {hidden, tokens});

    // Branch b issues calls c = b, b + parallel, ... on its own stream, forked from and joined to
    // the capture stream: independent gates (q, k, v of one visit) as parallel graph branches.
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<cudaStream_t> branches(static_cast<std::size_t>(parallel));
    std::vector<cudaEvent_t> joined(static_cast<std::size_t>(parallel));
    cudaEvent_t fork = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
    for (int b = 0; b < parallel; ++b) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&branches[b], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&joined[b], cudaEventDisableTiming));
    }
    TimedGraph graph;
    graph.capture(stream, [&](cudaStream_t s) {
        CUDA_CHECK(cudaEventRecord(fork, s));
        for (int b = 0; b < parallel; ++b) {
            CUDA_CHECK(cudaStreamWaitEvent(branches[b], fork, 0));
            Tensor to(static_cast<float*>(outs[b].p) + shift, DType::FP32, {hidden, tokens});
            for (int c = b; c < calls; c += parallel) {
                const WeightSet& w = sets[hot ? 0 : static_cast<std::size_t>(c)];
                ops::message_gate(th, tm, Tensor(w.down.p, DType::BF16, {2 * hidden, rank}),
                                  Tensor(w.forget.p, DType::FP32, {rank, hidden}),
                                  Tensor(w.gain.p, DType::FP32, {rank, hidden}), 1.0e-6F, to,
                                  branches[b]);
            }
            CUDA_CHECK(cudaEventRecord(joined[b], branches[b]));
            CUDA_CHECK(cudaStreamWaitEvent(s, joined[b], 0));
        }
    });
    L2FlushBuffer flush(std::size_t{128} << 20);
    const ColdTiming t = measure_cold_graph(graph, flush, stream, 5, 30);
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (int b = 0; b < parallel; ++b) {
        CUDA_CHECK(cudaStreamDestroy(branches[b]));
        CUDA_CHECK(cudaEventDestroy(joined[b]));
    }
    CUDA_CHECK(cudaEventDestroy(fork));

    const double weight_bytes = static_cast<double>(rank) * hidden * (2 * 2 + 2 * 4);
    const double bytes        = weight_bytes + 3.0 * static_cast<double>(columns) * 4.0;
    const double per_call     = t.median_us / calls;
    std::printf("message_gate H=%d R=%d T=%-4d %s%s branches=%d calls=%d  per call median=%8.2f us "
                " min=%8.2f us"
                "  %7.1f GB/s (weights + activations once)\n",
                hidden, rank, tokens, hot ? "hot " : "cold", shift ? " shifted" : "", parallel,
                calls, per_call, t.min_us / calls, bytes / per_call / 1.0e3);
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    int hidden   = 2304;
    int rank     = 64;
    int calls    = 32;
    bool hot     = false;
    int shift    = 0;
    int parallel = 1;
    std::vector<int> tokens;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--hidden") && i + 1 < argc) {
            hidden = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--rank") && i + 1 < argc) {
            rank = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--calls") && i + 1 < argc) {
            calls = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) {
            tokens.push_back(std::atoi(argv[++i]));
        } else if (!std::strcmp(argv[i], "--parallel") && i + 1 < argc) {
            parallel = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--shift")) {
            shift = 1;
        } else if (!std::strcmp(argv[i], "--hot")) {
            hot = true;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--hidden H] [--rank R] [--calls N] [--tokens T]... [--hot] "
                         "[--shift] [--parallel B]\n",
                         argv[0]);
            return 2;
        }
    }
    if (hidden <= 0 || rank <= 0 || calls <= 0 || parallel <= 0) {
        std::fprintf(stderr, "hidden, rank, calls and branches must be positive\n");
        return 2;
    }
    if (tokens.empty()) { tokens = {1, 256}; }
    for (const int t : tokens) { run(hidden, rank, t, calls, hot, shift, parallel); }
    return 0;
}
