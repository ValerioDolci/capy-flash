// message_gate at the model geometry (H = 2304, R = 64). One captured graph issues `--calls`
// gates back to back, each with its own weight set: with the default 32 sets the 57 MB of weights
// exceed L2, as in a decode step where every gate of every visit has distinct weights. `--hot`
// reuses one weight set (L2 resident); `--shift` places hidden, message and out 4 bytes past a
// 16-byte boundary (a valid contiguous placement that the Op serves with another route). The time
// per call is the graph time divided by the calls. `--parallel B` spreads the calls over B parallel
// graph branches (independent gates, such as the q, k and v gates of one visit). `--batch G` makes
// each call one message_gates_rmsnorm of G gates (a visit's q, k, v gates: G = 3; its FFN gate:
// G = 1), each gate with its own weights; `--lead` puts before each call the depth attention that
// precedes the gates in a decode visit (2 states, 4 queries), so that the call's dependent launch
// overlaps it as in the model.
#include "ninfer/ops/depth_attention.h"
#include "ninfer/ops/message_gate.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

struct WeightSet {
    DeviceBuffer down, forget, gain, forget16, gain16;
};

void run(int hidden, int rank, int tokens, int calls, bool hot, int shift, int parallel, int batch,
         bool lead, bool bf16) {
    const std::size_t columns = static_cast<std::size_t>(hidden) * tokens;
    DeviceBuffer h            = make_f32(columns + 1, 11U, -2.0F, 2.0F);
    DeviceBuffer m            = make_f32(columns + 1, 13U, -3.0F, 3.0F);
    std::vector<DeviceBuffer> outs;
    for (int b = 0; b < parallel; ++b) { outs.emplace_back((columns + 1) * sizeof(float)); }
    const int per_call = std::max(batch, 1);
    std::vector<WeightSet> sets(hot ? static_cast<std::size_t>(per_call)
                                    : static_cast<std::size_t>(calls) * per_call);
    for (std::size_t s = 0; s < sets.size(); ++s) {
        sets[s].down =
            make_bf16(static_cast<std::size_t>(2) * hidden * rank, 17U + s, -0.03F, 0.03F);
        sets[s].forget = make_f32(static_cast<std::size_t>(hidden) * rank, 19U + s, -0.2F, 0.2F);
        sets[s].gain   = make_f32(static_cast<std::size_t>(hidden) * rank, 23U + s, -0.2F, 0.2F);
        if (bf16) {
            sets[s].forget16 =
                make_bf16(static_cast<std::size_t>(hidden) * rank, 19U + s, -0.2F, 0.2F);
            sets[s].gain16 =
                make_bf16(static_cast<std::size_t>(hidden) * rank, 23U + s, -0.2F, 0.2F);
        }
    }
    const Tensor th(static_cast<float*>(h.p) + shift, DType::FP32, {hidden, tokens});
    const Tensor tm(static_cast<float*>(m.p) + shift, DType::FP32, {hidden, tokens});
    // Batch outputs: per gate an FP32 out and a BF16 norm; the norm weight.
    std::vector<DeviceBuffer> batch_out, batch_normed;
    for (int g = 0; g < batch; ++g) {
        batch_out.emplace_back((columns + 1) * sizeof(float));
        batch_normed.emplace_back(columns * sizeof(__nv_bfloat16));
    }
    DeviceBuffer norm = make_bf16(static_cast<std::size_t>(hidden), 29U, 0.5F, 1.5F);
    // The lead depth attention: 2 BF16 states and a BF16 current, 4 FP32 queries.
    DeviceBuffer lead_states[2] = {make_bf16(columns, 31U, -2.0F, 2.0F),
                                   make_bf16(columns, 37U, -2.0F, 2.0F)};
    DeviceBuffer lead_current   = make_bf16(columns, 41U, -2.0F, 2.0F);
    DeviceBuffer lead_query     = make_f32(static_cast<std::size_t>(hidden) * 4, 43U, -1.0F, 1.0F);
    DeviceBuffer lead_out(columns * 4 * sizeof(float));
    const Tensor lead_state_tensors[2] = {Tensor(lead_states[0].p, DType::BF16, {hidden, tokens}),
                                          Tensor(lead_states[1].p, DType::BF16, {hidden, tokens})};

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
                if (lead) {
                    Tensor lo(lead_out.p, DType::FP32, {hidden, tokens, 4});
                    ops::depth_attention(std::span<const Tensor>(lead_state_tensors, 2),
                                         Tensor(lead_current.p, DType::BF16, {hidden, tokens}),
                                         Tensor(lead_query.p, DType::FP32, {hidden, 4}), 1.0e-6F,
                                         0.02F, lo, branches[b]);
                }
                const auto set = [&](int g) -> const WeightSet& {
                    return sets[hot ? static_cast<std::size_t>(g)
                                    : static_cast<std::size_t>(c) * per_call + g];
                };
                if (batch > 0) {
                    std::vector<ops::MessageGateNormed> gates;
                    for (int g = 0; g < batch; ++g) {
                        const WeightSet& w = set(g);
                        if (bf16) {
                            gates.push_back(
                                {th, tm, Tensor(w.down.p, DType::BF16, {2 * hidden, rank}),
                                 Tensor(w.forget16.p, DType::BF16, {rank, hidden}),
                                 Tensor(w.gain16.p, DType::BF16, {rank, hidden}),
                                 Tensor(static_cast<float*>(batch_out[g].p) + shift, DType::FP32,
                                        {hidden, tokens}),
                                 Tensor(norm.p, DType::BF16, {hidden}),
                                 Tensor(batch_normed[g].p, DType::BF16, {hidden, tokens})});
                            continue;
                        }
                        gates.push_back({th, tm, Tensor(w.down.p, DType::BF16, {2 * hidden, rank}),
                                         Tensor(w.forget.p, DType::FP32, {rank, hidden}),
                                         Tensor(w.gain.p, DType::FP32, {rank, hidden}),
                                         Tensor(static_cast<float*>(batch_out[g].p) + shift,
                                                DType::FP32, {hidden, tokens}),
                                         Tensor(norm.p, DType::BF16, {hidden}),
                                         Tensor(batch_normed[g].p, DType::BF16, {hidden, tokens})});
                    }
                    ops::message_gates_rmsnorm(gates, 1.0e-6F, 1.0e-6F, branches[b]);
                    continue;
                }
                const WeightSet& w = set(0);
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

    const double weight_bytes = static_cast<double>(rank) * hidden * (2 * 2 + 2 * 4) * per_call;
    const double bytes        = weight_bytes + 3.0 * static_cast<double>(columns) * 4.0;
    const double per_call_us  = t.median_us / calls;
    std::printf("message_gate H=%d R=%d T=%-4d %s%s batch=%d%s branches=%d calls=%d  per call "
                "median=%8.2f us "
                " min=%8.2f us"
                "  %7.1f GB/s (weights + activations once)\n",
                hidden, rank, tokens, hot ? "hot " : "cold", shift ? " shifted" : "", batch,
                lead ? " lead" : "", parallel, calls, per_call_us, t.min_us / calls,
                bytes / per_call_us / 1.0e3);
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
    int batch    = 0;
    bool lead    = false;
    bool bf16    = false;
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
        } else if (!std::strcmp(argv[i], "--batch") && i + 1 < argc) {
            batch = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--lead")) {
            lead = true;
        } else if (!std::strcmp(argv[i], "--bf16")) {
            bf16 = true;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--hidden H] [--rank R] [--calls N] [--tokens T]... [--hot] "
                         "[--shift] [--parallel B] [--batch G] [--lead]\n",
                         argv[0]);
            return 2;
        }
    }
    if (hidden <= 0 || rank <= 0 || calls <= 0 || parallel <= 0 || batch < 0 ||
        batch > ops::kMessageGateMaxBatch) {
        std::fprintf(stderr, "hidden, rank, calls and branches must be positive\n");
        return 2;
    }
    if (tokens.empty()) { tokens = {1, 256}; }
    for (const int t : tokens) {
        run(hidden, rank, t, calls, hot, shift, parallel, batch, lead, bf16);
    }
    return 0;
}
