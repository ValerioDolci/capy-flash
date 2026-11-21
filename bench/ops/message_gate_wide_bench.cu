// message_gate at the model geometry (H = 2304, R = 64), narrow routes against the wide route.
// One captured graph per arm issues `--calls` gates back to back, each with its own weight set
// (together more than L2, as the 96 gates of a prompt block). Arm A = message_gate_launch (the
// cluster route), arm B = the wide route; arms alternate A B B A per repetition and the outputs
// are compared bit for bit. The time per call is the graph time divided by the calls.
#include "core/tensor.h"
#include "ninfer_bench_common.h"
#include "ops/message_gate/launch.h"
#include "ops/message_gate/wide.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

struct WeightSet {
    DeviceBuffer down, forget, gain;
};

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

void run(int tokens, int calls, int repeats) {
    constexpr int hidden = 2304, rank = 64;
    const std::size_t columns = static_cast<std::size_t>(hidden) * tokens;
    DeviceBuffer h            = make_f32(columns, 11U, -2.0F, 2.0F);
    DeviceBuffer m            = make_f32(columns, 13U, -3.0F, 3.0F);
    DeviceBuffer out_a(columns * sizeof(float)), out_b(columns * sizeof(float));
    std::vector<WeightSet> sets(static_cast<std::size_t>(calls));
    for (std::size_t s = 0; s < sets.size(); ++s) {
        sets[s].down =
            make_bf16(static_cast<std::size_t>(2) * hidden * rank, 17U + s, -0.03F, 0.03F);
        sets[s].forget = make_f32(static_cast<std::size_t>(hidden) * rank, 19U + s, -0.2F, 0.2F);
        sets[s].gain   = make_f32(static_cast<std::size_t>(hidden) * rank, 23U + s, -0.2F, 0.2F);
    }
    const Tensor th(h.p, DType::FP32, {hidden, tokens});
    const Tensor tm(m.p, DType::FP32, {hidden, tokens});
    Tensor ta(out_a.p, DType::FP32, {hidden, tokens});
    Tensor tb(out_b.p, DType::FP32, {hidden, tokens});
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const auto weights = [&](int c, Tensor& d, Tensor& f, Tensor& g) {
        const WeightSet& w = sets[static_cast<std::size_t>(c)];
        d                  = Tensor(w.down.p, DType::BF16, {2 * hidden, rank});
        f                  = Tensor(w.forget.p, DType::FP32, {rank, hidden});
        g                  = Tensor(w.gain.p, DType::FP32, {rank, hidden});
    };
    TimedGraph narrow, wide;
    narrow.capture(stream, [&](cudaStream_t s) {
        for (int c = 0; c < calls; ++c) {
            Tensor d, f, g;
            weights(c, d, f, g);
            ops::detail::message_gate_launch(th, tm, d, f, g, 1.0e-6F, ta, s);
        }
    });
    wide.capture(stream, [&](cudaStream_t s) {
        for (int c = 0; c < calls; ++c) {
            Tensor d, f, g;
            weights(c, d, f, g);
            if (!ops::detail::message_gate_wide(th, tm, d, f, g, 1.0e-6F, tb, s)) std::abort();
        }
    });
    L2FlushBuffer flush(std::size_t{128} << 20);
    std::vector<double> a, b;
    for (int r = 0; r < repeats; ++r) {
        a.push_back(measure_cold_graph(narrow, flush, stream, 3, 20).median_us / calls);
        b.push_back(measure_cold_graph(wide, flush, stream, 3, 20).median_us / calls);
        b.push_back(measure_cold_graph(wide, flush, stream, 3, 20).median_us / calls);
        a.push_back(measure_cold_graph(narrow, flush, stream, 3, 20).median_us / calls);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<unsigned char> ha(columns * 4), hb(columns * 4);
    CUDA_CHECK(cudaMemcpy(ha.data(), out_a.p, columns * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), out_b.p, columns * 4, cudaMemcpyDeviceToHost));
    const bool same = std::memcmp(ha.data(), hb.data(), columns * 4) == 0;
    const double ma = median(a), mb = median(b);
    std::printf("message_gate T=%-5d calls=%d  narrow %9.2f us  wide %9.2f us  x%.2f  bits %s\n",
                tokens, calls, ma, mb, ma / mb, same ? "identical" : "DIFFERENT");
    CUDA_CHECK(cudaStreamDestroy(stream));
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    int calls = 24, repeats = 3;
    std::vector<int> tokens;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) {
            tokens.push_back(std::atoi(argv[++i]));
        } else if (!std::strcmp(argv[i], "--calls") && i + 1 < argc) {
            calls = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--repeats") && i + 1 < argc) {
            repeats = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "usage: %s [--tokens T]... [--calls N] [--repeats N]\n", argv[0]);
            return 2;
        }
    }
    if (tokens.empty()) tokens = {16, 32, 64, 128, 512, 1024};
    if (calls <= 0 || repeats <= 0) return 2;
    for (const int t : tokens) run(t, calls, repeats);
    return 0;
}
