// Microbenchmark of the hot DSpark drafter Ops at the rank shapes of Inkling 2 Mini at tp 2: the
// block attention (12/4 heads, B = 8) over a local window of 1024 and over a global context of
// 1024 / 12800 / 131072 positions, and one recurrent head step over a vocabulary half (100029
// rows, rank 256), each against the bytes it must read.
//
//   ./ninfer_dspark_draft_bench
#include "ninfer/ops/dspark_draft.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr int kWarmup = 20;
constexpr int kRepeat = 200;

void attention(std::int32_t P, std::int32_t window, std::int32_t capacity, cudaStream_t stream) {
    const int nq = 12, kv = 4, B = 8, D = 128, rows = (nq + 2 * kv) * D;
    DeviceBuffer qkv = make_bf16(std::size_t(rows) * B, 1),
                 kc  = make_bf16(std::size_t(D) * kv * capacity, 2),
                 vc  = make_bf16(std::size_t(D) * kv * capacity, 3),
                 out = make_zeros(std::size_t(nq) * D * B * 2), frontier = make_zeros(4);
    frontier.copy_from_host(&P, 4);
    WorkspaceArena work(ops::block_attention_workspace_bytes(nq, B) + 4096);
    const Tensor tq(qkv.p, DType::BF16, {rows, B}), tf(frontier.p, DType::I32, {1});
    Tensor to(out.p, DType::BF16, {nq * D, B});
    const ops::DraftKVCacheView cache{Tensor(kc.p, DType::BF16, {D, kv, capacity}),
                                      Tensor(vc.p, DType::BF16, {D, kv, capacity}), capacity};
    const auto run = [&](cudaStream_t s) {
        ops::block_attention(tq, nq, kv, cache, tf, window, 1.0F / std::sqrt(128.0F), work, to, s);
    };
    const ColdTiming t  = measure_launch(run, stream, kWarmup, kRepeat);
    const int keys      = (window > 0 ? std::min(P, window) : P);
    const double mbytes = double(keys) * kv * D * 2 * 2 / 1e6;
    std::printf("block_attention P=%d window=%d: %.2f us (%.2f MB of keys/values, %.0f GB/s)\n", P,
                window, t.median_us, mbytes, mbytes * 1e3 / t.median_us);
}

void recurrent(cudaStream_t stream) {
    const int R = 256, H = 2304, G = 7, rows = 100029, V = 201024, in = 2 * R + 2 * H;
    DeviceBuffer w   = make_bf16(std::size_t(3) * R * in, 4, -0.02F, 0.02F),
                 emb = make_bf16(std::size_t(V) * R, 5), fold = make_bf16(std::size_t(rows) * R, 6),
                 fbias   = make_bf16(rows, 7),
                 base    = make_bf16(std::size_t(rows) * G, 8, -10.0F, 10.0F),
                 context = make_f32(std::size_t(3) * R * G, 9, -1.0F, 1.0F), s0 = make_zeros(R * 2),
                 s1 = make_zeros(R * 2), drafts = make_zeros(G * 4), choice = make_zeros(16),
                 anchor = make_zeros(4);
    ops::RecurrentHeadOperands head{Tensor(w.p, DType::BF16, {in, 3 * R}),
                                    Tensor(emb.p, DType::BF16, {R, V}),
                                    Tensor(fold.p, DType::BF16, {R, rows}),
                                    Tensor(fbias.p, DType::BF16, {rows}),
                                    Tensor(base.p, DType::BF16, {rows, G}),
                                    Tensor(context.p, DType::FP32, {3 * R, G}),
                                    30.0F,
                                    0,
                                    0,
                                    2};
    WorkspaceArena work(ops::recurrent_head_step_workspace_bytes() + 4096);
    const Tensor ta(anchor.p, DType::I32, {1}), si(s0.p, DType::BF16, {R});
    Tensor so(s1.p, DType::BF16, {R}), td(drafts.p, DType::I32, {G}),
        tc(choice.p, DType::FP32, {4});
    const auto run = [&](cudaStream_t s) {
        ops::recurrent_head_step(head, 1, ta, si, so, td, tc, work, s);
    };
    const ColdTiming t  = measure_launch(run, stream, kWarmup, kRepeat);
    const double mbytes = (double(rows) * R * 2 + double(3) * R * 2 * R * 2) / 1e6;
    std::printf("recurrent_head_step rows=%d: %.2f us (%.1f MB, %.0f GB/s)\n", rows, t.median_us,
                mbytes, mbytes * 1e3 / t.median_us);
}

} // namespace

int main() {
    try {
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        attention(1024, 1024, 1024, stream);
        attention(12800, 1024, 1024, stream);
        attention(1024, 0, 131072, stream);
        attention(12800, 0, 131072, stream);
        attention(131064, 0, 131072, stream);
        recurrent(stream);
        CUDA_CHECK(cudaStreamDestroy(stream));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "dspark_draft_bench: %s\n", error.what());
        return 1;
    }
    return 0;
}
