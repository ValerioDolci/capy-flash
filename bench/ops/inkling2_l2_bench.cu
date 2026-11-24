// Inkling 2 Mini BF16 projections at the tp 2 shard shapes with their weights cold, already in L2,
// or prefetched into L2 on a side branch of the graph while an idle window runs on the main
// stream (a spin kernel standing in for the decode's control kernels, which leave DRAM idle).
//
// Per shape and token count, every time per call, captured graphs, L2 flushed before each launch:
//   cold     `calls` projections back to back, each with its own weight (together more than L2)
//   hot      16 projections of one weight (in L2 after the first)
//   alone    `calls` prefetches of distinct weights back to back: ops::l2_prefetch (hint, bulk
//            prefetch hints) and ops::l2_preload with B blocks for every --blocks B
//   window W base: [spin W us] -> projection c; the gain of each prefetch of weight c forked at
//            the start of the window onto a side stream (joined once at the end)
//   --evict E: in both window arms the main stream also streams E MiB of other data after the
//            window and before the projection (a MoE bank between the prefetch and its use)
#include "core/weight.h"
#include "direct_bf16_weight.cuh"
#include "ninfer/ops/l2_preload.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/peer_mailbox.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

struct Shape {
    const char* label;
    std::int32_t n, k;
};

constexpr Shape kShapes[]{
    {"queries", 3276, 2304}, {"all", 4300, 2304},    {"keys", 528, 2304},
    {"key", 512, 2304},      {"output", 2304, 1536},
};

constexpr std::size_t kColdBytes = std::size_t{192} << 20;
constexpr int kHint              = -1;
constexpr int kNone              = -2;

__global__ void idle_window(std::uint64_t ns) {
    std::uint64_t start = 0;
    std::uint64_t now   = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now)); } while (now - start < ns);
}

// Streams `vectors` uint4 of `input` through L2 (a stand-in for other weights read in between).
__global__ void stream_other(const uint4* input, std::size_t vectors, uint4* sink) {
    uint4 acc{};
    for (auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < vectors;
         i += std::size_t(gridDim.x) * blockDim.x) {
        const uint4 v = input[i];
        acc.x ^= v.x;
        acc.y ^= v.y;
        acc.z ^= v.z;
        acc.w ^= v.w;
    }
    if ((acc.x & acc.y & acc.z & acc.w) == 0xffffffffU) { sink[0] = acc; }
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

void print_device() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int l2 = 0, persist = 0, sms = 0, bus = 0, mem_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&persist, cudaDevAttrMaxPersistingL2CacheSize, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, device));
    std::printf("device %d: L2 %.1f MiB, max persisting L2 %.1f MiB, SMs %d, bus %d bit, "
                "memory clock %d kHz\n",
                device, l2 / 1048576.0, persist / 1048576.0, sms, bus, mem_khz);
}

struct Options {
    std::vector<int> tokens;
    std::vector<std::string> only;
    std::vector<int> windows;
    std::vector<int> blocks;
    int evict_mib = 0;
    int repeats   = 3;
};

void run(const Shape& shape, const Options& o) {
    const std::size_t weight_bytes = std::size_t(shape.n) * shape.k * 2;
    const int calls =
        static_cast<int>(std::clamp<std::size_t>(kColdBytes / weight_bytes + 1, 2, 64));
    std::vector<DirectBf16Weight> weights;
    for (int c = 0; c < calls; ++c)
        weights.push_back(make_direct_bf16_weight(shape.n, shape.k, 0x51U + c));
    const int max_t = *std::max_element(o.tokens.begin(), o.tokens.end());
    DeviceBuffer x  = make_bf16(std::size_t(shape.k) * max_t, 7U, -1.0F, 1.0F);
    DeviceBuffer out(std::size_t(shape.n) * max_t * 2);
    const std::size_t other_bytes = std::size_t(o.evict_mib) << 20;
    DeviceBuffer other(std::max<std::size_t>(other_bytes * calls, 16));
    DeviceBuffer sink(16);
    if (other_bytes != 0) {
        CUDA_CHECK(cudaMemset(other.p, 1, other.bytes));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    cudaStream_t stream = nullptr, side = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking));
    cudaEvent_t fork = nullptr, join = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
    L2FlushBuffer flush(std::size_t{128} << 20);

    for (const int tokens : o.tokens) {
        const Tensor tx(x.p, DType::BF16, {shape.k, tokens});
        Tensor to(out.p, DType::BF16, {shape.n, tokens});
        const auto timed = [&](auto&& body) {
            TimedGraph g;
            g.capture(stream, body);
            std::vector<double> v;
            for (int r = 0; r < o.repeats; ++r)
                v.push_back(measure_cold_graph(g, flush, stream, 3, 20).median_us);
            return median(v);
        };
        const double cold = timed([&](cudaStream_t s) {
                                for (int c = 0; c < calls; ++c)
                                    ops::linear(tx, weights[c].weight, to, s);
                            }) /
                            calls;
        TimedGraph hot_graph;
        hot_graph.capture(stream, [&](cudaStream_t s) {
            for (int c = 0; c < 16; ++c) ops::linear(tx, weights[0].weight, to, s);
        });
        std::vector<double> hv;
        for (int r = 0; r < o.repeats; ++r)
            hv.push_back(measure_graph(hot_graph, stream, 3, 20).median_us / 16);
        const double hot = median(hv);
        // Arm a: kHint = ops::l2_prefetch, b > 0 = ops::l2_preload with b blocks.
        const auto issue = [&](int a, int c, cudaStream_t s) {
            if (a == kHint) {
                const ops::PeerL2Range r[1] = {{weights[c].storage.p, weight_bytes}};
                ops::l2_prefetch(r, s);
                return;
            }
            const ops::L2PreloadRange r[1] = {{weights[c].storage.p, weight_bytes}};
            ops::l2_preload(r, a, s);
        };
        std::vector<int> arms{kHint};
        arms.insert(arms.end(), o.blocks.begin(), o.blocks.end());
        std::printf("%-8s [%5d,%4d] T=%-2d calls=%-2d %5.1f MiB  cold %7.2f us (%5.0f GB/s)  "
                    "hot %7.2f us (%5.0f GB/s)  x%.2f\n",
                    shape.label, shape.n, shape.k, tokens, calls, weight_bytes / 1048576.0, cold,
                    weight_bytes / cold / 1e3, hot, weight_bytes / hot / 1e3, cold / hot);
        for (const int a : arms) {
            const double alone = timed([&](cudaStream_t s) {
                                     for (int c = 0; c < calls; ++c) issue(a, c, s);
                                 }) /
                                 calls;
            std::printf("%-8s   alone %-6s %7.2f us (%5.0f GB/s)\n", shape.label,
                        a == kHint ? "hint" : std::to_string(a).c_str(), alone,
                        weight_bytes / alone / 1e3);
        }
        for (const int w : o.windows) {
            const auto ns  = static_cast<std::uint64_t>(w) * 1000U;
            const auto arm = [&](int a) { // kNone: no prefetch
                return timed([&](cudaStream_t s) {
                           for (int c = 0; c < calls; ++c) {
                               if (a != kNone) {
                                   CUDA_CHECK(cudaEventRecord(fork, s));
                                   CUDA_CHECK(cudaStreamWaitEvent(side, fork, 0));
                                   issue(a, c, side);
                               }
                               if (w > 0) idle_window<<<1, 32, 0, s>>>(ns);
                               if (other_bytes != 0) {
                                   const auto* p = static_cast<const std::uint8_t*>(other.p) +
                                                   std::size_t(c) * other_bytes;
                                   stream_other<<<512, 256, 0, s>>>(
                                       reinterpret_cast<const uint4*>(p), other_bytes / 16,
                                       static_cast<uint4*>(sink.p));
                               }
                               ops::linear(tx, weights[c].weight, to, s);
                           }
                           if (a != kNone) {
                               CUDA_CHECK(cudaEventRecord(join, side));
                               CUDA_CHECK(cudaStreamWaitEvent(s, join, 0));
                           }
                       }) /
                       calls;
            };
            std::vector<int> all{kNone};
            all.insert(all.end(), arms.begin(), arms.end());
            std::vector<std::vector<double>> t(all.size());
            for (int r = 0; r < o.repeats; ++r) {
                for (std::size_t i = 0; i < all.size(); ++i) t[i].push_back(arm(all[i]));
                for (std::size_t i = all.size(); i-- > 0;) t[i].push_back(arm(all[i]));
            }
            const double mb = median(t[0]);
            std::printf(
                "%-8s T=%-2d window %3d us evict %3d MiB: base %7.2f us (window+cold %7.2f, "
                "window+hot %7.2f) gain:",
                shape.label, tokens, w, o.evict_mib, mb, w + cold, w + hot);
            for (std::size_t i = 1; i < all.size(); ++i) {
                std::printf("  %s %6.2f", all[i] == kHint ? "hint" : std::to_string(all[i]).c_str(),
                            mb - median(t[i]));
            }
            std::printf("  (max %.2f)\n", cold - hot);
        }
    }
    CUDA_CHECK(cudaEventDestroy(fork));
    CUDA_CHECK(cudaEventDestroy(join));
    CUDA_CHECK(cudaStreamDestroy(side));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    Options o;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) {
            o.tokens.push_back(std::atoi(argv[++i]));
        } else if (!std::strcmp(argv[i], "--shape") && i + 1 < argc) {
            o.only.emplace_back(argv[++i]);
        } else if (!std::strcmp(argv[i], "--window") && i + 1 < argc) {
            o.windows.push_back(std::atoi(argv[++i]));
        } else if (!std::strcmp(argv[i], "--blocks") && i + 1 < argc) {
            o.blocks.push_back(std::atoi(argv[++i]));
        } else if (!std::strcmp(argv[i], "--evict") && i + 1 < argc) {
            o.evict_mib = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--repeats") && i + 1 < argc) {
            o.repeats = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr,
                         "usage: %s [--tokens T]... [--shape LABEL]... [--window US]... "
                         "[--blocks B]... [--evict MIB] [--repeats N]\n",
                         argv[0]);
            return 2;
        }
    }
    if (o.tokens.empty()) o.tokens = {1};
    if (o.blocks.empty()) o.blocks = {8, 12, 16};
    for (const int b : o.blocks)
        if (b <= 0) return 2;
    if (o.repeats <= 0 || o.evict_mib < 0) return 2;
    print_device();
    for (const Shape& shape : kShapes) {
        if (!o.only.empty() && std::find(o.only.begin(), o.only.end(), shape.label) == o.only.end())
            continue;
        run(shape, o);
    }
    return 0;
}
