// Inkling 2 Mini BF16 shapes (shapes/inkling2.cu): the registered route gives the generic route's
// bits at every T <= 16 (the GEMV) and at prefill widths up to 1024 (the tiled kernel), column t
// of a T-column call equals the same column computed alone (T = 1) up to T = 16 and the same
// column of any other prefill width beyond, captured graphs replay the same bits, unaligned
// operands fall back to the generic route, and sampled outputs stay within one BF16 rounding of a
// float64 oracle. The stacked projections written block by block (inkling2_projection_blocks)
// give every block the bits of the same rows of ops::linear at T = 2..16, and decline other
// operands.
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/inkling2_projection.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/bf16/bf16_shapes.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace ninfer;

std::uint16_t to_bf16(float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    u += 0x7FFFU + ((u >> 16) & 1U);
    return static_cast<std::uint16_t>(u >> 16);
}

float from_bf16(std::uint16_t h) {
    const std::uint32_t u = std::uint32_t(h) << 16;
    float f               = 0.0F;
    std::memcpy(&f, &u, 4);
    return f;
}

std::vector<std::uint16_t> random_bf16(std::size_t count, std::uint32_t seed, float scale) {
    std::vector<std::uint16_t> v(count);
    for (auto& e : v) {
        seed = seed * 1664525U + 1013904223U;
        e    = to_bf16((static_cast<float>(seed >> 8) / 16777216.0F * 2.0F - 1.0F) * scale);
    }
    return v;
}

#define CHECK_CUDA(call)                                                                           \
    do {                                                                                           \
        if ((call) != cudaSuccess) {                                                               \
            std::printf("FAIL: %s\n", #call);                                                      \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

struct Device {
    void* p = nullptr;

    explicit Device(std::size_t bytes) {
        if (cudaMalloc(&p, bytes) != cudaSuccess) p = nullptr;
    }

    ~Device() { cudaFree(p); }

    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
};

Weight bf16_weight(void* data, std::int32_t n, std::int32_t k) {
    Weight w{};
    w.payload         = data;
    w.payload_bytes   = std::size_t(n) * k * 2;
    w.qtype           = QType::BF16;
    w.shape[0]        = n;
    w.shape[1]        = k;
    w.padded_shape[0] = n;
    w.padded_shape[1] = k;
    w.ndim            = 2;
    w.qdata           = data;
    w.n               = n;
    w.k               = k;
    w.layout          = QuantLayout::Contiguous;
    return w;
}

std::vector<std::uint16_t> download(const void* p, std::size_t count) {
    std::vector<std::uint16_t> v(count);
    cudaMemcpy(v.data(), p, count * 2, cudaMemcpyDeviceToHost);
    return v;
}

int run_shape(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    constexpr int kMaxT     = 17;
    constexpr int kPrefillT = 1024;
    int failures            = 0;
    const auto wh           = random_bf16(std::size_t(n) * k, seed, 0.05F);
    // One spare column so that the activation can start 2 bytes past a 16-byte boundary.
    const auto xh = random_bf16(std::size_t(k) * (kPrefillT + 1), seed + 1U, 1.0F);
    Device w(wh.size() * 2), x(xh.size() * 2), route(std::size_t(n) * kPrefillT * 2),
        generic(std::size_t(n) * kPrefillT * 2), single(std::size_t(n) * 2);
    if (!w.p || !x.p || !route.p || !generic.p || !single.p) {
        std::printf("FAIL: allocation\n");
        return 1;
    }
    CHECK_CUDA(cudaMemcpy(w.p, wh.data(), wh.size() * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(x.p, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice));
    const Weight weight     = bf16_weight(w.p, n, k);
    const std::string shape = "[" + std::to_string(n) + "," + std::to_string(k) + "]";

    for (int tokens = 1; tokens <= kMaxT; ++tokens) {
        const std::size_t count = std::size_t(n) * tokens;
        Tensor tx(x.p, DType::BF16, {k, tokens});
        Tensor tr(route.p, DType::BF16, {n, tokens});
        Tensor tg(generic.p, DType::BF16, {n, tokens});
        CHECK_CUDA(cudaMemset(route.p, 0xFF, count * 2));
        ops::linear(tx, weight, tr, nullptr);
        ops::detail::launch_bf16_generic(tx, weight, tg, nullptr);
        CHECK_CUDA(cudaDeviceSynchronize());
        const auto r = download(route.p, count), g = download(generic.p, count);
        if (r != g) {
            std::printf("FAIL %s T=%d: route bits differ from the generic GEMV\n", shape.c_str(),
                        tokens);
            ++failures;
        }
        // Every column equals the same column alone (the verification round against decode); beyond
        // T = 16 the generic route is the tiled WMMA kernel, whose bits are not the GEMV's.
        for (int t = 0; t < (tokens <= 16 ? tokens : 0); ++t) {
            Tensor xt(static_cast<std::uint16_t*>(x.p) + std::size_t(t) * k, DType::BF16, {k, 1});
            Tensor ts(single.p, DType::BF16, {n, 1});
            ops::linear(xt, weight, ts, nullptr);
            CHECK_CUDA(cudaDeviceSynchronize());
            if (download(single.p, n) !=
                std::vector<std::uint16_t>(r.begin() + std::size_t(t) * n,
                                           r.begin() + std::size_t(t + 1) * n)) {
                std::printf("FAIL %s T=%d: column %d differs from T=1\n", shape.c_str(), tokens, t);
                ++failures;
                break;
            }
        }
        // float64 oracle on sampled rows and every token.
        double worst = 0.0;
        for (const std::int32_t row : {0, 1, n / 3, n / 2, n - 2, n - 1}) {
            for (int t = 0; t < tokens; ++t) {
                double ref = 0.0, mag = 0.0;
                for (std::int32_t c = 0; c < k; ++c) {
                    const double p = double(from_bf16(wh[std::size_t(row) * k + c])) *
                                     from_bf16(xh[std::size_t(t) * k + c]);
                    ref += p;
                    mag += std::fabs(p);
                }
                const double got = from_bf16(r[std::size_t(t) * n + row]);
                const double err = std::fabs(got - ref) / (std::fabs(ref) / 128.0 + mag * 1e-4);
                worst            = std::max(worst, err);
            }
        }
        if (!(worst <= 1.0)) {
            std::printf("FAIL %s T=%d: oracle error %.3f of the bound\n", shape.c_str(), tokens,
                        worst);
            ++failures;
        }
    }

    // Prefill widths: the generic tiled kernel's bits (edges of the 128-token tile included), and
    // every column equal to the same column of the widest call (a prompt cut into other blocks).
    std::vector<std::uint16_t> widest;
    for (const int tokens : {kPrefillT, 17, 64, 127, 128, 129, 255, 300, 512, 541}) {
        const std::size_t count = std::size_t(n) * tokens;
        Tensor tx(x.p, DType::BF16, {k, tokens});
        Tensor tr(route.p, DType::BF16, {n, tokens});
        Tensor tg(generic.p, DType::BF16, {n, tokens});
        CHECK_CUDA(cudaMemset(route.p, 0xFF, count * 2));
        ops::linear(tx, weight, tr, nullptr);
        ops::detail::launch_bf16_generic(tx, weight, tg, nullptr);
        CHECK_CUDA(cudaDeviceSynchronize());
        const auto r = download(route.p, count);
        if (r != download(generic.p, count)) {
            std::printf("FAIL %s T=%d: route bits differ from the generic tiled kernel\n",
                        shape.c_str(), tokens);
            ++failures;
        }
        if (widest.empty()) {
            widest = r;
        } else if (!std::equal(r.begin(), r.end(), widest.begin())) {
            std::printf("FAIL %s T=%d: columns differ from the same columns at T=%d\n",
                        shape.c_str(), tokens, kPrefillT);
            ++failures;
        }
        double worst = 0.0;
        for (const std::int32_t row : {0, n / 2, n - 1}) {
            for (const int t : {0, tokens / 2, tokens - 1}) {
                double ref = 0.0, mag = 0.0;
                for (std::int32_t c = 0; c < k; ++c) {
                    const double p = double(from_bf16(wh[std::size_t(row) * k + c])) *
                                     from_bf16(xh[std::size_t(t) * k + c]);
                    ref += p;
                    mag += std::fabs(p);
                }
                const double got = from_bf16(r[std::size_t(t) * n + row]);
                worst =
                    std::max(worst, std::fabs(got - ref) / (std::fabs(ref) / 128.0 + mag * 1e-4));
            }
        }
        if (!(worst <= 1.0)) {
            std::printf("FAIL %s T=%d: oracle error %.3f of the bound\n", shape.c_str(), tokens,
                        worst);
            ++failures;
        }
    }

    // Graph replay at T = 1, 4, 8 and 300: same bits as eager.
    for (const int tokens : {1, 4, 8, 300}) {
        const std::size_t count = std::size_t(n) * tokens;
        Tensor tx(x.p, DType::BF16, {k, tokens});
        Tensor tr(route.p, DType::BF16, {n, tokens});
        ops::linear(tx, weight, tr, nullptr);
        CHECK_CUDA(cudaDeviceSynchronize());
        const auto eager = download(route.p, count);
        CHECK_CUDA(cudaMemset(route.p, 0xFF, count * 2));
        cudaStream_t stream  = nullptr;
        cudaGraph_t graph    = nullptr;
        cudaGraphExec_t exec = nullptr;
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        ops::linear(tx, weight, tr, stream);
        CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
        CHECK_CUDA(cudaGraphInstantiate(&exec, graph, 0));
        CHECK_CUDA(cudaGraphLaunch(exec, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        if (download(route.p, count) != eager) {
            std::printf("FAIL %s T=%d: graph replay differs from eager\n", shape.c_str(), tokens);
            ++failures;
        }
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
    }

    // Stacked projections block by block: the blocks of the forward's projections (queries: q,
    // gate, relative, sink; all: those and k, v; keys/values: k or v and its dynamic taps).
    std::vector<std::int32_t> stacked;
    if (n == 3276) stacked = {1536, 1536, 192, 12};
    if (n == 4300) stacked = {1536, 1536, 192, 12, 512, 512};
    if (n == 528) stacked = {512, 16};
    for (int tokens = 2; tokens <= 16 && !stacked.empty(); ++tokens) {
        const std::size_t count = std::size_t(n) * tokens;
        Tensor tx(x.p, DType::BF16, {k, tokens});
        Tensor tr(route.p, DType::BF16, {n, tokens});
        ops::linear(tx, weight, tr, nullptr);
        CHECK_CUDA(cudaMemset(generic.p, 0xFF, count * 2));
        std::vector<ops::Inkling2ProjectionBlock> blocks;
        std::size_t offset = 0;
        for (const std::int32_t rows : stacked) {
            blocks.push_back({static_cast<std::uint16_t*>(generic.p) + offset, rows});
            offset += std::size_t(rows) * tokens;
        }
        if (!ops::inkling2_projection_blocks(tx, weight, blocks, nullptr)) {
            std::printf("FAIL %s T=%d: stacked projection declined\n", shape.c_str(), tokens);
            ++failures;
            continue;
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        const auto r = download(route.p, count), b = download(generic.p, count);
        std::size_t first = 0, base = 0;
        bool same = true;
        for (const std::int32_t rows : stacked) {
            for (int t = 0; t < tokens; ++t) {
                const auto from = b.begin() + std::ptrdiff_t(base + std::size_t(t) * rows);
                same            = same && std::equal(from, from + rows,
                                                     r.begin() + std::ptrdiff_t(std::size_t(t) * n + first));
            }
            first += std::size_t(rows);
            base += std::size_t(rows) * tokens;
        }
        if (!same) {
            std::printf("FAIL %s T=%d: stacked blocks differ from the Linear\n", shape.c_str(),
                        tokens);
            ++failures;
        }
    }
    {
        // Declined: one column, rows that do not add up, an unstacked shape.
        Tensor t1(x.p, DType::BF16, {k, 1});
        Tensor t4(x.p, DType::BF16, {k, 4});
        const std::vector<ops::Inkling2ProjectionBlock> whole{{generic.p, n}};
        const std::vector<ops::Inkling2ProjectionBlock> short_rows{{generic.p, n - 1}};
        const bool stackable = !stacked.empty();
        if (ops::inkling2_projection_blocks(t1, weight, whole, nullptr) ||
            ops::inkling2_projection_blocks(t4, weight, short_rows, nullptr) ||
            (!stackable && ops::inkling2_projection_blocks(t4, weight, whole, nullptr))) {
            std::printf("FAIL %s: stacked projection accepted operands it must decline\n",
                        shape.c_str());
            ++failures;
        }
    }

    // An activation 2 bytes past a 16-byte boundary takes the generic route: same bits as it.
    for (const int tokens : {1, 4, 300}) {
        const std::size_t count = std::size_t(n) * tokens;
        Tensor tx(static_cast<std::uint16_t*>(x.p) + 1, DType::BF16, {k, tokens});
        Tensor tr(route.p, DType::BF16, {n, tokens});
        Tensor tg(generic.p, DType::BF16, {n, tokens});
        ops::detail::launch_bf16_generic(tx, weight, tg, nullptr);
        ops::detail::select_bf16_a16_launch(n, k, tokens)(tx, weight, tr, nullptr);
        CHECK_CUDA(cudaDeviceSynchronize());
        if (download(route.p, count) != download(generic.p, count)) {
            std::printf("FAIL %s T=%d unaligned: bits differ from the generic route\n",
                        shape.c_str(), tokens);
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }

    struct Shape {
        std::int32_t n, k;
    };

    constexpr Shape kShapes[]{{3276, 2304}, {4300, 2304}, {528, 2304}, {512, 2304}, {2304, 1536}};
    int failures       = 0;
    std::uint32_t seed = 101U;
    for (const Shape& s : kShapes) failures += run_shape(s.n, s.k, seed += 7U);
    std::printf("%s BF16 Inkling 2 shapes\n", failures == 0 ? "OK" : "FAIL");
    return failures == 0 ? 0 : 1;
}
