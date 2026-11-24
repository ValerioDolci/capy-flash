// Inkling 2 Mini BF16 projections at the tp 2 shard shapes, at decode and verification widths.
// One captured graph issues `--calls` linears back to back, each with its own weight (together
// more than L2, as in a decode step where every projection of every visit has distinct weights).
// Arm A = the generic route (launch_bf16_generic), arm B = the public Op (the shape's registered
// route); arms alternate A B B A per repetition, and the outputs of the two arms are compared bit
// for bit. The time per call is the graph time divided by the calls. --warm: every call reads the
// same weight and L2 is not flushed (as the weights preloaded in L2 at decode widths).
#include "core/weight.h"
#include "direct_bf16_weight.cuh"
#include "ninfer/ops/copy_rows.h"
#include "ninfer/ops/inkling2_projection.h"
#include "ninfer/ops/linear.h"
#include "ninfer_bench_common.h"
#include "ops/linear/bf16/bf16_shapes.h"

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
    std::vector<std::int32_t> blocks; // the rows of a stacked projection's blocks
};

const Shape kShapes[]{
    {"queries", 3276, 2304, {1536, 1536, 192, 12}},
    {"all", 4300, 2304, {1536, 1536, 192, 12, 512, 512}},
    {"keys", 528, 2304, {512, 16}},
    {"key", 512, 2304, {}},
    {"output", 2304, 1536, {}},
};

constexpr std::size_t kColdBytes = std::size_t{192} << 20;

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

void run(const Shape& shape, const std::vector<int>& token_counts, int repeats, bool warm) {
    const std::size_t weight_bytes = std::size_t(shape.n) * shape.k * 2;
    const int calls =
        warm ? 16 : static_cast<int>(std::clamp<std::size_t>(kColdBytes / weight_bytes + 1, 2, 64));
    std::vector<DirectBf16Weight> weights;
    for (int c = 0; c < (warm ? 1 : calls); ++c)
        weights.push_back(make_direct_bf16_weight(shape.n, shape.k, 0x51U + c));
    const int max_t = *std::max_element(token_counts.begin(), token_counts.end());
    DeviceBuffer x  = make_bf16(std::size_t(shape.k) * max_t, 7U, -1.0F, 1.0F);
    DeviceBuffer out_a(std::size_t(shape.n) * max_t * 2);
    DeviceBuffer out_b(std::size_t(shape.n) * max_t * 2);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    L2FlushBuffer flush(std::size_t{128} << 20);

    for (const int tokens : token_counts) {
        const Tensor tx(x.p, DType::BF16, {shape.k, tokens});
        Tensor ta(out_a.p, DType::BF16, {shape.n, tokens});
        Tensor tb(out_b.p, DType::BF16, {shape.n, tokens});
        TimedGraph generic, op;
        generic.capture(stream, [&](cudaStream_t s) {
            for (int c = 0; c < calls; ++c)
                ops::detail::launch_bf16_generic(tx, weights[warm ? 0 : c].weight, ta, s);
        });
        op.capture(stream, [&](cudaStream_t s) {
            for (int c = 0; c < calls; ++c) ops::linear(tx, weights[warm ? 0 : c].weight, tb, s);
        });
        const auto time = [&](const TimedGraph& g) {
            const ColdTiming t = warm ? measure_graph(g, stream, 3, 20)
                                      : measure_cold_graph(g, flush, stream, 3, 20);
            return t.median_us / calls;
        };
        std::vector<double> a, b;
        for (int r = 0; r < repeats; ++r) {
            a.push_back(time(generic));
            b.push_back(time(op));
            b.push_back(time(op));
            a.push_back(time(generic));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::size_t bytes = std::size_t(shape.n) * tokens * 2;
        std::vector<unsigned char> ha(bytes), hb(bytes);
        CUDA_CHECK(cudaMemcpy(ha.data(), out_a.p, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hb.data(), out_b.p, bytes, cudaMemcpyDeviceToHost));
        const bool same = std::memcmp(ha.data(), hb.data(), bytes) == 0;
        const double ma = median(a), mb = median(b);
        std::printf("%-8s%s [%6d,%4d] T=%-2d calls=%-2d  generic %8.2f us  op %8.2f us  x%.2f  "
                    "op %6.1f GB/s  bits %s\n",
                    shape.label, warm ? " warm" : "", shape.n, shape.k, tokens, calls, ma, mb,
                    ma / mb, double(weight_bytes) / mb / 1.0e3, same ? "identical" : "DIFFERENT");
        if (shape.blocks.empty() || tokens < 2) continue;
        // A stacked projection as the forward runs it: the Op and one row copy per block, against
        // the projection writing every block to its own buffer.
        std::vector<DeviceBuffer> parts;
        std::vector<ops::Inkling2ProjectionBlock> blocks;
        for (const std::int32_t rows : shape.blocks) {
            parts.emplace_back(std::size_t(rows) * tokens * 2);
            blocks.push_back({parts.back().p, rows});
        }
        TimedGraph copied, direct;
        copied.capture(stream, [&](cudaStream_t s) {
            for (int c = 0; c < calls; ++c) {
                ops::linear(tx, weights[warm ? 0 : c].weight, tb, s);
                std::size_t row = 0;
                for (std::size_t i = 0; i < blocks.size(); ++i) {
                    ops::copy_rows(parts[i].p, static_cast<const char*>(out_b.p) + row * 2,
                                   std::size_t(shape.n) * 2, std::size_t(blocks[i].rows) * 2,
                                   std::size_t(tokens), s);
                    row += std::size_t(blocks[i].rows);
                }
            }
        });
        bool launched = true;
        direct.capture(stream, [&](cudaStream_t s) {
            for (int c = 0; c < calls; ++c)
                launched = launched && ops::inkling2_projection_blocks(
                                           tx, weights[warm ? 0 : c].weight, blocks, s);
        });
        std::vector<double> pc, pd;
        for (int r = 0; r < repeats; ++r) {
            pc.push_back(time(copied));
            pd.push_back(time(direct));
            pd.push_back(time(direct));
            pc.push_back(time(copied));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        bool same_blocks = launched;
        std::size_t row  = 0;
        for (std::size_t i = 0; i < blocks.size() && same_blocks; ++i) {
            const std::size_t rb = std::size_t(blocks[i].rows) * 2;
            std::vector<unsigned char> hp(rb * tokens);
            CUDA_CHECK(cudaMemcpy(hp.data(), parts[i].p, hp.size(), cudaMemcpyDeviceToHost));
            for (int t = 0; t < tokens && same_blocks; ++t)
                same_blocks =
                    std::memcmp(hp.data() + t * rb,
                                ha.data() + std::size_t(t) * shape.n * 2 + row * 2, rb) == 0;
            row += std::size_t(blocks[i].rows);
        }
        std::printf("%-8s%s [%6d,%4d] T=%-2d calls=%-2d  op+copies %8.2f us  blocks %8.2f us  "
                    "x%.2f  bits %s\n",
                    shape.label, warm ? " warm" : "", shape.n, shape.k, tokens, calls, median(pc),
                    median(pd), median(pc) / median(pd),
                    !launched     ? "NOT LAUNCHED"
                    : same_blocks ? "identical"
                                  : "DIFFERENT");
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    std::vector<int> tokens;
    std::vector<std::string> only;
    int repeats = 3;
    bool warm   = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--warm")) {
            warm = true;
            continue;
        }
        if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) {
            tokens.push_back(std::atoi(argv[++i]));
        } else if (!std::strcmp(argv[i], "--shape") && i + 1 < argc) {
            only.emplace_back(argv[++i]);
        } else if (!std::strcmp(argv[i], "--repeats") && i + 1 < argc) {
            repeats = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr,
                         "usage: %s [--tokens T]... [--shape LABEL]... [--repeats N] [--warm]\n",
                         argv[0]);
            return 2;
        }
    }
    if (tokens.empty()) tokens = {1, 4, 8};
    if (repeats <= 0) return 2;
    for (const Shape& shape : kShapes) {
        if (!only.empty() && std::find(only.begin(), only.end(), shape.label) == only.end())
            continue;
        run(shape, tokens, repeats, warm);
    }
    return 0;
}
