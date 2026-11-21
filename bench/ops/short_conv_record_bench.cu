// Microbenchmark of the recording short convolutions against short_conv (the same output, the
// state transition replaced by the record) at the rank shapes of a verification call, and of one
// conv_tail_fold over the 192 tails of a rank at tp 2.
//
//   ./ninfer_short_conv_record_bench [--tokens T]
#include "ninfer/ops/short_conv.h"
#include "ninfer/ops/short_conv_record.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr int kWarmup = 50;
constexpr int kRepeat = 500;

// One shape: C channels, T tokens, G dynamic groups (0 = static taps).
void conv_shape(std::int32_t C, std::int32_t T, std::int32_t G, cudaStream_t stream) {
    DeviceBuffer x = make_bf16(std::size_t(C) * T, 1), state = make_bf16(std::size_t(C) * 3, 2);
    DeviceBuffer weight = make_f32(std::size_t(C) * 4, 3, -1.0F, 1.0F);
    DeviceBuffer dyn    = make_bf16(std::size_t(4) * std::max(G, 1) * T, 4);
    DeviceBuffer out    = make_zeros(std::size_t(C) * T * 2);
    DeviceBuffer record = make_zeros(std::size_t(C) * (T + 3) * 2);
    const Tensor tx(x.p, DType::BF16, {C, T});
    Tensor ts(state.p, DType::BF16, {C, 3});
    Tensor to(out.p, DType::BF16, {C, T});
    Tensor tr(record.p, DType::BF16, {C, T + 3});
    const Tensor dynamic(dyn.p, DType::BF16, {4 * std::max(G, 1), T});
    const float scale = G == 0 ? 1.0F / std::sqrt(5.0F) : 1.0F / 3.0F;
    const auto plain  = [&](cudaStream_t s) {
        if (G == 0) {
            ops::short_conv_static(tx, Tensor(weight.p, DType::FP32, {4, C}), scale, ts, ts, to, s);
        } else {
            ops::short_conv_dynamic(tx, Tensor(weight.p, DType::FP32, {C, 4}), dynamic, scale, ts,
                                     ts, to, s);
        }
    };
    const auto recorded = [&](cudaStream_t s) {
        if (G == 0) {
            ops::short_conv_static_record(tx, Tensor(weight.p, DType::FP32, {4, C}), scale, ts, tr,
                                          to, s);
        } else {
            ops::short_conv_dynamic_record(tx, Tensor(weight.p, DType::FP32, {C, 4}), dynamic,
                                           scale, ts, tr, to, s);
        }
    };
    const ColdTiming a = measure_launch(plain, stream, kWarmup, kRepeat);
    const ColdTiming b = measure_launch(recorded, stream, kWarmup, kRepeat);
    std::printf("%s C=%d T=%d%s: short_conv %.2f us (2 kernels), record %.2f us (1 kernel)\n",
                G == 0 ? "static " : "dynamic", C, T,
                G == 0 ? "" : (" G=" + std::to_string(G)).c_str(), a.median_us, b.median_us);
}

void fold_shape(std::int32_t T, cudaStream_t stream) {
    std::vector<DeviceBuffer> records, tails;
    std::vector<ops::ConvTailFoldEntry> host;
    for (int slot = 0; slot < 48; ++slot) {
        for (const std::int32_t C : {512, 512, 2304, 2304}) {
            records.push_back(make_bf16(std::size_t(C) * (T + 3), 10 + host.size()));
            tails.push_back(make_zeros(std::size_t(C) * 3 * 2));
            host.push_back({records.back().p, tails.back().p, C});
        }
    }
    DeviceBuffer entries(host.size() * sizeof(ops::ConvTailFoldEntry));
    entries.copy_from_host(host.data(), entries.bytes);
    DeviceBuffer committed = make_zeros(4);
    const std::int32_t n   = T / 2;
    committed.copy_from_host(&n, 4);
    const auto fold = [&](cudaStream_t s) {
        ops::conv_tail_fold(static_cast<const ops::ConvTailFoldEntry*>(entries.p),
                            static_cast<std::int32_t>(host.size()), T,
                            Tensor(committed.p, DType::I32, {1}), s);
    };
    const ColdTiming t = measure_launch(fold, stream, kWarmup, kRepeat);
    std::printf("conv_tail_fold %zu tails T=%d: %.2f us\n", host.size(), T, t.median_us);
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::int32_t T = 8;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            if (arg == "--tokens" && i + 1 < argc) {
                T = std::stoi(argv[++i]);
            } else {
                std::printf("usage: %s [--tokens T]\n", argv[0]);
                return 2;
            }
        }
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        conv_shape(2304, T, 0, stream);
        conv_shape(512, T, 0, stream);
        conv_shape(512, T, 4, stream);
        conv_shape(2304, 1, 0, stream);
        fold_shape(T, stream);
        CUDA_CHECK(cudaStreamDestroy(stream));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "short_conv_record_bench: %s\n", error.what());
        return 1;
    }
    return 0;
}
