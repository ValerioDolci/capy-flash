// message_gate's wide route (ops/message_gate/wide.cu) bit for bit against the narrow routes
// (message_gate_launch: the cluster route at the model geometry) at every T, in place (out =
// hidden, as the FFN gate is called), through the public Op beyond the threshold, and captured in
// a graph. The narrow routes carry the contract's oracle test (test_message_gate.cpp).
#include "core/tensor.h"
#include "ninfer/ops/message_gate.h"
#include "ops/message_gate/launch.h"
#include "ops/message_gate/wide.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using namespace ninfer;

constexpr int kHidden = 2304;
constexpr int kRank   = 64;
constexpr int kMaxT   = 1500;
constexpr float kEps  = 1.0e-6F;

std::uint32_t next(std::uint32_t& seed) { return seed = seed * 1664525U + 1013904223U; }

std::vector<float> random_f32(std::size_t count, std::uint32_t seed, float scale) {
    std::vector<float> v(count);
    for (auto& e : v) e = (static_cast<float>(next(seed) >> 8) / 16777216.0F * 2.0F - 1.0F) * scale;
    return v;
}

std::vector<std::uint16_t> random_bf16(std::size_t count, std::uint32_t seed, float scale) {
    std::vector<std::uint16_t> v(count);
    for (auto& e : v) {
        float f         = (static_cast<float>(next(seed) >> 8) / 16777216.0F * 2.0F - 1.0F) * scale;
        std::uint32_t u = 0;
        std::memcpy(&u, &f, 4);
        u += 0x7FFFU + ((u >> 16) & 1U);
        e = static_cast<std::uint16_t>(u >> 16);
    }
    return v;
}

struct Device {
    void* p = nullptr;

    explicit Device(std::size_t bytes) {
        if (cudaMalloc(&p, bytes) != cudaSuccess) p = nullptr;
    }

    ~Device() { cudaFree(p); }

    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
};

template <class T>
bool upload(void* d, const std::vector<T>& h) {
    return cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
}

std::vector<float> download(const void* p, std::size_t count) {
    std::vector<float> v(count);
    cudaMemcpy(v.data(), p, count * 4, cudaMemcpyDeviceToHost);
    return v;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    const std::size_t columns = std::size_t(kHidden) * kMaxT;
    Device h(columns * 4), m(columns * 4), narrow(columns * 4), wide(columns * 4),
        down(std::size_t(2) * kHidden * kRank * 2), forget(std::size_t(kHidden) * kRank * 4),
        gain(std::size_t(kHidden) * kRank * 4);
    if (!h.p || !m.p || !narrow.p || !wide.p || !down.p || !forget.p || !gain.p) {
        std::printf("FAIL: allocation\n");
        return 1;
    }
    const auto hh = random_f32(columns, 11U, 2.0F);
    if (!upload(h.p, hh) || !upload(m.p, random_f32(columns, 13U, 3.0F)) ||
        !upload(down.p, random_bf16(std::size_t(2) * kHidden * kRank, 17U, 0.03F)) ||
        !upload(forget.p, random_f32(std::size_t(kHidden) * kRank, 19U, 0.2F)) ||
        !upload(gain.p, random_f32(std::size_t(kHidden) * kRank, 23U, 0.2F))) {
        std::printf("FAIL: upload\n");
        return 1;
    }
    const Tensor td(down.p, DType::BF16, {2 * kHidden, kRank});
    const Tensor tf(forget.p, DType::FP32, {kRank, kHidden});
    const Tensor tg(gain.p, DType::FP32, {kRank, kHidden});
    int failures = 0;
    for (const int T : {1, 3, 4, 5, 8, 9, 16, 17, 63, 64, 100, 512, 1023, 1024, 1500}) {
        const std::size_t count = std::size_t(kHidden) * T;
        const Tensor th(h.p, DType::FP32, {kHidden, T});
        const Tensor tm(m.p, DType::FP32, {kHidden, T});
        Tensor tn(narrow.p, DType::FP32, {kHidden, T});
        Tensor tw(wide.p, DType::FP32, {kHidden, T});
        cudaMemset(wide.p, 0xFF, count * 4);
        ops::detail::message_gate_launch(th, tm, td, tf, tg, kEps, tn, nullptr);
        if (!ops::detail::message_gate_wide(th, tm, td, tf, tg, kEps, tw, nullptr)) {
            std::printf("FAIL T=%d: the wide route refused the model geometry\n", T);
            ++failures;
            continue;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::printf("FAIL T=%d: CUDA error\n", T);
            return 1;
        }
        const auto expected = download(narrow.p, count);
        if (!same_bits(download(wide.p, count), expected)) {
            std::printf("FAIL T=%d: wide route bits differ from the narrow route\n", T);
            ++failures;
        }
        // In place, as the FFN gate: out is the hidden storage.
        cudaMemcpy(wide.p, h.p, count * 4, cudaMemcpyDeviceToDevice);
        Tensor in_place(wide.p, DType::FP32, {kHidden, T});
        (void)ops::detail::message_gate_wide(in_place, tm, td, tf, tg, kEps, in_place, nullptr);
        cudaDeviceSynchronize();
        if (!same_bits(download(wide.p, count), expected)) {
            std::printf("FAIL T=%d: in-place wide route differs\n", T);
            ++failures;
        }
        // The public Op (its route for this T) gives the same bits.
        cudaMemset(wide.p, 0xFF, count * 4);
        ops::message_gate(th, tm, td, tf, tg, kEps, tw, nullptr);
        cudaDeviceSynchronize();
        if (!same_bits(download(wide.p, count), expected)) {
            std::printf("FAIL T=%d: public Op differs\n", T);
            ++failures;
        }
    }
    // Captured in a graph: the same bits as eager.
    {
        constexpr int T         = 300;
        const std::size_t count = std::size_t(kHidden) * T;
        const Tensor th(h.p, DType::FP32, {kHidden, T});
        const Tensor tm(m.p, DType::FP32, {kHidden, T});
        Tensor tn(narrow.p, DType::FP32, {kHidden, T});
        Tensor tw(wide.p, DType::FP32, {kHidden, T});
        ops::detail::message_gate_launch(th, tm, td, tf, tg, kEps, tn, nullptr);
        cudaMemset(wide.p, 0xFF, count * 4);
        cudaDeviceSynchronize();
        cudaStream_t stream  = nullptr;
        cudaGraph_t graph    = nullptr;
        cudaGraphExec_t exec = nullptr;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
        (void)ops::detail::message_gate_wide(th, tm, td, tf, tg, kEps, tw, stream);
        cudaStreamEndCapture(stream, &graph);
        if (graph == nullptr || cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess ||
            cudaGraphLaunch(exec, stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess) {
            std::printf("FAIL: graph capture of the wide route\n");
            ++failures;
        } else if (!same_bits(download(wide.p, count), download(narrow.p, count))) {
            std::printf("FAIL T=%d: graph replay differs\n", T);
            ++failures;
        }
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
    }
    std::printf("%s message_gate wide route\n", failures == 0 ? "OK" : "FAIL");
    return failures == 0 ? 0 : 1;
}
