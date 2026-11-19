// The tp-2 Q6 output head half [124160, 2560]: the shape's selector routes 2..8 columns to the
// 32 x 8 sliced-K tile, which gives the same bits as the 32 x 16 route it replaced.
#include "ops/linear/linear_test_common.h"
#include "ops/linear/q6/q6_launch.h"
#include "ops/linear/q6/q6_shapes.h"

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    using namespace ninfer;
    constexpr std::int32_t N = 124160, K = 2560, Tmax = 8;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { return 77; }
    const auto host             = test::linear::make_q6_g64_fp16_weight(N, K, 777U);
    void* payload               = nullptr;
    void* x                     = nullptr;
    void* out16                 = nullptr;
    void* out8                  = nullptr;
    const std::size_t out_bytes = std::size_t(N) * Tmax * 2;
    if (cudaMalloc(&payload, host.payload.size()) != cudaSuccess ||
        cudaMalloc(&x, std::size_t(K) * Tmax * 2) != cudaSuccess ||
        cudaMalloc(&out16, out_bytes) != cudaSuccess ||
        cudaMalloc(&out8, out_bytes) != cudaSuccess) {
        std::printf("FAIL: allocation\n");
        return 1;
    }
    cudaMemcpy(payload, host.payload.data(), host.payload.size(), cudaMemcpyHostToDevice);
    const Weight weight = host.device_weight(payload);
    std::vector<std::uint16_t> xb(std::size_t(K) * Tmax);
    std::uint32_t state = 12345U;
    for (auto& v : xb) {
        state   = state * 1664525U + 1013904223U;
        float f = static_cast<float>(static_cast<std::int32_t>(state >> 8) - (1 << 23)) / (1 << 23);
        std::uint32_t bits;
        std::memcpy(&bits, &f, 4);
        v = static_cast<std::uint16_t>(bits >> 16);
    }
    cudaMemcpy(x, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice);
    int failures = 0;
    std::vector<std::uint16_t> a(std::size_t(N) * Tmax), b(std::size_t(N) * Tmax);
    for (std::int32_t t = 2; t <= Tmax; ++t) {
        const ops::detail::Q6Launch route = ops::detail::select_q6_n124160_k2560(t);
        if (route != ops::detail::launch_q6_a16_sliced_r32_t8_w4_s2) {
            std::printf("FAIL: T=%d is not routed to the 32x8 tile\n", t);
            ++failures;
        }
        Tensor xt(x, DType::BF16, {K, t});
        Tensor o16(out16, DType::BF16, {N, t});
        Tensor o8(out8, DType::BF16, {N, t});
        cudaMemset(out16, 0x00, out_bytes);
        cudaMemset(out8, 0xff, out_bytes);
        ops::detail::launch_q6_a16_sliced_r32_t16_w4_s2(xt, weight, o16, nullptr);
        route(xt, weight, o8, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::printf("FAIL: T=%d launch\n", t);
            return 1;
        }
        const std::size_t n = std::size_t(N) * t;
        cudaMemcpy(a.data(), out16, n * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), out8, n * 2, cudaMemcpyDeviceToHost);
        if (std::memcmp(a.data(), b.data(), n * 2) != 0) {
            std::printf("FAIL: T=%d the selected route differs from the 32x16 route\n", t);
            ++failures;
        }
    }
    cudaFree(payload);
    cudaFree(x);
    cudaFree(out16);
    cudaFree(out8);
    std::printf(failures == 0 ? "PASS\n" : "FAILED\n");
    return failures == 0 ? 0 : 1;
}
