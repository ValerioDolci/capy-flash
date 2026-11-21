// message_gates_rmsnorm against message_gate + rmsnorm_f32 per gate, bit for bit: 1-4 gates (three
// of them reading the same hidden, as a local visit's gates do), T across the tile sizes, the
// in-place gate (out = hidden), and operands 4 bytes off a 16-byte boundary (the per-gate
// fallback).
#include "ninfer/ops/message_gate.h"
#include "ninfer/ops/rmsnorm_f32.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace ninfer;

std::uint32_t g_seed = 777U;

float uniform() {
    g_seed = g_seed * 1664525U + 1013904223U;
    return static_cast<float>(g_seed >> 8) / 16777216.0F * 2.0F - 1.0F;
}

std::uint16_t to_bf16(float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    u += 0x7FFFU + ((u >> 16) & 1U);
    return static_cast<std::uint16_t>(u >> 16);
}

struct Buffer {
    void* p           = nullptr;
    std::size_t bytes = 0;

    Buffer(std::size_t count, bool fp32, float scale) : bytes(count * (fp32 ? 4 : 2) + 16) {
        cudaMalloc(&p, bytes);
        std::vector<unsigned char> h(bytes);
        for (std::size_t i = 0; i + (fp32 ? 4 : 2) <= bytes; i += fp32 ? 4 : 2) {
            if (fp32) {
                const float v = uniform() * scale;
                std::memcpy(&h[i], &v, 4);
            } else {
                const std::uint16_t v = to_bf16(uniform() * scale);
                std::memcpy(&h[i], &v, 2);
            }
        }
        cudaMemcpy(p, h.data(), bytes, cudaMemcpyHostToDevice);
    }

    ~Buffer() { cudaFree(p); }

    Buffer(const Buffer&)            = delete;
    Buffer& operator=(const Buffer&) = delete;

    std::vector<unsigned char> host() const {
        std::vector<unsigned char> h(bytes);
        cudaDeviceSynchronize();
        cudaMemcpy(h.data(), p, bytes, cudaMemcpyDeviceToHost);
        return h;
    }

    void copy_from(const Buffer& o) const { cudaMemcpy(p, o.p, bytes, cudaMemcpyDeviceToDevice); }
};

using Ptr = std::unique_ptr<Buffer>;

int failures = 0;

bool same(const Buffer& a, const Buffer& b, std::size_t offset, std::size_t bytes) {
    const auto x = a.host(), y = b.host();
    return std::memcmp(x.data() + offset, y.data() + offset, bytes) == 0;
}

Tensor at(const Ptr& b, int shift, DType dtype, std::initializer_list<std::int32_t> ne) {
    return Tensor(static_cast<unsigned char*>(b->p) + shift, dtype, ne);
}

void run(int gates, int T, bool shared_hidden, bool in_place, int shift) {
    constexpr int H = 2304, R = 64;
    const std::string tag = "G=" + std::to_string(gates) + " T=" + std::to_string(T) +
                            (shared_hidden ? " shared" : "") + (in_place ? " in-place" : "") +
                            (shift ? " shifted" : "");
    const std::size_t HT = static_cast<std::size_t>(H) * T;
    std::vector<Ptr> hidden, message, down, forget, gain, norm;
    for (int g = 0; g < gates; ++g) {
        hidden.push_back(std::make_unique<Buffer>(HT, true, 2.0F));
        message.push_back(std::make_unique<Buffer>(HT, true, 3.0F));
        down.push_back(std::make_unique<Buffer>(static_cast<std::size_t>(2) * H * R, false, 0.05F));
        forget.push_back(std::make_unique<Buffer>(static_cast<std::size_t>(H) * R, true, 0.2F));
        gain.push_back(std::make_unique<Buffer>(static_cast<std::size_t>(H) * R, true, 0.2F));
        norm.push_back(std::make_unique<Buffer>(H, false, 1.5F));
    }
    // Reference and fused outputs (FP32 gate outputs, BF16 norms); in place, out is a copy of
    // hidden.
    std::vector<Ptr> ref_out, out, ref_norm, out_norm;
    for (int g = 0; g < gates; ++g) {
        ref_out.push_back(std::make_unique<Buffer>(HT, true, 1.0F));
        out.push_back(std::make_unique<Buffer>(HT, true, 1.0F));
        if (in_place) {
            ref_out[g]->copy_from(*hidden[g]);
            out[g]->copy_from(*hidden[g]);
        }
        ref_norm.push_back(std::make_unique<Buffer>(HT, false, 1.0F));
        out_norm.push_back(std::make_unique<Buffer>(HT, false, 1.0F));
    }
    const auto hidden_of = [&](int g, const std::vector<Ptr>& outs) {
        if (in_place) return at(outs[g], shift, DType::FP32, {H, T});
        return at(hidden[shared_hidden ? 0 : g], shift, DType::FP32, {H, T});
    };
    const float eps = 1e-6F;
    for (int g = 0; g < gates; ++g) {
        Tensor o = at(ref_out[g], shift, DType::FP32, {H, T});
        Tensor n = at(ref_norm[g], 0, DType::BF16, {H, T});
        ops::message_gate(hidden_of(g, ref_out), at(message[g], shift, DType::FP32, {H, T}),
                          at(down[g], 0, DType::BF16, {2 * H, R}),
                          at(forget[g], 0, DType::FP32, {R, H}),
                          at(gain[g], 0, DType::FP32, {R, H}), eps, o, nullptr);
        ops::rmsnorm_f32(o, at(norm[g], 0, DType::BF16, {H}), eps, n, nullptr);
    }
    std::vector<ops::MessageGateNormed> batch;
    for (int g = 0; g < gates; ++g) {
        batch.push_back({hidden_of(g, out), at(message[g], shift, DType::FP32, {H, T}),
                         at(down[g], 0, DType::BF16, {2 * H, R}),
                         at(forget[g], 0, DType::FP32, {R, H}), at(gain[g], 0, DType::FP32, {R, H}),
                         at(out[g], shift, DType::FP32, {H, T}), at(norm[g], 0, DType::BF16, {H}),
                         at(out_norm[g], 0, DType::BF16, {H, T})});
    }
    ops::message_gates_rmsnorm(batch, eps, eps, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("FAIL %s: CUDA error\n", tag.c_str());
        ++failures;
        return;
    }
    for (int g = 0; g < gates; ++g) {
        if (!same(*ref_out[g], *out[g], shift, HT * 4)) {
            std::printf("FAIL %s gate %d: gate output bits differ\n", tag.c_str(), g);
            ++failures;
        }
        if (!same(*ref_norm[g], *out_norm[g], 0, HT * 2)) {
            std::printf("FAIL %s gate %d: norm bits differ\n", tag.c_str(), g);
            ++failures;
        }
    }
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    try {
        for (const int T : {1, 2, 3, 4, 5, 6, 8, 9, 16, 17, 40}) {
            run(3, T, true, false, 0);
            run(3, T, false, false, 0);
            run(1, T, false, true, 0);
        }
        run(4, 8, false, false, 0);
        run(2, 5, false, false, 0);
        run(3, 4, true, false, 4);
        run(1, 1, false, true, 4);
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
    std::printf("%s message_gates_rmsnorm\n", failures == 0 ? "OK" : "FAIL");
    return failures == 0 ? 0 : 1;
}
