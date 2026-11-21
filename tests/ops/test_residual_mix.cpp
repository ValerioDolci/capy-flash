// residual_mix.h against the Ops it fuses, bit for bit: short_conv_static(_record) +
// residual_add_f32 + depth_attention + rmsnorm_f32 (residual_conv_mix), depth_attention +
// rmsnorm_f32 (depth_attention_rmsnorm), short_conv_static(_record) + residual_add_round_bf16
// (residual_conv_round). Every source count 1..9, Q 1..4, T 1..16, the state advanced in place
// (T = 1) and recorded (any T), a width that is a multiple of the CTA and one that is not.
#include "ninfer/ops/depth_attention.h"
#include "ninfer/ops/residual_f32.h"
#include "ninfer/ops/residual_mix.h"
#include "ninfer/ops/rmsnorm_f32.h"
#include "ninfer/ops/short_conv.h"
#include "ninfer/ops/short_conv_record.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace ninfer;

std::uint32_t g_seed = 12345U;

float next_uniform() {
    g_seed = g_seed * 1664525U + 1013904223U;
    return static_cast<float>(g_seed >> 8) / 16777216.0F * 2.0F - 1.0F;
}

std::uint16_t to_bf16(float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    u += 0x7FFFU + ((u >> 16) & 1U);
    return static_cast<std::uint16_t>(u >> 16);
}

// A device buffer filled with random BF16 or FP32 values times `scale`.
struct Buffer {
    void* p           = nullptr;
    std::size_t bytes = 0;

    Buffer(std::size_t count, DType dtype, float scale) {
        bytes = count * (dtype == DType::FP32 ? 4 : 2);
        cudaMalloc(&p, bytes);
        if (dtype == DType::FP32) {
            std::vector<float> h(count);
            for (auto& v : h) v = next_uniform() * scale;
            cudaMemcpy(p, h.data(), bytes, cudaMemcpyHostToDevice);
        } else {
            std::vector<std::uint16_t> h(count);
            for (auto& v : h) v = to_bf16(next_uniform() * scale);
            cudaMemcpy(p, h.data(), bytes, cudaMemcpyHostToDevice);
        }
    }

    ~Buffer() { cudaFree(p); }

    Buffer(const Buffer&)            = delete;
    Buffer& operator=(const Buffer&) = delete;

    void copy_from(const Buffer& other) const {
        cudaMemcpy(p, other.p, bytes, cudaMemcpyDeviceToDevice);
    }

    std::vector<unsigned char> host() const {
        std::vector<unsigned char> h(bytes);
        cudaDeviceSynchronize();
        cudaMemcpy(h.data(), p, bytes, cudaMemcpyDeviceToHost);
        return h;
    }
};

using BufferPtr = std::unique_ptr<Buffer>;

BufferPtr make(std::size_t count, DType dtype, float scale = 1.0F) {
    return std::make_unique<Buffer>(count, dtype, scale);
}

Tensor view(const BufferPtr& b, DType dtype, std::initializer_list<std::int32_t> ne) {
    return Tensor(b->p, dtype, ne);
}

int failures = 0;

void expect_same(const Buffer& a, const Buffer& b, const std::string& what) {
    if (a.host() != b.host()) {
        std::printf("FAIL %s: bits differ\n", what.c_str());
        ++failures;
    }
}

// One case of each fused Op against the separate Ops on the same operands.
void run_case(int H, int T, int stored, int Q, bool record, bool with_base, bool norms_on) {
    const std::string tag = "H=" + std::to_string(H) + " T=" + std::to_string(T) +
                            " n=" + std::to_string(stored) + " Q=" + std::to_string(Q) +
                            (record ? " record" : " state") + (with_base ? " base" : "") +
                            (norms_on ? " norms" : " mix");
    const std::size_t HT = static_cast<std::size_t>(H) * T;
    auto y               = make(HT, DType::BF16, 2.0F);
    auto conv_w          = make(static_cast<std::size_t>(H) * 4, DType::FP32, 0.5F);
    auto state0          = make(static_cast<std::size_t>(H) * 3, DType::BF16, 2.0F);
    auto base            = make(HT, DType::BF16, 4.0F);
    auto query           = make(static_cast<std::size_t>(H) * Q, DType::FP32, 0.05F);
    std::vector<BufferPtr> states;
    for (int s = 0; s < stored; ++s) states.push_back(make(HT, DType::BF16, 3.0F));
    std::vector<BufferPtr> norm_w;
    for (int q = 0; q < Q; ++q) norm_w.push_back(make(H, DType::BF16, 1.5F));
    const float conv_scale = 0.4472136F, mix_eps = 1e-6F, mix_scale = 0.0208333F, norm_eps = 1e-6F;
    std::vector<Tensor> st;
    for (auto& s : states) st.push_back(view(s, DType::BF16, {H, T}));
    std::vector<Tensor> nw;
    for (auto& w : norm_w) nw.push_back(view(w, DType::BF16, {H}));

    // Reference: separate Ops.
    auto ref_state = make(static_cast<std::size_t>(H) * 3, DType::BF16);
    ref_state->copy_from(*state0);
    auto ref_record = make(static_cast<std::size_t>(H) * (T + 3), DType::BF16);
    auto yc         = make(HT, DType::BF16);
    auto ref_res    = make(HT, DType::FP32);
    auto ref_mix    = make(HT * Q, DType::FP32);
    std::vector<BufferPtr> ref_normed;
    for (int q = 0; q < Q; ++q) ref_normed.push_back(make(HT, DType::BF16));
    {
        Tensor x = view(y, DType::BF16, {H, T}), w = view(conv_w, DType::FP32, {4, H});
        Tensor s = view(ref_state, DType::BF16, {H, 3}), out = view(yc, DType::BF16, {H, T});
        if (record) {
            Tensor rec = view(ref_record, DType::BF16, {H, T + 3});
            ops::short_conv_static_record(x, w, conv_scale, s, rec, out, nullptr);
        } else {
            ops::short_conv_static(x, w, conv_scale, s, s, out, nullptr);
        }
        Tensor b = view(base, DType::BF16, {H, T}), r = view(ref_res, DType::FP32, {H, T});
        ops::residual_add_f32(with_base ? &b : nullptr, out, r, nullptr);
        Tensor q = view(query, DType::FP32, {H, Q}), m = view(ref_mix, DType::FP32, {H, T, Q});
        ops::depth_attention(st, r, q, mix_eps, mix_scale, m, nullptr);
        for (int j = 0; j < Q; ++j) {
            Tensor o = view(ref_normed[j], DType::BF16, {H, T});
            ops::rmsnorm_f32(m.slice(2, j, 1).view({H, T}), nw[j], norm_eps, o, nullptr);
        }
    }
    // Fused residual_conv_mix.
    auto f_state = make(static_cast<std::size_t>(H) * 3, DType::BF16);
    f_state->copy_from(*state0);
    auto f_record = make(static_cast<std::size_t>(H) * (T + 3), DType::BF16);
    auto f_res    = make(HT, DType::FP32);
    auto f_mix    = make(HT * Q, DType::FP32);
    std::vector<BufferPtr> f_normed;
    for (int q = 0; q < Q; ++q) f_normed.push_back(make(HT, DType::BF16));
    {
        ops::ResidualConv conv{view(y, DType::BF16, {H, T}), view(conv_w, DType::FP32, {4, H}),
                               conv_scale, view(f_state, DType::BF16, {H, 3}),
                               record ? view(f_record, DType::BF16, {H, T + 3}) : Tensor{}};
        Tensor b = view(base, DType::BF16, {H, T}), r = view(f_res, DType::FP32, {H, T});
        Tensor q = view(query, DType::FP32, {H, Q}), m = view(f_mix, DType::FP32, {H, T, Q});
        std::vector<Tensor> outs;
        for (auto& o : f_normed) outs.push_back(view(o, DType::BF16, {H, T}));
        ops::residual_conv_mix(conv, with_base ? &b : nullptr, st, q, mix_eps, mix_scale,
                               norms_on ? std::span<const Tensor>(nw) : std::span<const Tensor>(),
                               norm_eps, r, norms_on && Q == 1 ? nullptr : &m,
                               norms_on ? std::span<Tensor>(outs) : std::span<Tensor>(), nullptr);
    }
    expect_same(*ref_res, *f_res, "residual_conv_mix residual " + tag);
    if (record) {
        expect_same(*ref_record, *f_record, "residual_conv_mix record " + tag);
        expect_same(*state0, *f_state, "residual_conv_mix state untouched " + tag);
    } else {
        expect_same(*ref_state, *f_state, "residual_conv_mix state " + tag);
    }
    if (!(norms_on && Q == 1)) expect_same(*ref_mix, *f_mix, "residual_conv_mix mix " + tag);
    if (norms_on) {
        for (int q = 0; q < Q; ++q) {
            expect_same(*ref_normed[q], *f_normed[q], "residual_conv_mix normed " + tag);
        }
    }

    // depth_attention_rmsnorm over the BF16 `base` as current.
    {
        auto r_mix = make(HT * Q, DType::FP32);
        Tensor cur = view(base, DType::BF16, {H, T}), q = view(query, DType::FP32, {H, Q});
        Tensor m = view(r_mix, DType::FP32, {H, T, Q});
        ops::depth_attention(st, cur, q, mix_eps, mix_scale, m, nullptr);
        std::vector<BufferPtr> r_out, d_out;
        std::vector<Tensor> outs;
        for (int j = 0; j < Q; ++j) {
            r_out.push_back(make(HT, DType::BF16));
            d_out.push_back(make(HT, DType::BF16));
            Tensor o = view(r_out[j], DType::BF16, {H, T});
            ops::rmsnorm_f32(m.slice(2, j, 1).view({H, T}), nw[j], norm_eps, o, nullptr);
            outs.push_back(view(d_out[j], DType::BF16, {H, T}));
        }
        ops::depth_attention_rmsnorm(st, cur, q, mix_eps, mix_scale, nw, norm_eps, outs, nullptr);
        for (int j = 0; j < Q; ++j) {
            expect_same(*r_out[j], *d_out[j], "depth_attention_rmsnorm " + tag);
        }
    }

    // residual_conv_round over the FP32 residual of the reference.
    {
        auto r_state = make(static_cast<std::size_t>(H) * 3, DType::BF16);
        auto g_state = make(static_cast<std::size_t>(H) * 3, DType::BF16);
        r_state->copy_from(*state0);
        g_state->copy_from(*state0);
        auto r_rec = make(static_cast<std::size_t>(H) * (T + 3), DType::BF16);
        auto g_rec = make(static_cast<std::size_t>(H) * (T + 3), DType::BF16);
        auto zc = make(HT, DType::BF16), r_out = make(HT, DType::BF16),
             g_out = make(HT, DType::BF16);
        Tensor x = view(y, DType::BF16, {H, T}), w = view(conv_w, DType::FP32, {4, H});
        Tensor s = view(r_state, DType::BF16, {H, 3}), z = view(zc, DType::BF16, {H, T});
        if (record) {
            Tensor rec = view(r_rec, DType::BF16, {H, T + 3});
            ops::short_conv_static_record(x, w, conv_scale, s, rec, z, nullptr);
        } else {
            ops::short_conv_static(x, w, conv_scale, s, s, z, nullptr);
        }
        Tensor res = view(ref_res, DType::FP32, {H, T}), ro = view(r_out, DType::BF16, {H, T});
        ops::residual_add_round_bf16(res, z, ro, nullptr);
        ops::ResidualConv conv{x, w, conv_scale, view(g_state, DType::BF16, {H, 3}),
                               record ? view(g_rec, DType::BF16, {H, T + 3}) : Tensor{}};
        Tensor go = view(g_out, DType::BF16, {H, T});
        ops::residual_conv_round(conv, res, go, nullptr);
        expect_same(*r_out, *g_out, "residual_conv_round out " + tag);
        if (record) {
            expect_same(*r_rec, *g_rec, "residual_conv_round record " + tag);
        } else {
            expect_same(*r_state, *g_state, "residual_conv_round state " + tag);
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("FAIL %s: CUDA error\n", tag.c_str());
        ++failures;
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
        for (const int H : {2304, 1000}) {
            for (int n = 0; n <= ops::kDepthAttentionMaxStates; ++n) {
                run_case(H, 1, n, 1, false, n > 0, true);
                run_case(H, 1, n, 3, false, true, true);
                run_case(H, 4, n, 1, true, true, true);
            }
            for (const int T : {1, 2, 5, 8, 16}) {
                run_case(H, T, 3, 1, true, true, false);
                run_case(H, T, 4, 4, true, false, true);
                run_case(H, T, 2, 3, true, true, true);
            }
            run_case(H, 1, 5, 1, false, true, false);
        }
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
    std::printf("%s residual_mix\n", failures == 0 ? "OK" : "FAIL");
    return failures == 0 ? 0 : 1;
}
