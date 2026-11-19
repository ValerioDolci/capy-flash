// Host MoE executor on GGUF layers (HostExpertLayer::format == Gguf) against an FP64 oracle.
//
// Layers: synthetic random ggml blocks for every gate/up x down pair of the Flash-Next banks (8
// experts each), and, with NINFER_GGUF_BANKS (tools/convert/gguf_experts.py extract), the real
// banks of the extracted layers (512 experts).
// Oracle: ggml dequantization of the weights; x quantized with the executor's codec (per 32 values
// d = amax / 127, q = roundf(v / d)); gate/up in FP64 from the dequantized codes; h = silu(g) * u
// in FP64, quantized with the same codec; out = sum alpha * W_down . h in FP64.
// Criteria:
//  - |out - oracle| <= 4e-3 * rms(oracle) per token (the executor's FP32 sums and its ~2-ulp exp
//    can move an h code across a rounding boundary; one such flip changes an output by at most
//    d_h * |w_down|, far inside this bound);
//  - determinism: identical bits for 1, 3 and 6 threads, and a token's output in a T = 8 call
//    equals the same token's output in a call of its own (T independence); identical bits with
//    Options::gguf_only;
//  - tokens without a host expert (all ids < 0) get 0.
#include "ops/host_moe/host_moe_executor.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace h = ninfer::ops::host;
namespace g = ninfer::ops::host::gguf;

namespace {

constexpr int H = 2560, I = 640, K = 10;
int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

std::uint16_t bf16(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<std::uint16_t>((u + 0x7FFFU + ((u >> 16) & 1U)) >> 16);
}

float from_bf16(std::uint16_t b) {
    const std::uint32_t u = std::uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

void dequantize(g::WeightFormat fmt, const void* row, float* y, int k) {
    switch (fmt) {
    case g::WeightFormat::IQ2_S:
        dequantize_row_iq2_s(static_cast<const block_iq2_s*>(row), y, k);
        return;
    case g::WeightFormat::IQ3_XXS:
        dequantize_row_iq3_xxs(static_cast<const block_iq3_xxs*>(row), y, k);
        return;
    case g::WeightFormat::IQ3_S:
        dequantize_row_iq3_s(static_cast<const block_iq3_s*>(row), y, k);
        return;
    case g::WeightFormat::IQ4_XS:
        dequantize_row_iq4_xs(static_cast<const block_iq4_xs*>(row), y, k);
        return;
    case g::WeightFormat::IQ4_NL:
        dequantize_row_iq4_nl(static_cast<const block_iq4_nl*>(row), y, k);
        return;
    case g::WeightFormat::Q2_0:
        dequantize_row_q2_0(static_cast<const block_q2_0*>(row), y, k);
        return;
    }
}

// The executor's codec on doubles holding float values: per 32, d = fl(amax / 127), q = round(v /
// d).
void q8f32(const std::vector<double>& v, std::vector<double>& out) {
    out.assign(v.size(), 0.0);
    for (std::size_t b = 0; b < v.size(); b += 32) {
        float amax = 0;
        for (std::size_t i = 0; i < 32; ++i)
            amax = std::max(amax, std::fabs(static_cast<float>(v[b + i])));
        if (amax == 0) continue;
        const float d = amax / 127.0F;
        for (std::size_t i = 0; i < 32; ++i) {
            out[b + i] = static_cast<double>(std::round(static_cast<float>(v[b + i]) / d)) * d;
        }
    }
}

struct Layer {
    std::string name;
    int experts = 0;
    h::HostExpertLayer layer;
    std::vector<std::uint8_t> gu, dn;
};

Layer synthetic(g::WeightFormat gu, g::WeightFormat dn, int experts, std::mt19937_64& rng) {
    Layer L;
    L.name    = std::string("synthetic ") + g::format_name(gu) + "+" + g::format_name(dn);
    L.experts = experts;
    const std::size_t rbg = g::weight_row_bytes(gu, H), rbd = g::weight_row_bytes(dn, I);
    L.gu.resize(rbg * 2 * I * experts);
    L.dn.resize(rbd * H * experts);
    std::uniform_real_distribution<float> dd(5e-4F, 2e-2F);
    const auto fill = [&](std::vector<std::uint8_t>& v, g::WeightFormat f, std::size_t rb, int k,
                          std::size_t rows) {
        for (auto& b : v) b = static_cast<std::uint8_t>(rng());
        const auto fi = g::format_info(f);
        for (std::size_t r = 0; r < rows; ++r)
            for (int b = 0; b < k / fi.block_elems; ++b) {
                const std::uint16_t hv = GGML_FP32_TO_FP16(dd(rng));
                std::memcpy(v.data() + r * rb + std::size_t(b) * fi.block_bytes, &hv, 2);
            }
    };
    fill(L.gu, gu, rbg, H, std::size_t(2) * I * experts);
    fill(L.dn, dn, rbd, I, std::size_t(H) * experts);
    return L;
}

bool format_of(const std::string& n, g::WeightFormat* f) {
    static const std::pair<const char*, g::WeightFormat> m[] = {
        {"gguf_iq2_s", g::WeightFormat::IQ2_S},   {"gguf_iq3_xxs", g::WeightFormat::IQ3_XXS},
        {"gguf_iq3_s", g::WeightFormat::IQ3_S},   {"gguf_iq4_xs", g::WeightFormat::IQ4_XS},
        {"gguf_iq4_nl", g::WeightFormat::IQ4_NL}, {"gguf_q2_0", g::WeightFormat::Q2_0}};
    for (const auto& [k, v] : m)
        if (n == k) {
            *f = v;
            return true;
        }
    return false;
}

void finish(Layer& L, g::WeightFormat gu, g::WeightFormat dn) {
    L.layer.format                 = h::HostExpertFormat::Gguf;
    L.layer.gguf.gate_up           = gu;
    L.layer.gguf.down              = dn;
    L.layer.gguf.gate_up_rows      = L.gu.data();
    L.layer.gguf.down_rows         = L.dn.data();
    L.layer.gguf.gate_up_row_bytes = g::weight_row_bytes(gu, H);
    L.layer.gguf.down_row_bytes    = g::weight_row_bytes(dn, I);
    L.layer.experts                = L.experts;
    L.layer.hidden                 = H;
    L.layer.intermediate           = I;
}

std::vector<double> oracle(const Layer& L, const std::vector<std::uint16_t>& x,
                           const std::vector<std::int32_t>& ids, const std::vector<float>& alpha,
                           int T) {
    const auto& G = L.layer.gguf;
    std::vector<double> out(std::size_t(T) * H, 0.0), xq, hq;
    std::vector<float> w(H);
    for (int t = 0; t < T; ++t) {
        std::vector<double> xv(H);
        for (int i = 0; i < H; ++i) xv[i] = from_bf16(x[std::size_t(t) * H + i]);
        q8f32(xv, xq);
        // Experts in ascending id (the executor's order; FP64 makes it irrelevant here).
        for (int k = 0; k < K; ++k) {
            const int e = ids[std::size_t(t) * K + k];
            if (e < 0) continue;
            std::vector<double> hv(I);
            for (int r = 0; r < I; ++r) {
                double gs = 0, us = 0;
                dequantize(G.gate_up,
                           G.gate_up_rows + (std::size_t(e) * 2 * I + r) * G.gate_up_row_bytes,
                           w.data(), H);
                for (int i = 0; i < H; ++i) gs += double(w[i]) * xq[i];
                dequantize(G.gate_up,
                           G.gate_up_rows + (std::size_t(e) * 2 * I + I + r) * G.gate_up_row_bytes,
                           w.data(), H);
                for (int i = 0; i < H; ++i) us += double(w[i]) * xq[i];
                hv[r] = static_cast<float>(gs / (1.0 + std::exp(-gs)) * us);
            }
            q8f32(hv, hq);
            for (int r = 0; r < H; ++r) {
                dequantize(G.down, G.down_rows + (std::size_t(e) * H + r) * G.down_row_bytes,
                           w.data(), I);
                double s = 0;
                for (int i = 0; i < I; ++i) s += double(w[i]) * hq[i];
                out[std::size_t(t) * H + r] += alpha[std::size_t(t) * K + k] * s;
            }
        }
    }
    return out;
}

void run_layer(const Layer& L, std::mt19937_64& rng, int Tmax) {
    std::normal_distribution<float> nd(0.F, 1.F);
    std::uniform_int_distribution<int> pick(0, L.experts - 1);
    std::vector<std::uint16_t> x(std::size_t(Tmax) * H);
    for (auto& v : x) v = bf16(nd(rng) * 0.7F);
    std::vector<std::int32_t> ids(std::size_t(Tmax) * K, -1);
    std::vector<float> alpha(std::size_t(Tmax) * K, 0.F);
    for (int t = 0; t < Tmax; ++t) {
        if (t == 2) continue; // a token without host experts
        const int n = 1 + t % 4;
        std::vector<int> used;
        for (int k = 0; k < n; ++k) {
            int e;
            do e = pick(rng);
            while (std::find(used.begin(), used.end(), e) != used.end() &&
                   int(used.size()) < L.experts);
            used.push_back(e);
            ids[std::size_t(t) * K + k]   = e;
            alpha[std::size_t(t) * K + k] = 0.05F + 0.1F * k;
        }
    }
    std::vector<std::vector<float>> outs;
    for (int threads : {1, 3, 6}) {
        const auto cpus = h::spread_cpus(threads);
        h::HostCpuPool pool(cpus);
        h::HostMoeExecutor ex(pool, H, I, std::max(Tmax, 8), K, h::HostMoeExecutor::Options{});
        std::vector<float> out(std::size_t(Tmax) * H, 123.F);
        ex.run(L.layer, h::HostMoeJob{Tmax, K, x.data(), ids.data(), alpha.data(), out.data()});
        outs.push_back(out);
        if (threads == 3) {
            // T independence: each token alone.
            for (int t = 0; t < std::min(Tmax, 8); ++t) {
                std::vector<float> one(H, 7.F);
                ex.run(L.layer, h::HostMoeJob{1, K, x.data() + std::size_t(t) * H,
                                              ids.data() + std::size_t(t) * K,
                                              alpha.data() + std::size_t(t) * K, one.data()});
                check(std::memcmp(one.data(), out.data() + std::size_t(t) * H, sizeof(float) * H) ==
                          0,
                      L.name + ": token result depends on the call width");
            }
        }
    }
    check(outs[0] == outs[1] && outs[1] == outs[2],
          L.name + ": result depends on the thread count");
    {
        // Big-core decoder and the E-core tail (on a hybrid CPU 12 threads mix P-
        // and E-cores; elsewhere both options leave the plain path): the same bits.
        h::HostCpuPool pool(h::spread_cpus(12));
        h::HostMoeExecutor::Options o{};
        o.gguf_big_gather = true;
        o.gguf_ecore_tail = 8;
        h::HostMoeExecutor ex(pool, H, I, std::max(Tmax, 8), K, o);
        std::vector<float> out(std::size_t(Tmax) * H, 123.F);
        ex.run(L.layer, h::HostMoeJob{Tmax, K, x.data(), ids.data(), alpha.data(), out.data()});
        check(out == outs[0], L.name + ": big-core decoder / E-core tail change the result");
    }
    {
        // Chunk-sized gate/up items (calls of at most 8 tokens) on one executor across calls of
        // 8, 1, 3 tokens, a wide call (slice items) and 8 again, with and without the big-core
        // decoder and the small-core tail: the same bits as the slice items (a token's result
        // does not depend on the call width, checked above).
        for (const bool big : {false, true}) {
            h::HostCpuPool pool(h::spread_cpus(big ? 12 : 3));
            h::HostMoeExecutor::Options o{};
            o.gguf_fine       = true;
            o.gguf_big_gather = big;
            o.gguf_ecore_tail = big ? 24 : 0;
            h::HostMoeExecutor ex(pool, H, I, std::max(Tmax, 8), K, o);
            for (const int Tc : {8, 1, 3, Tmax, 8}) {
                const int Tf = std::min(Tc, Tmax);
                std::vector<float> out(std::size_t(Tf) * H, 123.F);
                ex.run(L.layer,
                       h::HostMoeJob{Tf, K, x.data(), ids.data(), alpha.data(), out.data()});
                check(std::memcmp(out.data(), outs[0].data(), sizeof(float) * out.size()) == 0,
                      L.name + ": chunk-sized gate/up items change the result (T=" +
                          std::to_string(Tf) + ")");
            }
            check(ex.gguf_fine_calls() > 0, L.name + ": the chunk-sized gate/up items never ran");
        }
    }
    {
        // A GGUF-only executor (no NVFP4 activation buffers) computes the same bits.
        h::HostCpuPool pool(h::spread_cpus(3));
        h::HostMoeExecutor::Options options;
        options.gguf_only = true;
        h::HostMoeExecutor ex(pool, H, I, std::max(Tmax, 8), K, options);
        std::vector<float> out(std::size_t(Tmax) * H, 123.F);
        ex.run(L.layer, h::HostMoeJob{Tmax, K, x.data(), ids.data(), alpha.data(), out.data()});
        check(std::memcmp(out.data(), outs[1].data(), sizeof(float) * out.size()) == 0,
              L.name + ": a GGUF-only executor changes the result");
    }
    const int To   = std::min(Tmax, 8);
    const auto ref = oracle(L, x, ids, alpha, To);
    double worst   = 0;
    for (int t = 0; t < To; ++t) {
        double rms = 0, err = 0;
        for (int r = 0; r < H; ++r) {
            const double o = ref[std::size_t(t) * H + r];
            rms += o * o;
            err = std::max(err, std::fabs(outs[0][std::size_t(t) * H + r] - o));
        }
        rms = std::sqrt(rms / H);
        if (rms == 0) {
            check(err == 0, L.name + ": token without host experts not zero");
            continue;
        }
        worst = std::max(worst, err / rms);
    }
    check(worst <= 4e-3, L.name + ": error vs FP64 oracle " + std::to_string(worst) + " rms");
    std::printf("%-34s T=%d  max|err|/rms %.2e  threads 1/3/6 identical %s\n", L.name.c_str(), Tmax,
                worst, outs[0] == outs[1] && outs[1] == outs[2] ? "yes" : "NO");
}

} // namespace

int main(int argc, char** argv) {
    if (!h::host_experts_supported()) {
        std::printf("SKIP: CPU runs neither ISA of the host expert kernels (AVX2 + FMA + F16C)\n");
        return 77;
    }
    std::printf("GGUF host kernels: %s (NINFER_HOST_ISA selects)\n",
                h::host_isa_name(g::host_gguf_isa()));
    std::mt19937_64 rng(77);
    const std::pair<g::WeightFormat, g::WeightFormat> pairs[] = {
        {g::WeightFormat::IQ2_S, g::WeightFormat::Q2_0},
        {g::WeightFormat::IQ2_S, g::WeightFormat::IQ4_NL},
        {g::WeightFormat::IQ3_XXS, g::WeightFormat::IQ4_NL},
        {g::WeightFormat::IQ3_S, g::WeightFormat::Q2_0},
        {g::WeightFormat::IQ4_XS, g::WeightFormat::IQ4_NL},
        {g::WeightFormat::IQ4_NL, g::WeightFormat::IQ4_NL}};
    for (const auto& [gu, dn] : pairs) {
        Layer L = synthetic(gu, dn, 8, rng);
        finish(L, gu, dn);
        run_layer(L, rng, 8);
    }
    {
        Layer L = synthetic(g::WeightFormat::IQ3_XXS, g::WeightFormat::IQ4_NL, 8, rng);
        finish(L, g::WeightFormat::IQ3_XXS, g::WeightFormat::IQ4_NL);
        run_layer(L, rng, 40); // a prefill-width call
    }
    const char* dir = argc > 1 ? argv[1] : std::getenv("NINFER_GGUF_BANKS");
    if (dir && *dir) {
        std::ifstream in(std::string(dir) + "/manifest.json");
        if (in) {
            const auto m = nlohmann::json::parse(in);
            for (const auto& [layer, e] : m.at("layers").items()) {
                g::WeightFormat gu, dn;
                if (!format_of(e.at("gate_up").at("format"), &gu) ||
                    !format_of(e.at("down").at("format"), &dn))
                    continue;
                Layer L;
                L.name    = "layer " + layer + " " + g::format_name(gu) + "+" + g::format_name(dn);
                L.experts = 512;
                for (auto [role, buf] : {std::pair{"gate_up", &L.gu}, std::pair{"down", &L.dn}}) {
                    std::ifstream f(std::string(dir) + "/" +
                                        e.at(role).at("file").get<std::string>(),
                                    std::ios::binary);
                    buf->assign(std::istreambuf_iterator<char>(f), {});
                }
                finish(L, gu, dn);
                run_layer(L, rng, 8);
            }
        }
    }
    if (failures) {
        std::printf("FAILED: %d\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
