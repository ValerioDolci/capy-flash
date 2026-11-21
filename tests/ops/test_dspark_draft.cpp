// FP64 oracles of the DSpark drafter Ops (ninfer/ops/dspark_draft.h) at small shapes and at the
// rank shapes of Inkling 2 Mini's drafter (H 2304, 12/4 and 24/8 heads of 128, window 1024, rank
// 256, vocabulary halves of 100029 rows). The oracles evaluate each contract's formula in FP64 from
// the represented inputs, rounding to BF16 exactly where the contract names a BF16 value.
#include "ninfer/ops/dspark_draft.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int D = 128;
// One BF16 rounding of an FP32 evaluation (the final store).
constexpr PointwiseCriterion kRounded{1e-5, 1.0 / 128.0};
// Chains of several BF16 roundings (norm, then rotation; gates).
constexpr PointwiseCriterion kChained{2e-3, 1.0 / 64.0};

double bf(double x) { return bf16_to_f32(f32_to_bf16(float(x))); }

std::vector<float> values(std::size_t n, std::uint32_t seed, float lo, float hi) {
    std::vector<float> v(n);
    fill_uniform(v, seed, lo, hi);
    round_to_bf16(v);
    return v;
}

std::vector<double> download(const DeviceBuffer& d, std::size_t n) {
    return from_device_bf16(d, n);
}

int report(const char* name, int failures) {
    std::cout << (failures ? "FAIL " : "OK ") << name << '\n';
    return failures;
}

float inv_freq(float theta, int i) { return 1.0F / std::pow(theta, float(2 * i) / float(D)); }

// rotate(bf16(rms(x) w)) of one head, FP64 with the contract's FP32 angle.
std::vector<double> norm_rope(const float* x, const float* w, int p, float theta, double eps) {
    double ss = 0;
    for (int d = 0; d < D; ++d) ss += double(x[d]) * x[d];
    const double inv = 1.0 / std::sqrt(ss / D + eps);
    std::vector<double> n(D), out(D);
    for (int d = 0; d < D; ++d) n[d] = bf(x[d] * inv * w[d]);
    for (int i = 0; i < D / 2; ++i) {
        const double angle = double(float(p) * inv_freq(theta, i));
        const double c = std::cos(angle), s = std::sin(angle);
        out[i]         = n[i] * c - n[i + D / 2] * s;
        out[i + D / 2] = n[i + D / 2] * c + n[i] * s;
    }
    return out;
}

int test_mix(int H, int W, int count, std::uint32_t seed) {
    const int S = 4, R = 4;
    auto F      = values(std::size_t(H) * W * S, seed, -3, 3);
    auto logits = values(S * R, seed + 1, -2, 2);
    std::vector<std::vector<float>> norms;
    for (int r = 0; r < R; ++r) norms.push_back(values(H, seed + 2 + r, 0.5, 1.5));
    auto dF = to_device_bf16(F), dl = to_device_bf16(logits);
    std::vector<DeviceBuffer> dn;
    for (auto& n : norms) dn.push_back(to_device_bf16(n));
    DeviceBuffer out(std::size_t(H) * W * (R - 1) * 2), last(std::size_t(H) * 2);
    auto dc = to_device_i32({count});
    Tensor tF(dF.p, DType::BF16, {H, W, S}), tl(dl.p, DType::BF16, {S, R});
    std::array<Tensor, 4> tn;
    std::array<const Tensor*, 8> norm_views{};
    for (int r = 0; r < R; ++r) {
        tn[r]         = Tensor(dn[r].p, DType::BF16, {H});
        norm_views[r] = &tn[r];
    }
    Tensor to(out.p, DType::BF16, {H, W, R - 1}), tlast(last.p, DType::BF16, {H}),
        tc(dc.p, DType::I32, {1});
    ops::feature_mix_rmsnorm(tF, tl, norm_views, R, 1e-6F, &tc, to, tlast, nullptr);
    cuda_synchronize();
    std::vector<double> ref(std::size_t(H) * W * (R - 1)), ref_last(H);
    for (int r = 0; r < R; ++r) {
        double m = -1e30, e[4], sum = 0;
        for (int s = 0; s < S; ++s) m = std::max(m, double(logits[r * S + s]));
        for (int s = 0; s < S; ++s) sum += e[s] = std::exp(logits[r * S + s] - m);
        double alpha[4];
        for (int s = 0; s < S; ++s) alpha[s] = bf(e[s] / sum);
        for (int t = 0; t < W; ++t) {
            if (r == R - 1 && t != count - 1) continue;
            std::vector<double> mix(H);
            double ss = 0;
            for (int h = 0; h < H; ++h) {
                for (int s = 0; s < S; ++s)
                    mix[h] += F[(std::size_t(s) * W + t) * H + h] * alpha[s];
                ss += mix[h] * mix[h];
            }
            const double inv = 1 / std::sqrt(ss / H + 1e-6);
            for (int h = 0; h < H; ++h) {
                const double v = mix[h] * inv * norms[r][h];
                if (r == R - 1)
                    ref_last[h] = v;
                else
                    ref[(std::size_t(r) * W + t) * H + h] = v;
            }
        }
    }
    const std::string name =
        "feature_mix_rmsnorm H=" + std::to_string(H) + " W=" + std::to_string(W);
    int f = verify_pointwise(name, download(out, ref.size()), ref, kRounded);
    f += verify_pointwise(name + " last", download(last, H), ref_last, kRounded);
    return report(name.c_str(), f);
}

int test_rope(int nq, int nk, int T, int extra, int first, std::uint32_t seed) {
    const int rows = (nq + nk + extra) * D;
    auto x         = values(std::size_t(rows) * T, seed, -2, 2);
    auto qn = values(D, seed + 1, 0.5, 1.5), kn = values(D, seed + 2, 0.5, 1.5);
    std::vector<int> pos(T);
    for (int t = 0; t < T; ++t) pos[t] = first + t;
    auto dx = to_device_bf16(x), dq = to_device_bf16(qn), dk = to_device_bf16(kn);
    auto dp = to_device_i32(pos);
    Tensor tx(dx.p, DType::BF16, {rows, T}), tq(dq.p, DType::BF16, {D}), tk(dk.p, DType::BF16, {D}),
        tp(dp.p, DType::I32, {T});
    ops::head_rmsnorm_rope(tx, nq, nk, tq, tk, tp, 1e6F, 1e-6F, nullptr);
    cuda_synchronize();
    std::vector<double> ref(x.begin(), x.end());
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < nq + nk; ++h) {
            const std::size_t o = std::size_t(t) * rows + std::size_t(h) * D;
            const auto r = norm_rope(&x[o], h < nq ? qn.data() : kn.data(), pos[t], 1e6F, 1e-6);
            for (int d = 0; d < D; ++d) ref[o + d] = r[d];
        }
    }
    const std::string name = "head_rmsnorm_rope " + std::to_string(nq) + "/" + std::to_string(nk) +
                             " p=" + std::to_string(first);
    return report(name.c_str(), verify_pointwise(name, download(dx, ref.size()), ref, kChained));
}

int test_store(int kv, int W, int count, int first, int capacity, std::uint32_t seed) {
    auto raw = values(std::size_t(2) * kv * D * W, seed, -2, 2);
    auto kn  = values(D, seed + 1, 0.5, 1.5);
    std::vector<int> pos(W);
    for (int j = 0; j < W; ++j) pos[j] = first + j;
    const std::size_t plane = std::size_t(D) * kv * capacity;
    std::vector<float> fill(plane, 7.0F);
    auto dr = to_device_bf16(raw), dk = to_device_bf16(kn), dp = to_device_i32(pos),
         dc = to_device_i32({count}), kc = to_device_bf16(fill), vc = to_device_bf16(fill);
    Tensor tr(dr.p, DType::BF16, {2 * kv * D, W}), tn(dk.p, DType::BF16, {D}),
        tp(dp.p, DType::I32, {W}), tc(dc.p, DType::I32, {1});
    std::array<const Tensor*, 8> pr{&tr}, norm_views{&tn};
    std::array<ops::DraftKVCacheView, 8> caches{};
    caches[0] = {Tensor(kc.p, DType::BF16, {D, kv, capacity}),
                 Tensor(vc.p, DType::BF16, {D, kv, capacity}), capacity};
    ops::context_kv_store(pr, norm_views, caches, 1, tp, tc, 1e6F, 1e-6F, nullptr);
    cuda_synchronize();
    std::vector<double> rk(plane, 7.0), rv(plane, 7.0);
    const int end = first + count;
    for (int j = 0; j < count; ++j) {
        const int p = pos[j];
        if (p < end - capacity) continue;
        const std::size_t slot = std::size_t(p % capacity);
        for (int h = 0; h < kv; ++h) {
            const float* k = &raw[std::size_t(j) * 2 * kv * D + std::size_t(h) * D];
            const auto r   = norm_rope(k, kn.data(), p, 1e6F, 1e-6);
            for (int d = 0; d < D; ++d) {
                rk[(slot * kv + h) * D + d] = r[d];
                rv[(slot * kv + h) * D + d] =
                    raw[std::size_t(j) * 2 * kv * D + std::size_t(kv + h) * D + d];
            }
        }
    }
    const std::string name = "context_kv_store kv=" + std::to_string(kv) +
                             " W=" + std::to_string(W) + " count=" + std::to_string(count) +
                             " cap=" + std::to_string(capacity);
    int f = verify_pointwise(name + " K", download(kc, plane), rk, kChained);
    f += verify_pointwise(name + " V", download(vc, plane), rv, kRounded);
    return report(name.c_str(), f);
}

int test_attention(int nq, int kv, int B, int P, int window, int capacity, std::uint32_t seed) {
    const int rows          = (nq + 2 * kv) * D;
    auto qkv                = values(std::size_t(rows) * B, seed, -1, 1);
    const std::size_t plane = std::size_t(D) * kv * capacity;
    auto kc = values(plane, seed + 1, -1, 1), vc = values(plane, seed + 2, -1, 1);
    auto dq = to_device_bf16(qkv), dk = to_device_bf16(kc), dv = to_device_bf16(vc);
    auto df = to_device_i32({P});
    DeviceBuffer out(std::size_t(nq) * D * B * 2);
    WorkspaceArena work(ops::block_attention_workspace_bytes(nq, B) + 4096);
    Tensor tq(dq.p, DType::BF16, {rows, B}), tf(df.p, DType::I32, {1}),
        to(out.p, DType::BF16, {nq * D, B});
    ops::DraftKVCacheView cache{Tensor(dk.p, DType::BF16, {D, kv, capacity}),
                                Tensor(dv.p, DType::BF16, {D, kv, capacity}), capacity};
    const float scale = 1.0F / std::sqrt(float(D));
    ops::block_attention(tq, nq, kv, cache, tf, window, scale, work, to, nullptr);
    cuda_synchronize();
    const int lo = window > 0 ? std::max(0, P - window) : 0;
    std::vector<double> ref(std::size_t(nq) * D * B);
    const int G = nq / kv;
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < nq; ++h) {
            const int g    = h / G;
            const float* q = &qkv[std::size_t(b) * rows + std::size_t(h) * D];
            std::vector<double> score;
            std::vector<const float*> vals;
            const auto key = [&](const float* k, const float* v) {
                double dot = 0;
                for (int d = 0; d < D; ++d) dot += double(q[d]) * k[d];
                score.push_back(dot * scale);
                vals.push_back(v);
            };
            for (int p = lo; p < P; ++p) {
                const std::size_t o = (std::size_t(p % capacity) * kv + g) * D;
                key(&kc[o], &vc[o]);
            }
            for (int j = 0; j < B; ++j) {
                const std::size_t c = std::size_t(j) * rows;
                key(&qkv[c + std::size_t(nq + g) * D], &qkv[c + std::size_t(nq + kv + g) * D]);
            }
            const double m = *std::max_element(score.begin(), score.end());
            double sum     = 0;
            for (auto& s : score) sum += s = std::exp(s - m);
            for (int d = 0; d < D; ++d) {
                double acc = 0;
                for (std::size_t j = 0; j < score.size(); ++j) acc += score[j] * vals[j][d];
                ref[(std::size_t(b) * nq + h) * D + d] = acc / sum;
            }
        }
    }
    const std::string name = "block_attention " + std::to_string(nq) + "/" + std::to_string(kv) +
                             " B=" + std::to_string(B) + " P=" + std::to_string(P) +
                             " window=" + std::to_string(window);
    return report(name.c_str(), verify_pointwise(name, download(out, ref.size()), ref,
                                                 PointwiseCriterion{2e-3, 1.0 / 128.0}));
}

int test_add_rmsnorm(int H, int T, std::uint32_t seed) {
    auto r  = values(std::size_t(H) * T, seed, -3, 3),
         y  = values(std::size_t(H) * T, seed + 1, -3, 3);
    auto w  = values(H, seed + 2, 0.5, 1.5);
    auto dr = to_device_bf16(r), dy = to_device_bf16(y), dw = to_device_bf16(w);
    DeviceBuffer out(std::size_t(H) * T * 2);
    Tensor tr(dr.p, DType::BF16, {H, T}), ty(dy.p, DType::BF16, {H, T}), tw(dw.p, DType::BF16, {H}),
        to(out.p, DType::BF16, {H, T});
    ops::add_rmsnorm(tr, ty, tw, 1e-6F, to, nullptr);
    cuda_synchronize();
    std::vector<double> rres(r.size()), rout(r.size());
    for (int t = 0; t < T; ++t) {
        double ss = 0;
        for (int h = 0; h < H; ++h) {
            const double s = double(r[std::size_t(t) * H + h]) + y[std::size_t(t) * H + h];
            rres[std::size_t(t) * H + h] = s;
            ss += s * s;
        }
        const double inv = 1 / std::sqrt(ss / H + 1e-6);
        for (int h = 0; h < H; ++h)
            rout[std::size_t(t) * H + h] = rres[std::size_t(t) * H + h] * inv * w[h];
    }
    const std::string name = "add_rmsnorm H=" + std::to_string(H) + " T=" + std::to_string(T);
    int f = verify_pointwise(name + " residual", download(dr, r.size()), rres, kRounded);
    f += verify_pointwise(name + " out", download(out, r.size()), rout, kRounded);
    return report(name.c_str(), f);
}

// The recurrent head: context, then seven chained steps (tp 1, one vocabulary block) against the
// FP64 chain fed with the kernel's own previous token, state and choice.
int test_recurrent(int H, int rows, int first, int ranks, std::uint32_t seed) {
    const int R = 256, G = 7, in = 2 * R + 2 * H, V = first + rows + 5;
    auto W     = values(std::size_t(3) * R * in, seed, -1.0F / std::sqrt(float(in)),
                        1.0F / std::sqrt(float(in)));
    auto bias  = values(3 * R, seed + 1, -0.1F, 0.1F);
    auto hid   = values(std::size_t(H) * G, seed + 2, -2, 2);
    auto ctx   = values(H, seed + 3, -2, 2);
    auto emb   = values(std::size_t(V) * R, seed + 4, -1, 1);
    auto fold  = values(std::size_t(rows) * R, seed + 5, -0.5F, 0.5F);
    auto fbias = values(rows, seed + 6, -3, 3);
    auto base  = values(std::size_t(rows) * G, seed + 7, -20, 20);
    auto dW = to_device_bf16(W), db = to_device_bf16(bias), dh = to_device_bf16(hid),
         dc = to_device_bf16(ctx), de = to_device_bf16(emb), dfo = to_device_bf16(fold),
         dfb = to_device_bf16(fbias), dba = to_device_bf16(base);
    DeviceBuffer context(std::size_t(3) * R * G * 4), s0(R * 2), s1(R * 2), drafts(G * 4),
        choice(16);
    cuda_check(cudaMemset(s0.p, 0, R * 2), "memset");
    auto anchor = to_device_i32({3});
    Tensor tW(dW.p, DType::BF16, {in, 3 * R}), tb(db.p, DType::BF16, {3 * R}),
        th(dh.p, DType::BF16, {H, G}), tc(dc.p, DType::BF16, {H}),
        tctx(context.p, DType::FP32, {3 * R, G});
    ops::recurrent_head_context(tW, tb, th, tc, tctx, nullptr);
    cuda_synchronize();
    std::vector<double> rctx(std::size_t(3) * R * G);
    for (int s = 0; s < G; ++s) {
        for (int o = 0; o < 3 * R; ++o) {
            double acc = bias[o];
            for (int k = 0; k < H; ++k) {
                acc += double(W[std::size_t(o) * in + 2 * R + k]) * hid[std::size_t(s) * H + k];
                acc += double(W[std::size_t(o) * in + 2 * R + H + k]) * ctx[k];
            }
            rctx[std::size_t(s) * 3 * R + o] = acc;
        }
    }
    int f = verify_pointwise("recurrent_head_context", from_device_f32(context, rctx.size()), rctx,
                             PointwiseCriterion{1e-4, 1e-4});
    ops::RecurrentHeadOperands head{tW,
                                    Tensor(de.p, DType::BF16, {R, V}),
                                    Tensor(dfo.p, DType::BF16, {R, rows}),
                                    Tensor(dfb.p, DType::BF16, {rows}),
                                    Tensor(dba.p, DType::BF16, {rows, G}),
                                    tctx,
                                    30.0F,
                                    first,
                                    0,
                                    ranks};
    WorkspaceArena work(ops::recurrent_head_step_workspace_bytes() + 4096);
    Tensor ta(anchor.p, DType::I32, {1}), td(drafts.p, DType::I32, {G}),
        tch(choice.p, DType::FP32, {4});
    std::array<Tensor, 2> state{Tensor(s0.p, DType::BF16, {R}), Tensor(s1.p, DType::BF16, {R})};
    const auto gpu_ctx = from_device_f32(context, rctx.size());
    std::vector<double> st(R, 0.0);
    int prev = 3;
    for (int s = 0; s < G; ++s) {
        ops::recurrent_head_step(head, s, ta, state[s % 2], state[(s + 1) % 2], td, tch, work,
                                 nullptr);
        cuda_synchronize();
        // FP64 step from the same inputs (the kernel's context, its previous state and token).
        std::vector<double> raw(3 * R), lat(R), next(R);
        for (int o = 0; o < 3 * R; ++o) {
            double acc = gpu_ctx[std::size_t(s) * 3 * R + o];
            for (int k = 0; k < R; ++k) {
                acc += double(W[std::size_t(o) * in + k]) * st[k];
                acc += double(W[std::size_t(o) * in + R + k]) * emb[std::size_t(prev) * R + k];
            }
            raw[o] = acc;
        }
        for (int i = 0; i < R; ++i) {
            const double g = bf(1 / (1 + std::exp(-bf(raw[i]))));
            const double c = bf(std::tanh(bf(raw[R + i])));
            next[i]        = bf(bf(g * st[i]) + bf(bf(1 - g) * c));
            lat[i]         = bf(std::tanh(bf(raw[2 * R + i])));
        }
        double best = -1e30;
        std::vector<double> logit(rows);
        for (int v = 0; v < rows; ++v) {
            double dot = fbias[v];
            for (int k = 0; k < R; ++k) dot += lat[k] * fold[std::size_t(v) * R + k];
            const double x = bf(base[std::size_t(s) * rows + v] + bf(dot));
            logit[v]       = bf(30 * std::tanh(x / 30));
            best           = std::max(best, logit[v]);
        }
        const auto got_state = download(s % 2 ? s0 : s1, R);
        f += verify_pointwise("recurrent_head_step state", got_state, next, kChained);
        const auto ch = from_device_f32(choice, 4);
        const int id  = int(ch[1]) - first;
        // The chosen token is a maximum of the oracle's logits up to one BF16 step of the value.
        const bool ok = id >= 0 && id < rows &&
                        logit[id] >= best - std::ldexp(std::abs(best), -7) &&
                        std::abs(ch[0] - best) <= std::ldexp(std::abs(best), -7) &&
                        ch[2] == (ranks == 2 ? 0.0 : -std::numeric_limits<double>::infinity());
        if (!ok) {
            std::cerr << "recurrent_head_step " << s << ": choice " << ch[0] << "/" << ch[1]
                      << " against best " << best << '\n';
            ++f;
        }
        prev = int(ch[1]);
        st   = got_state;
        if (ranks == 2) {
            // The peer's slot as the all-reduce would bring it: a lower value, so this rank wins.
            const float peer[2] = {float(ch[0]) - 100.0F, 0.0F};
            cuda_check(
                cudaMemcpy(static_cast<float*>(choice.p) + 2, peer, 8, cudaMemcpyHostToDevice),
                "copy");
        }
    }
    ops::recurrent_head_select(tch, td, nullptr);
    cuda_synchronize();
    const auto got = from_device_i32(drafts, G);
    f += got[G - 1] == int(from_device_f32(choice, 4)[1]) ? 0 : 1;
    const std::string name =
        "recurrent_head rows=" + std::to_string(rows) + " ranks=" + std::to_string(ranks);
    return report(name.c_str(), f);
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    try {
        failures += test_mix(64, 3, 2, 11);
        failures += test_mix(2304, 8, 8, 12);
        failures += test_mix(2304, 8, 3, 13);
        failures += test_mix(2304, 256, 256, 14);
        failures += test_rope(12, 4, 8, 4, 0, 21);
        failures += test_rope(24, 8, 8, 8, 131000, 22);
        failures += test_rope(2, 1, 3, 1, 12345, 23);
        failures += test_store(4, 8, 8, 100, 1024, 31);
        failures += test_store(4, 8, 3, 2000, 1024, 32);
        failures += test_store(4, 64, 64, 1000, 32, 33);   // more columns than the ring
        failures += test_store(4, 8, 5, 12000, 16384, 34); // a global layer
        failures += test_attention(12, 4, 8, 0, 1024, 1024, 41);
        failures += test_attention(12, 4, 8, 700, 1024, 1024, 42);
        failures += test_attention(12, 4, 8, 3000, 1024, 1024, 43); // ring wrapped
        failures += test_attention(12, 4, 8, 12800, 0, 16384, 44);  // global
        failures += test_attention(24, 8, 8, 2500, 0, 4096, 45);
        failures += test_attention(6, 2, 8, 100, 64, 64, 46);
        failures += test_add_rmsnorm(2304, 8, 51);
        failures += test_add_rmsnorm(512, 3, 52);
        failures += test_recurrent(512, 3000, 0, 1, 61);
        failures += test_recurrent(2304, 100029, 100029, 2, 62);
    } catch (const std::exception& error) {
        std::cerr << "dspark_draft: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " dspark_draft\n";
    return failures ? 1 : 0;
}
