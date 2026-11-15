// Host (CPU) routed-expert path: NVFP4 W4A8 dots and the weight-major layer executor against an
// independent FP64 oracle that decodes the stored block_scale_k16_m128x4_v1 planes itself.
//
// Semantics under test (host_nvfp4.h): w = e2m1 * e4m3 / divisor; activations quantized to int8
// per 16 elements, q = rint(fl32(x * fl32(1 / fl32(amax / 127)))); the SwiGLU output is quantized
// the same way before the down projection. The oracle replicates that codec exactly and computes
// every sum in FP64.
//
// Criteria:
//  - dots: |got - ref| <= 2 (K/16 + 8) u * sum|w*a| (u = 2^-24): FP32 rounding of one product
//    per 16-block and of a K/16-term accumulation (worst case, not a statistical allowance);
//  - layer, in two stages split at the int8 SwiGLU output (a semantic quantization boundary):
//    A) every int8 intermediate equals the reference codec applied to fl32 of the FP64 h, except
//       by one at elements whose FP64 distance to a rounding boundary is within the worst-case
//       FP32 error of h (gate/up dot bounds propagated through SwiGLU; cancellation makes this
//       zone wide, so the count of actual flips is reported); block scales within the bound of
//       their absmax element;
//    B) the output vs FP64 down projections and expert sum of the executor's own intermediates:
//       2 (I/16 + 8 + U) u * sum|alpha w a| per output (no allowance needed);
//  - determinism: identical bits across 1/4/6 threads and repeated calls; the multi-token dot
//    of a token is bit-identical at every T;
//  - informative (printed, gross gate only): relative L2 vs the W4A16 reference (activations not
//    quantized) and vs an emulation of the GPU qf_moe W4A4 recipe (FP4 activations with E4M3
//    block scales and a global scale, bf16 roundings of gemm1 and SwiGLU).
#include "host_moe_fixture.h"
#include "ops/sparse_moe/host/host_moe_executor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer::ops::host;
using namespace ninfer::test::host_moe;

namespace {

int g_failures      = 0;
constexpr double kU = 1.0 / 16777216.0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++g_failures;
        std::printf("FAIL %s\n", what.c_str());
    }
}

// Reference W4A8 activation codec (natural element order).
struct QAct {
    std::vector<int> q;
    std::vector<float> s;

    std::vector<double> value(int k) const {
        std::vector<double> v(static_cast<std::size_t>(k));
        for (int i = 0; i < k; ++i) v[i] = static_cast<double>(q[i]) * s[i / 16];
        return v;
    }
};

QAct quantize_ref(const float* x, int k) {
    QAct a;
    a.q.resize(k);
    a.s.resize(k / 16);
    for (int b = 0; b < k / 16; ++b) {
        float amax = 0.0F;
        for (int j = 0; j < 16; ++j) amax = std::max(amax, std::fabs(x[b * 16 + j]));
        const float s   = amax > 0.0F ? amax / 127.0F : 1.0F;
        const float inv = 1.0F / s;
        for (int j = 0; j < 16; ++j) {
            const float p   = x[b * 16 + j] * inv;
            a.q[b * 16 + j] = static_cast<int>(std::nearbyint(p));
        }
        a.s[b] = s;
    }
    return a;
}

std::vector<float> random_activation(int k, std::mt19937_64& rng,
                                     std::vector<std::uint16_t>* bf16 = nullptr) {
    std::normal_distribution<double> normal(0.0, 1.0);
    std::lognormal_distribution<double> gain(0.0, 0.5);
    std::vector<float> x(static_cast<std::size_t>(k));
    if (bf16) bf16->resize(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) {
        double v = normal(rng) * gain(rng);
        if (i % 641 == 7) v *= 20.0; // a few outlier channels
        const std::uint16_t h = to_bf16(static_cast<float>(v));
        x[i]                  = from_bf16(h);
        if (bf16) (*bf16)[i] = h;
    }
    return x;
}

// ---------------------------------------------------------------- dots
void test_dots(std::mt19937_64& rng) {
    for (const int k : {2560, 640}) {
        const std::int64_t rows = 256;
        PlaneData pd(rows, k, 2, rng);
        const Nvfp4Plane plane = pd.plane();
        W4A8ActivationBuffer acts(k, kHostMaxTokens), acts_f32(k, kHostMaxTokens);
        std::vector<std::vector<float>> xs;
        std::vector<W4A8Activation> view, view_f32;
        int codec_mismatch = 0;
        for (int t = 0; t < kHostMaxTokens; ++t) {
            std::vector<std::uint16_t> h;
            xs.push_back(random_activation(k, rng, &h));
            quantize_bf16(h.data(), k, acts.at(t));
            quantize_blocks(xs.back().data(), 0, k / 16, acts_f32.at(t));
            view.push_back(acts.at(t));
            view_f32.push_back(acts_f32.at(t));
            const QAct ref         = quantize_ref(xs.back().data(), k);
            const W4A8Activation a = acts.at(t), f = acts_f32.at(t);
            for (int i = 0; i < k; ++i) {
                const int b = i / 16, j = i % 16;
                const int stored = b * 16 + (j % 2 ? 8 + j / 2 : j / 2);
                if (a.q[stored] != ref.q[i] || f.q[stored] != ref.q[i] ||
                    static_cast<std::uint8_t>(a.qu[stored]) !=
                        static_cast<std::uint8_t>(ref.q[i] + 128)) {
                    ++codec_mismatch;
                }
                if (j == 0 && (a.asb[b] != ref.s[b] || f.asb[b] != ref.s[b] ||
                               a.sb16[4 * b + 3] != ref.s[b])) {
                    ++codec_mismatch;
                }
                if (j % 4 == 0) {
                    int sum = 0;
                    for (int d = 0; d < 4; ++d) sum += a.q[b * 16 + (j / 4) * 4 + d];
                    if (a.corr[b * 4 + j / 4] != -12 * sum) ++codec_mismatch;
                }
            }
        }
        expect(codec_mismatch == 0, "activation codec k=" + std::to_string(k));

        // Oracle per (row, token).
        const std::int64_t r0 = 37;
        const int n           = 150; // crosses a 128-row tile and both divisors
        std::vector<double> ref(static_cast<std::size_t>(n) * kHostMaxTokens), l1(ref.size());
        for (int t = 0; t < kHostMaxTokens; ++t) {
            const auto a = quantize_ref(xs[t].data(), k).value(k);
            for (int i = 0; i < n; ++i) {
                double s = 0, m = 0;
                for (int c = 0; c < k; ++c) {
                    const double p = pd.weight(r0 + i, c) * a[c];
                    s += p;
                    m += std::abs(p);
                }
                ref[static_cast<std::size_t>(t) * n + i] = s;
                l1[static_cast<std::size_t>(t) * n + i]  = m;
            }
        }
        const double bound = 2.0 * (k / 16 + 8) * kU;
        std::vector<float> got(static_cast<std::size_t>(n) * kHostMaxTokens),
            single(static_cast<std::size_t>(n));
        for (int T = 1; T <= kHostMaxTokens; ++T) {
            double worst = 0;
            int variance = 0;
            rows_dot(plane, r0, n, view.data(), T, got.data(), n);
            for (int t = 0; t < T; ++t) {
                for (int i = 0; i < n; ++i) {
                    const std::size_t idx = static_cast<std::size_t>(t) * n + i;
                    worst =
                        std::max(worst, std::abs(got[idx] - ref[idx]) / (bound * l1[idx] + 1e-30));
                }
                rows_dot(plane, r0, n, &view[t], 1, single.data(), n);
                for (int i = 0; i < n; ++i)
                    variance += single[i] != got[static_cast<std::size_t>(t) * n + i];
            }
            expect(worst <= 1.0, "dot k=" + std::to_string(k) + " T=" + std::to_string(T) +
                                     " err/tol " + std::to_string(worst));
            expect(variance == 0,
                   "dot batch variance k=" + std::to_string(k) + " T=" + std::to_string(T));
            if (T == 1 || T == 4 || T == 8) {
                std::printf("dot k=%4d T=%d max err/tol %.3f (tol %.2e of sum|w*a|), bit-identical "
                            "to T=1\n",
                            k, T, worst, bound);
            }
        }
    }
}

// ---------------------------------------------------------------- layer oracle
struct Case {
    std::string name;
    int tokens = 0;
    std::vector<std::int32_t> ids; // [T][top_k], -1 = elsewhere
    std::vector<float> alpha;
    std::vector<std::uint16_t> x; // [T][H] bf16
};

// Stage A reference of one (expert, token) pair: FP64 h from the exactly quantized x, its int8
// codec on fl32(h), and which elements the executor's FP32 h may legitimately round the other way.
struct PairRef {
    int t = 0, e = 0;
    double alpha = 0;
    std::vector<int> q;          // natural order
    std::vector<float> s;        // block scales
    std::vector<char> amb;       // element within the FP32 bound of a rounding boundary
    std::vector<double> drel;    // per block: relative bound on the scale (absmax element's bound)
    std::vector<double> h16, h4; // W4A16 h, GPU-recipe W4A4 h (bf16)
};

struct CaseRef {
    std::vector<PairRef> pairs;
    std::map<int, std::vector<std::vector<double>>> down; // decoded down rows per expert
    std::vector<double> w4a8, w4a16, w4a4;                // end-to-end references [T][H]
    int ambiguous = 0;
    int experts   = 0;
};

double silu(double g) { return g / (1.0 + std::exp(-g)); }

double bf16_round(double v) { return from_bf16(to_bf16(static_cast<float>(v))); }

// qf_moe activation quantization (FlashInfer recipe): FP4 E2M1 with an E4M3 block scale relative
// to a global scale gs; returns the dequantized values.
std::vector<double> fp4_quant(const std::vector<double>& v, double gs) {
    std::vector<double> out(v.size());
    for (std::size_t b = 0; b < v.size() / 16; ++b) {
        double amax = 0;
        for (int j = 0; j < 16; ++j) amax = std::max(amax, std::abs(v[b * 16 + j]));
        const double sf = e4m3_value(e4m3_encode_positive(gs * amax / 6.0));
        const double os = amax != 0 && sf != 0 ? gs / sf : 0.0;
        for (int j = 0; j < 16; ++j)
            out[b * 16 + j] = e2m1_value(e2m1_encode(v[b * 16 + j] * os)) * sf / gs;
    }
    return out;
}

CaseRef build_ref(const LayerData& L, const Case& c, int top_k) {
    const int H = L.hidden, I = L.intermediate, T = c.tokens;
    CaseRef o;
    const std::size_t n = static_cast<std::size_t>(T) * H;
    o.w4a8.assign(n, 0.0), o.w4a16.assign(n, 0.0), o.w4a4.assign(n, 0.0);
    std::vector<std::vector<double>> xq(T), x64(T), x4(T);
    double xmax = 0;
    for (int t = 0; t < T; ++t) {
        std::vector<float> xf(H);
        x64[t].resize(H);
        for (int i = 0; i < H; ++i) {
            xf[i]     = from_bf16(c.x[static_cast<std::size_t>(t) * H + i]);
            x64[t][i] = xf[i];
            xmax      = std::max(xmax, std::abs(x64[t][i]));
        }
        xq[t] = quantize_ref(xf.data(), H).value(H);
    }
    const double a1_gs = 448.0 * 6.0 / xmax;
    for (int t = 0; t < T; ++t) x4[t] = fp4_quant(x64[t], a1_gs);
    std::vector<int> experts;
    for (int i = 0; i < T * top_k; ++i) {
        if (c.ids[i] >= 0 && std::find(experts.begin(), experts.end(), c.ids[i]) == experts.end())
            experts.push_back(c.ids[i]);
    }
    o.experts               = static_cast<int>(experts.size());
    const double gate_bound = 2.0 * (H / 16 + 8) * kU;
    double hmax4            = 0;
    for (const int e : experts) {
        std::vector<std::vector<double>> gu(2 * I);
        for (int r = 0; r < 2 * I; ++r) L.gate_up.decode_row(std::int64_t{e} * 2 * I + r, gu[r]);
        auto& dn = o.down[e];
        dn.resize(H);
        for (int r = 0; r < H; ++r) L.down.decode_row(std::int64_t{e} * H + r, dn[r]);
        for (int t = 0; t < T; ++t) {
            for (int s = 0; s < top_k; ++s) {
                if (c.ids[t * top_k + s] != e) continue;
                PairRef p;
                p.t = t, p.e = e, p.alpha = c.alpha[t * top_k + s];
                std::vector<double> h8(I), tau(I);
                p.h16.resize(I), p.h4.resize(I);
                for (int j = 0; j < I; ++j) {
                    double g8 = 0, u8 = 0, lg = 0, lu = 0, g16 = 0, u16 = 0, g4 = 0, u4 = 0;
                    for (int k = 0; k < H; ++k) {
                        g8 += gu[j][k] * xq[t][k], lg += std::abs(gu[j][k] * xq[t][k]);
                        u8 += gu[I + j][k] * xq[t][k], lu += std::abs(gu[I + j][k] * xq[t][k]);
                        g16 += gu[j][k] * x64[t][k], u16 += gu[I + j][k] * x64[t][k];
                        g4 += gu[j][k] * x4[t][k], u4 += gu[I + j][k] * x4[t][k];
                    }
                    h8[j] = silu(g8) * u8;
                    // |d silu/dg| <= 1.1; SwiGLU in FP32: a few ulps of h.
                    tau[j] = std::abs(u8) * 1.1 * gate_bound * lg +
                             (std::abs(silu(g8)) + 1.1 * gate_bound * lg) * gate_bound * lu +
                             8 * kU * std::abs(h8[j]);
                    p.h16[j] = silu(g16) * u16;
                    g4 = bf16_round(g4), u4 = bf16_round(u4);
                    p.h4[j] = bf16_round(static_cast<float>(silu(g4)) * static_cast<float>(u4));
                    hmax4   = std::max(hmax4, std::abs(p.h4[j]));
                }
                std::vector<float> h32(I);
                for (int j = 0; j < I; ++j) h32[j] = static_cast<float>(h8[j]);
                const QAct q = quantize_ref(h32.data(), I);
                p.q = q.q, p.s = q.s;
                p.amb.assign(I, 0);
                p.drel.assign(I / 16, 0.0);
                for (int b = 0; b < I / 16; ++b) {
                    int am = b * 16;
                    for (int j = b * 16; j < b * 16 + 16; ++j)
                        if (std::abs(h32[j]) > std::abs(h32[am])) am = j;
                    p.drel[b] = std::abs(h32[am]) > 0 ? tau[am] / std::abs(h32[am]) + 2 * kU : 0.0;
                    const double inv = 1.0 / q.s[b];
                    for (int j = b * 16; j < b * 16 + 16; ++j) {
                        const double v     = static_cast<double>(h32[j]) * inv;
                        const double dist  = std::abs(std::abs(v - std::floor(v)) - 0.5);
                        const double slack = (tau[j] + std::abs(h32[j]) * p.drel[b]) * inv +
                                             4 * kU * std::abs(v) + 1e-9;
                        if (dist <= slack) p.amb[j] = 1, ++o.ambiguous;
                    }
                }
                o.pairs.push_back(std::move(p));
            }
        }
    }
    const double a2_gs = hmax4 > 0 ? 448.0 * 6.0 / hmax4 : 1.0;
    for (const auto& p : o.pairs) {
        const auto& dn = o.down[p.e];
        const auto h4q = fp4_quant(p.h4, a2_gs);
        for (int r = 0; r < H; ++r) {
            double s8 = 0, s16 = 0, s4 = 0;
            for (int j = 0; j < I; ++j) {
                s8 += dn[r][j] * p.q[j] * static_cast<double>(p.s[j / 16]);
                s16 += dn[r][j] * p.h16[j];
                s4 += dn[r][j] * h4q[j];
            }
            const std::size_t idx = static_cast<std::size_t>(p.t) * H + r;
            o.w4a8[idx] += p.alpha * s8;
            o.w4a16[idx] += p.alpha * s16;
            o.w4a4[idx] += static_cast<float>(s4) * static_cast<float>(p.alpha);
        }
    }
    return o;
}

struct RunCheck {
    int flips = 0, bad_q = 0, bad_scale = 0, missing = 0;
    double stage_b = 0; // max err / tol of the down + expert sum stage
};

// Stage A: the executor's int8 intermediates vs the reference codec (differences only at ambiguous
// elements, by one); stage B: its output vs FP64 down projections of its own intermediates.
RunCheck check_run(const CaseRef& ref, const HostMoeExecutor& ex, const std::vector<float>& out,
                   int T, int H, int I) {
    RunCheck rc;
    std::vector<double> want(static_cast<std::size_t>(T) * H, 0.0), l1(want.size(), 0.0);
    const int pairs = ex.intermediate_count();
    rc.missing      = static_cast<int>(ref.pairs.size()) - pairs;
    for (int i = 0; i < pairs; ++i) {
        const HostMoeIntermediate m = ex.intermediate(i);
        const PairRef* p            = nullptr;
        for (const auto& r : ref.pairs)
            if (r.e == m.expert && r.t == m.token) p = &r;
        if (!p) {
            ++rc.missing;
            continue;
        }
        std::vector<double> a(I);
        for (int j = 0; j < I; ++j) {
            const int b = j / 16, k = j % 16;
            const int q = m.act.q[b * 16 + (k % 2 ? 8 + k / 2 : k / 2)];
            const int d = q - p->q[j];
            if (d != 0) {
                if (p->amb[j] && std::abs(d) == 1)
                    ++rc.flips;
                else
                    ++rc.bad_q;
            }
            a[j] = q * static_cast<double>(m.act.asb[b]);
        }
        for (int b = 0; b < I / 16; ++b) {
            if (std::abs(m.act.asb[b] - p->s[b]) > p->drel[b] * p->s[b]) ++rc.bad_scale;
        }
        const auto& dn = ref.down.at(m.expert);
        for (int r = 0; r < H; ++r) {
            double s = 0, sa = 0;
            for (int j = 0; j < I; ++j) s += dn[r][j] * a[j], sa += std::abs(dn[r][j] * a[j]);
            const std::size_t idx = static_cast<std::size_t>(m.token) * H + r;
            want[idx] += p->alpha * s;
            l1[idx] += std::abs(p->alpha) * sa;
        }
    }
    const double bound = 2.0 * (I / 16 + 8 + ref.experts) * kU;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double err = std::abs(out[i] - want[i]);
        rc.stage_b =
            std::max(rc.stage_b, l1[i] > 0 ? err / (bound * l1[i]) : (err == 0 ? 0.0 : 1e30));
    }
    return rc;
}

Case make_case(const std::string& name, int T, int top_k, int experts, std::mt19937_64& rng,
               const std::vector<std::vector<std::int32_t>>& host_ids, int H) {
    Case c;
    c.name   = name;
    c.tokens = T;
    c.ids.assign(static_cast<std::size_t>(T) * top_k, -1);
    c.alpha.assign(c.ids.size(), 0.0F);
    std::uniform_real_distribution<float> w(0.02F, 0.3F);
    for (int t = 0; t < T; ++t) {
        for (std::size_t s = 0; s < host_ids[t].size(); ++s)
            c.ids[t * top_k + s] = host_ids[t][s] % experts;
        for (int s = 0; s < top_k; ++s) c.alpha[t * top_k + s] = w(rng);
    }
    for (int t = 0; t < T; ++t) {
        std::vector<std::uint16_t> h;
        random_activation(H, rng, &h);
        c.x.insert(c.x.end(), h.begin(), h.end());
    }
    return c;
}

void test_layer(std::mt19937_64& rng, int E, int H, int I, bool statistics) {
    const int top_k = 10;
    LayerData L(E, H, I, rng);
    const HostExpertLayer layer = L.layer();
    std::vector<Case> cases;
    // Decode-window pattern at TP2: 3-5 missing experts per layer, 1-2 tokens each.
    cases.push_back(make_case("T4 real (4 experts, 1-2 tokens)", 4, top_k, E, rng,
                              {{3, 9}, {9, 1}, {5}, {1}}, H));
    // Every token routes 10 experts of a small bank: experts shared by up to 8 tokens (multi-token
    // T 3..8).
    {
        std::vector<std::vector<std::int32_t>> ids(8);
        for (int t = 0; t < 8; ++t) {
            std::vector<std::int32_t> all(E);
            for (int e = 0; e < E; ++e) all[e] = e;
            std::shuffle(all.begin(), all.end(), rng);
            std::swap(*std::find(all.begin(), all.end(), 0), all[0]); // expert 0 in every token
            ids[t].assign(all.begin(), all.begin() + std::min(top_k, E));
        }
        cases.push_back(make_case("T8 all host, shared", 8, top_k, E, rng, ids, H));
    }
    cases.push_back(
        make_case("T1 ten experts", 1, top_k, E, rng, {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}, H));
    cases.push_back(make_case("T3 shared by 3, one token without host experts", 3, top_k, E, rng,
                              {{2, 4}, {}, {4, 2, 5}}, H));
    cases.push_back(make_case("T4 no host expert", 4, top_k, E, rng, {{}, {}, {}, {}}, H));
    // Prefill-like call: 40 tokens, 6 host experts each from the bank, so an expert has up to ~15
    // tokens and runs in several dot groups of <= 8 (wide calls use one slice per work item).
    {
        std::vector<std::vector<std::int32_t>> ids(40);
        for (int t = 0; t < 40; ++t) {
            std::vector<std::int32_t> all(E);
            for (int e = 0; e < E; ++e) all[e] = e;
            std::shuffle(all.begin(), all.end(), rng);
            ids[t].assign(all.begin(), all.begin() + std::min(6, E));
        }
        cases.push_back(make_case("T40 prefill, dot groups", 40, top_k, E, rng, ids, H));
    }
    for (auto& c : cases) { // ids % E may repeat within a token at small E
        for (int t = 0; t < c.tokens; ++t) {
            for (int s = 0; s < top_k; ++s) {
                for (int s2 = 0; s2 < s; ++s2) {
                    if (c.ids[t * top_k + s] >= 0 && c.ids[t * top_k + s] == c.ids[t * top_k + s2])
                        c.ids[t * top_k + s] = -1;
                }
            }
        }
    }

    for (const auto& c : cases) {
        const CaseRef ref   = build_ref(L, c, top_k);
        const std::size_t n = static_cast<std::size_t>(c.tokens) * H;
        std::vector<float> first;
        for (const int threads : {1, 4, 6, 7}) {
            const auto cpus = spread_cpus(threads);
            HostCpuPool pool(cpus);
            HostMoeExecutor::Options opt;
            opt.record_members = threads == 7;
            // This test qualifies the W4A8 path against its oracle (A4 is the default).
            opt.a4 = false;
            HostMoeExecutor ex(pool, H, I, std::max(kHostMaxTokens, c.tokens), top_k, opt);
            std::vector<float> out(n, 1e30F), again(n, -1e30F);
            const HostMoeJob job2{c.tokens,     top_k,          c.x.data(),
                                  c.ids.data(), c.alpha.data(), again.data()};
            ex.run(layer, job2);
            const HostMoeJob job{c.tokens,     top_k,          c.x.data(),
                                 c.ids.data(), c.alpha.data(), out.data()};
            ex.run(layer, job);
            expect(out == again, c.name + ": repeated call differs");
            if (first.empty()) first = out;
            expect(out == first, c.name + ": result depends on the thread count (" +
                                     std::to_string(threads) + ")");
            if (threads != 6) continue;
            const RunCheck rc = check_run(ref, ex, out, c.tokens, H, I);
            expect(rc.missing == 0, c.name + ": intermediates do not match the routed pairs");
            expect(rc.bad_q == 0, c.name + ": " + std::to_string(rc.bad_q) +
                                      " int8 intermediates differ outside the bound");
            expect(rc.bad_scale == 0, c.name + ": " + std::to_string(rc.bad_scale) +
                                          " block scales outside the bound");
            expect(rc.stage_b <= 1.0,
                   c.name + ": down stage err/tol " + std::to_string(rc.stage_b));
            double d8 = 0, n8 = 0;
            for (std::size_t i = 0; i < n; ++i)
                d8 += (out[i] - ref.w4a8[i]) * (out[i] - ref.w4a8[i]),
                    n8 += ref.w4a8[i] * ref.w4a8[i];
            std::printf("layer H%d I%d %-48s: int8 h flips %d (of %d within the bound, %zu pairs), "
                        "down stage err/tol %.4f, rel L2 vs pure-W4A8 oracle %.2e\n",
                        H, I, c.name.c_str(), rc.flips, ref.ambiguous, ref.pairs.size(), rc.stage_b,
                        n8 > 0 ? std::sqrt(d8 / n8) : 0.0);
        }
        if (!statistics) continue;
        // Informative numerics of the served executor.
        double n16 = 0, d8 = 0, d4 = 0, d84 = 0;
        for (std::size_t i = 0; i < n; ++i) {
            n16 += ref.w4a16[i] * ref.w4a16[i];
            d8 += (first[i] - ref.w4a16[i]) * (first[i] - ref.w4a16[i]);
            d4 += (ref.w4a4[i] - ref.w4a16[i]) * (ref.w4a4[i] - ref.w4a16[i]);
            d84 += (first[i] - ref.w4a4[i]) * (first[i] - ref.w4a4[i]);
        }
        if (n16 > 0) {
            const double r8 = std::sqrt(d8 / n16), r4 = std::sqrt(d4 / n16),
                         r84 = std::sqrt(d84 / n16);
            std::printf("numerics %-48s rel L2 vs W4A16: host W4A8 %.4f, GPU-recipe W4A4 %.4f; "
                        "W4A8 vs W4A4 %.4f\n",
                        c.name.c_str(), r8, r4, r84);
            expect(r8 < 0.03, c.name + ": W4A8 vs W4A16 gross check");
        }
    }
}

// Issue #11: the automatic thread count leaves one core to the main thread when no SMT sibling is
// free, and is otherwise unchanged.
void test_auto_threads() {
    using ninfer::ops::host::auto_expert_cpu_threads;
    using ninfer::ops::host::HostCpuTopology;
    const auto topo = [](int logical, int physical, int l3) {
        HostCpuTopology t;
        t.logical_cpus   = logical;
        t.physical_cores = physical;
        t.l3_domains     = l3;
        t.numa_nodes     = 1;
        return t;
    };
    expect(auto_expert_cpu_threads(topo(24, 12, 2), true) == 12, "9900X GGUF: 12");
    expect(auto_expert_cpu_threads(topo(24, 12, 2), false) == 6, "9900X NVFP4: 6");
    expect(auto_expert_cpu_threads(topo(12, 12, 2), true) == 11, "9900X SMT off GGUF: 11");
    expect(auto_expert_cpu_threads(topo(12, 12, 2), false) == 6, "9900X SMT off NVFP4: 6");
    expect(auto_expert_cpu_threads(topo(32, 16, 16), false) == 16, "7F52 NVFP4: 16");
    expect(auto_expert_cpu_threads(topo(16, 16, 16), false) == 15, "7F52 taskset 0-15 NVFP4: 15");
    expect(auto_expert_cpu_threads(topo(16, 16, 16), true) == 15, "7F52 taskset 0-15 GGUF: 15");
    expect(auto_expert_cpu_threads(topo(1, 1, 1), true) == 1, "one CPU: 1");
    std::string why;
    (void)auto_expert_cpu_threads(topo(16, 16, 16), true, &why);
    expect(why.find("one core left to the main thread") != std::string::npos, "reason names it");
    std::printf("auto threads: %s\n", why.c_str());
}

// Issue #11: NINFER_HOST_MOE_CPUS=1-31 used to pin a thread to cpu -31.
void test_cpu_list() {
    using ninfer::ops::host::parse_cpu_list;
    expect(parse_cpu_list("3") == std::vector<int>{3}, "cpu list: one cpu");
    expect(parse_cpu_list("0,2,4") == std::vector<int>({0, 2, 4}), "cpu list: commas");
    expect(parse_cpu_list("1-4") == std::vector<int>({1, 2, 3, 4}), "cpu list: range");
    expect(parse_cpu_list("0-2,8,10-11\n") == std::vector<int>({0, 1, 2, 8, 10, 11}),
           "cpu list: sysfs line");
    expect(parse_cpu_list("1-31").size() == 31 && parse_cpu_list("1-31").back() == 31,
           "cpu list: 1-31");
    for (const char* bad : {"", "-3", "1-", "4-1", "1,,2", "1,", "a", "1 2", "1-2-3", "99999"}) {
        bool rejected = false;
        try {
            (void)parse_cpu_list(bad);
        } catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, std::string("cpu list rejects \"") + bad + "\"");
    }
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (!host_experts_supported()) {
        std::printf("SKIP: neither AVX2 + FMA + F16C nor AVX-512 VNNI\n");
        return 77;
    }
    std::printf("host ISA: %s\n", host_isa_name(host_isa()));
    std::mt19937_64 rng(20261007);
    test_auto_threads();
    test_cpu_list();
    test_dots(rng);
    test_layer(rng, 16, 2560, 640, true); // Flash-Next expert shape
    test_layer(rng, 5, 256, 128, false);  // small shape: one row tile, odd expert count
    std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "PASSED", g_failures);
    return g_failures ? 1 : 0;
}
