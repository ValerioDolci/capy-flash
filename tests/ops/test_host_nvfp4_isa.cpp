// The host NVFP4 kernels of every ISA against each other and against a scalar model of their
// operation order, bit for bit (host_nvfp4.h: "identical bit for bit").
//
// Scalar model: integer block sums per 4-element lane, then per 64-weight group and lane one FP32
// product (e4m3 weight scale x activation block scale) and one fused multiply-add into that lane's
// accumulator (std::fmaf), the 16 lanes reduced in _mm512_reduce_add_ps's order, out = s * 0.5 /
// divisor; the int8 and A4 codecs element by element with vpmovdb's truncation.
//
// Criteria (pre-registered): zero differing bits in every comparison below.
//   1. codecs: quantize_bf16 / quantize_blocks / quantize_*_a4 (q, qu, corr, sb16, asb) on random
//      activations with outliers, all-zero blocks, tiny and huge values, A4 grid ties;
//   2. dots: rows_dot, T = 1..8, synthetic planes
//      at k = 2560, 640, 1088 (16-block steps + 4-block tail), 192 (tail only), 64;
//   3. real rows (NINFER_HOST_NVFP4_BANKS = directory with layer_00.nvfp4.{gate_up,down}.bin, the
//      Flash-Next layer-0 banks as stored in the artifact): every row of 24 random experts, gate/up
//      and down, T = 1, 3, 4, 8, W4A8 and A4 activations; the scalar model on 4 of them;
//   4. the layer executor (HostMoeExecutor, 6 threads, a4 on and off, decode windows T = 1..8
//      (Static) and a 24-token prefill chunk (Dynamic)) with isa = Avx2 vs Avx512Vnni on the real
//      layer (or on a synthetic one without NINFER_HOST_NVFP4_BANKS).
// ISAs the CPU lacks are skipped (on a CPU without AVX-512 the test compares AVX2 to the model).
#include "host_moe_fixture.h"
#include "ops/sparse_moe/host/host_moe_executor.h"
#include "ops/sparse_moe/host/host_nvfp4_isa.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

using namespace ninfer::ops::host;
using namespace ninfer::test::host_moe;

namespace {

int g_failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++g_failures;
        std::printf("FAIL %s\n", what.c_str());
    }
}

std::vector<HostIsa> g_isas; // supported ISAs, widest first

// ---------------------------------------------------------------- scalar model
const std::int8_t kW[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

std::int8_t truncate8(float v) { // vcvtps2dq (nearest even, indefinite = INT_MIN) then vpmovdb
    std::int32_t q = std::numeric_limits<std::int32_t>::min();
    if (std::isfinite(v) && std::fabs(v) < 2147483520.0F)
        q = static_cast<std::int32_t>(std::nearbyint(v));
    return static_cast<std::int8_t>(
        static_cast<std::uint8_t>(static_cast<std::uint32_t>(q) & 0xFFU));
}

int stored_element(int p) { return p < 8 ? 2 * p : 2 * (p - 8) + 1; }

void model_store(const std::int8_t* q, float s, int b, W4A8Activation a) {
    for (int p = 0; p < 16; ++p) {
        a.q[b * 16 + p]  = q[p];
        a.qu[b * 16 + p] = static_cast<std::uint8_t>(q[p] ^ static_cast<std::int8_t>(0x80));
    }
    for (int d = 0; d < 4; ++d) {
        a.corr[b * 4 + d] = -12 * (q[4 * d] + q[4 * d + 1] + q[4 * d + 2] + q[4 * d + 3]);
        a.sb16[b * 4 + d] = s;
    }
    a.asb[b] = s;
}

void model_quantize(const float* x, int block0, int blocks, W4A8Activation a) {
    for (int i = 0; i < blocks; ++i) {
        const float* v = x + i * 16;
        float amax     = 0.0F;
        for (int j = 0; j < 16; ++j) amax = std::max(amax, std::fabs(v[j]));
        const float s   = amax > 0.0F ? amax / 127.0F : 1.0F;
        const float inv = 1.0F / s;
        std::int8_t q[16];
        for (int p = 0; p < 16; ++p) q[p] = truncate8(v[stored_element(p)] * inv);
        model_store(q, s, block0 + i, a);
    }
}

void model_quantize_a4(const float* x, int block0, int blocks, float gs, W4A8Activation a) {
    for (int i = 0; i < blocks; ++i) {
        const float* v = x + i * 16;
        float amax     = 0.0F;
        for (int j = 0; j < 16; ++j) amax = std::max(amax, std::fabs(v[j]));
        const float sf        = detail::a4_e4m3_rn(gs * (amax * (1.0F / 6.0F)));
        const float out_scale = (amax != 0.F && sf != 0.F) ? 1.0F / (sf * (1.0F / gs)) : 0.F;
        std::int8_t q[16];
        for (int p = 0; p < 16; ++p) {
            const float y  = v[stored_element(p)] * out_scale;
            const float ya = std::fabs(y);
            float r        = std::nearbyint(ya * 2.0F);
            if (ya > 2.0F) r = std::nearbyint(ya) * 2.0F;
            if (ya > 4.0F) r = std::min(std::nearbyint(ya * 0.5F) * 4.0F, 12.0F);
            const std::int8_t c = truncate8(r);
            q[p]                = y < 0.0F ? static_cast<std::int8_t>(-c) : c;
        }
        model_store(q, sf / (2.0F * gs), block0 + i, a);
    }
}

std::vector<float> bf16_floats(const std::uint16_t* x, int k) {
    std::vector<float> f(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) f[i] = from_bf16(x[i]);
    return f;
}

float reduce16(const float* l) {
    float s8[8], s4[4];
    for (int i = 0; i < 8; ++i) s8[i] = l[i + 8] + l[i];
    for (int i = 0; i < 4; ++i) s4[i] = s8[i + 4] + s8[i];
    return (s4[0] + s4[2]) + (s4[1] + s4[3]);
}

std::int32_t lane_sum(const Nvfp4Plane& p, std::int64_t r, int blk, int lane4,
                      const W4A8Activation& a) {
    const std::uint8_t* codes = p.row_codes(r) + static_cast<std::size_t>(blk) * 8;
    std::int32_t s            = 0;
    for (int p4 = 0; p4 < 4; ++p4) {
        const int pos  = 4 * lane4 + p4;
        const int e    = stored_element(pos);
        const int code = e % 2 ? codes[e / 2] >> 4 : codes[e / 2] & 15;
        s += kW[code] * a.q[blk * 16 + pos];
    }
    return s;
}

float scale_of(const Nvfp4Plane& p, std::int64_t r, int blk) {
    return ninfer::ops::host::e4m3_value(
        p.row_scales(r)[static_cast<std::size_t>(blk / 4) * kScaleQuadStride + blk % 4]);
}

// lanes[j] += di * f over the 64-weight group of blocks b..b+3
void model_group(const Nvfp4Plane& p, std::int64_t r, int b, const W4A8Activation& a, float* acc) {
    for (int j = 0; j < 16; ++j) {
        const int blk = b + j / 4;
        const float f = scale_of(p, r, blk) * a.sb16[b * 4 + j];
        acc[j]        = std::fmaf(static_cast<float>(lane_sum(p, r, blk, j % 4, a)), f, acc[j]);
    }
}

float model_dot(const Nvfp4Plane& p, std::int64_t r, const W4A8Activation& a) {
    const int nb  = p.k / 16;
    float acc[16] = {};
    for (int b = 0; b < nb; b += 4) model_group(p, r, b, a, acc);
    return reduce16(acc) * 0.5F / p.divisor(r);
}

// ---------------------------------------------------------------- helpers
std::vector<std::uint16_t> random_bf16(int k, std::mt19937_64& rng, int variant) {
    std::normal_distribution<double> normal(0.0, 1.0);
    std::lognormal_distribution<double> gain(0.0, 0.7);
    std::vector<std::uint16_t> x(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) {
        double v = normal(rng) * gain(rng);
        if (i % 641 == 7) v *= 30.0;                                   // outlier channels
        if (variant == 1 && (i / 16) % 7 == 3) v = 0.0;                // all-zero blocks
        if (variant == 2 && (i / 16) % 5 == 1) v *= 1e-30;             // tiny blocks
        if (variant == 2 && (i / 16) % 5 == 2) v *= 1e30;              // huge blocks
        if (variant == 3) v = std::ldexp(std::nearbyint(v * 4.0), -3); // A4 grid ties
        x[static_cast<std::size_t>(i)] = to_bf16(static_cast<float>(v));
    }
    return x;
}

float a4_global_scale(const std::vector<std::uint16_t>& x) { // 448 * 6 / amax, as the GPU recipe
    float amax = 0.0F;
    for (const std::uint16_t h : x) amax = std::max(amax, std::fabs(from_bf16(h)));
    return amax > 0.0F ? 448.0F * 6.0F / amax : 1.0F;
}

bool same_act(const W4A8Activation& x, const W4A8Activation& y, int k) {
    return !std::memcmp(x.q, y.q, k) && !std::memcmp(x.qu, y.qu, k) &&
           !std::memcmp(x.corr, y.corr, sizeof(std::int32_t) * (k / 4)) &&
           !std::memcmp(x.sb16, y.sb16, sizeof(float) * (k / 4)) &&
           !std::memcmp(x.asb, y.asb, sizeof(float) * (k / 16));
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && !std::memcmp(a.data(), b.data(), sizeof(float) * a.size());
}

// Activations of T tokens for each ISA (bit-identical, checked) and their prepared views.
struct Acts {
    std::vector<W4A8ActivationBuffer> buf; // per ISA
    std::vector<std::vector<W4A8Activation>> view;
};

Acts make_acts(int k, int T, bool a4, std::mt19937_64& rng, const std::string& what) {
    Acts acts;
    std::vector<std::vector<std::uint16_t>> xs;
    for (int t = 0; t < T; ++t) xs.push_back(random_bf16(k, rng, t % 4));
    for (const HostIsa isa : g_isas) {
        const HostNvfp4Kernels& kern = host_nvfp4_kernels(isa);
        acts.buf.emplace_back(k, T);
        acts.view.emplace_back();
        for (int t = 0; t < T; ++t) {
            const W4A8Activation a = acts.buf.back().at(t);
            if (a4)
                kern.quantize_bf16_a4(xs[t].data(), k, a4_global_scale(xs[t]), a);
            else
                kern.quantize_bf16(xs[t].data(), k, a);
            acts.view.back().push_back(a);
        }
    }
    for (std::size_t i = 1; i < g_isas.size(); ++i)
        for (int t = 0; t < T; ++t)
            expect(same_act(acts.view[0][t], acts.view[i][t], k), what + ": activation codec");
    return acts;
}

struct DotTally {
    long rows = 0, isa_diff = 0, model_rows = 0, model_diff = 0;
};

// Rows [r0, r0 + n) of `p` for T tokens: every ISA, and the model on rows where `model`.
void check_rows(const Nvfp4Plane& p, std::int64_t r0, int n, const Acts& acts, int T, bool model,
                DotTally& tally) {
    std::vector<std::vector<float>> out(g_isas.size(),
                                        std::vector<float>(static_cast<std::size_t>(T) * n));
    for (std::size_t i = 0; i < g_isas.size(); ++i) {
        host_nvfp4_kernels(g_isas[i]).rows_dot(p, r0, n, acts.view[i].data(), T, out[i].data(),
                                               static_cast<std::size_t>(n));
    }
    tally.rows += n;
    for (std::size_t i = 1; i < g_isas.size(); ++i)
        if (!same_bits(out[0], out[i])) {
            for (std::size_t j = 0; j < out[0].size(); ++j)
                tally.isa_diff += std::memcmp(&out[0][j], &out[i][j], 4) != 0;
        }
    if (!model) return;
    tally.model_rows += n;
    for (int i = 0; i < n; ++i)
        for (int t = 0; t < T; ++t) {
            const float m = model_dot(p, r0 + i, acts.view[0][t]);
            tally.model_diff +=
                std::memcmp(&m, &out[0][static_cast<std::size_t>(t) * n + i], 4) != 0;
        }
}

// ---------------------------------------------------------------- 1. codecs
void test_codecs(std::mt19937_64& rng) {
    long blocks = 0, diff = 0;
    for (const int k : {2560, 640, 64}) {
        for (int variant = 0; variant < 4; ++variant) {
            for (int rep = 0; rep < 8; ++rep) {
                const std::vector<std::uint16_t> x = random_bf16(k, rng, variant);
                const std::vector<float> xf        = bf16_floats(x.data(), k);
                const float gs                     = a4_global_scale(x) * (rep % 2 ? 0.37F : 1.0F);
                for (int mode = 0; mode < 4; ++mode) { // bf16, f32 blocks, a4 bf16, a4 f32 blocks
                    W4A8ActivationBuffer ref(k, 1);
                    if (mode < 2)
                        model_quantize(xf.data(), 0, k / 16, ref.at(0));
                    else
                        model_quantize_a4(xf.data(), 0, k / 16, gs, ref.at(0));
                    for (const HostIsa isa : g_isas) {
                        const HostNvfp4Kernels& kern = host_nvfp4_kernels(isa);
                        W4A8ActivationBuffer got(k, 1);
                        const W4A8Activation a = got.at(0);
                        switch (mode) {
                        case 0:
                            kern.quantize_bf16(x.data(), k, a);
                            break;
                        case 1: // two calls: blocks [0, 1) and [1, k/16)
                            kern.quantize_blocks(xf.data(), 0, 1, a);
                            kern.quantize_blocks(xf.data() + 16, 1, k / 16 - 1, a);
                            break;
                        case 2:
                            kern.quantize_bf16_a4(x.data(), k, gs, a);
                            break;
                        default:
                            kern.quantize_blocks_a4(xf.data(), 0, 1, gs, a);
                            kern.quantize_blocks_a4(xf.data() + 16, 1, k / 16 - 1, gs, a);
                        }
                        blocks += k / 16;
                        if (!same_act(a, ref.at(0), k)) {
                            ++diff;
                            std::printf("  codec mismatch: %s k=%d variant %d mode %d\n",
                                        host_isa_name(isa), k, variant, mode);
                        }
                    }
                }
            }
        }
    }
    std::printf("codecs: %ld blocks (all ISAs x 4 codec modes), %ld mismatching activations\n",
                blocks, diff);
    expect(diff == 0, "codecs vs scalar model");
}

// ---------------------------------------------------------------- 2. synthetic dots
void test_synthetic_dots(std::mt19937_64& rng) {
    DotTally tally;
    for (const int k : {2560, 640, 1088, 192, 64}) {
        PlaneData pd(256, k, 2, rng);
        const Nvfp4Plane plane = pd.plane();
        for (const bool a4 : {false, true}) {
            const Acts acts =
                make_acts(k, kHostMaxTokens, a4, rng, "synthetic k=" + std::to_string(k));
            for (int T = 1; T <= kHostMaxTokens; ++T) {
                check_rows(plane, 0, 64, acts, T, true, tally);
                check_rows(plane, 128 + 37, 19, acts, T, T <= 2, tally);
            }
        }
    }
    std::printf("synthetic dots: %ld row calls x tokens; ISA-vs-ISA differing outputs %ld; model: "
                "%ld rows, %ld differing outputs\n",
                tally.rows, tally.isa_diff, tally.model_rows, tally.model_diff);
    expect(tally.isa_diff == 0, "synthetic dots: ISA vs ISA");
    expect(tally.model_diff == 0, "synthetic dots: vs scalar model");
}

// ---------------------------------------------------------------- 3. real rows
struct Mapped {
    void* base        = nullptr;
    std::size_t bytes = 0;

    explicit Mapped(const std::string& path) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return;
        struct stat st{};
        if (::fstat(fd, &st) == 0) {
            bytes = static_cast<std::size_t>(st.st_size);
            base  = ::mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
            if (base == MAP_FAILED) base = nullptr;
        }
        ::close(fd);
    }

    ~Mapped() {
        if (base) ::munmap(base, bytes);
    }
};

constexpr int kE = 512, kH = 2560, kI = 640;

struct RealLayer {
    Mapped gu, dn;
    HostExpertLayer layer;
    bool ok = false;

    explicit RealLayer(const std::string& dir)
        : gu(dir + "/layer_00.nvfp4.gate_up.bin"), dn(dir + "/layer_00.nvfp4.down.bin") {
        const std::int64_t gu_rows = std::int64_t{kE} * 2 * kI, dn_rows = std::int64_t{kE} * kH;
        if (!gu.base || !dn.base || gu.bytes != Nvfp4Plane::payload_bytes(gu_rows, kH, 2 * kE) ||
            dn.bytes != Nvfp4Plane::payload_bytes(dn_rows, kI, kE)) {
            std::printf("real banks: missing or wrong size in %s\n", dir.c_str());
            return;
        }
        layer = {Nvfp4Plane::from_payload(gu.base, gu_rows, kH, 2 * kE),
                 Nvfp4Plane::from_payload(dn.base, dn_rows, kI, kE),
                 kE,
                 kH,
                 kI,
                 1.0F,
                 1.0F};
        ok    = true;
    }
};

void test_real_rows(const RealLayer& real, std::mt19937_64& rng) {
    DotTally tally;
    std::uniform_int_distribution<int> pick(0, kE - 1);
    for (int n = 0; n < 24; ++n) {
        const int e      = pick(rng);
        const bool model = n < 4;
        for (const bool a4 : {false, true}) {
            for (const int T : {1, 3, 4, 8}) {
                if (model && T > 3) continue; // the scalar model is slow; T = 1, 3 cover both V
                const Acts x = make_acts(kH, T, a4, rng, "real gate/up");
                const Acts h = make_acts(kI, T, a4, rng, "real down");
                check_rows(real.layer.gate_up, std::int64_t{e} * 2 * kI, 2 * kI, x, T,
                           model && T == 1, tally);
                check_rows(real.layer.down, std::int64_t{e} * kH, kH, h, T, model && T == 1, tally);
            }
        }
    }
    std::printf("real rows (layer 0, 24 experts): %ld rows; ISA-vs-ISA differing outputs %ld; "
                "model: %ld rows, %ld differing outputs\n",
                tally.rows, tally.isa_diff, tally.model_rows, tally.model_diff);
    expect(tally.isa_diff == 0, "real rows: ISA vs ISA");
    expect(tally.model_diff == 0, "real rows: vs scalar model");
}

// ---------------------------------------------------------------- 4. executor
void test_executor(const HostExpertLayer& layer, std::mt19937_64& rng, const char* what) {
    if (g_isas.size() < 2) {
        std::printf("executor (%s): one ISA only, skipped\n", what);
        return;
    }
    const std::vector<int> cpus = spread_cpus(6);
    HostCpuPool pool(cpus);
    long calls = 0, diff = 0;
    std::uniform_int_distribution<int> pick(0, layer.experts - 1);
    std::uniform_real_distribution<float> weight(0.01F, 0.3F);
    HostExpertLayer l       = layer;
    l.gate_up_input_divisor = 8.0F;
    l.down_input_divisor    = 40.0F;
    for (const bool a4 : {true, false}) {
        std::vector<std::unique_ptr<HostMoeExecutor>> ex;
        for (const HostIsa isa : g_isas) {
            HostMoeExecutor::Options o;
            o.a4  = a4;
            o.isa = isa;
            ex.push_back(std::make_unique<HostMoeExecutor>(pool, layer.hidden, layer.intermediate,
                                                           24, 10, o));
        }
        for (const int T : {1, 2, 3, 4, 5, 6, 7, 8, 24}) {
            std::vector<std::uint16_t> x;
            for (int t = 0; t < T; ++t) {
                const auto r = random_bf16(layer.hidden, rng, 0);
                x.insert(x.end(), r.begin(), r.end());
            }
            std::vector<std::int32_t> ids(static_cast<std::size_t>(T) * 10);
            std::vector<float> alpha(ids.size());
            const int pool_experts = std::min(layer.experts, 20); // tokens share experts
            for (std::size_t i = 0; i < ids.size(); ++i) {
                ids[i]   = (i % 10 < 3) ? -1 : (pick(rng) % pool_experts);
                alpha[i] = weight(rng);
            }
            for (int t = 0; t < T; ++t) { // distinct experts within a token
                std::sort(ids.begin() + t * 10, ids.begin() + t * 10 + 10);
                for (int s = 1; s < 10; ++s)
                    if (ids[t * 10 + s] >= 0 && ids[t * 10 + s] == ids[t * 10 + s - 1])
                        ids[t * 10 + s] = -1;
            }
            std::vector<std::vector<float>> out(
                g_isas.size(), std::vector<float>(static_cast<std::size_t>(T) * layer.hidden));
            for (std::size_t i = 0; i < g_isas.size(); ++i) {
                HostMoeJob job{T, 10, x.data(), ids.data(), alpha.data(), out[i].data()};
                (void)ex[i]->run(l, job);
            }
            ++calls;
            for (std::size_t i = 1; i < g_isas.size(); ++i) diff += !same_bits(out[0], out[i]);
        }
    }
    std::printf("executor (%s): %ld calls x %zu ISAs, %ld calls with differing outputs\n", what,
                calls, g_isas.size(), diff);
    expect(diff == 0, std::string("executor ISA vs ISA: ") + what);
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    for (const HostIsa isa : {HostIsa::Avx512Vnni, HostIsa::Avx2})
        if (host_isa_supported(isa)) g_isas.push_back(isa);
    if (g_isas.empty()) {
        std::printf("SKIP: neither AVX2 + FMA + F16C nor AVX-512 VNNI\n");
        return 77;
    }
    std::printf("ISAs:");
    for (const HostIsa isa : g_isas) std::printf(" %s", host_isa_name(isa));
    std::printf(" (+ scalar model)\n");
    std::mt19937_64 rng(20261019);
    test_codecs(rng);
    test_synthetic_dots(rng);
    if (const char* dir = std::getenv("NINFER_HOST_NVFP4_BANKS"); dir != nullptr && *dir) {
        const RealLayer real(dir);
        expect(real.ok, "real banks readable");
        if (real.ok) {
            test_real_rows(real, rng);
            test_executor(real.layer, rng, "real layer 0");
        }
    } else {
        std::printf("real rows: NINFER_HOST_NVFP4_BANKS not set, skipped\n");
        LayerData data(24, 2560, 640, rng);
        test_executor(data.layer(), rng, "synthetic 24 experts");
    }
    std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "PASSED", g_failures);
    return g_failures ? 1 : 0;
}
