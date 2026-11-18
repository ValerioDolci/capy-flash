// Host (CPU) GGUF routed-expert kernels (ops/host_moe/host_gguf.h, the fngine port) against
// ggml's reference code (third_party/ggml-iq: dequantize_row_*, quantize_row_q8_{K,0}_ref,
// ggml_vec_dot_*_generic).
//
// Rows: synthetic random blocks of every format (always) and, when a bank directory is given
// (argv[1] or NINFER_GGUF_BANKS; tools/convert/gguf_experts.py extract), real rows of every bank
// file it lists, spread over the 512 experts.
//
// Criteria (each format, T = 1..8 tokens):
//  - quantizers: quantize_row_q8_K_bf16 / q8_0_bf16 equal ggml's quantize_row_q8_K_ref /
//    q8_0_ref of the same values, byte for byte;
//  - dot: |kernel - ggml generic| <= 2^-17 * sum_i |w_i a_i| (both compute the same exact integer
//    sums per sub-block; they differ only in FP32 scale steps and accumulation order: a handful of
//    roundings of partial sums, each <= 2^-24 of a bound by sum |w a|);
//  - independence: a token's result is bit-identical for every T and every row tile (1 or 8 rows);
//  - BF16 dequantization (prefill path): decode_rows_bf16 equals RNE_bf16(dequantize_row), every
//    element;
//  - informative: relative L2 error of the kernel against FP64 products with the unquantized BF16
//    activations (the cost of the Q8 activation codec), printed per format.
#include "ops/host_moe/host_gguf.h"
#include "ops/host_moe/host_q8.h"

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

#include <fcntl.h>
#include <unistd.h>

namespace g = ninfer::ops::host::gguf;
using g::ActFormat;
using g::WeightFormat;

namespace {

constexpr int kMaxT = 8;
int g_failures      = 0;

#define CHECK(cond, ...)                                                                           \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            ++g_failures;                                                                          \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                              \
            std::fprintf(stderr, __VA_ARGS__);                                                     \
            std::fprintf(stderr, "\n");                                                            \
        }                                                                                          \
    } while (0)

std::uint16_t bf16_rne(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<std::uint16_t>((u + 0x7FFFU + ((u >> 16) & 1U)) >> 16);
}

float bf16_value(std::uint16_t b) {
    const std::uint32_t u = std::uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct AlignedBuf {
    std::uint8_t* p = nullptr;

    explicit AlignedBuf(std::size_t n) {
        p = static_cast<std::uint8_t*>(std::aligned_alloc(64, (n + 63) / 64 * 64));
        std::memset(p, 0, (n + 63) / 64 * 64);
    }

    ~AlignedBuf() { std::free(p); }

    AlignedBuf(const AlignedBuf&)            = delete;
    AlignedBuf& operator=(const AlignedBuf&) = delete;
};

void dequantize(WeightFormat fmt, const void* row, float* y, int k) {
    switch (fmt) {
    case WeightFormat::IQ2_S:
        dequantize_row_iq2_s(static_cast<const block_iq2_s*>(row), y, k);
        return;
    case WeightFormat::IQ3_XXS:
        dequantize_row_iq3_xxs(static_cast<const block_iq3_xxs*>(row), y, k);
        return;
    case WeightFormat::IQ3_S:
        dequantize_row_iq3_s(static_cast<const block_iq3_s*>(row), y, k);
        return;
    case WeightFormat::IQ4_XS:
        dequantize_row_iq4_xs(static_cast<const block_iq4_xs*>(row), y, k);
        return;
    case WeightFormat::IQ4_NL:
        dequantize_row_iq4_nl(static_cast<const block_iq4_nl*>(row), y, k);
        return;
    case WeightFormat::Q2_0:
        dequantize_row_q2_0(static_cast<const block_q2_0*>(row), y, k);
        return;
    }
}

float generic_dot(WeightFormat fmt, const void* row, const void* act, int k) {
    float s = 0.0F;
    switch (fmt) {
    case WeightFormat::IQ2_S:
        ggml_vec_dot_iq2_s_q8_K_generic(k, &s, 0, row, 0, act, 0, 1);
        break;
    case WeightFormat::IQ3_XXS:
        ggml_vec_dot_iq3_xxs_q8_K_generic(k, &s, 0, row, 0, act, 0, 1);
        break;
    case WeightFormat::IQ3_S:
        ggml_vec_dot_iq3_s_q8_K_generic(k, &s, 0, row, 0, act, 0, 1);
        break;
    case WeightFormat::IQ4_XS:
        ggml_vec_dot_iq4_xs_q8_K_generic(k, &s, 0, row, 0, act, 0, 1);
        break;
    case WeightFormat::IQ4_NL:
        ggml_vec_dot_iq4_nl_q8_0_generic(k, &s, 0, row, 0, act, 0, 1);
        break;
    case WeightFormat::Q2_0:
        ggml_vec_dot_q2_0_q8_0_generic(k, &s, 0, row, 0, act, 0, 1);
        break;
    }
    return s;
}

// Rows of one format: `rows` rows of row_bytes.
struct RowSet {
    std::string label;
    WeightFormat fmt;
    int k                 = 0;
    int rows              = 0;
    std::size_t row_bytes = 0;
    std::vector<std::uint8_t> data;
};

std::uint16_t half_bits(float f) { return GGML_FP32_TO_FP16(f); }

RowSet synthetic(WeightFormat fmt, int k, int rows, std::mt19937_64& rng) {
    RowSet s{std::string("synthetic ") + g::format_name(fmt),
             fmt,
             k,
             rows,
             g::weight_row_bytes(fmt, k),
             {}};
    s.data.resize(s.row_bytes * rows);
    for (auto& b : s.data) b = static_cast<std::uint8_t>(rng());
    const g::FormatInfo fi = g::format_info(fmt);
    std::uniform_real_distribution<float> d(1e-3F, 4e-2F);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < k / fi.block_elems; ++b) {
            const std::uint16_t h = half_bits(d(rng) * ((rng() & 1) ? 1.F : -1.F));
            std::memcpy(s.data.data() + r * s.row_bytes + std::size_t(b) * fi.block_bytes, &h, 2);
        }
    }
    return s;
}

bool format_of(const std::string& name, WeightFormat* out) {
    static const std::pair<const char*, WeightFormat> kMap[] = {
        {"gguf_iq2_s", WeightFormat::IQ2_S},   {"gguf_iq3_xxs", WeightFormat::IQ3_XXS},
        {"gguf_iq3_s", WeightFormat::IQ3_S},   {"gguf_iq4_xs", WeightFormat::IQ4_XS},
        {"gguf_iq4_nl", WeightFormat::IQ4_NL}, {"gguf_q2_0", WeightFormat::Q2_0}};
    for (const auto& [n, f] : kMap) {
        if (name == n) {
            *out = f;
            return true;
        }
    }
    return false;
}

// `rows` rows of a bank file, spread evenly over its row range.
std::vector<RowSet> bank_rows(const std::string& dir, int rows) {
    std::vector<RowSet> out;
    std::ifstream in(dir + "/manifest.json");
    if (!in) return out;
    const auto manifest = nlohmann::json::parse(in);
    for (const auto& [layer, entry] : manifest.at("layers").items()) {
        for (const char* role : {"gate_up", "down"}) {
            const auto& e = entry.at(role);
            WeightFormat fmt;
            if (!format_of(e.at("format").get<std::string>(), &fmt)) continue;
            RowSet s{"layer " + layer + " " + role + " " + g::format_name(fmt),
                     fmt,
                     e.at("shape")[1].get<int>(),
                     rows,
                     e.at("row_bytes").get<std::size_t>(),
                     {}};
            const long long total = e.at("shape")[0].get<long long>();
            s.data.resize(s.row_bytes * rows);
            const int fd = ::open((dir + "/" + e.at("file").get<std::string>()).c_str(), O_RDONLY);
            if (fd < 0) continue;
            for (int r = 0; r < rows; ++r) {
                const long long row = (total - 1) * r / std::max(rows - 1, 1);
                const ssize_t got =
                    ::pread(fd, s.data.data() + r * s.row_bytes, s.row_bytes,
                            static_cast<off_t>(row) * static_cast<off_t>(s.row_bytes));
                CHECK(got == static_cast<ssize_t>(s.row_bytes), "%s: short read", s.label.c_str());
            }
            ::close(fd);
            out.push_back(std::move(s));
        }
    }
    return out;
}

void check_rows(const RowSet& s, std::mt19937_64& rng) {
    const int k        = s.k;
    const ActFormat af = g::format_info(s.fmt).act;
    // Activations: BF16 columns with varied scales, a few outliers, one all-zero 32-block.
    std::normal_distribution<float> nd(0.F, 1.F);
    std::vector<std::uint16_t> x(std::size_t(kMaxT) * k);
    for (int t = 0; t < kMaxT; ++t) {
        const float scale = 0.05F + 2.F * float(t) / kMaxT;
        for (int i = 0; i < k; ++i) {
            float v = nd(rng) * scale;
            if (i % 97 == 13) v *= 20.F;
            if (t == 3 && i < 32) v = 0.F;
            x[std::size_t(t) * k + i] = bf16_rne(v);
        }
    }
    // (1) quantizers vs ggml's reference, and the packed activations.
    const std::size_t arb = g::act_row_bytes(af, k), pb = g::packed_act_bytes(af, k);
    std::vector<std::uint8_t> ours(arb * kMaxT), ref(arb * kMaxT);
    AlignedBuf packed(pb * kMaxT);
    const void* acts[kMaxT];
    std::vector<float> xf(k);
    for (int t = 0; t < kMaxT; ++t) {
        const std::uint16_t* xt = x.data() + std::size_t(t) * k;
        for (int i = 0; i < k; ++i) xf[i] = bf16_value(xt[i]);
        if (af == ActFormat::Q8_K) {
            g::quantize_row_q8_K_bf16(xt, reinterpret_cast<block_q8_K*>(ours.data() + arb * t), k);
            quantize_row_q8_K_ref(xf.data(), reinterpret_cast<block_q8_K*>(ref.data() + arb * t),
                                  k);
        } else {
            g::quantize_row_q8_0_bf16(xt, reinterpret_cast<block_q8_0*>(ours.data() + arb * t), k);
            quantize_row_q8_0_ref(xf.data(), reinterpret_cast<block_q8_0*>(ref.data() + arb * t),
                                  k);
        }
        g::quantize_pack_bf16(af, xt, k, packed.p + pb * t);
        acts[t] = packed.p + pb * t;
    }
    // ggml's reference leaves bsums of an all-zero Q8_K block unwritten (the port writes 0): no
    // such block here except t = 3's first 32 values, which do not zero a whole 256-block.
    CHECK(std::memcmp(ours.data(), ref.data(), ours.size()) == 0,
          "%s: Q8 quantizer differs from ggml ref", s.label.c_str());

    // (2)-(4) dots.
    const std::size_t drb = g::decoded_row_bytes(s.fmt, k);
    AlignedBuf dec(drb * 8);
    std::vector<float> w(k);
    double worst_generic = 0.0, err2 = 0.0, ref2 = 0.0;
    int tile_mismatch = 0, t_mismatch = 0;
    for (int r0 = 0; r0 < s.rows; r0 += 8) {
        const int nr             = std::min(8, s.rows - r0);
        const std::uint8_t* rows = s.data.data() + std::size_t(r0) * s.row_bytes;
        g::decode_rows(s.fmt, rows, s.row_bytes, nr, k, dec.p);
        float out8[8 * kMaxT];
        g::gemm_decoded(s.fmt, dec.p, nr, k, acts, kMaxT, out8, kMaxT);
        for (int r = 0; r < nr; ++r) {
            const std::uint8_t* row = rows + std::size_t(r) * s.row_bytes;
            // Single-row tile and every T must give the same bits.
            AlignedBuf one(drb);
            g::decode_rows(s.fmt, row, s.row_bytes, 1, k, one.p);
            for (int T = 1; T <= kMaxT; ++T) {
                float o[kMaxT];
                g::gemm_decoded(s.fmt, one.p, 1, k, acts, T, o, kMaxT);
                for (int t = 0; t < T; ++t) {
                    if (std::memcmp(&o[t], &out8[r * kMaxT + t], 4) != 0) {
                        T == kMaxT ? ++tile_mismatch : ++t_mismatch;
                    }
                }
            }
            dequantize(s.fmt, row, w.data(), k);
            for (int t = 0; t < kMaxT; ++t) {
                const float got = out8[r * kMaxT + t];
                const float gen = generic_dot(s.fmt, row, ours.data() + arb * t, k);
                // sum |w a| over the quantized activation values.
                std::vector<float> aq(k);
                if (af == ActFormat::Q8_K)
                    dequantize_row_q8_K(reinterpret_cast<const block_q8_K*>(ours.data() + arb * t),
                                        aq.data(), k);
                else
                    dequantize_row_q8_0(reinterpret_cast<const block_q8_0*>(ours.data() + arb * t),
                                        aq.data(), k);
                double mag = 0.0, exact = 0.0;
                for (int i = 0; i < k; ++i) {
                    mag += std::fabs(double(w[i]) * aq[i]);
                    exact += double(w[i]) * bf16_value(x[std::size_t(t) * k + i]);
                }
                const double rel =
                    mag > 0 ? std::fabs(double(got) - gen) / mag : std::fabs(double(got) - gen);
                worst_generic = std::max(worst_generic, rel);
                err2 += (got - exact) * (got - exact);
                ref2 += exact * exact;
            }
        }
    }
    CHECK(worst_generic <= std::ldexp(1.0, -17), "%s: |kernel - ggml generic| = %.3g sum|wa|",
          s.label.c_str(), worst_generic);
    CHECK(tile_mismatch == 0, "%s: %d results depend on the row tile", s.label.c_str(),
          tile_mismatch);
    CHECK(t_mismatch == 0, "%s: %d results depend on T", s.label.c_str(), t_mismatch);

    // (4b) Q8_K formats with per-32 activation scales (quantize_pack_bf16_q8k_g32): against FP64
    // products of the same Q8_0-quantized activations (no ggml routine pairs these), same bound.
    double g32_worst = 0.0, g32_err2 = 0.0, g32_ref2 = 0.0;
    if (af == ActFormat::Q8_K) {
        AlignedBuf fine(pb * kMaxT);
        const void* facts[kMaxT];
        std::vector<std::uint8_t> q0(g::act_row_bytes(ActFormat::Q8_0, k) * kMaxT);
        const std::size_t q0b = g::act_row_bytes(ActFormat::Q8_0, k);
        for (int t = 0; t < kMaxT; ++t) {
            g::quantize_pack_bf16_q8k_g32(x.data() + std::size_t(t) * k, k, fine.p + pb * t);
            facts[t] = fine.p + pb * t;
            g::quantize_row_q8_0_bf16(x.data() + std::size_t(t) * k,
                                      reinterpret_cast<block_q8_0*>(q0.data() + q0b * t), k);
        }
        std::vector<float> aq(k);
        for (int r0 = 0; r0 < s.rows; r0 += 8) {
            const int nr             = std::min(8, s.rows - r0);
            const std::uint8_t* rows = s.data.data() + std::size_t(r0) * s.row_bytes;
            g::decode_rows(s.fmt, rows, s.row_bytes, nr, k, dec.p);
            float out8[8 * kMaxT];
            g::gemm_decoded(s.fmt, dec.p, nr, k, facts, kMaxT, out8, kMaxT);
            for (int r = 0; r < nr; ++r) {
                dequantize(s.fmt, rows + std::size_t(r) * s.row_bytes, w.data(), k);
                for (int t = 0; t < kMaxT; ++t) {
                    dequantize_row_q8_0(reinterpret_cast<const block_q8_0*>(q0.data() + q0b * t),
                                        aq.data(), k);
                    double mag = 0.0, emu = 0.0, exact = 0.0;
                    for (int i = 0; i < k; ++i) {
                        mag += std::fabs(double(w[i]) * aq[i]);
                        emu += double(w[i]) * aq[i];
                        exact += double(w[i]) * bf16_value(x[std::size_t(t) * k + i]);
                    }
                    const double got = out8[r * kMaxT + t];
                    g32_worst        = std::max(g32_worst, mag > 0 ? std::fabs(got - emu) / mag
                                                                   : std::fabs(got - emu));
                    g32_err2 += (got - exact) * (got - exact);
                    g32_ref2 += exact * exact;
                }
            }
        }
        CHECK(g32_worst <= std::ldexp(1.0, -17), "%s: g32 |kernel - FP64| = %.3g sum|wa|",
              s.label.c_str(), g32_worst);
    }

    // (6) Big-core decoder (hybrid CPUs): the same decoded bytes as decode_rows.
    {
        const g::HostGgufKernels& K = g::host_gguf_kernels(g::host_gguf_isa());
        const std::size_t db        = g::decoded_row_bytes(s.fmt, k);
        AlignedBuf plain(db * std::size_t(s.rows)), big(db * std::size_t(s.rows));
        K.decode_rows(s.fmt, s.data.data(), s.row_bytes, s.rows, k, plain.p);
        K.decode_rows_big(s.fmt, s.data.data(), s.row_bytes, s.rows, k, big.p);
        CHECK(std::memcmp(plain.p, big.p, db * std::size_t(s.rows)) == 0,
              "%s: decode_rows_big differs from decode_rows", s.label.c_str());
    }

    // (5) BF16 dequantization.
    std::vector<std::uint16_t> b16(std::size_t(s.rows) * k);
    g::decode_rows_bf16(s.fmt, s.data.data(), s.row_bytes, s.rows, k, b16.data());
    long bad = 0;
    for (int r = 0; r < s.rows; ++r) {
        dequantize(s.fmt, s.data.data() + std::size_t(r) * s.row_bytes, w.data(), k);
        for (int i = 0; i < k; ++i) bad += b16[std::size_t(r) * k + i] != bf16_rne(w[i]);
    }
    CHECK(bad == 0, "%s: %ld BF16 dequantized values differ from RNE(ggml dequantize)",
          s.label.c_str(), bad);
    std::printf("%-34s rows %3d k %4d  |kernel-generic| <= %.2e sum|wa|  T/tile bit-identical %s  "
                "bf16 exact %s  rel.L2 vs BF16 activations %.3e\n",
                s.label.c_str(), s.rows, k, worst_generic,
                (tile_mismatch + t_mismatch) ? "NO" : "yes", bad ? "NO" : "yes",
                std::sqrt(err2 / std::max(ref2, 1e-300)));
    if (af == ActFormat::Q8_K) {
        std::printf(
            "%-34s   with per-32 activation scales: |kernel-FP64| <= %.2e sum|wa|  rel.L2 vs BF16 "
            "activations %.3e\n",
            "", g32_worst, std::sqrt(g32_err2 / std::max(g32_ref2, 1e-300)));
    }
}

} // namespace

int main(int argc, char** argv) {
    if (!ninfer::ops::host::host_experts_supported()) {
        std::printf(
            "SKIP: this CPU runs neither ISA of the GGUF host kernels (AVX2 + FMA + F16C)\n");
        return 77;
    }
    std::printf("GGUF host kernels: %s (NINFER_HOST_ISA selects)\n",
                ninfer::ops::host::host_isa_name(g::host_gguf_isa()));
    std::mt19937_64 rng(0x5EED1234ULL);

    const struct {
        WeightFormat fmt;
        int k;
    } kFormats[] = {{WeightFormat::IQ2_S, 2560},  {WeightFormat::IQ3_XXS, 2560},
                    {WeightFormat::IQ3_S, 2560},  {WeightFormat::IQ4_XS, 2560},
                    {WeightFormat::IQ4_NL, 640},  {WeightFormat::Q2_0, 640},
                    {WeightFormat::IQ4_NL, 2560}, {WeightFormat::Q2_0, 2560}};

    for (const auto& f : kFormats) check_rows(synthetic(f.fmt, f.k, 21, rng), rng);
    const char* dir = argc > 1 ? argv[1] : std::getenv("NINFER_GGUF_BANKS");
    if (dir != nullptr && *dir) {
        const auto sets = bank_rows(dir, 64);
        CHECK(!sets.empty(), "no bank rows under %s", dir);
        for (const auto& s : sets) check_rows(s, rng);
    } else {
        std::printf("(no bank directory: real GSQ-RCO rows not checked; pass it as argv[1] or "
                    "NINFER_GGUF_BANKS)\n");
    }
    if (g_failures) {
        std::printf("FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
