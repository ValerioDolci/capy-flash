// moe_slots_gguf (decode windows) and moe_grouped_gguf (prefill widths) against FP64 oracles at the
// Qwen3.8-Flash-Next geometry (hidden 2560, expert intermediate 640, top-10).
//
// Banks: synthetic random ggml blocks for every gate/up and down format (one layer per format
// pair, 16 experts), and, with NINFER_GGUF_BANKS (or a directory argument), 16 random experts of
// every layer_LL.{gate_up,down}.bin of the real extracted banks (formats from manifest.json,
// checked against the file sizes). Of the 16 test experts, 12 are resident; the others are absent
// from the slot table, or mapped to a slot >= layer.slots; routed ids also hold -1, ids >= E,
// repeated experts in a token row, a zero weight, and one token without any resident expert.
// Placements: blob (the experts copied into a device pool at arbitrary 256-aligned offsets, in a
// shuffled slot order, unused slots holding garbage) and planar (host-mapped gate|up and down
// banks indexed by a different slot assignment).
//
// Oracles (FP64 over the ggml dequantization, dequantize_row_* of third_party/ggml-iq):
//   decode:  x coded with the Op's activation codec (per 32 values d = fl(amax / 127),
//            q = roundf(v / d)); g, u in FP64; h = silu(g) * u; h coded with the same codec;
//            out = sum_k alpha_k W_down . h_q.
//   grouped: weights rounded to BF16 (RNE of the exact dequantization), BF16 x; g, u in FP64;
//            h rounded to BF16; out = sum_k alpha_k W_down,bf16 . h_bf16.
// An h code is "uncertain" when the Op's FP32 h can land on another code: the oracle carries an
// FP32 error bound e_h for every h (e_g = eps * sum over 32-value sub-blocks |partial|, eps = 2e-6
// for the dp4a GEMV, a rigorous bound for its 5 lane FMAs + 4 shuffle adds; 4e-6 for the BF16
// tensor-core GEMM, ~20x its random-walk error over 160 MMAs; silu and the product add 4e-6 |h|),
// and every code reachable from [h - e_h, h + e_h] (and from the block scale range for q8) is
// considered. Uncertain codes add their largest deviation times |W_down[r][i]| * |alpha| to the
// allowance A[r] of output r. The count and max A/rms are printed.
//
// Pre-registered criteria, per token t with resident experts and per output r:
//   decode:  |out - ref| - A[r] <= 1e-3 * rms_t(ref)
//   grouped: |out - ref| - A[r] <= 1e-3 * rms_t(ref)
//     (BF16 bound: with the BF16 weight and h roundings reproduced by the oracle and the
//     uncertain BF16 h codes in A, what remains is FP32 accumulation, ~1e-6 relative);
//   tokens without a resident expert: exactly 0 (Store);
//   exact (bitwise): repeated calls are identical; blob and planar placements give identical
//   results; Accumulate = old + Store (FP32); BF16 Store = RNE(FP32 Store); a call with
//   slots = 0 writes zeros (Store) or keeps `out` (Accumulate); a captured CUDA graph replays the
//   eager result before and after x, routing, slot table and slot offsets change; grouped
//   columns of a call one piece plus 52 columns wide (two pieces) equal the same columns of
//   narrower calls.
// Informational: whether a token's decode result is the same in calls of different widths.
//
// --bench: microseconds per call (blob placement, 10 resident experts routed by every token):
// decode T = 1 and T = 4 (eager and CUDA-graph replay), grouped T = 2048; no correctness checks.
#include "ninfer/ops/moe_slots_gguf.h"

#include "core/arena.h"
#include "ops/moe_slots_gguf/launch.h"
#include "ops/op_tester.h"

#include "ggml-iq/ggml-iq.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

using Fmt = ops::MoeGgufFormat;

constexpr int kH = 2560, kI = 640, kK = 10, kGU = 2 * kI;
constexpr int kNE           = 16; // test experts per bank
constexpr int kResident     = 12; // test experts j < 12 are resident
constexpr int kOutOfPool    = 14; // test expert mapped to a slot >= layer.slots
constexpr int kDecodeT      = 8;
constexpr int kGroupedT     = 300; // oracle columns of the grouped checks
constexpr int kPieceT       = ops::detail::kGgufGroupedChunkColumns;
constexpr int kWideT        = kPieceT + 52; // two grouped pieces
constexpr double kDecodeTol = 1e-3, kGroupedTol = 1e-3;
constexpr double kEpsDp4a = 2e-6, kEpsMma = 4e-6, kEpsSilu = 4e-6;

int failures = 0;

void fail(const std::string& what) {
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
}

void parallel_for(int n, const std::function<void(int)>& fn) {
    const int cores =
        static_cast<int>(std::max(1U, std::min(16U, std::thread::hardware_concurrency())));
    const int workers = std::max(1, std::min(n, cores));
    std::vector<std::thread> pool;
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(workers));
    for (int w = 0; w < workers; ++w) {
        pool.emplace_back([&, w] {
            try {
                for (int i = w; i < n; i += workers) fn(i);
            } catch (...) { errors[static_cast<std::size_t>(w)] = std::current_exception(); }
        });
    }
    for (auto& t : pool) t.join();
    for (auto& e : errors)
        if (e) std::rethrow_exception(e);
}

// ------------------------------------------------------------------------------- formats
const char* fmt_name(Fmt f) {
    switch (f) {
    case Fmt::IQ2_S:
        return "iq2_s";
    case Fmt::IQ3_XXS:
        return "iq3_xxs";
    case Fmt::IQ3_S:
        return "iq3_s";
    case Fmt::IQ4_XS:
        return "iq4_xs";
    case Fmt::IQ4_NL:
        return "iq4_nl";
    case Fmt::Q2_0:
        return "q2_0";
    }
    return "?";
}

bool fmt_of_name(const std::string& name, Fmt* f) {
    for (Fmt c : {Fmt::IQ2_S, Fmt::IQ3_XXS, Fmt::IQ3_S, Fmt::IQ4_XS, Fmt::IQ4_NL, Fmt::Q2_0}) {
        if (name == std::string("gguf_") + fmt_name(c)) {
            *f = c;
            return true;
        }
    }
    return false;
}

int block_elems(Fmt f) { return f == Fmt::IQ4_NL ? 32 : f == Fmt::Q2_0 ? 64 : 256; }

void dequant_row(Fmt f, const std::uint8_t* row, float* y, int k) {
    switch (f) {
    case Fmt::IQ2_S:
        dequantize_row_iq2_s(reinterpret_cast<const block_iq2_s*>(row), y, k);
        return;
    case Fmt::IQ3_XXS:
        dequantize_row_iq3_xxs(reinterpret_cast<const block_iq3_xxs*>(row), y, k);
        return;
    case Fmt::IQ3_S:
        dequantize_row_iq3_s(reinterpret_cast<const block_iq3_s*>(row), y, k);
        return;
    case Fmt::IQ4_XS:
        dequantize_row_iq4_xs(reinterpret_cast<const block_iq4_xs*>(row), y, k);
        return;
    case Fmt::IQ4_NL:
        dequantize_row_iq4_nl(reinterpret_cast<const block_iq4_nl*>(row), y, k);
        return;
    case Fmt::Q2_0:
        dequantize_row_q2_0(reinterpret_cast<const block_q2_0*>(row), y, k);
        return;
    }
    throw std::runtime_error("dequant_row: unknown format");
}

float bf16r(float v) { return bf16_to_f32(f32_to_bf16(v)); }

std::uint16_t fp16_bits(float f) {
    const __half_raw raw = __float2half_rn(f);
    return raw.x;
}

// ---------------------------------------------------------------------------------- banks
struct Bank {
    std::string name;
    Fmt gu = Fmt::IQ3_XXS, dn = Fmt::IQ4_NL;
    std::int64_t gu_row = 0, dn_row = 0, gu_bytes = 0, dn_bytes = 0, expert_bytes = 0;
    int table_size = 0;                          // E of the slot table
    std::vector<int> ids;                        // [kNE] expert id of test expert j
    std::vector<int> j_of_id;                    // [E] test expert of an id, or -1
    std::vector<std::vector<std::uint8_t>> blob; // [kNE] 640 gate, 640 up, 2560 down rows
    std::vector<std::vector<float>> wgu, wdn;    // dequantized (rounded to BF16 for grouped)
};

void init_geometry(Bank& b) {
    b.gu_row       = ops::moe_gguf_row_bytes(b.gu, kH);
    b.dn_row       = ops::moe_gguf_row_bytes(b.dn, kI);
    b.gu_bytes     = kGU * b.gu_row;
    b.dn_bytes     = kH * b.dn_row;
    b.expert_bytes = b.gu_bytes + b.dn_bytes;
    if (b.gu_row <= 0 || b.dn_row <= 0 || ops::moe_gguf_expert_bytes(b.gu, b.dn) != b.expert_bytes)
        throw std::runtime_error(b.name + ": inconsistent GGUF geometry helpers");
}

void finish_ids(Bank& b) {
    b.j_of_id.assign(static_cast<std::size_t>(b.table_size), -1);
    for (int j = 0; j < kNE; ++j) b.j_of_id[static_cast<std::size_t>(b.ids[j])] = j;
}

// Random bytes, each block's leading FP16 scale d drawn from [5e-4, 2e-2].
void fill_rows(std::uint8_t* p, std::int64_t rows, std::int64_t row_bytes, Fmt f, int k,
               std::mt19937_64& rng) {
    std::uniform_real_distribution<float> dd(5e-4F, 2e-2F);
    const int nb          = k / block_elems(f);
    const std::int64_t bb = row_bytes / nb;
    for (std::int64_t i = 0; i < rows * row_bytes; ++i) p[i] = static_cast<std::uint8_t>(rng());
    for (std::int64_t r = 0; r < rows; ++r)
        for (int b = 0; b < nb; ++b) {
            const std::uint16_t h = fp16_bits(dd(rng));
            std::memcpy(p + r * row_bytes + b * bb, &h, 2);
        }
}

Bank synthetic_bank(Fmt gu, Fmt dn, std::mt19937_64& rng) {
    Bank b;
    b.name = std::string("synthetic ") + fmt_name(gu) + "+" + fmt_name(dn);
    b.gu   = gu;
    b.dn   = dn;
    init_geometry(b);
    b.table_size = 40;
    std::vector<int> all(static_cast<std::size_t>(b.table_size));
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), rng);
    b.ids.assign(all.begin(), all.begin() + kNE);
    b.blob.resize(kNE);
    for (auto& e : b.blob) {
        e.resize(static_cast<std::size_t>(b.expert_bytes));
        fill_rows(e.data(), kGU, b.gu_row, gu, kH, rng);
        fill_rows(e.data() + b.gu_bytes, kH, b.dn_row, dn, kI, rng);
    }
    finish_ids(b);
    return b;
}

struct RealLayer {
    int layer = 0;
    Fmt gu = Fmt::IQ3_XXS, dn = Fmt::IQ4_NL;
    std::string gu_path, dn_path;
};

// The "format" value of the manifest object whose "file" is `file`.
std::string manifest_format(const std::string& manifest, const std::string& file) {
    const auto at = manifest.find("\"" + file + "\"");
    if (at == std::string::npos) return {};
    const auto key = manifest.find("\"format\"", at);
    if (key == std::string::npos) return {};
    const auto colon = manifest.find(':', key);
    const auto q0    = manifest.find('"', colon);
    const auto q1    = manifest.find('"', q0 + 1);
    if (colon == std::string::npos || q0 == std::string::npos || q1 == std::string::npos) return {};
    return manifest.substr(q0 + 1, q1 - q0 - 1);
}

std::vector<RealLayer> real_layers(const std::string& dir) {
    std::ifstream in(dir + "/manifest.json");
    if (!in) {
        fail("NINFER_GGUF_BANKS: no manifest.json in " + dir);
        return {};
    }
    const std::string manifest((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
    std::vector<RealLayer> out;
    for (int l = 0; l < 100; ++l) {
        char gu_file[64], dn_file[64];
        std::snprintf(gu_file, sizeof gu_file, "layer_%02d.gate_up.bin", l);
        std::snprintf(dn_file, sizeof dn_file, "layer_%02d.down.bin", l);
        RealLayer r;
        r.layer   = l;
        r.gu_path = dir + "/" + gu_file;
        r.dn_path = dir + "/" + dn_file;
        if (!std::filesystem::exists(r.gu_path) || !std::filesystem::exists(r.dn_path)) continue;
        if (!fmt_of_name(manifest_format(manifest, gu_file), &r.gu) ||
            !fmt_of_name(manifest_format(manifest, dn_file), &r.dn)) {
            fail(std::string("manifest.json: no supported format for ") + gu_file + " / " +
                 dn_file);
            continue;
        }
        out.push_back(r);
    }
    if (out.empty()) fail("NINFER_GGUF_BANKS: no layer_LL.{gate_up,down}.bin in " + dir);
    return out;
}

bool load_real_bank(const RealLayer& r, std::mt19937_64& rng, Bank& b) {
    b.name = "layer " + std::to_string(r.layer) + " " + fmt_name(r.gu) + "+" + fmt_name(r.dn);
    b.gu   = r.gu;
    b.dn   = r.dn;
    init_geometry(b);
    const auto gsize           = static_cast<std::int64_t>(std::filesystem::file_size(r.gu_path));
    const auto dsize           = static_cast<std::int64_t>(std::filesystem::file_size(r.dn_path));
    const std::int64_t experts = gsize / b.gu_bytes;
    if (gsize % b.gu_bytes != 0 || dsize != experts * b.dn_bytes || experts < kNE) {
        fail(b.name + ": bank sizes do not match the manifest formats");
        return false;
    }
    b.table_size = static_cast<int>(experts);
    std::vector<int> all(static_cast<std::size_t>(experts));
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), rng);
    b.ids.assign(all.begin(), all.begin() + kNE);
    std::ifstream g(r.gu_path, std::ios::binary), d(r.dn_path, std::ios::binary);
    b.blob.resize(kNE);
    for (int j = 0; j < kNE; ++j) {
        auto& e = b.blob[static_cast<std::size_t>(j)];
        e.resize(static_cast<std::size_t>(b.expert_bytes));
        g.seekg(static_cast<std::streamoff>(b.ids[j]) * b.gu_bytes);
        g.read(reinterpret_cast<char*>(e.data()), b.gu_bytes);
        d.seekg(static_cast<std::streamoff>(b.ids[j]) * b.dn_bytes);
        d.read(reinterpret_cast<char*>(e.data() + b.gu_bytes), b.dn_bytes);
        if (!g || !d) {
            fail(b.name + ": short read");
            return false;
        }
    }
    finish_ids(b);
    return true;
}

void dequantize_bank(Bank& b) {
    b.wgu.assign(kNE, {});
    b.wdn.assign(kNE, {});
    parallel_for(kNE, [&](int j) {
        const auto& e = b.blob[static_cast<std::size_t>(j)];
        auto& gu      = b.wgu[static_cast<std::size_t>(j)];
        auto& dn      = b.wdn[static_cast<std::size_t>(j)];
        gu.resize(static_cast<std::size_t>(kGU) * kH);
        dn.resize(static_cast<std::size_t>(kH) * kI);
        for (int r = 0; r < kGU; ++r)
            dequant_row(b.gu, e.data() + r * b.gu_row, gu.data() + std::size_t(r) * kH, kH);
        for (int r = 0; r < kH; ++r)
            dequant_row(b.dn, e.data() + b.gu_bytes + r * b.dn_row, dn.data() + std::size_t(r) * kI,
                        kI);
    });
}

void round_bank_bf16(Bank& b) {
    parallel_for(kNE, [&](int j) {
        for (float& v : b.wgu[static_cast<std::size_t>(j)]) v = bf16r(v);
        for (float& v : b.wdn[static_cast<std::size_t>(j)]) v = bf16r(v);
    });
}

// -------------------------------------------------------------------------------- routing
struct Routing {
    std::vector<std::uint16_t> x;  // [T][2560] BF16
    std::vector<std::int32_t> ids; // [T][10]
    std::vector<float> w;          // [T][10]
};

Routing make_routing(const Bank& b, int tokens, std::mt19937_64& rng) {
    Routing R;
    R.x.resize(static_cast<std::size_t>(tokens) * kH);
    R.ids.resize(static_cast<std::size_t>(tokens) * kK);
    R.w.resize(static_cast<std::size_t>(tokens) * kK);
    std::normal_distribution<float> nd(0.F, 0.7F);
    std::uniform_real_distribution<float> wd(0.02F, 0.3F);
    for (auto& v : R.x) v = f32_to_bf16(nd(rng));
    const auto id = [&](int j) { return b.ids[static_cast<std::size_t>(j)]; };
    for (int t = 0; t < tokens; ++t) {
        std::int32_t* row = R.ids.data() + std::size_t(t) * kK;
        float* w          = R.w.data() + std::size_t(t) * kK;
        std::vector<int> js(kNE);
        std::iota(js.begin(), js.end(), 0);
        std::shuffle(js.begin(), js.end(), rng);
        for (int k = 0; k < kK; ++k) {
            row[k] = id(js[static_cast<std::size_t>(k)]);
            w[k]   = wd(rng);
        }
        if (rng() % 32 == 0) row[rng() % kK] = -1;
        if (rng() % 32 == 0) row[rng() % kK] = b.table_size + 7;
        if (rng() % 16 == 0) { // a repeated expert
            const auto to   = rng() % kK;
            const auto from = rng() % kK;
            row[to]         = row[from];
        }
    }
    // Token 0: expert 0 three times, a zero weight, an absent expert, -1, an id >= E, a slot >=
    // slots.
    const std::int32_t t0[kK] = {id(0), id(0), id(1), id(12),           -1,
                                 id(2), id(0), id(3), b.table_size + 7, id(kOutOfPool)};
    std::copy(t0, t0 + kK, R.ids.begin());
    R.w[5] = 0.F;
    // Token 2: no resident expert.
    if (tokens > 2) {
        const std::int32_t t2[kK] = {id(12),           id(13), id(kOutOfPool), id(15), -1,
                                     b.table_size + 7, id(12), id(13),         id(15), -1};
        std::copy(t2, t2 + kK, R.ids.begin() + 2 * kK);
    }
    return R;
}

// --------------------------------------------------------------------------------- oracle
struct Ref {
    std::vector<double> out, allow; // [2560]
    int uncertain = 0;
    bool any      = false; // has a resident expert
};

// The activation codec over one 32-value block of h, with the codes reachable from
// [h - e, h + e]: hq gets the reference values (codec of float(h)), dev the largest deviation of
// an uncertain code (0 otherwise).
void codec_q8_interval(const double* h, const double* e, double* hq, double* dev, int& uncertain) {
    float amax_ref = 0.F;
    double amax_lo = 0.0, amax_hi = 0.0;
    for (int i = 0; i < 32; ++i) {
        amax_ref = std::max(amax_ref, std::fabs(static_cast<float>(h[i])));
        amax_lo  = std::max(amax_lo, std::fabs(h[i]) - e[i]);
        amax_hi  = std::max(amax_hi, std::fabs(h[i]) + e[i]);
    }
    const float d_ref = amax_ref / 127.0F;
    const double d_lo = amax_lo / 127.0, d_hi = amax_hi / 127.0;
    for (int i = 0; i < 32; ++i) {
        const double q_ref =
            amax_ref == 0.F ? 0.0
                            : static_cast<double>(std::round(static_cast<float>(h[i]) / d_ref));
        hq[i]  = q_ref * static_cast<double>(d_ref);
        dev[i] = 0.0;
        if (!(d_lo > 0.0)) {
            if (amax_hi > 0.0) {
                dev[i] = 2.0 * amax_hi;
                ++uncertain;
            }
            continue;
        }
        double rmin = std::numeric_limits<double>::infinity(), rmax = -rmin;
        for (double v : {h[i] - e[i], h[i] + e[i]})
            for (double d : {d_lo, d_hi}) {
                rmin = std::min(rmin, v / d);
                rmax = std::max(rmax, v / d);
            }
        const double qmin = std::clamp(std::round(rmin), -127.0, 127.0);
        const double qmax = std::clamp(std::round(rmax), -127.0, 127.0);
        if (qmin == q_ref && qmax == q_ref) continue;
        ++uncertain;
        double dv = 0.0;
        for (double q : {qmin, qmax, q_ref})
            for (double d : {d_lo, d_hi}) dv = std::max(dv, std::fabs(q * d - hq[i]));
        dev[i] = dv;
    }
}

// BF16 rounding of h with the values reachable from [h - e, h + e].
void codec_bf16_interval(double h, double e, double& hb, double& dev, int& uncertain) {
    hb              = bf16r(static_cast<float>(h));
    const double lo = bf16r(static_cast<float>(h - e)), hi = bf16r(static_cast<float>(h + e));
    dev = 0.0;
    if (lo == hb && hi == hb) return;
    ++uncertain;
    dev = std::max(std::fabs(lo - hb), std::fabs(hi - hb));
}

// The Op's activation codec of x on the host (bit-identical codes and scales).
void codec_x(const std::uint16_t* x, double* xq) {
    for (int b = 0; b < kH; b += 32) {
        float v[32], amax = 0.F;
        for (int i = 0; i < 32; ++i) {
            v[i] = bf16_to_f32(x[b + i]);
            amax = std::max(amax, std::fabs(v[i]));
        }
        const float d = amax / 127.0F;
        for (int i = 0; i < 32; ++i)
            xq[b + i] = amax == 0.F
                            ? 0.0
                            : static_cast<double>(std::round(v[i] / d)) * static_cast<double>(d);
    }
}

Ref oracle_token(const Bank& b, const Routing& R, int t, bool grouped) {
    Ref ref;
    ref.out.assign(kH, 0.0);
    ref.allow.assign(kH, 0.0);
    std::vector<double> xa(kH), g(kI), u(kI), eg(kI), eu(kI), h(kI), eh(kI), hq(kI), dev(kI);
    const std::uint16_t* xt = R.x.data() + std::size_t(t) * kH;
    if (grouped) {
        for (int i = 0; i < kH; ++i) xa[static_cast<std::size_t>(i)] = bf16_to_f32(xt[i]);
    } else {
        codec_x(xt, xa.data());
    }
    const double eps = grouped ? kEpsMma : kEpsDp4a;
    for (int k = 0; k < kK; ++k) {
        const int id = R.ids[std::size_t(t) * kK + k];
        if (id < 0 || id >= b.table_size) continue;
        const int j = b.j_of_id[static_cast<std::size_t>(id)];
        if (j < 0 || j >= kResident) continue;
        ref.any        = true;
        const float* W = b.wgu[static_cast<std::size_t>(j)].data();
        for (int r = 0; r < kGU; ++r) {
            const float* row = W + std::size_t(r) * kH;
            double total = 0.0, magnitude = 0.0;
            for (int s = 0; s < kH / 32; ++s) {
                double part = 0.0;
                for (int i = 32 * s; i < 32 * s + 32; ++i)
                    part += static_cast<double>(row[i]) * xa[static_cast<std::size_t>(i)];
                total += part;
                magnitude += std::fabs(part);
            }
            if (r < kI) {
                g[static_cast<std::size_t>(r)]  = total;
                eg[static_cast<std::size_t>(r)] = eps * magnitude;
            } else {
                u[static_cast<std::size_t>(r - kI)]  = total;
                eu[static_cast<std::size_t>(r - kI)] = eps * magnitude;
            }
        }
        for (std::size_t r = 0; r < static_cast<std::size_t>(kI); ++r) {
            const double sg = g[r] / (1.0 + std::exp(-g[r]));
            h[r]            = sg * u[r];
            // |silu'| <= 1.1
            eh[r] = 1.1 * eg[r] * std::fabs(u[r]) + std::fabs(sg) * eu[r] + 1.1 * eg[r] * eu[r] +
                    kEpsSilu * std::fabs(h[r]);
        }
        int uncertain = 0;
        if (grouped) {
            for (std::size_t r = 0; r < static_cast<std::size_t>(kI); ++r)
                codec_bf16_interval(h[r], eh[r], hq[r], dev[r], uncertain);
        } else {
            for (int blk = 0; blk < kI / 32; ++blk)
                codec_q8_interval(h.data() + 32 * blk, eh.data() + 32 * blk, hq.data() + 32 * blk,
                                  dev.data() + 32 * blk, uncertain);
        }
        ref.uncertain += uncertain;
        std::vector<int> unsure;
        for (int i = 0; i < kI; ++i)
            if (dev[static_cast<std::size_t>(i)] != 0.0) unsure.push_back(i);
        const double alpha = R.w[std::size_t(t) * kK + k];
        const float* D     = b.wdn[static_cast<std::size_t>(j)].data();
        for (int r = 0; r < kH; ++r) {
            const float* row = D + std::size_t(r) * kI;
            double s = 0.0, a = 0.0;
            for (int i = 0; i < kI; ++i)
                s += static_cast<double>(row[i]) * hq[static_cast<std::size_t>(i)];
            for (int i : unsure)
                a += dev[static_cast<std::size_t>(i)] * std::fabs(static_cast<double>(row[i]));
            ref.out[static_cast<std::size_t>(r)] += alpha * s;
            ref.allow[static_cast<std::size_t>(r)] += std::fabs(alpha) * a;
        }
    }
    return ref;
}

std::vector<Ref> oracle(const Bank& b, const Routing& R, int tokens, bool grouped) {
    std::vector<Ref> refs(static_cast<std::size_t>(tokens));
    parallel_for(
        tokens, [&](int t) { refs[static_cast<std::size_t>(t)] = oracle_token(b, R, t, grouped); });
    return refs;
}

void check_numeric(const std::string& label, const std::vector<float>& out,
                   const std::vector<Ref>& refs, int T, double tol) {
    double worst = 0.0, worst_allow = 0.0;
    int uncertain = 0, nonfinite = 0;
    for (int t = 0; t < T; ++t) {
        const Ref& R   = refs[static_cast<std::size_t>(t)];
        const float* o = out.data() + std::size_t(t) * kH;
        uncertain += R.uncertain;
        if (!R.any) {
            for (int r = 0; r < kH; ++r)
                if (o[r] != 0.F) {
                    fail(label + ": token " + std::to_string(t) +
                         " without a resident expert is not 0");
                    break;
                }
            continue;
        }
        double rms = 0.0;
        for (double v : R.out) rms += v * v;
        rms = std::sqrt(rms / kH);
        if (!(rms > 0.0)) {
            fail(label + ": token " + std::to_string(t) + " has a zero reference");
            continue;
        }
        for (int r = 0; r < kH; ++r) {
            if (!std::isfinite(o[r])) {
                ++nonfinite;
                continue;
            }
            const double ref = R.out[static_cast<std::size_t>(r)];
            const double a   = R.allow[static_cast<std::size_t>(r)];
            worst            = std::max(worst, (std::fabs(o[r] - ref) - a) / rms);
            worst_allow      = std::max(worst_allow, a / rms);
        }
    }
    std::printf("  %-52s max(|err|-A)/rms %.2e  max A/rms %.2e  uncertain h codes %d\n",
                label.c_str(), worst, worst_allow, uncertain);
    if (nonfinite > 0) fail(label + ": " + std::to_string(nonfinite) + " non-finite outputs");
    if (!(worst <= tol))
        fail(label + ": error " + std::to_string(worst) + " rms exceeds " + std::to_string(tol));
}

bool same_bits(const void* a, const void* b, std::size_t bytes) {
    return std::memcmp(a, b, bytes) == 0;
}

// ------------------------------------------------------------------------------------ GPU
struct Gpu {
    cudaStream_t stream = nullptr;
    std::unique_ptr<DeviceArena> ws;
};

struct Placement {
    DeviceBuffer table;   // I32 [E]
    DeviceBuffer offsets; // I64 [slots] (blob)
    DeviceBuffer pool;    // blob pool
    std::unique_ptr<PinnedHostBuffer> planar_gu, planar_dn;
    std::vector<std::int32_t> table_host;
    std::vector<std::int64_t> offsets_host;
    ops::MoeGgufLayer layer;
};

// Test experts j < resident get slot perm[j]; j = kOutOfPool gets a slot >= slots; others -1.
std::vector<std::int32_t> table_for(const Bank& b, const std::vector<int>& perm, int slots,
                                    int resident) {
    std::vector<std::int32_t> t(static_cast<std::size_t>(b.table_size), -1);
    for (int j = 0; j < resident; ++j)
        t[static_cast<std::size_t>(b.ids[j])] = perm[static_cast<std::size_t>(j)];
    t[static_cast<std::size_t>(b.ids[kOutOfPool])] = slots + 5;
    return t;
}

std::unique_ptr<Placement> make_blob(const Bank& b, std::mt19937_64& rng,
                                     int resident = kResident) {
    auto P          = std::make_unique<Placement>();
    const int slots = kResident + 2; // two slots hold garbage
    std::vector<int> perm(static_cast<std::size_t>(slots)), phys(static_cast<std::size_t>(slots));
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::iota(phys.begin(), phys.end(), 0);
    std::shuffle(phys.begin(), phys.end(), rng);
    const std::int64_t foot = (b.expert_bytes + 255) / 256 * 256;
    P->offsets_host.assign(static_cast<std::size_t>(slots), 0);
    std::int64_t at = 256 * static_cast<std::int64_t>(rng() % 4);
    for (int s : phys) {
        P->offsets_host[static_cast<std::size_t>(s)] = at;
        at += foot + 256 * static_cast<std::int64_t>(rng() % 3);
    }
    P->pool = DeviceBuffer(static_cast<std::size_t>(at));
    std::vector<std::uint8_t> garbage(static_cast<std::size_t>(at));
    for (auto& v : garbage) v = static_cast<std::uint8_t>(rng());
    P->pool.copy_from_host(garbage.data(), garbage.size());
    for (int j = 0; j < resident; ++j)
        P->pool.copy_from_host(
            b.blob[static_cast<std::size_t>(j)].data(), static_cast<std::size_t>(b.expert_bytes),
            static_cast<std::size_t>(
                P->offsets_host[static_cast<std::size_t>(perm[static_cast<std::size_t>(j)])]));
    P->table_host           = table_for(b, perm, slots, resident);
    P->table                = to_device(P->table_host);
    P->offsets              = to_device(P->offsets_host);
    P->layer.slot_table     = Tensor(P->table.p, DType::I32, {b.table_size});
    P->layer.slots          = slots;
    P->layer.slot_offset    = static_cast<const std::int64_t*>(P->offsets.p);
    P->layer.base           = static_cast<const std::byte*>(P->pool.p);
    P->layer.gate_up_format = b.gu;
    P->layer.down_format    = b.dn;
    return P;
}

std::unique_ptr<Placement> make_planar(const Bank& b, std::mt19937_64& rng) {
    auto P          = std::make_unique<Placement>();
    const int slots = kResident + 1;
    std::vector<int> perm(static_cast<std::size_t>(slots));
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    P->planar_gu = std::make_unique<PinnedHostBuffer>(static_cast<std::size_t>(slots * b.gu_bytes),
                                                      HostMapping::Yes);
    P->planar_dn = std::make_unique<PinnedHostBuffer>(static_cast<std::size_t>(slots * b.dn_bytes),
                                                      HostMapping::Yes);
    auto* gu     = static_cast<std::uint8_t*>(P->planar_gu->data());
    auto* dn     = static_cast<std::uint8_t*>(P->planar_dn->data());
    std::memset(gu, 0xA5, P->planar_gu->size());
    std::memset(dn, 0x5A, P->planar_dn->size());
    for (int j = 0; j < kResident; ++j) {
        const auto& e = b.blob[static_cast<std::size_t>(j)];
        const auto s  = static_cast<std::int64_t>(perm[static_cast<std::size_t>(j)]);
        std::memcpy(gu + s * b.gu_bytes, e.data(), static_cast<std::size_t>(b.gu_bytes));
        std::memcpy(dn + s * b.dn_bytes, e.data() + b.gu_bytes,
                    static_cast<std::size_t>(b.dn_bytes));
    }
    P->table_host           = table_for(b, perm, slots, kResident);
    P->table                = to_device(P->table_host);
    P->layer.slot_table     = Tensor(P->table.p, DType::I32, {b.table_size});
    P->layer.slots          = slots;
    P->layer.gate_up_base   = static_cast<const std::byte*>(P->planar_gu->device_data());
    P->layer.down_base      = static_cast<const std::byte*>(P->planar_dn->device_data());
    P->layer.gate_up_format = b.gu;
    P->layer.down_format    = b.dn;
    return P;
}

struct Inputs {
    DeviceBuffer x, ids, w;
    int T = 0;
};

Inputs upload(const Routing& R, int first, int T) {
    Inputs in;
    in.T   = T;
    in.x   = to_device(std::vector<std::uint16_t>(R.x.begin() + std::ptrdiff_t(first) * kH,
                                                  R.x.begin() + std::ptrdiff_t(first + T) * kH));
    in.ids = to_device(std::vector<std::int32_t>(R.ids.begin() + std::ptrdiff_t(first) * kK,
                                                 R.ids.begin() + std::ptrdiff_t(first + T) * kK));
    in.w   = to_device(std::vector<float>(R.w.begin() + std::ptrdiff_t(first) * kK,
                                          R.w.begin() + std::ptrdiff_t(first + T) * kK));
    return in;
}

// Rewrites `in` in place (same device buffers) with tokens [first, first + in.T) of R.
void rewrite(Inputs& in, const Routing& R, int first) {
    in.x.copy_from_host(R.x.data() + std::size_t(first) * kH, std::size_t(in.T) * kH * 2);
    in.ids.copy_from_host(R.ids.data() + std::size_t(first) * kK, std::size_t(in.T) * kK * 4);
    in.w.copy_from_host(R.w.data() + std::size_t(first) * kK, std::size_t(in.T) * kK * 4);
}

void call_decode(Gpu& g, const ops::MoeGgufLayer& layer, const Inputs& in, ops::MoeSlotsOutput mode,
                 DeviceBuffer& out, DType out_dtype) {
    const Tensor x(in.x.p, DType::BF16, {kH, in.T});
    const Tensor ids(in.ids.p, DType::I32, {kK, in.T});
    const Tensor w(in.w.p, DType::FP32, {kK, in.T});
    Tensor o(out.p, out_dtype, {kH, in.T});
    ops::moe_slots_gguf(x, ids, w, layer, mode, o, *g.ws, g.stream);
    cuda_check_last_launch("moe_slots_gguf");
}

void call_grouped(Gpu& g, const ops::MoeGgufLayer& layer, const Inputs& in, DeviceBuffer& out) {
    const Tensor x(in.x.p, DType::BF16, {kH, in.T});
    const Tensor ids(in.ids.p, DType::I32, {kK, in.T});
    const Tensor w(in.w.p, DType::FP32, {kK, in.T});
    Tensor o(out.p, DType::FP32, {kH, in.T});
    ops::moe_grouped_gguf(x, ids, w, layer, o, *g.ws, g.stream);
    cuda_check_last_launch("moe_grouped_gguf");
}

// Output of one call; `out` starts as NaN bytes (Store must write every value) or as `init`.
std::vector<float> run_decode(Gpu& g, const ops::MoeGgufLayer& layer, const Inputs& in,
                              ops::MoeSlotsOutput mode, const std::vector<float>* init = nullptr) {
    const std::size_t n = std::size_t(in.T) * kH;
    DeviceBuffer out(n * 4);
    if (init != nullptr)
        out.copy_from_host(init->data(), n * 4);
    else
        cuda_check(cudaMemsetAsync(out.p, 0xFF, n * 4, g.stream), "memset");
    call_decode(g, layer, in, mode, out, DType::FP32);
    cuda_synchronize(g.stream);
    return from_device<float>(out, n);
}

std::vector<std::uint16_t> run_decode_bf16(Gpu& g, const ops::MoeGgufLayer& layer,
                                           const Inputs& in) {
    const std::size_t n = std::size_t(in.T) * kH;
    DeviceBuffer out(n * 2);
    cuda_check(cudaMemsetAsync(out.p, 0xFF, n * 2, g.stream), "memset");
    call_decode(g, layer, in, ops::MoeSlotsOutput::Store, out, DType::BF16);
    cuda_synchronize(g.stream);
    return from_device<std::uint16_t>(out, n);
}

std::vector<float> run_grouped(Gpu& g, const ops::MoeGgufLayer& layer, const Inputs& in) {
    const std::size_t n = std::size_t(in.T) * kH;
    DeviceBuffer out(n * 4);
    cuda_check(cudaMemsetAsync(out.p, 0xFF, n * 4, g.stream), "memset");
    call_grouped(g, layer, in, out);
    cuda_synchronize(g.stream);
    return from_device<float>(out, n);
}

struct Graph {
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;

    ~Graph() {
        if (exec != nullptr) cudaGraphExecDestroy(exec);
        if (graph != nullptr) cudaGraphDestroy(graph);
    }
};

void capture(Gpu& g, Graph& G, const std::function<void()>& body) {
    cuda_check(cudaStreamBeginCapture(g.stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    body();
    cuda_check(cudaStreamEndCapture(g.stream, &G.graph), "end capture");
    cuda_check(cudaGraphInstantiate(&G.exec, G.graph, 0), "instantiate");
}

std::vector<float> replay(Gpu& g, Graph& G, DeviceBuffer& out, std::size_t n) {
    cuda_check(cudaMemsetAsync(out.p, 0xFF, n * 4, g.stream), "memset");
    cuda_check(cudaGraphLaunch(G.exec, g.stream), "graph launch");
    cuda_synchronize(g.stream);
    return from_device<float>(out, n);
}

// Moves every slot of a blob placement to slot sigma(s) (offset table and slot table rewritten
// consistently) and drops test experts [8, 12) from the table; restore() undoes it.
void reshuffle_blob(Placement& P, const Bank& b, std::mt19937_64& rng) {
    const int slots = P.layer.slots;
    std::vector<int> sigma(static_cast<std::size_t>(slots));
    std::iota(sigma.begin(), sigma.end(), 0);
    std::shuffle(sigma.begin(), sigma.end(), rng);
    std::vector<std::int64_t> off(static_cast<std::size_t>(slots));
    for (int s = 0; s < slots; ++s)
        off[static_cast<std::size_t>(sigma[static_cast<std::size_t>(s)])] =
            P.offsets_host[static_cast<std::size_t>(s)];
    std::vector<std::int32_t> table = P.table_host;
    for (auto& v : table)
        if (v >= 0 && v < slots) v = sigma[static_cast<std::size_t>(v)];
    for (int j = 8; j < kResident; ++j) table[static_cast<std::size_t>(b.ids[j])] = -1;
    P.offsets.copy_from_host(off.data(), off.size() * 8);
    P.table.copy_from_host(table.data(), table.size() * 4);
}

void restore(Placement& P) {
    P.offsets.copy_from_host(P.offsets_host.data(), P.offsets_host.size() * 8);
    P.table.copy_from_host(P.table_host.data(), P.table_host.size() * 4);
}

// ---------------------------------------------------------------------------- decode cases
void decode_cases(const Bank& b, const Routing& R, Gpu& g, Placement& blob, Placement& planar,
                  std::mt19937_64& rng) {
    const std::vector<Ref> refs = oracle(b, R, kDecodeT, false);
    std::vector<std::vector<float>> stores(kDecodeT + 1);
    std::normal_distribution<float> nd(0.F, 1.F);
    for (int T = 1; T <= kDecodeT; ++T) {
        const std::string tag = b.name + " decode T=" + std::to_string(T);
        const Inputs in       = upload(R, 0, T);
        const std::size_t n   = std::size_t(T) * kH;
        const auto s1         = run_decode(g, blob.layer, in, ops::MoeSlotsOutput::Store);
        check_numeric(tag + " blob", s1, refs, T, kDecodeTol);
        const auto s2 = run_decode(g, blob.layer, in, ops::MoeSlotsOutput::Store);
        if (!same_bits(s1.data(), s2.data(), n * 4)) fail(tag + ": repeated calls differ");
        const auto p = run_decode(g, planar.layer, in, ops::MoeSlotsOutput::Store);
        if (!same_bits(s1.data(), p.data(), n * 4))
            fail(tag + ": blob and planar placements differ");
        std::vector<float> old(n), expect(n);
        for (std::size_t i = 0; i < n; ++i) {
            old[i]    = nd(rng);
            expect[i] = old[i] + s1[i];
        }
        const auto acc = run_decode(g, blob.layer, in, ops::MoeSlotsOutput::Accumulate, &old);
        if (!same_bits(acc.data(), expect.data(), n * 4)) fail(tag + ": Accumulate != old + Store");
        const auto bf = run_decode_bf16(g, blob.layer, in);
        for (std::size_t i = 0; i < n; ++i)
            if (bf[i] != f32_to_bf16(s1[i])) {
                fail(tag + ": BF16 Store != RNE(FP32 Store)");
                break;
            }
        stores[static_cast<std::size_t>(T)] = s1;
    }
    int differing = 0;
    for (int T = 1; T < kDecodeT; ++T)
        for (int t = 0; t < T; ++t)
            differing += same_bits(stores[static_cast<std::size_t>(T)].data() + std::size_t(t) * kH,
                                   stores[kDecodeT].data() + std::size_t(t) * kH, kH * 4)
                             ? 0
                             : 1;
    std::printf(
        "  %-52s token results equal across call widths: %s (%d of 28 differ; informational)\n",
        (b.name + " decode").c_str(), differing == 0 ? "yes" : "no", differing);

    // slots = 0: nothing is resident.
    {
        const Inputs in        = upload(R, 0, 4);
        const std::size_t n    = std::size_t(4) * kH;
        ops::MoeGgufLayer none = blob.layer;
        none.slots             = 0;
        const auto z           = run_decode(g, none, in, ops::MoeSlotsOutput::Store);
        for (float v : z)
            if (v != 0.F) {
                fail(b.name + " decode slots=0: Store did not write zeros");
                break;
            }
        std::vector<float> old(n);
        for (auto& v : old) v = nd(rng);
        const auto keep = run_decode(g, none, in, ops::MoeSlotsOutput::Accumulate, &old);
        if (!same_bits(keep.data(), old.data(), n * 4))
            fail(b.name + " decode slots=0: Accumulate changed out");
    }

    // CUDA graph: captured once, replayed before and after inputs, slot table and offsets change.
    {
        const std::string tag = b.name + " decode graph";
        Inputs in             = upload(R, 0, kDecodeT);
        const std::size_t n   = std::size_t(kDecodeT) * kH;
        DeviceBuffer out(n * 4);
        call_decode(g, blob.layer, in, ops::MoeSlotsOutput::Store, out, DType::FP32);
        cuda_synchronize(g.stream);
        const auto eager_a = from_device<float>(out, n);
        Graph G;
        capture(g, G, [&] {
            call_decode(g, blob.layer, in, ops::MoeSlotsOutput::Store, out, DType::FP32);
        });
        if (!same_bits(replay(g, G, out, n).data(), eager_a.data(), n * 4))
            fail(tag + ": replay != eager");
        rewrite(in, R, kDecodeT);
        reshuffle_blob(blob, b, rng);
        cuda_check(cudaMemsetAsync(out.p, 0xFF, n * 4, g.stream), "memset");
        call_decode(g, blob.layer, in, ops::MoeSlotsOutput::Store, out, DType::FP32);
        cuda_synchronize(g.stream);
        const auto eager_b = from_device<float>(out, n);
        if (!same_bits(replay(g, G, out, n).data(), eager_b.data(), n * 4))
            fail(tag + ": replay after the inputs changed != eager");
        if (same_bits(eager_a.data(), eager_b.data(), n * 4))
            fail(tag + ": the replay check is vacuous");
        restore(blob);
    }
}

// Per-expert assignment counts at the tile edges (rig fork, 10. 11. 2026: the grouped kernels map
// the two token halves of a tile to warps 0-3 / 4-7 and skip the m16 sub-tiles past the tile's
// tokens): resident expert j gets exactly counts[j] tokens, in two passes since a token routes at
// most 10 experts.
void grouped_edge_counts(const Bank& b, Gpu& g, Placement& blob, std::mt19937_64& rng) {
    const std::vector<std::vector<int>> passes = {{1, 8, 15, 16, 17, 31, 32, 33, 63, 64},
                                                  {65, 129, 16, 48}};
    for (std::size_t pass = 0; pass < passes.size(); ++pass) {
        const std::vector<int>& counts = passes[pass];
        const int T                    = *std::max_element(counts.begin(), counts.end());
        Routing R;
        R.x.resize(std::size_t(T) * kH);
        R.ids.assign(std::size_t(T) * kK, -1);
        R.w.resize(std::size_t(T) * kK);
        std::normal_distribution<float> nd(0.F, 0.7F);
        std::uniform_real_distribution<float> wd(0.02F, 0.3F);
        for (auto& v : R.x) v = f32_to_bf16(nd(rng));
        for (auto& v : R.w) v = wd(rng);
        for (int t = 0; t < T; ++t) {
            int k = 0;
            for (std::size_t j = 0; j < counts.size() && k < kK; ++j)
                if (t < counts[j]) R.ids[std::size_t(t) * kK + std::size_t(k++)] = b.ids[j];
        }
        const std::string tag = b.name + " grouped edge counts pass " + std::to_string(pass + 1);
        const std::vector<Ref> refs = oracle(b, R, T, true);
        const auto out              = run_grouped(g, blob.layer, upload(R, 0, T));
        check_numeric(tag, out, refs, T, kGroupedTol);
    }
}

// --------------------------------------------------------------------------- grouped cases
void grouped_cases(const Bank& b, const Routing& R, Gpu& g, Placement& blob, Placement& planar,
                   std::mt19937_64& rng) {
    const std::vector<Ref> refs = oracle(b, R, kGroupedT, true);
    std::vector<float> narrow;
    for (int T : {1, 9, 64, kGroupedT}) {
        const std::string tag = b.name + " grouped T=" + std::to_string(T);
        const Inputs in       = upload(R, 0, T);
        const std::size_t n   = std::size_t(T) * kH;
        const auto s1         = run_grouped(g, blob.layer, in);
        check_numeric(tag + " blob", s1, refs, T, kGroupedTol);
        const auto s2 = run_grouped(g, blob.layer, in);
        if (!same_bits(s1.data(), s2.data(), n * 4)) fail(tag + ": repeated calls differ");
        const auto p = run_grouped(g, planar.layer, in);
        if (!same_bits(s1.data(), p.data(), n * 4))
            fail(tag + ": blob and planar placements differ");
        {
            // capy-flash zero-copy prefill: every other blob-resident expert dropped from the slot
            // table and read in place from the host bank through host_table (same bits).
            std::vector<std::int32_t> t = blob.table_host, h(t.size(), -1);
            for (int j = 0; j < kResident; j += 2) {
                const auto e = static_cast<std::size_t>(b.ids[j]);
                t[e]         = -1;
                h[e]         = planar.table_host[e];
            }
            DeviceBuffer td = to_device(t), hd = to_device(h);
            ops::MoeGgufLayer hy = blob.layer;
            hy.slot_table        = Tensor(td.p, DType::I32, {b.table_size});
            hy.host_table        = static_cast<const std::int32_t*>(hd.p);
            hy.gate_up_base      = planar.layer.gate_up_base;
            hy.down_base         = planar.layer.down_base;
            const auto hv        = run_grouped(g, hy, in);
            if (!same_bits(s1.data(), hv.data(), n * 4))
                fail(tag + ": hybrid (blob + host) differs");
        }
        if (T == kGroupedT) narrow = s1;
    }
    // Two pieces: columns [0, 300) as in the 300-column call, [kPieceT, kWideT) as in a call of
    // their own.
    {
        const std::string tag = b.name + " grouped T=" + std::to_string(kWideT);
        const auto wide       = run_grouped(g, blob.layer, upload(R, 0, kWideT));
        if (!same_bits(wide.data(), narrow.data(), narrow.size() * 4))
            fail(tag + ": columns [0,300) differ from the 300-column call");
        const auto tail = run_grouped(g, blob.layer, upload(R, kPieceT, kWideT - kPieceT));
        if (!same_bits(wide.data() + std::size_t(kPieceT) * kH, tail.data(), tail.size() * 4))
            fail(tag + ": columns past the first piece differ from a call of their own");
    }
    // CUDA graph.
    {
        const std::string tag = b.name + " grouped graph";
        constexpr int T       = 64;
        Inputs in             = upload(R, 0, T);
        const std::size_t n   = std::size_t(T) * kH;
        DeviceBuffer out(n * 4);
        call_grouped(g, blob.layer, in, out);
        cuda_synchronize(g.stream);
        const auto eager_a = from_device<float>(out, n);
        Graph G;
        capture(g, G, [&] { call_grouped(g, blob.layer, in, out); });
        if (!same_bits(replay(g, G, out, n).data(), eager_a.data(), n * 4))
            fail(tag + ": replay != eager");
        rewrite(in, R, T);
        reshuffle_blob(blob, b, rng);
        call_grouped(g, blob.layer, in, out);
        cuda_synchronize(g.stream);
        const auto eager_b = from_device<float>(out, n);
        if (!same_bits(replay(g, G, out, n).data(), eager_b.data(), n * 4))
            fail(tag + ": replay after the inputs changed != eager");
        if (same_bits(eager_a.data(), eager_b.data(), n * 4))
            fail(tag + ": the replay check is vacuous");
        restore(blob);
    }
}

// ----------------------------------------------------------------------------------- bench
// Every token routes the 10 resident test experts 0..9 (shuffled order).
Routing bench_routing(const Bank& b, int tokens, std::mt19937_64& rng) {
    Routing R;
    R.x.resize(std::size_t(tokens) * kH);
    R.ids.resize(std::size_t(tokens) * kK);
    R.w.assign(std::size_t(tokens) * kK, 0.1F);
    std::normal_distribution<float> nd(0.F, 0.7F);
    for (auto& v : R.x) v = f32_to_bf16(nd(rng));
    std::vector<int> js(kK);
    for (int t = 0; t < tokens; ++t) {
        std::iota(js.begin(), js.end(), 0);
        std::shuffle(js.begin(), js.end(), rng);
        for (int k = 0; k < kK; ++k)
            R.ids[std::size_t(t) * kK + k] =
                b.ids[static_cast<std::size_t>(js[static_cast<std::size_t>(k)])];
    }
    return R;
}

double time_us(Gpu& g, int iters, const std::function<void()>& body) {
    cudaEvent_t start = nullptr, stop = nullptr;
    cuda_check(cudaEventCreate(&start), "event");
    cuda_check(cudaEventCreate(&stop), "event");
    cuda_check(cudaEventRecord(start, g.stream), "record");
    for (int i = 0; i < iters; ++i) body();
    cuda_check(cudaEventRecord(stop, g.stream), "record");
    cuda_check(cudaEventSynchronize(stop), "sync");
    float ms = 0.F;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 1000.0 * ms / iters;
}

void bench_bank(const Bank& b, Gpu& g, std::mt19937_64& rng) {
    auto P = make_blob(b, rng, kK);
    for (int T : {1, 4}) {
        const Routing R = bench_routing(b, T, rng);
        const Inputs in = upload(R, 0, T);
        DeviceBuffer out(std::size_t(T) * kH * 4);
        const auto call = [&] {
            call_decode(g, P->layer, in, ops::MoeSlotsOutput::Store, out, DType::FP32);
        };
        for (int i = 0; i < 20; ++i) call();
        cuda_synchronize(g.stream);
        const double eager = time_us(g, 500, call);
        Graph G;
        capture(g, G, call);
        for (int i = 0; i < 20; ++i) cuda_check(cudaGraphLaunch(G.exec, g.stream), "graph launch");
        const double graph =
            time_us(g, 500, [&] { cuda_check(cudaGraphLaunch(G.exec, g.stream), "graph launch"); });
        const double gbs = double(kK) * double(b.expert_bytes) / (graph * 1e3);
        std::printf("  %-34s decode T=%d   eager %8.1f us   graph %8.1f us   (%.0f GB/s of expert "
                    "weights)\n",
                    b.name.c_str(), T, eager, graph, gbs);
    }
    {
        constexpr int T = 2048;
        const Routing R = bench_routing(b, T, rng);
        const Inputs in = upload(R, 0, T);
        DeviceBuffer out(std::size_t(T) * kH * 4);
        const auto call = [&] { call_grouped(g, P->layer, in, out); };
        for (int i = 0; i < 3; ++i) call();
        cuda_synchronize(g.stream);
        const double us    = time_us(g, 20, call);
        const double flops = 2.0 * T * kK * (double(kGU) * kH + double(kH) * kI);
        std::printf("  %-34s grouped T=%d  %10.1f us   (%.1f TFLOP/s)\n", b.name.c_str(), T, us,
                    flops / (us * 1e6));
    }
}

// ------------------------------------------------------------------------------------ main
void run_bank(Bank& b, Gpu& g, std::mt19937_64& rng, bool bench) {
    std::printf("%s\n", b.name.c_str());
    if (bench) {
        bench_bank(b, g, rng);
        return;
    }
    dequantize_bank(b);
    const Routing R = make_routing(b, kWideT, rng);
    auto blob       = make_blob(b, rng);
    auto planar     = make_planar(b, rng);
    decode_cases(b, R, g, *blob, *planar, rng);
    round_bank_bf16(b);
    grouped_cases(b, R, g, *blob, *planar, rng);
    grouped_edge_counts(b, g, *blob, rng);
    b.wgu.clear();
    b.wdn.clear();
}

// ----------------------------------------------------------------------------- probe bench
// --probe FILE --layer L [--rank R] [--group G] [--reps N] [--dump OUT] (with a bank directory):
// times moe_grouped_gguf on real prefill routing (probe files, which no engine build writes: per
// call I32 T, x BF16 [2560,T], ids I32 [10,T], weights FP32 [10,T]) of real layer L, as rank R of
// tp 2 runs it:
// every expert of the layer in a blob pool, the slot table keeping the experts with id % 2 == R
// (the parity partition), calls cut into G-column groups as the T62 exchange does (default 2048).
// Prints per group size the time, the useful TFLOPS (assignments of the partition x 2 x (1280 x
// 2560 + 2560 x 640)) and the per-expert assignment histogram;
// --dump appends every output (FP32 [2560, columns]) for bit comparisons between kernel builds.
struct ProbeCall {
    int T = 0;
    std::vector<std::uint16_t> x;
    std::vector<std::int32_t> ids;
    std::vector<float> w;
};

std::vector<ProbeCall> read_probe(const std::string& path, int min_t) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open probe " + path);
    std::vector<ProbeCall> calls;
    for (;;) {
        std::int32_t T = 0;
        if (!in.read(reinterpret_cast<char*>(&T), 4)) break;
        ProbeCall c;
        c.T = T;
        c.x.resize(std::size_t(T) * kH);
        c.ids.resize(std::size_t(T) * kK);
        c.w.resize(std::size_t(T) * kK);
        in.read(reinterpret_cast<char*>(c.x.data()), std::streamsize(c.x.size() * 2));
        in.read(reinterpret_cast<char*>(c.ids.data()), std::streamsize(c.ids.size() * 4));
        in.read(reinterpret_cast<char*>(c.w.data()), std::streamsize(c.w.size() * 4));
        if (!in) throw std::runtime_error("short probe record in " + path);
        if (T >= min_t) calls.push_back(std::move(c));
    }
    return calls;
}

int probe_bench(const std::string& dir, const std::string& probe, int layer, int rank, int group,
                int reps, const char* dump) {
    const RealLayer* found              = nullptr;
    const std::vector<RealLayer> layers = real_layers(dir);
    for (const RealLayer& r : layers)
        if (r.layer == layer) found = &r;
    if (found == nullptr)
        throw std::runtime_error("layer " + std::to_string(layer) + " not in " + dir);
    Bank b;
    b.name =
        "layer " + std::to_string(layer) + " " + fmt_name(found->gu) + "+" + fmt_name(found->dn);
    b.gu = found->gu;
    b.dn = found->dn;
    init_geometry(b);
    const auto experts = static_cast<int>(std::filesystem::file_size(found->gu_path) / b.gu_bytes);
    std::vector<std::uint8_t> gu(std::size_t(experts) * b.gu_bytes),
        dn(std::size_t(experts) * b.dn_bytes);
    {
        std::ifstream g(found->gu_path, std::ios::binary), d(found->dn_path, std::ios::binary);
        g.read(reinterpret_cast<char*>(gu.data()), std::streamsize(gu.size()));
        d.read(reinterpret_cast<char*>(dn.data()), std::streamsize(dn.size()));
        if (!g || !d) throw std::runtime_error("short bank read");
    }
    // Blob pool: slot s = expert 2s + rank (gate|up then down), 256-byte aligned feet.
    const int slots         = experts / 2;
    const std::int64_t foot = (b.expert_bytes + 255) / 256 * 256;
    std::vector<std::int64_t> offsets(static_cast<std::size_t>(slots));
    std::vector<std::int32_t> table(static_cast<std::size_t>(experts), -1);
    DeviceBuffer pool(static_cast<std::size_t>(foot * slots));
    for (int s = 0; s < slots; ++s) {
        const int e             = 2 * s + rank;
        offsets[std::size_t(s)] = s * foot;
        table[std::size_t(e)]   = s;
        pool.copy_from_host(gu.data() + std::size_t(e) * b.gu_bytes, std::size_t(b.gu_bytes),
                            std::size_t(s * foot));
        pool.copy_from_host(dn.data() + std::size_t(e) * b.dn_bytes, std::size_t(b.dn_bytes),
                            std::size_t(s * foot + b.gu_bytes));
    }
    DeviceBuffer table_d = to_device(table), offsets_d = to_device(offsets);
    ops::MoeGgufLayer L;
    L.slot_table     = Tensor(table_d.p, DType::I32, {experts});
    L.slots          = slots;
    L.slot_offset    = static_cast<const std::int64_t*>(offsets_d.p);
    L.base           = static_cast<const std::byte*>(pool.p);
    L.gate_up_format = b.gu;
    L.down_format    = b.dn;

    const std::vector<ProbeCall> calls = read_probe(probe, 2048);
    std::printf("%s, rank %d, %zu prefill calls, group %d columns\n", b.name.c_str(), rank,
                calls.size(), group);
    Gpu g;
    cuda_check(cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking), "stream");
    g.ws = std::make_unique<DeviceArena>(ops::moe_grouped_gguf_workspace_bytes(group) + 4096);
    std::FILE* out_file = dump != nullptr ? std::fopen(dump, "wb") : nullptr;
    if (dump != nullptr && out_file == nullptr)
        throw std::runtime_error(std::string("cannot write ") + dump);
    cudaEvent_t e0, e1;
    cuda_check(cudaEventCreate(&e0), "event");
    cuda_check(cudaEventCreate(&e1), "event");
    double total_ms = 0, total_flop = 0;
    std::vector<long> hist(8, 0); // assignments per expert per group: 0, 1-15, 16-31, 32-63,
                                  // 64-127, 128-255, 256+, (unused)
    long tiles64 = 0, rows64 = 0, assigned = 0;
    for (const ProbeCall& c : calls) {
        for (int first = 0; first < c.T; first += group) {
            const int n = std::min(group, c.T - first);
            Inputs in;
            in.T = n;
            in.x =
                to_device(std::vector<std::uint16_t>(c.x.begin() + std::ptrdiff_t(first) * kH,
                                                     c.x.begin() + std::ptrdiff_t(first + n) * kH));
            in.ids = to_device(
                std::vector<std::int32_t>(c.ids.begin() + std::ptrdiff_t(first) * kK,
                                          c.ids.begin() + std::ptrdiff_t(first + n) * kK));
            in.w = to_device(std::vector<float>(c.w.begin() + std::ptrdiff_t(first) * kK,
                                                c.w.begin() + std::ptrdiff_t(first + n) * kK));
            std::vector<int> per(static_cast<std::size_t>(experts), 0);
            for (int i = first * kK; i < (first + n) * kK; ++i) {
                const int e = c.ids[std::size_t(i)];
                if (e >= 0 && e < experts && table[std::size_t(e)] >= 0) ++per[std::size_t(e)];
            }
            long a = 0;
            for (int e = rank; e < experts; e += 2) {
                const int k = per[std::size_t(e)];
                a += k;
                tiles64 += (k + 63) / 64;
                rows64 += (k + 63) / 64 * 64;
                hist[k == 0    ? 0
                     : k < 16  ? 1
                     : k < 32  ? 2
                     : k < 64  ? 3
                     : k < 128 ? 4
                     : k < 256 ? 5
                               : 6] += 1;
            }
            assigned += a;
            DeviceBuffer out(std::size_t(n) * kH * 4);
            call_grouped(g, L, in, out); // warm-up
            cuda_check(cudaEventRecord(e0, g.stream), "record");
            for (int r = 0; r < reps; ++r) call_grouped(g, L, in, out);
            cuda_check(cudaEventRecord(e1, g.stream), "record");
            cuda_synchronize(g.stream);
            float ms = 0;
            cuda_check(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
            total_ms += ms / reps;
            total_flop += double(a) * 2.0 * (double(kGU) * kH + double(kH) * kI);
            if (out_file != nullptr) {
                const auto host = from_device<float>(out, std::size_t(n) * kH);
                std::fwrite(host.data(), 4, host.size(), out_file);
            }
        }
    }
    if (out_file != nullptr) std::fclose(out_file);
    std::printf("  groups total %.3f ms, useful %.1f TFLOPS, %ld assignments, 64-row tiles %ld "
                "(fill %.1f %%)\n",
                total_ms, total_flop / (total_ms * 1e-3) / 1e12, assigned, tiles64,
                100.0 * double(assigned) / double(std::max(rows64, 1L)));
    std::printf("  experts per group by assignments: 0: %ld, 1-15: %ld, 16-31: %ld, 32-63: %ld, "
                "64-127: %ld, "
                "128-255: %ld, 256+: %ld\n",
                hist[0], hist[1], hist[2], hist[3], hist[4], hist[5], hist[6]);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    g.ws.reset();
    cudaStreamDestroy(g.stream);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    bool bench        = false;
    const char* dir   = std::getenv("NINFER_GGUF_BANKS");
    const char* probe = nullptr;
    const char* dump  = nullptr;
    int probe_layer = -1, rank = 0, group = 2048, reps = 5, device = -1;
    for (int i = 1; i < argc; ++i) {
        const auto value = [&](const char* name) {
            if (i + 1 >= argc) throw std::runtime_error(std::string(name) + " needs a value");
            return argv[++i];
        };
        if (std::strcmp(argv[i], "--bench") == 0)
            bench = true;
        else if (std::strcmp(argv[i], "--probe") == 0)
            probe = value("--probe");
        else if (std::strcmp(argv[i], "--layer") == 0)
            probe_layer = std::atoi(value("--layer"));
        else if (std::strcmp(argv[i], "--rank") == 0)
            rank = std::atoi(value("--rank"));
        else if (std::strcmp(argv[i], "--group") == 0)
            group = std::atoi(value("--group"));
        else if (std::strcmp(argv[i], "--reps") == 0)
            reps = std::atoi(value("--reps"));
        else if (std::strcmp(argv[i], "--dump") == 0)
            dump = value("--dump");
        else if (std::strcmp(argv[i], "--device") == 0)
            device = std::atoi(value("--device"));
        else
            dir = argv[i];
    }
    if (device >= 0) cuda_check(cudaSetDevice(device), "cudaSetDevice");
    if (probe != nullptr) {
        try {
            if (dir == nullptr || probe_layer < 0)
                throw std::runtime_error("--probe needs a bank directory and --layer");
            cudaDeviceProp p{};
            int d = 0;
            cuda_check(cudaGetDevice(&d), "cudaGetDevice");
            cuda_check(cudaGetDeviceProperties(&p, d), "props");
            std::printf("device %d: %s\n", d, p.name);
            return probe_bench(dir, probe, probe_layer, rank, group, reps, dump);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "FAIL: %s\n", e.what());
            return 1;
        }
    }
    try {
        if (cuda_unavailable()) {
            std::printf("SKIP: no CUDA device\n");
            return 77;
        }
        int device = 0;
        cudaDeviceProp prop{};
        cuda_check(cudaGetDevice(&device), "cudaGetDevice");
        cuda_check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
        if (prop.major != 12) {
            std::printf("SKIP: the build targets sm_120a, device is sm_%d%d\n", prop.major,
                        prop.minor);
            return 77;
        }
        Gpu g;
        cuda_check(cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking), "stream");
        const std::size_t ws = std::max(ops::moe_slots_gguf_workspace_bytes(kDecodeT),
                                        ops::moe_grouped_gguf_workspace_bytes(kWideT));
        g.ws                 = std::make_unique<DeviceArena>(ws + 4096);
        std::mt19937_64 rng(20261019);
        const std::pair<Fmt, Fmt> pairs[] = {
            {Fmt::IQ2_S, Fmt::Q2_0}, {Fmt::IQ2_S, Fmt::IQ4_NL},  {Fmt::IQ3_XXS, Fmt::IQ4_NL},
            {Fmt::IQ3_S, Fmt::Q2_0}, {Fmt::IQ4_XS, Fmt::IQ4_NL}, {Fmt::IQ4_NL, Fmt::IQ4_NL}};
        for (const auto& [gu, dn] : pairs) {
            Bank b = synthetic_bank(gu, dn, rng);
            run_bank(b, g, rng, bench);
        }
        if (dir != nullptr && *dir != '\0') {
            for (const RealLayer& r : real_layers(dir)) {
                Bank b;
                if (load_real_bank(r, rng, b)) run_bank(b, g, rng, bench);
            }
        } else {
            std::printf("real banks: NINFER_GGUF_BANKS not set, synthetic banks only\n");
        }
        g.ws.reset();
        cudaStreamDestroy(g.stream);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    if (failures != 0) {
        std::printf("FAILED: %d\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
