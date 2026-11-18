// The AVX2 GGUF host kernels (host_gguf_avx2.cpp, host_q8_avx2.cpp) against the AVX-512 ones
// (host_gguf_avx512.cpp, host_q8.cpp), bit for bit, on a CPU that runs both (skips otherwise).
//
// Rows: synthetic random blocks of every format (Q8_0 formats also with a trailing half group),
// an edge set (IQ4_NL weights all -127 against activations of -128 and 127: the largest pair sums of
// the AVX2 vpmaddubsw path) and, with NINFER_GGUF_BANKS (tools/convert/gguf_experts.py extract),
// 64 real rows of every bank file, spread over the 512 experts.
//
// Criteria, all exact (memcmp), each format:
//  1. decode_rows: the same decoded bytes (weights and per-lane scales), tiles of 1..8 rows;
//  2. packed activations: quantize_pack_bf16 / _f32 (ggml Q8_K / Q8_0 codes), _q8f32 (bf16, f32),
//     _q8k_g32, pack_act_q8_K / _q8_0 of ggml reference blocks, quantize_pack_q8_0_half(_q8f32),
//     and the host_q8.h quantizers: the same bytes;
//  3. gemm_decoded on each ISA's own decoded rows and packed activations: the same outputs for
//     every T = 1..8 and row tile 1..8 (and each ISA's outputs independent of T and tile);
//  4. silu_mul: the same outputs for n = 1..40, 640, over normal, clamped, subnormal-result and
//     non-finite inputs;
//  5. prefill: decode_rows_bf16 the same BF16 values; gemm_bf16 the same outputs (reported);
//  6. HostMoeExecutor (Options::isa = Avx2 vs Avx512Vnni) on GGUF layers: the same outputs for
//     calls of T = 1..8 tokens (synthetic layers and, with NINFER_GGUF_BANKS, the real ones).
#include "ops/host_moe/host_gguf.h"
#include "ops/host_moe/host_moe_executor.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace h = ninfer::ops::host;
namespace g = ninfer::ops::host::gguf;
using g::ActFormat;
using g::WeightFormat;

namespace {

constexpr int kMaxT = 8;
int g_failures = 0;

struct Tally {
  long long compared = 0, differing = 0;
};
std::map<std::string, Tally> g_tally;

// Counts `n` compared items of `what`, `bad` of them differing.
void tally(const std::string& what, long long n, long long bad, const std::string& where) {
  auto& t = g_tally[what];
  t.compared += n;
  t.differing += bad;
  if (bad) {
    ++g_failures;
    std::fprintf(stderr, "FAIL %s: %s: %lld of %lld differ\n", where.c_str(), what.c_str(), bad, n);
  }
}

long long diff_bytes(const void* a, const void* b, std::size_t n) {
  const auto* x = static_cast<const std::uint8_t*>(a);
  const auto* y = static_cast<const std::uint8_t*>(b);
  long long d = 0;
  for (std::size_t i = 0; i < n; ++i) d += x[i] != y[i];
  return d;
}
long long diff_words(const void* a, const void* b, std::size_t n, std::size_t size) {
  const auto* x = static_cast<const std::uint8_t*>(a);
  const auto* y = static_cast<const std::uint8_t*>(b);
  long long d = 0;
  for (std::size_t i = 0; i < n; ++i) d += std::memcmp(x + i * size, y + i * size, size) != 0;
  return d;
}

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
  std::size_t n = 0;
  explicit AlignedBuf(std::size_t bytes, std::uint8_t fill = 0xA5) {
    n = (bytes + 63) / 64 * 64;
    p = static_cast<std::uint8_t*>(std::aligned_alloc(64, n));
    std::memset(p, fill, n);
  }
  ~AlignedBuf() { std::free(p); }
  AlignedBuf(const AlignedBuf&) = delete;
  AlignedBuf& operator=(const AlignedBuf&) = delete;
};

struct RowSet {
  std::string label;
  WeightFormat fmt;
  int k = 0;
  int rows = 0;
  std::size_t row_bytes = 0;
  std::vector<std::uint8_t> data;
};

RowSet synthetic(WeightFormat fmt, int k, int rows, std::mt19937_64& rng) {
  RowSet s{std::string("synthetic ") + g::format_name(fmt) + " k" + std::to_string(k), fmt, k, rows,
           g::weight_row_bytes(fmt, k), {}};
  s.data.resize(s.row_bytes * rows);
  for (auto& b : s.data) b = static_cast<std::uint8_t>(rng());
  const g::FormatInfo fi = g::format_info(fmt);
  std::uniform_real_distribution<float> d(1e-3F, 4e-2F);
  for (int r = 0; r < rows; ++r) {
    for (int b = 0; b < k / fi.block_elems; ++b) {
      const std::uint16_t hv = GGML_FP32_TO_FP16(d(rng) * ((rng() & 1) ? 1.F : -1.F));
      std::memcpy(s.data.data() + r * s.row_bytes + std::size_t(b) * fi.block_bytes, &hv, 2);
    }
  }
  return s;
}

// IQ4_NL rows whose codes are all 0 (w = -127, the largest |w| of any format).
RowSet edge_iq4nl(int k, int rows) {
  RowSet s{"edge IQ4_NL w=-127 k" + std::to_string(k), WeightFormat::IQ4_NL, k, rows,
           g::weight_row_bytes(WeightFormat::IQ4_NL, k), {}};
  s.data.assign(s.row_bytes * rows, 0);
  for (int r = 0; r < rows; ++r)
    for (int b = 0; b < k / 32; ++b) {
      const std::uint16_t hv = GGML_FP32_TO_FP16(0.01F * float(1 + (r + b) % 5));
      std::memcpy(s.data.data() + r * s.row_bytes + std::size_t(b) * sizeof(block_iq4_nl), &hv, 2);
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
      RowSet s{"layer " + layer + " " + role + " " + g::format_name(fmt), fmt, e.at("shape")[1].get<int>(), rows,
               e.at("row_bytes").get<std::size_t>(), {}};
      const long long total = e.at("shape")[0].get<long long>();
      s.data.resize(s.row_bytes * rows);
      const int fd = ::open((dir + "/" + e.at("file").get<std::string>()).c_str(), O_RDONLY);
      if (fd < 0) continue;
      for (int r = 0; r < rows; ++r) {
        const long long row = (total - 1) * r / std::max(rows - 1, 1);
        const ssize_t got = ::pread(fd, s.data.data() + r * s.row_bytes, s.row_bytes,
                                    static_cast<off_t>(row) * static_cast<off_t>(s.row_bytes));
        if (got != static_cast<ssize_t>(s.row_bytes)) tally("bank reads", 1, 1, s.label);
      }
      ::close(fd);
      out.push_back(std::move(s));
    }
  }
  return out;
}

// One packed-activation variant of T rows for both ISAs.
struct Acts {
  std::string name;
  std::size_t pb = 0;
  std::vector<std::unique_ptr<AlignedBuf>> buf;  // [2]: AVX-512, AVX2
  const void* ptr[2][kMaxT] = {};
};

void check_set(const RowSet& s, std::mt19937_64& rng, const g::HostGgufKernels* K[2], bool edge = false) {
  const int k = s.k;
  const ActFormat af = g::format_info(s.fmt).act;
  // Activations: BF16 and FP32 rows with varied scales, outliers, one all-zero 32-block, ties.
  std::normal_distribution<float> nd(0.F, 1.F);
  std::vector<std::uint16_t> xb(std::size_t(kMaxT) * k);
  std::vector<float> xf(std::size_t(kMaxT) * k);
  for (int t = 0; t < kMaxT; ++t) {
    const float scale = 0.05F + 2.F * float(t) / kMaxT;
    for (int i = 0; i < k; ++i) {
      float v = nd(rng) * scale;
      if (i % 97 == 13) v *= 20.F;
      if (t == 3 && i < 32) v = 0.F;
      if (t == 5 && i % 64 == 7) v = 0.5F * float(int(v * 4.F));  // halves: round_away ties
      xb[std::size_t(t) * k + i] = bf16_rne(v);
      xf[std::size_t(t) * k + i] = t == 6 ? v : bf16_value(bf16_rne(v));
    }
  }
  const std::size_t pb = g::packed_act_bytes(af, k);
  const std::size_t arb = g::act_row_bytes(af, k);
  std::vector<Acts> acts;
  const auto make = [&](const std::string& name, auto&& fill) {
    Acts a;
    a.name = name;
    a.pb = pb;
    for (int i = 0; i < 2; ++i) {
      a.buf.push_back(std::make_unique<AlignedBuf>(pb * kMaxT));
      for (int t = 0; t < kMaxT; ++t) {
        fill(*K[i], t, a.buf[i]->p + pb * t);
        a.ptr[i][t] = a.buf[i]->p + pb * t;
      }
    }
    tally("packed activation bytes (" + name + ")", (long long)(pb * kMaxT),
          diff_bytes(a.buf[0]->p, a.buf[1]->p, pb * kMaxT), s.label);
    acts.push_back(std::move(a));
  };
  make("quantize_pack_bf16", [&](const g::HostGgufKernels& Kk, int t, void* d) {
    Kk.quantize_pack_bf16(af, xb.data() + std::size_t(t) * k, k, d);
  });
  make("quantize_pack_f32", [&](const g::HostGgufKernels& Kk, int t, void* d) {
    Kk.quantize_pack_f32(af, xf.data() + std::size_t(t) * k, k, d);
  });
  make("quantize_pack_bf16_q8f32", [&](const g::HostGgufKernels& Kk, int t, void* d) {
    Kk.quantize_pack_bf16_q8f32(af, xb.data() + std::size_t(t) * k, k, d);
  });
  make("quantize_pack_f32_q8f32", [&](const g::HostGgufKernels& Kk, int t, void* d) {
    Kk.quantize_pack_f32_q8f32(af, xf.data() + std::size_t(t) * k, k, d);
  });
  // ggml reference blocks -> pack_act.
  std::vector<std::uint8_t> ref(arb * kMaxT);
  for (int t = 0; t < kMaxT; ++t) {
    if (af == ActFormat::Q8_K)
      quantize_row_q8_K_ref(xf.data() + std::size_t(t) * k, reinterpret_cast<block_q8_K*>(ref.data() + arb * t), k);
    else
      quantize_row_q8_0_ref(xf.data() + std::size_t(t) * k, reinterpret_cast<block_q8_0*>(ref.data() + arb * t), k);
  }
  if (edge && af == ActFormat::Q8_0) {
    // Codes -128 and 127 (no ggml quantizer produces -128; pack_act accepts any block).
    for (int t = 0; t < kMaxT; ++t)
      for (int b = 0; b < k / 32; ++b) {
        auto* blk = reinterpret_cast<block_q8_0*>(ref.data() + arb * t) + b;
        for (int i = 0; i < 32; ++i) blk->qs[i] = int8_t((t + b + i) % 3 == 0 ? 127 : -128);
      }
  }
  make("pack_act ggml ref", [&](const g::HostGgufKernels& Kk, int t, void* d) {
    if (af == ActFormat::Q8_K) Kk.pack_act_q8_K(reinterpret_cast<const block_q8_K*>(ref.data() + arb * t), k, d);
    else Kk.pack_act_q8_0(reinterpret_cast<const block_q8_0*>(ref.data() + arb * t), k, d);
  });
  if (af == ActFormat::Q8_K) {
    make("quantize_pack_bf16_q8k_g32", [&](const g::HostGgufKernels& Kk, int t, void* d) {
      Kk.quantize_pack_bf16_q8k_g32(xb.data() + std::size_t(t) * k, k, d);
    });
  } else if (k % 64 == 0) {
    // The executor's h path: 32-value halves written one at a time.
    make("quantize_pack_q8_0_half_q8f32", [&](const g::HostGgufKernels& Kk, int t, void* d) {
      for (int b = 0; b < k / 32; ++b) Kk.quantize_pack_q8_0_half_q8f32(xf.data() + std::size_t(t) * k + 32 * b, b / 2, b % 2, d);
    });
    make("quantize_pack_q8_0_half", [&](const g::HostGgufKernels& Kk, int t, void* d) {
      for (int b = 0; b < k / 32; ++b) Kk.quantize_pack_q8_0_half(xf.data() + std::size_t(t) * k + 32 * b, b / 2, b % 2, d);
    });
  }
  // host_q8.h quantizers.
  {
    std::vector<std::uint8_t> qa(arb * kMaxT, 0x5A), qb(arb * kMaxT, 0x5A);
    for (int t = 0; t < kMaxT; ++t) {
      for (int i = 0; i < 2; ++i) {
        std::uint8_t* d = (i == 0 ? qa : qb).data() + arb * t;
        const bool f32 = t % 2 == 1;
        if (af == ActFormat::Q8_K) {
          if (f32) K[i]->quantize_row_q8_K_f32(xf.data() + std::size_t(t) * k, reinterpret_cast<block_q8_K*>(d), k);
          else K[i]->quantize_row_q8_K_bf16(xb.data() + std::size_t(t) * k, reinterpret_cast<block_q8_K*>(d), k);
        } else {
          if (f32) K[i]->quantize_row_q8_0_f32(xf.data() + std::size_t(t) * k, reinterpret_cast<block_q8_0*>(d), k);
          else K[i]->quantize_row_q8_0_bf16(xb.data() + std::size_t(t) * k, reinterpret_cast<block_q8_0*>(d), k);
        }
      }
    }
    tally("host_q8 quantizer bytes", (long long)qa.size(), diff_bytes(qa.data(), qb.data(), qa.size()), s.label);
  }

  // (1) decode, (3) dots.
  const std::size_t drb = g::decoded_row_bytes(s.fmt, k);
  AlignedBuf dec0(drb * 8), dec1(drb * 8);
  for (int r0 = 0; r0 < s.rows; r0 += 8) {
    const int nr = std::min(8, s.rows - r0);
    const std::uint8_t* rows = s.data.data() + std::size_t(r0) * s.row_bytes;
    for (int tile = 1; tile <= nr; ++tile) {
      std::memset(dec0.p, 0x33, dec0.n);
      std::memset(dec1.p, 0x33, dec1.n);
      for (int r = 0; r < nr; r += tile) {
        const int n = std::min(tile, nr - r);
        K[0]->decode_rows(s.fmt, rows + std::size_t(r) * s.row_bytes, s.row_bytes, n, k, dec0.p + drb * r);
        K[1]->decode_rows(s.fmt, rows + std::size_t(r) * s.row_bytes, s.row_bytes, n, k, dec1.p + drb * r);
      }
      tally("decoded bytes", (long long)(drb * nr), diff_bytes(dec0.p, dec1.p, drb * nr), s.label);
    }
    for (const Acts& a : acts) {
      float ref8[2][8 * kMaxT];
      for (int i = 0; i < 2; ++i)
        K[i]->gemm_decoded(s.fmt, i == 0 ? dec0.p : dec1.p, nr, k, a.ptr[i], kMaxT, ref8[i], kMaxT);
      long long bad = 0, n = 0, dep = 0;
      for (int tile = 1; tile <= nr; ++tile) {
        for (int T = 1; T <= kMaxT; ++T) {
          float o[2][8 * kMaxT];
          for (int i = 0; i < 2; ++i) {
            const std::uint8_t* d = i == 0 ? dec0.p : dec1.p;
            for (int r = 0; r < nr; r += tile)
              K[i]->gemm_decoded(s.fmt, d + drb * r, std::min(tile, nr - r), k, a.ptr[i], T, o[i] + r * kMaxT, kMaxT);
          }
          for (int r = 0; r < nr; ++r)
            for (int t = 0; t < T; ++t) {
              ++n;
              bad += std::memcmp(&o[0][r * kMaxT + t], &o[1][r * kMaxT + t], 4) != 0;
              for (int i = 0; i < 2; ++i) dep += std::memcmp(&o[i][r * kMaxT + t], &ref8[i][r * kMaxT + t], 4) != 0;
            }
        }
      }
      tally("gemm_decoded outputs (" + a.name + ")", n, bad, s.label);
      tally("gemm_decoded T/tile independence (both ISAs)", 2 * n, dep, s.label);
    }
  }

  // (5) prefill.
  std::vector<std::uint16_t> b0(std::size_t(s.rows) * k, 1), b1(std::size_t(s.rows) * k, 2);
  K[0]->decode_rows_bf16(s.fmt, s.data.data(), s.row_bytes, s.rows, k, b0.data());
  K[1]->decode_rows_bf16(s.fmt, s.data.data(), s.row_bytes, s.rows, k, b1.data());
  tally("decode_rows_bf16 values", (long long)b0.size(), diff_words(b0.data(), b1.data(), b0.size(), 2), s.label);
  {
    const void* ap[kMaxT];
    for (int t = 0; t < kMaxT; ++t) ap[t] = xb.data() + std::size_t(t) * k;
    std::vector<float> o0(std::size_t(s.rows) * kMaxT), o1(o0.size());
    long long bad = 0;
    double worst = 0;
    for (int T : {1, 3, 8}) {
      K[0]->gemm_bf16(b0.data(), s.rows, k, ap, T, o0.data(), kMaxT);
      K[1]->gemm_bf16(b0.data(), s.rows, k, ap, T, o1.data(), kMaxT);
      for (int r = 0; r < s.rows; ++r)
        for (int t = 0; t < T; ++t) {
          const float a = o0[r * kMaxT + t], b = o1[r * kMaxT + t];
          if (std::memcmp(&a, &b, 4)) {
            ++bad;
            worst = std::max(worst, std::fabs(double(a) - b) / std::max(std::fabs(double(a)), 1e-30));
          }
        }
    }
    tally("gemm_bf16 outputs", (long long)s.rows * (1 + 3 + 8), bad, s.label);
    if (bad) std::fprintf(stderr, "  gemm_bf16: worst relative difference %.3g\n", worst);
  }
  std::printf("%-36s rows %3d k %4d checked\n", s.label.c_str(), s.rows, k);
}

void check_silu(const g::HostGgufKernels* K[2], std::mt19937_64& rng) {
  std::vector<float> gv, uv;
  std::normal_distribution<float> nd(0.F, 1.F);
  std::uniform_real_distribution<float> big(-115.F, 115.F);
  for (int i = 0; i < 4096; ++i) {
    gv.push_back(i % 4 == 0 ? big(rng) : nd(rng) * 6.F);
    uv.push_back(nd(rng) * 3.F);
  }
  // Clamp edges, subnormal-result region (g in [87, 104]), n = 128 (g near -88.5), non-finite.
  for (float v = 80.F; v < 110.F; v += 0.37F) gv.push_back(v), uv.push_back(1.F);
  for (float v = -110.F; v < -80.F; v += 0.37F) gv.push_back(v), uv.push_back(1.F);
  for (float v : {-103.F, 103.F, -88.5F, 88.5F, 0.F, -0.F, 1e-40F, -1e-40F, std::numeric_limits<float>::infinity(),
                  -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
    gv.push_back(v);
    uv.push_back(1.5F);
  }
  long long n = 0, bad = 0;
  int reported = 0;
  std::vector<float> h0(gv.size() + 1), h1(gv.size() + 1);
  std::vector<int> lens;
  for (int l = 1; l <= 40; ++l) lens.push_back(l);
  lens.push_back(640);
  lens.push_back(int(gv.size()));
  for (int len : lens) {
    for (std::size_t off = 0; off + len <= gv.size(); off += std::max<std::size_t>(len, 97)) {
      std::fill(h0.begin(), h0.end(), 7.F);
      std::fill(h1.begin(), h1.end(), 7.F);
      K[0]->silu_mul(gv.data() + off, uv.data() + off, h0.data(), len);
      K[1]->silu_mul(gv.data() + off, uv.data() + off, h1.data(), len);
      n += len + 1;
      bad += diff_words(h0.data(), h1.data(), std::size_t(len) + 1, 4);  // + the untouched next value
      for (int i = 0; i < len; ++i)
        if (std::memcmp(&h0[i], &h1[i], 4) && reported++ < 4)
          std::fprintf(stderr, "  silu_mul(g = %a, u = %a): AVX-512 %a, AVX2 %a\n", gv[off + i], uv[off + i], h0[i], h1[i]);
    }
  }
  tally("silu_mul outputs", n, bad, "silu");
}

// ---------------------------------------------------------------- executor
constexpr int H = 2560, I = 640, KTOP = 10;

struct Layer {
  std::string name;
  h::HostExpertLayer layer;
  std::vector<std::uint8_t> gu, dn;
};

void finish(Layer& L, WeightFormat gu, WeightFormat dn, int experts) {
  L.layer.format = h::HostExpertFormat::Gguf;
  L.layer.gguf.gate_up = gu;
  L.layer.gguf.down = dn;
  L.layer.gguf.gate_up_rows = L.gu.data();
  L.layer.gguf.down_rows = L.dn.data();
  L.layer.gguf.gate_up_row_bytes = g::weight_row_bytes(gu, H);
  L.layer.gguf.down_row_bytes = g::weight_row_bytes(dn, I);
  L.layer.experts = experts;
  L.layer.hidden = H;
  L.layer.intermediate = I;
}

void check_executor(const Layer& L, std::mt19937_64& rng, h::HostCpuPool& pool) {
  h::HostMoeExecutor::Options o512, o2;
  o512.isa = h::HostIsa::Avx512Vnni;
  o2.isa = h::HostIsa::Avx2;
  h::HostMoeExecutor e512(pool, H, I, kMaxT, KTOP, o512), e2(pool, H, I, kMaxT, KTOP, o2);
  std::normal_distribution<float> nd(0.F, 1.F);
  std::uniform_int_distribution<int> pick(0, L.layer.experts - 1);
  long long n = 0, bad = 0;
  for (int T = 1; T <= kMaxT; ++T) {
    std::vector<std::uint16_t> x(std::size_t(T) * H);
    for (auto& v : x) v = bf16_rne(nd(rng) * 0.7F);
    std::vector<std::int32_t> ids(std::size_t(T) * KTOP, -1);
    std::vector<float> alpha(std::size_t(T) * KTOP, 0.F);
    for (int t = 0; t < T; ++t) {
      const int m = 1 + (t * 3 + T) % 6;
      for (int j = 0; j < m; ++j) {
        int e;
        do e = pick(rng);
        while (std::find(ids.begin() + t * KTOP, ids.begin() + t * KTOP + j, e) != ids.begin() + t * KTOP + j);
        ids[std::size_t(t) * KTOP + j] = e;
        alpha[std::size_t(t) * KTOP + j] = 0.05F + 0.07F * float(j);
      }
    }
    std::vector<float> a(std::size_t(T) * H, 1.F), b(std::size_t(T) * H, 2.F);
    e512.run(L.layer, h::HostMoeJob{T, KTOP, x.data(), ids.data(), alpha.data(), a.data()});
    e2.run(L.layer, h::HostMoeJob{T, KTOP, x.data(), ids.data(), alpha.data(), b.data()});
    n += (long long)a.size();
    bad += diff_words(a.data(), b.data(), a.size(), 4);
  }
  tally("executor outputs", n, bad, L.name);
  std::printf("%-36s executor T = 1..%d checked\n", L.name.c_str(), kMaxT);
}

Layer synthetic_layer(WeightFormat gu, WeightFormat dn, int experts, std::mt19937_64& rng) {
  Layer L;
  L.name = std::string("executor synthetic ") + g::format_name(gu) + "+" + g::format_name(dn);
  const std::size_t rbg = g::weight_row_bytes(gu, H), rbd = g::weight_row_bytes(dn, I);
  L.gu.resize(rbg * 2 * I * experts);
  L.dn.resize(rbd * H * experts);
  std::uniform_real_distribution<float> dd(5e-4F, 2e-2F);
  const auto fill = [&](std::vector<std::uint8_t>& v, WeightFormat f, std::size_t rb, int k, std::size_t rows) {
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
  finish(L, gu, dn, experts);
  return L;
}

}  // namespace

int main(int argc, char** argv) {
  if (!g::host_gguf_isa_supported(h::HostIsa::Avx512Vnni) || !g::host_gguf_isa_supported(h::HostIsa::Avx2)) {
    std::printf("SKIP: comparing the ISAs needs a CPU that runs both (AVX-512 GGUF set and AVX2)\n");
    return 77;
  }
  const g::HostGgufKernels* K[2] = {&g::host_gguf_kernels(h::HostIsa::Avx512Vnni),
                                    &g::host_gguf_kernels(h::HostIsa::Avx2)};
  std::mt19937_64 rng(0x15A2ULL);
  const struct {
    WeightFormat fmt;
    int k;
  } kFormats[] = {{WeightFormat::IQ2_S, 2560},  {WeightFormat::IQ3_XXS, 2560}, {WeightFormat::IQ3_S, 2560},
                  {WeightFormat::IQ4_XS, 2560}, {WeightFormat::IQ4_NL, 640},   {WeightFormat::Q2_0, 640},
                  {WeightFormat::IQ4_NL, 2560}, {WeightFormat::Q2_0, 2560},    {WeightFormat::IQ4_NL, 672},
                  {WeightFormat::IQ2_S, 256},   {WeightFormat::IQ3_S, 512}};
  for (const auto& f : kFormats) check_set(synthetic(f.fmt, f.k, 21, rng), rng, K);
  check_set(edge_iq4nl(640, 16), rng, K, true);
  check_silu(K, rng);
  const char* dir = argc > 1 ? argv[1] : std::getenv("NINFER_GGUF_BANKS");
  const bool banks = dir != nullptr && *dir;
  if (banks) {
    const auto sets = bank_rows(dir, 64);
    if (sets.empty()) tally("bank rows", 1, 1, dir);
    for (const auto& s : sets) check_set(s, rng, K);
  } else {
    std::printf("(NINFER_GGUF_BANKS not set: real rows and layers not checked)\n");
  }
  {
    const auto cpus = h::spread_cpus(4);
    h::HostCpuPool pool(cpus);
    const std::pair<WeightFormat, WeightFormat> pairs[] = {{WeightFormat::IQ2_S, WeightFormat::Q2_0},
                                                           {WeightFormat::IQ3_XXS, WeightFormat::IQ4_NL},
                                                           {WeightFormat::IQ3_S, WeightFormat::Q2_0},
                                                           {WeightFormat::IQ4_XS, WeightFormat::IQ4_NL}};
    for (const auto& [gu, dn] : pairs) {
      const Layer L = synthetic_layer(gu, dn, 12, rng);
      check_executor(L, rng, pool);
    }
    if (banks) {
      std::ifstream in(std::string(dir) + "/manifest.json");
      const auto m = nlohmann::json::parse(in);
      for (const auto& [layer, e] : m.at("layers").items()) {
        WeightFormat gu, dn;
        if (!format_of(e.at("gate_up").at("format"), &gu) || !format_of(e.at("down").at("format"), &dn)) continue;
        Layer L;
        L.name = "executor layer " + layer + " " + g::format_name(gu) + "+" + g::format_name(dn);
        for (auto [role, buf] : {std::pair{"gate_up", &L.gu}, std::pair{"down", &L.dn}}) {
          std::ifstream f(std::string(dir) + "/" + e.at(role).at("file").get<std::string>(), std::ios::binary);
          buf->assign(std::istreambuf_iterator<char>(f), {});
        }
        finish(L, gu, dn, 512);
        check_executor(L, rng, pool);
      }
    }
  }
  std::printf("\n%-58s %14s %10s\n", "compared (AVX2 vs AVX-512)", "values", "differing");
  for (const auto& [what, t] : g_tally) std::printf("%-58s %14lld %10lld\n", what.c_str(), t.compared, t.differing);
  if (g_failures) {
    std::printf("FAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
