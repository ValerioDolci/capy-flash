// Two-device parity of the Qwen4Exp routed-expert MoE at tp 2: routed experts partitioned by
// expert ID (rank r computes the experts with id % 2 == r: its cache-resident ones with
// moe_slots_nvfp4 Store, the ones missing from its cache from the CPU path, accumulated into the
// same FP32 partial), the two FP32 rank partials summed by ONE allreduce_sum_f32 and rounded to
// BF16 once. Reference: the single-device op over every expert resident (Store FP32, then BF16).
//
// The CPU-path partial is emulated on the device by a second slot pool holding the rank's
// "missing" experts and moe_slots_nvfp4 Accumulate into the rank's FP32 partial: the data flow and
// the rounding point (everything a rank contributes accumulated in FP32, one BF16 rounding after
// the cross-rank sum) are the production ones; test_host_moe checks the CPU arithmetic itself.
//
// Criteria (stated before the first run):
//  * FP32 sum vs the single-device FP32 partial: |tp2 - full| <= 2^-15 * max_t|full[:,t]|. The two
//    sums add the same <= 10 weighted expert outputs per element in different orders (rank,
//    resident/missing, then the cross-rank add): the reassociation error is <= 2 * 9 * 2^-24 times
//    the sum of the magnitudes, a few times max|full|; 2^-15 is ~2^-17 margin over that and ~64x
//    below one BF16 ulp, so a dropped, doubled or misweighted expert (>= 1e-2 relative) fails.
//  * BF16 out == RNE(FP32 sum) exactly on both ranks (the Op's definition), both ranks identical,
//    and within one BF16 ulp of the single-device BF16 output (they differ only where the two FP32
//    values straddle a rounding boundary; the count is reported).
//  * W4A4 recipe oracle (FP64, test_moe_slots.cpp's): relative L2 <= 5e-3, as the one-GPU Op.
//  * Exact: a call whose routed experts all belong to one rank and are all resident equals the
//    single-device result bit for bit (same order, the other rank adds +0).
// Transports: the staged all-reduce eagerly, and the whole rank computation plus the all-reduce
// captured into one two-device CUDA graph with the pipelined mailbox, replayed.
//
// Fixture (synthetic NVFP4 experts at the real geometry, calibration, FP64 recipe oracle) copied
// from tests/ops/test_moe_slots.cpp.
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/moe_slots.h"
#include "ninfer/ops/peer_mailbox.h"

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/nvfp4_test_codec.h"
#include "ops/op_tester.h"
#include "ops/split_test_support.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kH = 2560, kI = 640, kK = 10, kE = 512, kRows13 = 2 * kI;
constexpr int kHB = kH / 16, kIB = kI / 16;
constexpr std::int64_t kSlotBytes = ops::kMoeSlotsGateUpCodeBytes + ops::kMoeSlotsGateUpScaleBytes +
                                    ops::kMoeSlotsDownCodeBytes +
                                    ops::kMoeSlotsDownScaleBytes; // 2 764 800

// W4A4 vs the unquantized FP64 oracle. Two NVFP4 activation quantizations in series (x before
// gemm1, h before gemm2); the single-quantization A4 allowance of the Linear suites is 0.16 and
// F0-A measured 0.157-0.164 for this recipe (author's kernel, same data model). 0.18 covers the
// two stages with ~10 % margin over F0-A. Gross cap: the largest error of a 2560*T Gaussian-like
// error field relative to the largest reference value is ~ the relative L2 (both extremes
// ~4 sigma), so 0.30 is ~2x.
constexpr ReductionCriterion kW4A4{0.18, 0.0, 0.30};
// Recipe oracle: same quantization formula evaluated in FP64/FP32 on the host. Differences are
// FP32 accumulation order, rcp.approx/__expf vs exact, which flip a BF16 rounding of a gemm1
// output or h with probability ~1e-4 and an E2M1 code only rarely; one flipped h code moves one
// token's output by < 1e-2 relative. 5e-3 relative L2 and 5e-2 gross are well above that and
// ~30x below the W4A4 error, so any scale, divisor or slot fault fails.
constexpr ReductionCriterion kRecipe{5.0e-3, 0.0, 5.0e-2};
constexpr float kDivisorJitter = 0.98742F; // see reference()

// The documented activation profile for 16 values: codes (as values) and the E4M3 scale.
void quant16(const float* v, float gs, float* codes, float& sf) {
    float amax = 0.F;
    for (int i = 0; i < 16; ++i) amax = std::max(amax, std::fabs(v[i]));
    sf                    = e4m3_rn(gs * (amax * (1.0F / 6.0F)));
    const float out_scale = (amax != 0.F && sf != 0.F) ? 1.0F / (sf * (1.0F / gs)) : 0.F;
    for (int i = 0; i < 16; ++i) codes[i] = e2m1_rn(v[i] * out_scale);
}

float bf16_round(float v) { return bf16_to_f32(f32_to_bf16(v)); }

std::size_t swizzled_offset(int row, int group, int k_tiles) {
    const int inner = row % 128;
    return (static_cast<std::size_t>(row / 128) * k_tiles + group / 4) * 512 +
           static_cast<std::size_t>(inner % 32) * 16 + static_cast<std::size_t>(inner / 32) * 4 +
           group % 4;
}

void parallel_for(int n, const std::function<void(int)>& fn) {
    const int cores   = static_cast<int>(std::min(12U, std::thread::hardware_concurrency()));
    const int workers = std::max(1, std::min(n, cores));
    std::vector<std::thread> pool;
    std::vector<std::exception_ptr> errors(workers);
    for (int w = 0; w < workers; ++w) {
        pool.emplace_back([&, w] {
            try {
                for (int i = w; i < n; i += workers) fn(i);
            } catch (...) { errors[w] = std::current_exception(); }
        });
    }
    for (auto& t : pool) t.join();
    for (auto& e : errors)
        if (e) std::rethrow_exception(e);
}

// One routed expert in the bank's row order: gate|up [1280,2560] (gate rows first), down
// [2560,640]; natural (row-major) scale words.
struct Expert {
    std::vector<std::uint8_t> w13, s13, w2, s2;
};

Expert make_expert(std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte(0, 255), scale(0x30, 0x3F), zero(0, 63);
    Expert x;
    x.w13.resize(static_cast<std::size_t>(kRows13) * kH / 2);
    x.s13.resize(static_cast<std::size_t>(kRows13) * kHB);
    x.w2.resize(static_cast<std::size_t>(kH) * kI / 2);
    x.s2.resize(static_cast<std::size_t>(kH) * kIB);
    for (auto& b : x.w13) b = static_cast<std::uint8_t>(byte(rng));
    for (auto& b : x.w2) b = static_cast<std::uint8_t>(byte(rng));
    // valid stored scales: positive finite words, a few zero blocks
    for (auto& b : x.s13) b = static_cast<std::uint8_t>(zero(rng) == 0 ? 0 : scale(rng));
    for (auto& b : x.s2) b = static_cast<std::uint8_t>(zero(rng) == 0 ? 0 : scale(rng));
    return x;
}

// code * block scale, without the divisor (exact in FP32)
void decode(const std::vector<std::uint8_t>& codes, const std::vector<std::uint8_t>& scales,
            int rows, int cols, std::vector<float>& out) {
    out.resize(static_cast<std::size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const std::uint8_t byte = codes[(static_cast<std::size_t>(r) * cols + c) / 2];
            const int code          = (c % 2 == 0) ? (byte & 15) : (byte >> 4);
            out[static_cast<std::size_t>(r) * cols + c] =
                kE2M1[code] *
                e4m3_decode(scales[static_cast<std::size_t>(r) * (cols / 16) + c / 16]);
        }
}

struct Model {
    std::vector<int> ids;      // materialized expert ids
    std::vector<int> index_of; // [512] -> index in experts or -1
    std::vector<Expert> experts;
    std::vector<float> div13, div2; // [2E], [E]
};

Model make_model(int count, std::uint32_t seed) {
    Model m;
    std::mt19937 rng(seed);
    std::vector<int> all(kE);
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), rng);
    m.ids.assign(all.begin(), all.begin() + count);
    m.index_of.assign(kE, -1);
    m.experts.resize(count);
    for (int i = 0; i < count; ++i) m.index_of[m.ids[i]] = i;
    parallel_for(count, [&](int i) { m.experts[i] = make_expert(seed * 7919U + 17U * i + 1U); });
    // distinct gate/up/down divisors per expert (ModelOpt: 1/weight_scale_2), 250..1000
    std::uniform_real_distribution<float> div(250.0F, 1000.0F);
    m.div13.resize(2 * kE);
    m.div2.resize(kE);
    for (auto& d : m.div13) d = div(rng);
    for (auto& d : m.div2) d = div(rng);
    return m;
}

enum class Layout { Planar, Contiguous };

// A device slot pool; placement[i] = slot of materialized expert i, or -1. Unused slots hold
// random bytes (including NaN scale words) that must never be read.
struct DevicePool {
    DeviceBuffer memory;
    ops::MoeSlotPool pool;
};

DevicePool make_pool(const Model& m, const std::vector<std::vector<int>>& placement, int slots,
                     Layout layout, std::uint32_t seed) {
    const std::int64_t plane_bytes[4] = {ops::kMoeSlotsGateUpCodeBytes,
                                         ops::kMoeSlotsGateUpScaleBytes,
                                         ops::kMoeSlotsDownCodeBytes, ops::kMoeSlotsDownScaleBytes};
    std::int64_t base[4], stride[4];
    if (layout == Layout::Planar) {
        std::int64_t at = 0;
        for (int q = 0; q < 4; ++q) {
            base[q]   = at;
            stride[q] = plane_bytes[q];
            at += plane_bytes[q] * slots;
        }
    } else {
        std::int64_t at = 0;
        for (int q = 0; q < 4; ++q) {
            base[q]   = at;
            stride[q] = kSlotBytes;
            at += plane_bytes[q];
        }
    }
    const std::size_t total = static_cast<std::size_t>(kSlotBytes) * slots;
    std::vector<std::uint8_t> host(total);
    std::mt19937 rng(seed);
    for (std::size_t i = 0; i < total; i += 4) {
        const std::uint32_t r = rng();
        std::memcpy(host.data() + i, &r, std::min<std::size_t>(4, total - i));
    }
    for (std::size_t i = 0; i < placement.size(); ++i) {
        const Expert& x = m.experts[i];
        for (const int s : placement[i]) {
            std::uint8_t* p[4];
            for (int q = 0; q < 4; ++q) p[q] = host.data() + base[q] + stride[q] * s;
            std::memcpy(p[0], x.w13.data(), x.w13.size());
            std::memcpy(p[2], x.w2.data(), x.w2.size());
            for (int r = 0; r < kRows13; ++r)
                for (int g = 0; g < kHB; ++g)
                    p[1][swizzled_offset(r, g, kHB / 4)] =
                        x.s13[static_cast<std::size_t>(r) * kHB + g];
            for (int r = 0; r < kH; ++r)
                for (int g = 0; g < kIB; ++g)
                    p[3][swizzled_offset(r, g, kIB / 4)] =
                        x.s2[static_cast<std::size_t>(r) * kIB + g];
        }
    }
    DevicePool d;
    d.memory = DeviceBuffer(total);
    d.memory.copy_from_host(host.data(), total);
    auto* dev             = static_cast<std::uint8_t*>(d.memory.p);
    d.pool.gate_up_codes  = {dev + base[0], stride[0]};
    d.pool.gate_up_scales = {dev + base[1], stride[1]};
    d.pool.down_codes     = {dev + base[2], stride[2]};
    d.pool.down_scales    = {dev + base[3], stride[3]};
    d.pool.slots          = slots;
    return d;
}

std::vector<int> table_from(const Model& m, const std::vector<std::vector<int>>& placement,
                            int which = 0) {
    std::vector<int> table(kE, -1);
    for (std::size_t i = 0; i < placement.size(); ++i)
        if (static_cast<int>(placement[i].size()) > which) table[m.ids[i]] = placement[i][which];
    return table;
}

struct Call {
    int tokens = 0;
    std::vector<float> x;   // [T][H], BF16-representable
    std::vector<int> ids;   // [T][K]
    std::vector<float> w;   // [T][K]
    std::vector<int> table; // [E]
    float a1 = 0.F, a2 = 0.F;
};

// Each row: 10 distinct ids from the materialized set; weights = softmax of random logits.
Call make_call(const Model& m, int tokens, std::uint32_t seed) {
    std::mt19937 rng(seed);
    Call c;
    c.tokens = tokens;
    c.x.resize(static_cast<std::size_t>(tokens) * kH);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (auto& v : c.x) v = bf16_round(normal(rng));
    // an all-zero block and a tiny block (scale underflow) in token 0
    for (int i = 0; i < 16; ++i) c.x[32 + i] = 0.0F;
    for (int i = 0; i < 16; ++i) c.x[64 + i] = bf16_round(1.0e-6F * normal(rng));
    c.ids.resize(static_cast<std::size_t>(tokens) * kK);
    c.w.resize(c.ids.size());
    std::vector<int> pick = m.ids;
    for (int t = 0; t < tokens; ++t) {
        std::shuffle(pick.begin(), pick.end(), rng);
        float sum = 0.F;
        for (int k = 0; k < kK; ++k) {
            c.ids[t * kK + k] = pick[k];
            c.w[t * kK + k]   = std::exp(normal(rng));
            sum += c.w[t * kK + k];
        }
        for (int k = 0; k < kK; ++k) c.w[t * kK + k] /= sum;
    }
    return c;
}

bool resident(const Call& c, int id) { return id >= 0 && id < kE && c.table[id] >= 0; }

struct Reference {
    std::vector<double> exact, recipe; // [T][H]
    double hmax = 0.0;
};

// Calibrates the call's input divisors (x: 2688 / max|x|, h: 2688 / max|h| over the resident
// (expert, token) pairs, 2688 = 6 * 448, the F0-A data model) unless `calibrate` is false, then
// evaluates both oracles. The x divisor carries a non-dyadic factor kDivisorJitter: 2688 / max|x|
// with a BF16 max|x| makes v * divisor / scale exactly an E2M1 rounding midpoint for many BF16 v
// (e.g. 1.6015625 * 2688 / (3.84375 * 320) = 3.5), and the kernel resolves those exact ties
// through rcp.approx (the FlashInfer fast-math recipe) where the host uses exact division, so a
// tie can round to the other code. The factor moves every product off the midpoints; tie
// breaking itself is a property of the recipe that this oracle does not reproduce.
Reference reference(const Model& m, Call& c, bool calibrate) {
    const int T = c.tokens;
    // per resident expert: tokens and summed weights
    std::vector<int> experts;
    for (int i = 0; i < T * kK; ++i) {
        const int id = c.ids[i];
        if (resident(c, id) && m.index_of[id] >= 0 &&
            std::find(experts.begin(), experts.end(), id) == experts.end())
            experts.push_back(id);
    }
    if (calibrate) {
        float xmax = 0.F;
        for (const float v : c.x) xmax = std::max(xmax, std::fabs(v));
        c.a1 = 2688.0F / xmax * kDivisorJitter;
    }
    // x quantized once per token (the input divisor is per layer)
    std::vector<float> xc(c.x.size()), xsf(static_cast<std::size_t>(T) * kHB);
    for (int t = 0; t < T; ++t)
        for (int b = 0; b < kHB; ++b)
            quant16(&c.x[static_cast<std::size_t>(t) * kH + b * 16], c.a1,
                    &xc[static_cast<std::size_t>(t) * kH + b * 16], xsf[t * kHB + b]);

    const int n = static_cast<int>(experts.size());
    std::vector<std::vector<double>> h_exact(n), act(n); // [T][I] per expert
    std::vector<std::vector<float>> d2(n);
    parallel_for(n, [&](int j) {
        const int id     = experts[j];
        const Expert& ex = m.experts[m.index_of[id]];
        std::vector<float> d13;
        decode(ex.w13, ex.s13, kRows13, kH, d13);
        decode(ex.w2, ex.s2, kH, kI, d2[j]);
        h_exact[j].assign(static_cast<std::size_t>(T) * kI, 0.0);
        act[j].assign(static_cast<std::size_t>(T) * kI, 0.0);
        const double dg = m.div13[2 * id], du = m.div13[2 * id + 1];
        for (int t = 0; t < T; ++t) {
            bool used = false;
            for (int k = 0; k < kK; ++k) used |= c.ids[t * kK + k] == id;
            if (!used) continue;
            const float* xt  = &c.x[static_cast<std::size_t>(t) * kH];
            const float* xqt = &xc[static_cast<std::size_t>(t) * kH];
            for (int r = 0; r < kI; ++r) {
                double g = 0, u = 0, gq = 0, uq = 0;
                const float* wg = &d13[static_cast<std::size_t>(r) * kH];
                const float* wu = &d13[static_cast<std::size_t>(kI + r) * kH];
                for (int b = 0; b < kHB; ++b) {
                    double gb = 0, ub = 0;
                    for (int i = b * 16; i < b * 16 + 16; ++i) {
                        g += double(wg[i]) * xt[i];
                        u += double(wu[i]) * xt[i];
                        gb += double(wg[i]) * xqt[i];
                        ub += double(wu[i]) * xqt[i];
                    }
                    gq += gb * xsf[t * kHB + b];
                    uq += ub * xsf[t * kHB + b];
                }
                g /= dg;
                u /= du;
                h_exact[j][static_cast<std::size_t>(t) * kI + r] = g / (1.0 + std::exp(-g)) * u;
                const float gr = bf16_round(static_cast<float>(gq / (dg * c.a1)));
                const float ur = bf16_round(static_cast<float>(uq / (du * c.a1)));
                act[j][static_cast<std::size_t>(t) * kI + r] =
                    bf16_round(gr * (1.0F / (1.0F + std::exp(-gr))) * ur);
            }
        }
    });
    Reference ref;
    for (int j = 0; j < n; ++j)
        for (const double v : h_exact[j]) ref.hmax = std::max(ref.hmax, std::fabs(v));
    if (calibrate) c.a2 = static_cast<float>(2688.0 / ref.hmax);

    ref.exact.assign(static_cast<std::size_t>(T) * kH, 0.0);
    ref.recipe.assign(static_cast<std::size_t>(T) * kH, 0.0);
    std::vector<std::vector<double>> oe(n), orc(n); // per expert [T][H], weighted
    parallel_for(n, [&](int j) {
        const int id    = experts[j];
        const double dd = m.div2[id];
        oe[j].assign(static_cast<std::size_t>(T) * kH, 0.0);
        orc[j].assign(static_cast<std::size_t>(T) * kH, 0.0);
        for (int t = 0; t < T; ++t) {
            double w = 0.0;
            for (int k = 0; k < kK; ++k)
                if (c.ids[t * kK + k] == id) w += c.w[t * kK + k];
            if (w == 0.0) continue;
            std::vector<float> hc(kI), hsf(kIB);
            std::vector<float> a(kI);
            for (int r = 0; r < kI; ++r)
                a[r] = static_cast<float>(act[j][static_cast<std::size_t>(t) * kI + r]);
            for (int b = 0; b < kIB; ++b) quant16(&a[b * 16], c.a2, &hc[b * 16], hsf[b]);
            const double* he = &h_exact[j][static_cast<std::size_t>(t) * kI];
            for (int r = 0; r < kH; ++r) {
                const float* wd = &d2[j][static_cast<std::size_t>(r) * kI];
                double e = 0, q = 0;
                for (int b = 0; b < kIB; ++b) {
                    double qb = 0;
                    for (int i = b * 16; i < b * 16 + 16; ++i) {
                        e += double(wd[i]) * he[i];
                        qb += double(wd[i]) * hc[i];
                    }
                    q += qb * hsf[b];
                }
                oe[j][static_cast<std::size_t>(t) * kH + r]  = w * e / dd;
                orc[j][static_cast<std::size_t>(t) * kH + r] = w * q / (dd * c.a2);
            }
        }
    });
    for (int j = 0; j < n; ++j)
        for (std::size_t i = 0; i < ref.exact.size(); ++i) {
            ref.exact[i] += oe[j][i];
            ref.recipe[i] += orc[j][i];
        }
    return ref;
}

double relative_l2(const std::vector<double>& got, const std::vector<double>& ref) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        num += (got[i] - ref[i]) * (got[i] - ref[i]);
        den += ref[i] * ref[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

// ---- two-device partition
// -------------------------------------------------------------------------

constexpr int kRanks = 2;

std::vector<float> as_float(const std::vector<std::uint32_t>& words) {
    std::vector<float> out(words.size());
    std::memcpy(out.data(), words.data(), words.size() * 4);
    return out;
}

// Which of a rank's experts its cache holds; the others take the (emulated) CPU path. A fixed
// hash of the ID, ~70 % resident.
bool cache_resident(int id) { return ((static_cast<unsigned>(id) * 2654435761U) >> 7) % 10 < 7; }

int owner(int id) { return id % kRanks; }

// Slot placements of rank `rank`'s resident or missing experts (one slot each, compact).
std::vector<std::vector<int>> partition_placement(const Model& m, int rank, bool resident,
                                                  int& slots) {
    std::vector<std::vector<int>> p(m.ids.size());
    slots = 0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        const int id = m.ids[i];
        if (owner(id) == rank && cache_resident(id) == resident) { p[i] = {slots++}; }
    }
    return p;
}

// One rank's device state (or the single-device reference's, rank 0).
struct RankState {
    DeviceBuffer div13, div2, table_resident, table_missing, x, ids, w;
    std::unique_ptr<WorkspaceArena> workspace;
    DevicePool resident, missing;
    std::unique_ptr<GuardedDeviceBuffer> partial, staging, out;
    int resident_slots = 0, missing_slots = 0;

    ops::MoeSlotsLayer layer(const DeviceBuffer& table, const Call& c) const {
        ops::MoeSlotsLayer l;
        l.slot_table            = Tensor(table.p, DType::I32, {kE});
        l.gate_up_divisors      = Tensor(div13.p, DType::FP32, {2 * kE});
        l.down_divisors         = Tensor(div2.p, DType::FP32, {kE});
        l.gate_up_input_divisor = c.a1;
        l.down_input_divisor    = c.a2;
        return l;
    }
};

std::unique_ptr<RankState> make_rank(const Model& m, int rank, bool whole) {
    auto s       = std::make_unique<RankState>();
    s->div13     = to_device(m.div13);
    s->div2      = to_device(m.div2);
    s->workspace = std::make_unique<WorkspaceArena>(
        ops::moe_slots_nvfp4_workspace_bytes(ops::kMoeSlotsMaxTokens));
    std::vector<int> table_r, table_m;
    if (whole) {
        std::vector<std::vector<int>> all(m.ids.size());
        for (std::size_t i = 0; i < all.size(); ++i) { all[i] = {static_cast<int>(i)}; }
        s->resident_slots = static_cast<int>(m.ids.size());
        s->resident       = make_pool(m, all, s->resident_slots, Layout::Planar, 91U);
        table_r           = table_from(m, all);
    } else {
        const auto pr = partition_placement(m, rank, true, s->resident_slots);
        const auto pm = partition_placement(m, rank, false, s->missing_slots);
        // Contiguous slots in the cache pool (ninfer-ext's layout), planar for the CPU stand-in;
        // a few spare slots of garbage in each.
        s->resident = make_pool(m, pr, s->resident_slots + 3, Layout::Contiguous, 93U + rank);
        s->missing =
            make_pool(m, pm, std::max(s->missing_slots, 1) + 2, Layout::Planar, 95U + rank);
        table_r = table_from(m, pr);
        table_m = table_from(m, pm);
    }
    s->table_resident       = to_device_i32(table_r);
    s->table_missing        = to_device_i32(whole ? std::vector<int>(kE, -1) : table_m);
    const std::size_t count = static_cast<std::size_t>(ops::kMoeSlotsMaxTokens) * kH;
    s->partial              = std::make_unique<GuardedDeviceBuffer>(count * 4);
    s->staging              = std::make_unique<GuardedDeviceBuffer>(count * 4);
    s->out                  = std::make_unique<GuardedDeviceBuffer>(count * 2);
    s->staging->fill(0);
    s->x   = DeviceBuffer(count * 2);
    s->ids = DeviceBuffer(static_cast<std::size_t>(ops::kMoeSlotsMaxTokens) * kK * 4);
    s->w   = DeviceBuffer(static_cast<std::size_t>(ops::kMoeSlotsMaxTokens) * kK * 4);
    return s;
}

void upload_call(RankState& s, const Call& c) {
    std::vector<std::uint16_t> xb(c.x.size());
    for (std::size_t i = 0; i < xb.size(); ++i) { xb[i] = f32_to_bf16(c.x[i]); }
    s.x.copy_from_host(xb.data(), xb.size() * 2);
    s.ids.copy_from_host(c.ids.data(), c.ids.size() * 4);
    s.w.copy_from_host(c.w.data(), c.w.size() * 4);
    s.partial->fill(0xFF);
    s.out->fill(0xFF);
}

struct CallTensors {
    Tensor x, ids, w, partial, staging, out;
};

CallTensors tensors(RankState& s, int tokens) {
    return {Tensor(s.x.p, DType::BF16, {kH, tokens}),
            Tensor(s.ids.p, DType::I32, {kK, tokens}),
            Tensor(s.w.p, DType::FP32, {kK, tokens}),
            Tensor(s.partial->data(), DType::FP32, {kH, tokens}),
            Tensor(s.staging->data(), DType::FP32, {kH, tokens}),
            Tensor(s.out->data(), DType::BF16, {kH, tokens})};
}

// The rank's FP32 partial: its cache-resident experts (Store), then its missing experts from the
// CPU path (Accumulate). Issued on the rank's own stream, device current.
void rank_partial(RankState& s, const Call& c, CallTensors& t, cudaStream_t stream) {
    ops::moe_slots_nvfp4(t.x, t.ids, t.w, s.layer(s.table_resident, c), s.resident.pool,
                         ops::MoeSlotsOutput::Store, t.partial, *s.workspace, stream);
    if (s.missing_slots > 0) {
        ops::moe_slots_nvfp4(t.x, t.ids, t.w, s.layer(s.table_missing, c), s.missing.pool,
                             ops::MoeSlotsOutput::Accumulate, t.partial, *s.workspace, stream);
    }
}

struct Result {
    std::array<std::vector<std::uint32_t>, kRanks> sum;
    std::array<std::vector<std::uint16_t>, kRanks> bf16;
};

Result read_result(const ExecutionContext& ec, std::array<std::unique_ptr<RankState>, kRanks>& r,
                   int tokens, int& failures, const std::string& label) {
    Result out;
    const std::size_t count = static_cast<std::size_t>(tokens) * kH;
    for (int rank = 0; rank < kRanks; ++rank) {
        set_device(ec, rank);
        out.sum[rank]                = from_device<std::uint32_t>(r[rank]->partial->data(), count);
        out.bf16[rank]               = from_device<std::uint16_t>(r[rank]->out->data(), count);
        const std::string rank_label = label + " rank " + std::to_string(rank);
        failures += r[rank]->partial->verify_guards(rank_label + " partial");
        failures += r[rank]->staging->verify_guards(rank_label + " staging");
        failures += r[rank]->out->verify_guards(rank_label + " out");
    }
    return out;
}

int check(const std::string& label, const Result& tp2, const std::vector<std::uint32_t>& full_sum,
          const std::vector<std::uint16_t>& full_bf16, const Reference& ref, int tokens,
          bool expect_exact) {
    int failures = 0;
    failures += verify_exact((label + " ranks agree (fp32)").c_str(), tp2.sum[1], tp2.sum[0]);
    failures += verify_exact((label + " ranks agree (bf16)").c_str(), tp2.bf16[1], tp2.bf16[0]);
    const auto got  = as_float(tp2.sum[0]);
    const auto want = as_float(full_sum);
    std::vector<std::uint16_t> rounded(got.size());
    for (std::size_t i = 0; i < got.size(); ++i) { rounded[i] = f32_to_bf16(got[i]); }
    failures += verify_exact((label + " bf16 == RNE(fp32 sum)").c_str(), tp2.bf16[0], rounded);
    double worst  = 0.0;
    int ulp_diffs = 0;
    for (int t = 0; t < tokens; ++t) {
        double column_max = 0.0;
        for (int n = 0; n < kH; ++n) {
            column_max = std::max(column_max, std::fabs(double(want[t * kH + n])));
        }
        const double limit = std::ldexp(column_max, -15);
        for (int n = 0; n < kH; ++n) {
            const std::size_t i = static_cast<std::size_t>(t) * kH + n;
            const double error  = std::fabs(double(got[i]) - double(want[i]));
            worst               = std::max(worst, column_max > 0 ? error / column_max : error);
            if (error > limit) {
                if (failures < 8) {
                    std::cerr << label << ": fp32 sum differs at token " << t << " row " << n
                              << ": " << got[i] << " vs " << want[i] << '\n';
                }
                ++failures;
            }
            if (tp2.bf16[0][i] != full_bf16[i]) {
                ++ulp_diffs;
                const float a = bf16_to_f32(tp2.bf16[0][i]), b = bf16_to_f32(full_bf16[i]);
                const float ulp =
                    std::ldexp(1.0F, std::ilogb(std::max(std::fabs(a), std::fabs(b))) - 7);
                if (std::fabs(a - b) > ulp) {
                    std::cerr << label << ": bf16 differs by more than one ulp at " << i << '\n';
                    ++failures;
                }
            }
        }
    }
    std::vector<double> got_d(got.begin(), got.end());
    const double rl2 = relative_l2(got_d, ref.recipe);
    std::cout << label << ": max |tp2 - 1gpu| / column max = " << worst << ", bf16 elements "
              << "differing by one ulp " << ulp_diffs << " of " << got.size()
              << ", rel L2 vs recipe oracle " << rl2 << '\n';
    failures += verify_reduction(label + " recipe oracle", got_d, ref.recipe, kRecipe);
    if (expect_exact) {
        failures += verify_exact((label + " exact vs 1 GPU (fp32)").c_str(), tp2.sum[0], full_sum);
        failures +=
            verify_exact((label + " exact vs 1 GPU (bf16)").c_str(), tp2.bf16[0], full_bf16);
    }
    return failures;
}

// Single-device reference on rank 0's device: every expert resident, Store FP32 and BF16.
void reference_run(const ExecutionContext& ec, RankState& whole, const Call& c, int tokens,
                   std::vector<std::uint32_t>& sum, std::vector<std::uint16_t>& bf16) {
    set_device(ec, 0);
    upload_call(whole, c);
    auto t              = tensors(whole, tokens);
    cudaStream_t stream = ec.dev[0]->stream;
    ops::moe_slots_nvfp4(t.x, t.ids, t.w, whole.layer(whole.table_resident, c), whole.resident.pool,
                         ops::MoeSlotsOutput::Store, t.partial, *whole.workspace, stream);
    ops::moe_slots_nvfp4(t.x, t.ids, t.w, whole.layer(whole.table_resident, c), whole.resident.pool,
                         ops::MoeSlotsOutput::Store, t.out, *whole.workspace, stream);
    cuda_check(cudaStreamSynchronize(stream), "reference");
    const std::size_t count = static_cast<std::size_t>(tokens) * kH;
    sum                     = from_device<std::uint32_t>(whole.partial->data(), count);
    bf16                    = from_device<std::uint16_t>(whole.out->data(), count);
}

void rank_pass(const ExecutionContext& ec, std::array<std::unique_ptr<RankState>, kRanks>& r,
               const Call& c, int tokens, const ops::PeerEvents& events) {
    std::array<CallTensors, kRanks> t;
    for (int rank = 0; rank < kRanks; ++rank) {
        set_device(ec, rank);
        t[rank] = tensors(*r[rank], tokens);
        rank_partial(*r[rank], c, t[rank], ec.dev[rank]->stream);
    }
    const std::array<Tensor, 2> partial{t[0].partial, t[1].partial};
    const std::array<Tensor, 2> staging{t[0].staging, t[1].staging};
    const std::array<Tensor, 2> out{t[0].out, t[1].out};
    ops::allreduce_sum_f32(partial, staging, &out, ec, events);
}

// Calls with every routed expert on rank 0 and resident there: identical to one GPU.
Call one_rank_call(const Model& m, int tokens, std::uint32_t seed) {
    Call c = make_call(m, tokens, seed);
    std::vector<int> local;
    for (const int id : m.ids) {
        if (owner(id) == 0 && cache_resident(id)) { local.push_back(id); }
    }
    std::mt19937 rng(seed);
    for (int t = 0; t < tokens; ++t) {
        std::shuffle(local.begin(), local.end(), rng);
        for (int k = 0; k < kK; ++k) { c.ids[t * kK + k] = local[k]; }
    }
    return c;
}

int run(const ExecutionContext& ec) {
    const Model m = make_model(64, 23U);
    set_device(ec, 0);
    auto whole = make_rank(m, 0, true);
    std::array<std::unique_ptr<RankState>, kRanks> ranks;
    for (int rank = 0; rank < kRanks; ++rank) {
        set_device(ec, rank);
        ranks[rank] = make_rank(m, rank, false);
        std::cout << "rank " << rank << ": " << ranks[rank]->resident_slots
                  << " cache-resident experts, " << ranks[rank]->missing_slots
                  << " on the CPU path\n";
    }
    retire_staging(ec);
    const ops::PeerEvents staged(ec);
    // The Program's slot for this window: [2560, T] FP32 (40 KiB at T = 4, MTP-3).
    ops::PeerMailbox mailbox(ec, static_cast<std::size_t>(kH) * ops::kMoeSlotsMaxTokens * 4);
    ops::PeerEvents mailbox_events(ec);
    mailbox_events.attach_mailbox(&mailbox);

    int failures = 0;

    struct Case {
        int tokens;
        std::uint32_t seed;
        bool one_rank;
    };

    for (const Case& k : {Case{1, 31U, false}, Case{2, 33U, false}, Case{4, 35U, false},
                          Case{8, 37U, false}, Case{4, 39U, true}, Case{1, 41U, true}}) {
        Call c  = k.one_rank ? one_rank_call(m, k.tokens, k.seed) : make_call(m, k.tokens, k.seed);
        c.table = table_from(m, [&] {
            std::vector<std::vector<int>> all(m.ids.size());
            for (std::size_t i = 0; i < all.size(); ++i) { all[i] = {static_cast<int>(i)}; }
            return all;
        }());
        const Reference ref = reference(m, c, true);
        std::vector<std::uint32_t> full_sum;
        std::vector<std::uint16_t> full_bf16;
        reference_run(ec, *whole, c, k.tokens, full_sum, full_bf16);
        const std::string label = std::string("tp2 moe T=") + std::to_string(k.tokens) +
                                  (k.one_rank ? " (one rank)" : "");

        // Eager, staged all-reduce.
        for (int rank = 0; rank < kRanks; ++rank) {
            set_device(ec, rank);
            upload_call(*ranks[rank], c);
        }
        retire_staging(ec);
        rank_pass(ec, ranks, c, k.tokens, staged);
        synchronize_both(ec);
        const auto eager = read_result(ec, ranks, k.tokens, failures, label + " staged");
        failures += check(label + " staged", eager, full_sum, full_bf16, ref, k.tokens, k.one_rank);

        // Both ranks' MoE and the all-reduce captured into one graph on the mailbox, replayed
        // twice: bit-identical to the eager result.
        for (int rank = 0; rank < kRanks; ++rank) {
            set_device(ec, rank);
            upload_call(*ranks[rank], c);
        }
        retire_staging(ec);
        const DecodeGraphPeerBridge bridge(ec.dev[0]->device, ec.dev[1]->device);
        DecodeGraphDefinition definition;
        set_device(ec, 0);
        definition.capture(
            ec.dev[0]->stream,
            [&] {
                rank_pass(ec, ranks, c, k.tokens, mailbox_events);
                set_device(ec, 0);
            },
            DecodeGraphPeerCapture{.bridge = &bridge, .stream = ec.dev[1]->stream});
        DecodeGraphExecutable executable;
        executable.instantiate(definition);
        for (int replay = 0; replay < 2; ++replay) {
            set_device(ec, 0);
            executable.launch(ec.dev[0]->stream);
            synchronize_both(ec);
            const auto graph = read_result(ec, ranks, k.tokens, failures,
                                           label + " graph replay " + std::to_string(replay));
            for (int rank = 0; rank < kRanks; ++rank) {
                failures += verify_exact((label + " graph == eager (fp32)").c_str(),
                                         graph.sum[rank], eager.sum[rank]);
                failures += verify_exact((label + " graph == eager (bf16)").c_str(),
                                         graph.bf16[rank], eager.bf16[rank]);
            }
        }
        if (mailbox.hang_reported()) {
            std::cerr << label << ": mailbox hang\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) {
            std::cout << "SKIP: no CUDA device\n";
            return 77;
        }
        int devices = 0;
        cuda_check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
        if (devices < 2) {
            std::cout << "SKIP: the two-device MoE needs two CUDA devices\n";
            return 77;
        }
        const ExecutionContext ec({0, 1});
        (void)ops::enable_peer_access(ec);
        const int failures = run(ec);
        std::cout << (failures ? "FAIL" : "OK") << " flash_next_tp2_moe\n";
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "flash_next_tp2_moe: " << e.what() << '\n';
        return 1;
    }
}
