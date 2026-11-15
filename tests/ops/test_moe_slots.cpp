// moe_slots_nvfp4 against FP64 oracles at the Qwen3.8-Flash-Next geometry (hidden 2560, expert
// intermediate 640, top-10, T = 1..8), with experts read from cache slots through a slot table.
//
// Oracles (both evaluated in FP64 from the represented inputs: BF16 x, exact NVFP4 decode of
// codes, E4M3FN block scales and FP32 divisors):
//   * mathematical: unquantized activations (the F0-A metric "W4A4 vs fp32");
//   * recipe: the same formula with the documented W4A4 activation profile applied (x and h
//     quantized to E2M1 with E4M3FN scales per 16 values against the input divisors, gate/up and
//     h rounded to BF16). It isolates indexing/scaling faults that the loose W4A4 bound cannot
//     see (a swapped gate/up divisor or a wrong slot moves it by orders of magnitude).
// Exact (bitwise) checks: the output does not depend on which slot holds an expert or on the
// pool layout; skipped positions equal zero-weight positions; a repeated expert equals one
// position with the summed weight; Accumulate = old + Store; BF16 = RNE(FP32 Store); a call with
// no resident expert writes zeros (Store) or keeps `out` (Accumulate); a captured CUDA graph
// replays the eager result after the slot table, x and routing change.
#include "ninfer/ops/moe_slots.h"

#include "core/arena.h"
#include "ops/nvfp4_test_codec.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
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

class Runner {
public:
    explicit Runner(const Model& m)
        : workspace_(ops::moe_slots_nvfp4_workspace_bytes(ops::kMoeSlotsMaxTokens)),
          div13_(to_device(m.div13)), div2_(to_device(m.div2)), table_(kE * sizeof(int)) {
        cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream");
    }

    ~Runner() { cudaStreamDestroy(stream_); }

    cudaStream_t stream() const { return stream_; }

    WorkspaceArena& workspace() { return workspace_; }

    ops::MoeSlotsLayer layer(const Call& c, void* table_device) const {
        ops::MoeSlotsLayer l;
        l.slot_table            = Tensor(table_device, DType::I32, {kE});
        l.gate_up_divisors      = Tensor(div13_.p, DType::FP32, {2 * kE});
        l.down_divisors         = Tensor(div2_.p, DType::FP32, {kE});
        l.gate_up_input_divisor = c.a1;
        l.down_input_divisor    = c.a2;
        return l;
    }

    // Returns the raw output words (FP32 bits or BF16 bits widened); checks the guards.
    std::vector<std::uint32_t> run(const Call& c, const ops::MoeSlotPool& pool,
                                   ops::MoeSlotsOutput mode, DType dtype,
                                   const std::vector<std::uint32_t>* init, int& failures,
                                   const std::string& label) {
        const int T = c.tokens;
        table_.copy_from_host(c.table.data(), c.table.size() * sizeof(int));
        std::vector<std::uint16_t> xb(c.x.size());
        for (std::size_t i = 0; i < xb.size(); ++i) xb[i] = f32_to_bf16(c.x[i]);
        DeviceBuffer dx = to_device(xb), di = to_device(c.ids), dw = to_device(c.w);
        const std::size_t elem  = dtype == DType::FP32 ? 4 : 2;
        const std::size_t count = static_cast<std::size_t>(T) * kH;
        GuardedDeviceBuffer out(count * elem);
        if (init != nullptr) {
            if (dtype == DType::FP32)
                out.copy_from_host(init->data(), count * 4);
            else {
                std::vector<std::uint16_t> h(count);
                for (std::size_t i = 0; i < count; ++i)
                    h[i] = static_cast<std::uint16_t>((*init)[i]);
                out.copy_from_host(h.data(), count * 2);
            }
        } else {
            out.fill(0xFF); // NaN: every element must be written
        }
        cuda_synchronize(); // memsets/copies on the legacy stream before the non-blocking stream
        Tensor tx(dx.p, DType::BF16, {kH, T}), ti(di.p, DType::I32, {kK, T}),
            tw(dw.p, DType::FP32, {kK, T}), to(out.data(), dtype, {kH, T});
        ops::moe_slots_nvfp4(tx, ti, tw, layer(c, table_.p), pool, mode, to, workspace_, stream_);
        cuda_check_last_launch("moe_slots_nvfp4");
        cuda_synchronize(stream_);
        failures += out.verify_guards(label);
        std::vector<std::uint32_t> words(count);
        if (dtype == DType::FP32)
            out.copy_to_host(words.data(), count * 4);
        else {
            std::vector<std::uint16_t> h(count);
            out.copy_to_host(h.data(), count * 2);
            for (std::size_t i = 0; i < count; ++i) words[i] = h[i];
        }
        return words;
    }

    std::vector<std::uint32_t> run(const Call& c, const ops::MoeSlotPool& pool, int& failures,
                                   const std::string& label) {
        return run(c, pool, ops::MoeSlotsOutput::Store, DType::FP32, nullptr, failures, label);
    }

private:
    cudaStream_t stream_ = nullptr;
    WorkspaceArena workspace_;
    DeviceBuffer div13_, div2_, table_;
};

std::vector<double> as_double(const std::vector<std::uint32_t>& words) {
    std::vector<double> out(words.size());
    for (std::size_t i = 0; i < words.size(); ++i) {
        float f;
        std::memcpy(&f, &words[i], 4);
        out[i] = f;
    }
    return out;
}

std::vector<float> as_float(const std::vector<std::uint32_t>& words) {
    std::vector<float> out(words.size());
    std::memcpy(out.data(), words.data(), words.size() * 4);
    return out;
}

int check_oracles(const std::string& label, const std::vector<std::uint32_t>& got_words,
                  const Reference& ref) {
    const auto got = as_double(got_words);
    std::cout << label << ": rel_l2 vs FP64 (unquantized activations) "
              << relative_l2(got, ref.exact) << ", vs recipe " << relative_l2(got, ref.recipe)
              << '\n';
    return verify_reduction(label + " W4A4 vs FP64", got, ref.exact, kW4A4) +
           verify_reduction(label + " recipe", got, ref.recipe, kRecipe);
}

std::vector<std::vector<int>> identity_placement(const Model& m) {
    std::vector<std::vector<int>> p(m.ids.size());
    for (std::size_t i = 0; i < p.size(); ++i) p[i] = {static_cast<int>(i)};
    return p;
}

std::vector<std::vector<int>> random_placement(const Model& m, int slots, std::uint32_t seed,
                                               int copies = 1) {
    std::vector<int> s(slots);
    std::iota(s.begin(), s.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(s.begin(), s.end(), rng);
    std::vector<std::vector<int>> p(m.ids.size());
    for (std::size_t i = 0; i < p.size(); ++i)
        for (int c = 0; c < copies; ++c) p[i].push_back(s[i * copies + c]);
    return p;
}

// (1) Oracle at every T with all experts resident, planar pool, identity slots.
int oracle_cases(const Model& m, Runner& runner, const DevicePool& pool) {
    int failures = 0;
    for (const int T : {1, 2, 3, 4, 8}) {
        Call c                  = make_call(m, T, 100U + T);
        c.table                 = table_from(m, identity_placement(m));
        const Reference ref     = reference(m, c, true);
        const std::string label = "oracle T=" + std::to_string(T);
        failures += check_oracles(label, runner.run(c, pool.pool, failures, label), ref);
    }
    return failures;
}

// (2) Slot indirection: the same call through random placements in larger pools of both layouts
// (garbage in every other slot) is bit-identical to the identity placement.
int permutation_cases(const Model& m, Runner& runner, const DevicePool& identity) {
    int failures = 0;
    for (const int T : {3, 8}) {
        Call c  = make_call(m, T, 200U + T);
        c.table = table_from(m, identity_placement(m));
        reference(m, c, true); // calibration only
        const auto base = runner.run(c, identity.pool, failures, "perm base");
        for (int p = 0; p < 4; ++p) {
            const int slots         = 40 + 7 * p;
            const auto place        = random_placement(m, slots, 300U + 10U * p + T);
            const Layout lay        = p % 2 == 0 ? Layout::Contiguous : Layout::Planar;
            const DevicePool d      = make_pool(m, place, slots, lay, 400U + p);
            Call cp                 = c;
            cp.table                = table_from(m, place);
            const std::string label = "permutation T=" + std::to_string(T) + " #" +
                                      std::to_string(p) +
                                      (lay == Layout::Contiguous ? " contiguous" : " planar");
            failures += verify_exact(label.c_str(), runner.run(cp, d.pool, failures, label), base);
        }
    }
    return failures;
}

// (3) Missing experts: non-resident ids, -1 and out-of-range ids are skipped; the result meets
// both oracles on the resident experts and equals (as FP32 values) the call where those positions
// are resident with weight 0.
int missing_cases(const Model& m, Runner& runner, const DevicePool& pool) {
    int failures = 0;
    for (const int T : {1, 4, 8}) {
        Call c = make_call(m, T, 500U + T);
        std::mt19937 rng(600U + T);
        auto place = identity_placement(m);
        // ~30 % of the materialized experts not resident on this GPU
        for (std::size_t i = 0; i < place.size(); ++i)
            if (rng() % 10 < 3) place[i].clear();
        c.table = table_from(m, place);
        // the first 3 positions of every token: invalid ids (-1, out of range)
        const int invalid[4] = {-1, kE, 7777, -5};
        std::vector<int> zero_positions;
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < 3; ++k) c.ids[t * kK + k] = invalid[(t + k) % 4];
        for (int i = 0; i < T * kK; ++i)
            if (!resident(c, c.ids[i])) zero_positions.push_back(i);
        const Reference ref     = reference(m, c, true);
        const std::string label = "missing T=" + std::to_string(T);
        const auto got          = runner.run(c, pool.pool, failures, label);
        failures += check_oracles(label, got, ref);

        Call z  = c;
        z.table = table_from(m, identity_placement(m));
        for (const int i : zero_positions)
            if (z.ids[i] >= 0 && z.ids[i] < kE) z.w[i] = 0.0F;
        failures += verify_exact((label + " == zero-weight residency").c_str(),
                                 as_float(runner.run(z, pool.pool, failures, label + " zw")),
                                 as_float(got));
    }
    return failures;
}

// (4) A repeated expert in one token row counts with each weight: equals one position with the
// summed weight (summed in top-k order, as the Op does), and meets the oracles.
int duplicate_case(const Model& m, Runner& runner, const DevicePool& pool) {
    int failures        = 0;
    Call c              = make_call(m, 4, 700U);
    c.table             = table_from(m, identity_placement(m));
    c.ids[1 * kK + 6]   = c.ids[1 * kK + 1]; // token 1: positions 1 and 6
    c.ids[3 * kK + 9]   = c.ids[3 * kK + 0]; // token 3: positions 0 and 9
    const Reference ref = reference(m, c, true);
    const auto got      = runner.run(c, pool.pool, failures, "duplicate");
    failures += check_oracles("duplicate", got, ref);
    Call merged            = c;
    merged.w[1 * kK + 1]   = (0.0F + c.w[1 * kK + 1]) + c.w[1 * kK + 6];
    merged.ids[1 * kK + 6] = -1;
    merged.w[3 * kK + 0]   = (0.0F + c.w[3 * kK + 0]) + c.w[3 * kK + 9];
    merged.ids[3 * kK + 9] = -1;
    failures +=
        verify_exact("duplicate == merged weight",
                     as_float(runner.run(merged, pool.pool, failures, "merged")), as_float(got));
    return failures;
}

// (5) Output modes and the empty call.
int mode_cases(const Model& m, Runner& runner, const DevicePool& pool) {
    int failures = 0;
    Call c       = make_call(m, 5, 800U);
    auto place   = identity_placement(m);
    for (std::size_t i = 0; i < place.size(); i += 3) place[i].clear();
    c.table = table_from(m, place);
    reference(m, c, true);
    const auto store = runner.run(c, pool.pool, failures, "store");
    // Accumulate onto random FP32 values = old + Store, exactly
    std::vector<std::uint32_t> init(store.size());
    std::mt19937 rng(801U);
    std::normal_distribution<float> normal(0.F, 1.F);
    std::vector<float> initf(store.size());
    for (std::size_t i = 0; i < init.size(); ++i) {
        initf[i] = normal(rng);
        std::memcpy(&init[i], &initf[i], 4);
    }
    const auto acc = runner.run(c, pool.pool, ops::MoeSlotsOutput::Accumulate, DType::FP32, &init,
                                failures, "accumulate");
    std::vector<float> expect(store.size());
    const auto sf = as_float(store);
    for (std::size_t i = 0; i < expect.size(); ++i) expect[i] = initf[i] + sf[i];
    failures += verify_exact("accumulate == old + store", as_float(acc), expect);
    // BF16 = RNE(FP32 Store)
    const auto b16 = runner.run(c, pool.pool, ops::MoeSlotsOutput::Store, DType::BF16, nullptr,
                                failures, "bf16");
    std::vector<std::uint32_t> expect16(store.size());
    for (std::size_t i = 0; i < expect16.size(); ++i) expect16[i] = f32_to_bf16(sf[i]);
    failures += verify_exact("bf16 == rne(store)", b16, expect16);

    // no resident expert at all
    Call empty = c;
    empty.table.assign(kE, -1);
    const std::vector<std::uint32_t> zeros(store.size(), 0U);
    failures +=
        verify_exact("empty store == 0", as_float(runner.run(empty, pool.pool, failures, "empty")),
                     std::vector<float>(store.size(), 0.0F));
    failures += verify_exact("empty accumulate keeps out",
                             runner.run(empty, pool.pool, ops::MoeSlotsOutput::Accumulate,
                                        DType::FP32, &init, failures, "empty acc"),
                             init);
    failures += verify_exact("empty bf16 == 0",
                             runner.run(empty, pool.pool, ops::MoeSlotsOutput::Store, DType::BF16,
                                        nullptr, failures, "empty bf16"),
                             zeros);
    // a token without resident experts gets 0 while the others are computed
    Call partial = c;
    for (int k = 0; k < kK; ++k) partial.ids[2 * kK + k] = -1;
    const auto pr = as_float(runner.run(partial, pool.pool, failures, "token without experts"));
    failures += verify_exact("token without experts == 0",
                             std::vector<float>(pr.begin() + 2 * kH, pr.begin() + 3 * kH),
                             std::vector<float>(kH, 0.0F));
    return failures;
}

// (6) CUDA graph: three layer calls captured once (shared workspace, PDL chain); replays after the
// slot tables, x and routing change reproduce eager calls bit for bit.
int graph_case(const Model& m, Runner& runner) {
    int failures              = 0;
    const int slots           = 2 * static_cast<int>(m.ids.size()) + 5;
    const auto place          = random_placement(m, slots, 900U, 2); // every expert in two slots
    const DevicePool d        = make_pool(m, place, slots, Layout::Contiguous, 901U);
    constexpr int kLayers     = 3;
    const int tokens[kLayers] = {4, 1, 8};
    std::vector<Call> v1(kLayers), v2(kLayers);
    for (int l = 0; l < kLayers; ++l) {
        v1[l]       = make_call(m, tokens[l], 910U + l);
        v1[l].table = table_from(m, place, 0);
        reference(m, v1[l], true);
        v2[l]       = make_call(m, tokens[l], 920U + l);
        v2[l].table = table_from(m, place, 1); // other copy of every expert
        for (int e = 0; e < kE; e += 4) v2[l].table[e] = -1;
        v2[l].a1 = v1[l].a1; // input divisors are launch parameters fixed at capture
        v2[l].a2 = v1[l].a2;
    }
    std::vector<std::vector<std::uint32_t>> eager1(kLayers), eager2(kLayers);
    for (int l = 0; l < kLayers; ++l) {
        eager1[l] = runner.run(v1[l], d.pool, failures, "graph eager v1");
        eager2[l] = runner.run(v2[l], d.pool, failures, "graph eager v2");
    }
    // device inputs per layer, overwritten between replays
    std::vector<DeviceBuffer> dx, di, dw, dt, dout;
    for (int l = 0; l < kLayers; ++l) {
        dx.emplace_back(static_cast<std::size_t>(tokens[l]) * kH * 2);
        di.emplace_back(static_cast<std::size_t>(tokens[l]) * kK * 4);
        dw.emplace_back(static_cast<std::size_t>(tokens[l]) * kK * 4);
        dt.emplace_back(kE * 4);
        dout.emplace_back(static_cast<std::size_t>(tokens[l]) * kH * 4);
    }
    auto upload = [&](const std::vector<Call>& calls) {
        for (int l = 0; l < kLayers; ++l) {
            std::vector<std::uint16_t> xb(calls[l].x.size());
            for (std::size_t i = 0; i < xb.size(); ++i) xb[i] = f32_to_bf16(calls[l].x[i]);
            dx[l].copy_from_host(xb.data(), xb.size() * 2);
            di[l].copy_from_host(calls[l].ids.data(), calls[l].ids.size() * 4);
            dw[l].copy_from_host(calls[l].w.data(), calls[l].w.size() * 4);
            dt[l].copy_from_host(calls[l].table.data(), kE * 4);
            dout[l].fill(0xFF);
        }
        cuda_synchronize();
    };
    auto download = [&](int l) {
        std::vector<std::uint32_t> words(static_cast<std::size_t>(tokens[l]) * kH);
        dout[l].copy_to_host(words.data(), words.size() * 4);
        return words;
    };
    upload(v1);
    cudaStream_t s       = runner.stream();
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
    cuda_check(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "begin capture");
    for (int l = 0; l < kLayers; ++l) {
        Tensor tx(dx[l].p, DType::BF16, {kH, tokens[l]}), ti(di[l].p, DType::I32, {kK, tokens[l]}),
            tw(dw[l].p, DType::FP32, {kK, tokens[l]}), to(dout[l].p, DType::FP32, {kH, tokens[l]});
        ops::moe_slots_nvfp4(tx, ti, tw, runner.layer(v1[l], dt[l].p), d.pool,
                             ops::MoeSlotsOutput::Store, to, runner.workspace(), s);
    }
    cuda_check(cudaStreamEndCapture(s, &graph), "end capture");
    cuda_check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    for (int round = 0; round < 2; ++round) {
        upload(v1);
        cuda_check(cudaGraphLaunch(exec, s), "launch");
        cuda_synchronize(s);
        for (int l = 0; l < kLayers; ++l)
            failures += verify_exact(("graph v1 layer " + std::to_string(l)).c_str(), download(l),
                                     eager1[l]);
        upload(v2);
        cuda_check(cudaGraphLaunch(exec, s), "launch");
        cuda_synchronize(s);
        for (int l = 0; l < kLayers; ++l)
            failures += verify_exact(("graph v2 layer " + std::to_string(l)).c_str(), download(l),
                                     eager2[l]);
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    for (int l = 0; l < kLayers; ++l)
        if (eager1[l] == eager2[l]) {
            std::cerr << "graph: v1 and v2 outputs are identical (the replay check is vacuous)\n";
            ++failures;
        }
    return failures;
}

int validation_cases(const Model& m, Runner& runner, const DevicePool& pool) {
    int failures = 0;
    Call c       = make_call(m, 2, 1000U);
    c.table      = table_from(m, identity_placement(m));
    reference(m, c, true);
    DeviceBuffer dx(9 * kH * 2), di(9 * kK * 4), dw(9 * kK * 4), dt(kE * 4), dout(9 * kH * 4);
    dt.copy_from_host(c.table.data(), kE * 4);
    auto expect_throw = [&](const char* label, auto&& fn) {
        try {
            fn();
        } catch (const std::invalid_argument&) { return; }
        std::cerr << label << ": expected std::invalid_argument\n";
        ++failures;
    };
    const auto layer = runner.layer(c, dt.p);
    Tensor tx(dx.p, DType::BF16, {kH, 2}), ti(di.p, DType::I32, {kK, 2}),
        tw(dw.p, DType::FP32, {kK, 2});
    expect_throw("T=9", [&] {
        Tensor x9(dx.p, DType::BF16, {kH, 9}), i9(di.p, DType::I32, {kK, 9}),
            w9(dw.p, DType::FP32, {kK, 9}), o9(dout.p, DType::FP32, {kH, 9});
        ops::moe_slots_nvfp4(x9, i9, w9, layer, pool.pool, ops::MoeSlotsOutput::Store, o9,
                             runner.workspace(), runner.stream());
    });
    expect_throw("accumulate into BF16", [&] {
        Tensor o(dout.p, DType::BF16, {kH, 2});
        ops::moe_slots_nvfp4(tx, ti, tw, layer, pool.pool, ops::MoeSlotsOutput::Accumulate, o,
                             runner.workspace(), runner.stream());
    });
    expect_throw("misaligned code stride", [&] {
        auto p = pool.pool;
        p.gate_up_codes.stride += 4;
        Tensor o(dout.p, DType::FP32, {kH, 2});
        ops::moe_slots_nvfp4(tx, ti, tw, layer, p, ops::MoeSlotsOutput::Store, o,
                             runner.workspace(), runner.stream());
    });
    expect_throw("workspace too small", [&] {
        WorkspaceArena small(ops::moe_slots_nvfp4_workspace_bytes(2) - 256);
        Tensor o(dout.p, DType::FP32, {kH, 2});
        ops::moe_slots_nvfp4(tx, ti, tw, layer, pool.pool, ops::MoeSlotsOutput::Store, o, small,
                             runner.stream());
    });
    return failures;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) {
            std::cout << "SKIP: no CUDA device\n";
            return 77;
        }
        const Model m = make_model(24, 11U);
        Runner runner(m);
        const DevicePool identity = make_pool(m, identity_placement(m),
                                              static_cast<int>(m.ids.size()), Layout::Planar, 12U);
        int failures              = 0;
        failures += oracle_cases(m, runner, identity);
        failures += permutation_cases(m, runner, identity);
        failures += missing_cases(m, runner, identity);
        failures += duplicate_case(m, runner, identity);
        failures += mode_cases(m, runner, identity);
        failures += graph_case(m, runner);
        failures += validation_cases(m, runner, identity);
        if (failures != 0) {
            std::cerr << "moe_slots_nvfp4: " << failures << " failure(s)\n";
            return 1;
        }
        std::cout << "moe_slots_nvfp4: all checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "moe_slots_nvfp4 test: " << e.what() << '\n';
        return 1;
    }
}
