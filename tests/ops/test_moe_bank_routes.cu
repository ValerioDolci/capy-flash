// moe_bank_capped_swiglu (narrow route, up to 24 tokens) and moe_sigmoid_route against their
// reference executions: the first,
// one-warp-per-row implementation of the contract, kept here (not in the library) as the bit
// reference of the production routes. Every route of the Op must give the same partial bit for bit,
// Store and Accumulate, at the synthetic and the real geometry, nvfp4 and bf16 banks, decode and
// prefill widths. The mathematical acceptance is test_moe_bank.cpp (FP64 oracle); this suite pins
// the arithmetic so a faster route cannot drift silently.
#include "core/arena.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "ninfer/ops/moe_bank.h"
#include "ninfer/ops/moe_sigmoid_route.h"
#include "ops/op_tester.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace reference {

constexpr int kWarps   = 8;
constexpr int kThreads = kWarps * 32;

struct Plane {
    const std::uint8_t* data      = nullptr;
    bool nvfp4                    = false;
    std::int64_t row_begin        = 0;
    std::int32_t columns          = 0;
    std::uint64_t scale_offset    = 0;
    std::uint64_t divisor_offset  = 0;
    std::int64_t rows_per_divisor = 0;
};

struct Shape {
    std::int32_t hidden, intermediate, experts, first_expert, top_k, tokens;
};

__constant__ float kE2M1[16] = {0.0F,  0.5F,  1.0F,  1.5F,  2.0F,  3.0F,  4.0F,  6.0F,
                                -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F};

__device__ __forceinline__ float e4m3fn(std::uint8_t c) {
    const int e = (c >> 3) & 15, m = c & 7;
    const float v = e == 0 ? static_cast<float>(m) * 0x1p-9F
                           : ldexpf(1.0F + static_cast<float>(m) * 0.125F, e - 7);
    return (c & 0x80) ? -v : v;
}

__device__ __forceinline__ float bf16_round(float v) {
    return __bfloat162float(__float2bfloat16_rn(v));
}

__device__ __forceinline__ float warp_sum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) x += __shfl_xor_sync(0xffffffffU, x, offset);
    return x;
}

__device__ float row_dot(const Plane& plane, std::int64_t row, const __nv_bfloat16* v, int lane) {
    const int columns = plane.columns;
    float acc         = 0.0F;
    if (plane.nvfp4) {
        const auto* codes          = plane.data + row * (columns / 2);
        const auto* scales         = plane.data + plane.scale_offset;
        const std::int64_t k_tiles = columns / 64;
        const std::int64_t inner   = row % 128;
        const std::int64_t tile_base =
            (row / 128) * k_tiles * 512 + (inner % 32) * 16 + (inner / 32) * 4;
        for (int g = lane; g < columns / 16; g += 32) {
            const uint2 packed       = *reinterpret_cast<const uint2*>(codes + g * 8);
            const std::uint8_t scale = scales[tile_base + (g / 4) * 512 + g % 4];
            const __nv_bfloat16* vg  = v + g * 16;
            float block              = 0.0F;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const std::uint32_t word = c < 4 ? packed.x : packed.y;
                const std::uint32_t byte = (word >> ((c % 4) * 8)) & 0xFFU;
                block = fmaf(kE2M1[byte & 0xFU], __bfloat162float(vg[2 * c]), block);
                block = fmaf(kE2M1[byte >> 4], __bfloat162float(vg[2 * c + 1]), block);
            }
            acc = fmaf(block, e4m3fn(scale), acc);
        }
    } else {
        const auto* w = reinterpret_cast<const __nv_bfloat16*>(plane.data) + row * columns;
        for (int c = lane * 8; c < columns; c += 32 * 8) {
            const uint4 wq = *reinterpret_cast<const uint4*>(w + c);
            const uint4 vq = *reinterpret_cast<const uint4*>(v + c);
            const auto* wp = reinterpret_cast<const __nv_bfloat162*>(&wq);
            const auto* vp = reinterpret_cast<const __nv_bfloat162*>(&vq);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 a = __bfloat1622float2(wp[i]);
                const float2 b = __bfloat1622float2(vp[i]);
                acc            = fmaf(a.x, b.x, fmaf(a.y, b.y, acc));
            }
        }
    }
    return acc;
}

__device__ float full_dot(const Plane& plane, std::int64_t row, const __nv_bfloat16* v, int lane) {
    const float dot = warp_sum(row_dot(plane, row, v, lane));
    if (!plane.nvfp4) return dot;
    const auto* words = reinterpret_cast<const float*>(plane.data + plane.divisor_offset);
    return dot / words[row / plane.rows_per_divisor];
}

__device__ __forceinline__ float softcap(float v, float cap) { return cap * tanhf(v / cap); }

__global__ void gate_up_kernel(const __nv_bfloat16* x, const std::int32_t* ids, Plane gate_up,
                               Shape shape, float gate_cap, float up_cap, __nv_bfloat16* hidden) {
    const int assignment = static_cast<int>(blockIdx.x);
    const int b          = ids[assignment] - shape.first_expert;
    if (ids[assignment] < 0 || b < 0 || b >= shape.experts) return;
    const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
    const int channel = static_cast<int>(blockIdx.y) * kWarps + warp;
    if (channel >= shape.intermediate) return;
    const int token        = assignment / shape.top_k;
    const __nv_bfloat16* u = x + static_cast<std::int64_t>(token) * shape.hidden;
    const std::int64_t base =
        gate_up.row_begin + static_cast<std::int64_t>(b) * 2 * shape.intermediate;
    const float g = bf16_round(full_dot(gate_up, base + channel, u, lane));
    const float a = bf16_round(full_dot(gate_up, base + shape.intermediate + channel, u, lane));
    if (lane == 0) {
        const float sig = 1.0F / (1.0F + expf(-g));
        const float h   = softcap(g, gate_cap) * sig * softcap(a, up_cap);
        hidden[static_cast<std::int64_t>(assignment) * shape.intermediate + channel] =
            __float2bfloat16_rn(h);
    }
}

__global__ void down_kernel(const std::int32_t* ids, const float* weights,
                            const __nv_bfloat16* hidden, Plane down, Shape shape, bool accumulate,
                            float* partial) {
    const int token = static_cast<int>(blockIdx.x);
    const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
    const int row = static_cast<int>(blockIdx.y) * kWarps + warp;
    if (row >= shape.hidden) return;
    float sum = 0.0F;
    for (int j = 0; j < shape.top_k; ++j) {
        const int assignment = token * shape.top_k + j;
        const int id = ids[assignment], b = id - shape.first_expert;
        if (id < 0 || b < 0 || b >= shape.experts) continue;
        const std::int64_t prow =
            down.row_begin + static_cast<std::int64_t>(b) * shape.hidden + row;
        const float o = bf16_round(full_dot(
            down, prow, hidden + static_cast<std::int64_t>(assignment) * shape.intermediate, lane));
        sum           = __fadd_rn(sum, __fmul_rn(weights[assignment], o));
    }
    if (lane == 0) {
        float* out = partial + static_cast<std::int64_t>(token) * shape.hidden + row;
        *out       = accumulate ? __fadd_rn(*out, sum) : sum;
    }
}

} // namespace reference

// The first moe_sigmoid_route implementation (one block per column), the bit reference of the
// router's production route.
namespace router_reference {

constexpr int kWarps   = 8;
constexpr int kThreads = kWarps * 32;

__device__ __forceinline__ bool better(float key, int id, float best_key, int best_id) {
    if (id < 0) { return false; }
    if (best_id < 0) { return true; }
    return key > best_key || (key == best_key && id < best_id);
}

__global__ void __launch_bounds__(kThreads)
    moe_sigmoid_route_kernel(const __nv_bfloat16* __restrict__ x, const float* __restrict__ router,
                             const float* __restrict__ bias, std::int32_t hidden,
                             std::int32_t experts, std::int32_t top_k, float route_scale,
                             std::int32_t* __restrict__ ids, float* __restrict__ weights,
                             const std::int32_t* __restrict__ valid_columns,
                             std::int32_t lane_width) {
    extern __shared__ float shared[];
    float* score = shared;           // [E]
    float* key   = shared + experts; // [E]
    __shared__ float best_key_of[kWarps];
    __shared__ int best_id_of[kWarps];
    __shared__ int chosen[64];

    const int column      = static_cast<int>(blockIdx.x);
    const int warp        = static_cast<int>(threadIdx.x) >> 5;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    std::int32_t* out_ids = ids + static_cast<std::int64_t>(column) * top_k;
    float* out_weights    = weights + static_cast<std::int64_t>(column) * top_k;

    if (valid_columns != nullptr && column % lane_width >= valid_columns[column / lane_width]) {
        for (int j = static_cast<int>(threadIdx.x); j < top_k; j += kThreads) {
            out_ids[j]     = -1;
            out_weights[j] = 0.0F;
        }
        return;
    }

    const __nv_bfloat16* u = x + static_cast<std::int64_t>(column) * hidden;
    for (int e = warp; e < experts; e += kWarps) {
        const float* row = router + static_cast<std::int64_t>(e) * hidden;
        float sum        = 0.0F;
        for (int h = lane; h < hidden; h += 32) { sum = fmaf(row[h], __bfloat162float(u[h]), sum); }
        sum = reference::warp_sum(sum);
        if (lane == 0) {
            const float s = 1.0F / (1.0F + expf(-sum));
            score[e]      = s;
            key[e]        = s + bias[e];
        }
    }
    __syncthreads();

    for (int j = 0; j < top_k; ++j) {
        float best_key = 0.0F;
        int best_id    = -1;
        for (int e = static_cast<int>(threadIdx.x); e < experts; e += kThreads) {
            bool taken = false;
            for (int i = 0; i < j; ++i) { taken = taken || chosen[i] == e; }
            if (!taken && better(key[e], e, best_key, best_id)) {
                best_key = key[e];
                best_id  = e;
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_key = __shfl_xor_sync(0xffffffffU, best_key, offset);
            const int other_id    = __shfl_xor_sync(0xffffffffU, best_id, offset);
            if (better(other_key, other_id, best_key, best_id)) {
                best_key = other_key;
                best_id  = other_id;
            }
        }
        if (lane == 0) {
            best_key_of[warp] = best_key;
            best_id_of[warp]  = best_id;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float k_best = 0.0F;
            int id_best  = -1;
            for (int w = 0; w < kWarps; ++w) {
                if (better(best_key_of[w], best_id_of[w], k_best, id_best)) {
                    k_best  = best_key_of[w];
                    id_best = best_id_of[w];
                }
            }
            chosen[j] = id_best;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float total = 0.0F;
        for (int j = 0; j < top_k; ++j) { total += score[chosen[j]]; }
        const float denominator = total + 1e-20F;
        for (int j = 0; j < top_k; ++j) {
            out_ids[j]     = chosen[j];
            out_weights[j] = score[chosen[j]] / denominator * route_scale;
        }
    }
}


} // namespace router_reference

namespace {

std::uint64_t splitmix(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z               = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z               = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

struct Parent {
    DeviceBuffer data;
    WeightParent parent;
};

std::unique_ptr<Parent> make_parent(bool nvfp4, std::uint64_t rows, std::uint64_t cols,
                                    std::uint64_t divisors, std::uint64_t seed) {
    auto p = std::make_unique<Parent>();
    const std::array<std::uint64_t, 2> shape{rows, cols};
    p->parent.geometry =
        weight_geometry(nvfp4 ? QType::NVFP4 : QType::BF16,
                        nvfp4 ? QuantLayout::BlockScaleK16M128x4 : QuantLayout::Contiguous, shape,
                        nvfp4 ? divisors : 1);
    const auto& g = p->parent.geometry;
    std::vector<std::uint8_t> host(g.bytes);
    std::uint64_t s = seed;
    if (nvfp4) {
        for (std::size_t i = 0; i + 8 <= g.code_bytes; i += 8) {
            const std::uint64_t r = splitmix(s);
            std::memcpy(host.data() + i, &r, 8);
        }
        for (std::size_t i = 0; i < g.scale_bytes; ++i) {
            const std::uint64_t r    = splitmix(s);
            const int e              = 11 + static_cast<int>(r % 5),
                      m              = static_cast<int>((r >> 8) % (e == 15 ? 7 : 8));
            host[g.scale_offset + i] = static_cast<std::uint8_t>((e << 3) | m);
        }
        for (std::uint64_t d = 0; d < divisors; ++d) {
            const float word = 8000.0F + static_cast<float>(splitmix(s) % 20000);
            std::memcpy(host.data() + g.divisor_offset + d * 4, &word, 4);
        }
    } else {
        const double a = 0.06 * std::sqrt(2304.0 / static_cast<double>(cols));
        for (std::uint64_t i = 0; i < rows * cols; ++i) {
            const double u           = static_cast<double>(splitmix(s) >> 11) * 0x1p-53;
            const std::uint16_t bits = f32_to_bf16(static_cast<float>((2 * u - 1) * a));
            std::memcpy(host.data() + i * 2, &bits, 2);
        }
    }
    p->data = DeviceBuffer(g.bytes);
    p->data.copy_from_host(host.data(), g.bytes);
    p->parent.data = static_cast<const std::byte*>(p->data.p);
    return p;
}

WeightView rows_of(const Parent& p, std::uint64_t begin, std::uint64_t count) {
    const auto cols = p.parent.geometry.shape[1];
    WeightView v;
    v.shape = {count, cols};
    v.parts = {{&p.parent, begin * cols, (begin + count) * cols}};
    return v;
}

reference::Plane plane_of(const Parent& p, std::uint64_t row_begin) {
    const auto& g = p.parent.geometry;
    reference::Plane out;
    out.data             = static_cast<const std::uint8_t*>(p.data.p);
    out.nvfp4            = g.format == QType::NVFP4;
    out.row_begin        = static_cast<std::int64_t>(row_begin);
    out.columns          = static_cast<std::int32_t>(g.shape[1]);
    out.scale_offset     = g.scale_offset;
    out.divisor_offset   = g.divisor_offset;
    out.rows_per_divisor = static_cast<std::int64_t>(g.rows_per_divisor());
    return out;
}

struct Config {
    std::string label;
    bool nvfp4;
    int hidden, intermediate, parent_experts, view_first, bank_experts, first_expert, total;
    int gate_up_sources;
};

int run(const Config& c, const std::vector<int>& widths, std::uint32_t seed) {
    const int H = c.hidden, F = c.intermediate, K = 8;
    const auto gu =
        make_parent(c.nvfp4, static_cast<std::uint64_t>(c.parent_experts) * 2 * F, H,
                    static_cast<std::uint64_t>(c.parent_experts) * c.gate_up_sources, seed);
    const auto dn       = make_parent(c.nvfp4, static_cast<std::uint64_t>(c.parent_experts) * H, F,
                                      static_cast<std::uint64_t>(c.parent_experts), seed + 1);
    const auto gu_begin = static_cast<std::uint64_t>(c.view_first) * 2 * F;
    const auto dn_begin = static_cast<std::uint64_t>(c.view_first) * H;
    const WeightView gu_view =
        rows_of(*gu, gu_begin, static_cast<std::uint64_t>(c.bank_experts) * 2 * F);
    const WeightView dn_view =
        rows_of(*dn, dn_begin, static_cast<std::uint64_t>(c.bank_experts) * H);
    const ops::MoeBank bank{gu_view, dn_view, c.first_expert};
    int failures = 0;
    std::mt19937 rng(seed + 2);
    for (const int T : widths) {
        if (T > 47) { throw std::invalid_argument("the reference execution is the narrow route"); }
        std::vector<float> x(static_cast<std::size_t>(T) * H), w(static_cast<std::size_t>(T) * K),
            init(static_cast<std::size_t>(T) * H);
        fill_uniform(x, seed + 3 + T, -1.7F, 1.7F);
        fill_uniform(w, seed + 4 + T, 0.1F, 2.0F);
        fill_uniform(init, seed + 5 + T, -3.0F, 3.0F);
        std::vector<std::uint16_t> xb(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) xb[i] = f32_to_bf16(x[i]);
        std::vector<std::int32_t> ids(static_cast<std::size_t>(T) * K);
        std::vector<int> all(static_cast<std::size_t>(c.total));
        for (int e = 0; e < c.total; ++e) all[static_cast<std::size_t>(e)] = e;
        for (int t = 0; t < T; ++t) {
            std::shuffle(all.begin(), all.end(), rng);
            for (int j = 0; j < K; ++j)
                ids[static_cast<std::size_t>(t) * K + j] = all[static_cast<std::size_t>(j)];
            if (t % 5 == 2) ids[static_cast<std::size_t>(t) * K + 3] = -1;
            if (t % 7 == 3)
                ids[static_cast<std::size_t>(t) * K + 1] = ids[static_cast<std::size_t>(t) * K];
        }
        const std::size_t n = static_cast<std::size_t>(T) * H;
        DeviceBuffer dx(xb.size() * 2), dids(ids.size() * 4), dw(w.size() * 4), dop(n * 4),
            dref(n * 4), dopa(n * 4), drefa(n * 4), dh(static_cast<std::size_t>(T) * K * F * 2);
        dx.copy_from_host(xb.data(), dx.bytes);
        dids.copy_from_host(ids.data(), dids.bytes);
        dw.copy_from_host(w.data(), dw.bytes);
        dop.fill(0x7f);
        dref.fill(0x7f);
        dopa.copy_from_host(init.data(), n * 4);
        drefa.copy_from_host(init.data(), n * 4);
        Tensor tx(dx.p, DType::BF16, {H, T});
        Tensor tids(dids.p, DType::I32, {K, T});
        Tensor tw(dw.p, DType::FP32, {K, T});
        Tensor top(dop.p, DType::FP32, {H, T});
        Tensor topa(dopa.p, DType::FP32, {H, T});
        WorkspaceArena work(ops::moe_bank_capped_swiglu_workspace_bytes(K, F, T));
        ops::moe_bank_capped_swiglu(tx, tids, tw, bank, {4.0F, 25.0F}, ops::MoeBankOutput::Store,
                                    top, work, nullptr);
        ops::moe_bank_capped_swiglu(tx, tids, tw, bank, {4.0F, 25.0F},
                                    ops::MoeBankOutput::Accumulate, topa, work, nullptr);
        const reference::Shape shape{H, F, c.bank_experts, c.first_expert, K, T};
        const auto* xp = static_cast<const __nv_bfloat16*>(dx.p);
        const auto* ip = static_cast<const std::int32_t*>(dids.p);
        auto* hp       = static_cast<__nv_bfloat16*>(dh.p);
        const dim3 g1(static_cast<unsigned>(T * K), (F + 7) / 8),
            g2(static_cast<unsigned>(T), (H + 7) / 8);
        reference::gate_up_kernel<<<g1, reference::kThreads>>>(xp, ip, plane_of(*gu, gu_begin),
                                                               shape, 4.0F, 25.0F, hp);
        for (const bool acc : {false, true}) {
            reference::down_kernel<<<g2, reference::kThreads>>>(
                ip, static_cast<const float*>(dw.p), hp, plane_of(*dn, dn_begin), shape, acc,
                static_cast<float*>(acc ? drefa.p : dref.p));
        }
        cuda_check_last_launch("reference kernels");
        cuda_synchronize();
        const auto a     = from_device<std::uint32_t>(dop.p, n),
                   b     = from_device<std::uint32_t>(dref.p, n);
        const auto aa    = from_device<std::uint32_t>(dopa.p, n),
                   ba    = from_device<std::uint32_t>(drefa.p, n);
        std::size_t diff = 0, diff_acc = 0;
        for (std::size_t i = 0; i < n; ++i) {
            diff += a[i] != b[i];
            diff_acc += aa[i] != ba[i];
        }
        std::cout << "CASE " << c.label << " T=" << T << " store_bits_differ=" << diff
                  << " accumulate_bits_differ=" << diff_acc
                  << ((diff || diff_acc) ? " FAIL" : " ok") << '\n';
        failures += (diff || diff_acc) ? 1 : 0;
    }
    return failures;
}

// Wide route: the result does not depend on the workspace size (one pass with the four-argument
// size, several token passes with the three-argument one), and two calls give the same bits.
int run_wide(const Config& c, const std::vector<int>& widths, std::uint32_t seed) {
    const int H = c.hidden, F = c.intermediate, K = 8;
    const auto gu =
        make_parent(c.nvfp4, static_cast<std::uint64_t>(c.parent_experts) * 2 * F, H,
                    static_cast<std::uint64_t>(c.parent_experts) * c.gate_up_sources, seed);
    const auto dn = make_parent(c.nvfp4, static_cast<std::uint64_t>(c.parent_experts) * H, F,
                                static_cast<std::uint64_t>(c.parent_experts), seed + 1);
    const WeightView gu_view = rows_of(*gu, static_cast<std::uint64_t>(c.view_first) * 2 * F,
                                       static_cast<std::uint64_t>(c.bank_experts) * 2 * F);
    const WeightView dn_view = rows_of(*dn, static_cast<std::uint64_t>(c.view_first) * H,
                                       static_cast<std::uint64_t>(c.bank_experts) * H);
    const ops::MoeBank bank{gu_view, dn_view, c.first_expert};
    int failures = 0;
    std::mt19937 rng(seed + 2);
    for (const int T : widths) {
        std::vector<float> x(static_cast<std::size_t>(T) * H), w(static_cast<std::size_t>(T) * K);
        fill_uniform(x, seed + 3 + T, -1.7F, 1.7F);
        fill_uniform(w, seed + 4 + T, 0.1F, 2.0F);
        std::vector<std::uint16_t> xb(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) xb[i] = f32_to_bf16(x[i]);
        std::vector<std::int32_t> ids(static_cast<std::size_t>(T) * K);
        std::vector<int> all(static_cast<std::size_t>(c.total));
        for (int e = 0; e < c.total; ++e) all[static_cast<std::size_t>(e)] = e;
        for (int t = 0; t < T; ++t) {
            std::shuffle(all.begin(), all.end(), rng);
            for (int j = 0; j < K; ++j)
                ids[static_cast<std::size_t>(t) * K + j] = all[static_cast<std::size_t>(j)];
            if (t % 5 == 2) ids[static_cast<std::size_t>(t) * K + 3] = -1;
            if (t % 7 == 3)
                ids[static_cast<std::size_t>(t) * K + 1] = ids[static_cast<std::size_t>(t) * K];
        }
        const std::size_t n = static_cast<std::size_t>(T) * H;
        DeviceBuffer dx(xb.size() * 2), dids(ids.size() * 4), dw(w.size() * 4), d1(n * 4),
            d2(n * 4), d3(n * 4);
        dx.copy_from_host(xb.data(), dx.bytes);
        dids.copy_from_host(ids.data(), dids.bytes);
        dw.copy_from_host(w.data(), dw.bytes);
        Tensor tx(dx.p, DType::BF16, {H, T});
        Tensor tids(dids.p, DType::I32, {K, T});
        Tensor tw(dw.p, DType::FP32, {K, T});
        Tensor t1(d1.p, DType::FP32, {H, T}), t2(d2.p, DType::FP32, {H, T}),
            t3(d3.p, DType::FP32, {H, T});
        WorkspaceArena full(ops::moe_bank_capped_swiglu_workspace_bytes(K, F, H, T));
        WorkspaceArena small(ops::moe_bank_capped_swiglu_workspace_bytes(K, F, T));
        ops::moe_bank_capped_swiglu(tx, tids, tw, bank, {4.0F, 25.0F}, ops::MoeBankOutput::Store,
                                    t1, full, nullptr);
        ops::moe_bank_capped_swiglu(tx, tids, tw, bank, {4.0F, 25.0F}, ops::MoeBankOutput::Store,
                                    t2, small, nullptr);
        ops::moe_bank_capped_swiglu(tx, tids, tw, bank, {4.0F, 25.0F}, ops::MoeBankOutput::Store,
                                    t3, full, nullptr);
        cuda_synchronize();
        const auto a = from_device<std::uint32_t>(d1.p, n), b = from_device<std::uint32_t>(d2.p, n),
                   r          = from_device<std::uint32_t>(d3.p, n);
        std::size_t diff_size = 0, diff_repeat = 0;
        for (std::size_t i = 0; i < n; ++i) {
            diff_size += a[i] != b[i];
            diff_repeat += a[i] != r[i];
        }
        std::cout << "CASE wide " << c.label << " T=" << T << " passes_bits_differ=" << diff_size
                  << " repeat_bits_differ=" << diff_repeat
                  << ((diff_size || diff_repeat) ? " FAIL" : " ok") << '\n';
        failures += (diff_size || diff_repeat) ? 1 : 0;
    }
    return failures;
}

// An MXFP8 parent (block_scale_k32_v1) and a BF16 parent holding its exact values.
struct ParentPair {
    std::unique_ptr<Parent> mxfp8, bf16;
};

ParentPair make_mxfp8_pair(std::uint64_t rows, std::uint64_t cols, std::uint64_t seed) {
    ParentPair out;
    const std::array<std::uint64_t, 2> shape{rows, cols};
    out.mxfp8 = std::make_unique<Parent>();
    out.bf16  = std::make_unique<Parent>();
    out.mxfp8->parent.geometry =
        weight_geometry(QType::MXFP8_E4M3FN_G32_UE8M0, QuantLayout::BlockScaleK32, shape);
    out.bf16->parent.geometry = weight_geometry(QType::BF16, QuantLayout::Contiguous, shape);
    const auto& g             = out.mxfp8->parent.geometry;
    std::vector<std::uint8_t> host(g.bytes, 0);
    std::vector<std::uint16_t> values(rows * cols);
    std::uint64_t s = seed;
    for (std::uint64_t r = 0; r < rows; ++r) {
        for (std::uint64_t b = 0; b < cols / 32; ++b) {
            const auto exponent = static_cast<std::uint8_t>(118 + splitmix(s) % 4);
            host[g.scale_offset + r * (cols / 32) + b] = exponent;
            for (std::uint64_t c = 0; c < 32; ++c) {
                const std::uint64_t random = splitmix(s);
                // Sign, exponent field 0..8, mantissa: never the NaN codes.
                const auto code = static_cast<std::uint8_t>(
                    ((random & 1U) << 7) | ((random >> 1) % 9) << 3 | ((random >> 8) & 7U));
                host[r * cols + b * 32 + c] = code;
                const int e = (code >> 3) & 15, m = code & 7;
                double v =
                    e == 0 ? m * std::ldexp(1.0, -9) : (1.0 + m / 8.0) * std::ldexp(1.0, e - 7);
                v = (code & 0x80) ? -v : v;
                values[r * cols + b * 32 + c] =
                    f32_to_bf16(static_cast<float>(std::ldexp(v, exponent - 127)));
            }
        }
    }
    out.mxfp8->data = DeviceBuffer(g.bytes);
    out.mxfp8->data.copy_from_host(host.data(), g.bytes);
    out.mxfp8->parent.data = static_cast<const std::byte*>(out.mxfp8->data.p);
    out.bf16->data         = DeviceBuffer(values.size() * 2);
    out.bf16->data.copy_from_host(values.data(), values.size() * 2);
    out.bf16->parent.data = static_cast<const std::byte*>(out.bf16->data.p);
    return out;
}

// An MXFP8 bank against the BF16 bank of its values: the same bits on both routes.
int run_mxfp8(const std::string& label, int H, int F, int experts, int first_expert, int total,
              const std::vector<int>& widths, std::uint32_t seed) {
    const int K      = 8;
    const auto gu    = make_mxfp8_pair(static_cast<std::uint64_t>(experts) * 2 * F, H, seed);
    const auto dn    = make_mxfp8_pair(static_cast<std::uint64_t>(experts) * H, F, seed + 1);
    const auto whole = [](const Parent& p) { return rows_of(p, 0, p.parent.geometry.shape[0]); };
    const WeightView gu8 = whole(*gu.mxfp8), dn8 = whole(*dn.mxfp8);
    const WeightView gu16 = whole(*gu.bf16), dn16 = whole(*dn.bf16);
    const ops::MoeBank bank8{gu8, dn8, first_expert}, bank16{gu16, dn16, first_expert};
    int failures = 0;
    std::mt19937 rng(seed + 2);
    for (const int T : widths) {
        std::vector<float> x(static_cast<std::size_t>(T) * H), w(static_cast<std::size_t>(T) * K);
        fill_uniform(x, seed + 3 + T, -1.7F, 1.7F);
        fill_uniform(w, seed + 4 + T, 0.1F, 2.0F);
        std::vector<std::uint16_t> xb(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) xb[i] = f32_to_bf16(x[i]);
        std::vector<std::int32_t> ids(static_cast<std::size_t>(T) * K);
        std::vector<int> all(static_cast<std::size_t>(total));
        for (int e = 0; e < total; ++e) all[static_cast<std::size_t>(e)] = e;
        for (int t = 0; t < T; ++t) {
            std::shuffle(all.begin(), all.end(), rng);
            for (int j = 0; j < K; ++j)
                ids[static_cast<std::size_t>(t) * K + j] = all[static_cast<std::size_t>(j)];
        }
        const std::size_t n = static_cast<std::size_t>(T) * H;
        DeviceBuffer dx(xb.size() * 2), dids(ids.size() * 4), dw(w.size() * 4), d8(n * 4),
            d16(n * 4);
        dx.copy_from_host(xb.data(), dx.bytes);
        dids.copy_from_host(ids.data(), dids.bytes);
        dw.copy_from_host(w.data(), dw.bytes);
        Tensor tx(dx.p, DType::BF16, {H, T});
        Tensor tids(dids.p, DType::I32, {K, T});
        Tensor tw(dw.p, DType::FP32, {K, T});
        Tensor t8(d8.p, DType::FP32, {H, T}), t16(d16.p, DType::FP32, {H, T});
        WorkspaceArena work(ops::moe_bank_capped_swiglu_workspace_bytes(K, F, H, T));
        ops::moe_bank_capped_swiglu(tx, tids, tw, bank8, {4.0F, 25.0F}, ops::MoeBankOutput::Store,
                                    t8, work, nullptr);
        ops::moe_bank_capped_swiglu(tx, tids, tw, bank16, {4.0F, 25.0F}, ops::MoeBankOutput::Store,
                                    t16, work, nullptr);
        cuda_synchronize();
        const auto a     = from_device<std::uint32_t>(d8.p, n),
                   b     = from_device<std::uint32_t>(d16.p, n);
        std::size_t diff = 0, nonzero = 0;
        for (std::size_t i = 0; i < n; ++i) {
            diff += a[i] != b[i];
            nonzero += (a[i] & 0x7FFFFFFFU) != 0U;
        }
        const bool ok = diff == 0 && nonzero > 0;
        std::cout << "CASE mxfp8=bf16 " << label << " T=" << T << " bits_differ=" << diff
                  << " nonzero=" << nonzero << (ok ? " ok" : " FAIL") << '\n';
        failures += ok ? 0 : 1;
    }
    return failures;
}

int run_router(int hidden, int experts, int top_k, int tokens, std::uint32_t seed) {
    std::vector<float> x(static_cast<std::size_t>(tokens) * hidden),
        router(static_cast<std::size_t>(experts) * hidden), bias(static_cast<std::size_t>(experts));
    fill_uniform(x, seed, -1.7F, 1.7F);
    fill_uniform(router, seed + 1, -0.07F, 0.07F);
    fill_uniform(bias, seed + 2, -0.5F, 0.5F);
    // exact key ties: identical rows and biases
    std::copy_n(router.begin(), hidden,
                router.begin() + static_cast<std::ptrdiff_t>(hidden) * (experts - 1));
    bias[static_cast<std::size_t>(experts - 1)] = bias[0];
    std::vector<std::uint16_t> xb(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) xb[i] = f32_to_bf16(x[i]);
    const std::size_t outs = static_cast<std::size_t>(tokens) * top_k;
    DeviceBuffer dx(xb.size() * 2), dr(router.size() * 4), db(bias.size() * 4), di(outs * 4),
        dw(outs * 4), ri(outs * 4), rw(outs * 4);
    dx.copy_from_host(xb.data(), dx.bytes);
    dr.copy_from_host(router.data(), dr.bytes);
    db.copy_from_host(bias.data(), db.bytes);
    Tensor tx(dx.p, DType::BF16, {hidden, tokens});
    Tensor tr(dr.p, DType::FP32, {hidden, experts});
    Tensor tb(db.p, DType::FP32, {experts});
    Tensor ti(di.p, DType::I32, {top_k, tokens});
    Tensor tw(dw.p, DType::FP32, {top_k, tokens});
    ops::moe_sigmoid_route(tx, tr, tb, top_k, 8.0F, ti, tw, nullptr);
    router_reference::moe_sigmoid_route_kernel<<<tokens, router_reference::kThreads,
                                                 2 * static_cast<std::size_t>(experts) * 4>>>(
        static_cast<const __nv_bfloat16*>(dx.p), static_cast<const float*>(dr.p),
        static_cast<const float*>(db.p), hidden, experts, top_k, 8.0F,
        static_cast<std::int32_t*>(ri.p), static_cast<float*>(rw.p), nullptr, tokens);
    cuda_check_last_launch("router reference");
    cuda_synchronize();
    const auto a     = from_device<std::uint32_t>(di.p, outs),
               b     = from_device<std::uint32_t>(ri.p, outs);
    const auto c     = from_device<std::uint32_t>(dw.p, outs),
               d     = from_device<std::uint32_t>(rw.p, outs);
    std::size_t diff = 0;
    for (std::size_t i = 0; i < outs; ++i) diff += (a[i] != b[i]) + (c[i] != d[i]);
    std::cout << "CASE router H=" << hidden << " E=" << experts << " k=" << top_k << " T=" << tokens
              << " bits_differ=" << diff << (diff ? " FAIL" : " ok") << '\n';
    return diff ? 1 : 0;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    try {
        failures += run({"syn nvfp4 view 8-15", true, 512, 256, 16, 8, 8, 8, 16, 2},
                        {1, 2, 4, 8, 9, 16, 24, 47}, 31);
        failures += run({"syn bf16", false, 512, 256, 8, 0, 8, 0, 16, 1}, {1, 8, 24}, 32);
        failures += run({"real nvfp4 E'=64 off 64", true, 2304, 1152, 64, 0, 64, 64, 128, 1},
                        {1, 2, 3, 4, 5, 8, 16, 24, 47}, 33);
        failures +=
            run({"real bf16 E'=64", false, 2304, 1152, 64, 0, 64, 0, 128, 1}, {1, 8, 24}, 34);
        failures += run_wide({"syn nvfp4 view 8-15", true, 512, 256, 16, 8, 8, 8, 16, 2},
                             {48, 64, 512}, 35);
        failures += run_wide({"syn bf16", false, 512, 256, 8, 0, 8, 0, 16, 1}, {48, 100}, 36);
        failures += run_wide({"real nvfp4 E'=64 off 64", true, 2304, 1152, 64, 0, 64, 64, 128, 1},
                             {512}, 37);
        failures +=
            run_wide({"real bf16 E'=64", false, 2304, 1152, 64, 0, 64, 0, 128, 1}, {64}, 38);
        failures += run_mxfp8("syn", 512, 256, 8, 8, 16, {1, 8, 47, 48, 512}, 39);
        failures += run_mxfp8("real 16 experts", 2304, 1152, 16, 0, 32, {1, 8, 64}, 40);
        for (const int t : {1, 4, 16, 63, 64, 127, 128, 131, 512, 1024}) {
            failures += run_router(2304, 128, 8, t, 40U + static_cast<std::uint32_t>(t));
        }
        failures += run_router(512, 16, 8, 16, 41U);
        failures += run_router(100, 7, 7, 3, 42U);
        failures += run_router(512, 16, 8, 136, 43U);
        failures += run_router(100, 7, 7, 130, 44U);
    } catch (const std::exception& error) {
        std::cerr << "moe_bank routes test error: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << "moe_bank routes: " << failures << " failures\n";
        return 1;
    }
    std::cout << "OK moe_bank routes bit-identical to the reference execution\n";
    return 0;
}
