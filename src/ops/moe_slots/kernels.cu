// Decode-size NVFP4 (W4A4) mixture-of-experts over expert-cache slots, Qwen3.8-Flash-Next geometry.
//
// Derived from qf_moe.cu of https://github.com/SSHdotCodes/qwen-3.8-flash-next-pro6000
// (commit 8e5be44, serve/qwenfast/qf_moe.cu), Copyright 2026 SSHdotCodes contributors,
// licensed under the Apache License, Version 2.0.
// Modifications Copyright 2026 Valerio Dolci, Apache License, Version 2.0: routed ids are mapped
// to cache slots in both kernels (slot table, -1 = not resident), weights are read from slot
// planes with per-plane strides in the bank row order (gate rows first), per-expert weight
// divisors are read by expert id with separate gate/up divisors, activation divisors are launch
// parameters, the partial is kept in FP32 (store/accumulate) or BF16, a call without resident
// experts still defines its output, per-token weights of a repeated expert are summed in top-k
// order, the arrival counters are cleared by the first kernel (no state between calls), and the
// workspace is caller-owned; both kernels run over a compacted list of the distinct resident
// experts on persistent grids with a rolling weight prefetch (per-(row, token) arithmetic
// unchanged, output bit-identical to the one-CTA-per-position kernels).
//
// Same quantization recipe as FlashInfer's CUTLASS MoE (TensorRT-LLM kernels): activations are
// quantized to E2M1 with E4M3 block scales (16 elements) using the fast-math reciprocal recipe;
// gemm1 outputs are rounded to BF16, SwiGLU silu(gate) * up is rounded to BF16 and re-quantized
// before gemm2. Products of two E2M1 codes are small integers (after x2 scaling), so every
// 16-element block sum is exact with dp4a and scaled by both E4M3 block scales.
//
//   gemm1 : items (distinct resident expert, 16 intermediate rows); every resident expert is read
//           once; x -> FP4 once per CTA; SwiGLU; h -> FP4
//   gemm2 : items (distinct resident expert, 128 output rows); the last item per row tile sums the
//           (token, position) partials in top-k order
#include "ops/moe_slots/launch.h"

#include "core/device.h"
#include "core/pdl.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int H = kMoeSlotsHidden, I = kMoeSlotsIntermediate, TOPK = kMoeSlotsTopK;
constexpr int MAXT = kMoeSlotsMaxTokens, MAXSLOT = MAXT * TOPK;
constexpr int HB = H / 16, IB = I / 16; // 160, 40 scale blocks per row
constexpr int HQ = HB / 4, IQ = IB / 4; // 40, 10 quads (64 elements) per row
constexpr int G1_THREADS = 128, G1_BLOCKS_PER_POS = I / 16;
constexpr int G2_THREADS = 256, G2_ROWS = 128, G2_TILES = H / G2_ROWS;
// Compacted kernels: register caps for 3 gemm1 / 2 gemm2 CTAs per SM (no spills at these caps),
// persistent grids of G1_GRID_PER_SM / G2_GRID_PER_SM CTAs per SM (sweep on the RTX 5070 Ti:
// 2/2 best at the decode window shapes; the free gemm1 capacity lets gemm2 CTAs start and
// prefetch their weights while gemm1 drains).
constexpr int G1_MIN_BLOCKS = 3, G2_MIN_BLOCKS = 2;
constexpr int G1_GRID_PER_SM = 2, G2_GRID_PER_SM = 2;

static_assert(MAXSLOT <= G1_THREADS && MAXSLOT <= G2_THREADS);
static_assert(G2_TILES <= G1_THREADS);

struct PoolArgs {
    const std::uint8_t* w13;
    const std::uint8_t* w13_sf;
    const std::uint8_t* w2;
    const std::uint8_t* w2_sf;
    long long w13_stride, w13_sf_stride, w2_stride, w2_sf_stride;
    int slots;
};

struct LayerArgs {
    const int* table;
    const float* gate_up_div;
    const float* down_div;
    int experts;
    float a1_gs, a2_gs;
};

__device__ __forceinline__ int slot_of(const int* __restrict__ ids, const LayerArgs& layer,
                                       int slots, int i) {
    const int e = ids[i];
    if (e < 0 || e >= layer.experts) return -1;
    const int s = layer.table[e];
    return (s >= 0 && s < slots) ? s : -1;
}

__device__ __forceinline__ float rcp_approx(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float e4m3_to_float(std::uint32_t code) {
    __nv_fp8_e4m3 t;
    t.__x = static_cast<__nv_fp8_storage_t>(code);
    return static_cast<float>(t);
}

// 16 values -> 8 bytes of E2M1 codes (low nibble = even element) + E4M3 scale, FlashInfer
// fast-math recipe.
__device__ __forceinline__ uint2 quant16(const float (&v)[16], float gs, float& sf_val) {
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
    const __nv_fp8_e4m3 code(gs * (amax * rcp_approx(6.0f)));
    const float sf        = static_cast<float>(code);
    const float out_scale = amax != 0.f ? rcp_approx(sf * rcp_approx(gs)) : 0.f;
    sf_val                = sf;
    std::uint32_t w[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        std::uint32_t packed = 0;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const float a = v[h * 8 + 2 * p] * out_scale, b = v[h * 8 + 2 * p + 1] * out_scale;
            std::uint16_t r;
            asm("{\n.reg .b8 t;\ncvt.rn.satfinite.e2m1x2.f32 t, %2, %1;\ncvt.u16.u8 %0, t;\n}"
                : "=h"(r)
                : "f"(a), "f"(b));
            packed |= static_cast<std::uint32_t>(r & 0xFF) << (8 * p);
        }
        w[h] = packed;
    }
    return make_uint2(w[0], w[1]);
}

// 8 E2M1 codes in one word -> int8 values x2 ({0,1,2,3,4,6,8,12} with sign), even/odd split.
__device__ __forceinline__ int2 fp4x8_to_i8(std::uint32_t q) {
    constexpr std::uint32_t t0 = 0x03020100u, t1 = 0x0C080604u, t2 = 0xFDFEFF00u, t3 = 0xF4F8FAFCu;
    const std::uint32_t sel = 0x32103210u | ((q & 0x88888888u) >> 1);
    std::uint32_t tmp[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const std::uint32_t s  = q >> (16 * i);
        const std::uint32_t lo = __byte_perm(t0, t1, s);
        const std::uint32_t hi = __byte_perm(t2, t3, s);
        tmp[i]                 = __byte_perm(lo, hi, sel >> (16 * i));
    }
    return make_int2(static_cast<int>(__byte_perm(tmp[0], tmp[1], 0x6420)),
                     static_cast<int>(__byte_perm(tmp[0], tmp[1], 0x7531)));
}

// block_scale_k16_m128x4_v1: the 4 consecutive scale words of quad `quad` of row `row`.
__device__ __forceinline__ std::size_t sf_quad_offset(int row, int quad, int kq) {
    return (static_cast<std::size_t>(row >> 7) * kq + quad) * 512 + (row & 31) * 16 +
           ((row & 127) >> 5) * 4;
}

__device__ __forceinline__ int nth_token(unsigned mask, int rank) {
    for (int k = 0; k < rank; ++k) mask &= mask - 1;
    return __ffs(mask) - 1;
}

__device__ __forceinline__ void quant_x_block(const __nv_bfloat16* __restrict__ src, float gs,
                                              int2* dst8, float& sf) {
    const uint4 u0 = __ldg(reinterpret_cast<const uint4*>(src));
    const uint4 u1 = __ldg(reinterpret_cast<const uint4*>(src) + 1);
    float v[16];
    const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&u0);
    const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&u1);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 a = __bfloat1622float2(h0[i]), c = __bfloat1622float2(h1[i]);
        v[2 * i]         = a.x;
        v[2 * i + 1]     = a.y;
        v[8 + 2 * i]     = c.x;
        v[8 + 2 * i + 1] = c.y;
    }
    const uint2 q = quant16(v, gs, sf);
    dst8[0]       = fp4x8_to_i8(q.x);
    dst8[1]       = fp4x8_to_i8(q.y);
}

// ---- compacted persistent kernels ----------------------------------------------------------
// Every CTA builds the call's list of distinct resident experts (first position order) from the
// routed ids, then loops over (expert, row tile) items: the grid is a fixed number of CTAs per SM,
// no CTA is spent on non-resident or repeated positions, x is quantized once per CTA (all T
// tokens), and the weights of the CTA's next item are loaded while the current one is computed.

// Distinct resident slots of the call in first-position order: s_list[u] = first position of the
// u-th distinct slot, s_rank[p] = u of position p's slot (or -1). Returns U. Positions >= n and
// threads >= n contribute nothing. Requires s_slot filled and a barrier before the call.
__device__ __forceinline__ int distinct_slots(const int* s_slot, int n, int tid, int* s_list,
                                              int* s_rank, int* s_warp) {
    const int lane = tid & 31, warp = tid >> 5;
    int fp = -1;
    if (tid < n && s_slot[tid] >= 0) {
        fp = tid;
        for (int k = 0; k < tid; ++k)
            if (s_slot[k] == s_slot[tid]) {
                fp = k;
                break;
            }
    }
    const bool first = fp == tid && fp >= 0;
    const unsigned b = __ballot_sync(0xffffffffu, first);
    if (lane == 0 && warp < (MAXSLOT + 31) / 32) s_warp[warp] = __popc(b);
    __syncthreads();
    int base = 0, total = 0;
#pragma unroll
    for (int w = 0; w < (MAXSLOT + 31) / 32; ++w) {
        const int c = s_warp[w];
        base += w < warp ? c : 0;
        total += c;
    }
    if (first) {
        const int u = base + __popc(b & ((1u << lane) - 1u));
        s_list[u]   = tid;
        s_rank[tid] = u;
    }
    __syncthreads();
    if (tid < n) s_rank[tid] = fp >= 0 ? s_rank[fp] : -1;
    __syncthreads();
    return total;
}

__global__ void __launch_bounds__(G1_THREADS, G1_MIN_BLOCKS)
    moe_slots_gemm1_kernel(const __nv_bfloat16* __restrict__ x, int T, const int* __restrict__ ids,
                           LayerArgs layer, PoolArgs pool, std::uint32_t* __restrict__ hq,
                           float* __restrict__ hsf, std::uint32_t* __restrict__ done) {
    extern __shared__ int2 s_x[]; // [T][H / 8] int8 x2 split, then [T][HB] floats
    __shared__ int s_slot[MAXSLOT], s_id[MAXSLOT], s_list[MAXSLOT], s_rank[MAXSLOT];
    __shared__ unsigned s_umask[MAXSLOT];
    __shared__ int s_warp[(MAXSLOT + 31) / 32];
    __shared__ float s_dot[32][MAXT];
    __shared__ float s_act[MAXT][16];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n = T * TOPK;
    pdl::wait_for_dependencies(); // routed ids, x, previous users of the workspace
    if (tid < n) {
        s_id[tid]    = ids[tid];
        s_slot[tid]  = slot_of(ids, layer, pool.slots, tid);
        s_umask[tid] = 0;
    }
    if (blockIdx.x == 0 && tid < G2_TILES) done[tid] = 0; // gemm2 counts after its own wait
    __syncthreads();
    pdl::trigger_dependents();
    const int U = distinct_slots(s_slot, n, tid, s_list, s_rank, s_warp);
    if (tid < n && s_rank[tid] >= 0) atomicOr(&s_umask[s_rank[tid]], 1u << (tid / TOPK));
    __syncthreads();
    float* s_xsf    = reinterpret_cast<float*>(s_x + T * (H / 8));
    const int items = U * G1_BLOCKS_PER_POS;
    const int r = warp * 8 + (lane >> 2), q0 = lane & 3;
    int item = blockIdx.x;
    if (item >= items) return;
    // Rolling prefetch: the weight registers of quad i are refilled with the next item's quad i
    // as soon as quad i has been decoded, so the next item's loads overlap this item's math.
    const auto row_of = [&](int it) {
        return (r < 16 ? I : 0) + (it % G1_BLOCKS_PER_POS) * 16 + (r & 15);
    };
    const auto slot_at = [&](int it) { return s_slot[s_list[it / G1_BLOCKS_PER_POS]]; };
    uint4 wv[10][2];
    std::uint32_t sv[10];
    {
        const int s = slot_at(item), grow = row_of(item);
        const std::uint8_t* wrow = pool.w13 + static_cast<long long>(s) * pool.w13_stride +
                                   static_cast<long long>(grow) * (H / 2);
        const std::uint8_t* sfe = pool.w13_sf + static_cast<long long>(s) * pool.w13_sf_stride;
#pragma unroll
        for (int i = 0; i < 10; ++i) {
            const int q    = q0 + 4 * i;
            const uint4* p = reinterpret_cast<const uint4*>(wrow + q * 32);
            wv[i][0]       = __ldcs(p);
            wv[i][1]       = __ldcs(p + 1);
            sv[i] = __ldg(reinterpret_cast<const unsigned int*>(sfe + sf_quad_offset(grow, q, HQ)));
        }
    }
    // every token once per CTA, while the first item's weights are in flight
    for (int c = tid; c < T * HB; c += G1_THREADS) {
        const int t = c / HB, b = c % HB;
        quant_x_block(x + static_cast<std::size_t>(t) * H + b * 16, layer.a1_gs,
                      s_x + t * (H / 8) + b * 2, s_xsf[t * HB + b]);
    }
    for (; item < items; item += gridDim.x) {
        const int u = item / G1_BLOCKS_PER_POS, jt = item % G1_BLOCKS_PER_POS;
        const int pos = s_list[u], e = s_id[pos];
        const int next  = item + static_cast<int>(gridDim.x);
        const bool more = next < items;
        const int ns = more ? slot_at(next) : 0, ngrow = row_of(more ? next : item);
        const std::uint8_t* nwrow = pool.w13 + static_cast<long long>(ns) * pool.w13_stride +
                                    static_cast<long long>(ngrow) * (H / 2);
        const std::uint8_t* nsfe = pool.w13_sf + static_cast<long long>(ns) * pool.w13_sf_stride;
        __syncthreads(); // x (first item); the previous item's epilogue
        const unsigned mask = s_umask[u];
        const int te        = __popc(mask);
        int tok[MAXT];
        {
            unsigned m = mask;
#pragma unroll
            for (int rk = 0; rk < MAXT; ++rk) {
                tok[rk] = m ? __ffs(m) - 1 : 0;
                m &= m - 1;
            }
        }

        float acc[MAXT];
#pragma unroll
        for (int t = 0; t < MAXT; ++t) acc[t] = 0.f;
#pragma unroll
        for (int i = 0; i < 10; ++i) {
            const int q = q0 + 4 * i;
            int2 wi[8];
            const std::uint32_t* ww = reinterpret_cast<const std::uint32_t*>(&wv[i][0]);
#pragma unroll
            for (int k = 0; k < 8; ++k) wi[k] = fp4x8_to_i8(ww[k]);
            float wsf[4];
#pragma unroll
            for (int b = 0; b < 4; ++b) wsf[b] = e4m3_to_float((sv[i] >> (8 * b)) & 0xFF);
            if (more) {
                const uint4* p = reinterpret_cast<const uint4*>(nwrow + q * 32);
                wv[i][0]       = __ldcs(p);
                wv[i][1]       = __ldcs(p + 1);
                sv[i]          = __ldg(
                    reinterpret_cast<const unsigned int*>(nsfe + sf_quad_offset(ngrow, q, HQ)));
            }
#pragma unroll
            for (int rk = 0; rk < MAXT; ++rk) {
                if (rk >= te) break;
                const int t = tok[rk];
                float part  = 0.f;
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    int sum = 0;
#pragma unroll
                    for (int k = 0; k < 2; ++k) {
                        const int2 xv = s_x[t * (H / 8) + q * 8 + b * 2 + k];
                        sum           = __dp4a(wi[b * 2 + k].x, xv.x, sum);
                        sum           = __dp4a(wi[b * 2 + k].y, xv.y, sum);
                    }
                    part = fmaf(static_cast<float>(sum), wsf[b] * s_xsf[t * HB + q * 4 + b], part);
                }
                acc[rk] += part;
            }
        }
        const float divisor = layer.gate_up_div[2 * e + (r < 16 ? 1 : 0)];
        const float alpha   = 0.25f / (divisor * layer.a1_gs);
#pragma unroll
        for (int rk = 0; rk < MAXT; ++rk) {
            if (rk >= te) break;
            float v = acc[rk];
            v += __shfl_xor_sync(0xffffffff, v, 1);
            v += __shfl_xor_sync(0xffffffff, v, 2);
            if (q0 == 0) s_dot[r][rk] = v * alpha;
        }
        __syncthreads();
        for (int c = tid; c < 16 * te; c += G1_THREADS) {
            const int j = c & 15, rk = c >> 4;
            const float up   = __bfloat162float(__float2bfloat16(s_dot[j][rk]));
            const float gate = __bfloat162float(__float2bfloat16(s_dot[16 + j][rk]));
            s_act[rk][j] =
                __bfloat162float(__float2bfloat16(gate * (1.f / (1.f + __expf(-gate))) * up));
        }
        __syncthreads();
        if (tid < te) {
            const int rk = tid;
            const int t  = nth_token(mask, rk);
            float v[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) v[j] = s_act[rk][j];
            float sf;
            const uint2 q                   = quant16(v, layer.a2_gs, sf);
            const std::size_t pair          = static_cast<std::size_t>(pos) * T + t;
            hq[pair * (I / 8) + 2 * jt]     = q.x;
            hq[pair * (I / 8) + 2 * jt + 1] = q.y;
            hsf[pair * IB + jt]             = sf;
        }
    }
}

__global__ void __launch_bounds__(G2_THREADS, G2_MIN_BLOCKS)
    moe_slots_gemm2_kernel(int T, const int* __restrict__ ids, const float* __restrict__ tw,
                           LayerArgs layer, PoolArgs pool, const std::uint32_t* __restrict__ hq,
                           const float* __restrict__ hsf, float* __restrict__ part_out,
                           std::uint32_t* __restrict__ done, float* __restrict__ out_f32,
                           __nv_bfloat16* __restrict__ out_bf16, int accumulate) {
    __shared__ int s_slot[MAXSLOT], s_id[MAXSLOT], s_list[MAXSLOT], s_rank[MAXSLOT];
    __shared__ unsigned char s_row_first[MAXSLOT];
    __shared__ int s_warp[(MAXSLOT + 31) / 32];
    __shared__ unsigned s_mask;
    __shared__ float s_w[MAXT];
    __shared__ int s_k[MAXT];
    __shared__ int2 s_h[MAXT][I / 8];
    __shared__ float s_hsf[MAXT][IB];
    __shared__ bool s_last;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n = T * TOPK;
    // routed ids/weights and the table were complete before gemm1 passed its own wait, which
    // precedes its trigger, so they are read before ours
    if (tid < n) {
        s_id[tid]   = ids[tid];
        s_slot[tid] = slot_of(ids, layer, pool.slots, tid);
    }
    __syncthreads();
    if (tid < n) {
        const int sl   = s_slot[tid];
        bool row_first = sl >= 0;
        for (int k = (tid / TOPK) * TOPK; k < tid && row_first; ++k)
            if (s_slot[k] == sl) row_first = false;
        s_row_first[tid] = row_first ? 1 : 0;
    }
    const int U = distinct_slots(s_slot, n, tid, s_list, s_rank, s_warp);
    if (U == 0) {
        // no resident expert in the call: define the output once
        pdl::wait_for_dependencies();
        if (accumulate) return;
        for (int c = blockIdx.x * G2_THREADS + tid; c < T * H; c += gridDim.x * G2_THREADS) {
            if (out_f32 != nullptr)
                out_f32[c] = 0.f;
            else
                out_bf16[c] = __float2bfloat16(0.f);
        }
        return;
    }
    const int items = U * G2_TILES;
    const int row = warp * 16 + (lane >> 1), q0 = lane & 1;
    int item = blockIdx.x;
    if (item >= items) return;
    // Rolling prefetch of the next item's weights, as in gemm1.
    uint4 wv[5][2];
    std::uint32_t sv[5];
    {
        const int s = s_slot[s_list[item / G2_TILES]], grow = (item % G2_TILES) * G2_ROWS + row;
        const std::uint8_t* wrow = pool.w2 + static_cast<long long>(s) * pool.w2_stride +
                                   static_cast<long long>(grow) * (I / 2);
        const std::uint8_t* sfe = pool.w2_sf + static_cast<long long>(s) * pool.w2_sf_stride;
#pragma unroll
        for (int i = 0; i < 5; ++i) {
            const int q    = q0 + 2 * i;
            const uint4* p = reinterpret_cast<const uint4*>(wrow + q * 32);
            wv[i][0]       = __ldcs(p);
            wv[i][1]       = __ldcs(p + 1);
            sv[i] = __ldg(reinterpret_cast<const unsigned int*>(sfe + sf_quad_offset(grow, q, IQ)));
        }
    }
    for (; item < items; item += gridDim.x) {
        const int u = item / G2_TILES, tile = item % G2_TILES;
        const int pos = s_list[u], s = s_slot[pos], e = s_id[pos];
        const int grow            = tile * G2_ROWS + row;
        const int next            = item + static_cast<int>(gridDim.x);
        const bool more           = next < items;
        const int ns              = more ? s_slot[s_list[next / G2_TILES]] : 0;
        const int ngrow           = ((more ? next : item) % G2_TILES) * G2_ROWS + row;
        const std::uint8_t* nwrow = pool.w2 + static_cast<long long>(ns) * pool.w2_stride +
                                    static_cast<long long>(ngrow) * (I / 2);
        const std::uint8_t* nsfe = pool.w2_sf + static_cast<long long>(ns) * pool.w2_sf_stride;
        // this expert's tokens: summed routing weight (top-k order) and first position in the row
        if (warp == 0) {
            float w = 0.f;
            int kf  = -1;
            if (lane < T) {
                for (int k = 0; k < TOPK; ++k) {
                    if (s_slot[lane * TOPK + k] != s) continue;
                    if (kf < 0) kf = k;
                    w += tw[lane * TOPK + k];
                }
                s_w[lane] = w;
                s_k[lane] = kf;
            }
            const unsigned m = __ballot_sync(0xffffffffu, kf >= 0);
            if (lane == 0) s_mask = m;
        }
        pdl::wait_for_dependencies(); // gemm1's h (returns at once after the first item)
        __syncthreads();
        const unsigned mask = s_mask;
        const int te        = __popc(mask);
        for (int c = tid; c < te * (I / 8); c += G2_THREADS) {
            const int rk = c / (I / 8), k = c % (I / 8);
            const int t            = nth_token(mask, rk);
            const std::size_t pair = static_cast<std::size_t>(pos) * T + t;
            s_h[rk][k]             = fp4x8_to_i8(__ldcg(hq + pair * (I / 8) + k));
        }
        for (int c = tid; c < te * IB; c += G2_THREADS) {
            const int rk = c / IB, k = c % IB;
            const int t  = nth_token(mask, rk);
            s_hsf[rk][k] = __ldcg(hsf + (static_cast<std::size_t>(pos) * T + t) * IB + k);
        }
        __syncthreads();

        float part[MAXT];
#pragma unroll
        for (int rk = 0; rk < MAXT; ++rk) part[rk] = 0.f;
#pragma unroll
        for (int i = 0; i < 5; ++i) {
            const int q = q0 + 2 * i;
            int2 wi[8];
            const std::uint32_t* ww = reinterpret_cast<const std::uint32_t*>(&wv[i][0]);
#pragma unroll
            for (int k = 0; k < 8; ++k) wi[k] = fp4x8_to_i8(ww[k]);
            float wsf[4];
#pragma unroll
            for (int b = 0; b < 4; ++b) wsf[b] = e4m3_to_float((sv[i] >> (8 * b)) & 0xFF);
            if (more) {
                const uint4* p = reinterpret_cast<const uint4*>(nwrow + q * 32);
                wv[i][0]       = __ldcs(p);
                wv[i][1]       = __ldcs(p + 1);
                sv[i]          = __ldg(
                    reinterpret_cast<const unsigned int*>(nsfe + sf_quad_offset(ngrow, q, IQ)));
            }
#pragma unroll
            for (int rk = 0; rk < MAXT; ++rk) {
                if (rk >= te) break;
                float p = 0.f;
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    int sum = 0;
#pragma unroll
                    for (int k = 0; k < 2; ++k) {
                        const int2 hv = s_h[rk][q * 8 + b * 2 + k];
                        sum           = __dp4a(wi[b * 2 + k].x, hv.x, sum);
                        sum           = __dp4a(wi[b * 2 + k].y, hv.y, sum);
                    }
                    p = fmaf(static_cast<float>(sum), wsf[b] * s_hsf[rk][q * 4 + b], p);
                }
                part[rk] += p;
            }
        }
        const float alpha = 0.25f / (layer.down_div[e] * layer.a2_gs);
        unsigned m        = mask;
#pragma unroll
        for (int rk = 0; rk < MAXT; ++rk) {
            if (rk >= te) break;
            const float v = part[rk] + __shfl_xor_sync(0xffffffff, part[rk], 1);
            const int t   = __ffs(m) - 1;
            m &= m - 1;
            if (q0 == 0)
                part_out[(static_cast<std::size_t>(t) * TOPK + s_k[t]) * H + grow] =
                    v * alpha * s_w[t];
        }
        // last item of this row tile: sum the partials in top-k order and write the output
        __threadfence();
        __syncthreads();
        if (tid == 0) s_last = atomicAdd(done + tile, 1u) == static_cast<unsigned>(U) - 1;
        __syncthreads();
        if (s_last) {
            __threadfence();
            for (int c = tid; c < T * (G2_ROWS / 4); c += G2_THREADS) {
                const int t = c / (G2_ROWS / 4), r4 = (c % (G2_ROWS / 4)) * 4;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                for (int k = 0; k < TOPK; ++k) {
                    if (!s_row_first[t * TOPK + k]) continue;
                    const float4 a = __ldcg(reinterpret_cast<const float4*>(
                        part_out + (static_cast<std::size_t>(t) * TOPK + k) * H + tile * G2_ROWS +
                        r4));
                    v.x += a.x;
                    v.y += a.y;
                    v.z += a.z;
                    v.w += a.w;
                }
                const std::size_t o = static_cast<std::size_t>(t) * H + tile * G2_ROWS + r4;
                if (out_f32 != nullptr) {
                    float4* dst = reinterpret_cast<float4*>(out_f32 + o);
                    if (accumulate) {
                        const float4 old = *dst;
                        v = make_float4(old.x + v.x, old.y + v.y, old.z + v.z, old.w + v.w);
                    }
                    *dst = v;
                } else {
                    __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y);
                    __nv_bfloat162 hi = __floats2bfloat162_rn(v.z, v.w);
                    uint2 packed;
                    packed.x = *reinterpret_cast<std::uint32_t*>(&lo);
                    packed.y = *reinterpret_cast<std::uint32_t*>(&hi);
                    *reinterpret_cast<uint2*>(out_bf16 + o) = packed;
                }
            }
        }
        __syncthreads(); // s_h, s_w, s_k, s_mask, s_last are rewritten by the next item
    }
}

// Resident CTAs per SM of each kernel (per T for gemm1's dynamic shared memory), per device.
struct Capacity {
    int sms          = 0;
    int g1[MAXT + 1] = {};
    int g2           = 0;
};

const Capacity& capacity() {
    constexpr int kMaxDevices = 16;
    static Capacity table[kMaxDevices];
    static bool known[kMaxDevices];
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= kMaxDevices)
        throw std::runtime_error("moe_slots_nvfp4: device index");
    if (!known[device]) {
        Capacity c;
        CUDA_CHECK(cudaDeviceGetAttribute(&c.sms, cudaDevAttrMultiProcessorCount, device));
        for (int t = 1; t <= MAXT; ++t) {
            const auto smem =
                static_cast<std::size_t>(t) * ((H / 8) * sizeof(int2) + HB * sizeof(float));
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &c.g1[t], moe_slots_gemm1_kernel, G1_THREADS, smem));
        }
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&c.g2, moe_slots_gemm2_kernel,
                                                                 G2_THREADS, 0));
        table[device] = c;
        known[device] = true;
    }
    return table[device];
}

std::size_t align256(std::size_t v) { return (v + 255) / 256 * 256; }

} // namespace

std::size_t moe_slots_workspace_layout(std::int32_t tokens, std::size_t offsets[4]) {
    const auto t            = static_cast<std::size_t>(tokens);
    const std::size_t pairs = TOPK * t * t; // (first position, token)
    std::size_t at          = 0;
    offsets[0]              = at;
    at += align256(pairs * (I / 8) * sizeof(std::uint32_t));
    offsets[1] = at;
    at += align256(pairs * IB * sizeof(float));
    offsets[2] = at;
    at += align256(t * TOPK * H * sizeof(float));
    offsets[3] = at;
    at += align256(G2_TILES * sizeof(std::uint32_t));
    return at;
}

void moe_slots_launch(const Tensor& x, const Tensor& routed_ids, const Tensor& routed_weights,
                      const MoeSlotsLayer& layer, const MoeSlotPool& pool, MoeSlotsOutput mode,
                      Tensor& out, const MoeSlotsWorkspace& workspace, cudaStream_t stream) {
    const int T = x.ne[1];
    const PoolArgs p{static_cast<const std::uint8_t*>(pool.gate_up_codes.base),
                     static_cast<const std::uint8_t*>(pool.gate_up_scales.base),
                     static_cast<const std::uint8_t*>(pool.down_codes.base),
                     static_cast<const std::uint8_t*>(pool.down_scales.base),
                     pool.gate_up_codes.stride,
                     pool.gate_up_scales.stride,
                     pool.down_codes.stride,
                     pool.down_scales.stride,
                     pool.slots};
    const LayerArgs l{static_cast<const int*>(layer.slot_table.data),
                      static_cast<const float*>(layer.gate_up_divisors.data),
                      static_cast<const float*>(layer.down_divisors.data),
                      static_cast<int>(layer.slot_table.numel()),
                      layer.gate_up_input_divisor,
                      layer.down_input_divisor};
    const int* ids  = static_cast<const int*>(routed_ids.data);
    const auto smem = static_cast<std::size_t>(T) * ((H / 8) * sizeof(int2) + HB * sizeof(float));
    float* out_f32  = out.dtype == DType::FP32 ? static_cast<float*>(out.data) : nullptr;
    __nv_bfloat16* out_bf16 =
        out.dtype == DType::BF16 ? static_cast<__nv_bfloat16*>(out.data) : nullptr;
    const Capacity& cap = capacity();
    const int g1 =
        std::min(T * TOPK * G1_BLOCKS_PER_POS, cap.sms * std::min(G1_GRID_PER_SM, cap.g1[T]));
    const int g2 = std::min(T * TOPK * G2_TILES, cap.sms * std::min(G2_GRID_PER_SM, cap.g2));
    CUDA_CHECK(pdl::launch_dependent(pdl::LaunchConfig{dim3(g1), dim3(G1_THREADS), smem, stream},
                                     moe_slots_gemm1_kernel,
                                     static_cast<const __nv_bfloat16*>(x.data), T, ids, l, p,
                                     workspace.h_codes, workspace.h_scales, workspace.tile_done));
    CUDA_CHECK(pdl::launch_dependent(
        pdl::LaunchConfig{dim3(g2), dim3(G2_THREADS), 0, stream}, moe_slots_gemm2_kernel, T, ids,
        static_cast<const float*>(routed_weights.data), l, p,
        static_cast<const std::uint32_t*>(workspace.h_codes),
        static_cast<const float*>(workspace.h_scales), workspace.partial, workspace.tile_done,
        out_f32, out_bf16, mode == MoeSlotsOutput::Accumulate ? 1 : 0));
}

} // namespace ninfer::ops::detail
