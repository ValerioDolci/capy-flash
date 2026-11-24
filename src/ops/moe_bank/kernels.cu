// Implements: include/ninfer/ops/moe_bank.h
// Two kernels, one warp per output row. The first runs per (token, position) in the bank and per
// group of expert channels: the gate and up dots of channel i (half a warp each), their BF16
// roundings, the capped SwiGLU in FP32 and h rounded to BF16 into the workspace. The second runs
// per token and per group of hidden rows: the down dots of the token's positions in the bank (a
// group of lanes each), their BF16 roundings and the weighted FP32 sum, then the store or the one
// accumulating addition. Weights are decoded exactly in registers (NVFP4: 16 E2M1 codes times their
// E4M3FN scale, the divisor applied once to the dot).
#include "ops/moe_bank/launch.h"

#include "core/device.h"
#include "ops/common/pdl_launch.cuh" // CUDA_CHECK
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/launcher/kernel_attr_once.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

constexpr int kWarps   = 8;
constexpr int kThreads = kWarps * 32;

// Exact decodes: E2M1 pairs (low nibble = even element) and E4M3FN scales widen without rounding.
__device__ __forceinline__ float2 e2m1x2(std::uint32_t byte) {
    __nv_fp4x2_e2m1 value;
    value.__x = static_cast<__nv_fp4x2_storage_t>(byte);
    return static_cast<float2>(value);
}

__device__ __forceinline__ float e4m3fn(std::uint8_t c) {
    __nv_fp8_e4m3 value;
    value.__x = static_cast<__nv_fp8_storage_t>(c);
    return static_cast<float>(value);
}

// UE8M0 block scale 2^(s - 127), exactly (s = 0 is the subnormal 2^-127; 255 is excluded).
__device__ __forceinline__ float ue8m0(std::uint32_t s) {
    return __uint_as_float(s != 0U ? s << 23 : 0x00400000U);
}

// The exact values of the two MXFP8 codes in bytes `pair` (low byte first) times `scale`.
__device__ __forceinline__ float2 mxfp8x2(std::uint32_t pair, float scale) {
    return make_float2(e4m3fn(static_cast<std::uint8_t>(pair & 0xFFU)) * scale,
                       e4m3fn(static_cast<std::uint8_t>((pair >> 8) & 0xFFU)) * scale);
}

__device__ __forceinline__ float bf16_round(float v) {
    return __bfloat162float(__float2bfloat16_rn(v));
}

// Lane-partial FP32 dot of parent row `row` of `plane` with the BF16 vector `v` (length
// plane.columns) over kLanes lanes (lane < kLanes): NVFP4 16-column groups, MXFP8 and BF16
// 8-column pieces dealt round robin; the caller sums the lanes. NVFP4 rows return the dot before
// the divisor.
template <int kLanes = 32>
__device__ float row_dot(const MoeBankPlane& plane, std::int64_t row,
                         const __nv_bfloat16* __restrict__ v, int lane) {
    const int columns = plane.columns;
    float acc         = 0.0F;
    if (plane.nvfp4) {
        const auto* codes          = plane.data + row * (columns / 2);
        const auto* scales         = plane.data + plane.scale_offset;
        const std::int64_t k_tiles = columns / 64;
        const std::int64_t inner   = row % 128;
        const std::int64_t tile_base =
            (row / 128) * k_tiles * 512 + (inner % 32) * 16 + (inner / 32) * 4;
        // A lane share of a few groups keeps several groups' loads in flight.
#pragma unroll(kLanes < 32 ? 3 : 1)
        for (int g = lane; g < columns / 16; g += kLanes) {
            const uint2 packed = *reinterpret_cast<const uint2*>(codes + g * 8);
            const std::uint8_t scale =
                static_cast<std::uint8_t>(scales[tile_base + (g / 4) * 512 + g % 4]);
            const auto* vg = reinterpret_cast<const uint4*>(v + g * 16);
            const uint4 v0 = vg[0], v1 = vg[1];
            const std::uint32_t vw[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
            float block               = 0.0F;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const std::uint32_t word = c < 4 ? packed.x : packed.y;
                const float2 w           = e2m1x2((word >> ((c % 4) * 8)) & 0xFFU);
                const float2 a =
                    __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vw[c]));
                block = fmaf(w.x, a.x, block);
                block = fmaf(w.y, a.y, block);
            }
            acc = fmaf(block, e4m3fn(scale), acc);
        }
    } else if (plane.mxfp8) {
        // The BF16 loop below with the weights decoded from their codes: the same values in the
        // same order, so an MXFP8 bank gives the bits of a BF16 bank holding its values.
        const auto* codes  = plane.data + row * columns;
        const auto* scales = plane.data + plane.scale_offset + row * (columns / 32);
#pragma unroll(kLanes < 32 ? 3 : 1)
        for (int c = lane * 8; c < columns; c += kLanes * 8) {
            const uint2 wq               = *reinterpret_cast<const uint2*>(codes + c);
            const float scale            = ue8m0(static_cast<std::uint8_t>(scales[c / 32]));
            const uint4 vq               = *reinterpret_cast<const uint4*>(v + c);
            const auto* vp               = reinterpret_cast<const __nv_bfloat162*>(&vq);
            const std::uint32_t pairs[4] = {wq.x & 0xFFFFU, wq.x >> 16, wq.y & 0xFFFFU, wq.y >> 16};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 a = mxfp8x2(pairs[i], scale);
                const float2 b = __bfloat1622float2(vp[i]);
                acc            = fmaf(a.x, b.x, fmaf(a.y, b.y, acc));
            }
        }
    } else {
        const auto* w = reinterpret_cast<const __nv_bfloat16*>(plane.data) + row * columns;
        for (int c = lane * 8; c < columns; c += kLanes * 8) {
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

__device__ __forceinline__ std::int64_t scale_tile_base(const MoeBankPlane& plane,
                                                        std::int64_t row) {
    const std::int64_t inner = row % 128;
    return static_cast<std::int64_t>(plane.scale_offset) +
           (row / 128) * (plane.columns / 64) * 512 + (inner % 32) * 16 + (inner / 32) * 4;
}

__device__ __forceinline__ float row_divisor(const MoeBankPlane& plane, std::int64_t row) {
    if (!plane.nvfp4) { return 1.0F; }
    const auto* words = reinterpret_cast<const float*>(plane.data + plane.divisor_offset);
    return words[row / plane.rows_per_divisor];
}

__device__ __forceinline__ float softcap(float v, float cap) { return cap * tanhf(v / cap); }

__device__ __forceinline__ __nv_bfloat16 capped_swiglu(float g, float a, CappedSwiglu activation) {
    const float sig = 1.0F / (1.0F + expf(-g));
    return __float2bfloat16_rn(softcap(g, activation.gate_cap) * sig *
                               softcap(a, activation.up_cap));
}

constexpr int kGateUpLanes = 16;

__global__ void __launch_bounds__(kThreads)
    moe_bank_gate_up_kernel(const __nv_bfloat16* __restrict__ x,
                            const std::int32_t* __restrict__ ids, MoeBankPlane gate_up,
                            MoeBankShape shape, CappedSwiglu activation,
                            __nv_bfloat16* __restrict__ hidden_out) {
    pdl_enter();
    const int assignment = static_cast<int>(blockIdx.x); // t * k + j
    const int b          = ids[assignment] - shape.first_expert;
    if (ids[assignment] < 0 || b < 0 || b >= shape.experts) { return; }
    const int warp    = static_cast<int>(threadIdx.x) >> 5;
    const int lane    = static_cast<int>(threadIdx.x) & 31;
    const int channel = static_cast<int>(blockIdx.y) * kWarps + warp;
    if (channel >= shape.intermediate) { return; }
    const int token        = assignment / shape.top_k;
    const __nv_bfloat16* u = x + static_cast<std::int64_t>(token) * shape.hidden;
    // Lanes 0-15 take the gate row, lanes 16-31 the up row, each half its share of the dot and its
    // pairwise sum (l0 + l1) + (l2 + l3) ... over the 16 lanes.
    const int half = lane / kGateUpLanes, part = lane % kGateUpLanes;
    const std::int64_t row = gate_up.row_begin +
                             static_cast<std::int64_t>(b) * 2 * shape.intermediate +
                             (half != 0 ? shape.intermediate : 0) + channel;
    float dot = row_dot<kGateUpLanes>(gate_up, row, u, part);
#pragma unroll
    for (int offset = 1; offset < kGateUpLanes; offset <<= 1) {
        dot += __shfl_xor_sync(kFullWarpMask, dot, offset);
    }
    const float v = bf16_round(gate_up.nvfp4 ? dot / row_divisor(gate_up, row) : dot);
    const float a = __shfl_sync(kFullWarpMask, v, kGateUpLanes);
    if (lane == 0) {
        hidden_out[static_cast<std::int64_t>(assignment) * shape.intermediate + channel] =
            capped_swiglu(v, a, activation);
    }
}

// Wide calls (more than kWideMinTokens - 1 tokens): grouped by expert on tensor cores. NVFP4 and
// MXFP8 planes run wide_tile when the rows fit its tiles (same arithmetic, larger tiles); the
// others these 64 x 64 kernels. A block owns one expert of the bank and
// kWideChannels channels, i.e. a 64-row tile of its gate rows
// (first 32) and the matching up rows (last 32), and runs every (token, position) routed to the
// expert through it in tiles of 64: weights decoded to BF16 exactly (an E2M1 code times its
// E4M3FN scale has at most six significant bits), BF16 MMA with FP32 accumulation over H, the
// NVFP4 divisor applied to the accumulated dot. The accumulation order is the tensor cores', not
// the per-lane chain of the narrow kernels: wide and narrow calls do not give the same bits.
constexpr int kWideMinTokens = 48;
constexpr int kWideThreads   = 256;
constexpr int kWideChannels  = 32;
constexpr int kWideRows      = 64; // weight rows per tile
constexpr int kWideCols      = 64; // (token, position) columns per tile
constexpr int kWideK         = 64;
constexpr int kWideStride    = kWideK + 8; // BF16 elements per shared row (144 B: ldmatrix-aligned)

// The pass's plan: for each expert b of the bank, its columns of [first_column, end_column) in
// increasing order, at sorted[offsets[b] .. offsets[b + 1]) (offsets [E' + 1], then sorted). Block
// b counts the in-bank columns of lower experts (strided, coalesced), then writes its own columns
// in order over windows of kPlanThreads * kPlanPerThread ids held in registers.
constexpr int kPlanThreads   = 1024;
constexpr int kPlanPerThread = 8;

__global__ void __launch_bounds__(kPlanThreads)
    moe_bank_plan_kernel(const std::int32_t* __restrict__ ids, MoeBankShape shape, int first_column,
                         int end_column, std::int32_t* __restrict__ plan) {
    constexpr int kWarpsPerBlock = kPlanThreads / 32;
    __shared__ int s_sum[kWarpsPerBlock];
    __shared__ int s_scan[kWarpsPerBlock];
    const int b    = static_cast<int>(blockIdx.x);
    const int t    = static_cast<int>(threadIdx.x);
    const int warp = t >> 5, lane = t & 31;
    const int expert = shape.first_expert + b;
    const auto local = [&](int id) {
        const int l = id - shape.first_expert;
        return id >= 0 && l >= 0 && l < shape.experts ? l : -1;
    };
    int less = 0;
    for (int i = first_column + t; i < end_column; i += kPlanThreads) {
        const int l = local(ids[i]);
        less += l >= 0 && l < b ? 1 : 0;
    }
    less = warp_sum(less);
    if (lane == 0) { s_sum[warp] = less; }
    __syncthreads();
    int before = 0;
    for (int w = 0; w < kWarpsPerBlock; ++w) { before += s_sum[w]; }
    if (t == 0) { plan[b] = before; }
    std::int32_t* sorted = plan + shape.experts + 1;
    int written          = 0;
    for (int base = first_column; base < end_column; base += kPlanThreads * kPlanPerThread) {
        const int lo = base + t * kPlanPerThread;
        int hit[kPlanPerThread];
        int mine = 0;
#pragma unroll
        for (int j = 0; j < kPlanPerThread; ++j) {
            hit[j] = lo + j < end_column && ids[lo + j] == expert ? 1 : 0;
            mine += hit[j];
        }
        int scan = mine;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int v = __shfl_up_sync(kFullWarpMask, scan, o);
            if (lane >= o) { scan += v; }
        }
        __syncthreads(); // the previous window's totals are read
        if (lane == 31) { s_scan[warp] = scan; }
        __syncthreads();
        int warp_before = 0, total = 0;
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            warp_before += w < warp ? s_scan[w] : 0;
            total += s_scan[w];
        }
        int at = before + written + warp_before + scan - mine;
#pragma unroll
        for (int j = 0; j < kPlanPerThread; ++j) {
            if (hit[j] != 0) { sorted[at++] = lo + j; }
        }
        written += total;
    }
    if (t == 0 && b == shape.experts - 1) { plan[shape.experts] = before + written; }
}

// One thread's share of a 64 x 64 weight tile (row threadIdx/4, 16 columns): the raw words,
// fetched from global memory ahead of their use, then decoded to BF16 into shared memory.
struct WeightFetch {
    uint4 raw[2];
};

__device__ __forceinline__ std::int64_t tile_row_index() {
    return static_cast<std::int64_t>(threadIdx.x) >> 2;
}

template <class RowOf>
__device__ __forceinline__ WeightFetch fetch_weight(const MoeBankPlane& plane, RowOf row_of,
                                                    int k0) {
    const int q            = static_cast<int>(threadIdx.x) & 3;
    const std::int64_t row = row_of(static_cast<int>(tile_row_index()));
    const int columns      = plane.columns;
    WeightFetch f;
    if (plane.nvfp4) {
        const int g = k0 / 16 + q;
        const uint2 packed =
            *reinterpret_cast<const uint2*>(plane.data + row * (columns / 2) + g * 8);
        const std::int64_t inner = row % 128;
        const auto scale         = static_cast<std::uint8_t>(
            plane.data[plane.scale_offset + ((row / 128) * (columns / 64) + g / 4) * 512 +
                       (inner % 32) * 16 + (inner / 32) * 4 + g % 4]);
        f.raw[0] = make_uint4(packed.x, packed.y, scale, 0U);
    } else if (plane.mxfp8) {
        const int c = k0 + q * 16;
        f.raw[0]    = *reinterpret_cast<const uint4*>(plane.data + row * columns + c);
        f.raw[1].x  = static_cast<std::uint8_t>(
            plane.data[plane.scale_offset + row * (columns / 32) + c / 32]);
    } else {
        const auto* src = reinterpret_cast<const uint4*>(
            reinterpret_cast<const __nv_bfloat16*>(plane.data) + row * columns + k0 + q * 16);
        f.raw[0] = src[0];
        f.raw[1] = src[1];
    }
    return f;
}

__device__ __forceinline__ void store_weight(const WeightFetch& f, const MoeBankPlane& plane,
                                             __nv_bfloat16* tile) {
    const int q = static_cast<int>(threadIdx.x) & 3;
    uint4 out[2];
    if (plane.mxfp8) {
        const float scale        = ue8m0(f.raw[1].x);
        auto* pairs              = reinterpret_cast<__nv_bfloat162*>(out);
        const std::uint32_t w[4] = {f.raw[0].x, f.raw[0].y, f.raw[0].z, f.raw[0].w};
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const float2 v = mxfp8x2(w[c / 2] >> ((c % 2) * 16), scale);
            pairs[c]       = __floats2bfloat162_rn(v.x, v.y);
        }
    } else if (plane.nvfp4) {
        const float scale = e4m3fn(static_cast<std::uint8_t>(f.raw[0].z));
        auto* pairs       = reinterpret_cast<__nv_bfloat162*>(out);
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const std::uint32_t word = c < 4 ? f.raw[0].x : f.raw[0].y;
            const float2 w           = e2m1x2((word >> ((c % 4) * 8)) & 0xFFU);
            pairs[c]                 = __floats2bfloat162_rn(w.x * scale, w.y * scale);
        }
    } else {
        out[0] = f.raw[0];
        out[1] = f.raw[1];
    }
    auto* dst = reinterpret_cast<uint4*>(tile + tile_row_index() * kWideStride + q * 16);
    dst[0]    = out[0];
    dst[1]    = out[1];
}

// One thread's share of a 64-column activation tile: column threadIdx/4, 16 values of the
// column's BF16 vector `column(c)` (null: an empty column, zeros).
template <class Column>
__device__ __forceinline__ void fetch_columns(Column column, int k0, uint4 (&v)[2]) {
    const int c = static_cast<int>(threadIdx.x) >> 2, q = static_cast<int>(threadIdx.x) & 3;
    const __nv_bfloat16* src = column(c);
    if (src != nullptr) {
        const auto* p = reinterpret_cast<const uint4*>(src + k0 + q * 16);
        v[0]          = p[0];
        v[1]          = p[1];
    } else {
        v[0] = make_uint4(0U, 0U, 0U, 0U);
        v[1] = v[0];
    }
}

__device__ __forceinline__ void store_columns(const uint4 (&v)[2], __nv_bfloat16* tile) {
    const int c = static_cast<int>(threadIdx.x) >> 2, q = static_cast<int>(threadIdx.x) & 3;
    auto* dst = reinterpret_cast<uint4*>(tile + c * kWideStride + q * 16);
    dst[0]    = v[0];
    dst[1]    = v[1];
}

// 64 x 64 x 64 tile product: warp (wm, wn) accumulates rows 16 wm.. and columns 32 wn.. .
__device__ __forceinline__ void mma_tile(const __nv_bfloat16* a_tile, const __nv_bfloat16* b_tile,
                                         float (&acc)[4][4]) {
    const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
    const int wm = warp & 3, wn = warp >> 2;
#pragma unroll
    for (int kk = 0; kk < kWideK; kk += 16) {
        unsigned a[4];
        const __nv_bfloat16* a_ptr =
            a_tile + (wm * 16 + (lane & 15)) * kWideStride + kk + (lane >> 4) * 8;
        ldmatrix_x4(a[0], a[1], a[2], a[3], static_cast<unsigned>(__cvta_generic_to_shared(a_ptr)));
#pragma unroll
        for (int nt = 0; nt < 4; nt += 2) {
            unsigned b[4];
            const __nv_bfloat16* b_ptr =
                b_tile + (wn * 32 + nt * 8 + (lane >> 4) * 8 + (lane & 7)) * kWideStride + kk +
                ((lane >> 3) & 1) * 8;
            ldmatrix_x4(b[0], b[1], b[2], b[3],
                        static_cast<unsigned>(__cvta_generic_to_shared(b_ptr)));
            mma_bf16(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], a[0], a[1], a[2], a[3], b[0],
                     b[1]);
            mma_bf16(acc[nt + 1][0], acc[nt + 1][1], acc[nt + 1][2], acc[nt + 1][3], a[0], a[1],
                     a[2], a[3], b[2], b[3]);
        }
    }
}

// acc += W[rows, 0:K) . V[0:K, columns] for a 64 x 64 tile, the next K step fetched into
// registers while the current one is multiplied. Ends with a barrier (shared memory free).
template <class RowOf, class Column>
__device__ __forceinline__ void wide_tile_product(const MoeBankPlane& plane, RowOf row_of,
                                                  Column column, int K, __nv_bfloat16* s_a,
                                                  __nv_bfloat16* s_b, float (&acc)[4][4]) {
    WeightFetch w = fetch_weight(plane, row_of, 0);
    uint4 v[2];
    fetch_columns(column, 0, v);
    store_weight(w, plane, s_a);
    store_columns(v, s_b);
    __syncthreads();
    for (int k0 = 0; k0 < K; k0 += kWideK) {
        const bool more = k0 + kWideK < K;
        if (more) {
            w = fetch_weight(plane, row_of, k0 + kWideK);
            fetch_columns(column, k0 + kWideK, v);
        }
        mma_tile(s_a, s_b, acc);
        __syncthreads();
        if (more) {
            store_weight(w, plane, s_a);
            store_columns(v, s_b);
            __syncthreads();
        }
    }
}

// Shared memory of the wide kernels: the two operand tiles, reused by the epilogue's 64 x 64
// FP32 outputs.
constexpr int kWideTileBytes = kWideRows * kWideStride * 2;
constexpr int kWideOutBytes  = kWideRows * (kWideCols + 1) * 4;

// NVFP4 wide products stream through a kWideStages-deep cp.async pipeline in shared memory: per K
// step the raw codes (32 bytes a row) and the scale word (the k-tile's four E4M3 scales) of the 64
// weight rows, and the 64 activation columns as BF16. A step's codes are decoded to BF16 (exactly,
// as store_weight does) into one weight tile, then the same MMAs as wide_tile_product run in the
// same K order: the same bits.
constexpr int kWideStages    = 3;
constexpr int kRawCodeBytes  = kWideRows * (kWideK / 2);
constexpr int kRawScaleBytes = kWideRows * 4;
constexpr int kStagedB       = kWideCols * kWideStride; // BF16 elements of one activation stage
constexpr int kPipelineBytes =
    kWideTileBytes + kWideStages * (kStagedB * 2 + kRawCodeBytes + kRawScaleBytes);
constexpr int kWideSharedBytes = kPipelineBytes > kWideOutBytes ? kPipelineBytes : kWideOutBytes;
static_assert(kWideOutBytes <= 2 * kWideTileBytes, "the epilogue fits the synchronous tiles");

__host__ __device__ __forceinline__ bool wide_pipelined(const MoeBankPlane& plane, int K) {
    return plane.nvfp4 && K % kWideK == 0 && plane.scale_offset % 4 == 0;
}

template <class RowOf, class Column>
__device__ __forceinline__ void
wide_tile_product_pipelined(const MoeBankPlane& plane, RowOf row_of, Column column, int K,
                            unsigned char* smem, float (&acc)[4][4]) {
    auto* a_tile   = reinterpret_cast<__nv_bfloat16*>(smem);
    auto* b_stages = a_tile + kWideRows * kWideStride;
    auto* codes    = reinterpret_cast<unsigned char*>(b_stages + kWideStages * kStagedB);
    auto* scales   = codes + kWideStages * kRawCodeBytes;
    const int t    = static_cast<int>(threadIdx.x);
    // This thread's copies, the same at every K step: two 16-byte pieces of activation columns,
    // then (threads 0-127) half a row of codes or (threads 128-191) a row's scale word.
    const __nv_bfloat16* b_src[2];
    int b_dst[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int piece = t + kWideThreads * j, c = piece >> 3, part = piece & 7;
        const __nv_bfloat16* src = column(c);
        b_src[j]                 = src != nullptr ? src + part * 8 : nullptr;
        b_dst[j]                 = c * kWideStride + part * 8;
    }
    const std::byte* w_src = nullptr;
    int w_dst              = 0;
    if (t < 2 * kWideRows) {
        const int r = t >> 1, half = t & 1;
        w_src = plane.data + row_of(r) * (plane.columns / 2) + half * 16;
        w_dst = r * (kWideK / 2) + half * 16;
    } else if (t < 3 * kWideRows) {
        const int r = t - 2 * kWideRows;
        w_src       = plane.data + scale_tile_base(plane, row_of(r));
        w_dst       = r * 4;
    }
    const int steps  = K / kWideK;
    const auto issue = [&](int step) {
        const int s = step % kWideStages, k0 = step * kWideK;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            void* dst = b_stages + s * kStagedB + b_dst[j];
            if (b_src[j] != nullptr) {
                cp_async<16>(dst, b_src[j] + k0);
            } else {
                cp_async_zfill<16>(dst, plane.data, 0);
            }
        }
        if (t < 2 * kWideRows) {
            cp_async<16>(codes + s * kRawCodeBytes + w_dst, w_src + k0 / 2);
        } else if (t < 3 * kWideRows) {
            cp_async<4>(scales + s * kRawScaleBytes + w_dst, w_src + (k0 / kWideK) * 512);
        }
    };
#pragma unroll
    for (int s = 0; s < kWideStages - 1; ++s) {
        if (s < steps) { issue(s); }
        cp_commit();
    }
    for (int step = 0; step < steps; ++step) {
        const int s = step % kWideStages;
        cp_wait<kWideStages - 2>();
        __syncthreads(); // stage `step` landed; every warp is past the previous step's MMAs
        if (step + kWideStages - 1 < steps) { issue(step + kWideStages - 1); }
        cp_commit();
        {
            const int r = t >> 2, q = t & 3;
            const uint2 packed = *reinterpret_cast<const uint2*>(codes + s * kRawCodeBytes +
                                                                 r * (kWideK / 2) + q * 8);
            const float scale  = e4m3fn(scales[s * kRawScaleBytes + r * 4 + q]);
            uint4 out[2];
            auto* pairs = reinterpret_cast<__nv_bfloat162*>(out);
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const std::uint32_t word = c < 4 ? packed.x : packed.y;
                const float2 w           = e2m1x2((word >> ((c % 4) * 8)) & 0xFFU);
                pairs[c]                 = __floats2bfloat162_rn(w.x * scale, w.y * scale);
            }
            auto* dst = reinterpret_cast<uint4*>(a_tile + r * kWideStride + q * 16);
            dst[0]    = out[0];
            dst[1]    = out[1];
        }
        __syncthreads();
        mma_tile(a_tile, b_stages + s * kStagedB, acc);
    }
    cp_wait<0>();
    __syncthreads();
}

__global__ void __launch_bounds__(kWideThreads)
    moe_bank_gate_up_wide_kernel(const __nv_bfloat16* __restrict__ x, MoeBankPlane gate_up,
                                 MoeBankShape shape, CappedSwiglu activation, int first_column,
                                 const std::int32_t* __restrict__ plan,
                                 __nv_bfloat16* __restrict__ hidden_out) {
    extern __shared__ __align__(16) unsigned char s_raw[];
    __shared__ int s_cols[kWideCols];
    auto* s_a     = reinterpret_cast<__nv_bfloat16*>(s_raw);
    auto* s_b     = reinterpret_cast<__nv_bfloat16*>(s_raw + kWideTileBytes);
    auto* s_out   = reinterpret_cast<float (*)[kWideCols + 1]>(s_raw);
    const int b   = static_cast<int>(blockIdx.x);
    const int ch0 = static_cast<int>(blockIdx.y) * kWideChannels;
    const int H = shape.hidden, F = shape.intermediate;
    const std::int64_t base = gate_up.row_begin + static_cast<std::int64_t>(b) * 2 * F;
    const auto row_of       = [&](int r) {
        return base + (r < kWideChannels ? ch0 + r : F + ch0 + r - kWideChannels);
    };
    const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
    const int wm = warp & 3, wn = warp >> 2;
    const std::int32_t* list = plan + shape.experts + 1 + plan[b];
    const int count          = plan[b + 1] - plan[b];
    {
        for (int col0 = 0; col0 < count; col0 += kWideCols) {
            __syncthreads(); // the previous tile's columns are no longer read
            if (threadIdx.x < kWideCols) {
                const int c = static_cast<int>(threadIdx.x);
                s_cols[c]   = col0 + c < count ? list[col0 + c] : -1;
            }
            __syncthreads();
            const auto column = [&](int c) -> const __nv_bfloat16* {
                if (s_cols[c] < 0) { return nullptr; }
                return x + static_cast<std::int64_t>(s_cols[c] / shape.top_k) * H;
            };
            float acc[4][4] = {};
            wide_tile_product(gate_up, row_of, column, H, s_a, s_b, acc);
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int r = wm * 16 + (lane >> 2) + (e >= 2 ? 8 : 0);
                    const int c = wn * 32 + nt * 8 + (lane & 3) * 2 + (e & 1);
                    float dot   = acc[nt][e];
                    if (gate_up.nvfp4) { dot = dot / row_divisor(gate_up, row_of(r)); }
                    s_out[r][c] = bf16_round(dot);
                }
            }
            __syncthreads();
            for (int i = static_cast<int>(threadIdx.x); i < kWideChannels * kWideCols;
                 i += kWideThreads) {
                const int ch = i % kWideChannels, c = i / kWideChannels;
                if (col0 + c < count) {
                    hidden_out[static_cast<std::int64_t>(s_cols[c] - first_column) * F + ch0 + ch] =
                        capped_swiglu(s_out[ch][c], s_out[ch + kWideChannels][c], activation);
                }
            }
            __syncthreads();
        }
    }
}

// The BF16 pair of NVFP4 byte `byte` (low nibble first) times `scale`, exactly as store_weight.
__device__ __forceinline__ unsigned nvfp4_bf16x2(std::uint32_t byte, float scale) {
    const float2 w               = e2m1x2(byte);
    const __nv_bfloat162 product = __floats2bfloat162_rn(w.x * scale, w.y * scale);
    return *reinterpret_cast<const unsigned*>(&product);
}

// Wide calls on NVFP4 and MXFP8 planes (gate/up and down). A block owns one expert, kTileRows rows
// and one tile of up to kTileCols of the expert's columns; the blocks of an expert with many
// columns split its tiles. Warp w multiplies rows 16w.. 16w + 15 of the block by every column of
// the tile. Gate/up: the block's rows are the gate and up rows of 64 channels, rows 0-7 of a warp's
// m16 tile the gate rows of channels 8w.. 8w + 7 and rows 8-15 their up rows. Down: 128
// consecutive hidden rows. Per K step the raw codes, (NVFP4) the scale words and the input columns
// stream through a cp.async pipeline (MXFP8 scales are read ahead into registers); each warp
// decodes the A fragments of its own rows in registers (exactly, as store_weight), then the same
// mma.m16n8k16 run in the same K order as the 64 x 64 tiles, followed by the same divisor,
// roundings and capped SwiGLU: the same bits.
constexpr int kTileRows      = 128;
constexpr int kTileChannels  = kTileRows / 2;
constexpr int kTileCols      = 128;
constexpr int kTileStride    = kWideK + 8; // BF16 elements per staged column (ldmatrix-aligned)
constexpr int kTileColBytes  = kTileCols * kTileStride * 2;
constexpr int kTileDotStride = kTileRows + 1; // FP32 dots per staged column

// The pipeline of one format: NVFP4 4 stages of codes (32 bytes a row), scale words and
// columns; MXFP8 3 stages of codes (64 bytes a row) and columns.
template <bool kMx>
struct TilePipe {
    static constexpr int kStages     = kMx ? 3 : 4;
    static constexpr int kRowBytes   = kMx ? kWideK : kWideK / 2;
    static constexpr int kCodeBytes  = kTileRows * kRowBytes;
    static constexpr int kScaleBytes = kMx ? 0 : kTileRows * 4;
    static constexpr int kStageBytes = kCodeBytes + kScaleBytes + kTileColBytes;
    static constexpr int kBytes      = kStages * kStageBytes;
    static_assert((kTileCols * kTileDotStride + kTileRows) * 4 <= kBytes,
                  "the epilogue fits the pipeline");
    static_assert(kCodeBytes % (16 * kWideThreads) == 0, "whole 16-byte pieces per thread");
};

static_assert(kWideThreads / 32 * 16 == kTileRows, "eight warps of 16 rows");

// Whether a wide call's projection runs on wide_tile (rows = 2F for gate/up, H for down).
__host__ __device__ __forceinline__ bool wide_tiles(const MoeBankPlane& plane, int K, int rows) {
    return (plane.mxfp8 || wide_pipelined(plane, K)) && rows % kTileRows == 0;
}

// The BF16 pair of MXFP8 bytes `pair` (low byte first) times `scale`, exactly as store_weight.
__device__ __forceinline__ unsigned mxfp8_bf16x2(std::uint32_t pair, float scale) {
    const float2 v               = mxfp8x2(pair, scale);
    const __nv_bfloat162 product = __floats2bfloat162_rn(v.x, v.y);
    return *reinterpret_cast<const unsigned*>(&product);
}

template <bool kDown, bool kMx>
__device__ __forceinline__ void
wide_tile(const __nv_bfloat16* __restrict__ in, const MoeBankPlane& plane,
          const MoeBankShape& shape, CappedSwiglu activation, int first_column,
          const std::int32_t* __restrict__ plan, __nv_bfloat16* __restrict__ out) {
    extern __shared__ __align__(16) unsigned char s_raw[];
    __shared__ int s_cols[kTileCols];
    __shared__ int s_sums[kWideThreads / 32];
    __shared__ int s_tile[3]; // expert, first column of the tile, valid columns
    const int t = static_cast<int>(threadIdx.x), warp = t >> 5, lane = t & 31;
    // This block's tile: the blockIdx.x-th of the experts' tiles in expert order, the experts'
    // tile counts scanned kWideThreads at a time.
    {
        const int tile = static_cast<int>(blockIdx.x);
        if (t == 0) { s_tile[0] = -1; }
        int start = 0;
        for (int e0 = 0; e0 < shape.experts; e0 += kWideThreads) {
            const int e     = e0 + t;
            const int count = e < shape.experts ? plan[e + 1] - plan[e] : 0;
            const int tiles = (count + kTileCols - 1) / kTileCols;
            int end         = tiles;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_up_sync(kFullWarpMask, end, o);
                if (lane >= o) { end += v; }
            }
            if (lane == 31) { s_sums[warp] = end; }
            __syncthreads();
            int total = start;
#pragma unroll
            for (int w = 0; w < kWideThreads / 32; ++w) {
                end += w < warp ? s_sums[w] : 0;
                total += s_sums[w];
            }
            end += start;
            if (tile >= end - tiles && tile < end) {
                s_tile[0] = e;
                s_tile[1] = (tile - end + tiles) * kTileCols;
                s_tile[2] = min(kTileCols, count - s_tile[1]);
            }
            start = total;
            __syncthreads();
            if (tile < start) { break; }
        }
    }
    const int b = s_tile[0];
    if (b < 0) { return; }
    const int col0 = s_tile[1], valid = s_tile[2];
    if (t < valid) { s_cols[t] = plan[shape.experts + 1 + plan[b] + col0 + t]; }
    const int H = shape.hidden, F = shape.intermediate, K = kDown ? F : H;
    const int r0 = static_cast<int>(blockIdx.y) * (kDown ? kTileRows : kTileChannels);
    const std::int64_t base =
        plane.row_begin + static_cast<std::int64_t>(b) * (kDown ? H : 2 * F) + r0;
    const auto row_of = [&](int r) -> std::int64_t {
        if constexpr (kDown) { return base + r; }
        return base + ((r & 15) < 8 ? 0 : F) + 8 * (r >> 4) + (r & 7);
    };
    __syncthreads();
    using Pipe = TilePipe<kMx>;
    // This thread's copies, the same at every K step: kCodePieces 16-byte pieces of code rows,
    // (NVFP4, t < kTileRows) a row's scale word, and the pieces of the valid input columns among
    // four.
    constexpr int kCodePieces = Pipe::kCodeBytes / (16 * kWideThreads);
    constexpr int kRowPieces  = Pipe::kRowBytes / 16;
    const std::byte* code_src[kCodePieces];
    int code_dst[kCodePieces];
#pragma unroll
    for (int j = 0; j < kCodePieces; ++j) {
        const int piece = t + kWideThreads * j, r = piece / kRowPieces, part = piece % kRowPieces;
        code_src[j] = plane.data + row_of(r) * (K / (kMx ? 1 : 2)) + part * 16;
        code_dst[j] = r * Pipe::kRowBytes + part * 16;
    }
    const std::byte* scale_src =
        !kMx && t < kTileRows ? plane.data + scale_tile_base(plane, row_of(t)) : nullptr;
    const __nv_bfloat16* col_src[4];
    int col_dst[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int piece = t + kWideThreads * j, c = piece >> 3, part = piece & 7;
        const std::int64_t column =
            c < valid ? (kDown ? s_cols[c] - first_column : s_cols[c] / shape.top_k) : -1;
        col_src[j] = column >= 0 ? in + column * K + part * 8 : nullptr;
        col_dst[j] = c * kTileStride + part * 8;
    }
    const auto stage_codes  = [&](int s) { return s_raw + s * Pipe::kStageBytes; };
    const auto stage_scales = [&](int s) {
        return s_raw + s * Pipe::kStageBytes + Pipe::kCodeBytes;
    };
    const auto stage_cols = [&](int s) {
        return reinterpret_cast<__nv_bfloat16*>(s_raw + s * Pipe::kStageBytes + Pipe::kCodeBytes +
                                                Pipe::kScaleBytes);
    };
    const int steps  = K / kWideK;
    const auto issue = [&](int step) {
        const int s = step % Pipe::kStages, k0 = step * kWideK;
#pragma unroll
        for (int j = 0; j < kCodePieces; ++j) {
            cp_async<16, Cache::cg>(stage_codes(s) + code_dst[j], code_src[j] + k0 / (kMx ? 1 : 2));
        }
        if (scale_src != nullptr) { cp_async<4>(stage_scales(s) + t * 4, scale_src + step * 512); }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (col_src[j] != nullptr) {
                cp_async<16, Cache::cg>(stage_cols(s) + col_dst[j], col_src[j] + k0);
            }
        }
    };
#pragma unroll
    for (int s = 0; s < Pipe::kStages - 1; ++s) {
        if (s < steps) { issue(s); }
        cp_commit();
    }
    const int g = lane >> 2, q = lane & 3;
    const int row_lo = warp * 16 + g, row_hi = row_lo + 8;
    // MXFP8: this thread's rows' UE8M0 scales (two a K step), read one step ahead.
    const std::byte* mx_lo = plane.data + plane.scale_offset + row_of(row_lo) * (K / 32);
    const std::byte* mx_hi = plane.data + plane.scale_offset + row_of(row_hi) * (K / 32);
    const auto mx_scales   = [&](int step) {
        if (!kMx || step >= steps) { return 0U; }
        const auto* lo = reinterpret_cast<const std::uint8_t*>(mx_lo) + step * 2;
        const auto* hi = reinterpret_cast<const std::uint8_t*>(mx_hi) + step * 2;
        return static_cast<unsigned>(lo[0]) | static_cast<unsigned>(lo[1]) << 8 |
               static_cast<unsigned>(hi[0]) << 16 | static_cast<unsigned>(hi[1]) << 24;
    };
    float acc[kTileCols / 8][4] = {};
    // The K loop for the tile's first kPairs 16-column pairs (a predicated-off MMA costs as much
    // as an issued one, so the pair count is a template constant). Staged columns past the last
    // valid one are never copied: their dots are garbage and never stored.
    const auto product = [&](auto pairs) {
        constexpr int kPairs = decltype(pairs)::value;
        unsigned mx          = mx_scales(0);
        for (int step = 0; step < steps; ++step) {
            const int s = step % Pipe::kStages;
            cp_wait<Pipe::kStages - 2>();
            __syncthreads(); // stage `step` landed; every warp is past the previous step's MMAs
            const unsigned mx_step = mx;
            mx                     = mx_scales(step + 1);
            // This thread's code words of its two rows: NVFP4 group kk is words 2 kk (columns
            // 0-7) and 2 kk + 1 (8-15), this thread's pairs byte q of each, the scale byte kk of
            // the row's scale word; MXFP8 group kk is words 4 kk.. 4 kk + 3, this thread's pairs
            // half q % 2 of words 4 kk + q / 2 and 4 kk + 2 + q / 2, the scale byte kk / 2.
            constexpr int kWords = Pipe::kRowBytes / 4;
            const auto* codes    = reinterpret_cast<const uint4*>(stage_codes(s));
            std::uint32_t lw[kWords], hw[kWords];
#pragma unroll
            for (int i = 0; i < kWords / 4; ++i) {
                const uint4 l = codes[row_lo * (kWords / 4) + i];
                const uint4 h = codes[row_hi * (kWords / 4) + i];
                lw[4 * i] = l.x, lw[4 * i + 1] = l.y, lw[4 * i + 2] = l.z, lw[4 * i + 3] = l.w;
                hw[4 * i] = h.x, hw[4 * i + 1] = h.y, hw[4 * i + 2] = h.z, hw[4 * i + 3] = h.w;
            }
            std::uint32_t ls = 0, hs = 0;
            if constexpr (!kMx) {
                const auto* scale_words = reinterpret_cast<const std::uint32_t*>(stage_scales(s));
                ls                      = scale_words[row_lo];
                hs                      = scale_words[row_hi];
            }
            const __nv_bfloat16* cols = stage_cols(s);
#pragma unroll
            for (int kk = 0; kk < kWideK / 16; ++kk) {
                unsigned a0, a1, a2, a3;
                if constexpr (kMx) {
                    const float lscale = ue8m0((mx_step >> (8 * (kk / 2))) & 0xFFU);
                    const float hscale = ue8m0((mx_step >> (16 + 8 * (kk / 2))) & 0xFFU);
                    // Selects, not a lane-dependent index: the words stay in registers.
                    const bool odd = q >= 2;
                    const int half = 16 * (q % 2), w = 4 * kk;
                    a0 = mxfp8_bf16x2((odd ? lw[w + 1] : lw[w]) >> half, lscale);
                    a1 = mxfp8_bf16x2((odd ? hw[w + 1] : hw[w]) >> half, hscale);
                    a2 = mxfp8_bf16x2((odd ? lw[w + 3] : lw[w + 2]) >> half, lscale);
                    a3 = mxfp8_bf16x2((odd ? hw[w + 3] : hw[w + 2]) >> half, hscale);
                } else {
                    const float lscale = e4m3fn(static_cast<std::uint8_t>(ls >> (8 * kk)));
                    const float hscale = e4m3fn(static_cast<std::uint8_t>(hs >> (8 * kk)));
                    a0                 = nvfp4_bf16x2(lw[2 * kk] >> (8 * q), lscale);
                    a1                 = nvfp4_bf16x2(hw[2 * kk] >> (8 * q), hscale);
                    a2                 = nvfp4_bf16x2(lw[2 * kk + 1] >> (8 * q), lscale);
                    a3                 = nvfp4_bf16x2(hw[2 * kk + 1] >> (8 * q), hscale);
                }
#pragma unroll
                for (int p = 0; p < kPairs; ++p) {
                    unsigned bq[4];
                    const __nv_bfloat16* b_ptr =
                        cols + (p * 16 + (lane >> 4) * 8 + (lane & 7)) * kTileStride + kk * 16 +
                        ((lane >> 3) & 1) * 8;
                    ldmatrix_x4(bq[0], bq[1], bq[2], bq[3],
                                static_cast<unsigned>(__cvta_generic_to_shared(b_ptr)));
                    mma_bf16(acc[2 * p][0], acc[2 * p][1], acc[2 * p][2], acc[2 * p][3], a0, a1, a2,
                             a3, bq[0], bq[1]);
                    mma_bf16(acc[2 * p + 1][0], acc[2 * p + 1][1], acc[2 * p + 1][2],
                             acc[2 * p + 1][3], a0, a1, a2, a3, bq[2], bq[3]);
                }
                // The next stage's copies go out once this step's first MMAs are queued.
                if (kk == 0) {
                    if (step + Pipe::kStages - 1 < steps) { issue(step + Pipe::kStages - 1); }
                    cp_commit();
                }
            }
        }
    };
    switch ((valid + 15) / 16) {
    case 1:
        product(std::integral_constant<int, 1>{});
        break;
    case 2:
        product(std::integral_constant<int, 2>{});
        break;
    case 3:
        product(std::integral_constant<int, 3>{});
        break;
    case 4:
        product(std::integral_constant<int, 4>{});
        break;
    case 5:
        product(std::integral_constant<int, 5>{});
        break;
    case 6:
        product(std::integral_constant<int, 6>{});
        break;
    case 7:
        product(std::integral_constant<int, 7>{});
        break;
    default:
        product(std::integral_constant<int, kTileCols / 16>{});
        break;
    }
    cp_wait<0>();
    __syncthreads(); // the stages are free for the epilogue
    // The dots go to shared memory (column-major tile), then one rolled pass per output.
    auto* s_dot = reinterpret_cast<float*>(s_raw);
#pragma unroll
    for (int nt = 0; nt < kTileCols / 8; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int c                                           = nt * 8 + q * 2 + (e & 1);
            s_dot[c * kTileDotStride + (e < 2 ? row_lo : row_hi)] = acc[nt][e];
        }
    }
    float* s_div = s_dot + kTileCols * kTileDotStride;
    if (t < kTileRows) { s_div[t] = row_divisor(plane, row_of(t)); }
    __syncthreads();
    if constexpr (kDown) {
#pragma unroll 1
        for (int i = t; i < valid * kTileRows; i += kWideThreads) {
            const int c = i / kTileRows, r = i % kTileRows;
            out[static_cast<std::int64_t>(s_cols[c] - first_column) * H + r0 + r] =
                __float2bfloat16_rn(s_dot[c * kTileDotStride + r] / s_div[r]);
        }
    } else {
#pragma unroll 1
        for (int i = t; i < valid * kTileChannels; i += kWideThreads) {
            const int c = i / kTileChannels, ch = i % kTileChannels;
            const int gate = 16 * (ch / 8) + ch % 8, up = gate + 8;
            const float* dots = s_dot + c * kTileDotStride;
            out[static_cast<std::int64_t>(s_cols[c] - first_column) * F + r0 + ch] = capped_swiglu(
                bf16_round(dots[gate] / s_div[gate]), bf16_round(dots[up] / s_div[up]), activation);
        }
    }
}

template <bool kMx>
__global__ void __launch_bounds__(kWideThreads, 1)
    moe_bank_gate_up_wide_tile_kernel(const __nv_bfloat16* __restrict__ x, MoeBankPlane gate_up,
                                      MoeBankShape shape, CappedSwiglu activation, int first_column,
                                      const std::int32_t* __restrict__ plan,
                                      __nv_bfloat16* __restrict__ hidden_out) {
    wide_tile<false, kMx>(x, gate_up, shape, activation, first_column, plan, hidden_out);
}

template <bool kMx>
__global__ void __launch_bounds__(kWideThreads, 1)
    moe_bank_down_wide_tile_kernel(const __nv_bfloat16* __restrict__ hidden_in, MoeBankPlane down,
                                   MoeBankShape shape, int first_column,
                                   const std::int32_t* __restrict__ plan,
                                   __nv_bfloat16* __restrict__ out) {
    wide_tile<true, kMx>(hidden_in, down, shape, CappedSwiglu{}, first_column, plan, out);
}

// Launch of a tile kernel with its pipeline's dynamic shared memory (opted in once per device).
template <bool kMx, class Kernel, class... Args>
void launch_tiles(Kernel kernel, FuncAttrPerDevice& attr, dim3 grid, cudaStream_t stream,
                  Args... args) {
    attr.ensure(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, TilePipe<kMx>::kBytes);
    kernel<<<grid, kWideThreads, TilePipe<kMx>::kBytes, stream>>>(args...);
}

// Down projection of wide calls on tensor cores: a block owns one expert and 64 hidden rows and
// writes o = BF16(W_d h / divisor) of every column routed to the expert into `out` (row-major per
// column of the chunk); moe_bank_sum_kernel then sums each token's positions in order.
__global__ void __launch_bounds__(kWideThreads)
    moe_bank_down_wide_kernel(const __nv_bfloat16* __restrict__ hidden_in, MoeBankPlane down,
                              MoeBankShape shape, int first_column,
                              const std::int32_t* __restrict__ plan,
                              __nv_bfloat16* __restrict__ out) {
    extern __shared__ __align__(16) unsigned char s_raw[];
    __shared__ int s_cols[kWideCols];
    auto* s_a    = reinterpret_cast<__nv_bfloat16*>(s_raw);
    auto* s_b    = reinterpret_cast<__nv_bfloat16*>(s_raw + kWideTileBytes);
    const int b  = static_cast<int>(blockIdx.x);
    const int r0 = static_cast<int>(blockIdx.y) * kWideRows;
    const int H = shape.hidden, F = shape.intermediate;
    const std::int64_t base = down.row_begin + static_cast<std::int64_t>(b) * H + r0;
    const auto row_of       = [&](int r) { return base + r; };
    const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
    const int wm = warp & 3, wn = warp >> 2;
    const std::int32_t* list = plan + shape.experts + 1 + plan[b];
    const int count          = plan[b + 1] - plan[b];
    {
        for (int col0 = 0; col0 < count; col0 += kWideCols) {
            __syncthreads(); // the previous tile's columns are no longer read
            if (threadIdx.x < kWideCols) {
                const int c = static_cast<int>(threadIdx.x);
                s_cols[c]   = col0 + c < count ? list[col0 + c] : -1;
            }
            __syncthreads();
            const auto column = [&](int c) -> const __nv_bfloat16* {
                if (s_cols[c] < 0) { return nullptr; }
                return hidden_in + static_cast<std::int64_t>(s_cols[c] - first_column) * F;
            };
            float acc[4][4] = {};
            if (wide_pipelined(down, F)) {
                wide_tile_product_pipelined(down, row_of, column, F, s_raw, acc);
            } else {
                wide_tile_product(down, row_of, column, F, s_a, s_b, acc);
            }
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int r = wm * 16 + (lane >> 2) + (e >= 2 ? 8 : 0);
                    const int c = wn * 32 + nt * 8 + (lane & 3) * 2 + (e & 1);
                    if (col0 + c < count) {
                        float dot = acc[nt][e];
                        if (down.nvfp4) { dot = dot / row_divisor(down, row_of(r)); }
                        out[static_cast<std::int64_t>(s_cols[c] - first_column) * H + r0 + r] =
                            __float2bfloat16_rn(dot);
                    }
                }
            }
        }
    }
}

// partial[:, t] (=, +=) sum over the token's in-bank positions, in position order, of
// weights * o, for the tokens [first_token, first_token + gridDim.x).
__global__ void moe_bank_sum_kernel(const std::int32_t* __restrict__ ids,
                                    const float* __restrict__ weights,
                                    const __nv_bfloat16* __restrict__ o, MoeBankShape shape,
                                    int first_token, bool accumulate, float* __restrict__ partial) {
    const int local = static_cast<int>(blockIdx.x);
    const int token = first_token + local;
    const int row   = static_cast<int>(blockIdx.y) * blockDim.x + static_cast<int>(threadIdx.x);
    if (row >= shape.hidden) { return; }
    float sum = 0.0F;
    for (int j = 0; j < shape.top_k; ++j) {
        const int assignment = token * shape.top_k + j;
        const int id         = ids[assignment];
        const int b          = id - shape.first_expert;
        if (id < 0 || b < 0 || b >= shape.experts) { continue; }
        const float value = __bfloat162float(
            o[static_cast<std::int64_t>(local * shape.top_k + j) * shape.hidden + row]);
        sum = __fadd_rn(sum, __fmul_rn(weights[assignment], value));
    }
    float* out = partial + static_cast<std::int64_t>(token) * shape.hidden + row;
    *out       = accumulate ? __fadd_rn(*out, sum) : sum;
}

// Down projection of narrow calls: one warp per hidden row and token. The token's positions run
// in passes of kDownSlots, a group of kDownLanes lanes per position: each lane its share of the
// row's dot (row_dot<kDownLanes>), the group's sum ((l0 + l1) + (l2 + l3)), the divisor, the BF16
// rounding of o and the weight; then the pass's terms in a pairwise tree over the groups, and the
// passes in order. Positions outside the bank add +0.
constexpr int kDownLanes = 4;
constexpr int kDownSlots = 32 / kDownLanes;

__global__ void __launch_bounds__(kThreads)
    moe_bank_down_kernel(const std::int32_t* __restrict__ ids, const float* __restrict__ weights,
                         const __nv_bfloat16* __restrict__ hidden_in, MoeBankPlane down,
                         MoeBankShape shape, bool accumulate, float* __restrict__ partial) {
    pdl_enter();
    const int token = static_cast<int>(blockIdx.x);
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int row   = static_cast<int>(blockIdx.y) * kWarps + warp;
    if (row >= shape.hidden) { return; }
    const int slot = lane / kDownLanes, part = lane % kDownLanes;
    float sum = 0.0F;
    for (int j0 = 0; j0 < shape.top_k; j0 += kDownSlots) {
        const int j          = j0 + slot;
        const int assignment = token * shape.top_k + j;
        const int id         = j < shape.top_k ? ids[assignment] : -1;
        const int b          = id - shape.first_expert;
        const bool in_bank   = id >= 0 && b >= 0 && b < shape.experts;
        const std::int64_t parent_row =
            down.row_begin + static_cast<std::int64_t>(b) * shape.hidden + row;
        float dot = in_bank
                        ? row_dot<kDownLanes>(down, parent_row,
                                              hidden_in + static_cast<std::int64_t>(assignment) *
                                                              shape.intermediate,
                                              part)
                        : 0.0F;
#pragma unroll
        for (int offset = 1; offset < kDownLanes; offset <<= 1) {
            dot += __shfl_xor_sync(kFullWarpMask, dot, offset);
        }
        float term = 0.0F;
        if (in_bank) {
            const float o = bf16_round(down.nvfp4 ? dot / row_divisor(down, parent_row) : dot);
            term          = __fmul_rn(weights[assignment], o);
        }
#pragma unroll
        for (int offset = kDownLanes; offset < 32; offset <<= 1) {
            term += __shfl_xor_sync(kFullWarpMask, term, offset);
        }
        sum = __fadd_rn(sum, term);
    }
    if (lane == 0) {
        float* out = partial + static_cast<std::int64_t>(token) * shape.hidden + row;
        *out       = accumulate ? __fadd_rn(*out, sum) : sum;
    }
}

__global__ void moe_partial_to_bf16_kernel(const float* __restrict__ partial,
                                           __nv_bfloat16* __restrict__ y, std::int64_t count) {
    pdl_enter();
    const std::int64_t i =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
    if (i < count) { y[i] = __float2bfloat16_rn(partial[i]); }
}

// Dynamic shared memory of a wide kernel: the pipeline's for NVFP4 planes, else the two tiles.
std::size_t wide_shared(const MoeBankPlane& plane, int K) {
    return wide_pipelined(plane, K) ? kWideSharedBytes : 2 * kWideTileBytes;
}

} // namespace

bool moe_bank_wide_route(const MoeBankShape& shape) { return shape.tokens >= kWideMinTokens; }

void moe_bank_capped_swiglu_launch(const Tensor& x, const Tensor& ids, const Tensor& weights,
                                   const MoeBankPlane& gate_up, const MoeBankPlane& down,
                                   const MoeBankShape& shape, CappedSwiglu activation,
                                   bool accumulate, Tensor& partial, const MoeBankWorkspace& work,
                                   cudaStream_t stream) {
    auto* hidden    = reinterpret_cast<__nv_bfloat16*>(work.hidden);
    const auto* xp  = static_cast<const __nv_bfloat16*>(x.data);
    const auto* idp = static_cast<const std::int32_t*>(ids.data);
    const auto* wp  = static_cast<const float*>(weights.data);
    auto* pp        = static_cast<float*>(partial.data);
    if (moe_bank_wide_route(shape)) {
        auto* out = reinterpret_cast<__nv_bfloat16*>(work.out);
        for (int t0 = 0; t0 < shape.tokens; t0 += work.chunk_tokens) {
            const int tokens = min(work.chunk_tokens, shape.tokens - t0);
            const int c0 = t0 * shape.top_k, c1 = (t0 + tokens) * shape.top_k;
            moe_bank_plan_kernel<<<shape.experts, kPlanThreads, 0, stream>>>(idp, shape, c0, c1,
                                                                             work.plan);
            CUDA_CHECK(cudaGetLastError());
            // Tile grids: at most one partial tile per expert beyond the full ones.
            const auto tiles =
                static_cast<unsigned>((c1 - c0 + kTileCols - 1) / kTileCols + shape.experts);
            if (wide_tiles(gate_up, shape.hidden, 2 * shape.intermediate)) {
                static FuncAttrPerDevice nv_attr, mx_attr;
                const dim3 grid(tiles, static_cast<unsigned>(shape.intermediate / kTileChannels));
                if (gate_up.mxfp8) {
                    launch_tiles<true>(moe_bank_gate_up_wide_tile_kernel<true>, mx_attr, grid,
                                       stream, xp, gate_up, shape, activation, c0, work.plan,
                                       hidden);
                } else {
                    launch_tiles<false>(moe_bank_gate_up_wide_tile_kernel<false>, nv_attr, grid,
                                        stream, xp, gate_up, shape, activation, c0, work.plan,
                                        hidden);
                }
            } else {
                const dim3 gate_grid(static_cast<unsigned>(shape.experts),
                                     static_cast<unsigned>(shape.intermediate / kWideChannels));
                moe_bank_gate_up_wide_kernel<<<gate_grid, kWideThreads, 2 * kWideTileBytes,
                                               stream>>>(xp, gate_up, shape, activation, c0,
                                                         work.plan, hidden);
            }
            CUDA_CHECK(cudaGetLastError());
            if (wide_tiles(down, shape.intermediate, shape.hidden)) {
                static FuncAttrPerDevice nv_attr, mx_attr;
                const dim3 grid(tiles, static_cast<unsigned>(shape.hidden / kTileRows));
                const __nv_bfloat16* h = hidden;
                if (down.mxfp8) {
                    launch_tiles<true>(moe_bank_down_wide_tile_kernel<true>, mx_attr, grid, stream,
                                       h, down, shape, c0, work.plan, out);
                } else {
                    launch_tiles<false>(moe_bank_down_wide_tile_kernel<false>, nv_attr, grid,
                                        stream, h, down, shape, c0, work.plan, out);
                }
            } else {
                const dim3 down_grid(static_cast<unsigned>(shape.experts),
                                     static_cast<unsigned>(shape.hidden / kWideRows));
                moe_bank_down_wide_kernel<<<down_grid, kWideThreads,
                                            wide_shared(down, shape.intermediate), stream>>>(
                    hidden, down, shape, c0, work.plan, out);
            }
            CUDA_CHECK(cudaGetLastError());
            const dim3 sum_grid(static_cast<unsigned>(tokens),
                                static_cast<unsigned>((shape.hidden + 255) / 256));
            moe_bank_sum_kernel<<<sum_grid, 256, 0, stream>>>(idp, wp, out, shape, t0, accumulate,
                                                              pp);
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }
    const dim3 gate_grid(static_cast<unsigned>(shape.tokens * shape.top_k),
                         (shape.intermediate + kWarps - 1) / kWarps);
    // The narrow route serves decode and verification widths (the wide one takes the rest).
    const bool dependent = shape.tokens <= kDependentLaunchMaxTokens;
    launch_maybe_dependent(dependent, gate_grid, dim3(kThreads), 0, stream, moe_bank_gate_up_kernel,
                           xp, idp, gate_up, shape, activation, hidden);
    const dim3 down_grid(static_cast<unsigned>(shape.tokens), (shape.hidden + kWarps - 1) / kWarps);
    launch_maybe_dependent(dependent, down_grid, dim3(kThreads), 0, stream, moe_bank_down_kernel,
                           idp, wp, hidden, down, shape, accumulate, pp);
}

void moe_partial_to_bf16_launch(const Tensor& partial, Tensor& y, cudaStream_t stream) {
    const std::int64_t count = partial.numel();
    constexpr int threads    = 256;
    const auto blocks        = static_cast<unsigned>((count + threads - 1) / threads);
    launch_maybe_dependent(count <= kDependentLaunchMaxTokens * partial.ne[0], dim3(blocks),
                           dim3(threads), 0, stream, moe_partial_to_bf16_kernel,
                           static_cast<const float*>(partial.data),
                           static_cast<__nv_bfloat16*>(y.data), count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
