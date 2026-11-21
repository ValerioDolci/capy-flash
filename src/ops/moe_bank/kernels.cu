// Implements: include/ninfer/ops/moe_bank.h
// Two kernels, one warp per output row. The first runs per (token, position) in the bank and per
// group of expert channels: the gate and up dots of channel i, their BF16 roundings, the capped
// SwiGLU in FP32 and h rounded to BF16 into the workspace. The second runs per token and per group
// of hidden rows: for each position of the token in the bank (in position order) the down dot,
// its BF16 rounding and the weighted FP32 sum, then the store or the one accumulating addition.
// Weights are decoded exactly in registers (NVFP4: 16 E2M1 codes times their E4M3FN scale, the
// divisor applied once to the dot).
#include "ops/moe_bank/launch.h"

#include "core/device.h"
#include "ops/common/pdl_launch.cuh" // CUDA_CHECK
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>

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
// plane.columns); the caller sums the lanes. NVFP4 rows return the dot before the divisor.
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
        for (int g = lane; g < columns / 16; g += 32) {
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
        for (int c = lane * 8; c < columns; c += 32 * 8) {
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

__device__ __forceinline__ float full_dot(const MoeBankPlane& plane, std::int64_t row,
                                          const __nv_bfloat16* __restrict__ v, int lane) {
    const float dot = warp_sum(row_dot(plane, row, v, lane));
    return plane.nvfp4 ? dot / row_divisor(plane, row) : dot;
}

__device__ __forceinline__ float softcap(float v, float cap) { return cap * tanhf(v / cap); }

__device__ __forceinline__ __nv_bfloat16 capped_swiglu(float g, float a, CappedSwiglu activation) {
    const float sig = 1.0F / (1.0F + expf(-g));
    return __float2bfloat16_rn(softcap(g, activation.gate_cap) * sig *
                               softcap(a, activation.up_cap));
}

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
    const std::int64_t base =
        gate_up.row_begin + static_cast<std::int64_t>(b) * 2 * shape.intermediate;
    const float g = bf16_round(full_dot(gate_up, base + channel, u, lane));
    const float a = bf16_round(full_dot(gate_up, base + shape.intermediate + channel, u, lane));
    if (lane == 0) {
        hidden_out[static_cast<std::int64_t>(assignment) * shape.intermediate + channel] =
            capped_swiglu(g, a, activation);
    }
}

// Wide calls (more than kWideMinTokens - 1 tokens): grouped by expert on tensor cores. A block
// owns one expert of the bank and kWideChannels channels, i.e. a 64-row tile of its gate rows
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
            if (wide_pipelined(gate_up, H)) {
                wide_tile_product_pipelined(gate_up, row_of, column, H, s_raw, acc);
            } else {
                wide_tile_product(gate_up, row_of, column, H, s_a, s_b, acc);
            }
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
    float sum = 0.0F;
    for (int j = 0; j < shape.top_k; ++j) {
        const int assignment = token * shape.top_k + j;
        const int id         = ids[assignment];
        const int b          = id - shape.first_expert;
        if (id < 0 || b < 0 || b >= shape.experts) { continue; }
        const std::int64_t parent_row =
            down.row_begin + static_cast<std::int64_t>(b) * shape.hidden + row;
        const float o = bf16_round(
            full_dot(down, parent_row,
                     hidden_in + static_cast<std::int64_t>(assignment) * shape.intermediate, lane));
        sum = __fadd_rn(sum, __fmul_rn(weights[assignment], o));
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
            const dim3 gate_grid(static_cast<unsigned>(shape.experts),
                                 static_cast<unsigned>(shape.intermediate / kWideChannels));
            moe_bank_gate_up_wide_kernel<<<gate_grid, kWideThreads,
                                           wide_shared(gate_up, shape.hidden), stream>>>(
                xp, gate_up, shape, activation, c0, work.plan, hidden);
            CUDA_CHECK(cudaGetLastError());
            const dim3 down_grid(static_cast<unsigned>(shape.experts),
                                 static_cast<unsigned>(shape.hidden / kWideRows));
            moe_bank_down_wide_kernel<<<down_grid, kWideThreads,
                                        wide_shared(down, shape.intermediate), stream>>>(
                hidden, down, shape, c0, work.plan, out);
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
