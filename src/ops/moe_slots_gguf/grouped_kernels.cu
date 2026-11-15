// Modified for capy-flash from fngine:src/kernels/moe/moe_grouped_gemm.cu and
// fngine:src/kernels/moe/moe_prefill.cu @ c20a885 (satellitedown/fngine, Apache License 2.0). See
// NOTICE.
//
// Implements: include/ninfer/ops/moe_slots_gguf.h (moe_grouped_gguf)
//
// Prefill grouped expert GEMMs over the native GGUF block formats: 64-token x 64-row tiles over
// per-expert token lists. Weights are dequantized exactly (ggml reference arithmetic) into BF16
// shared memory, activations are gathered with cp.async, mma.m16n8k16 BF16 with FP32
// accumulation; h = silu(g) * u is rounded to BF16. Kept from fngine: both GEMM kernels (tile
// shape, raw-weight staging, pinned dequantization ahead of the MMAs, swizzled shared tiles) and
// the count / scan / scatter shape of its prefill routing.
// Changes in this tree:
//   * the routing is this Op's input: routed_ids through the layer's slot table (an id without a
//     slot is skipped) instead of a router top-k; the per-expert lists hold assignments
//     (token, position); experts are counted by id (E <= kGgufGroupedMaxExperts);
//   * an expert's gate|up and down rows come from the slot's blob offset or the planar bank bases
//     (pointer tables written by the scan) instead of a per-expert blob array;
//   * the down kernel writes the unweighted FP32 output of each assignment (grouped order) and a
//     merge kernel sums the positions of a token in top-k order (fmaf with the routed weight),
//     instead of fngine's fixed-point atomics into y; the exchange publish and CPU-tier combine
//     are not carried over;
//   * pieces of <= kGgufGroupedChunkColumns columns (moe_grouped_gguf), as moe_grouped_nvfp4.
#include "ops/moe_slots_gguf/iq_device.cuh"
#include "ops/moe_slots_gguf/launch.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using Format = MoeGgufFormat;
using gguf_device::dequant8;
using gguf_device::kRowBytes;

constexpr int kHidden       = kMoeSlotsHidden;       // 2560
constexpr int kInter        = kMoeSlotsIntermediate; // 640
constexpr int kTopK         = kMoeSlotsTopK;         // 10
constexpr int kThreads      = 256;
constexpr int kBK           = 64;                     // K elements per pipeline step
constexpr int kChunksPerRow = kBK / 8;                // 16-byte chunks per smem row
constexpr int kTileM        = kGgufGroupedTileTokens; // tokens
constexpr int kTileN        = 64;                     // weight rows (per matrix)
constexpr int kGuRowTiles   = kInter / kTileN;        // 10
constexpr int kDownRowTiles = kHidden / kTileN;       // 40
constexpr int kScanThreads  = kGgufGroupedMaxExperts;

// Swizzled 16-byte chunk offset (in bf16 elements) inside a [rows][64] bf16 tile.
__device__ __forceinline__ int swz(int row, int chunk) {
    return row * kBK + ((chunk ^ (row & 7)) << 3);
}

__device__ __forceinline__ int4 pack_bf16x8(const float v[8]) {
    int4 r;
    r.x = int(pack_bf16x2(v[0], v[1]));
    r.y = int(pack_bf16x2(v[2], v[3]));
    r.z = int(pack_bf16x2(v[4], v[5]));
    r.w = int(pack_bf16x2(v[6], v[7]));
    return r;
}

// Dequantize `Groups` consecutive 8-element groups of one staged weight row into BF16 registers.
// The values are materialized here (asm barrier), ahead of the volatile MMA sequence that follows
// in the callers: letting the compiler interleave the dequant table lookups with the MMAs stalls
// the tensor pipe (fngine measured ~3.5x slower gate/up GEMM).
template <Format F, int Groups>
__device__ __forceinline__ void dequant_groups_pinned(const std::uint8_t* row, int g0,
                                                      int4 (&v)[Groups]) {
#pragma unroll
    for (int i = 0; i < Groups; ++i) {
        float f[8];
        dequant8<F>(row, g0 + i, f);
        v[i] = pack_bf16x8(f);
        asm volatile("" : "+r"(v[i].x), "+r"(v[i].y), "+r"(v[i].z), "+r"(v[i].w));
    }
}

// Store `Groups` 16-byte chunks (chunk0..) of one row of a swizzled [rows][64] BF16 tile.
template <int Groups>
__device__ __forceinline__ void store_groups(__nv_bfloat16* tile, int row, int chunk0,
                                             const int4 (&v)[Groups]) {
#pragma unroll
    for (int i = 0; i < Groups; ++i) *reinterpret_cast<int4*>(tile + swz(row, chunk0 + i)) = v[i];
}

// A fragments for two m16 tiles (rows m0.. m0+31) at k16 step kk.
__device__ __forceinline__ void load_a_frags(const __nv_bfloat16* A, int m0, int kk, int lane,
                                             unsigned a[2][4]) {
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        const int row   = m0 + 16 * mt + (lane & 15);
        const int chunk = kk * 2 + (lane >> 4);
        ldmatrix_x4(a[mt][0], a[mt][1], a[mt][2], a[mt][3], smem_addr(A + swz(row, chunk)));
    }
}

// B fragments for two n8 tiles (weight rows n0..n0+15) at k16 step kk.
__device__ __forceinline__ void load_b_frags(const __nv_bfloat16* W, int n0, int kk, int lane,
                                             unsigned b[2][2]) {
    const int row   = n0 + (lane & 7) + ((lane >> 4) << 3);
    const int chunk = kk * 2 + ((lane >> 3) & 1);
    ldmatrix_x4(b[0][0], b[0][1], b[1][0], b[1][1], smem_addr(W + swz(row, chunk)));
}

// Gather `Rows` activation rows (16-byte chunks via cp.async, zero-filled past `n`).
template <int Rows = kTileM, int Threads = kThreads>
__device__ __forceinline__ void load_a_tile(__nv_bfloat16* A, const __nv_bfloat16* const* rows,
                                            int k0, int n) {
    static_assert((Rows * kChunksPerRow) % Threads == 0);
#pragma unroll
    for (int i = 0; i < (Rows * kChunksPerRow) / Threads; ++i) {
        const int id = threadIdx.x + i * Threads;
        const int r = id / kChunksPerRow, c = id % kChunksPerRow;
        const __nv_bfloat16* src = rows[r] + k0 + c * 8;
        cp_async_zfill<16, Cache::cg>(A + swz(r, c), src, r < n ? 16 : 0);
    }
}

// Stage one byte segment of each of `Rows` weight rows into shared memory rows of `Stride` bytes:
// 4-byte cp.async from the 4-byte aligned address at or below each segment start (row_ptr[r] +
// seg_off), so the segment sits at offset (start & 3) of its smem row. Bytes at or past `limit`
// are zero-filled (never read from global memory). The stride is an odd number of words so that
// consecutive rows start in different banks.
constexpr int raw_stride4(int bytes) {
    const int words = (bytes + 3 + 3) / 4;
    return (words | 1) * 4;
}

template <int Rows, int Stride, int Threads = kThreads>
__device__ __forceinline__ void stage_raw4(std::uint8_t* raw, const std::uint8_t* const* row_ptr,
                                           std::int64_t seg_off, const std::uint8_t* limit) {
    static_assert(Stride % 4 == 0);
    constexpr int kWords = Stride / 4;
    for (int id = threadIdx.x; id < Rows * kWords; id += Threads) {
        const int r = id / kWords, wd = id % kWords;
        const std::uint8_t* start = row_ptr[r] + seg_off;
        const std::uint8_t* src   = start - (reinterpret_cast<std::uintptr_t>(start) & 3) + wd * 4;
        const std::int64_t left   = limit - src;
        const int bytes           = left >= 4 ? 4 : (left > 0 ? int(left) : 0);
        cp_async_zfill<4>(raw + r * Stride + wd * 4, src, bytes);
    }
}

__device__ __forceinline__ int raw_misalign4(const std::uint8_t* row, std::int64_t seg_off) {
    return int((reinterpret_cast<std::uintptr_t>(row) + std::uintptr_t(seg_off)) & 3);
}

template <Format F>
struct GateUpCfg {
    static constexpr int kRow         = kRowBytes<F, kHidden>;
    static constexpr int kSegElems    = 256;                          // one raw stage
    static constexpr int kSegBytes    = kRow / (kHidden / kSegElems); // bytes of 256 elements
    static constexpr int kRawStride   = raw_stride4(kSegBytes);
    static constexpr int kStepsPerSeg = kSegElems / kBK;         // 4
    static constexpr int kAElems      = kTileM * kBK;            // 2 buffers
    static constexpr int kWElems      = 2 * kTileN * kBK;        // single buffer
    static constexpr int kRawBytes    = 2 * kTileN * kRawStride; // single buffer
    static constexpr int kSmem        = 2 * kAElems * 2 + kWElems * 2 + kRawBytes;
};

// ------------------------------------------------------------------------------- gate/up
// CTA = (64-token tile of one expert, 64 intermediate rows): gate and up rows are computed
// together so the epilogue can form h. Per 64-wide K step ks: A(ks+1) (token rows) is gathered
// with cp.async; W(ks+1) is dequantized from the staged raw weight bytes into registers while the
// MMAs of step ks run and stored to the single BF16 W buffer after a barrier; the raw bytes of the
// next 256 elements are staged into the single raw buffer right after the last dequant of the
// current segment.
template <Format F>
__global__ void __launch_bounds__(kThreads, 2)
    grouped_gate_up_kernel(const int4* tiles, const std::int32_t* tile_count,
                           const std::uint8_t* const* expert_gate_up,
                           const std::int32_t* pair_token, const __nv_bfloat16* x,
                           __nv_bfloat16* h) {
    using C            = GateUpCfg<F>;
    const int tile_idx = blockIdx.x / kGuRowTiles, rt = blockIdx.x % kGuRowTiles;
    if (tile_idx >= *tile_count) return;
    constexpr int kSteps = kHidden / kBK; // 40
    extern __shared__ __align__(128) std::uint8_t smem_raw[];
    __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(smem_raw);       // [2][64][64]
    __nv_bfloat16* Ws = As + 2 * C::kAElems;                              // [128][64]
    std::uint8_t* Raw = reinterpret_cast<std::uint8_t*>(Ws + C::kWElems); // [128][kRawStride]
    __shared__ const __nv_bfloat16* s_rows[kTileM];
    __shared__ const std::uint8_t* s_wrow[2 * kTileN];

    const int4 tile = tiles[tile_idx];
    const int pb = tile.x, n = tile.y - tile.x;
    const std::uint8_t* gate_up = expert_gate_up[tile.z];
    if (threadIdx.x < kTileM) {
        const int r = threadIdx.x;
        s_rows[r]   = x + std::size_t(pair_token[pb + (r < n ? r : 0)]) * kHidden;
    }
    if (threadIdx.x < 2 * kTileN) {
        const int r = threadIdx.x;
        s_wrow[r]   = gate_up + (r < kTileN ? 0 : std::int64_t(kInter) * C::kRow) +
                    std::int64_t(rt * kTileN + (r & (kTileN - 1))) * C::kRow;
    }
    const std::uint8_t* limit = gate_up + std::int64_t(2 * kInter) * C::kRow; // gate|up end
    __syncthreads();

    // Dequant assignment: thread -> weight row (0..63 gate, 64..127 up), 4 groups (32 elements).
    const int wrow             = threadIdx.x >> 1;
    const int wgrp             = (threadIdx.x & 1) * 4;
    const std::uint8_t* my_row = s_wrow[wrow];
    auto dequant_regs          = [&](int k, int4(&v)[4]) {
        const int seg = k / C::kStepsPerSeg;
        const std::uint8_t* src =
            Raw + wrow * C::kRawStride + raw_misalign4(my_row, std::int64_t(seg) * C::kSegBytes);
        dequant_groups_pinned<F, 4>(src, (k % C::kStepsPerSeg) * kChunksPerRow + wgrp, v);
    };

    stage_raw4<2 * kTileN, C::kRawStride>(Raw, s_wrow, 0, limit);
    load_a_tile(As, s_rows, 0, n);
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    {
        int4 v[4];
        dequant_regs(0, v);
        store_groups<4>(Ws, wrow, wgrp, v);
    }

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    // Token halves on warps 0-3 / 4-7 (warp w issues on scheduler w % 4): a tile of <= 32 tokens
    // keeps all four tensor units of the SM busy instead of two.
    const int wm = warp >> 2, wn = warp & 3;
    const bool active = wm * 32 < n;
    float acc[2][4][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;

#pragma unroll 1
    for (int ks = 0; ks < kSteps; ++ks) {
        // A(ks) and any raw stage issued since the previous barrier have landed; the barrier also
        // publishes W(ks) and frees A buffer (ks+1)&1.
        cp_wait<0>();
        __syncthreads();
        if (ks + 1 < kSteps) {
            load_a_tile(As + ((ks + 1) & 1) * C::kAElems, s_rows, (ks + 1) * kBK, n);
            cp_commit();
        }

        int4 wnext[4];
        if (ks + 1 < kSteps) dequant_regs(ks + 1, wnext);

        if (active) {
            const __nv_bfloat16* A = As + (ks & 1) * C::kAElems;
#pragma unroll
            for (int kk = 0; kk < kBK / 16; ++kk) {
                unsigned a[2][4], bg[2][2], bu[2][2];
                load_a_frags(A, wm * 32, kk, lane, a);
                load_b_frags(Ws, wn * 16, kk, lane, bg);
                load_b_frags(Ws, kTileN + wn * 16, kk, lane, bu);
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    if (wm * 32 + mt * 16 >= n) continue; // padding rows only, never stored
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        mma_bf16(acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2], acc[mt][nt][3],
                                 a[mt][0], a[mt][1], a[mt][2], a[mt][3], bg[nt][0], bg[nt][1]);
                        mma_bf16(acc[mt][2 + nt][0], acc[mt][2 + nt][1], acc[mt][2 + nt][2],
                                 acc[mt][2 + nt][3], a[mt][0], a[mt][1], a[mt][2], a[mt][3],
                                 bu[nt][0], bu[nt][1]);
                    }
                }
            }
        }
        if (ks + 1 < kSteps) {
            __syncthreads(); // every warp is done reading W(ks) (and the raw segment, if finished)
            store_groups<4>(Ws, wrow, wgrp, wnext);
            const int k1 = ks + 1;
            if (k1 % C::kStepsPerSeg == C::kStepsPerSeg - 1 && k1 + 1 < kSteps) {
                // W(k1) was the last dequant of its segment: stage the next one (waited at the
                // next top).
                stage_raw4<2 * kTileN, C::kRawStride>(
                    Raw, s_wrow, std::int64_t(k1 / C::kStepsPerSeg + 1) * C::kSegBytes, limit);
                cp_commit();
            }
        }
    }

    // Epilogue: h = silu(g) * u, BF16, rows of valid tokens only.
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int tok = wm * 32 + mt * 16 + (lane >> 2) + half * 8;
            if (tok >= n) continue;
            __nv_bfloat16* hrow = h + std::size_t(pb + tok) * kInter + rt * kTileN;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int col  = wn * 16 + nt * 8 + (lane & 3) * 2;
                const float g0 = acc[mt][nt][half * 2 + 0], g1 = acc[mt][nt][half * 2 + 1];
                const float u0 = acc[mt][2 + nt][half * 2 + 0], u1 = acc[mt][2 + nt][half * 2 + 1];
                *reinterpret_cast<std::uint32_t*>(hrow + col) =
                    pack_bf16x2(silu(g0) * u0, silu(g1) * u1);
            }
        }
    }
}

template <Format F>
struct DownCfg {
    static constexpr int kRow       = kRowBytes<F, kInter>;
    static constexpr int kRawStride = raw_stride4(kRow); // whole 640-element row staged once
    static constexpr int kAElems    = kTileM * kBK;      // 2 buffers
    static constexpr int kWElems    = kTileN * kBK;      // single buffer
    static constexpr int kSmem      = 2 * kAElems * 2 + kWElems * 2 + kTileN * kRawStride;
};

// ---------------------------------------------------------------------------------- down
// CTA = (64-token tile of one expert, 64 output rows). The 64 weight rows (K = 640) are staged raw
// once in the prologue; per 64-wide K step, A(ks+1) is gathered with cp.async, W(ks+1) is
// dequantized into registers ahead of the MMAs of step ks and stored to the single BF16 W buffer
// after a barrier. The epilogue writes the FP32 output row of every assignment of the tile.
template <Format F>
__global__ void __launch_bounds__(kThreads, 2)
    grouped_down_kernel(const int4* tiles, const std::int32_t* tile_count,
                        const std::uint8_t* const* expert_down, const __nv_bfloat16* h,
                        float* grouped) {
    using C            = DownCfg<F>;
    const int tile_idx = blockIdx.x / kDownRowTiles, rt = blockIdx.x % kDownRowTiles;
    if (tile_idx >= *tile_count) return;
    constexpr int kSteps = kInter / kBK; // 10
    extern __shared__ __align__(128) std::uint8_t smem_raw[];
    __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(smem_raw);       // [2][64][64]
    __nv_bfloat16* Ws = As + 2 * C::kAElems;                              // [64][64]
    std::uint8_t* Raw = reinterpret_cast<std::uint8_t*>(Ws + C::kWElems); // [64][kRawStride]
    __shared__ const __nv_bfloat16* s_rows[kTileM];
    __shared__ const std::uint8_t* s_wrow[kTileN];

    const int4 tile = tiles[tile_idx];
    const int pb = tile.x, n = tile.y - tile.x;
    const std::uint8_t* down = expert_down[tile.z];
    if (threadIdx.x < kTileM) {
        const int r = threadIdx.x;
        s_rows[r]   = h + std::size_t(pb + (r < n ? r : 0)) * kInter;
        s_wrow[r]   = down + std::int64_t(rt * kTileN + r) * C::kRow;
    }
    const std::uint8_t* limit = down + std::int64_t(kHidden) * C::kRow;
    __syncthreads();

    // Dequant assignment: thread -> weight row, 2 groups (16 elements) per step.
    const int wrow             = threadIdx.x >> 2;
    const int wgrp             = (threadIdx.x & 3) * 2;
    const std::uint8_t* my_raw = Raw + wrow * C::kRawStride + raw_misalign4(s_wrow[wrow], 0);

    stage_raw4<kTileN, C::kRawStride>(Raw, s_wrow, 0, limit);
    load_a_tile(As, s_rows, 0, n);
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    {
        int4 v[2];
        dequant_groups_pinned<F, 2>(my_raw, wgrp, v);
        store_groups<2>(Ws, wrow, wgrp, v);
    }

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    // Token halves on warps 0-3 / 4-7 (warp w issues on scheduler w % 4): a tile of <= 32 tokens
    // keeps all four tensor units of the SM busy instead of two.
    const int wm = warp >> 2, wn = warp & 3;
    const bool active = wm * 32 < n;
    float acc[2][2][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 2; ++b)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;

#pragma unroll 1
    for (int ks = 0; ks < kSteps; ++ks) {
        cp_wait<0>();
        __syncthreads();
        if (ks + 1 < kSteps) {
            load_a_tile(As + ((ks + 1) & 1) * C::kAElems, s_rows, (ks + 1) * kBK, n);
            cp_commit();
        }
        int4 wnext[2];
        if (ks + 1 < kSteps)
            dequant_groups_pinned<F, 2>(my_raw, (ks + 1) * kChunksPerRow + wgrp, wnext);
        if (active) {
            const __nv_bfloat16* A = As + (ks & 1) * C::kAElems;
#pragma unroll
            for (int kk = 0; kk < kBK / 16; ++kk) {
                unsigned a[2][4], b[2][2];
                load_a_frags(A, wm * 32, kk, lane, a);
                load_b_frags(Ws, wn * 16, kk, lane, b);
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    if (wm * 32 + mt * 16 >= n) continue; // padding rows only, never stored
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt)
                        mma_bf16(acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2], acc[mt][nt][3],
                                 a[mt][0], a[mt][1], a[mt][2], a[mt][3], b[nt][0], b[nt][1]);
                }
            }
        }
        if (ks + 1 < kSteps) {
            __syncthreads(); // every warp is done reading W(ks)
            store_groups<2>(Ws, wrow, wgrp, wnext);
        }
    }

#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int tok = wm * 32 + mt * 16 + (lane >> 2) + half * 8;
            if (tok >= n) continue;
            float* yrow = grouped + std::size_t(pb + tok) * kHidden + rt * kTileN;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int col = wn * 16 + nt * 8 + (lane & 3) * 2;
                *reinterpret_cast<float2*>(yrow + col) =
                    make_float2(acc[mt][nt][half * 2 + 0], acc[mt][nt][half * 2 + 1]);
            }
        }
    }
}

// ------------------------------------------------------------------------------ grouping
// One exclusive scan over the CTA (fngine moe_common.cuh); blockDim.x a multiple of 32.
__device__ __forceinline__ int block_exclusive_scan(int v, int* scratch, int* total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    int incl = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int n = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += n;
    }
    if (lane == 31) scratch[warp] = incl;
    __syncthreads();
    if (warp == 0) {
        int s = lane < warps ? scratch[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int n = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += n;
        }
        scratch[lane] = s; // inclusive warp totals
    }
    __syncthreads();
    const int base = warp > 0 ? scratch[warp - 1] : 0;
    *total         = scratch[warps - 1];
    __syncthreads();
    return base + incl - v;
}

__device__ __forceinline__ int slot_of(const std::int32_t* ids, const GgufExperts& ex, int a) {
    const int e = ids[a];
    if (e < 0 || e >= ex.experts) return -1;
    const int s = ex.table[e];
    if (s >= 0 && s < ex.slots) return s;
    // Hybrid blob form: a non-resident expert read in place from the host bank.
    if (ex.host_table != nullptr && ex.slot_offset != nullptr && ex.host_table[e] >= 0)
        return ex.slots;
    return -1;
}

// rank[a] orders assignment a among those of its expert; -1 = no slot.
__global__ void count_kernel(const std::int32_t* ids, GgufExperts ex, int assignments,
                             std::int32_t* counts, std::int32_t* rank) {
    const int a = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (a >= assignments) return;
    rank[a] = slot_of(ids, ex, a) >= 0 ? atomicAdd(&counts[ids[a]], 1) : -1;
}

// One CTA: expert offsets, the (expert, 64-assignment tile) list and the experts' row pointers.
__global__ void __launch_bounds__(kScanThreads)
    scan_kernel(GgufExperts ex, const std::int32_t* counts, std::int32_t* offsets, int4* tiles,
                std::int32_t* tile_count, const std::uint8_t** expert_gate_up,
                const std::uint8_t** expert_down) {
    __shared__ int s_scan[32];
    const int e     = threadIdx.x;
    const int c     = e < ex.experts ? counts[e] : 0;
    const int nt    = (c + kTileM - 1) / kTileM;
    int total_pairs = 0, total_tiles = 0;
    const int off   = block_exclusive_scan(c, s_scan, &total_pairs);
    const int tile0 = block_exclusive_scan(nt, s_scan, &total_tiles);
    if (e < ex.experts) {
        offsets[e] = off;
        for (int j = 0; j < nt; ++j)
            tiles[tile0 + j] = make_int4(off + j * kTileM, off + min((j + 1) * kTileM, c), e, 0);
        if (c > 0) {
            const int s = ex.table[e];
            const std::uint8_t* gate_up;
            const std::uint8_t* down;
            if (ex.slot_offset != nullptr && !(s >= 0 && s < ex.slots)) {
                // Hybrid: counted only through host_table (slot_of).
                const std::int64_t row = ex.host_table[e];
                gate_up                = ex.gate_up_base + row * ex.gate_up_bytes;
                down                   = ex.down_base + row * ex.down_bytes;
            } else if (ex.slot_offset != nullptr) {
                gate_up = ex.base + ex.slot_offset[s];
                down    = gate_up + ex.gate_up_bytes;
            } else {
                gate_up = ex.gate_up_base + std::int64_t(s) * ex.gate_up_bytes;
                down    = ex.down_base + std::int64_t(s) * ex.down_bytes;
            }
            expert_gate_up[e] = gate_up;
            expert_down[e]    = down;
        }
    }
    if (e == 0) {
        offsets[ex.experts] = total_pairs;
        *tile_count         = total_tiles;
    }
}

__global__ void place_kernel(const std::int32_t* ids, const std::int32_t* rank, int assignments,
                             const std::int32_t* offsets, std::int32_t* packed_token,
                             std::int32_t* packed_index) {
    const int a = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (a >= assignments) return;
    const int r = rank[a];
    if (r < 0) {
        packed_index[a] = -1;
        return;
    }
    const int position     = offsets[ids[a]] + r;
    packed_token[position] = a / kTopK;
    packed_index[a]        = position;
}

// One CTA per column; thread i owns features [8i, 8i+8). Routes without a slot add nothing.
constexpr int kMergeThreads = kHidden / 8;

__global__ void __launch_bounds__(kMergeThreads)
    merge_kernel(const float* grouped, const std::int32_t* packed_index, const float* weights,
                 float* out) {
    const int column  = int(blockIdx.x);
    const int feature = int(threadIdx.x) * 8;
    float sum[8]      = {};
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        const int row = packed_index[column * kTopK + k];
        if (row < 0) continue;
        const float weight  = weights[column * kTopK + k];
        const float* source = grouped + std::int64_t(row) * kHidden + feature;
        const float4 a      = *reinterpret_cast<const float4*>(source);
        const float4 b      = *reinterpret_cast<const float4*>(source + 4);
        const float v[8]    = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) sum[i] = fmaf(weight, v[i], sum[i]);
    }
    float* destination                          = out + std::int64_t(column) * kHidden + feature;
    *reinterpret_cast<float4*>(destination)     = make_float4(sum[0], sum[1], sum[2], sum[3]);
    *reinterpret_cast<float4*>(destination + 4) = make_float4(sum[4], sum[5], sum[6], sum[7]);
}

// ------------------------------------------------------------------------------ dispatch
constexpr int kMaxDevices = 16;

// Raises the dynamic shared-memory limit of `Kernel` once per device.
template <auto Kernel>
void allow_smem(int bytes) {
    static bool done[kMaxDevices] = {};
    int device                    = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= kMaxDevices)
        throw std::runtime_error("moe_grouped_gguf: device index out of range");
    if (!done[device]) {
        CUDA_CHECK(
            cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        done[device] = true;
    }
}

template <Format F>
void launch_gate_up(const GgufGroupedPiece& c, int grid_tiles, cudaStream_t stream) {
    constexpr int smem    = GateUpCfg<F>::kSmem;
    constexpr auto kernel = &grouped_gate_up_kernel<F>;
    allow_smem<kernel>(smem);
    kernel<<<grid_tiles * kGuRowTiles, kThreads, smem, stream>>>(
        c.tiles, c.tile_count, c.expert_gate_up, c.packed_token, c.x, c.h);
    CUDA_CHECK(cudaGetLastError());
}

template <Format F>
void launch_down(const GgufGroupedPiece& c, int grid_tiles, cudaStream_t stream) {
    constexpr int smem    = DownCfg<F>::kSmem;
    constexpr auto kernel = &grouped_down_kernel<F>;
    allow_smem<kernel>(smem);
    kernel<<<grid_tiles * kDownRowTiles, kThreads, smem, stream>>>(c.tiles, c.tile_count,
                                                                   c.expert_down, c.h, c.grouped);
    CUDA_CHECK(cudaGetLastError());
}

std::size_t align256(std::size_t v) { return (v + 255) / 256 * 256; }

} // namespace

GgufGroupedLayout gguf_grouped_layout(std::int32_t columns) {
    const auto pairs = static_cast<std::size_t>(columns) * kTopK;
    const auto tiles =
        static_cast<std::size_t>(gguf_grouped_max_tiles(columns, kGgufGroupedMaxExperts));
    constexpr auto experts = static_cast<std::size_t>(kGgufGroupedMaxExperts);
    GgufGroupedLayout l{};
    std::size_t at = 0;
    auto take      = [&](std::size_t bytes) {
        const std::size_t here = at;
        at += align256(bytes);
        return here;
    };
    l.counts         = take(experts * 4);
    l.offsets        = take((experts + 1) * 4);
    l.rank           = take(pairs * 4);
    l.packed_token   = take(pairs * 4);
    l.packed_index   = take(pairs * 4);
    l.tiles          = take(tiles * sizeof(int4));
    l.tile_count     = take(4);
    l.expert_gate_up = take(experts * sizeof(void*));
    l.expert_down    = take(experts * sizeof(void*));
    l.h              = take(pairs * kInter * 2);
    l.grouped        = take(pairs * kHidden * 4);
    l.total          = at;
    return l;
}

void gguf_grouped_launch(const GgufGroupedPiece& c, cudaStream_t stream) {
    const int assignments = c.columns * kTopK;
    const int blocks      = (assignments + 255) / 256;
    const int grid_tiles  = gguf_grouped_max_tiles(c.columns, c.experts.experts);
    CUDA_CHECK(cudaMemsetAsync(c.counts, 0, sizeof(std::int32_t) * c.experts.experts, stream));
    count_kernel<<<blocks, 256, 0, stream>>>(c.ids, c.experts, assignments, c.counts, c.rank);
    CUDA_CHECK(cudaGetLastError());
    scan_kernel<<<1, kScanThreads, 0, stream>>>(c.experts, c.counts, c.offsets, c.tiles,
                                                c.tile_count, c.expert_gate_up, c.expert_down);
    CUDA_CHECK(cudaGetLastError());
    place_kernel<<<blocks, 256, 0, stream>>>(c.ids, c.rank, assignments, c.offsets, c.packed_token,
                                             c.packed_index);
    CUDA_CHECK(cudaGetLastError());
    switch (c.experts.gate_up_format) {
    case Format::IQ2_S:
        launch_gate_up<Format::IQ2_S>(c, grid_tiles, stream);
        break;
    case Format::IQ3_XXS:
        launch_gate_up<Format::IQ3_XXS>(c, grid_tiles, stream);
        break;
    case Format::IQ3_S:
        launch_gate_up<Format::IQ3_S>(c, grid_tiles, stream);
        break;
    case Format::IQ4_XS:
        launch_gate_up<Format::IQ4_XS>(c, grid_tiles, stream);
        break;
    case Format::IQ4_NL:
        launch_gate_up<Format::IQ4_NL>(c, grid_tiles, stream);
        break;
    default:
        throw std::invalid_argument("moe_grouped_gguf: unsupported gate/up format");
    }
    switch (c.experts.down_format) {
    case Format::IQ4_NL:
        launch_down<Format::IQ4_NL>(c, grid_tiles, stream);
        break;
    case Format::Q2_0:
        launch_down<Format::Q2_0>(c, grid_tiles, stream);
        break;
    default:
        throw std::invalid_argument("moe_grouped_gguf: unsupported down format");
    }
    merge_kernel<<<c.columns, kMergeThreads, 0, stream>>>(c.grouped, c.packed_index, c.weights,
                                                          c.out);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
