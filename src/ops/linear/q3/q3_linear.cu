// q3_g128_fp16 Linear, residual-add, SwiGLU and row-split epilogues.
// See q3_linear.h for the contract and q3_rowsplit_storage.cuh for the storage.
#include "ops/linear/q3/q3_linear.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/launcher/kernel_attr_once.h"
#include "ops/linear/q3/q3_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

using S = Q3RowSplitStorage;

struct Q3Operands {
    const __nv_bfloat16* x; // [K,T]: token t starts at x + t * k
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    std::int64_t code_row_bytes;
    std::int64_t scale_row_bytes;
    std::int32_t rows; // weight rows N
    std::int32_t k;
    std::int32_t tokens;
};

__device__ __forceinline__ float q3_scale(const Q3Operands& op, int row, int group) {
    const auto* p = reinterpret_cast<const __half*>(op.scales + row * op.scale_row_bytes) + group;
    return __half2float(__ldg(p));
}

// ---------------------------------------------------------------------------------------------
// Epilogues. store(row, token, value) receives the FP32 result of weight row `row`.

struct SegmentsEpilogue {
    Q3RowSegments seg;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        int begin = 0;
#pragma unroll
        for (int s = 0; s < Q3RowSegments::kMax; ++s) {
            if (s < seg.count && row < seg.end[s]) {
                seg.data[s][token * seg.ld[s] + (row - begin)] = __float2bfloat16_rn(value);
                return;
            }
            if (s < seg.count) begin = seg.end[s];
        }
    }

    // Rows row..row+7 of one token (row % 8 == 0): one 16-byte store when they lie in one segment
    // at a 16-byte aligned place, else element by element (rows past the last segment dropped).
    __device__ __forceinline__ void store8(int row, int token, const float (&v)[8]) const {
        int begin = 0;
#pragma unroll
        for (int s = 0; s < Q3RowSegments::kMax; ++s) {
            if (s < seg.count && row < seg.end[s]) {
                __nv_bfloat16* p = seg.data[s] + token * seg.ld[s] + (row - begin);
                if (row + 8 <= seg.end[s] && (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0) {
                    uint4 o;
                    auto* h = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        h[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
                    *reinterpret_cast<uint4*>(p) = o;
                    return;
                }
                break;
            }
            if (s < seg.count) begin = seg.end[s];
        }
        const int rows = seg.end[seg.count - 1];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (row + i < rows) store(row + i, token, v[i]);
    }
};

struct AddEpilogue {
    __nv_bfloat16* residual; // [N,T]
    std::int32_t rows;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        __nv_bfloat16* p = residual + static_cast<std::int64_t>(token) * rows + row;
        *p               = __float2bfloat16_rn(__bfloat162float(*p) + value);
    }

    __device__ __forceinline__ void store8(int row, int token, const float (&v)[8]) const {
        __nv_bfloat16* p = residual + static_cast<std::int64_t>(token) * rows + row;
        if (row + 8 <= rows && (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0) {
            uint4 o = *reinterpret_cast<const uint4*>(p);
            auto* h = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 r = __bfloat1622float2(h[i]);
                h[i]           = __floats2bfloat162_rn(r.x + v[2 * i], r.y + v[2 * i + 1]);
            }
            *reinterpret_cast<uint4*>(p) = o;
            return;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (row + i < rows) store(row + i, token, v[i]);
    }
};

// SwiGLU writes out[(N/2) x T] from a gate row and its up row (row + N/2).
struct SwigluOut {
    __nv_bfloat16* out;
    std::int32_t half_rows;

    __device__ __forceinline__ void store(int row, int token, float gate, float up) const {
        out[static_cast<std::int64_t>(token) * half_rows + row] =
            __float2bfloat16_rn(silu(gate) * up);
    }

    __device__ __forceinline__ void store8(int row, int token, const float (&gate)[8],
                                           const float (&up)[8]) const {
        __nv_bfloat16* p = out + static_cast<std::int64_t>(token) * half_rows + row;
        if (row + 8 <= half_rows && (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0) {
            uint4 o;
            auto* h = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
            for (int i = 0; i < 4; ++i)
                h[i] = __floats2bfloat162_rn(silu(gate[2 * i]) * up[2 * i],
                                             silu(gate[2 * i + 1]) * up[2 * i + 1]);
            *reinterpret_cast<uint4*>(p) = o;
            return;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (row + i < half_rows) store(row + i, token, gate[i], up[i]);
    }
};

// ---------------------------------------------------------------------------------------------
// GEMM, T > 8: a CTA computes 128 weight rows x 64 tokens with 8 warps (4 row x 2 token warps,
// each 32 x 32 = 2 m16 x 4 n8 tiles of mma.sync.m16n8k16 BF16 -> FP32). One K step = one group:
// codes become exact BF16 integers in shared memory, the step's FP32 partial is scaled by the
// row's binary16 multiplier and added to the running sum. SwiGLU tiles take 64 gate rows and their
// 64 up rows; the sums meet in shared memory before silu(gate) * up.

constexpr int kTileRows    = 128;
constexpr int kTileTokens  = 64;
constexpr int kTilePitch   = S::kGroupK + 8; // BF16 per staged row (conflict-free u32 reads)
constexpr int kGemmThreads = 256;
constexpr int kASmemBytes  = kTileRows * kTilePitch * 2;
constexpr int kBSmemBytes  = kTileTokens * kTilePitch * 2;
constexpr int kGemmSmem    = kASmemBytes + kBSmemBytes;
constexpr int kCPitch      = kTileTokens + 4; // FP32 per staged row in the SwiGLU epilogue
static_assert(kTileRows * kCPitch * 4 <= kASmemBytes, "SwiGLU epilogue reuses the A tile");

__device__ __forceinline__ void mma_bf16_16816(float (&d)[4], const std::uint32_t (&a)[4],
                                               const std::uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Weight row of tile row `i` (0..127) of tile `tile`.
template <bool Swiglu>
__device__ __forceinline__ int q3_tile_row(const Q3Operands& op, int tile, int i) {
    if constexpr (Swiglu) {
        const int half = op.rows / 2;
        return i < 64 ? tile * 64 + i : half + tile * 64 + (i - 64);
    } else {
        return tile * kTileRows + i;
    }
}

template <bool Swiglu>
__device__ __forceinline__ bool q3_tile_row_live(const Q3Operands& op, int tile, int i) {
    if constexpr (Swiglu) {
        return tile * 64 + (i & 63) < op.rows / 2;
    } else {
        return tile * kTileRows + i < op.rows;
    }
}

template <bool Swiglu, class Epilogue>
__global__ void __launch_bounds__(kGemmThreads) q3_gemm_kernel(Q3Operands op, Epilogue epilogue) {
    extern __shared__ __align__(16) unsigned char smem[];
    auto* as        = reinterpret_cast<__nv_bfloat16*>(smem);
    auto* bs        = reinterpret_cast<__nv_bfloat16*>(smem + kASmemBytes);
    const int tile  = blockIdx.x;
    const int t0    = blockIdx.y * kTileTokens;
    const int warp  = threadIdx.x >> 5;
    const int lane  = threadIdx.x & 31;
    const int wr    = warp >> 1; // rows 32 * wr
    const int wt    = warp & 1;  // tokens 32 * wt
    const int g     = lane >> 2;
    const int tq    = lane & 3;
    const int steps = op.k / S::kGroupK;

    float total[2][4][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e) total[mi][ni][e] = 0.0F;

    for (int step = 0; step < steps; ++step) {
        __syncthreads();
        // Activations: 64 tokens x 128 values = 1024 vectors of 8 BF16.
        for (int i = threadIdx.x; i < kTileTokens * 16; i += kGemmThreads) {
            const int t = i >> 4;
            const int v = i & 15;
            uint4 value = make_uint4(0, 0, 0, 0);
            if (t0 + t < op.tokens) {
                value = __ldg(reinterpret_cast<const uint4*>(
                    op.x + static_cast<std::int64_t>(t0 + t) * op.k + step * S::kGroupK + v * 8));
            }
            *reinterpret_cast<uint4*>(bs + t * kTilePitch + v * 8) = value;
        }
        // Weights: 128 rows x 4 units of 32 codes -> exact BF16 integers.
        for (int i = threadIdx.x; i < kTileRows * S::kUnitsGroup; i += kGemmThreads) {
            const int r = i >> 2;
            const int u = i & 3;
            float codes[32];
            if (q3_tile_row_live<Swiglu>(op, tile, r)) {
                const int row = q3_tile_row<Swiglu>(op, tile, r);
                const auto* p = reinterpret_cast<const std::uint32_t*>(
                    op.codes + row * op.code_row_bytes +
                    static_cast<std::int64_t>(step * S::kUnitsGroup + u) * S::kUnitBytes);
                q3_decode_unit(__ldg(p), __ldg(p + 1), __ldg(p + 2), codes);
            } else {
#pragma unroll
                for (int j = 0; j < 32; ++j) codes[j] = 0.0F;
            }
            auto* dst = reinterpret_cast<__nv_bfloat162*>(as + r * kTilePitch + u * 32);
#pragma unroll
            for (int j = 0; j < 16; ++j)
                dst[j] = __floats2bfloat162_rn(codes[2 * j], codes[2 * j + 1]);
        }
        __syncthreads();

        float part[2][4][4];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 4; ++ni)
#pragma unroll
                for (int e = 0; e < 4; ++e) part[mi][ni][e] = 0.0F;
#pragma unroll
        for (int kk = 0; kk < S::kGroupK; kk += 16) {
            std::uint32_t a[2][4], b[4][2];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const __nv_bfloat16* base = as + (wr * 32 + mi * 16 + g) * kTilePitch + kk + 2 * tq;
                a[mi][0]                  = *reinterpret_cast<const std::uint32_t*>(base);
                a[mi][1] = *reinterpret_cast<const std::uint32_t*>(base + 8 * kTilePitch);
                a[mi][2] = *reinterpret_cast<const std::uint32_t*>(base + 8);
                a[mi][3] = *reinterpret_cast<const std::uint32_t*>(base + 8 * kTilePitch + 8);
            }
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const __nv_bfloat16* base = bs + (wt * 32 + ni * 8 + g) * kTilePitch + kk + 2 * tq;
                b[ni][0]                  = *reinterpret_cast<const std::uint32_t*>(base);
                b[ni][1]                  = *reinterpret_cast<const std::uint32_t*>(base + 8);
            }
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) mma_bf16_16816(part[mi][ni], a[mi], b[ni]);
        }
        // Scale this group's partial by each row's multiplier (rows g and g + 8 of every m tile).
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            const int r0   = wr * 32 + mi * 16 + g;
            const float s0 = q3_tile_row_live<Swiglu>(op, tile, r0)
                                 ? q3_scale(op, q3_tile_row<Swiglu>(op, tile, r0), step)
                                 : 0.0F;
            const float s1 = q3_tile_row_live<Swiglu>(op, tile, r0 + 8)
                                 ? q3_scale(op, q3_tile_row<Swiglu>(op, tile, r0 + 8), step)
                                 : 0.0F;
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                total[mi][ni][0] = fmaf(s0, part[mi][ni][0], total[mi][ni][0]);
                total[mi][ni][1] = fmaf(s0, part[mi][ni][1], total[mi][ni][1]);
                total[mi][ni][2] = fmaf(s1, part[mi][ni][2], total[mi][ni][2]);
                total[mi][ni][3] = fmaf(s1, part[mi][ni][3], total[mi][ni][3]);
            }
        }
    }

    if constexpr (Swiglu) {
        __syncthreads();
        auto* cs = reinterpret_cast<float*>(smem);
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 4; ++ni)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int r         = wr * 32 + mi * 16 + g + (e >= 2 ? 8 : 0);
                    const int t         = wt * 32 + ni * 8 + 2 * tq + (e & 1);
                    cs[r * kCPitch + t] = total[mi][ni][e];
                }
        __syncthreads();
        for (int i = threadIdx.x; i < 64 * kTileTokens; i += kGemmThreads) {
            const int r = i / kTileTokens;
            const int t = i - r * kTileTokens;
            if (tile * 64 + r < op.rows / 2 && t0 + t < op.tokens) {
                epilogue.store(tile * 64 + r, t0 + t, cs[r * kCPitch + t],
                               cs[(r + 64) * kCPitch + t]);
            }
        }
    } else {
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 4; ++ni)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int r = tile * kTileRows + wr * 32 + mi * 16 + g + (e >= 2 ? 8 : 0);
                    const int t = t0 + wt * 32 + ni * 8 + 2 * tq + (e & 1);
                    if (r < op.rows && t < op.tokens) epilogue.store(r, t, total[mi][ni][e]);
                }
    }
}

// ---------------------------------------------------------------------------------------------
// Register-direct small-T route (T <= 8, K % 128 == 0). Weights are the MMA A operand
// (M = 16 rows), activations the B operand (N = 8 token columns). The K order inside each 16-wide
// MMA slice is permuted identically on both operands (a dot product does not depend on the order
// of its terms): thread (g, t) of a warp owns codes [32t, 32t+32) of one 128-code group for rows g
// and g+8, and in slice s feeds codes 32t+4s .. +3, so it decodes them straight from its own three
// 32-bit code words into BF16 fragment registers (exact integers via the 128 + u bit trick) and
// reads the matching activations as two consecutive 32-bit words. No shared memory on the weight
// path; each warp streams 16 rows x 48 bytes per group. The 8 warps of a CTA split the K groups
// and meet once in shared memory; each group's partial is scaled by the row multiplier in FP32.

__device__ __forceinline__ std::uint32_t rd_pair(std::uint32_t u, int f) {
    // Two offset-form fields f, f+1 of a 24-bit word -> BF16x2 bits of (128 + u'), low field low.
    const std::uint32_t lo = (u >> (3 * f)) & 7u;
    const std::uint32_t hi = (u >> (3 * f + 3)) & 7u;
    return lo | (hi << 16) | 0x43004300u;
}

__device__ __forceinline__ std::uint32_t rd_sub132(std::uint32_t biased) {
    const __nv_bfloat162 v = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&biased),
                                     __floats2bfloat162_rn(132.0F, 132.0F));
    return *reinterpret_cast<const std::uint32_t*>(&v);
}

template <int M, bool Swiglu, class Epilogue>
__global__ void __launch_bounds__(256) q3_rd_kernel(Q3Operands op, Epilogue epilogue) {
    static_assert(M == 1 || M == 2);
    __shared__ float partial[8][M][32][4];
    const int tile   = blockIdx.x;
    const int warp   = threadIdx.x >> 5;
    const int lane   = threadIdx.x & 31;
    const int g      = lane >> 2;
    const int tq     = lane & 3;
    const int groups = op.k / S::kGroupK;
    // Rows of this thread: tile m holds rows base_m + {g, g + 8}.
    int row[M][2];
    bool live[M][2];
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const int base = Swiglu ? (m == 0 ? tile * 16 : op.rows / 2 + tile * 16) : tile * 16;
        const int lim  = Swiglu ? (m == 0 ? op.rows / 2 : op.rows) : op.rows;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            row[m][h]  = base + g + 8 * h;
            live[m][h] = row[m][h] < lim && (!Swiglu || tile * 16 + g + 8 * h < op.rows / 2);
            if (!live[m][h]) row[m][h] = 0;
        }
    }
    const bool token_live     = g < op.tokens; // B column n = g is token g
    const __nv_bfloat16* xrow = op.x + static_cast<std::int64_t>(token_live ? g : 0) * op.k;
    float total[M][4];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
        for (int e = 0; e < 4; ++e) total[m][e] = 0.0F;

    for (int grp = warp; grp < groups; grp += 8) {
        // Activations: x[g][grp*128 + 32t .. +32] as 16 BF16 pairs (zero for padded tokens).
        uint4 xv[4];
        const auto* xp = reinterpret_cast<const uint4*>(xrow + grp * S::kGroupK + 32 * tq);
#pragma unroll
        for (int q = 0; q < 4; ++q) xv[q] = token_live ? __ldg(xp + q) : make_uint4(0, 0, 0, 0);
        const std::uint32_t* xw = reinterpret_cast<const std::uint32_t*>(xv);
        std::uint32_t word[M][2][4]; // 24-bit offset-form code words per row
        float scale[M][2];
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const auto* p = reinterpret_cast<const std::uint32_t*>(
                    op.codes + row[m][h] * op.code_row_bytes +
                    static_cast<std::int64_t>(grp) * S::kCodeBytesPerGroup + tq * S::kUnitBytes);
                std::uint32_t w0 = 0, w1 = 0, w2 = 0;
                if (live[m][h]) {
                    w0 = __ldg(p);
                    w1 = __ldg(p + 1);
                    w2 = __ldg(p + 2);
                }
                std::uint32_t w24[4];
                q3_unit_words(w0, w1, w2, w24);
#pragma unroll
                for (int j = 0; j < 4; ++j) word[m][h][j] = w24[j] ^ kQ3SignFlip24;
                scale[m][h] = live[m][h] ? q3_scale(op, row[m][h], grp) : 0.0F;
            }
        float part[M][4];
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
            for (int e = 0; e < 4; ++e) part[m][e] = 0.0F;
#pragma unroll
        for (int sl = 0; sl < 8; ++sl) {
            const std::uint32_t b[2] = {xw[2 * sl], xw[2 * sl + 1]};
            const int j              = sl >> 1;      // 24-bit word holding codes 4sl..4sl+3
            const int f              = (sl & 1) * 4; // first field inside it
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const std::uint32_t a[4] = {rd_sub132(rd_pair(word[m][0][j], f)),
                                            rd_sub132(rd_pair(word[m][1][j], f)),
                                            rd_sub132(rd_pair(word[m][0][j], f + 2)),
                                            rd_sub132(rd_pair(word[m][1][j], f + 2))};
                mma_bf16_16816(part[m], a, b);
            }
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
            total[m][0] = fmaf(scale[m][0], part[m][0], total[m][0]);
            total[m][1] = fmaf(scale[m][0], part[m][1], total[m][1]);
            total[m][2] = fmaf(scale[m][1], part[m][2], total[m][2]);
            total[m][3] = fmaf(scale[m][1], part[m][3], total[m][3]);
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
        for (int e = 0; e < 4; ++e) partial[warp][m][lane][e] = total[m][e];
    __syncthreads();
    if (warp != 0) return;
    float sum[M][4];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            float v = 0.0F;
#pragma unroll
            for (int w = 0; w < 8; ++w) v += partial[w][m][lane][e];
            sum[m][e] = v;
        }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int h = e >= 2 ? 1 : 0;
        const int t = 2 * tq + (e & 1);
        if (t >= op.tokens) continue;
        if constexpr (Swiglu) {
            if (live[0][h]) epilogue.store(tile * 16 + g + 8 * h, t, sum[0][e], sum[1][e]);
        } else {
            if (live[0][h]) epilogue.store(row[0][h], t, sum[0][e]);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// A8 tile route for prefill widths (T >= 64; the fallback of the tall route below when the caller
// has no workspace): int8 tensor cores
// (mma.sync m16n8k32 s8). Each CTA quantizes its activation tile per (token, 128-value group) to
// int8 with an FP32 scale (amax / 127, round to nearest) in shared memory; the 3-bit codes are
// exact int8, decoded straight into A-fragment registers with the same K permutation trick as the
// register-direct small-T route (thread t owns codes [32t, 32t+32) of a group; slice s takes codes
// 32t+8s .. +7). Per group: int32 partial -> FP32 * weight multiplier * activation scale. The grid
// walks token tiles fastest so a row tile's codes are reused from L2 by every token tile.
// This changes the arithmetic (activation quantization): NINFER_Q3_A8=0 keeps the A16 tiles.

__device__ __forceinline__ void mma_s8_16832(int (&d)[4], const std::uint32_t (&a)[4],
                                             const std::uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Four offset-form 3-bit fields of a 24-bit word starting at field f -> four signed int8 lanes.
__device__ __forceinline__ std::uint32_t a8_quad(std::uint32_t u, int f) {
    const std::uint32_t v = u >> (3 * f);
    const std::uint32_t packed =
        (v & 7u) | (((v >> 3) & 7u) << 8) | (((v >> 6) & 7u) << 16) | (((v >> 9) & 7u) << 24);
    return __vsub4(packed, 0x04040404u);
}

constexpr int kA8XPitch = S::kGroupK + 8; // BF16 per staged token row (272 B)
constexpr int kA8QPitch = S::kGroupK + 4; // int8 per quantized token row (132 B: conflict-free)

template <int M, bool Swiglu, class Epilogue>
__global__ void __launch_bounds__(256) q3_a8_gemm_kernel(Q3Operands op, Epilogue epilogue) {
    static_assert(M == 1 || M == 2);
    constexpr int Tt = M == 1 ? 64 : 32;
    constexpr int NT = Tt / 8;
    __shared__ __align__(16) std::uint16_t xs[2][Tt][kA8XPitch];
    __shared__ __align__(16) std::int8_t xq[Tt][kA8QPitch];
    __shared__ float sx[Tt];
    const int t0       = blockIdx.x * Tt;
    const int rt       = blockIdx.y;
    const int warp     = threadIdx.x >> 5;
    const int lane     = threadIdx.x & 31;
    const int g        = lane >> 2;
    const int tq       = lane & 3;
    const int groups   = op.k / S::kGroupK;
    const int out_rows = Swiglu ? op.rows / 2 : op.rows;
    int row[M][2];
    bool live[M][2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int o = rt * 128 + warp * 16 + g + 8 * h;
#pragma unroll
        for (int m = 0; m < M; ++m) {
            live[m][h] = o < out_rows;
            row[m][h]  = live[m][h] ? (m == 0 ? o : op.rows / 2 + o) : 0;
        }
    }
    auto issue = [&](int grp, int buf) {
        for (int i = threadIdx.x; i < Tt * 16; i += 256) {
            const int t   = i >> 4;
            const int v   = i & 15;
            const bool ok = t0 + t < op.tokens;
            const auto* src =
                op.x + static_cast<std::int64_t>(ok ? t0 + t : 0) * op.k + grp * S::kGroupK + v * 8;
            cp_async_zfill<16, Cache::cg>(&xs[buf][t][v * 8], src, ok ? 16 : 0);
        }
        cp_commit();
    };
    float total[M][NT][4];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) total[m][n][e] = 0.0F;
    issue(0, 0);
    for (int grp = 0; grp < groups; ++grp) {
        // Weight codes and multipliers of this group (registers), issued before the barriers.
        std::uint32_t word[M][2][4];
        float sw[M][2];
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const auto* p = reinterpret_cast<const std::uint32_t*>(
                    op.codes + row[m][h] * op.code_row_bytes +
                    static_cast<std::int64_t>(grp) * S::kCodeBytesPerGroup + tq * S::kUnitBytes);
                std::uint32_t w0 = 0, w1 = 0, w2 = 0;
                if (live[m][h]) {
                    w0 = __ldg(p);
                    w1 = __ldg(p + 1);
                    w2 = __ldg(p + 2);
                }
                std::uint32_t w24[4];
                q3_unit_words(w0, w1, w2, w24);
#pragma unroll
                for (int j = 0; j < 4; ++j) word[m][h][j] = w24[j] ^ kQ3SignFlip24;
                sw[m][h] = live[m][h] ? q3_scale(op, row[m][h], grp) : 0.0F;
            }
        if (grp + 1 < groups)
            issue(grp + 1, (grp + 1) & 1);
        else
            cp_commit();
        cp_wait<1>();
        __syncthreads(); // x tile of grp landed; previous MMAs are done with xq
        // Quantize: warp w handles tokens w, w+8, ...; lane holds 4 values.
        for (int t = warp; t < Tt; t += 8) {
            const std::uint16_t* src = &xs[grp & 1][t][lane * 4];
            float v[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
                v[i] = __uint_as_float(static_cast<std::uint32_t>(src[i]) << 16);
            float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
            const float scale    = amax > 0.0F ? amax / 127.0F : 0.0F;
            const float inv      = amax > 0.0F ? 127.0F / amax : 0.0F;
            std::uint32_t packed = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int q = max(-127, min(127, __float2int_rn(v[i] * inv)));
                packed |= (static_cast<std::uint32_t>(q) & 0xffu) << (8 * i);
            }
            *reinterpret_cast<std::uint32_t*>(&xq[t][lane * 4]) = packed;
            if (lane == 0) sx[t] = scale;
        }
        __syncthreads();
        int part[M][NT][4];
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
            for (int n = 0; n < NT; ++n)
#pragma unroll
                for (int e = 0; e < 4; ++e) part[m][n][e] = 0;
#pragma unroll
        for (int sl = 0; sl < 4; ++sl) {
            std::uint32_t a[M][4];
#pragma unroll
            for (int m = 0; m < M; ++m) {
                a[m][0] = a8_quad(word[m][0][sl], 0);
                a[m][1] = a8_quad(word[m][1][sl], 0);
                a[m][2] = a8_quad(word[m][0][sl], 4);
                a[m][3] = a8_quad(word[m][1][sl], 4);
            }
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const std::int8_t* bp    = &xq[n * 8 + g][32 * tq + 8 * sl];
                const std::uint32_t b[2] = {*reinterpret_cast<const std::uint32_t*>(bp),
                                            *reinterpret_cast<const std::uint32_t*>(bp + 4)};
#pragma unroll
                for (int m = 0; m < M; ++m) mma_s8_16832(part[m][n], a[m], b);
            }
        }
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const float s0 = sx[n * 8 + 2 * tq];
            const float s1 = sx[n * 8 + 2 * tq + 1];
#pragma unroll
            for (int m = 0; m < M; ++m) {
                total[m][n][0] =
                    fmaf(sw[m][0] * s0, static_cast<float>(part[m][n][0]), total[m][n][0]);
                total[m][n][1] =
                    fmaf(sw[m][0] * s1, static_cast<float>(part[m][n][1]), total[m][n][1]);
                total[m][n][2] =
                    fmaf(sw[m][1] * s0, static_cast<float>(part[m][n][2]), total[m][n][2]);
                total[m][n][3] =
                    fmaf(sw[m][1] * s1, static_cast<float>(part[m][n][3]), total[m][n][3]);
            }
        }
    }
    cp_wait<0>();
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int h = e >= 2 ? 1 : 0;
            const int t = t0 + n * 8 + 2 * tq + (e & 1);
            if (t >= op.tokens || !live[0][h]) continue;
            if constexpr (Swiglu) {
                epilogue.store(rt * 128 + warp * 16 + g + 8 * h, t, total[0][n][e], total[1][n][e]);
            } else {
                epilogue.store(row[0][h], t, total[0][n][e]);
            }
        }
}

// ---------------------------------------------------------------------------------------------
// A8 tall route: the arithmetic of q3_a8_gemm_kernel, bit for bit, without its
// per-CTA overheads.
//  1. q3_a8_quantize_kernel quantizes the activations once per call (per token and 128-value
//     group, the same amax / 127 rule and the same lane split) into a caller workspace:
//     xq [T][K] int8 and sx [K/128][Tp] FP32, token-contiguous per group (Tp = T rounded up to 4,
//     pad entries zero).
//  2. q3_a8t_kernel: a CTA computes 128 weight rows x 128 tokens (SwiGLU: 64 gate rows and their 64
//     up rows) with 8 warps (4 row warps x 2 token warps, 32 rows x 64 tokens each) and walks K one
//     group per stage through a 3-stage cp.async ring (the group's int8 activation tile, raw 3-bit
//     codes, activation scales); one barrier per group. Codes go shared memory -> registers ->
//     int8 A fragments with the K permutation of the register-direct route; B fragments are two
//     16-byte shared loads per token per group. Per group: int32 partial -> FP32 times (weight
//     multiplier * activation scale), accumulated in group order exactly like q3_a8_gemm_kernel.

constexpr int kA8tRows       = 128;
constexpr int kA8tTokens     = 128;
constexpr int kA8tStages     = 3;
constexpr int kA8tXPitch     = S::kGroupK + 16; // int8 per staged token row (144 B: conflict-free)
constexpr int kA8tXBytes     = kA8tTokens * kA8tXPitch;
constexpr int kA8tCBytes     = kA8tRows * S::kCodeBytesPerGroup;
constexpr int kA8tSBytes     = kA8tTokens * 4;
constexpr int kA8tStageBytes = kA8tXBytes + kA8tCBytes + kA8tSBytes;
constexpr int kA8tSmem       = kA8tStages * kA8tStageBytes;
constexpr int kA8tCPitch     = kA8tRows + 4; // FP32 per staged token row in the epilogue
static_assert(kA8tTokens * kA8tCPitch * 4 <= kA8tSmem, "the epilogue reuses the stage ring");
static_assert(kA8tXBytes % 16 == 0 && kA8tCBytes % 16 == 0 && kA8tStageBytes % 16 == 0);

// One warp per (token, group); lane l quantizes values 4l..4l+3 (the q3_a8_gemm_kernel split).
__global__ void __launch_bounds__(256)
    q3_a8_quantize_kernel(const __nv_bfloat16* __restrict__ x, int tokens, int k, int tp,
                          std::int8_t* __restrict__ xq, float* __restrict__ sx) {
    const int groups = k / S::kGroupK;
    const std::int64_t item =
        static_cast<std::int64_t>(blockIdx.x) * 8 + static_cast<std::int64_t>(threadIdx.x >> 5);
    if (item >= static_cast<std::int64_t>(tokens) * groups) return;
    const int t           = static_cast<int>(item / groups);
    const int grp         = static_cast<int>(item - static_cast<std::int64_t>(t) * groups);
    const int lane        = threadIdx.x & 31;
    const std::int64_t at = static_cast<std::int64_t>(t) * k + grp * S::kGroupK + lane * 4;
    const uint2 raw       = __ldg(reinterpret_cast<const uint2*>(x + at));
    float v[4]            = {__uint_as_float(raw.x << 16), __uint_as_float(raw.x & 0xffff0000u),
                             __uint_as_float(raw.y << 16), __uint_as_float(raw.y & 0xffff0000u)};
    float amax            = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    const float scale    = amax > 0.0F ? amax / 127.0F : 0.0F;
    const float inv      = amax > 0.0F ? 127.0F / amax : 0.0F;
    std::uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int q = max(-127, min(127, __float2int_rn(v[i] * inv)));
        packed |= (static_cast<std::uint32_t>(q) & 0xffu) << (8 * i);
    }
    *reinterpret_cast<std::uint32_t*>(xq + at) = packed;
    if (lane == 0) {
        float* row = sx + static_cast<std::int64_t>(grp) * tp;
        row[t]     = scale;
        if (t == tokens - 1) {
            for (int p = tokens; p < tp; ++p) row[p] = 0.0F;
        }
    }
}

template <bool Swiglu>
__device__ __forceinline__ int a8t_row(const Q3Operands& op, int rt, int i) {
    if constexpr (Swiglu) {
        return i < 64 ? rt * 64 + i : op.rows / 2 + rt * 64 + (i - 64);
    } else {
        return rt * kA8tRows + i;
    }
}

template <bool Swiglu>
__device__ __forceinline__ bool a8t_row_live(const Q3Operands& op, int rt, int i) {
    if constexpr (Swiglu) {
        return rt * 64 + (i & 63) < op.rows / 2;
    } else {
        return rt * kA8tRows + i < op.rows;
    }
}

template <bool Swiglu, class Epilogue>
__global__ void __launch_bounds__(256, 1)
    q3_a8t_kernel(Q3Operands op, const std::int8_t* __restrict__ xq, const float* __restrict__ sx,
                  int tp, Epilogue epilogue) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int t0     = blockIdx.x * kA8tTokens;
    const int rt     = blockIdx.y;
    const int warp   = threadIdx.x >> 5;
    const int lane   = threadIdx.x & 31;
    const int wr     = warp >> 1; // row warp
    const int wt     = warp & 1;  // token warp: tokens 64 * wt
    const int g      = lane >> 2;
    const int tq     = lane & 3;
    const int groups = op.k / S::kGroupK;

    // Tile-local row of (m tile, half): SwiGLU m = 0 gate rows, m = 1 the matching up rows.
    int lrow[2][2];
    int grow[2][2];
    bool live[2][2];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            lrow[m][h] = Swiglu ? m * 64 + wr * 16 + g + 8 * h : wr * 32 + m * 16 + g + 8 * h;
            live[m][h] = a8t_row_live<Swiglu>(op, rt, lrow[m][h]);
            grow[m][h] = live[m][h] ? a8t_row<Swiglu>(op, rt, lrow[m][h]) : 0;
        }

    auto load = [&](int grp, int buf) {
        unsigned char* base = smem + buf * kA8tStageBytes;
        for (int i = threadIdx.x; i < kA8tTokens * 8; i += 256) {
            const int t   = i >> 3;
            const int c   = i & 7;
            const bool ok = t0 + t < op.tokens;
            const std::int8_t* src =
                xq + static_cast<std::int64_t>(ok ? t0 + t : 0) * op.k + grp * S::kGroupK + c * 16;
            cp_async_zfill<16, Cache::cg>(base + t * kA8tXPitch + c * 16, src, ok ? 16 : 0);
        }
        for (int i = threadIdx.x; i < kA8tRows * 3; i += 256) {
            const int r   = i / 3;
            const int c   = i - 3 * r;
            const bool ok = a8t_row_live<Swiglu>(op, rt, r);
            const std::uint8_t* src =
                op.codes +
                static_cast<std::int64_t>(ok ? a8t_row<Swiglu>(op, rt, r) : 0) * op.code_row_bytes +
                static_cast<std::int64_t>(grp) * S::kCodeBytesPerGroup + c * 16;
            cp_async_zfill<16, Cache::cg>(base + kA8tXBytes + r * S::kCodeBytesPerGroup + c * 16,
                                          src, ok ? 16 : 0);
        }
        if (threadIdx.x < kA8tTokens / 4) {
            const int t      = t0 + threadIdx.x * 4;
            const bool ok    = t < tp;
            const float* src = sx + static_cast<std::int64_t>(grp) * tp + (ok ? t : 0);
            cp_async_zfill<16, Cache::cg>(base + kA8tXBytes + kA8tCBytes + threadIdx.x * 16, src,
                                          ok ? 16 : 0);
        }
    };

    float total[2][8][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) total[m][n][e] = 0.0F;

#pragma unroll
    for (int s = 0; s < kA8tStages - 1; ++s) {
        if (s < groups) load(s, s);
        cp_commit();
    }
    for (int grp = 0; grp < groups; ++grp) {
        cp_wait<kA8tStages - 2>();
        __syncthreads(); // stage grp landed; every warp is done with the buffer reloaded below
        {
            const int next = grp + kA8tStages - 1;
            if (next < groups) load(next, next % kA8tStages);
            cp_commit();
        }
        const unsigned char* base = smem + (grp % kA8tStages) * kA8tStageBytes;
        const std::int8_t* xs     = reinterpret_cast<const std::int8_t*>(base);
        const unsigned char* cs   = base + kA8tXBytes;
        const float* ss           = reinterpret_cast<const float*>(base + kA8tXBytes + kA8tCBytes);

        float sw[2][2];
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
            for (int h = 0; h < 2; ++h)
                sw[m][h] = live[m][h] ? q3_scale(op, grow[m][h], grp) : 0.0F;

        std::uint32_t a[2][4][4];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            std::uint32_t word[2][4];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const auto* p = reinterpret_cast<const std::uint32_t*>(
                    cs + lrow[m][h] * S::kCodeBytesPerGroup + tq * S::kUnitBytes);
                std::uint32_t w24[4];
                q3_unit_words(p[0], p[1], p[2], w24);
#pragma unroll
                for (int j = 0; j < 4; ++j) word[h][j] = w24[j] ^ kQ3SignFlip24;
            }
#pragma unroll
            for (int sl = 0; sl < 4; ++sl) {
                a[m][sl][0] = a8_quad(word[0][sl], 0);
                a[m][sl][1] = a8_quad(word[1][sl], 0);
                a[m][sl][2] = a8_quad(word[0][sl], 4);
                a[m][sl][3] = a8_quad(word[1][sl], 4);
            }
        }

        int part[2][8][4];
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
            for (int n = 0; n < 8; ++n)
#pragma unroll
                for (int e = 0; e < 4; ++e) part[m][n][e] = 0;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            const std::int8_t* bp       = xs + (wt * 64 + n * 8 + g) * kA8tXPitch + 32 * tq;
            const uint4 lo              = *reinterpret_cast<const uint4*>(bp);
            const uint4 hi              = *reinterpret_cast<const uint4*>(bp + 16);
            const std::uint32_t b[4][2] = {{lo.x, lo.y}, {lo.z, lo.w}, {hi.x, hi.y}, {hi.z, hi.w}};
#pragma unroll
            for (int sl = 0; sl < 4; ++sl)
#pragma unroll
                for (int m = 0; m < 2; ++m) mma_s8_16832(part[m][n], a[m][sl], b[sl]);
        }
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            const float s0 = ss[wt * 64 + n * 8 + 2 * tq];
            const float s1 = ss[wt * 64 + n * 8 + 2 * tq + 1];
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                total[m][n][0] =
                    fmaf(sw[m][0] * s0, static_cast<float>(part[m][n][0]), total[m][n][0]);
                total[m][n][1] =
                    fmaf(sw[m][0] * s1, static_cast<float>(part[m][n][1]), total[m][n][1]);
                total[m][n][2] =
                    fmaf(sw[m][1] * s0, static_cast<float>(part[m][n][2]), total[m][n][2]);
                total[m][n][3] =
                    fmaf(sw[m][1] * s1, static_cast<float>(part[m][n][3]), total[m][n][3]);
            }
        }
    }
    // Epilogue: the FP32 tile meets in shared memory as [token][row], then every thread stores
    // 8 consecutive rows of one token (16-byte accesses; the residual reads are independent).
    cp_wait<0>();
    __syncthreads();
    float* cs = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int t                          = wt * 64 + n * 8 + 2 * tq + (e & 1);
                cs[t * kA8tCPitch + lrow[m][e >> 1]] = total[m][n][e];
            }
    __syncthreads();
    constexpr int kVecs = Swiglu ? 8 : 16; // 8-row vectors per token
    for (int i = threadIdx.x; i < kA8tTokens * kVecs; i += 256) {
        const int t = i / kVecs;
        const int r = (i - t * kVecs) * 8;
        if (t0 + t >= op.tokens) continue;
        const float4* c = reinterpret_cast<const float4*>(cs + t * kA8tCPitch + r);
        const float4 c0 = c[0], c1 = c[1];
        const float v[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
        if constexpr (Swiglu) {
            const float4* u = reinterpret_cast<const float4*>(cs + t * kA8tCPitch + 64 + r);
            const float4 u0 = u[0], u1 = u[1];
            const float w[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
            epilogue.store8(rt * 64 + r, t0 + t, v, w);
        } else {
            epilogue.store8(rt * kA8tRows + r, t0 + t, v);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Host side.

constexpr int kSmallTMaxTokens = 8;

Q3Operands q3_operands(const Tensor& x, const Weight& w) {
    const std::int64_t groups = w.padded_shape[1] / S::kGroupK;
    return {static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales),
            groups * S::kCodeBytesPerGroup,
            w.scale_nb[1] != 0 ? w.scale_nb[1] : groups * S::kScaleBytesPerGroup,
            w.n,
            w.k,
            x.ne[1]};
}

void require_activation(const Tensor& x, const Weight& w, const char* label) {
    if (x.dtype != DType::BF16 || x.ne[0] != w.k || x.ne[1] <= 0 || x.ne[2] != 1 || x.ne[3] != 1 ||
        !x.is_contiguous() || x.data == nullptr ||
        (reinterpret_cast<std::uintptr_t>(x.data) & 15u) != 0) {
        throw std::invalid_argument(std::string(label) +
                                    ": x must be contiguous 16-byte-aligned BF16 [K,T]");
    }
}

template <bool Swiglu, class Epilogue>
void launch_gemm(const Q3Operands& op, const Epilogue& epilogue, cudaStream_t stream) {
    static FuncAttrPerDevice attr;
    auto* kernel = q3_gemm_kernel<Swiglu, Epilogue>;
    attr.ensure(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kGemmSmem);
    const int tiles = Swiglu ? (op.rows / 2 + 63) / 64 : (op.rows + kTileRows - 1) / kTileRows;
    const dim3 grid(static_cast<unsigned>(tiles),
                    static_cast<unsigned>((op.tokens + kTileTokens - 1) / kTileTokens));
    if (grid.y > 65535u) throw std::invalid_argument("q3 linear: T too large");
    kernel<<<grid, kGemmThreads, kGemmSmem, stream>>>(op, epilogue);
}

template <bool Swiglu, class Epilogue>
void launch_small_t(const Q3Operands& op, const Epilogue& epilogue, cudaStream_t stream) {
    const int rows = Swiglu ? op.rows / 2 : op.rows;
    const dim3 grid(static_cast<unsigned>((rows + 15) / 16));
    q3_rd_kernel<Swiglu ? 2 : 1, Swiglu, Epilogue><<<grid, 256, 0, stream>>>(op, epilogue);
}

// Route override for tests (q3_force_route_for_test): 0 = automatic.
int g_route_override = 0;

// NINFER_Q3_A8 selects the prefill route for T >= 64. Default (unset or 1) = A8, the tall route
// when the caller passes a workspace (PPL within 0.5 % of A16, prefill ~3.5x); 0 = A16 tiles
// (exact BF16 activations). T < 64 (decode, MTP verify, small batches) is always A16.
int a8_env_mode() {
    static const int mode = [] {
        const char* v = std::getenv("NINFER_Q3_A8");
        if (v == nullptr || v[0] == '\0') return 3;
        return v[0] == '0' ? 1 : 3;
    }();
    return mode;
}

int route_mode() { return g_route_override != 0 ? g_route_override : a8_env_mode(); }

constexpr int kA8MinTokens = 64;

bool a8_enabled(const Q3Operands& op) { return route_mode() >= 2 && op.tokens >= kA8MinTokens; }

std::size_t a8_scratch_code_bytes(std::int64_t tokens, std::int64_t k) {
    return static_cast<std::size_t>((tokens * k + 255) / 256 * 256);
}

std::size_t a8_scratch_bytes(std::int64_t tokens, std::int64_t k) {
    const std::int64_t tp = (tokens + 3) / 4 * 4;
    return a8_scratch_code_bytes(tokens, k) + static_cast<std::size_t>(k / S::kGroupK * tp * 4);
}

template <bool Swiglu, class Epilogue>
void launch_a8(const Q3Operands& op, const Epilogue& epilogue, cudaStream_t stream) {
    constexpr int Tt   = Swiglu ? 32 : 64;
    const int out_rows = Swiglu ? op.rows / 2 : op.rows;
    const dim3 grid(static_cast<unsigned>((op.tokens + Tt - 1) / Tt),
                    static_cast<unsigned>((out_rows + 127) / 128));
    if (grid.y > 65535u) throw std::invalid_argument("q3 linear: too many row tiles");
    q3_a8_gemm_kernel<Swiglu ? 2 : 1, Swiglu, Epilogue><<<grid, 256, 0, stream>>>(op, epilogue);
}

template <bool Swiglu, class Epilogue>
void launch_a8t(const Q3Operands& op, const Epilogue& epilogue, const Q3Workspace& ws,
                cudaStream_t stream) {
    const int tp             = (op.tokens + 3) / 4 * 4;
    const int groups         = op.k / S::kGroupK;
    auto* xq                 = static_cast<std::int8_t*>(ws.data);
    auto* sx                 = reinterpret_cast<float*>(static_cast<unsigned char*>(ws.data) +
                                                        a8_scratch_code_bytes(op.tokens, op.k));
    const std::int64_t items = static_cast<std::int64_t>(op.tokens) * groups;
    q3_a8_quantize_kernel<<<static_cast<unsigned>((items + 7) / 8), 256, 0, stream>>>(
        op.x, op.tokens, op.k, tp, xq, sx);
    static FuncAttrPerDevice attr;
    auto* kernel = q3_a8t_kernel<Swiglu, Epilogue>;
    attr.ensure(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kA8tSmem);
    const int out_rows  = Swiglu ? op.rows / 2 : op.rows;
    const int tile_rows = Swiglu ? kA8tRows / 2 : kA8tRows;
    const dim3 grid(static_cast<unsigned>((op.tokens + kA8tTokens - 1) / kA8tTokens),
                    static_cast<unsigned>((out_rows + tile_rows - 1) / tile_rows));
    if (grid.y > 65535u) throw std::invalid_argument("q3 linear: too many row tiles");
    kernel<<<grid, 256, kA8tSmem, stream>>>(op, xq, sx, tp, epilogue);
}

// Runs the A8 route when it applies and returns true. With a large enough workspace (and 16-byte
// aligned code rows) the tall route; otherwise the in-kernel quantizing tile (identical bits).
template <bool Swiglu, class Epilogue>
bool try_a8(const Q3Operands& op, const Epilogue& epilogue, const Q3Workspace& ws,
            cudaStream_t stream) {
    if (!a8_enabled(op)) return false;
    const bool tall = route_mode() == 3 && ws.data != nullptr &&
                      (reinterpret_cast<std::uintptr_t>(ws.data) & 255u) == 0 &&
                      ws.bytes >= a8_scratch_bytes(op.tokens, op.k) &&
                      (reinterpret_cast<std::uintptr_t>(op.codes) & 15u) == 0 &&
                      op.code_row_bytes % 16 == 0;
    if (tall) {
        launch_a8t<Swiglu>(op, epilogue, ws, stream);
    } else {
        if (route_mode() == 3) {
            static std::once_flag warned;
            std::call_once(warned, [] {
                std::fprintf(stderr, "q3 linear: A8 call without workspace; using the slower "
                                     "in-kernel quantizing tile\n");
            });
        }
        launch_a8<Swiglu>(op, epilogue, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

template <class Epilogue>
void run(const Q3Operands& op, const Epilogue& epilogue, const Q3Workspace& ws,
         cudaStream_t stream) {
    if (try_a8<false>(op, epilogue, ws, stream)) return;
    if (op.tokens <= kSmallTMaxTokens) {
        launch_small_t<false>(op, epilogue, stream);
    } else {
        launch_gemm<false>(op, epilogue, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q3_force_route_for_test(int route) { g_route_override = route; }

std::size_t q3_workspace_bytes(std::int32_t k, std::int32_t max_tokens) {
    const Q3Operands probe{nullptr, nullptr, nullptr, 0, 0, 0, k, max_tokens};
    if (k <= 0 || k % S::kGroupK != 0 || !a8_enabled(probe)) return 0;
    return a8_scratch_bytes(max_tokens, k);
}

void validate_q3_weight(const Weight& w, const char* label) {
    const auto fail = [&](const char* what) {
        throw std::invalid_argument(std::string(label) + ": " + what);
    };
    if (w.qtype != QType::Q3_G128_FP16 || w.layout != QuantLayout::RowSplit) {
        fail("weight must be q3_g128_fp16 row-split");
    }
    if (w.group_size != 128 || w.scale_dtype != DType::FP16 || w.qhigh != nullptr) {
        fail("q3 weight must have FP16 scales per 128-value group and no high plane");
    }
    if (w.n <= 0 || w.k <= 0 || w.k % S::kGroupK != 0 || w.padded_shape[1] != w.k) {
        fail("q3 weight needs K a positive multiple of 128");
    }
    if (w.qdata == nullptr || w.scales == nullptr ||
        (reinterpret_cast<std::uintptr_t>(w.qdata) & 3u) != 0 ||
        (reinterpret_cast<std::uintptr_t>(w.scales) & 1u) != 0) {
        fail("q3 weight planes must be non-null and aligned");
    }
}

void q3_linear_segments_dispatch(const Tensor& x, const Weight& w, const Q3RowSegments& segments,
                                 cudaStream_t stream, const Q3Workspace& ws) {
    validate_q3_weight(w, "q3 linear");
    require_activation(x, w, "q3 linear");
    if (segments.count <= 0 || segments.count > Q3RowSegments::kMax ||
        segments.end[segments.count - 1] != w.n) {
        throw std::invalid_argument("q3 linear: row segments must cover the weight rows");
    }
    for (int s = 0; s < segments.count; ++s) {
        const int begin = s == 0 ? 0 : segments.end[s - 1];
        if (segments.data[s] == nullptr || segments.end[s] <= begin ||
            segments.ld[s] < segments.end[s] - begin) {
            throw std::invalid_argument("q3 linear: invalid row segment");
        }
    }
    run(q3_operands(x, w), SegmentsEpilogue{segments}, ws, stream);
}

void q3_linear_dispatch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream,
                        const Q3Workspace& ws) {
    if (out.dtype != DType::BF16 || out.ne[0] != w.n || out.ne[1] != x.ne[1] ||
        !out.is_contiguous() || out.data == nullptr) {
        throw std::invalid_argument("q3 linear: out must be contiguous BF16 [N,T]");
    }
    Q3RowSegments segments;
    segments.count   = 1;
    segments.data[0] = static_cast<__nv_bfloat16*>(out.data);
    segments.ld[0]   = w.n;
    segments.end[0]  = w.n;
    q3_linear_segments_dispatch(x, w, segments, stream, ws);
}

void q3_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream,
                            const Q3Workspace& ws) {
    validate_q3_weight(w, "q3 linear_add");
    require_activation(x, w, "q3 linear_add");
    if (residual.dtype != DType::BF16 || residual.ne[0] != w.n || residual.ne[1] != x.ne[1] ||
        !residual.is_contiguous() || residual.data == nullptr) {
        throw std::invalid_argument("q3 linear_add: residual must be contiguous BF16 [N,T]");
    }
    run(q3_operands(x, w), AddEpilogue{static_cast<__nv_bfloat16*>(residual.data), w.n}, ws,
        stream);
}

void q3_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream,
                               const Q3Workspace& ws) {
    validate_q3_weight(w, "q3 linear_swiglu");
    require_activation(x, w, "q3 linear_swiglu");
    if (w.n % 2 != 0 || out.dtype != DType::BF16 || out.ne[0] != w.n / 2 || out.ne[1] != x.ne[1] ||
        !out.is_contiguous() || out.data == nullptr) {
        throw std::invalid_argument("q3 linear_swiglu: out must be contiguous BF16 [N/2,T]");
    }
    const Q3Operands op = q3_operands(x, w);
    const SwigluOut epilogue{static_cast<__nv_bfloat16*>(out.data), w.n / 2};
    if (try_a8<true>(op, epilogue, ws, stream)) return;
    if (op.tokens <= kSmallTMaxTokens) {
        launch_small_t<true>(op, epilogue, stream);
    } else {
        launch_gemm<true>(op, epilogue, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
