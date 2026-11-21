#include "core/device.h"
#include "ops/linear/bf16/bf16_shapes.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
// Inkling 2 Mini BF16 projections at decode and verification widths (T <= 16): the tp 2 shards
// of the attention inputs (queries, all, key/value, keys/values) and of the attention output (the
// output head half already reads at ~95 % of the DRAM bandwidth on the generic route). Same
// arithmetic as the generic GEMV (bf16_generic.cu), hence the same bits, and the same bits in
// every column at every T: each output row is split over the same 32-lane groups (eight warps
// per row below 4096 rows, one warp per row from 4096 rows), each lane runs the same 8-wide FMA
// chain, then the same butterfly and the same sum of the warp partials. What changes is the work
// around it: K is static (all weight loads of a row in flight before the first FMA), the token
// count is a template (no guards for absent tokens) and at T = 1 two rows share a CTA and its
// activation loads. Unaligned operands take the generic route (its scalar path sums in another
// order).
//
// Beyond T = 16 (prefill blocks) the generic route is the 64x64 tiled WMMA kernel. The tiled
// kernel here keeps its arithmetic per output element: FP32 accumulators from 0, one BF16 x BF16
// m16n8k16 tensor-core product per 16 values of K, in increasing K, one BF16 rounding of the
// result (a WMMA 16x16x16 product is two m16n8k16 products over the same 16 values of K), hence
// the generic route's bits. What changes is the work around it: 128-token x 128-row tiles (64
// rows for short outputs), cp.async double buffering of the 32-wide K slices, ldmatrix fragments
// and the result written from the registers.
namespace {

constexpr int kThreads        = 256;
constexpr int kWarps          = kThreads / 32;
constexpr int kTokensPerBlock = 8;    // tokens per grid row, as the generic GEMV
constexpr int kSplitBelowRows = 4096; // the generic GEMV's eight-warp rows
constexpr int kMaxTokens      = 16;   // the generic GEMV's widest T

__device__ __forceinline__ void unpack8(const uint4& v, float (&f)[8]) {
    const auto* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 p = __bfloat1622float2(h[i]);
        f[2 * i]       = p.x;
        f[2 * i + 1]   = p.y;
    }
}

__device__ __forceinline__ void butterfly(float& value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xFFFFFFFFU, value, offset);
}

// Lane `lane` of a row's 32-lane group g reads the 8-wide vectors g * 32 + lane + j * Stride.
template <int K, int Rows, int Tokens, int Stride>
__device__ __forceinline__ void
accumulate_rows(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w, int n,
                int row0, int first, int t0, int count, float (&acc)[Rows][Tokens]) {
    constexpr int kVectors = K / 8;
    constexpr int kSteps   = (kVectors + Stride - 1) / Stride;
    uint4 wv[Rows][kSteps];
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        const auto* wr = reinterpret_cast<const uint4*>(w + static_cast<std::size_t>(row0 + r) * K);
#pragma unroll
        for (int s = 0; s < kSteps; ++s) {
            const int v = first + s * Stride;
            wv[r][s]    = (v < kVectors && row0 + r < n) ? __ldg(wr + v) : make_uint4(0, 0, 0, 0);
        }
    }
#pragma unroll
    for (int s = 0; s < kSteps; ++s) {
        const int v = first + s * Stride;
        if (v < kVectors) {
            float wf[Rows][8];
#pragma unroll
            for (int r = 0; r < Rows; ++r) unpack8(wv[r][s], wf[r]);
#pragma unroll
            for (int t = 0; t < Tokens; ++t) {
                if (t < count) {
                    float xf[8];
                    unpack8(__ldg(reinterpret_cast<const uint4*>(
                                      x + static_cast<std::size_t>(t0 + t) * K) +
                                  v),
                            xf);
#pragma unroll
                    for (int r = 0; r < Rows; ++r) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) acc[r][t] = fmaf(wf[r][i], xf[i], acc[r][t]);
                    }
                }
            }
        }
    }
}

// Rows below 4096: the eight warps of the CTA are the eight 32-lane groups of each of its Rows
// rows; the row's value is the sum of the eight group totals, from 0 and in group order.
template <int K, int Rows, int Tokens>
__global__ void __launch_bounds__(kThreads)
    inkling2_gemv_split_kernel(const __nv_bfloat16* __restrict__ x,
                               const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out,
                               int n, int tokens) {
    __shared__ float partial[Rows][kWarps][Tokens];
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int row0  = static_cast<int>(blockIdx.x) * Rows;
    const int t0    = static_cast<int>(blockIdx.y) * kTokensPerBlock;
    const int count = min(Tokens, tokens - t0);
    float acc[Rows][Tokens];
#pragma unroll
    for (int r = 0; r < Rows; ++r)
#pragma unroll
        for (int t = 0; t < Tokens; ++t) acc[r][t] = 0.0F;
    accumulate_rows<K, Rows, Tokens, kThreads>(x, w, n, row0, static_cast<int>(threadIdx.x), t0,
                                               count, acc);
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            butterfly(acc[r][t]);
            if (lane == 0) partial[r][warp][t] = acc[r][t];
        }
    }
    __syncthreads();
    for (int i = static_cast<int>(threadIdx.x); i < Rows * Tokens; i += kThreads) {
        const int r = i / Tokens, t = i % Tokens;
        if (row0 + r < n && t < count) {
            float sum = 0.0F;
#pragma unroll
            for (int g = 0; g < kWarps; ++g) sum += partial[r][g][t];
            out[static_cast<std::size_t>(t0 + t) * n + row0 + r] = __float2bfloat16(sum);
        }
    }
}

// Rows from 4096: one warp is the row's only 32-lane group; each warp takes Rows rows.
template <int K, int Rows, int Tokens>
__global__ void __launch_bounds__(kThreads)
    inkling2_gemv_warp_kernel(const __nv_bfloat16* __restrict__ x,
                              const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out,
                              int n, int tokens) {
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int row0  = (static_cast<int>(blockIdx.x) * kWarps + warp) * Rows;
    const int t0    = static_cast<int>(blockIdx.y) * kTokensPerBlock;
    const int count = min(Tokens, tokens - t0);
    float acc[Rows][Tokens];
#pragma unroll
    for (int r = 0; r < Rows; ++r)
#pragma unroll
        for (int t = 0; t < Tokens; ++t) acc[r][t] = 0.0F;
    accumulate_rows<K, Rows, Tokens, 32>(x, w, n, row0, lane, t0, count, acc);
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            butterfly(acc[r][t]);
            if (lane == 0 && row0 + r < n && t < count)
                out[static_cast<std::size_t>(t0 + t) * n + row0 + r] = __float2bfloat16(acc[r][t]);
        }
    }
}

bool aligned16(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 0xFU) == 0; }

// Split (fewer than 4096 rows): two rows per CTA at T = 1, one beyond (T = 4 and 8 were slower
// with two or more: more accumulators, fewer CTAs). Warp (from 4096 rows): one row per warp, T = 1
// only (cold-L2 sweep on RTX 5070 Ti: at T >= 4 it was no faster than the generic GEMV).
template <int K, bool Split>
void launch_inkling2_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const int n = w.n, tokens = x.ne[1];
    const auto* xp = static_cast<const __nv_bfloat16*>(x.data);
    const auto* wp = static_cast<const __nv_bfloat16*>(w.qdata);
    auto* op       = static_cast<__nv_bfloat16*>(out.data);
    if (w.k != K || (n < kSplitBelowRows) != Split || tokens <= 0 ||
        tokens > (Split ? kMaxTokens : 1) || xp == nullptr || wp == nullptr || op == nullptr ||
        !aligned16(xp) || !aligned16(wp)) {
        launch_bf16_generic(x, w, out, stream);
        return;
    }
    const auto token_blocks =
        static_cast<unsigned>((tokens + kTokensPerBlock - 1) / kTokensPerBlock);
    const auto split = [&]<int Rows, int Tokens>() {
        const auto blocks = static_cast<unsigned>((n + Rows - 1) / Rows);
        inkling2_gemv_split_kernel<K, Rows, Tokens>
            <<<dim3(blocks, token_blocks), kThreads, 0, stream>>>(xp, wp, op, n, tokens);
    };
    if constexpr (Split) {
        if (tokens == 1)
            split.template operator()<2, 1>();
        else if (tokens == 2)
            split.template operator()<1, 2>();
        else if (tokens <= 4)
            split.template operator()<1, 4>();
        else
            split.template operator()<1, 8>();
    } else {
        const auto blocks = static_cast<unsigned>((n + kWarps - 1) / kWarps);
        inkling2_gemv_warp_kernel<K, 1, 1><<<blocks, kThreads, 0, stream>>>(xp, wp, op, n, tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

constexpr int kTileTokens = 128;        // tokens per CTA
constexpr int kTileK      = 32;         // K slice per stage, as the generic tiled kernel
constexpr int kTileLds    = kTileK + 8; // BF16 leading dimension: conflict-free ldmatrix rows

__device__ __forceinline__ unsigned shared_address(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

// 16 bytes from global to shared; `valid` false fills zeros (rows past the operand).
__device__ __forceinline__ void copy16(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(shared_address(dst)),
                 "l"(src), "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(shared_address(p)));
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], const unsigned (&a)[4], unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// One TileRows-row x 128-token tile per CTA; eight warps of (TileRows / 2) rows x 32 tokens.
template <int K, int TileRows>
__global__ void __launch_bounds__(kThreads)
    inkling2_gemm_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                         __nv_bfloat16* __restrict__ out, int n, int tokens) {
    constexpr int kWarpRows = TileRows / 2;
    constexpr int kM        = kWarpRows / 16; // m16 tiles per warp
    constexpr int kN        = 4;              // n8 tiles per warp (32 tokens)
    constexpr int kSlices   = K / kTileK;
    __shared__ __align__(128) __nv_bfloat16 a_tile[2][TileRows * kTileLds];
    __shared__ __align__(128) __nv_bfloat16 b_tile[2][kTileTokens * kTileLds];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp & 1, wn = warp >> 1;
    const int n0 = static_cast<int>(blockIdx.x) * TileRows;
    const int t0 = static_cast<int>(blockIdx.y) * kTileTokens;

    const auto load = [&](int stage, int k0) {
#pragma unroll
        for (int i = tid; i < TileRows * (kTileK / 8); i += kThreads) {
            const int r = i / (kTileK / 8), c8 = (i % (kTileK / 8)) * 8;
            const bool ok = n0 + r < n;
            copy16(&a_tile[stage][r * kTileLds + c8],
                   w + static_cast<std::size_t>(ok ? n0 + r : 0) * K + k0 + c8, ok);
        }
#pragma unroll
        for (int i = tid; i < kTileTokens * (kTileK / 8); i += kThreads) {
            const int r = i / (kTileK / 8), c8 = (i % (kTileK / 8)) * 8;
            const bool ok = t0 + r < tokens;
            copy16(&b_tile[stage][r * kTileLds + c8],
                   x + static_cast<std::size_t>(ok ? t0 + r : 0) * K + k0 + c8, ok);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    float acc[kM][kN][4];
#pragma unroll
    for (int i = 0; i < kM; ++i)
#pragma unroll
        for (int j = 0; j < kN; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0F;

    load(0, 0);
    for (int slice = 0; slice < kSlices; ++slice) {
        const int stage = slice & 1;
        if (slice + 1 < kSlices) {
            load(stage ^ 1, (slice + 1) * kTileK);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        const __nv_bfloat16* a_s = a_tile[stage];
        const __nv_bfloat16* b_s = b_tile[stage];
#pragma unroll
        for (int kk = 0; kk < kTileK; kk += 16) {
            unsigned a[kM][4], b[kN][2];
#pragma unroll
            for (int i = 0; i < kM; ++i) {
                const int row = wm * kWarpRows + i * 16 + (lane & 15);
                ldmatrix_x4(a[i], a_s + row * kTileLds + kk + (lane >> 4) * 8);
            }
#pragma unroll
            for (int p = 0; p < kN / 2; ++p) {
                unsigned r[4];
                const int tok = wn * 32 + p * 16 + (lane & 7) + (lane >> 4) * 8;
                ldmatrix_x4(r, b_s + tok * kTileLds + kk + ((lane >> 3) & 1) * 8);
                b[2 * p][0] = r[0], b[2 * p][1] = r[1];
                b[2 * p + 1][0] = r[2], b[2 * p + 1][1] = r[3];
            }
#pragma unroll
            for (int i = 0; i < kM; ++i)
#pragma unroll
                for (int j = 0; j < kN; ++j) mma_bf16(acc[i][j], a[i], b[j][0], b[j][1]);
        }
        __syncthreads();
    }

    const int g = lane >> 2, c = lane & 3;
#pragma unroll
    for (int i = 0; i < kM; ++i) {
#pragma unroll
        for (int j = 0; j < kN; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = n0 + wm * kWarpRows + i * 16 + g + (e >> 1) * 8;
                const int tok = t0 + wn * 32 + j * 8 + c * 2 + (e & 1);
                if (row < n && tok < tokens)
                    out[static_cast<std::size_t>(tok) * n + row] = __float2bfloat16(acc[i][j][e]);
            }
        }
    }
}

// Prefill widths (T > 16). 64-row tiles for outputs under 2048 rows, so that the grid still
// covers the SMs.
template <int K>
void launch_inkling2_gemm(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    static_assert(K % kTileK == 0);
    const int n = w.n, tokens = x.ne[1];
    const auto* xp = static_cast<const __nv_bfloat16*>(x.data);
    const auto* wp = static_cast<const __nv_bfloat16*>(w.qdata);
    auto* op       = static_cast<__nv_bfloat16*>(out.data);
    if (w.k != K || n <= 0 || tokens <= kMaxTokens || xp == nullptr || wp == nullptr ||
        op == nullptr || !aligned16(xp) || !aligned16(wp)) {
        launch_bf16_generic(x, w, out, stream);
        return;
    }
    const auto token_tiles = static_cast<unsigned>((tokens + kTileTokens - 1) / kTileTokens);
    if (n >= 2048) {
        inkling2_gemm_kernel<K, 128>
            <<<dim3(static_cast<unsigned>((n + 127) / 128), token_tiles), kThreads, 0, stream>>>(
                xp, wp, op, n, tokens);
    } else {
        inkling2_gemm_kernel<K, 64>
            <<<dim3(static_cast<unsigned>((n + 63) / 64), token_tiles), kThreads, 0, stream>>>(
                xp, wp, op, n, tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

template <int Rows, int K>
Bf16Launch select_bf16_inkling2(std::int32_t tokens) {
    constexpr bool kSplit = Rows < kSplitBelowRows;
    if (tokens > kMaxTokens) return launch_inkling2_gemm<K>;
    if (tokens > (kSplit ? kMaxTokens : 1)) return launch_bf16_generic;
    return launch_inkling2_gemv<K, kSplit>;
}

template Bf16Launch select_bf16_inkling2<3276, 2304>(std::int32_t);
template Bf16Launch select_bf16_inkling2<4300, 2304>(std::int32_t);
template Bf16Launch select_bf16_inkling2<528, 2304>(std::int32_t);
template Bf16Launch select_bf16_inkling2<512, 2304>(std::int32_t);
template Bf16Launch select_bf16_inkling2<2304, 1536>(std::int32_t);
} // namespace ninfer::ops::detail
