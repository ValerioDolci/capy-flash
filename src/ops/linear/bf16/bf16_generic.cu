// Generic BF16 A16 linear for the shapes the finite
// registry (bf16_dispatch.cpp) does not name, e.g. Qwen3.8-Flash-Next's GDN q|k|v|z
// [16384,2560] / [8192,2560] kept in BF16, and for the small-T widths of the named
// Qwen3.8-Flash-Next shapes (shapes/qwen4_exp.cu). Correctness first: any N, K, T (16-byte vector
// loads when K % 8 == 0 and the operands are aligned, scalar otherwise), FP32 accumulation, one
// BF16 rounding of the output. Plain launches, no workspace, no attributes: safe under CUDA graph
// capture and at tp 2 (each rank calls it on its own stream).
//   T <= 16: GEMV, a warp (or 8 warps for short N) per output row, 8 tokens per grid row.
//   T  > 16: 64x64 tiled WMMA (bf16 16x16x16, FP32 accumulators), zero-filled edges.
#include "core/device.h"
#include "ops/linear/bf16/bf16_launch.h"

#include <cuda_bf16.h>
#include <mma.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kGemvTokens = 8;  // tokens per GEMV grid row
constexpr int kGemvMaxT   = 16; // larger T takes the tiled route
constexpr int kTileRows   = 64;
constexpr int kTileTokens = 64;
constexpr int kTileK      = 32;
constexpr int kTileLds    = kTileK + 8; // BF16 leading dimension (multiple of 8)
constexpr int kTileLdc    = kTileRows + 4;

__device__ __forceinline__ void unpack8(const uint4& v, float (&f)[8]) {
    const auto* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 p = __bfloat1622float2(h[i]);
        f[2 * i]       = p.x;
        f[2 * i + 1]   = p.y;
    }
}

// WarpsPerRow warps share one output row (split K, reduced in shared memory); 256 threads.
template <int WarpsPerRow>
__global__ void __launch_bounds__(256)
    bf16_generic_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                             const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out,
                             int n, int k, int tokens, bool vectorized) {
    constexpr int kRows = 8 / WarpsPerRow;
    const int warp      = threadIdx.x >> 5;
    const int lane      = threadIdx.x & 31;
    const int row       = blockIdx.x * kRows + warp / WarpsPerRow;
    const int sub       = warp % WarpsPerRow;
    const int t0        = blockIdx.y * kGemvTokens;
    const int count     = min(kGemvTokens, tokens - t0);
    float acc[kGemvTokens];
#pragma unroll
    for (int t = 0; t < kGemvTokens; ++t) acc[t] = 0.0F;
    if (row < n) {
        const __nv_bfloat16* wr = w + static_cast<std::size_t>(row) * k;
        if (vectorized) {
            const int vectors = k / 8;
            const auto* wv    = reinterpret_cast<const uint4*>(wr);
            for (int v = sub * 32 + lane; v < vectors; v += WarpsPerRow * 32) {
                float wf[8];
                unpack8(__ldg(wv + v), wf);
#pragma unroll
                for (int t = 0; t < kGemvTokens; ++t) {
                    if (t < count) {
                        const auto* xv = reinterpret_cast<const uint4*>(
                            x + static_cast<std::size_t>(t0 + t) * k);
                        float xf[8];
                        unpack8(__ldg(xv + v), xf);
#pragma unroll
                        for (int i = 0; i < 8; ++i) acc[t] = fmaf(wf[i], xf[i], acc[t]);
                    }
                }
            }
        } else {
            for (int c = sub * 32 + lane; c < k; c += WarpsPerRow * 32) {
                const float wf = __bfloat162float(wr[c]);
#pragma unroll
                for (int t = 0; t < kGemvTokens; ++t) {
                    if (t < count) {
                        acc[t] =
                            fmaf(wf, __bfloat162float(x[static_cast<std::size_t>(t0 + t) * k + c]),
                                 acc[t]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < kGemvTokens; ++t) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            acc[t] += __shfl_xor_sync(0xFFFFFFFFU, acc[t], offset);
    }
    if constexpr (WarpsPerRow > 1) {
        __shared__ float partial[8][kGemvTokens];
        if (lane == 0) {
#pragma unroll
            for (int t = 0; t < kGemvTokens; ++t) partial[warp][t] = acc[t];
        }
        __syncthreads();
        if (sub == 0 && lane < count && row < n) {
            float sum = 0.0F;
            for (int s = 0; s < WarpsPerRow; ++s) sum += partial[warp + s][lane];
            out[static_cast<std::size_t>(t0 + lane) * n + row] = __float2bfloat16(sum);
        }
    } else {
        if (lane == 0 && row < n) {
#pragma unroll
            for (int t = 0; t < kGemvTokens; ++t) {
                if (t < count)
                    out[static_cast<std::size_t>(t0 + t) * n + row] = __float2bfloat16(acc[t]);
            }
        }
    }
}

// One 64-row x 64-token tile per block, 4 warps of 32x32 (2x2 WMMA fragments).
__global__ void __launch_bounds__(128)
    bf16_generic_gemm_kernel(const __nv_bfloat16* __restrict__ x,
                             const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out,
                             int n, int k, int tokens, bool vectorized) {
    using namespace nvcuda;
    __shared__ __align__(32) __nv_bfloat16 a_tile[kTileRows * kTileLds];
    __shared__ __align__(32) __nv_bfloat16 b_tile[kTileTokens * kTileLds];
    __shared__ __align__(32) float c_tile[kTileTokens * kTileLdc];
    const int n0   = blockIdx.x * kTileRows;
    const int t0   = blockIdx.y * kTileTokens;
    const int warp = threadIdx.x >> 5;
    const int wm   = warp & 1;  // 32-row half
    const int wn   = warp >> 1; // 32-token half
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0F);
    const __nv_bfloat16 zero = __float2bfloat16(0.0F);
    const auto load          = [&](const __nv_bfloat16* source, int limit, int base, int k0,
                          __nv_bfloat16* tile) {
        for (int i = threadIdx.x; i < kTileRows * kTileK / 8; i += 128) {
            const int r      = i / (kTileK / 8);
            const int c8     = (i % (kTileK / 8)) * 8;
            const int g      = base + r;
            const int gk     = k0 + c8;
            __nv_bfloat16* d = tile + r * kTileLds + c8;
            if (vectorized && g < limit && gk + 8 <= k) {
                *reinterpret_cast<uint4*>(d) = __ldg(
                    reinterpret_cast<const uint4*>(source + static_cast<std::size_t>(g) * k + gk));
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    d[j] = (g < limit && gk + j < k)
                               ? source[static_cast<std::size_t>(g) * k + gk + j]
                               : zero;
                }
            }
        }
    };
    for (int k0 = 0; k0 < k; k0 += kTileK) {
        load(w, n, n0, k0, a_tile);      // A(row, k): row-major [64][kTileLds]
        load(x, tokens, t0, k0, b_tile); // B(k, token): column-major [64][kTileLds]
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < kTileK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b[2];
#pragma unroll
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(a[i], a_tile + (wm * 32 + i * 16) * kTileLds + kk, kTileLds);
#pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(b[j], b_tile + (wn * 32 + j * 16) * kTileLds + kk, kTileLds);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(c_tile + (wn * 32 + j * 16) * kTileLdc + (wm * 32 + i * 16),
                                    acc[i][j], kTileLdc, wmma::mem_col_major);
    __syncthreads();
    for (int i = threadIdx.x; i < kTileRows * kTileTokens; i += 128) {
        const int t = i / kTileRows;
        const int r = i % kTileRows;
        if (n0 + r < n && t0 + t < tokens) {
            out[static_cast<std::size_t>(t0 + t) * n + n0 + r] =
                __float2bfloat16(c_tile[t * kTileLdc + r]);
        }
    }
}

bool aligned16(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 0xFU) == 0; }

} // namespace

void launch_bf16_generic(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const int n = w.n, k = w.k, tokens = x.ne[1];
    if (n <= 0 || k <= 0 || tokens <= 0 || w.qdata == nullptr || x.data == nullptr ||
        out.data == nullptr) {
        throw std::invalid_argument("bf16 generic linear: invalid operands");
    }
    const auto* xp        = static_cast<const __nv_bfloat16*>(x.data);
    const auto* wp        = static_cast<const __nv_bfloat16*>(w.qdata);
    auto* op              = static_cast<__nv_bfloat16*>(out.data);
    const bool vectorized = k % 8 == 0 && aligned16(xp) && aligned16(wp);
    if (tokens <= kGemvMaxT) {
        const unsigned token_blocks =
            static_cast<unsigned>((tokens + kGemvTokens - 1) / kGemvTokens);
        // Short outputs: 8 warps per row so that the grid still covers every SM.
        if (n < 4096) {
            bf16_generic_gemv_kernel<8>
                <<<dim3(static_cast<unsigned>(n), token_blocks), 256, 0, stream>>>(
                    xp, wp, op, n, k, tokens, vectorized);
        } else {
            bf16_generic_gemv_kernel<1>
                <<<dim3(static_cast<unsigned>((n + 7) / 8), token_blocks), 256, 0, stream>>>(
                    xp, wp, op, n, k, tokens, vectorized);
        }
    } else {
        const dim3 grid(static_cast<unsigned>((n + kTileRows - 1) / kTileRows),
                        static_cast<unsigned>((tokens + kTileTokens - 1) / kTileTokens));
        if (grid.y > 65535U) throw std::invalid_argument("bf16 generic linear: T too large");
        bf16_generic_gemm_kernel<<<grid, 128, 0, stream>>>(xp, wp, op, n, k, tokens, vectorized);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
