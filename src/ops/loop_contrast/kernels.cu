// Implements: include/ninfer/ops/loop_contrast.h
// Extrapolation: elementwise grid-stride kernel. Logits: one block per column; with `adaptive` a
// first sweep reduces the column's two largest values and its softmax denominator (per-thread
// strided partials, then a fixed shuffle and shared-memory tree), a second sweep combines.
#include "ops/loop_contrast/launch.h"

#include "core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock        = 256;
constexpr int kColumnBlock  = 1024;
constexpr int kColumnWarps  = kColumnBlock / 32;
constexpr unsigned kAllMask = 0xffffffffU;

__global__ void extrapolate_kernel(const __nv_bfloat16* last, const __nv_bfloat16* __restrict__ ref,
                                   float w, __nv_bfloat16* out, std::int64_t n) {
    const std::int64_t step = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += step) {
        const float a = __bfloat162float(last[i]);
        const float d = __fsub_rn(a, __bfloat162float(ref[i]));
        out[i]        = __float2bfloat16_rn(__fmaf_rn(w, d, a));
    }
}

// The two largest values of a set and sum exp(v - m1) over it; empty: m1 = m2 = -inf, s = 0.
struct Top2 {
    float m1, m2, s;
};

__device__ __forceinline__ Top2 merge(Top2 a, Top2 b) {
    if (b.s == 0.0F) { return a; }
    if (a.s == 0.0F) { return b; }
    if (b.m1 > a.m1) {
        const Top2 t = a;
        a            = b;
        b            = t;
    }
    return {a.m1, fmaxf(a.m2, b.m1), a.s + b.s * expf(b.m1 - a.m1)};
}

__device__ __forceinline__ Top2 shuffle_merge(Top2 t, int offset) {
    const Top2 other{__shfl_down_sync(kAllMask, t.m1, offset),
                     __shfl_down_sync(kAllMask, t.m2, offset),
                     __shfl_down_sync(kAllMask, t.s, offset)};
    return merge(t, other);
}

__global__ void __launch_bounds__(kColumnBlock)
    contrast_logits_kernel(const float* capped, const float* __restrict__ ref, float w,
                           bool adaptive, float* out, __nv_bfloat16* __restrict__ logits,
                           std::int64_t rows) {
    const std::int64_t base = static_cast<std::int64_t>(blockIdx.x) * rows;
    const float* c          = capped + base;
    const float* r          = ref + base;
    __shared__ Top2 partial[kColumnWarps];
    __shared__ float column_w;
    if (adaptive) {
        Top2 t{-INFINITY, -INFINITY, 0.0F};
        for (std::int64_t v = threadIdx.x; v < rows; v += kColumnBlock) {
            const float x = c[v];
            if (t.s == 0.0F) {
                t = {x, -INFINITY, 1.0F};
            } else if (x > t.m1) {
                t = {x, t.m1, t.s * expf(t.m1 - x) + 1.0F};
            } else {
                t.m2 = fmaxf(t.m2, x);
                t.s += expf(x - t.m1);
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) { t = shuffle_merge(t, offset); }
        const int lane = static_cast<int>(threadIdx.x) & 31;
        const int warp = static_cast<int>(threadIdx.x) >> 5;
        if (lane == 0) { partial[warp] = t; }
        __syncthreads();
        if (warp == 0) {
            t = partial[lane];
            for (int offset = 16; offset > 0; offset >>= 1) { t = shuffle_merge(t, offset); }
            if (lane == 0) {
                const float p1 = 1.0F / t.s;
                const float p2 = expf(t.m2 - t.m1) / t.s;
                column_w       = w * (1.0F - (p1 - p2));
            }
        }
        __syncthreads();
    }
    const float wj    = adaptive ? column_w : w;
    float* o          = out + base;
    __nv_bfloat16* lo = logits + base;
    for (std::int64_t v = threadIdx.x; v < rows; v += kColumnBlock) {
        const float x = c[v];
        const float y = __fmaf_rn(wj, __fsub_rn(x, r[v]), x);
        o[v]          = y;
        lo[v]         = __float2bfloat16_rn(y);
    }
}

} // namespace

void loop_extrapolate_launch(const Tensor& last, const Tensor& reference, float w, Tensor& out,
                             cudaStream_t stream) {
    const std::int64_t n = last.numel();
    const int grid =
        static_cast<int>(std::clamp<std::int64_t>((n + kBlock - 1) / kBlock, 1, 1 << 20));
    extrapolate_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(last.data),
        static_cast<const __nv_bfloat16*>(reference.data), w, static_cast<__nv_bfloat16*>(out.data),
        n);
    CUDA_CHECK(cudaGetLastError());
}

void loop_contrast_logits_launch(const Tensor& capped, const Tensor& reference, float w,
                                 bool adaptive, Tensor& out, Tensor& logits, cudaStream_t stream) {
    const auto columns = static_cast<unsigned>(capped.ne[1]);
    contrast_logits_kernel<<<columns, kColumnBlock, 0, stream>>>(
        static_cast<const float*>(capped.data), static_cast<const float*>(reference.data), w,
        adaptive, static_cast<float*>(out.data), static_cast<__nv_bfloat16*>(logits.data),
        capped.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
