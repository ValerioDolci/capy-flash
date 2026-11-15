// Implements: include/ninfer/ops/hyper_connection.h
// Straightforward FP32-accumulating SIMT kernels. Every kernel is a column-parallel map or a
// per-(group, column) reduction; none carries state between calls.
#include "ops/hyper_connection/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;

__device__ __forceinline__ float block_sum(float value, float* scratch) {
    value          = warp_sum(value);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) { scratch[warp] = value; }
    __syncthreads();
    float total = 0.0F;
    if (warp == 0) {
        total = lane < static_cast<int>(blockDim.x >> 5) ? scratch[lane] : 0.0F;
        total = warp_sum(total);
        if (lane == 0) { scratch[0] = total; }
    }
    __syncthreads();
    total = scratch[0];
    __syncthreads();
    return total;
}

// One CTA per (group, column).
__global__ void grouped_offset_rmsnorm_kernel(const __nv_bfloat16* __restrict__ x,
                                              const __nv_bfloat16* __restrict__ weight,
                                              __nv_bfloat16* __restrict__ out, std::int32_t width,
                                              std::int32_t groups, float eps) {
    __shared__ float scratch[32];
    const std::int32_t group  = static_cast<std::int32_t>(blockIdx.x) % groups;
    const std::int64_t column = static_cast<std::int64_t>(blockIdx.x) / groups;
    const std::int64_t base   = column * width * groups + static_cast<std::int64_t>(group) * width;
    float sum                 = 0.0F;
    for (int h = static_cast<int>(threadIdx.x); h < width; h += blockDim.x) {
        const float v = __bfloat162float(x[base + h]);
        sum += v * v;
    }
    const float inv = rsqrtf(block_sum(sum, scratch) / static_cast<float>(width) + eps);
    for (int h = static_cast<int>(threadIdx.x); h < width; h += blockDim.x) {
        const float w =
            1.0F + __bfloat162float(weight[static_cast<std::int64_t>(group) * width + h]);
        out[base + h] = __float2bfloat16_rn(__bfloat162float(x[base + h]) * inv * w);
    }
}

__global__ void hyper_connection_gates_kernel(const __nv_bfloat16* __restrict__ projection,
                                              __nv_bfloat16* __restrict__ lowrank,
                                              float* __restrict__ inject, std::int32_t rows,
                                              std::int32_t lowrank_rows, std::int32_t streams,
                                              std::int64_t columns) {
    const std::int64_t total = static_cast<std::int64_t>(rows) * columns;
    const float inv_streams  = 1.0F / static_cast<float>(streams);
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
         i < total; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int32_t row    = static_cast<std::int32_t>(i % rows);
        const std::int64_t column = i / rows;
        const float v             = __bfloat162float(projection[i]) * inv_streams;
        if (row < lowrank_rows) {
            lowrank[column * lowrank_rows + row] = __float2bfloat16_rn(silu(v));
        } else {
            inject[column * streams + (row - lowrank_rows)] = 2.0F * sigmoid(v);
        }
    }
}

__global__ void hyper_connection_collapse_kernel(const __nv_bfloat16* __restrict__ up,
                                                 const __nv_bfloat16* __restrict__ normalized,
                                                 __nv_bfloat16* __restrict__ mixed,
                                                 std::int32_t width, std::int32_t streams,
                                                 std::int64_t columns) {
    const std::int64_t total = static_cast<std::int64_t>(width) * columns;
    const float inv_streams  = 1.0F / static_cast<float>(streams);
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
         i < total; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int32_t h      = static_cast<std::int32_t>(i % width);
        const std::int64_t column = i / width;
        const std::int64_t base   = column * width * streams + h;
        float sum                 = 0.0F;
        for (int s = 0; s < streams; ++s) {
            const std::int64_t index = base + static_cast<std::int64_t>(s) * width;
            sum += sigmoid(__bfloat162float(up[index])) * __bfloat162float(normalized[index]);
        }
        mixed[i] = __float2bfloat16_rn(sum * inv_streams);
    }
}

__global__ void hyper_connection_combine_kernel(const __nv_bfloat16* __restrict__ y,
                                                const float* __restrict__ inject,
                                                __nv_bfloat16* __restrict__ residual,
                                                std::int32_t width, std::int32_t streams,
                                                std::int64_t columns) {
    const std::int64_t wide  = static_cast<std::int64_t>(width) * streams;
    const std::int64_t total = wide * columns;
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
         i < total; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t column = i / wide;
        const std::int32_t row    = static_cast<std::int32_t>(i % wide);
        const std::int32_t stream = row / width;
        const std::int32_t h      = row - stream * width;
        const float update =
            inject[column * streams + stream] * __bfloat162float(y[column * width + h]);
        residual[i] = __float2bfloat16_rn(__bfloat162float(residual[i]) + update);
    }
}

__global__ void hyper_connection_expand_kernel(const __nv_bfloat16* __restrict__ x,
                                               __nv_bfloat16* __restrict__ residual,
                                               std::int32_t width, std::int32_t streams,
                                               std::int64_t columns) {
    const std::int64_t wide  = static_cast<std::int64_t>(width) * streams;
    const std::int64_t total = wide * columns;
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
         i < total; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t column = i / wide;
        const std::int32_t h      = static_cast<std::int32_t>((i % wide) % width);
        residual[i]               = x[column * width + h];
    }
}

// grouped_offset_rmsnorm_kernel for widths of whole 16-byte chunks, at most kVectorChunks per
// thread: every thread loads its chunks once and keeps them in registers across both passes.
constexpr int kVectorChunks = 2;

__global__ void grouped_offset_rmsnorm_vector_kernel(const __nv_bfloat16* __restrict__ x,
                                                     const __nv_bfloat16* __restrict__ weight,
                                                     __nv_bfloat16* __restrict__ out,
                                                     std::int32_t width, std::int32_t groups,
                                                     float eps) {
    __shared__ float scratch[32];
    const std::int32_t group  = static_cast<std::int32_t>(blockIdx.x) % groups;
    const std::int64_t column = static_cast<std::int64_t>(blockIdx.x) / groups;
    const std::int64_t base   = column * width * groups + static_cast<std::int64_t>(group) * width;
    const int chunks          = width / 8;
    const auto* source        = reinterpret_cast<const uint4*>(x + base);
    uint4 values[kVectorChunks];
    float sum = 0.0F;
#pragma unroll
    for (int c = 0; c < kVectorChunks; ++c) {
        const int chunk = static_cast<int>(threadIdx.x) + c * static_cast<int>(blockDim.x);
        values[c]       = chunk < chunks ? source[chunk] : make_uint4(0, 0, 0, 0);
        const auto* v   = reinterpret_cast<const __nv_bfloat162*>(&values[c]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(v[j]);
            sum += f.x * f.x + f.y * f.y;
        }
    }
    const float inv = rsqrtf(block_sum(sum, scratch) / static_cast<float>(width) + eps);
    const auto* w =
        reinterpret_cast<const uint4*>(weight + static_cast<std::int64_t>(group) * width);
    auto* target = reinterpret_cast<uint4*>(out + base);
#pragma unroll
    for (int c = 0; c < kVectorChunks; ++c) {
        const int chunk = static_cast<int>(threadIdx.x) + c * static_cast<int>(blockDim.x);
        if (chunk >= chunks) { continue; }
        const uint4 weights = w[chunk];
        const auto* v       = reinterpret_cast<const __nv_bfloat162*>(&values[c]);
        const auto* g       = reinterpret_cast<const __nv_bfloat162*>(&weights);
        uint4 result;
        auto* r = reinterpret_cast<__nv_bfloat162*>(&result);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(v[j]);
            const float2 s = __bfloat1622float2(g[j]);
            r[j] = __floats2bfloat162_rn(f.x * inv * (1.0F + s.x), f.y * inv * (1.0F + s.y));
        }
        target[chunk] = result;
    }
}

// Prefill-width (>= kVectorColumns columns) forms of collapse and combine: one thread per 8
// consecutive features of one column (16-byte loads and stores, one 64-bit division per 8
// elements instead of per element), each element computed with the scalar kernels' arithmetic in
// the same order, so the bits are those of the scalar kernels.
constexpr std::int64_t kVectorColumns = 64;

__device__ __forceinline__ void unpack8(uint4 v, float (&f)[8]) {
    const auto* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 t = __bfloat1622float2(p[j]);
        f[2 * j]       = t.x;
        f[2 * j + 1]   = t.y;
    }
}

__device__ __forceinline__ uint4 pack8(const float (&f)[8]) {
    uint4 v;
    auto* p = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) { p[j] = __floats2bfloat162_rn(f[2 * j], f[2 * j + 1]); }
    return v;
}

__global__ void
hyper_connection_collapse_vector_kernel(const __nv_bfloat16* __restrict__ up,
                                        const __nv_bfloat16* __restrict__ normalized,
                                        __nv_bfloat16* __restrict__ mixed, std::int32_t width,
                                        std::int32_t streams, std::int64_t columns) {
    const std::int32_t chunks = width / 8;
    const std::int64_t total  = static_cast<std::int64_t>(chunks) * columns;
    const float inv_streams   = 1.0F / static_cast<float>(streams);
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
         i < total; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t column = i / chunks;
        const std::int32_t h      = static_cast<std::int32_t>(i - column * chunks) * 8;
        const std::int64_t base   = column * width * streams + h;
        float sum[8]              = {};
        for (int s = 0; s < streams; ++s) {
            const std::int64_t index = base + static_cast<std::int64_t>(s) * width;
            float u[8], n[8];
            unpack8(*reinterpret_cast<const uint4*>(up + index), u);
            unpack8(*reinterpret_cast<const uint4*>(normalized + index), n);
#pragma unroll
            for (int j = 0; j < 8; ++j) { sum[j] += sigmoid(u[j]) * n[j]; }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) { sum[j] = sum[j] * inv_streams; }
        *reinterpret_cast<uint4*>(mixed + column * width + h) = pack8(sum);
    }
}

__global__ void hyper_connection_combine_vector_kernel(const __nv_bfloat16* __restrict__ y,
                                                       const float* __restrict__ inject,
                                                       __nv_bfloat16* __restrict__ residual,
                                                       std::int32_t width, std::int32_t streams,
                                                       std::int64_t columns) {
    const std::int32_t chunks = width * streams / 8;
    const std::int64_t total  = static_cast<std::int64_t>(chunks) * columns;
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
         i < total; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t column = i / chunks;
        const std::int32_t row    = static_cast<std::int32_t>(i - column * chunks) * 8;
        const std::int32_t stream = row / width;
        const std::int32_t h      = row - stream * width;
        const float gate          = inject[column * streams + stream];
        float yv[8], r[8];
        unpack8(*reinterpret_cast<const uint4*>(y + column * width + h), yv);
        auto* target = reinterpret_cast<uint4*>(residual + i * 8);
        unpack8(*target, r);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float update = gate * yv[j];
            r[j]               = r[j] + update;
        }
        *target = pack8(r);
    }
}

bool vector_ready(std::int32_t width, std::int64_t columns, std::initializer_list<const void*> p) {
    if (width % 8 != 0 || columns < kVectorColumns) { return false; }
    for (const void* q : p) {
        if ((reinterpret_cast<std::uintptr_t>(q) & 0xfu) != 0) { return false; }
    }
    return true;
}

int grid_for(std::int64_t work) {
    return static_cast<int>(std::clamp<std::int64_t>((work + kBlock - 1) / kBlock, 1, 65535LL * 8));
}

} // namespace

void grouped_offset_rmsnorm_launch(const Tensor& x, const Tensor& weight, std::int32_t groups,
                                   float eps, Tensor& out, cudaStream_t stream) {
    const std::int32_t width   = x.ne[0] / groups;
    const std::int64_t columns = x.numel() / x.ne[0];
    const auto aligned         = [](const void* p) {
        return (reinterpret_cast<std::uintptr_t>(p) & 0xfu) == 0;
    };
    if (width % 8 == 0 && width / 8 <= kBlock * kVectorChunks && aligned(x.data) &&
        aligned(weight.data) && aligned(out.data)) {
        grouped_offset_rmsnorm_vector_kernel<<<static_cast<unsigned>(columns * groups), kBlock, 0,
                                               stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.data), static_cast<__nv_bfloat16*>(out.data),
            width, groups, eps);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    grouped_offset_rmsnorm_kernel<<<static_cast<unsigned>(columns * groups), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<__nv_bfloat16*>(out.data), width, groups, eps);
    CUDA_CHECK(cudaGetLastError());
}

void hyper_connection_gates_launch(const Tensor& projection, std::int32_t streams, Tensor& lowrank,
                                   Tensor* inject, cudaStream_t stream) {
    const std::int64_t columns = projection.numel() / projection.ne[0];
    hyper_connection_gates_kernel<<<grid_for(projection.numel()), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projection.data),
        static_cast<__nv_bfloat16*>(lowrank.data),
        inject != nullptr ? static_cast<float*>(inject->data) : nullptr, projection.ne[0],
        lowrank.ne[0], streams, columns);
    CUDA_CHECK(cudaGetLastError());
}

void hyper_connection_collapse_launch(const Tensor& up, const Tensor& normalized,
                                      std::int32_t streams, Tensor& mixed, cudaStream_t stream) {
    const std::int64_t columns = mixed.numel() / mixed.ne[0];
    if (vector_ready(mixed.ne[0], columns, {up.data, normalized.data, mixed.data})) {
        hyper_connection_collapse_vector_kernel<<<grid_for(mixed.numel() / 8), kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(up.data),
            static_cast<const __nv_bfloat16*>(normalized.data),
            static_cast<__nv_bfloat16*>(mixed.data), mixed.ne[0], streams, columns);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    hyper_connection_collapse_kernel<<<grid_for(mixed.numel()), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(up.data),
        static_cast<const __nv_bfloat16*>(normalized.data), static_cast<__nv_bfloat16*>(mixed.data),
        mixed.ne[0], streams, columns);
    CUDA_CHECK(cudaGetLastError());
}

void hyper_connection_combine_launch(const Tensor& y, const Tensor& inject, Tensor& residual,
                                     cudaStream_t stream) {
    const std::int64_t columns = y.numel() / y.ne[0];
    if (vector_ready(y.ne[0], columns, {y.data, residual.data})) {
        hyper_connection_combine_vector_kernel<<<grid_for(residual.numel() / 8), kBlock, 0,
                                                 stream>>>(
            static_cast<const __nv_bfloat16*>(y.data), static_cast<const float*>(inject.data),
            static_cast<__nv_bfloat16*>(residual.data), y.ne[0], inject.ne[0], columns);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    hyper_connection_combine_kernel<<<grid_for(residual.numel()), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(y.data), static_cast<const float*>(inject.data),
        static_cast<__nv_bfloat16*>(residual.data), y.ne[0], inject.ne[0], columns);
    CUDA_CHECK(cudaGetLastError());
}

void hyper_connection_expand_launch(const Tensor& x, std::int32_t streams, Tensor& residual,
                                    cudaStream_t stream) {
    const std::int64_t columns = x.numel() / x.ne[0];
    hyper_connection_expand_kernel<<<grid_for(residual.numel()), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(residual.data),
        x.ne[0], streams, columns);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
