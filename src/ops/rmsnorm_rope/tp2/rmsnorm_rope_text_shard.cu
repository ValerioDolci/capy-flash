// Implements: include/ninfer/ops/tp2/rmsnorm_rope.h (rmsnorm_rope_text_shard). Our file.
// Match: Qwen3.8-27B two-device attention shard, q BF16 [256,12,T], k BF16 [256,2,T], unit-offset
// per-head norm, R=64 split-half RoPE with one or three (interleaved MRoPE) position axes.
// Algorithm assumptions: the arithmetic of the two kernels it replaces, statement for statement:
// rmsnorm_warp_bf16x2_kernel<Offset, 128, true, 256> (kernel/rmsnorm.cuh) for the norm, the BF16
// materialization of the normalized head, then rope_generic_kernel (kernel/rope.cuh) for the
// rotation, including its powf frequency. One warp owns one head, as in the norm kernel; the
// rotation partner of lane l's first BF16x2 pair is lane l^16's, exchanged by shuffle.
#include "ninfer/ops/tp2/rmsnorm_rope.h"

#include "core/device.h"
#include "ops/common/warp.cuh"
#include "ops/kernel/rmsnorm.cuh"
#include "ops/kernel/text_rope_yarn.cuh"
#include "ops/launcher/rope.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kHeadDim      = 256;
constexpr int kRotaryDim    = 64;
constexpr int kQueryHeads   = 12;
constexpr int kKeyHeads     = 2;
constexpr int kHeadsPerCta  = 7;
constexpr int kCombined     = kQueryHeads + kKeyHeads;
constexpr int kCtasPerToken = (kCombined + kHeadsPerCta - 1) / kHeadsPerCta;
constexpr int kPairsPerLane = kHeadDim / 2 / kWarpSize;
constexpr int kHalf         = kRotaryDim / 2;
static_assert(kPairsPerLane == 4 && kHalf == kWarpSize,
              "the rotary span is exactly the first BF16x2 pair of every lane");
static_assert(kCombined % kHeadsPerCta == 0, "every warp owns a head");

// Yarn: the coefficients of the YaRN table installed on the device, as rope_generic_kernel<true>.
template <bool Yarn>
__global__ __launch_bounds__(kHeadsPerCta* kWarpSize) void rmsnorm_rope_text_shard_kernel(
    const std::int32_t* __restrict__ positions, std::int32_t axes, float theta,
    const __nv_bfloat162* __restrict__ q, const __nv_bfloat162* __restrict__ k,
    const __nv_bfloat162* __restrict__ q_weight, const __nv_bfloat162* __restrict__ k_weight,
    __nv_bfloat162* __restrict__ q_out, __nv_bfloat162* __restrict__ k_out, std::int32_t tokens,
    float eps) {
    const int token = static_cast<int>(blockIdx.x);
    const int lane  = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int head =
        static_cast<int>(blockIdx.y) * kHeadsPerCta + static_cast<int>(threadIdx.x) / kWarpSize;

    __shared__ float cos_cache[kHalf];
    __shared__ float sin_cache[kHalf];
    if (threadIdx.x < kHalf) {
        // The coefficients of rope_generic_kernel<Yarn> for a Text head.
        const int pair = static_cast<int>(threadIdx.x);
        const int axis = text_rope_axis(axes, pair);
        if constexpr (Yarn) {
            text_yarn_sincos(positions[static_cast<std::int64_t>(axis) * tokens + token], pair,
                             &sin_cache[pair], &cos_cache[pair]);
        } else {
            const float exponent =
                -2.0F * static_cast<float>(pair) / static_cast<float>(kRotaryDim);
            const float frequency = powf(theta, exponent);
            const float angle =
                static_cast<float>(positions[static_cast<std::int64_t>(axis) * tokens + token]) *
                frequency;
            sincosf(angle, &sin_cache[pair], &cos_cache[pair]);
        }
    }

    const bool query = head < kQueryHeads;
    const std::int64_t row_base =
        (static_cast<std::int64_t>(token) * (query ? kQueryHeads : kKeyHeads) +
         (query ? head : head - kQueryHeads)) *
        (kHeadDim / 2);
    const __nv_bfloat162* x      = query ? q : k;
    const __nv_bfloat162* weight = query ? q_weight : k_weight;
    __nv_bfloat162* out          = query ? q_out : k_out;

    // rmsnorm_warp_bf16x2_kernel<Offset, 128, Prefetch=true, 256>.
    __nv_bfloat162 values[kPairsPerLane];
    __nv_bfloat162 weights[kPairsPerLane];
    float sum = 0.0f;
#pragma unroll
    for (int p = 0; p < kPairsPerLane; ++p) {
        const int pair  = lane + p * kWarpSize;
        values[p]       = x[row_base + pair];
        weights[p]      = weight[pair];
        const float2 xf = __bfloat1622float2(values[p]);
        sum += xf.x * xf.x + xf.y * xf.y;
    }
    sum       = warp_reduce_sum(sum);
    float inv = lane == 0 ? rsqrtf(sum / static_cast<float>(kHeadDim) + eps) : 0.0f;
    inv       = __shfl_sync(kFullWarpMask, inv, 0);

    __nv_bfloat162 normalized[kPairsPerLane];
#pragma unroll
    for (int p = 0; p < kPairsPerLane; ++p) {
        const float2 xf = __bfloat1622float2(values[p]);
        const float2 wf = __bfloat1622float2(weights[p]);
        normalized[p] =
            __floats2bfloat162_rn(rmsnorm_epilogue<RmsEpilogue::Offset>(xf.x, inv, wf.x, 0.0f),
                                  rmsnorm_epilogue<RmsEpilogue::Offset>(xf.y, inv, wf.y, 0.0f));
    }
    // Elements 2l, 2l+1 of lane l < 16 rotate against elements 2l+32, 2l+33 of lane l+16.
    __nv_bfloat162 partner = normalized[0];
    {
        unsigned bits;
        static_assert(sizeof(bits) == sizeof(partner));
        memcpy(&bits, &partner, sizeof(bits));
        bits = __shfl_xor_sync(kFullWarpMask, bits, kHalf / 2);
        memcpy(&partner, &bits, sizeof(bits));
    }
    __syncthreads();

    // rope_generic_kernel on the BF16 values just written by the norm.
    const bool second_half = lane >= kHalf / 2;
    const int pair0        = 2 * lane - (second_half ? kHalf : 0);
    const float c0 = cos_cache[pair0], c1 = cos_cache[pair0 + 1];
    const float s0 = sin_cache[pair0], s1 = sin_cache[pair0 + 1];
    const float2 mine  = __bfloat1622float2(normalized[0]);
    const float2 other = __bfloat1622float2(partner);
    __nv_bfloat162 rotated;
    if (!second_half) {
        const float first_x = mine.x, first_y = mine.y, second_x = other.x, second_y = other.y;
        rotated = __halves2bfloat162(__float2bfloat16_rn(first_x * c0 - second_x * s0),
                                     __float2bfloat16_rn(first_y * c1 - second_y * s1));
    } else {
        const float first_x = other.x, first_y = other.y, second_x = mine.x, second_y = mine.y;
        rotated = __halves2bfloat162(__float2bfloat16_rn(second_x * c0 + first_x * s0),
                                     __float2bfloat16_rn(second_y * c1 + first_y * s1));
    }
    out[row_base + lane] = rotated;
#pragma unroll
    for (int p = 1; p < kPairsPerLane; ++p) {
        out[row_base + lane + p * kWarpSize] = normalized[p];
    }
}

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto a0 = reinterpret_cast<std::uintptr_t>(a.data);
    const auto b0 = reinterpret_cast<std::uintptr_t>(b.data);
    return a0 < b0 + b.bytes() && b0 < a0 + a.bytes();
}

void require(const Tensor& t, DType dtype, std::int32_t n0, std::int32_t n1, std::int32_t n2,
             std::int32_t n3, const char* label) {
    if (t.dtype != dtype || t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != n2 || t.ne[3] != n3 ||
        !t.is_contiguous() || t.data == nullptr ||
        (reinterpret_cast<std::uintptr_t>(t.data) & 3U) != 0) {
        throw std::invalid_argument(std::string("rmsnorm_rope_text_shard: invalid ") + label);
    }
}

} // namespace

void rmsnorm_rope_text_shard(const Tensor& positions, int rotary_dim, float theta,
                             const Tensor& q_norm_weight, const Tensor& k_norm_weight, float eps,
                             const Tensor& q, const Tensor& k, Tensor& q_out, Tensor& k_out,
                             cudaStream_t stream) {
    const std::int32_t tokens = q.ne[2];
    if (tokens <= 0) { throw std::invalid_argument("rmsnorm_rope_text_shard: T must be positive"); }
    if (rotary_dim != kRotaryDim) {
        throw std::invalid_argument("rmsnorm_rope_text_shard: rotary_dim must be 64");
    }
    if (!(theta > 0.0f) || !std::isfinite(theta) || !(eps > 0.0f) || !std::isfinite(eps)) {
        throw std::invalid_argument("rmsnorm_rope_text_shard: theta and eps must be positive");
    }
    const std::int32_t axes = positions.ne[1];
    if (axes != 1 && axes != 3) {
        throw std::invalid_argument("rmsnorm_rope_text_shard: positions must be [T] or [T,3]");
    }
    require(positions, DType::I32, tokens, axes, 1, 1, "positions");
    require(q, DType::BF16, kHeadDim, kQueryHeads, tokens, 1, "q");
    require(k, DType::BF16, kHeadDim, kKeyHeads, tokens, 1, "k");
    require(q_out, DType::BF16, kHeadDim, kQueryHeads, tokens, 1, "q_out");
    require(k_out, DType::BF16, kHeadDim, kKeyHeads, tokens, 1, "k_out");
    require(q_norm_weight, DType::BF16, kHeadDim, 1, 1, 1, "q norm weight");
    require(k_norm_weight, DType::BF16, kHeadDim, 1, 1, 1, "k norm weight");
    for (const Tensor* out : {&q_out, &k_out}) {
        for (const Tensor* in : {&q, &k, &positions, &q_norm_weight, &k_norm_weight}) {
            if (overlaps(*out, *in)) {
                throw std::invalid_argument("rmsnorm_rope_text_shard: outputs overlap an input");
            }
        }
    }
    if (overlaps(q_out, k_out)) {
        throw std::invalid_argument("rmsnorm_rope_text_shard: q_out and k_out overlap");
    }
    const dim3 grid(static_cast<unsigned>(tokens), kCtasPerToken);
    auto* kernel = detail::text_rope_yarn(rotary_dim, theta) != nullptr
                       ? rmsnorm_rope_text_shard_kernel<true>
                       : rmsnorm_rope_text_shard_kernel<false>;
    kernel<<<grid, kHeadsPerCta * kWarpSize, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data), axes, theta,
        static_cast<const __nv_bfloat162*>(q.data), static_cast<const __nv_bfloat162*>(k.data),
        static_cast<const __nv_bfloat162*>(q_norm_weight.data),
        static_cast<const __nv_bfloat162*>(k_norm_weight.data),
        static_cast<__nv_bfloat162*>(q_out.data), static_cast<__nv_bfloat162*>(k_out.data), tokens,
        eps);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
