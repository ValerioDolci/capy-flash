#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/append/geometry.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

template <typename Geometry>
__device__ __forceinline__ void
kv_cache_append_full_fp8_row(const __nv_bfloat16* __restrict__ k,
                             const __nv_bfloat16* __restrict__ v,
                             std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
                             __half* __restrict__ scale_k, __half* __restrict__ scale_v, int token,
                             int kv_head, int physical_page, int page_off, int lane) {
    constexpr unsigned FullMask = 0xffffffffU;
    float values[8];
    float local_absmax = 0.0F;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        values[r]   = __bfloat162float(k[kv_cache_fp8_src_index<Geometry>(kv_head, d, token)]);
    }
    normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (float value : values) { local_absmax = fmaxf(local_absmax, fabsf(value)); }
    const auto k_quant = kv_cache_fp8_quant_params(warp_max(local_absmax, FullMask));
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        cache_k[kv_cache_fp8_code_index<Geometry>(physical_page, kv_head, d, page_off)] =
            kv_cache_fp8_quant_code(values[r], k_quant.inverse_scale);
    }
    if (lane == 0) {
        scale_k[kv_cache_fp8_scale_index<Geometry>(physical_page, kv_head, page_off)] =
            k_quant.scale;
    }

    local_absmax = 0.0F;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d  = lane + 32 * r;
        values[r]    = __bfloat162float(v[kv_cache_fp8_src_index<Geometry>(kv_head, d, token)]);
        local_absmax = fmaxf(local_absmax, fabsf(values[r]));
    }
    const auto v_quant = kv_cache_fp8_quant_params(warp_max(local_absmax, FullMask));
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        cache_v[kv_cache_fp8_code_index<Geometry>(physical_page, kv_head, d, page_off)] =
            kv_cache_fp8_quant_code(values[r], v_quant.inverse_scale);
    }
    if (lane == 0) {
        scale_v[kv_cache_fp8_scale_index<Geometry>(physical_page, kv_head, page_off)] =
            v_quant.scale;
    }
}

template <typename Geometry, typename Metadata>
__global__ void kv_cache_append_full_bf16_kernel(const __nv_bfloat16* __restrict__ k,
                                                 const __nv_bfloat16* __restrict__ v,
                                                 const std::int32_t* __restrict__ positions,
                                                 Metadata metadata,
                                                 __nv_bfloat16* __restrict__ cache_k,
                                                 __half* __restrict__ cache_v, std::int32_t width) {
    constexpr int VecElems = 8;
    const int tokens       = metadata.valid_tokens(width);
    const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t n   = static_cast<std::int64_t>(tokens) * Geometry::KVHeads *
                           (kKVCacheAppendFullHeadDim / VecElems);
    if (idx >= n) return;

    const int vec      = static_cast<int>(idx % (kKVCacheAppendFullHeadDim / VecElems));
    const int tmp      = static_cast<int>(idx / (kKVCacheAppendFullHeadDim / VecElems));
    const int kv_head  = tmp % Geometry::KVHeads;
    const int token    = tmp / Geometry::KVHeads;
    const int d        = vec * VecElems;
    const int position = positions[0] + token;
    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    const std::int64_t src_off =
        static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kKVCacheAppendFullHeadDim) *
                                           (kv_head + Geometry::KVHeads * token);
    const int4 k_value = load_vec<int4>(&k[src_off]);
    const int4 v_value = bf16x8_bits_to_f16x8_bits(load_vec<int4>(&v[src_off]));
    physical_page      = __shfl_sync(0xffffffffu, physical_page, 0);
    const std::int64_t cache_off =
        paged_kv_element_offset<kKVCacheAppendFullHeadDim, Geometry::KVHeads>(
            physical_page, kv_head, position & kPagedKVPageMask, d);
    store_vec(&cache_k[cache_off], k_value);
    store_vec(&cache_v[cache_off], v_value);
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void kv_cache_append_full_fp8_kernel(const __nv_bfloat16* __restrict__ k,
                                         const __nv_bfloat16* __restrict__ v,
                                         const std::int32_t* __restrict__ positions,
                                         Metadata metadata, std::uint8_t* __restrict__ cache_k,
                                         std::uint8_t* __restrict__ cache_v,
                                         __half* __restrict__ scale_k, __half* __restrict__ scale_v,
                                         std::int32_t width) {
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int unit              = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units             = tokens * Geometry::KVHeads;
    if (unit >= units) return;

    const int kv_head               = unit % Geometry::KVHeads;
    const int token                 = unit / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    physical_page                   = __shfl_sync(FullMask, physical_page, 0);
    kv_cache_append_full_fp8_row<Geometry>(k, v, cache_k, cache_v, scale_k, scale_v, token, kv_head,
                                           physical_page, position & kPagedKVPageMask, lane);
}

// Independent request rows share physical planes, but select separate input and table rows.
template <typename Geometry, bool Masked>
__launch_bounds__(256) __global__
    void kv_cache_append_full_fp8_batch_kernel(const __nv_bfloat16* k, const __nv_bfloat16* v,
                                               const std::int32_t* positions,
                                               PagedKVBatchMetadata<Masked> metadata,
                                               std::uint8_t* cache_k, std::uint8_t* cache_v,
                                               __half* scale_k, __half* scale_v,
                                               std::int32_t width) {
    const int batch = blockIdx.z;
    metadata.table_rows += batch;
    if constexpr (Masked) metadata.valid_columns += batch;
    const int tokens = metadata.valid_tokens(width);
    const int unit   = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (unit >= tokens * Geometry::KVHeads) return;
    const int token         = unit / Geometry::KVHeads;
    const int kv_head       = unit % Geometry::KVHeads;
    const int lane          = threadIdx.x & 31;
    const auto input_offset = static_cast<std::int64_t>(batch) * width * 256 * Geometry::KVHeads;
    const int position      = positions[batch * width] + token;
    const auto* table       = metadata.block_table();
    int page                = lane == 0 ? paged_kv_physical_page(table, position) : 0;
    page                    = __shfl_sync(0xffffffffU, page, 0);
    kv_cache_append_full_fp8_row<Geometry>(k + input_offset, v + input_offset, cache_k, cache_v,
                                           scale_k, scale_v, token, kv_head, page,
                                           position & kPagedKVPageMask, lane);
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void kv_cache_append_full_fp8_page_kernel(const __nv_bfloat16* __restrict__ k,
                                              const __nv_bfloat16* __restrict__ v,
                                              const std::int32_t* __restrict__ positions,
                                              Metadata metadata, std::uint8_t* __restrict__ cache_k,
                                              std::uint8_t* __restrict__ cache_v,
                                              __half* __restrict__ scale_k,
                                              __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int TokensPerTile = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int kv_head           = static_cast<int>(blockIdx.y);
    const int tile_delta        = static_cast<int>(blockIdx.x);
    const int base_position     = positions[0];
    const int tile_position     = (base_position / TokensPerTile + tile_delta) * TokensPerTile;
    const int logical_page      = tile_position >> kPagedKVPageShift;
    const int token_begin       = max(0, tile_position - base_position);
    const int token_end         = min(tokens, tile_position + TokensPerTile - base_position);
    if (token_begin >= token_end) return;

    const int token = token_begin + warp;
    if (token >= token_end) return;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? block_table[logical_page] : 0;
    physical_page                   = __shfl_sync(FullMask, physical_page, 0);
    const int position              = base_position + token;
    kv_cache_append_full_fp8_row<Geometry>(k, v, cache_k, cache_v, scale_k, scale_v, token, kv_head,
                                           physical_page, position & kPagedKVPageMask, lane);
}

template <typename Geometry, typename Metadata, bool MultiBatch = false>
__launch_bounds__(256) __global__
    void kv_cache_append_full_i8_kernel(const __nv_bfloat16* __restrict__ k,
                                        const __nv_bfloat16* __restrict__ v,
                                        const std::int32_t* __restrict__ positions,
                                        Metadata metadata, std::int8_t* __restrict__ cache_k,
                                        std::int8_t* __restrict__ cache_v,
                                        __half* __restrict__ scale_k, __half* __restrict__ scale_v,
                                        std::int32_t width) {
    if constexpr (MultiBatch) {
        const int batch = blockIdx.z;
        metadata.table_rows += batch;
        if (metadata.valid_columns) metadata.valid_columns += batch;
        const auto offset = static_cast<std::int64_t>(batch) * width * 256 * Geometry::KVHeads;
        k += offset;
        v += offset;
        positions += batch * width;
    }
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffu;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int unit              = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units             = tokens * Geometry::KVHeads;
    if (unit >= units) return;

    const int kv_head               = unit % Geometry::KVHeads;
    const int token                 = unit / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int page                        = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    const int page_off              = position & kPagedKVPageMask;
    page                            = __shfl_sync(FullMask, page, 0);

    float k_values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        k_values[r] =
            __bfloat162float(k[kv_cache_int8_quant_src_index<Geometry>(kv_head, d, token)]);
    }
    normalized_hadamard_d256_inplace(k_values, lane);

#pragma unroll
    for (int group = 0; group < kKVCacheInt8Groups; ++group) {
        const int d0                 = group * kKVCacheInt8Group + lane;
        const int d1                 = d0 + 32;
        const float k0               = k_values[2 * group];
        const float k1               = k_values[2 * group + 1];
        const std::int64_t src0      = kv_cache_int8_quant_src_index<Geometry>(kv_head, d0, token);
        const std::int64_t src1      = kv_cache_int8_quant_src_index<Geometry>(kv_head, d1, token);
        const float v0               = __bfloat162float(v[src0]);
        const float v1               = __bfloat162float(v[src1]);
        const float k_abs            = warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
        const float v_abs            = warp_max(fmaxf(fabsf(v0), fabsf(v1)), FullMask);
        const auto k_quant           = kv_cache_int8_quant_params(k_abs);
        const auto v_quant           = kv_cache_int8_quant_params(v_abs);
        const std::int64_t code_base = kv_cache_int8_quant_code_index<Geometry>(
            page, kv_head, group * kKVCacheInt8Group, page_off);
        cache_k[code_base + lane]      = kv_cache_int8_quant_code(k0, k_quant.inverse_scale);
        cache_k[code_base + lane + 32] = kv_cache_int8_quant_code(k1, k_quant.inverse_scale);
        cache_v[code_base + lane]      = kv_cache_int8_quant_code(v0, v_quant.inverse_scale);
        cache_v[code_base + lane + 32] = kv_cache_int8_quant_code(v1, v_quant.inverse_scale);
        if (lane == 0) {
            const std::int64_t scale_off =
                kv_cache_int8_quant_scale_index<Geometry>(page, kv_head, group, page_off);
            scale_k[scale_off] = k_quant.scale;
            scale_v[scale_off] = v_quant.scale;
        }
    }
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void kv_cache_append_full_i8_page_kernel(const __nv_bfloat16* __restrict__ k,
                                             const __nv_bfloat16* __restrict__ v,
                                             const std::int32_t* __restrict__ positions,
                                             Metadata metadata, std::int8_t* __restrict__ cache_k,
                                             std::int8_t* __restrict__ cache_v,
                                             __half* __restrict__ scale_k,
                                             __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int TokensPerTile = 8;
    constexpr unsigned FullMask = 0xffffffffu;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int kv_head           = static_cast<int>(blockIdx.y);
    const int tile_delta        = static_cast<int>(blockIdx.x);
    const int base_position     = positions[0];
    const int tile_position     = (base_position / TokensPerTile + tile_delta) * TokensPerTile;
    const int logical_page      = tile_position >> kPagedKVPageShift;
    const int token_begin       = max(0, tile_position - base_position);
    const int token_end         = min(tokens, tile_position + TokensPerTile - base_position);
    if (token_begin >= token_end) return;

    const int token = token_begin + warp;
    if (token >= token_end) return;

    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? block_table[logical_page] : 0;
    physical_page                   = __shfl_sync(FullMask, physical_page, 0);

    const int position = base_position + token;
    const int page_off = position & kPagedKVPageMask;

    float k_values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        k_values[r] =
            __bfloat162float(k[kv_cache_int8_quant_src_index<Geometry>(kv_head, d, token)]);
    }
    normalized_hadamard_d256_inplace(k_values, lane);

#pragma unroll
    for (int group = 0; group < kKVCacheInt8Groups; ++group) {
        const int d0                 = group * kKVCacheInt8Group + lane;
        const int d1                 = d0 + 32;
        const float k0               = k_values[2 * group];
        const float k1               = k_values[2 * group + 1];
        const std::int64_t src0      = kv_cache_int8_quant_src_index<Geometry>(kv_head, d0, token);
        const std::int64_t src1      = kv_cache_int8_quant_src_index<Geometry>(kv_head, d1, token);
        const float v0               = __bfloat162float(v[src0]);
        const float v1               = __bfloat162float(v[src1]);
        const float k_abs            = warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
        const float v_abs            = warp_max(fmaxf(fabsf(v0), fabsf(v1)), FullMask);
        const auto k_quant           = kv_cache_int8_quant_params(k_abs);
        const auto v_quant           = kv_cache_int8_quant_params(v_abs);
        const std::int64_t code_base = kv_cache_int8_quant_code_index<Geometry>(
            physical_page, kv_head, group * kKVCacheInt8Group, page_off);
        cache_k[code_base + lane]      = kv_cache_int8_quant_code(k0, k_quant.inverse_scale);
        cache_k[code_base + lane + 32] = kv_cache_int8_quant_code(k1, k_quant.inverse_scale);
        cache_v[code_base + lane]      = kv_cache_int8_quant_code(v0, v_quant.inverse_scale);
        cache_v[code_base + lane + 32] = kv_cache_int8_quant_code(v1, v_quant.inverse_scale);
        if (lane == 0) {
            const std::int64_t scale_offset =
                kv_cache_int8_quant_scale_index<Geometry>(physical_page, kv_head, group, page_off);
            scale_k[scale_offset] = k_quant.scale;
            scale_v[scale_offset] = v_quant.scale;
        }
    }
}

} // namespace ninfer::ops
