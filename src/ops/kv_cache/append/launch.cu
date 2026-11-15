#include "ops/kv_cache/append/launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kv_cache/append/kernel.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;

template <typename Geometry, typename CacheView, typename Metadata>
void launch_full(const Tensor& k, const Tensor& v, const Tensor& positions, CacheView cache,
                 Metadata metadata, cudaStream_t stream) {
    const auto tokens = static_cast<std::int32_t>(k.ne[2]);
    Tensor& cache_k   = cache.k_pages;
    Tensor& cache_v   = cache.v_pages;
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        Tensor& cache_k_scale = cache.k_scale_pages;
        Tensor& cache_v_scale = cache.v_scale_pages;
        if (tokens >= 128 && Geometry::KVHeads == 2) {
            constexpr int TokensPerTile = 8;
            const int max_tiles         = div_up(tokens + TokensPerTile - 1, TokensPerTile);
            const dim3 fill_grid(static_cast<unsigned>(max_tiles),
                                 static_cast<unsigned>(Geometry::KVHeads));
            kv_cache_append_full_fp8_page_kernel<Geometry, Metadata>
                <<<fill_grid, kBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data),
                    static_cast<const std::int32_t*>(positions.data), metadata,
                    static_cast<std::uint8_t*>(cache_k.data),
                    static_cast<std::uint8_t*>(cache_v.data),
                    static_cast<__half*>(cache_k_scale.data),
                    static_cast<__half*>(cache_v_scale.data), tokens);
        } else {
            constexpr int FillWarps       = kBlock / 32;
            const std::int64_t fill_units = static_cast<std::int64_t>(tokens) * Geometry::KVHeads;
            const int fill_grid =
                static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(FillWarps)));
            kv_cache_append_full_fp8_kernel<Geometry, Metadata><<<fill_grid, kBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(k.data),
                static_cast<const __nv_bfloat16*>(v.data),
                static_cast<const std::int32_t*>(positions.data), metadata,
                static_cast<std::uint8_t*>(cache_k.data), static_cast<std::uint8_t*>(cache_v.data),
                static_cast<__half*>(cache_k_scale.data), static_cast<__half*>(cache_v_scale.data),
                tokens);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (cache.storage == KvCacheStorage::Int8Group64) {
        Tensor& cache_k_scale = cache.k_scale_pages;
        Tensor& cache_v_scale = cache.v_scale_pages;
        if (tokens >= 128 && Geometry::KVHeads == 2) {
            constexpr int TokensPerTile = 8;
            const int max_tiles         = div_up(tokens + TokensPerTile - 1, TokensPerTile);
            const dim3 fill_grid(static_cast<unsigned>(max_tiles),
                                 static_cast<unsigned>(Geometry::KVHeads));
            kv_cache_append_full_i8_page_kernel<Geometry, Metadata>
                <<<fill_grid, kBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data),
                    static_cast<const std::int32_t*>(positions.data), metadata,
                    static_cast<std::int8_t*>(cache_k.data),
                    static_cast<std::int8_t*>(cache_v.data),
                    static_cast<__half*>(cache_k_scale.data),
                    static_cast<__half*>(cache_v_scale.data), tokens);
        } else {
            constexpr int FillWarps       = kBlock / 32;
            const std::int64_t fill_units = static_cast<std::int64_t>(tokens) * Geometry::KVHeads;
            const int fill_grid =
                static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(FillWarps)));
            kv_cache_append_full_i8_kernel<Geometry, Metadata><<<fill_grid, kBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(k.data),
                static_cast<const __nv_bfloat16*>(v.data),
                static_cast<const std::int32_t*>(positions.data), metadata,
                static_cast<std::int8_t*>(cache_k.data), static_cast<std::int8_t*>(cache_v.data),
                static_cast<__half*>(cache_k_scale.data), static_cast<__half*>(cache_v_scale.data),
                tokens);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    constexpr int Block         = Geometry::KVHeads == 4 ? 128 : 96;
    constexpr int VecElems      = 8;
    const std::int64_t elements = static_cast<std::int64_t>(tokens) * Geometry::KVHeads *
                                  (kKVCacheAppendFullHeadDim / VecElems);
    const int fill_grid = static_cast<int>(div_up(elements, static_cast<std::int64_t>(Block)));
    kv_cache_append_full_bf16_kernel<Geometry, Metadata><<<fill_grid, Block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const std::int32_t*>(positions.data), metadata,
        static_cast<__nv_bfloat16*>(cache_k.data), static_cast<__half*>(cache_v.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void kv_cache_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                            PagedKVLayerView cache, cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    if (k.ne[1] == KVCacheAppendD256Kv4::KVHeads) {
        launch_full<KVCacheAppendD256Kv4>(k, v, positions, cache, metadata, stream);
        return;
    }
    launch_full<KVCacheAppendD256Kv2>(k, v, positions, cache, metadata, stream);
}

void kv_cache_append_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                  const Tensor& valid_columns, const Tensor& table_rows,
                                  PagedKVBatchLayerView cache, cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        kv_cache_append_k8v4_batch_launch(k, v, positions, valid_columns, table_rows, cache,
                                          stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        kv_cache_append_nvfp4_batch_launch(k, v, positions, valid_columns, table_rows, cache,
                                           stream);
        return;
    }
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        if (cache.storage == KvCacheStorage::Fp8E4M3Row256 && k.ne[3] > 1) {
            const auto append = [&]<class Geometry>() {
                const dim3 grid(div_up(k.ne[2] * Geometry::KVHeads, 8), 1, k.ne[3]);
                kv_cache_append_full_fp8_batch_kernel<Geometry, Masked>
                    <<<grid, kBlock, 0, stream>>>(
                        static_cast<const __nv_bfloat16*>(k.data),
                        static_cast<const __nv_bfloat16*>(v.data),
                        static_cast<const std::int32_t*>(positions.data), metadata,
                        static_cast<std::uint8_t*>(cache.k_pages.data),
                        static_cast<std::uint8_t*>(cache.v_pages.data),
                        static_cast<__half*>(cache.k_scale_pages.data),
                        static_cast<__half*>(cache.v_scale_pages.data), k.ne[2]);
            };
            if (k.ne[1] == KVCacheAppendD256Kv4::KVHeads)
                append.template operator()<KVCacheAppendD256Kv4>();
            else
                append.template operator()<KVCacheAppendD256Kv2>();
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (cache.storage == KvCacheStorage::Int8Group64 && k.ne[3] > 1) {
            const auto append = [&]<class Geometry>() {
                const dim3 grid(div_up(k.ne[2] * Geometry::KVHeads, 8), 1, k.ne[3]);
                kv_cache_append_full_i8_kernel<Geometry, PagedKVBatchMetadata<Masked>, true>
                    <<<grid, kBlock, 0, stream>>>(
                        static_cast<const __nv_bfloat16*>(k.data),
                        static_cast<const __nv_bfloat16*>(v.data),
                        static_cast<const std::int32_t*>(positions.data), metadata,
                        static_cast<std::int8_t*>(cache.k_pages.data),
                        static_cast<std::int8_t*>(cache.v_pages.data),
                        static_cast<__half*>(cache.k_scale_pages.data),
                        static_cast<__half*>(cache.v_scale_pages.data), k.ne[2]);
            };
            if (k.ne[1] == KVCacheAppendD256Kv4::KVHeads)
                append.template operator()<KVCacheAppendD256Kv4>();
            else
                append.template operator()<KVCacheAppendD256Kv2>();
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (k.ne[1] == KVCacheAppendD256Kv4::KVHeads) {
            launch_full<KVCacheAppendD256Kv4>(k, v, positions, cache, metadata, stream);
            return;
        }
        launch_full<KVCacheAppendD256Kv2>(k, v, positions, cache, metadata, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
