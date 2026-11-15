#include "ninfer/ops/kv_cache_append.h"

#include "core/paged_kv_storage.h"
#include "ops/kv_cache/append/launch.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kFullHeadDim = 256;
constexpr const char* kAppendOp     = "kv_cache_append";

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* op, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != n3) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

void require_contiguous_nonnull(const Tensor& tensor, const char* op, const char* name) {
    if (!tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name +
                                    " must be contiguous and non-null");
    }
}

std::uint32_t validate_full_cache(const PagedKVLayerView& cache, std::int32_t kv_heads) {
    PagedKVStorageLayout layout{};
    try {
        layout = paged_kv_storage_layout(cache.storage, kFullHeadDim);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument("kv_cache_append: invalid cache geometry or storage");
    }
    if (cache.num_kv_heads != kv_heads || cache.head_dim != kFullHeadDim) {
        throw std::invalid_argument("kv_cache_append: invalid cache geometry or storage");
    }

    const std::int32_t physical_pages = cache.k_pages.ne[3];
    const std::int32_t logical_pages  = cache.block_table.ne[0];
    const std::int64_t capacity       = static_cast<std::int64_t>(logical_pages) * kPagedKVPageSize;
    if (physical_pages <= 0 || logical_pages <= 0 ||
        capacity > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("kv_cache_append: invalid cache capacity");
    }

    if (cache.k_pages.dtype != layout.key.data_dtype ||
        cache.v_pages.dtype != layout.value.data_dtype) {
        throw std::invalid_argument("kv_cache_append: invalid cache data dtype");
    }
    require_shape(cache.k_pages, layout.key.data_leading_extent, kPagedKVPageSize, kv_heads,
                  physical_pages, kAppendOp, "cache k pages");
    require_shape(cache.v_pages, layout.value.data_leading_extent, kPagedKVPageSize, kv_heads,
                  physical_pages, kAppendOp, "cache v pages");
    require_contiguous_nonnull(cache.k_pages, kAppendOp, "cache k pages");
    require_contiguous_nonnull(cache.v_pages, kAppendOp, "cache v pages");
    if (cache.block_table.dtype != DType::I32) {
        throw std::invalid_argument("kv_cache_append: block table must be I32");
    }
    require_shape(cache.block_table, logical_pages, 1, 1, 1, kAppendOp, "block table");
    require_contiguous_nonnull(cache.block_table, kAppendOp, "block table");

    const auto validate_scale = [&](const Tensor& tensor, const PagedKVVectorLayout& vector,
                                    const char* name) {
        if (!vector.has_scale()) {
            if (tensor.data != nullptr) {
                throw std::invalid_argument("kv_cache_append: unscaled cache must not have scales");
            }
            return;
        }
        if (tensor.dtype != vector.scale_dtype) {
            throw std::invalid_argument("kv_cache_append: invalid cache scale dtype");
        }
        require_shape(tensor, vector.scale_leading_extent, kPagedKVPageSize, kv_heads,
                      physical_pages, kAppendOp, name);
        require_contiguous_nonnull(tensor, kAppendOp, name);
    };
    validate_scale(cache.k_scale_pages, layout.key, "cache k scale pages");
    validate_scale(cache.v_scale_pages, layout.value, "cache v scale pages");
    return static_cast<std::uint32_t>(capacity);
}

} // namespace

void kv_cache_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                     PagedKVLayerView cache, cudaStream_t stream) {
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("kv_cache_append: k/v must be BF16");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("kv_cache_append: positions must be I32");
    }
    const std::int32_t kv_heads = k.ne[1];
    if (kv_heads != 4 && kv_heads != 2) {
        throw std::invalid_argument("kv_cache_append: unsupported KV head geometry");
    }
    const std::int32_t tokens = k.ne[2];
    if (tokens <= 0) { throw std::invalid_argument("kv_cache_append: T must be positive"); }
    require_shape(k, kFullHeadDim, kv_heads, tokens, 1, kAppendOp, "k");
    require_shape(v, kFullHeadDim, kv_heads, tokens, 1, kAppendOp, "v");
    require_shape(positions, tokens, 1, 1, 1, kAppendOp, "positions");
    require_contiguous_nonnull(k, kAppendOp, "k");
    require_contiguous_nonnull(v, kAppendOp, "v");
    require_contiguous_nonnull(positions, kAppendOp, "positions");
    const std::uint32_t capacity = validate_full_cache(cache, kv_heads);
    if (static_cast<std::uint32_t>(tokens) > capacity) {
        throw std::invalid_argument("kv_cache_append: T exceeds cache capacity");
    }
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        detail::kv_cache_append_k8v4_launch(k, v, positions, cache, stream);
    } else if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        detail::kv_cache_append_nvfp4_launch(k, v, positions, cache, stream);
    } else {
        detail::kv_cache_append_launch(k, v, positions, cache, stream);
    }
}

} // namespace ninfer::ops
