#include "ninfer/ops/relative_sink_attention.h"

#include "core/layout.h"
#include "ops/relative_sink_attention/launch.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr const char* kOp           = "relative_sink_attention";
constexpr std::int32_t kMaxHeadDim  = 256;
constexpr std::int32_t kMaxRelative = 32;

[[noreturn]] void reject(const std::string& what) {
    throw std::invalid_argument(std::string(kOp) + ": " + what);
}

void require_tensor(const Tensor& tensor, DType dtype, std::int32_t n0, std::int32_t n1,
                    std::int32_t n2, std::int32_t n3, const char* name) {
    if (tensor.data == nullptr || tensor.dtype != dtype || tensor.ne[0] != n0 ||
        tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != n3 || !tensor.is_contiguous()) {
        reject(std::string("invalid ") + name);
    }
}

void require_geometry(AttentionHeadGeometry geometry, std::int32_t relative_dim) {
    if (!valid_attention_head_geometry(geometry) || geometry.head_dim % 32 != 0 ||
        geometry.head_dim > kMaxHeadDim || relative_dim < 1 || relative_dim > kMaxRelative) {
        reject("unsupported head geometry or relative dimension");
    }
}

void require_params(const RelativeSinkAttentionParams& params) {
    const auto positive    = [](float x) { return std::isfinite(x) && x > 0.0f; };
    const auto nonnegative = [](float x) { return std::isfinite(x) && x >= 0.0f; };
    if (params.relative_extent < 1 || !positive(params.qk_clip) ||
        !positive(params.relative_clip) || !nonnegative(params.clip_epsilon) ||
        !nonnegative(params.value_epsilon) || !positive(params.scale) ||
        !std::isfinite(params.log_scale_alpha) || !positive(params.log_scale_floor)) {
        reject("invalid parameters");
    }
}

// Key and value rows are read as 16-byte vectors.
void require_vector_rows(const Tensor& keys) {
    bool aligned = reinterpret_cast<std::uintptr_t>(keys.data) % 16 == 0;
    for (int axis = 1; axis < 4; ++axis) aligned = aligned && keys.nb[axis] % 16 == 0;
    if (!aligned) reject("cache K/V rows must be 16-byte aligned");
}

std::int64_t element_stride(const Tensor& tensor, int axis) {
    const std::int64_t bytes = tensor.nb[axis];
    const auto size          = static_cast<std::int64_t>(dtype_size(tensor.dtype));
    if (bytes <= 0 || bytes % size != 0) reject("cache stride is not element aligned");
    return bytes / size;
}

struct Prepared {
    Tensor q, r, k, v, chunk_max, chunk_sum, chunk_out;
};

template <class Allocator>
Prepared allocate_prepared(Allocator& allocator, AttentionHeadGeometry geometry,
                           std::int32_t relative_dim, std::int32_t tokens, std::int32_t batch) {
    Prepared prepared;
    prepared.q =
        allocator.alloc(DType::BF16, {geometry.head_dim, geometry.query_heads, tokens, batch});
    prepared.r = allocator.alloc(DType::BF16, {relative_dim, geometry.query_heads, tokens, batch});
    prepared.k =
        allocator.alloc(DType::BF16, {geometry.head_dim, geometry.kv_heads, tokens, batch});
    prepared.v =
        allocator.alloc(DType::FP16, {geometry.head_dim, geometry.kv_heads, tokens, batch});
    if (tokens <= detail::kRelativeSinkSplitMaxTokens) {
        const std::int32_t rows = tokens * batch * geometry.query_heads;
        prepared.chunk_max = allocator.alloc(DType::FP32, {detail::kRelativeSinkMaxChunks, rows});
        prepared.chunk_sum = allocator.alloc(DType::FP32, {detail::kRelativeSinkMaxChunks, rows});
        prepared.chunk_out =
            allocator.alloc(DType::FP32, {geometry.head_dim, detail::kRelativeSinkMaxChunks, rows});
    }
    return prepared;
}

detail::RelativeSinkOperands
validate_operands(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& r,
                  const Tensor& sink, const Tensor& positions, const Tensor& rows,
                  const Tensor& q_scale, const Tensor& r_scale, const Tensor& rel_proj,
                  AttentionHeadGeometry geometry, const RelativeSinkAttentionParams& params,
                  const Tensor* gate, const Tensor& out) {
    require_geometry(geometry, params.relative_dim);
    require_params(params);
    const std::int32_t D  = geometry.head_dim;
    const std::int32_t Hq = geometry.query_heads;
    const std::int32_t Hk = geometry.kv_heads;
    const std::int32_t dr = params.relative_dim;
    const std::int32_t T  = q.ne[2];
    const std::int32_t B  = q.ne[3];
    if (T < 1 || B < 1 || B > 65535) reject("invalid token or batch extent");
    require_tensor(q, DType::BF16, D, Hq, T, B, "q");
    require_tensor(k, DType::BF16, D, Hk, T, B, "k");
    require_tensor(v, DType::BF16, D, Hk, T, B, "v");
    require_tensor(r, DType::BF16, dr, Hq, T, B, "r");
    require_tensor(sink, DType::BF16, Hq, T, B, 1, "sink");
    require_tensor(positions, DType::I32, T, B, 1, 1, "positions");
    require_tensor(rows, DType::I32, B, 1, 1, 1, "cache rows");
    require_tensor(q_scale, DType::BF16, D, Hq, 1, 1, "q_scale");
    require_tensor(r_scale, DType::BF16, dr, Hq, 1, 1, "r_scale");
    require_tensor(rel_proj, DType::BF16, params.relative_extent, dr, 1, 1, "rel_proj");
    require_tensor(out, DType::BF16, D, Hq, T, B, "out");
    if (gate != nullptr) {
        require_tensor(*gate, DType::BF16, D, Hq, T, B, "gate");
        if (gate->data == out.data) reject("gate and out must not overlap");
    }

    detail::RelativeSinkOperands operands;
    operands.q         = static_cast<const __nv_bfloat16*>(q.data);
    operands.k         = static_cast<const __nv_bfloat16*>(k.data);
    operands.v         = static_cast<const __nv_bfloat16*>(v.data);
    operands.r         = static_cast<const __nv_bfloat16*>(r.data);
    operands.sink      = static_cast<const __nv_bfloat16*>(sink.data);
    operands.gate      = gate != nullptr ? static_cast<const __nv_bfloat16*>(gate->data) : nullptr;
    operands.q_scale   = static_cast<const __nv_bfloat16*>(q_scale.data);
    operands.r_scale   = static_cast<const __nv_bfloat16*>(r_scale.data);
    operands.rel_proj  = static_cast<const __nv_bfloat16*>(rel_proj.data);
    operands.positions = static_cast<const std::int32_t*>(positions.data);
    operands.rows      = static_cast<const std::int32_t*>(rows.data);
    operands.out       = static_cast<__nv_bfloat16*>(out.data);
    operands.head_dim  = D;
    operands.query_heads = Hq;
    operands.kv_heads    = Hk;
    operands.tokens      = T;
    operands.batch       = B;
    operands.params      = params;
    return operands;
}

void bind_workspace(detail::RelativeSinkOperands& operands, const Prepared& prepared) {
    operands.q_prepared = static_cast<__nv_bfloat16*>(prepared.q.data);
    operands.r_prepared = static_cast<__nv_bfloat16*>(prepared.r.data);
    operands.k_prepared = static_cast<__nv_bfloat16*>(prepared.k.data);
    operands.v_prepared = static_cast<__half*>(prepared.v.data);
    operands.chunk_max  = static_cast<float*>(prepared.chunk_max.data);
    operands.chunk_sum  = static_cast<float*>(prepared.chunk_sum.data);
    operands.chunk_out  = static_cast<float*>(prepared.chunk_out.data);
}

} // namespace

std::size_t relative_sink_attention_workspace_capacity_bytes(AttentionHeadGeometry geometry,
                                                             std::int32_t relative_dim,
                                                             std::int32_t batch_size,
                                                             std::int32_t min_tokens,
                                                             std::int32_t max_tokens) {
    require_geometry(geometry, relative_dim);
    if (batch_size < 1 || batch_size > 65535 || min_tokens < 1 || max_tokens < min_tokens) {
        reject("invalid workspace interval");
    }
    // The split route's partials exist only up to its token bound: take the larger need of the
    // two routes over the interval.
    std::size_t capacity = 0;
    for (const std::int32_t tokens :
         {max_tokens, std::min(max_tokens, detail::kRelativeSinkSplitMaxTokens)}) {
        if (tokens < min_tokens) continue;
        WorkspaceLayoutBuilder layout;
        (void)allocate_prepared(layout, geometry, relative_dim, tokens, batch_size);
        capacity = std::max(capacity, layout.peak_bytes(1));
    }
    return capacity;
}

void relative_sink_attention_windowed(const Tensor& q, const Tensor& k, const Tensor& v,
                                      const Tensor& r, const Tensor& sink, const Tensor& positions,
                                      const Tensor& lanes, const Tensor& q_scale,
                                      const Tensor& r_scale, const Tensor& rel_proj,
                                      AttentionHeadGeometry geometry,
                                      const RelativeSinkAttentionParams& params,
                                      std::int32_t window, const CyclicKVCacheLayerView& cache,
                                      const Tensor* gate, WorkspaceArena& workspace, Tensor& out,
                                      cudaStream_t stream) {
    auto operands = validate_operands(q, k, v, r, sink, positions, lanes, q_scale, r_scale,
                                      rel_proj, geometry, params, gate, out);
    if (window < 1) reject("window must be positive");
    if (cache.capacity == 0 ||
        cache.capacity > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        cache.capacity + 1U < static_cast<std::uint32_t>(window) ||
        cache.padded_capacity < cache.capacity || cache.head_dim != geometry.head_dim ||
        cache.num_kv_heads != geometry.kv_heads || cache.lane_capacity < 1) {
        reject("invalid cyclic cache geometry");
    }
    const auto padded = static_cast<std::int32_t>(cache.padded_capacity);
    require_tensor(cache.k, DType::BF16, geometry.head_dim, padded, geometry.kv_heads,
                   cache.lane_capacity, "cyclic cache K");
    require_tensor(cache.v, DType::FP16, geometry.head_dim, padded, geometry.kv_heads,
                   cache.lane_capacity, "cyclic cache V");

    require_vector_rows(cache.k);
    require_vector_rows(cache.v);
    detail::RelativeSinkRing ring;
    ring.k        = static_cast<__nv_bfloat16*>(cache.k.data);
    ring.v        = static_cast<__half*>(cache.v.data);
    ring.capacity = static_cast<std::int32_t>(cache.capacity);
    ring.window   = window;
    for (int axis = 1; axis < 4; ++axis) {
        ring.k_stride[axis - 1] = element_stride(cache.k, axis);
        ring.v_stride[axis - 1] = element_stride(cache.v, axis);
    }

    auto scope          = workspace.scope();
    const auto prepared = allocate_prepared(workspace, geometry, params.relative_dim,
                                            operands.tokens, operands.batch);
    bind_workspace(operands, prepared);
    detail::relative_sink_attention_launch(operands, ring, stream);
}

void relative_sink_attention_paged(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& r, const Tensor& sink, const Tensor& positions,
                                   const Tensor& kv_table_rows, const Tensor& q_scale,
                                   const Tensor& r_scale, const Tensor& rel_proj,
                                   AttentionHeadGeometry geometry,
                                   const RelativeSinkAttentionParams& params,
                                   const PagedKVBatchLayerView& cache, const Tensor* gate,
                                   WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    auto operands = validate_operands(q, k, v, r, sink, positions, kv_table_rows, q_scale, r_scale,
                                      rel_proj, geometry, params, gate, out);
    if (cache.storage != KvCacheStorage::BFloat16 || cache.head_dim != geometry.head_dim ||
        cache.num_kv_heads != geometry.kv_heads) {
        reject("paged cache must be BFloat16 storage of the call's KV geometry");
    }
    const auto require_plane = [&](const Tensor& plane, DType dtype, const char* name) {
        if (plane.data == nullptr || plane.dtype != dtype || plane.ne[0] != geometry.head_dim ||
            plane.ne[1] != kPagedKVPageSize || plane.ne[2] != geometry.kv_heads ||
            plane.ne[3] < 1 || plane.nb[0] != static_cast<std::int64_t>(dtype_size(dtype))) {
            reject(std::string("invalid page-major ") + name);
        }
    };
    require_plane(cache.k_pages, DType::BF16, "K plane");
    require_plane(cache.v_pages, DType::FP16, "V plane");
    if (cache.block_tables.data == nullptr || cache.block_tables.dtype != DType::I32 ||
        cache.block_tables.ne[0] < 1 || cache.block_tables.ne[1] < 1 ||
        cache.block_tables.ne[2] != 1 || cache.block_tables.ne[3] != 1) {
        reject("invalid block tables");
    }

    require_vector_rows(cache.k_pages);
    require_vector_rows(cache.v_pages);
    detail::RelativeSinkPages pages;
    pages.k           = static_cast<__nv_bfloat16*>(cache.k_pages.data);
    pages.v           = static_cast<__half*>(cache.v_pages.data);
    pages.block_table = static_cast<const std::int32_t*>(cache.block_tables.data);
    for (int axis = 1; axis < 4; ++axis) {
        pages.k_stride[axis - 1] = element_stride(cache.k_pages, axis);
        pages.v_stride[axis - 1] = element_stride(cache.v_pages, axis);
    }
    pages.table_stride[0] = element_stride(cache.block_tables, 0);
    pages.table_stride[1] = element_stride(cache.block_tables, 1);

    auto scope          = workspace.scope();
    const auto prepared = allocate_prepared(workspace, geometry, params.relative_dim,
                                            operands.tokens, operands.batch);
    bind_workspace(operands, prepared);
    detail::relative_sink_attention_launch(operands, pages, stream);
}

} // namespace ninfer::ops
