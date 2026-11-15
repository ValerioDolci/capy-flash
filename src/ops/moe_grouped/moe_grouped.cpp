// ninfer::ops - moe_grouped_nvfp4: validate the public contract, carve the workspace, launch the
// pieces.
#include "ninfer/ops/moe_grouped.h"

#include "ops/moe_grouped/launch.h"
#include "ops/common/validate.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr const char* kOp       = "moe_grouped_nvfp4";
constexpr std::int32_t kExperts = 512;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::invalid_argument(std::string(kOp) + ": " + message); }
}

void require_dense(const Tensor& t, DType dtype, const char* label) {
    require(t.dtype == dtype && t.data != nullptr && t.is_contiguous() && t.numel() > 0,
            std::string(label) + " must be a contiguous non-empty " +
                (dtype == DType::I32    ? "I32"
                 : dtype == DType::FP32 ? "FP32"
                                        : "BF16") +
                " tensor");
}

void require_columns(const Tensor& t, std::int32_t rows, std::int32_t tokens, const char* label) {
    require(t.ne[0] == rows && t.ne[1] == tokens && t.ne[2] == 1 && t.ne[3] == 1,
            std::string(label) + " must be [" + std::to_string(rows) + ",T]");
}

void require_plane(const MoeSlotPlane& plane, std::int64_t bytes, std::int64_t alignment,
                   const char* label) {
    const auto base = reinterpret_cast<std::uintptr_t>(plane.base);
    require(plane.base != nullptr, std::string(label) + " plane base must be non-null");
    require(base % static_cast<std::uintptr_t>(alignment) == 0,
            std::string(label) + " plane base must be " + std::to_string(alignment) +
                "-byte aligned");
    require(plane.stride == bytes,
            std::string(label) + " plane must be planar (stride = one expert's plane bytes)");
}

constexpr std::size_t align(std::size_t value) { return (value + 255) / 256 * 256; }

} // namespace

namespace detail {

MoeGroupedLayout moe_grouped_layout(std::int32_t columns) {
    const auto tokens             = static_cast<std::size_t>(columns);
    const std::size_t assignments = tokens * kMoeSlotsTopK;
    const auto jobs               = static_cast<std::size_t>(moe_grouped_max_jobs(columns));
    MoeGroupedLayout out{};
    std::size_t cursor = 0;
    auto take          = [&](std::size_t bytes) {
        const std::size_t at = cursor;
        cursor += align(bytes);
        return at;
    };
    out.input_codes   = take(tokens * kMoeSlotsHidden / 2);
    out.input_scales  = take(tokens * kMoeSlotsHidden / 16);
    out.counts        = take(static_cast<std::size_t>(kExperts) * 4);
    out.offsets       = take(static_cast<std::size_t>(kExperts + 1) * 4);
    out.rank          = take(assignments * 4);
    out.packed_token  = take(assignments * 4);
    out.packed_index  = take(assignments * 4);
    out.job_experts   = take(jobs * 4);
    out.job_columns   = take(jobs * 4);
    out.job_slots     = take(jobs * 4);
    out.job_count     = take(4);
    out.middle_codes  = take(assignments * kMoeSlotsIntermediate / 2);
    out.middle_scales = take(assignments * kMoeSlotsIntermediate / 16);
    out.grouped       = take(assignments * kMoeSlotsHidden * 4);
    out.total         = cursor;
    return out;
}

MoeGroupedWideLayout moe_grouped_wide_layout(std::int32_t columns) {
    const auto tokens             = static_cast<std::size_t>(columns);
    const std::size_t assignments = tokens * kMoeSlotsTopK;
    const auto jobs               = static_cast<std::size_t>(moe_grouped_max_jobs(columns));
    const auto piece_assignments =
        static_cast<std::size_t>(std::min(columns, kMoeGroupedChunkColumns)) * kMoeSlotsTopK;
    const auto piece_jobs =
        static_cast<std::size_t>(moe_grouped_max_jobs(std::min(columns, kMoeGroupedChunkColumns)));
    MoeGroupedWideLayout out{};
    std::size_t cursor = 0;
    auto take          = [&](std::size_t bytes) {
        const std::size_t at = cursor;
        cursor += align(bytes);
        return at;
    };
    out.input_codes       = take(tokens * kMoeSlotsHidden / 2);
    out.input_scales      = take(tokens * kMoeSlotsHidden / 16);
    out.counts            = take(static_cast<std::size_t>(kExperts) * 4);
    out.offsets           = take(static_cast<std::size_t>(kExperts + 1) * 4);
    out.rank              = take(assignments * 4);
    out.packed_token      = take(assignments * 4);
    out.packed_index      = take(assignments * 4);
    out.job_experts       = take(jobs * 4);
    out.job_columns       = take(jobs * 4);
    out.job_slots         = take(jobs * 4);
    out.job_count         = take(4);
    out.middle_codes      = take(assignments * kMoeSlotsIntermediate / 2);
    out.middle_scales     = take(assignments * kMoeSlotsIntermediate / 16);
    out.piece_counts      = take(static_cast<std::size_t>(kExperts) * 4);
    out.piece_offsets     = take(static_cast<std::size_t>(kExperts + 1) * 4);
    out.piece_rank        = take(piece_assignments * 4);
    out.piece_index       = take(piece_assignments * 4);
    out.piece_middle_row  = take(piece_assignments * 4);
    out.piece_job_experts = take(piece_jobs * 4);
    out.piece_job_columns = take(piece_jobs * 4);
    out.piece_job_slots   = take(piece_jobs * 4);
    out.piece_job_count   = take(4);
    out.grouped           = take(piece_assignments * kMoeSlotsHidden * 4);
    out.total             = cursor;
    return out;
}

} // namespace detail

std::size_t moe_grouped_nvfp4_workspace_bytes(std::int32_t max_tokens) {
    require(max_tokens >= 1, "max_tokens must be positive");
    // Wider calls group the whole call for gate/up: the wide layout of max_tokens covers
    // every narrower call (its arrays only grow with the width).
    const std::size_t piece =
        detail::moe_grouped_layout(std::min(max_tokens, kMoeGroupedChunkColumns)).total;
    const std::size_t wide = max_tokens > kMoeGroupedChunkColumns
                                 ? detail::moe_grouped_wide_layout(max_tokens).total
                                 : 0;
    return std::max(piece, wide) + 256;
}

void moe_grouped_nvfp4(const Tensor& x, const Tensor& routed_ids, const Tensor& routed_weights,
                       const MoeSlotsLayer& layer, const MoeSlotPool& pool, Tensor& out,
                       WorkspaceArena& workspace, cudaStream_t stream) {
    require_dense(x, DType::BF16, "x");
    const std::int32_t tokens = x.ne[1];
    require(tokens >= 1, "T must be positive");
    require_columns(x, kMoeSlotsHidden, tokens, "x");
    require_dense(routed_ids, DType::I32, "routed_ids");
    require_columns(routed_ids, kMoeSlotsTopK, tokens, "routed_ids");
    require_dense(routed_weights, DType::FP32, "routed_weights");
    require_columns(routed_weights, kMoeSlotsTopK, tokens, "routed_weights");

    require_dense(layer.slot_table, DType::I32, "slot_table");
    require(layer.slot_table.numel() == kExperts, "slot_table must be [512]");
    require_dense(layer.gate_up_divisors, DType::FP32, "gate_up_divisors");
    require(layer.gate_up_divisors.numel() == 2 * kExperts, "gate_up_divisors must be [1024]");
    require_dense(layer.down_divisors, DType::FP32, "down_divisors");
    require(layer.down_divisors.numel() == kExperts, "down_divisors must be [512]");
    for (const float d : {layer.gate_up_input_divisor, layer.down_input_divisor}) {
        require(std::isfinite(d) && d > 0.0F, "input divisors must be finite and positive");
    }

    require(pool.slots >= 1 &&
                static_cast<std::int64_t>(pool.slots) * kMoeSlotsHidden < (std::int64_t(1) << 31),
            "pool slots must be in [1, 2^31 / 2560)");
    require_plane(pool.gate_up_codes, kMoeSlotsGateUpCodeBytes, 16, "gate_up_codes");
    require_plane(pool.gate_up_scales, kMoeSlotsGateUpScaleBytes, 16, "gate_up_scales");
    require_plane(pool.down_codes, kMoeSlotsDownCodeBytes, 16, "down_codes");
    require_plane(pool.down_scales, kMoeSlotsDownScaleBytes, 16, "down_scales");

    require_dense(out, DType::FP32, "out");
    require_columns(out, kMoeSlotsHidden, tokens, "out");

    const std::int32_t chunk              = std::min(tokens, kMoeGroupedChunkColumns);
    const bool wide                       = tokens > kMoeGroupedChunkColumns;
    const detail::MoeGroupedLayout layout = detail::moe_grouped_layout(chunk);
    const std::size_t need = wide ? detail::moe_grouped_wide_layout(tokens).total : layout.total;
    require(workspace.base() != nullptr && workspace.capacity() >= need &&
                workspace.used() <= workspace.capacity() - need,
            "insufficient workspace capacity");
    for (const Tensor* t : {&x, &routed_ids, &routed_weights, &layer.slot_table,
                            &layer.gate_up_divisors, &layer.down_divisors}) {
        require(!detail::overlaps(out.data, out.bytes(), t->data, t->bytes()),
                "out must not overlap an input");
    }
    require(!detail::overlaps(out.data, out.bytes(), workspace.base(), workspace.capacity()),
            "out must not overlap the workspace");

    auto scope        = workspace.scope();
    auto* base        = static_cast<std::byte*>(workspace.alloc_bytes(need).data);
    const auto common = [&](detail::MoeGroupedPiece& c) {
        c.table                 = static_cast<const std::int32_t*>(layer.slot_table.data);
        c.slots                 = pool.slots;
        c.gate_up_divisors      = static_cast<const float*>(layer.gate_up_divisors.data);
        c.down_divisors         = static_cast<const float*>(layer.down_divisors.data);
        c.gate_up_input_divisor = layer.gate_up_input_divisor;
        c.down_input_divisor    = layer.down_input_divisor;
        c.planes[0]             = static_cast<const std::uint8_t*>(pool.gate_up_codes.base);
        c.planes[1]             = static_cast<const std::uint8_t*>(pool.gate_up_scales.base);
        c.planes[2]             = static_cast<const std::uint8_t*>(pool.down_codes.base);
        c.planes[3]             = static_cast<const std::uint8_t*>(pool.down_scales.base);
    };
    if (wide) {
        const detail::MoeGroupedWideLayout l = detail::moe_grouped_wide_layout(tokens);
        const auto at                        = [&](std::size_t offset) { return base + offset; };
        detail::MoeGroupedWide w{};
        detail::MoeGroupedPiece& c = w.call;
        c.x                        = static_cast<const __nv_bfloat16*>(x.data);
        c.ids                      = static_cast<const std::int32_t*>(routed_ids.data);
        c.weights                  = static_cast<const float*>(routed_weights.data);
        c.out                      = static_cast<float*>(out.data);
        c.columns                  = tokens;
        common(c);
        c.input_codes       = reinterpret_cast<std::uint8_t*>(at(l.input_codes));
        c.input_scales      = reinterpret_cast<std::uint8_t*>(at(l.input_scales));
        c.counts            = reinterpret_cast<std::int32_t*>(at(l.counts));
        c.offsets           = reinterpret_cast<std::int32_t*>(at(l.offsets));
        c.rank              = reinterpret_cast<std::int32_t*>(at(l.rank));
        c.packed_token      = reinterpret_cast<std::int32_t*>(at(l.packed_token));
        c.packed_index      = reinterpret_cast<std::int32_t*>(at(l.packed_index));
        c.job_experts       = reinterpret_cast<std::int32_t*>(at(l.job_experts));
        c.job_columns       = reinterpret_cast<std::int32_t*>(at(l.job_columns));
        c.job_slots         = reinterpret_cast<std::int32_t*>(at(l.job_slots));
        c.job_count         = reinterpret_cast<std::int32_t*>(at(l.job_count));
        c.middle_codes      = reinterpret_cast<std::uint8_t*>(at(l.middle_codes));
        c.middle_scales     = reinterpret_cast<std::uint8_t*>(at(l.middle_scales));
        c.grouped           = reinterpret_cast<float*>(at(l.grouped));
        w.piece_counts      = reinterpret_cast<std::int32_t*>(at(l.piece_counts));
        w.piece_offsets     = reinterpret_cast<std::int32_t*>(at(l.piece_offsets));
        w.piece_rank        = reinterpret_cast<std::int32_t*>(at(l.piece_rank));
        w.piece_index       = reinterpret_cast<std::int32_t*>(at(l.piece_index));
        w.piece_middle_row  = reinterpret_cast<std::int32_t*>(at(l.piece_middle_row));
        w.piece_job_experts = reinterpret_cast<std::int32_t*>(at(l.piece_job_experts));
        w.piece_job_columns = reinterpret_cast<std::int32_t*>(at(l.piece_job_columns));
        w.piece_job_slots   = reinterpret_cast<std::int32_t*>(at(l.piece_job_slots));
        w.piece_job_count   = reinterpret_cast<std::int32_t*>(at(l.piece_job_count));
        detail::moe_grouped_wide_launch(w, stream);
        return;
    }
    for (std::int32_t begin = 0; begin < tokens; begin += chunk) {
        const std::int64_t k0 = static_cast<std::int64_t>(begin) * kMoeSlotsTopK;
        const std::int64_t h0 = static_cast<std::int64_t>(begin) * kMoeSlotsHidden;
        detail::MoeGroupedPiece c{};
        c.x                     = static_cast<const __nv_bfloat16*>(x.data) + h0;
        c.ids                   = static_cast<const std::int32_t*>(routed_ids.data) + k0;
        c.weights               = static_cast<const float*>(routed_weights.data) + k0;
        c.out                   = static_cast<float*>(out.data) + h0;
        c.columns               = std::min(chunk, tokens - begin);
        c.table                 = static_cast<const std::int32_t*>(layer.slot_table.data);
        c.slots                 = pool.slots;
        c.gate_up_divisors      = static_cast<const float*>(layer.gate_up_divisors.data);
        c.down_divisors         = static_cast<const float*>(layer.down_divisors.data);
        c.gate_up_input_divisor = layer.gate_up_input_divisor;
        c.down_input_divisor    = layer.down_input_divisor;
        c.planes[0]             = static_cast<const std::uint8_t*>(pool.gate_up_codes.base);
        c.planes[1]             = static_cast<const std::uint8_t*>(pool.gate_up_scales.base);
        c.planes[2]             = static_cast<const std::uint8_t*>(pool.down_codes.base);
        c.planes[3]             = static_cast<const std::uint8_t*>(pool.down_scales.base);
        c.input_codes           = reinterpret_cast<std::uint8_t*>(base + layout.input_codes);
        c.input_scales          = reinterpret_cast<std::uint8_t*>(base + layout.input_scales);
        c.counts                = reinterpret_cast<std::int32_t*>(base + layout.counts);
        c.offsets               = reinterpret_cast<std::int32_t*>(base + layout.offsets);
        c.rank                  = reinterpret_cast<std::int32_t*>(base + layout.rank);
        c.packed_token          = reinterpret_cast<std::int32_t*>(base + layout.packed_token);
        c.packed_index          = reinterpret_cast<std::int32_t*>(base + layout.packed_index);
        c.job_experts           = reinterpret_cast<std::int32_t*>(base + layout.job_experts);
        c.job_columns           = reinterpret_cast<std::int32_t*>(base + layout.job_columns);
        c.job_slots             = reinterpret_cast<std::int32_t*>(base + layout.job_slots);
        c.job_count             = reinterpret_cast<std::int32_t*>(base + layout.job_count);
        c.middle_codes          = reinterpret_cast<std::uint8_t*>(base + layout.middle_codes);
        c.middle_scales         = reinterpret_cast<std::uint8_t*>(base + layout.middle_scales);
        c.grouped               = reinterpret_cast<float*>(base + layout.grouped);
        detail::moe_grouped_launch(c, stream);
    }
}

} // namespace ninfer::ops
