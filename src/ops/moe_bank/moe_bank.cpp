// ninfer::ops - moe_bank_capped_swiglu / moe_partial_to_bf16: validate the public contract,
// resolve the bank's weight planes, carve workspace, launch.
#include "ninfer/ops/moe_bank.h"

#include "ops/moe_bank/launch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kMaxTopK = 64;

constexpr std::size_t align256(std::size_t value) { return (value + 255) / 256 * 256; }

void require(bool condition, const char* op, const std::string& message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

void require_dense(const Tensor& t, DType dtype, const char* op, const char* label) {
    require(t.dtype == dtype && t.data != nullptr && t.is_contiguous() && t.numel() > 0, op,
            std::string(label) + " must be contiguous and non-empty with the expected dtype");
}

bool matrix(const Tensor& t, std::int32_t rows, std::int32_t columns) {
    return t.ne[0] == rows && t.ne[1] == columns && t.ne[2] == 1 && t.ne[3] == 1;
}

// The plane of one bank view: one contiguous range of whole rows of a resident nvfp4
// (block_scale_k16_m128x4_v1) or bf16 (contiguous) parent.
detail::MoeBankPlane resolve_plane(const WeightView& view, const char* op, const char* label) {
    const std::string name(label);
    require(view.shape.size() == 2, op, name + " must be a matrix view");
    const WeightRegion region  = contiguous_weight_region(view);
    const WeightParent& parent = *region.parent;
    const WeightGeometry& g    = parent.geometry;
    require(parent.data != nullptr, op, name + " parent is not resident");
    require(g.shape.size() == 2 && g.shape[1] == view.shape[1] && region.begin % g.shape[1] == 0,
            op, name + " must be whole rows of a matrix parent");
    detail::MoeBankPlane plane;
    plane.data      = parent.data;
    plane.row_begin = static_cast<std::int64_t>(region.begin / g.shape[1]);
    require(view.shape[1] <= static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()),
            op, name + " has too many columns");
    plane.columns = static_cast<std::int32_t>(view.shape[1]);
    if (g.format == QType::NVFP4 && g.layout == QuantLayout::BlockScaleK16M128x4) {
        plane.nvfp4            = true;
        plane.scale_offset     = g.scale_offset;
        plane.divisor_offset   = g.divisor_offset;
        plane.rows_per_divisor = static_cast<std::int64_t>(g.rows_per_divisor());
    } else if (g.format == QType::MXFP8_E4M3FN_G32_UE8M0 &&
               g.layout == QuantLayout::BlockScaleK32) {
        plane.mxfp8        = true;
        plane.scale_offset = g.scale_offset;
    } else {
        require(g.format == QType::BF16 && g.layout == QuantLayout::Contiguous, op,
                name + " must be nvfp4 block_scale_k16_m128x4_v1, mxfp8_e4m3fn_g32_ue8m0 "
                       "block_scale_k32_v1 or contiguous bf16");
    }
    return plane;
}

// Bank experts the four-argument workspace size plans for in one pass (more run in passes).
constexpr std::size_t kPlanExperts = 4096;

// The wide route's plan of one pass of `tokens` tokens: offsets [E' + 1] and sorted columns.
std::size_t plan_bytes(std::size_t experts, std::size_t columns) {
    return align256((experts + 1 + columns) * sizeof(std::int32_t));
}

} // namespace

std::size_t moe_bank_capped_swiglu_workspace_bytes(std::int32_t top_k, std::int32_t intermediate,
                                                   std::int32_t max_tokens) {
    constexpr const char* op = "moe_bank_capped_swiglu_workspace_bytes";
    require(top_k >= 1 && top_k <= kMaxTopK && intermediate >= 1 && max_tokens >= 1, op,
            "k, F and T must be positive (k <= 64)");
    return static_cast<std::size_t>(max_tokens) * static_cast<std::size_t>(top_k) *
           static_cast<std::size_t>(intermediate) * sizeof(std::uint16_t);
}

std::size_t moe_bank_capped_swiglu_workspace_bytes(std::int32_t top_k, std::int32_t intermediate,
                                                   std::int32_t hidden, std::int32_t max_tokens) {
    constexpr const char* op = "moe_bank_capped_swiglu_workspace_bytes";
    require(hidden >= 1, op, "H must be positive");
    const std::size_t h_bytes =
        moe_bank_capped_swiglu_workspace_bytes(top_k, intermediate, max_tokens);
    const std::size_t o_bytes = static_cast<std::size_t>(max_tokens) *
                                static_cast<std::size_t>(top_k) * static_cast<std::size_t>(hidden) *
                                sizeof(std::uint16_t);
    return align256(h_bytes) + align256(o_bytes) +
           plan_bytes(kPlanExperts,
                      static_cast<std::size_t>(max_tokens) * static_cast<std::size_t>(top_k));
}

void moe_bank_capped_swiglu(const Tensor& x, const Tensor& ids, const Tensor& weights,
                            const MoeBank& bank, CappedSwiglu activation, MoeBankOutput mode,
                            Tensor& partial, WorkspaceArena& workspace, cudaStream_t stream) {
    constexpr const char* op = "moe_bank_capped_swiglu";
    require_dense(x, DType::BF16, op, "x");
    require_dense(ids, DType::I32, op, "ids");
    require_dense(weights, DType::FP32, op, "weights");
    require_dense(partial, DType::FP32, op, "partial");
    const std::int32_t hidden = x.ne[0], tokens = x.ne[1], top_k = ids.ne[0];
    require(matrix(x, hidden, tokens), op, "x must be [H,T]");
    require(top_k >= 1 && top_k <= kMaxTopK, op, "k outside [1, 64]");
    require(matrix(ids, top_k, tokens) && matrix(weights, top_k, tokens), op,
            "ids and weights must be [k,T]");
    require(matrix(partial, hidden, tokens), op, "partial must be [H,T]");
    require(reinterpret_cast<std::uintptr_t>(x.data) % 16 == 0, op, "x must be 16-byte aligned");
    require(bank.first_expert >= 0, op, "first_expert must be non-negative");
    require(std::isfinite(activation.gate_cap) && activation.gate_cap > 0.0F &&
                std::isfinite(activation.up_cap) && activation.up_cap > 0.0F,
            op, "caps must be finite and positive");

    const auto gate_up = resolve_plane(bank.gate_up, op, "gate_up");
    const auto down    = resolve_plane(bank.down, op, "down");
    require(gate_up.nvfp4 == down.nvfp4 && gate_up.mxfp8 == down.mxfp8, op,
            "gate_up and down must share one format");
    const std::int32_t intermediate = down.columns;
    require(gate_up.columns == hidden, op, "gate_up must have H columns");
    require(hidden % 64 == 0 && intermediate % 64 == 0, op, "H and F must be multiples of 64");
    const std::uint64_t down_rows = bank.down.shape[0];
    require(down_rows % static_cast<std::uint64_t>(hidden) == 0, op, "down rows must be E' * H");
    const std::uint64_t experts = down_rows / static_cast<std::uint64_t>(hidden);
    require(experts >= 1 &&
                experts <= static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max() -
                                                      bank.first_expert),
            op, "E' outside the id range");
    require(bank.gate_up.shape[0] == experts * 2 * static_cast<std::uint64_t>(intermediate), op,
            "gate_up rows must be E' * 2F");

    detail::MoeBankShape shape;
    shape.hidden       = hidden;
    shape.intermediate = intermediate;
    shape.experts      = static_cast<std::int32_t>(experts);
    shape.first_expert = bank.first_expert;
    shape.top_k        = top_k;
    shape.tokens       = tokens;

    auto scope = workspace.scope();
    detail::MoeBankWorkspace work;
    work.chunk_tokens = tokens;
    if (detail::moe_bank_wide_route(shape)) {
        // Tokens per pass: the most whose h and o fit the caller's workspace (all of them with
        // the four-argument size; fewer passes read the weights fewer times, same result).
        const std::size_t per_token =
            static_cast<std::size_t>(top_k) *
            (static_cast<std::size_t>(intermediate + hidden) * sizeof(std::uint16_t) +
             sizeof(std::int32_t));
        const std::size_t start = align256(workspace.used());
        const std::size_t room  = workspace.capacity() > start ? workspace.capacity() - start : 0;
        const auto h_bytes      = [&](std::size_t t) {
            return moe_bank_capped_swiglu_workspace_bytes(top_k, intermediate,
                                                               static_cast<std::int32_t>(t));
        };
        const auto o_bytes = [&](std::size_t t) {
            return t * static_cast<std::size_t>(top_k) * hidden * sizeof(std::uint16_t);
        };
        const auto need = [&](std::size_t t) {
            return align256(h_bytes(t)) + align256(o_bytes(t)) +
                   plan_bytes(experts, t * static_cast<std::size_t>(top_k));
        };
        std::size_t chunk =
            std::min<std::size_t>(static_cast<std::size_t>(tokens), room / per_token);
        while (chunk >= 1 && need(chunk) > room) { --chunk; }
        require(chunk >= 1, op, "workspace below moe_bank_capped_swiglu_workspace_bytes(k, F, T)");
        work.chunk_tokens = static_cast<int>(chunk);
        work.hidden       = static_cast<std::uint16_t*>(workspace.alloc_bytes(h_bytes(chunk)).data);
        work.out          = static_cast<std::uint16_t*>(workspace.alloc_bytes(o_bytes(chunk)).data);
        work.plan         = static_cast<std::int32_t*>(
            workspace.alloc_bytes(plan_bytes(experts, chunk * static_cast<std::size_t>(top_k)))
                .data);
    } else {
        work.hidden = static_cast<std::uint16_t*>(
            workspace
                .alloc_bytes(moe_bank_capped_swiglu_workspace_bytes(top_k, intermediate, tokens))
                .data);
    }
    detail::moe_bank_capped_swiglu_launch(x, ids, weights, gate_up, down, shape, activation,
                                          mode == MoeBankOutput::Accumulate, partial, work, stream);
}

void moe_partial_to_bf16(const Tensor& partial, Tensor& y, cudaStream_t stream) {
    constexpr const char* op = "moe_partial_to_bf16";
    require_dense(partial, DType::FP32, op, "partial");
    require_dense(y, DType::BF16, op, "y");
    require(partial.ne[0] == y.ne[0] && partial.ne[1] == y.ne[1] && partial.ne[2] == y.ne[2] &&
                partial.ne[3] == y.ne[3],
            op, "partial and y must have the same shape");
    detail::moe_partial_to_bf16_launch(partial, y, stream);
}

} // namespace ninfer::ops
