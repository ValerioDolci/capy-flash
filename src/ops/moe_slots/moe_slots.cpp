// ninfer::ops - moe_slots_nvfp4: validate the public contract, carve the workspace, launch.
#include "ninfer/ops/moe_slots.h"

#include "ops/moe_slots/launch.h"
#include "ops/common/validate.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr const char* kOp = "moe_slots_nvfp4";

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
    require(base % static_cast<std::uintptr_t>(alignment) == 0 && plane.stride % alignment == 0,
            std::string(label) + " plane base and stride must be " + std::to_string(alignment) +
                "-byte aligned");
    require(plane.stride >= bytes, std::string(label) + " plane stride must hold one expert");
}

} // namespace

std::size_t moe_slots_nvfp4_workspace_bytes(std::int32_t max_tokens) {
    require(max_tokens >= 1 && max_tokens <= kMoeSlotsMaxTokens, "max_tokens must be in [1,8]");
    std::size_t offsets[4];
    return detail::moe_slots_workspace_layout(max_tokens, offsets);
}

void moe_slots_nvfp4(const Tensor& x, const Tensor& routed_ids, const Tensor& routed_weights,
                     const MoeSlotsLayer& layer, const MoeSlotPool& pool, MoeSlotsOutput mode,
                     Tensor& out, WorkspaceArena& workspace, cudaStream_t stream) {
    require_dense(x, DType::BF16, "x");
    const std::int32_t tokens = x.ne[1];
    require(tokens >= 1 && tokens <= kMoeSlotsMaxTokens, "T must be in [1,8]");
    require_columns(x, kMoeSlotsHidden, tokens, "x");
    require_dense(routed_ids, DType::I32, "routed_ids");
    require_columns(routed_ids, kMoeSlotsTopK, tokens, "routed_ids");
    require_dense(routed_weights, DType::FP32, "routed_weights");
    require_columns(routed_weights, kMoeSlotsTopK, tokens, "routed_weights");

    require_dense(layer.slot_table, DType::I32, "slot_table");
    const std::int64_t experts = layer.slot_table.numel();
    require(experts <= (std::int64_t(1) << 30), "slot_table is too large");
    require_dense(layer.gate_up_divisors, DType::FP32, "gate_up_divisors");
    require(layer.gate_up_divisors.numel() == 2 * experts, "gate_up_divisors must be [2E]");
    require_dense(layer.down_divisors, DType::FP32, "down_divisors");
    require(layer.down_divisors.numel() == experts, "down_divisors must be [E]");
    for (const float d : {layer.gate_up_input_divisor, layer.down_input_divisor}) {
        require(std::isfinite(d) && d > 0.0F, "input divisors must be finite and positive");
    }

    require(pool.slots >= 1, "pool must hold at least one slot");
    require_plane(pool.gate_up_codes, kMoeSlotsGateUpCodeBytes, 16, "gate_up_codes");
    require_plane(pool.gate_up_scales, kMoeSlotsGateUpScaleBytes, 4, "gate_up_scales");
    require_plane(pool.down_codes, kMoeSlotsDownCodeBytes, 16, "down_codes");
    require_plane(pool.down_scales, kMoeSlotsDownScaleBytes, 4, "down_scales");

    require(out.dtype == DType::FP32 || out.dtype == DType::BF16, "out must be FP32 or BF16");
    require_dense(out, out.dtype, "out");
    require_columns(out, kMoeSlotsHidden, tokens, "out");
    require(mode == MoeSlotsOutput::Store || out.dtype == DType::FP32,
            "Accumulate requires an FP32 out");

    std::size_t offsets[4];
    const std::size_t required = detail::moe_slots_workspace_layout(tokens, offsets);
    require(workspace.base() != nullptr && workspace.capacity() >= required &&
                workspace.used() <= workspace.capacity() - required,
            "insufficient workspace capacity");
    for (const Tensor* t : {&x, &routed_ids, &routed_weights, &layer.slot_table,
                            &layer.gate_up_divisors, &layer.down_divisors}) {
        require(!detail::overlaps(out.data, out.bytes(), t->data, t->bytes()),
                "out must not overlap an input");
    }
    require(!detail::overlaps(out.data, out.bytes(), workspace.base(), workspace.capacity()),
            "out must not overlap the workspace");

    auto scope           = workspace.scope();
    const DeviceSpan raw = workspace.alloc_bytes(required);
    auto* base           = static_cast<std::byte*>(raw.data);
    const detail::MoeSlotsWorkspace views{
        reinterpret_cast<std::uint32_t*>(base + offsets[0]),
        reinterpret_cast<float*>(base + offsets[1]),
        reinterpret_cast<float*>(base + offsets[2]),
        reinterpret_cast<std::uint32_t*>(base + offsets[3]),
    };
    detail::moe_slots_launch(x, routed_ids, routed_weights, layer, pool, mode, out, views, stream);
}

} // namespace ninfer::ops
