// ninfer::ops - moe_slots_gguf / moe_grouped_gguf: validate the public contract, carve the
// workspace, launch.
#include "ninfer/ops/moe_slots_gguf.h"

#include "ops/moe_slots_gguf/launch.h"
#include "ops/common/validate.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kGateUpRows = 2 * kMoeSlotsIntermediate; // 1280
constexpr std::int32_t kDownRows   = kMoeSlotsHidden;           // 2560

void require(bool condition, const char* op, const std::string& message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

void require_dense(const Tensor& t, DType dtype, const char* op, const char* label) {
    require(t.dtype == dtype && t.data != nullptr && t.is_contiguous() && t.numel() > 0, op,
            std::string(label) + " must be a contiguous non-empty " +
                (dtype == DType::I32    ? "I32"
                 : dtype == DType::FP32 ? "FP32"
                                        : "BF16") +
                " tensor");
}

void require_columns(const Tensor& t, std::int32_t rows, std::int32_t tokens, const char* op,
                     const char* label) {
    require(t.ne[0] == rows && t.ne[1] == tokens && t.ne[2] == 1 && t.ne[3] == 1, op,
            std::string(label) + " must be [" + std::to_string(rows) + ",T]");
}

bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

// The shared part of both Ops' contracts: inputs, layer, formats; returns T.
std::int32_t validate_inputs(const Tensor& x, const Tensor& routed_ids,
                             const Tensor& routed_weights, const MoeGgufLayer& layer,
                             const char* op) {
    require_dense(x, DType::BF16, op, "x");
    const std::int32_t tokens = x.ne[1];
    require(tokens >= 1, op, "T must be positive");
    require_columns(x, kMoeSlotsHidden, tokens, op, "x");
    require(aligned16(x.data), op, "x must be 16-byte aligned");
    require_dense(routed_ids, DType::I32, op, "routed_ids");
    require_columns(routed_ids, kMoeSlotsTopK, tokens, op, "routed_ids");
    require_dense(routed_weights, DType::FP32, op, "routed_weights");
    require_columns(routed_weights, kMoeSlotsTopK, tokens, op, "routed_weights");

    require_dense(layer.slot_table, DType::I32, op, "slot_table");
    require(layer.slot_table.numel() <= (std::int64_t(1) << 30), op, "slot_table is too large");
    require(layer.slots >= 0, op, "slots must be non-negative");
    require(moe_gguf_gate_up_supported(layer.gate_up_format), op, "unsupported gate/up format");
    require(moe_gguf_down_supported(layer.down_format), op, "unsupported down format");
    if (layer.slot_offset != nullptr) {
        require(layer.base != nullptr && aligned16(layer.base), op,
                "blob form needs a 16-byte aligned base");
        require(reinterpret_cast<std::uintptr_t>(layer.slot_offset) % 8 == 0, op,
                "slot_offset must be 8-byte aligned");
        require(layer.host_table == nullptr ||
                    (layer.gate_up_base != nullptr && layer.down_base != nullptr &&
                     aligned16(layer.gate_up_base) && aligned16(layer.down_base)),
                op, "host_table needs 16-byte aligned gate_up_base and down_base");
    } else {
        require(layer.gate_up_base != nullptr && layer.down_base != nullptr &&
                    aligned16(layer.gate_up_base) && aligned16(layer.down_base),
                op, "planar form needs 16-byte aligned gate_up_base and down_base");
    }
    return tokens;
}

detail::GgufExperts experts_of(const MoeGgufLayer& layer) {
    detail::GgufExperts e;
    e.table          = static_cast<const std::int32_t*>(layer.slot_table.data);
    e.experts        = static_cast<std::int32_t>(layer.slot_table.numel());
    e.slots          = layer.slots;
    e.slot_offset    = layer.slot_offset;
    e.base           = reinterpret_cast<const std::uint8_t*>(layer.base);
    e.gate_up_base   = reinterpret_cast<const std::uint8_t*>(layer.gate_up_base);
    e.down_base      = reinterpret_cast<const std::uint8_t*>(layer.down_base);
    e.host_table     = layer.host_table;
    e.gate_up_bytes  = kGateUpRows * moe_gguf_row_bytes(layer.gate_up_format, kMoeSlotsHidden);
    e.down_bytes     = kDownRows * moe_gguf_row_bytes(layer.down_format, kMoeSlotsIntermediate);
    e.gate_up_format = layer.gate_up_format;
    e.down_format    = layer.down_format;
    return e;
}

void require_workspace(WorkspaceArena& workspace, std::size_t need, const char* op) {
    require(workspace.base() != nullptr && workspace.capacity() >= need &&
                workspace.used() <= workspace.capacity() - need,
            op, "insufficient workspace capacity");
}

void require_no_overlap(const Tensor& out, const Tensor& x, const Tensor& ids, const Tensor& w,
                        const MoeGgufLayer& layer, WorkspaceArena& workspace, const char* op) {
    for (const Tensor* t : {&x, &ids, &w, &layer.slot_table}) {
        require(!detail::overlaps(out.data, out.bytes(), t->data, t->bytes()), op,
                "out must not overlap an input");
    }
    require(!detail::overlaps(out.data, out.bytes(), workspace.base(), workspace.capacity()), op,
            "out must not overlap the workspace");
}

} // namespace

std::size_t moe_slots_gguf_workspace_bytes(std::int32_t max_tokens) {
    require(max_tokens >= 1 && max_tokens <= kMoeSlotsMaxTokens, "moe_slots_gguf",
            "max_tokens must be in [1,8]");
    std::size_t offsets[5];
    return detail::gguf_decode_workspace_layout(max_tokens, offsets);
}

void moe_slots_gguf(const Tensor& x, const Tensor& routed_ids, const Tensor& routed_weights,
                    const MoeGgufLayer& layer, MoeSlotsOutput mode, Tensor& out,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    constexpr const char* kOp = "moe_slots_gguf";
    const std::int32_t tokens = validate_inputs(x, routed_ids, routed_weights, layer, kOp);
    require(tokens <= kMoeSlotsMaxTokens, kOp, "T must be in [1,8]");
    require(out.dtype == DType::FP32 || out.dtype == DType::BF16, kOp, "out must be FP32 or BF16");
    require_dense(out, out.dtype, kOp, "out");
    require_columns(out, kMoeSlotsHidden, tokens, kOp, "out");
    require(aligned16(out.data), kOp, "out must be 16-byte aligned");
    require(mode == MoeSlotsOutput::Store || out.dtype == DType::FP32, kOp,
            "Accumulate requires an FP32 out");

    std::size_t offsets[5];
    const std::size_t need = detail::gguf_decode_workspace_layout(tokens, offsets);
    require_workspace(workspace, need, kOp);
    require_no_overlap(out, x, routed_ids, routed_weights, layer, workspace, kOp);

    auto scope = workspace.scope();
    auto* base = static_cast<std::byte*>(workspace.alloc_bytes(need).data);
    const detail::GgufDecodeWorkspace ws{
        reinterpret_cast<detail::GgufDecodeWork*>(base + offsets[0]),
        reinterpret_cast<std::int8_t*>(base + offsets[1]),
        reinterpret_cast<float*>(base + offsets[2]),
        reinterpret_cast<float*>(base + offsets[3]),
        reinterpret_cast<float*>(base + offsets[4]),
    };
    detail::GgufDecodeCall call;
    call.x          = static_cast<const __nv_bfloat16*>(x.data);
    call.ids        = static_cast<const std::int32_t*>(routed_ids.data);
    call.weights    = static_cast<const float*>(routed_weights.data);
    call.tokens     = tokens;
    call.experts    = experts_of(layer);
    call.accumulate = mode == MoeSlotsOutput::Accumulate;
    call.out_f32    = out.dtype == DType::FP32 ? static_cast<float*>(out.data) : nullptr;
    call.out_bf16   = out.dtype == DType::BF16 ? static_cast<__nv_bfloat16*>(out.data) : nullptr;
    detail::gguf_decode_launch(call, ws, stream);
}

std::size_t moe_grouped_gguf_workspace_bytes(std::int32_t max_tokens) {
    require(max_tokens >= 1, "moe_grouped_gguf", "max_tokens must be positive");
    return detail::gguf_grouped_layout(std::min(max_tokens, detail::kGgufGroupedChunkColumns))
        .total;
}

void moe_grouped_gguf(const Tensor& x, const Tensor& routed_ids, const Tensor& routed_weights,
                      const MoeGgufLayer& layer, Tensor& out, WorkspaceArena& workspace,
                      cudaStream_t stream) {
    constexpr const char* kOp = "moe_grouped_gguf";
    const std::int32_t tokens = validate_inputs(x, routed_ids, routed_weights, layer, kOp);
    require(layer.slot_table.numel() <= detail::kGgufGroupedMaxExperts, kOp,
            "slot_table must have at most 1024 experts");
    require_dense(out, DType::FP32, kOp, "out");
    require_columns(out, kMoeSlotsHidden, tokens, kOp, "out");
    require(aligned16(out.data), kOp, "out must be 16-byte aligned");

    const std::int32_t chunk          = std::min(tokens, detail::kGgufGroupedChunkColumns);
    const detail::GgufGroupedLayout l = detail::gguf_grouped_layout(chunk);
    require_workspace(workspace, l.total, kOp);
    require_no_overlap(out, x, routed_ids, routed_weights, layer, workspace, kOp);

    auto scope = workspace.scope();
    auto* base = static_cast<std::byte*>(workspace.alloc_bytes(l.total).data);
    const detail::GgufExperts experts = experts_of(layer);
    for (std::int32_t begin = 0; begin < tokens; begin += chunk) {
        const std::int64_t k0 = static_cast<std::int64_t>(begin) * kMoeSlotsTopK;
        const std::int64_t h0 = static_cast<std::int64_t>(begin) * kMoeSlotsHidden;
        detail::GgufGroupedPiece c;
        c.x              = static_cast<const __nv_bfloat16*>(x.data) + h0;
        c.ids            = static_cast<const std::int32_t*>(routed_ids.data) + k0;
        c.weights        = static_cast<const float*>(routed_weights.data) + k0;
        c.out            = static_cast<float*>(out.data) + h0;
        c.columns        = std::min(chunk, tokens - begin);
        c.experts        = experts;
        c.counts         = reinterpret_cast<std::int32_t*>(base + l.counts);
        c.offsets        = reinterpret_cast<std::int32_t*>(base + l.offsets);
        c.rank           = reinterpret_cast<std::int32_t*>(base + l.rank);
        c.packed_token   = reinterpret_cast<std::int32_t*>(base + l.packed_token);
        c.packed_index   = reinterpret_cast<std::int32_t*>(base + l.packed_index);
        c.tiles          = reinterpret_cast<int4*>(base + l.tiles);
        c.tile_count     = reinterpret_cast<std::int32_t*>(base + l.tile_count);
        c.expert_gate_up = reinterpret_cast<const std::uint8_t**>(base + l.expert_gate_up);
        c.expert_down    = reinterpret_cast<const std::uint8_t**>(base + l.expert_down);
        c.h              = reinterpret_cast<__nv_bfloat16*>(base + l.h);
        c.grouped        = reinterpret_cast<float*>(base + l.grouped);
        detail::gguf_grouped_launch(c, stream);
    }
}

} // namespace ninfer::ops
