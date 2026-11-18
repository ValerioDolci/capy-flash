#include "core/weight.h"
#include "ops/linear/q3/q3_arena.h"
#include "ops/linear/q3/q3_linear.h"
#include "ninfer/ops/attn_input_proj.h"

#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"
#include "ops/attn_input_proj/q8/q8_attn_input_plan.h"
#include "ops/linear/fp8/fp8_geometry.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/nvfp4/nvfp4_layout.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ninfer/ops/tp2/attn_input_proj.h"
#include "ops/attn_input_proj/tp2/fp8_attn_input_shard.h"
#include "ops/attn_input_proj/tp2/nvfp4_attn_input_shard.h"
#include "ops/common/split_launch.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t cols, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != cols ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void require_rowsplit(const Weight& weight, QType qtype, std::int32_t rows, const char* label) {
    const bool q4_planes =
        qtype != QType::Q4_G64_FP16 || (weight.qhigh == nullptr && weight.high_plane_bytes == 0);
    const bool q5_planes =
        qtype != QType::Q5_G64_FP16 || (weight.qhigh != nullptr && weight.high_plane_bytes != 0);
    if (weight.qtype != qtype || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 64 || weight.group != 64 ||
        weight.ndim != 2 || weight.n != rows || weight.k != 5120 || weight.shape[0] != rows ||
        weight.shape[1] != 5120 || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != 5120 || !q4_planes || !q5_planes ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 4) ||
        (qtype == QType::Q5_G64_FP16 && !aligned_to(weight.qhigh, 16))) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void require_q8_rowsplit(const Weight& weight, std::int32_t rows, std::int32_t hidden,
                         const char* label) {
    if (weight.qtype != QType::Q8_G32_FP16 || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 32 || weight.group != 32 ||
        weight.ndim != 2 || weight.n != rows || weight.k != hidden || weight.shape[0] != rows ||
        weight.shape[1] != hidden || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != hidden || weight.qhigh != nullptr ||
        weight.high_plane_bytes != 0 || !aligned_to(weight.qdata, 16) ||
        !aligned_to(weight.scales, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void require_bf16_contiguous(const Weight& weight, std::int32_t rows, std::int32_t hidden,
                             const char* label) {
    const std::uint64_t payload_bytes = static_cast<std::uint64_t>(rows) *
                                        static_cast<std::uint64_t>(hidden) * sizeof(std::uint16_t);
    if (weight.qtype != QType::BF16 || weight.layout != QuantLayout::Contiguous ||
        weight.payload_bytes < payload_bytes || weight.high_plane_bytes != 0 || weight.ndim != 2 ||
        weight.n != rows || weight.k != hidden || weight.shape[0] != rows ||
        weight.shape[1] != hidden || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != hidden || weight.qhigh != nullptr || weight.scales != nullptr ||
        weight.group_size != 0 || weight.group != 0 || !aligned_to(weight.qdata, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
    case LinearPolicy::AllowA4Decode:
        return;
    }
    throw std::invalid_argument("attn_input_proj: invalid compute policy");
}

// The registered single-parent problem: the q|k|gate|v parent [14336,5120] of 24 query / 4 KV
// heads of 256 over hidden 5120, in Q3, BF16, NVFP4 and FP8.
constexpr std::int32_t kParentQueryRows = 6144;
constexpr std::int32_t kParentKeyRows   = 1024;
constexpr std::int32_t kParentRows      = detail::Nvfp4N14336K5120::kOutputRows;
constexpr std::int32_t kParentHidden    = detail::Nvfp4N14336K5120::kInputRows;
static_assert(2 * (kParentQueryRows + kParentKeyRows) == kParentRows &&
              detail::Fp8N14336K5120::kOutputRows == kParentRows &&
              detail::Fp8N14336K5120::kInputRows == kParentHidden);

bool parent_problem(QType qtype, std::int32_t parent_rows, std::int32_t input_rows) {
    switch (qtype) {
    case QType::Q3_G128_FP16:
    case QType::BF16:
    case QType::NVFP4:
    case QType::FP8_E4M3FN_ROW_BF16:
        return parent_rows == kParentRows && input_rows == kParentHidden;
    default:
        return false;
    }
}

void dispatch_single_parent(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                            Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                            cudaStream_t stream) {
    validate_policy(policy);
    if (weight.qtype == QType::BF16) {
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        require_bf16_contiguous(weight, kRows, kHidden, "query/key/gate/value weight");
        detail::bf16_attn_input_dispatch(x, weight, q, gate, k, v, stream);
        return;
    }

    if (weight.qtype == QType::Q3_G128_FP16) {
        // One Q|K|Gate|V parent [14336,5120], rows split into the four outputs.
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        detail::validate_q3_weight(weight, "q3 attn_input_proj");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("q3 attn_input_proj: unsupported weight shape");
        }
        detail::Q3RowSegments segments;
        segments.count             = 4;
        Tensor* outputs[4]         = {&q, &k, &gate, &v};
        const std::int32_t rows[4] = {kQRows, kKvRows, kQRows, kKvRows};
        std::int32_t end           = 0;
        for (int s = 0; s < 4; ++s) {
            end += rows[s];
            segments.data[s] = static_cast<__nv_bfloat16*>(outputs[s]->data);
            segments.ld[s]   = rows[s];
            segments.end[s]  = end;
        }
        const detail::Q3ArenaWorkspace q3ws(workspace, weight.k, cols);
        detail::q3_linear_segments_dispatch(x, weight, segments, stream, q3ws.get());
        return;
    }

    if (weight.qtype == QType::NVFP4) {
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        detail::validate_nvfp4_weight(weight, "nvfp4 attn_input_proj");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("nvfp4 attn_input_proj: unsupported weight shape");
        }
        detail::nvfp4_attn_input_dispatch(x, weight, q, gate, k, v, policy, workspace, stream);
        return;
    }

    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        detail::validate_fp8_weight(weight, "fp8 attn_input_proj");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("fp8 attn_input_proj: unsupported weight shape");
        }
        detail::fp8_attn_input_dispatch(x, weight, q, gate, k, v, policy, workspace, stream);
        return;
    }

    throw std::invalid_argument("attn_input_proj: unsupported query/key/gate/value weight format");
}

} // namespace

bool attn_input_proj_registered(QType parent_qtype, std::int32_t query_rows, std::int32_t key_rows,
                                std::int32_t input_rows) {
    return query_rows == kParentQueryRows && key_rows == kParentKeyRows &&
           parent_problem(parent_qtype, 2 * (query_rows + key_rows), input_rows);
}

std::size_t attn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                                     std::int32_t input_rows, LinearPolicy policy,
                                                     std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("attn_input_proj workspace: invalid token interval");
    }

    switch (parent_qtype) {
    case QType::Q3_G128_FP16:
        if (!parent_problem(parent_qtype, parent_rows, input_rows)) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported Q3 profile");
        }
        return detail::q3_workspace_bytes(input_rows, max_tokens);
    case QType::BF16:
        if (!parent_problem(parent_qtype, parent_rows, input_rows)) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported BF16 profile");
        }
        return 0;
    case QType::NVFP4:
        if (!parent_problem(parent_qtype, parent_rows, input_rows)) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_attn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    case QType::FP8_E4M3FN_ROW_BF16:
        if (!parent_problem(parent_qtype, parent_rows, input_rows)) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported FP8 profile");
        }
        return detail::fp8_attn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    case QType::Q8_G32_FP16:
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
    case QType::FP32:
    case QType::INT32:
        break;
    }
    throw std::invalid_argument("attn_input_proj workspace: unsupported parent qtype");
}

void attn_input_proj(const Tensor& x, const Weight& query_key_weight,
                     const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                     cudaStream_t stream) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQRows  = 6144;
    constexpr std::int32_t kKvRows = 1024;
    const std::int32_t cols        = x.ne[1];
    require_matrix(x, kHidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(gate, kQRows, cols, "gate");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    require_rowsplit(query_key_weight, QType::Q4_G64_FP16, kQRows + kKvRows, "query/key weight");
    require_rowsplit(gate_value_weight, QType::Q5_G64_FP16, kQRows + kKvRows, "gate/value weight");

    detail::q4_q5_attn_input_dispatch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                      stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy,
                     WorkspaceArena& workspace, cudaStream_t stream) {
    dispatch_single_parent(x, query_key_gate_value_weight, q, gate, k, v, policy, &workspace,
                           stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    dispatch_single_parent(x, query_key_gate_value_weight, q, gate, k, v, LinearPolicy::A16Only,
                           nullptr, stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_value_weight, Tensor& q, Tensor& k,
                     Tensor& v, cudaStream_t stream) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQRows  = 4096;
    constexpr std::int32_t kKvRows = 1024;
    constexpr std::int32_t kRows   = 6144;
    const std::int32_t cols        = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
    require_matrix(x, kHidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    require_q8_rowsplit(query_key_value_weight, kRows, kHidden, "query/key/value weight");

    detail::q8_attn_input_dispatch(x, query_key_value_weight, q, k, v, stream);
}

// Two-device (tp 2) forms of attn_input_proj, declared in ninfer/ops/tp2/attn_input_proj.h.

namespace {

// One rank of the column-parallel form: an FP8 or NVFP4 [7168,5120] head-local shard of the fused
// parent.
static_assert(detail::Fp8N7168K5120::kOutputRows == detail::Nvfp4N7168K5120::kOutputRows &&
              detail::Fp8N7168K5120::kInputRows == detail::Nvfp4N7168K5120::kInputRows);
constexpr std::int32_t kHidden = detail::Fp8N7168K5120::kInputRows;
constexpr std::int32_t kQRows  = 3072;
constexpr std::int32_t kKvRows = 512;
constexpr std::int32_t kRows   = detail::Fp8N7168K5120::kOutputRows;
static_assert(2 * (kQRows + kKvRows) == kRows);

void validate_column_parallel_rank(const Tensor& x, const Weight& weight, const Tensor& q,
                                   const Tensor& gate, const Tensor& k, const Tensor& v,
                                   LinearPolicy policy) {
    validate_policy(policy);
    const std::int32_t cols = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
    require_matrix(x, kHidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(gate, kQRows, cols, "gate");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    if (weight.qtype == QType::Q3_G128_FP16) {
        detail::validate_q3_weight(weight, "q3 attn_input_proj column-parallel");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument(
                "q3 attn_input_proj column-parallel: unsupported shard shape");
        }
        return;
    }
    if (weight.qtype == QType::NVFP4) {
        detail::validate_nvfp4_weight(weight, "nvfp4 attn_input_proj column-parallel");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument(
                "nvfp4 attn_input_proj column-parallel: unsupported shard shape");
        }
        return;
    }
    if (weight.qtype != QType::FP8_E4M3FN_ROW_BF16) {
        throw std::invalid_argument("attn_input_proj column-parallel: only FP8_E4M3FN_ROW_BF16 and "
                                    "NVFP4 shards are registered");
    }
    detail::validate_fp8_weight(weight, "fp8 attn_input_proj column-parallel");
    if (weight.n != kRows || weight.k != kHidden) {
        throw std::invalid_argument("fp8 attn_input_proj column-parallel: unsupported shard shape");
    }
}

} // namespace

bool attn_input_proj_column_parallel_registered(QType shard_qtype, std::int32_t query_rows,
                                                std::int32_t key_rows, std::int32_t input_rows) {
    return (shard_qtype == QType::FP8_E4M3FN_ROW_BF16 || shard_qtype == QType::NVFP4) &&
           query_rows == kQRows && key_rows == kKvRows && input_rows == kHidden;
}

std::size_t attn_input_proj_column_parallel_workspace_capacity_bytes(QType shard_qtype,
                                                                     LinearPolicy policy,
                                                                     std::int32_t min_tokens,
                                                                     std::int32_t max_tokens) {
    validate_policy(policy);
    // The shard keeps the parent's input rows; the FP8 workspace also holds its split-K partials.
    switch (shard_qtype) {
    case QType::Q3_G128_FP16:
        return detail::q3_workspace_bytes(5120, max_tokens); // the shard keeps K = 5120
    case QType::FP8_E4M3FN_ROW_BF16:
        return detail::fp8_attn_input_shard_workspace_capacity_bytes(policy, min_tokens,
                                                                     max_tokens);
    case QType::NVFP4:
        return detail::nvfp4_attn_input_shard_workspace_capacity_bytes(policy, min_tokens,
                                                                       max_tokens);
    case QType::BF16:
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
    case QType::Q8_G32_FP16:
    case QType::FP32:
    case QType::INT32:
        break;
    }
    throw std::invalid_argument(
        "attn_input_proj column-parallel workspace: unsupported shard qtype");
}

void attn_input_proj_column_parallel(
    const std::array<Tensor, 2>& x, const std::array<Weight, 2>& query_key_gate_value_weight,
    const std::array<Tensor, 2>& q, const std::array<Tensor, 2>& gate,
    const std::array<Tensor, 2>& k, const std::array<Tensor, 2>& v, LinearPolicy policy,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& ec) {
    const std::array<Weight, 2>& w = query_key_gate_value_weight;
    constexpr const char* kOp      = "attn_input_proj column-parallel";
    detail::require_split_pair(ec, x, w, detail::SplitAxis::Output, kOp);
    // Validate both ranks before issuing either, so a rejected pair enqueues nothing.
    for (std::size_t rank = 0; rank < 2; ++rank) {
        validate_column_parallel_rank(x[rank], w[rank], q[rank], gate[rank], k[rank], v[rank],
                                      policy);
        constexpr const char* kResidency =
            "attn_input_proj split: rank arguments must reside on its device";
        detail::require_rank_residency(ec, static_cast<int>(rank), x[rank].data, w[rank].payload,
                                       q[rank].data, kResidency);
        detail::require_rank_residency(ec, static_cast<int>(rank), gate[rank].data, k[rank].data,
                                       v[rank].data, kResidency);
    }
    const std::size_t bytes = attn_input_proj_column_parallel_workspace_capacity_bytes(
        w[0].qtype, policy, x[0].ne[1], x[0].ne[1]);
    detail::require_split_workspace(workspace, {bytes, bytes}, kOp);
    std::array<Tensor, 2> q_out{q[0], q[1]};
    std::array<Tensor, 2> gate_out{gate[0], gate[1]};
    std::array<Tensor, 2> k_out{k[0], k[1]};
    std::array<Tensor, 2> v_out{v[0], v[1]};
    detail::for_each_rank(ec, [&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (w[slot].qtype == QType::Q3_G128_FP16) {
            // One rank's head-local Q|K|Gate|V shard [7168,5120].
            detail::Q3RowSegments segments;
            segments.count     = 4;
            Tensor* outputs[4] = {&q_out[slot], &k_out[slot], &gate_out[slot], &v_out[slot]};
            const std::int32_t rows[4] = {3072, 512, 3072, 512};
            std::int32_t end           = 0;
            for (int s = 0; s < 4; ++s) {
                end += rows[s];
                segments.data[s] = static_cast<__nv_bfloat16*>(outputs[s]->data);
                segments.ld[s]   = rows[s];
                segments.end[s]  = end;
            }
            const detail::Q3ArenaWorkspace q3ws(workspace[slot], w[slot].k, x[slot].ne[1]);
            detail::q3_linear_segments_dispatch(x[slot], w[slot], segments, ec.dev[slot]->stream,
                                                q3ws.get());
        } else if (w[slot].qtype == QType::NVFP4) {
            detail::nvfp4_attn_input_shard_dispatch(x[slot], w[slot], q_out[slot], gate_out[slot],
                                                    k_out[slot], v_out[slot], policy,
                                                    workspace[slot], ec.dev[slot]->stream);
        } else {
            detail::fp8_attn_input_shard_dispatch(x[slot], w[slot], q_out[slot], gate_out[slot],
                                                  k_out[slot], v_out[slot], policy, workspace[slot],
                                                  ec.dev[slot]->stream);
        }
    });
}

void attn_input_proj_column_parallel(const std::array<Tensor, 2>& x,
                                     const std::array<Weight, 2>& query_key_gate_value_weight,
                                     const std::array<Tensor, 2>& q,
                                     const std::array<Tensor, 2>& gate,
                                     const std::array<Tensor, 2>& k, const std::array<Tensor, 2>& v,
                                     const ExecutionContext& ec) {
    attn_input_proj_column_parallel(x, query_key_gate_value_weight, q, gate, k, v,
                                    LinearPolicy::A16Only, {nullptr, nullptr}, ec);
}

} // namespace ninfer::ops
