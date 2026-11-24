#pragma once

#include "core/tensor.h"
#include "core/weight_view.h"
#include "models/inkling2/model.h"
#include "ninfer/ops/weight_input.h"

#include <array>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace ninfer::models::inkling2::execution {

// Native operands of one rank (models/inkling2/weights.h names the logical parameters). Linear
// projections are prepared Weights; every other parameter is a contiguous Tensor whose ne order
// is the logical shape reversed (a [rows, columns] matrix has ne = {columns, rows}). At tp 2 the
// head-split parameters are this rank's shards (load/sharding.h): 12 of 24 query heads and 4 of
// 8 KV heads for the released model, the output head's vocabulary half.
using LinearParameters = ops::SingleProjectionWeight;

// Local layers: depthwise taps [C_r, K] FP32 (tap K-1 = current token).
struct StaticKvConvParameters {
    Tensor key, value;
};

// Global layers: static taps [K, C_r] FP32 (row 0 = current token) and the per-token tap
// projections [nkv_r * K, H] (row h*K + lag).
struct DynamicKvConvParameters {
    Tensor key_static, value_static;
    LinearParameters key_projection, value_projection;
};

struct AttentionParameters {
    // From the query input: query [nh_r*D, H], gate [nh_r*D, H], relative [nh_r*dr, H], sink
    // [nh_r, H]. From the key and value inputs: key and value [nkv_r*D, H]. output [H, nh_r*D].
    LinearParameters query, gate, relative, sink, key, value, output;
    // The same rows stacked per input (one parent each, tools/convert/inkling2_recipes.py), so one
    // Linear serves an input: queries = query|gate|relative|sink [nh_r*(2D+dr+1), H]; a local
    // layer's all = queries|key|value (its three inputs coincide outside the cross-pass messages);
    // a global layer's keys = key|key_projection and values = value|value_projection.
    LinearParameters queries;
    std::optional<LinearParameters> all, keys, values;
    Tensor query_scale;      // BF16 [nh_r, D]
    Tensor relative_scale;   // BF16 [nh_r, dr]
    Tensor relative_profile; // BF16 [dr, extent]
    std::variant<StaticKvConvParameters, DynamicKvConvParameters> kv_conv;
};

// One bank of routed experts, IDs [first_expert, first_expert + experts), at half the
// intermediate width F (models/inkling2/expert_bank.h): whole views of the bank's parents,
// gate_up [experts*F, H] (per expert F/2 gate rows then the F/2 matching up rows) and down
// [experts*H, F/2], NVFP4 (one divisor per expert), MXFP8 or BF16. These are the operands of
// ops::moe_bank_capped_swiglu (MoeBank{gate_up, down, first_expert}).
struct ExpertBankParameters {
    WeightView gate_up, down;
    std::int32_t first_expert = 0;
    std::int32_t experts      = 0;
};

struct MoeParameters {
    Tensor router;      // FP32 [E, H]
    Tensor router_bias; // FP32 [E]
    // The banks this rank computes: both halves of F at tp 1, rank r's half at tp 2.
    std::vector<ExpertBankParameters> banks;
};

struct MessageGateParameters {
    Tensor down;                   // BF16 [r, 2H]
    Tensor forget, gain;           // FP32 [H, r]
    Tensor forget_bf16, gain_bf16; // the same values in BF16 (message_gates_rmsnorm)
};

struct CoreParameters {
    Tensor message_queries; // FP32 [4, H]: Q, K, V, FFN
    // Gates of the attention query, key and value inputs and of the FFN input.
    std::array<MessageGateParameters, 4> gates;
};

struct LayerParameters {
    std::optional<Tensor> attention_queries; // FP32 [1|3, H]; absent for layer 0
    // BF16 [H]: one for a local layer, the query, key and value norms for a global layer.
    std::vector<Tensor> input_norms;
    AttentionParameters attention;
    Tensor post_attention_conv; // FP32 [H, K]
    Tensor ffn_queries;         // FP32 [1, H]
    Tensor ffn_norm;            // BF16 [H]
    MoeParameters moe;
    Tensor post_ffn_conv; // FP32 [H, K]
    std::optional<CoreParameters> core;
};

struct TextParameters {
    Tensor token_embedding; // BF16 [V, H]
    Tensor embedding_norm;  // BF16 [H]
    std::vector<LayerParameters> layers;
    Tensor final_queries;         // FP32 [1, H]
    Tensor final_norm;            // BF16 [H]
    LinearParameters output_head; // [Vu_r, H]: this rank's vocabulary rows
};

// Cold native preparation of one rank; every address borrows the source Model.
class Parameters {
public:
    explicit Parameters(const Model& source) : Parameters(source, 0) {}

    Parameters(const Model& source, int device);
    Parameters(const Parameters&)            = delete;
    Parameters& operator=(const Parameters&) = delete;
    Parameters(Parameters&&)                 = delete;
    Parameters& operator=(Parameters&&)      = delete;

    const Model& model;
    const int device;
    TextParameters text;
};

} // namespace ninfer::models::inkling2::execution
