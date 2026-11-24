#pragma once

#include "core/weight_view.h"
#include "ninfer/ops/linear.h"

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace ninfer::models::inkling2 {

// Logical parameters of the Text component (tools/convert/inkling2.py writes them; the names are
// the artifact contract). <L> is the physical layer, <e> the expert, H the hidden size, V the
// padded and Vu the unpadded vocabulary, nh/nkv the query/KV heads of width D, dr the relative
// width, K the convolution taps, E the experts of width F, r the message gate rank. "@ x" names
// the parameter's linear Use. Under text/:
//
//   token_embedding [V,H] BF16; embedding_norm [H] BF16
//   layers/<L>/attention_queries [1|3,H] FP32: L > 0; three rows (Q, K, V) for a global layer
//   layers/<L>/input_norm [H] BF16 (local); {query,key,value}_input_norm [H] BF16 (global)
//   layers/<L>/attention/query [nh*D,H], gate [nh*D,H], relative [nh*dr,H], sink [nh,H]:
//     BF16 @ layers/<L>/attention/query_input
//   layers/<L>/attention/key [nkv*D,H] BF16 @ attention/key_input; value likewise @ value_input
//   layers/<L>/attention/output [H,nh*D] BF16 @ attention/gated_output
//   layers/<L>/attention/query_scale [nh,D], relative_scale [nh,dr] BF16
//   layers/<L>/attention/relative_profile [dr,X] BF16: X = window (local), rel_extent (global)
//   layers/<L>/attention/{key,value}_conv [nkv*D,K] FP32 (local): tap K-1 weighs the current token
//   layers/<L>/attention/{key,value}_conv_static [K,nkv*D] FP32 (global): row 0 = current token
//   layers/<L>/attention/{key,value}_conv_projection [nkv*K,H] BF16 (global): row h*K + lag,
//     @ attention/{key,value}_input
//   layers/<L>/post_attention_conv [H,K] FP32, post_ffn_conv [H,K] FP32: tap K-1 = current
//   layers/<L>/ffn_queries [1,H] FP32; ffn_norm [H] BF16
//   layers/<L>/moe/router [E,H] FP32; moe/router_bias [E] FP32
//   layers/<L>/moe/experts/<e>/gate_up [2F,H]: rows [0,F) gate, [F,2F) up; @ ffn_input
//   layers/<L>/moe/experts/<e>/down [H,F] @ moe/experts/<e>/product
//     NVFP4 (block_scale_k16_m128x4) in the layers stored NVFP4 by the checkpoint, BF16 otherwise
//   layers/<L>/message_queries [4,H] FP32 (core L): rows Q, K, V, FFN
//   layers/<L>/message_gate/<role>/down [r,2H] BF16 @ message_gate/<role>/input (core L)
//   layers/<L>/message_gate/<role>/{forget,gain} [H,r] FP32 (core L); role: query key value ffn
//   final_queries [1,H] FP32; final_norm [H] BF16
//   output_head [Vu,H] BF16 @ text/final_hidden
//
// The experts of a layer are stored as two banks per matrix, IDs [0, E/2) and [E/2, E): each
// bank is one parent (an NVFP4 bank keeps one weight divisor per expert), whose rows are its
// experts' rows in ID order. The model maps them from the file and cuts the banks it computes
// with, every expert at half the intermediate width (expert_bank.h).
//
// The DSpark drafter (component mtp, draft/config.h; <i> its layer, nh/nkv/D its heads, F its
// MLP width, r the recurrent rank), every parameter BF16. Under mtp/:
//
//   fuse_weight [layers+1, 4]: mixing logits of the four read states, row i = layer i, the last
//     row the recurrent head; mask_embedding [H]; final_norm [H]
//   layers/<i>/context_norm [H], input_norm [H], post_attention_norm [H]
//   layers/<i>/attention/query [nh*D,H], key [nkv*D,H], value [nkv*D,H] @ attention/input
//   layers/<i>/attention/output [H,nh*D] @ attention/output_input
//   layers/<i>/attention/context_key [nkv*D,H], context_value [nkv*D,H] @ layers/<i>/context
//   layers/<i>/attention/query_norm [D], key_norm [D], context_key_norm [D]
//   layers/<i>/mlp/gate [F,H], up [F,H] @ mlp/input; mlp/down [H,F] @ mlp/product
//   markov/context_norm [H]; markov/token_embedding [V,r]; markov/recurrence [3r, 2r+2H] (input
//     columns state | token_embedding[prev] | hidden | context, output rows gate | candidate |
//     output) and recurrence_bias [3r]; markov/folded_output [Vu,r] and folded_bias [Vu]: the
//     output projection of the recurrent latent folded into output_head rows in FP64
//
// The drafter's query|key|value, context_key|context_value and gate|up are stacked parents.

struct WeightId {
    std::size_t index                          = std::numeric_limits<std::size_t>::max();
    friend bool operator==(WeightId, WeightId) = default;
};

struct WeightUseId {
    WeightId parameter;
    std::size_t use_index = std::numeric_limits<std::size_t>::max();
};

struct WeightUse {
    std::string input;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
};

struct BoundWeight {
    std::string name;
    std::vector<std::string> source_objects;
    WeightView view;
    std::vector<WeightUse> uses;
};

// Static depthwise convolutions of K and V (local layers).
struct StaticKvConvWeights {
    WeightId key, value;
};

// Dynamic convolutions of K and V (global layers): static taps plus a per-token projection.
struct DynamicKvConvWeights {
    WeightId key_static, key_projection, value_static, value_projection;
};

struct AttentionWeights {
    WeightId query, gate, relative, sink, key, value, output;
    WeightId query_scale, relative_scale, relative_profile;
    std::variant<StaticKvConvWeights, DynamicKvConvWeights> kv_conv;
};

struct ExpertWeights {
    WeightId gate_up, down;
};

struct MoeWeights {
    WeightId router, router_bias;
    std::vector<ExpertWeights> experts;
};

struct MessageGateWeights {
    WeightId down, forget, gain;
};

// Cross-pass messages of a core layer (passes after the first).
struct CoreWeights {
    WeightId message_queries;
    // Gates of the attention query, key and value inputs and of the FFN input, in that order.
    std::array<MessageGateWeights, 4> gates;
};

struct LayerWeights {
    std::optional<WeightId> attention_queries; // absent for layer 0
    // Local: one input norm for Q, K and V. Global: the query, key and value input norms.
    std::vector<WeightId> input_norms;
    AttentionWeights attention;
    WeightId post_attention_conv;
    WeightId ffn_queries, ffn_norm;
    MoeWeights moe;
    WeightId post_ffn_conv;
    std::optional<CoreWeights> core;
};

struct TextWeights {
    WeightId token_embedding, embedding_norm;
    std::vector<LayerWeights> layers;
    WeightId final_queries, final_norm, output_head;
    WeightUseId output_head_use;
};

struct DraftLayerWeights {
    WeightId context_norm, input_norm, post_attention_norm;
    WeightId query, key, value, output, query_norm, key_norm;
    WeightId context_key, context_value, context_key_norm;
    WeightId gate, up, down;
};

struct DraftWeights {
    WeightId fuse_weight, mask_embedding, final_norm;
    std::vector<DraftLayerWeights> layers;
    WeightId markov_context_norm, markov_token_embedding, recurrence, recurrence_bias;
    WeightId folded_output, folded_bias;
};

struct ModelWeights {
    TextWeights text;
    std::optional<DraftWeights> draft; // with the speculative backend
};

} // namespace ninfer::models::inkling2
