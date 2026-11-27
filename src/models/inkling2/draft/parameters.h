#pragma once

// Native operands of the DSpark drafter on one rank (weights.h names the logical parameters, mtp/).
// Linear projections are prepared Weights; every other parameter is a contiguous Tensor whose ne
// order is the logical shape reversed. At tp 2 a rank holds its 12 of 24 query heads and 4 of 8 KV
// heads of every layer, half of the MLP, and the folded head's rows of its output-head vocabulary
// half (load/sharding.h); the norms, the mixing logits, the mask embedding and the recurrence are
// replicated. The drafter also reads the Text component's token embedding, embedding norm and
// output head (execution::TextParameters of the same rank).

#include "core/tensor.h"
#include "models/inkling2/draft/config.h"
#include "models/inkling2/execution/parameters.h"
#include "models/inkling2/model.h"

#include <vector>

namespace ninfer::models::inkling2::draft {

using LinearParameters = execution::LinearParameters;

struct LayerParameters {
    Tensor context_norm, input_norm, post_attention_norm; // BF16 [H]
    // The rank's query|key|value rows [(nh_r + 2 nkv_r) D, H] (one stacked parent), and the
    // attention output columns [H, nh_r D].
    LinearParameters attention_input, output;
    // The rank's context_key|context_value rows [2 nkv_r D, H] (one stacked parent).
    LinearParameters context;
    Tensor query_norm, key_norm, context_key_norm; // BF16 [D]
    // The rank's gate|up rows [2 F_r, H] (one stacked parent) and down columns [H, F_r].
    LinearParameters gate_up, down;
};

class Parameters {
public:
    // Requires the Model to carry the drafter (Config::draft). `config` is the rank's geometry
    // (shard_draft_config).
    Parameters(const Model& source, int device, const DraftConfig& config);
    Parameters(const Parameters&)            = delete;
    Parameters& operator=(const Parameters&) = delete;

    const Model& model;
    const int device;
    Tensor fuse_weight;            // BF16 [layers + 1, 4]
    Tensor mask_embedding;         // BF16 [H]
    Tensor final_norm;             // BF16 [H]
    Tensor markov_context_norm;    // BF16 [H]
    Tensor markov_token_embedding; // BF16 [V, r]
    Tensor recurrence;             // BF16 [3r, 2r + 2H]
    Tensor recurrence_bias;        // BF16 [3r]
    Tensor folded_output;          // BF16 [Vu_r, r]
    Tensor folded_bias;            // BF16 [Vu_r]
    std::vector<LayerParameters> layers;
};

// The drafter geometry a rank computes with at tensor-parallel width `tp` (1 or 2): the query and
// KV heads and the MLP width divided by tp.
[[nodiscard]] DraftConfig shard_draft_config(const DraftConfig& config, int tp);

// The DFlash2 drafter's operands on one rank. At tp 2 a rank holds its query and KV heads of every
// layer (query|key|value rows, output columns) and half of the MLP; the context's key|value rows
// are the same parent's rows. Everything else is replicated: the feature projection, the norms, the
// convolutions, the selector's projection and codebooks.
struct DFlash2ConvParameters {
    Tensor base_kernel;                 // BF16 [2, 2, H] (ne {H, taps, sides})
    LinearParameters kernel_projection; // BF16 [4G, H]
};

struct DFlash2LayerParameters {
    Tensor input_norm, post_attention_norm; // BF16 [H]
    Tensor query_norm, key_norm;            // BF16 [D]
    // The rank's query|key|value rows [(nh_r + 2 nkv_r) D, H] and, of the same parent, its
    // key|value rows [2 nkv_r D, H] (the context projection).
    LinearParameters attention_input, context;
    LinearParameters output;        // [H, nh_r D]
    LinearParameters gate_up, down; // [2 F_r, H], [H, F_r]
    DFlash2ConvParameters attention_conv, mlp_conv;
};

class DFlash2Parameters {
public:
    // Requires the Model to carry the DFlash2 drafter. `config` is the rank's geometry
    // (shard_dflash2_config).
    DFlash2Parameters(const Model& source, int device, const DFlash2Config& config);
    DFlash2Parameters(const DFlash2Parameters&)            = delete;
    DFlash2Parameters& operator=(const DFlash2Parameters&) = delete;

    const Model& model;
    const int device;
    LinearParameters feature_projection; // Q8 [H, 4H]
    Tensor context_norm, final_norm;     // BF16 [H]
    std::vector<DFlash2LayerParameters> layers;
    LinearParameters hidden_projection;              // BF16 [r, H]
    Tensor predecessor_codebook, successor_codebook; // BF16 [V, r]
};

[[nodiscard]] DFlash2Config shard_dflash2_config(const DFlash2Config& config, int tp);

} // namespace ninfer::models::inkling2::draft
