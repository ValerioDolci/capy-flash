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

} // namespace ninfer::models::inkling2::draft
