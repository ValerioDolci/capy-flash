#include "models/inkling2/draft/parameters.h"

#include "core/weight_view.h"

#include <array>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

namespace ninfer::models::inkling2::draft {
namespace {

std::int32_t extent(std::uint64_t value) {
    if (value > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("drafter dimension exceeds the Tensor integer domain");
    }
    return static_cast<std::int32_t>(value);
}

class Prepare {
public:
    Prepare(const Model& model, int device) : model_(model), device_(device) {}

    // A contiguous BF16 parameter as a Tensor with ne = the logical shape reversed.
    [[nodiscard]] Tensor tensor(WeightId id) const {
        const auto& bound = model_.weight(id, device_);
        const auto& shape = bound.view.shape;
        if (shape.empty() || shape.size() > 4) {
            throw std::invalid_argument(bound.name + ": direct parameter rank is not 1-4");
        }
        std::array<std::int32_t, 4> ne{1, 1, 1, 1};
        for (std::size_t i = 0; i < shape.size(); ++i) {
            ne[i] = extent(shape[shape.size() - 1 - i]);
        }
        Tensor out = weight_tensor(bound.view, {ne[0], ne[1], ne[2], ne[3]});
        if (out.dtype != DType::BF16) {
            throw std::invalid_argument(bound.name + ": drafter parameters are BF16");
        }
        return out;
    }

    [[nodiscard]] LinearParameters linear(WeightId id) const {
        return ops::prepare_linear_weight(model_.input(id, device_));
    }

    // Consecutive rows of one parent as one Linear operand (throws for separate parents).
    [[nodiscard]] LinearParameters stacked(std::initializer_list<WeightId> ids) const {
        std::vector<ops::WeightInput> rows;
        for (const auto id : ids) { rows.push_back(model_.input(id, device_)); }
        return ops::prepare_linear_weight(std::span<const ops::WeightInput>(rows));
    }

private:
    const Model& model_;
    int device_;
};

} // namespace

DraftConfig shard_draft_config(const DraftConfig& config, int tp) {
    if (tp != 1 && tp != 2) {
        throw std::invalid_argument("drafter tensor parallelism must be 1 or 2");
    }
    const auto parts = static_cast<std::uint32_t>(tp);
    if (config.num_key_value_heads % parts || config.intermediate_size % parts) {
        throw std::invalid_argument("drafter heads and MLP width must divide by tp");
    }
    DraftConfig out = config;
    out.num_attention_heads /= parts;
    out.num_key_value_heads /= parts;
    out.intermediate_size /= parts;
    return out;
}

Parameters::Parameters(const Model& source, int device, const DraftConfig& config)
    : model(source), device(device) {
    const auto& weights = source.weights().draft;
    if (!weights) { throw std::invalid_argument("the Model was loaded without the drafter"); }
    const Prepare p(source, device);
    fuse_weight            = p.tensor(weights->fuse_weight);
    mask_embedding         = p.tensor(weights->mask_embedding);
    final_norm             = p.tensor(weights->final_norm);
    markov_context_norm    = p.tensor(weights->markov_context_norm);
    markov_token_embedding = p.tensor(weights->markov_token_embedding);
    recurrence             = p.tensor(weights->recurrence);
    recurrence_bias        = p.tensor(weights->recurrence_bias);
    folded_output          = p.tensor(weights->folded_output);
    folded_bias            = p.tensor(weights->folded_bias);
    for (const auto& w : weights->layers) {
        LayerParameters layer;
        layer.context_norm        = p.tensor(w.context_norm);
        layer.input_norm          = p.tensor(w.input_norm);
        layer.post_attention_norm = p.tensor(w.post_attention_norm);
        layer.attention_input     = p.stacked({w.query, w.key, w.value});
        layer.output              = p.linear(w.output);
        layer.context             = p.stacked({w.context_key, w.context_value});
        layer.query_norm          = p.tensor(w.query_norm);
        layer.key_norm            = p.tensor(w.key_norm);
        layer.context_key_norm    = p.tensor(w.context_key_norm);
        layer.gate_up             = p.stacked({w.gate, w.up});
        layer.down                = p.linear(w.down);
        const auto D              = static_cast<std::int32_t>(config.head_dim);
        if (layer.attention_input.weight.n !=
                D * static_cast<std::int32_t>(config.num_attention_heads +
                                              2 * config.num_key_value_heads) ||
            layer.context.weight.n !=
                2 * D * static_cast<std::int32_t>(config.num_key_value_heads) ||
            layer.gate_up.weight.n != 2 * static_cast<std::int32_t>(config.intermediate_size)) {
            throw std::invalid_argument("drafter layer shards do not match the rank geometry");
        }
        layers.push_back(layer);
    }
}

} // namespace ninfer::models::inkling2::draft
