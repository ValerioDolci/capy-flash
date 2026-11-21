#include "models/inkling2/draft/config.h"

#include "models/inkling2/config.h"

#include <cmath>
#include <limits>
#include <string>

namespace ninfer::models::inkling2::draft {
namespace {

using artifact::ArtifactError;
using artifact::Json;

std::uint32_t dimension(const Json& config, const char* name) {
    const auto n = artifact::require_u64(config.at(name), name, true);
    if (n > std::numeric_limits<std::int32_t>::max()) {
        throw ArtifactError(std::string(name) + ": config dimension exceeds i32");
    }
    return static_cast<std::uint32_t>(n);
}

float positive_float(const Json& config, const char* name) {
    const auto& value = config.at(name);
    if (!value.is_number()) { throw ArtifactError(std::string(name) + " must be a real value"); }
    const float result = value.get<float>();
    if (!std::isfinite(result) || result <= 0) {
        throw ArtifactError(std::string(name) + " must be positive finite FP32");
    }
    return result;
}

void fixed(const Json& config, const char* name, const Json& expected) {
    if (config.contains(name) && config.at(name) != expected) {
        throw ArtifactError(std::string(name) + ": supported value is " + expected.dump());
    }
}

} // namespace

DraftConfig parse_draft_config(const Json& value, const TextConfig& text) {
    try {
        artifact::require_members(
            value,
            {"architectures", "model_type", "hidden_size", "num_hidden_layers",
             "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size",
             "rms_norm_eps", "rope_theta", "local_layer_ids", "sliding_window_size", "block_size",
             "markov_rank", "markov_head_type", "target_layer_ids"},
            {"max_position_embeddings", "torch_dtype", "q_bias", "o_bias"}, "Inkling 2 mtp config");
        const auto& names = value.at("architectures");
        if (!names.is_array() || names.size() != 1 || names[0] != std::string(kArchitecture) ||
            value.at("model_type") != std::string(kModelType)) {
            throw ArtifactError("mtp component is not " + std::string(kArchitecture) + "/" +
                                std::string(kModelType));
        }
        DraftConfig out;
        out.hidden_size         = dimension(value, "hidden_size");
        out.num_layers          = dimension(value, "num_hidden_layers");
        out.num_attention_heads = dimension(value, "num_attention_heads");
        out.num_key_value_heads = dimension(value, "num_key_value_heads");
        out.head_dim            = dimension(value, "head_dim");
        out.intermediate_size   = dimension(value, "intermediate_size");
        out.sliding_window      = dimension(value, "sliding_window_size");
        out.block_size          = dimension(value, "block_size");
        out.markov_rank         = dimension(value, "markov_rank");
        out.rms_norm_eps        = positive_float(value, "rms_norm_eps");
        out.rope_theta          = positive_float(value, "rope_theta");
        fixed(value, "markov_head_type", "rnn_hidden");
        fixed(value, "torch_dtype", "bfloat16");
        fixed(value, "q_bias", false);
        fixed(value, "o_bias", false);
        if (value.contains("max_position_embeddings") &&
            dimension(value, "max_position_embeddings") < text.max_position_embeddings) {
            throw ArtifactError("max_position_embeddings is below the Text component's");
        }
        if (out.hidden_size != text.hidden_size) {
            throw ArtifactError("hidden_size must be the Text component's");
        }
        if (out.num_attention_heads % out.num_key_value_heads || out.head_dim % 2 ||
            out.block_size < 2) {
            throw ArtifactError("attention geometry or block size is invalid");
        }
        out.local.assign(out.num_layers, false);
        const auto& local = value.at("local_layer_ids");
        if (!local.is_array()) { throw ArtifactError("local_layer_ids must be an array"); }
        for (const auto& id : local) {
            const auto layer = artifact::require_u64(id, "local_layer_ids", false);
            if (layer >= out.num_layers || out.local[layer]) {
                throw ArtifactError("local_layer_ids names no layer or repeats one");
            }
            out.local[layer] = true;
        }
        const auto& targets = value.at("target_layer_ids");
        if (!targets.is_array() || targets.size() != kFeatures) {
            throw ArtifactError("target_layer_ids must name four states");
        }
        const auto visits = static_cast<std::uint64_t>(text.visits.size());
        for (std::size_t j = 0; j < kFeatures; ++j) {
            const auto id = artifact::require_u64(targets[j], "target_layer_ids", false);
            if (j + 1 < kFeatures) {
                if (id >= visits) {
                    throw ArtifactError("target_layer_ids: a visit outside the schedule");
                }
                out.feature_visits[j] = static_cast<std::uint32_t>(id);
            } else if (id != visits) {
                throw ArtifactError("target_layer_ids must end with the head-input sentinel " +
                                    std::to_string(visits));
            }
        }
        return out;
    } catch (const std::exception& error) {
        throw ArtifactError(std::string("Inkling 2 mtp config: ") + error.what());
    }
}

} // namespace ninfer::models::inkling2::draft
