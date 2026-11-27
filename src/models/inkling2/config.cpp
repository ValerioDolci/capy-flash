#include "models/inkling2/config.h"

#include <cmath>
#include <limits>
#include <string>

namespace ninfer::models::inkling2 {
namespace {

using artifact::ArtifactError;
using artifact::Json;

// The reference derives the schedule from the layer index alone: every sixth layer attends over
// the whole context, the others over the sliding window.
constexpr std::uint32_t kGlobalLayerPeriod = 6;

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

// A member the reference reads but this implementation supports at one value only.
void fixed(const Json& config, const char* name, const Json& expected) {
    if (config.contains(name) && config.at(name) != expected) {
        throw ArtifactError(std::string(name) + ": supported value is " + expected.dump());
    }
}

void equal_dimension(const Json& config, const char* name, std::uint32_t expected) {
    if (config.contains(name) && dimension(config, name) != expected) {
        throw ArtifactError(std::string(name) + " must equal the attention geometry");
    }
}

} // namespace

std::vector<Visit> visit_schedule(const TextConfig& c) {
    std::vector<Visit> out;
    const auto visit = [&](std::uint32_t layer, std::uint32_t pass, std::uint32_t slot) {
        out.push_back(
            {layer, pass, slot, c.layer_types[layer], pass > 1, layer % c.block_layers == 0});
    };
    for (std::uint32_t layer = 0; layer < c.core_end; ++layer) { visit(layer, 1, layer); }
    for (std::uint32_t pass = 2; pass <= c.num_passes; ++pass) {
        for (std::uint32_t i = 0; i < c.core_layers(); ++i) {
            visit(c.core_start + i, pass, c.num_hidden_layers + (pass - 2) * c.core_layers() + i);
        }
    }
    for (std::uint32_t layer = c.core_end; layer < c.num_hidden_layers; ++layer) {
        visit(layer, 1, layer);
    }
    if (c.contrast.mode == LoopContrast::Logits) {
        const auto slots = static_cast<std::uint32_t>(out.size());
        for (std::uint32_t layer = c.core_end; layer < c.num_hidden_layers; ++layer) {
            visit(layer, 1, slots + layer - c.core_end);
            out.back().reference = true;
        }
    }
    return out;
}

TextConfig with_loop_contrast(TextConfig config, const LoopContrastOptions& options) {
    config.contrast = options;
    if (options.mode == LoopContrast::Off || options.strength == 0.0F) {
        config.contrast = {};
    } else if (!std::isfinite(options.strength) || options.strength < 0.0F ||
               options.reference_pass < 1 || options.reference_pass >= config.num_passes ||
               (options.adaptive && options.mode != LoopContrast::Logits)) {
        throw ArtifactError("loop contrast: the strength must be finite and positive, the "
                            "reference pass in [1, " +
                            std::to_string(config.num_passes - 1) +
                            "], adaptive only on the logits");
    }
    config.visits = visit_schedule(config);
    return config;
}

TextConfig parse_text_config(const Json& value) {
    artifact::require_members(value,
                              {"architectures",
                               "model_type",
                               "hidden_size",
                               "num_hidden_layers",
                               "vocab_size",
                               "unpadded_vocab_size",
                               "model_max_length",
                               "rms_norm_eps",
                               "num_attention_heads",
                               "num_key_value_heads",
                               "head_dim",
                               "d_rel",
                               "rel_extent",
                               "sliding_window_size",
                               "qk_l2_clip_threshold",
                               "rel_l2_clip_threshold",
                               "log_scaling_n_floor",
                               "log_scaling_alpha",
                               "sconv_kernel_size",
                               "n_routed_experts",
                               "num_experts_per_tok",
                               "intermediate_size",
                               "route_scale",
                               "mlp_gate_softcap",
                               "mlp_up_softcap",
                               "final_logit_softcapping",
                               "n_layers_per_attn_res_block",
                               "loop_config"},
                              {"torch_dtype", "q_bias", "o_bias", "use_value_norm_no_gamma",
                               "attn_res_normalize_values", "n_shared_experts", "use_gate_bias",
                               "scoring_func", "swa_head_dim", "swa_num_attention_heads",
                               "swa_num_key_value_heads", "local_layer_ids", "layer_types"},
                              "Inkling 2 text config");
    const auto& names = value.at("architectures");
    if (!names.is_array() || names.size() != 1 || names[0] != std::string(kTextArchitecture) ||
        value.at("model_type") != std::string(kTextModelType)) {
        throw ArtifactError("text component is not " + std::string(kTextArchitecture) + "/" +
                            std::string(kTextModelType));
    }
    TextConfig out;
    out.hidden_size             = dimension(value, "hidden_size");
    out.num_hidden_layers       = dimension(value, "num_hidden_layers");
    out.vocab_size              = dimension(value, "vocab_size");
    out.unpadded_vocab_size     = dimension(value, "unpadded_vocab_size");
    out.max_position_embeddings = dimension(value, "model_max_length");
    out.rms_norm_eps            = positive_float(value, "rms_norm_eps");
    out.num_attention_heads     = dimension(value, "num_attention_heads");
    out.num_key_value_heads     = dimension(value, "num_key_value_heads");
    out.head_dim                = dimension(value, "head_dim");
    out.relative_dim            = dimension(value, "d_rel");
    out.relative_extent         = dimension(value, "rel_extent");
    out.sliding_window          = dimension(value, "sliding_window_size");
    out.qk_clip_threshold       = positive_float(value, "qk_l2_clip_threshold");
    out.relative_clip_threshold = positive_float(value, "rel_l2_clip_threshold");
    out.log_scaling_floor       = dimension(value, "log_scaling_n_floor");
    out.log_scaling_alpha       = positive_float(value, "log_scaling_alpha");
    out.conv_kernel             = dimension(value, "sconv_kernel_size");
    out.num_experts             = dimension(value, "n_routed_experts");
    out.experts_per_token       = dimension(value, "num_experts_per_tok");
    out.expert_intermediate     = dimension(value, "intermediate_size");
    out.route_scale             = positive_float(value, "route_scale");
    out.gate_softcap            = positive_float(value, "mlp_gate_softcap");
    out.up_softcap              = positive_float(value, "mlp_up_softcap");
    out.final_logit_softcap     = positive_float(value, "final_logit_softcapping");
    out.block_layers            = dimension(value, "n_layers_per_attn_res_block");

    const auto& loop = value.at("loop_config");
    artifact::require_members(loop, {"core_start", "core_end", "num_passes", "gate_rank"},
                              {"is_varloop"}, "Inkling 2 loop_config");
    fixed(loop, "is_varloop", false);
    out.core_start = dimension(loop, "core_start");
    out.core_end   = dimension(loop, "core_end");
    out.num_passes = dimension(loop, "num_passes");
    out.gate_rank  = dimension(loop, "gate_rank");

    fixed(value, "torch_dtype", "bfloat16");
    fixed(value, "q_bias", false);
    fixed(value, "o_bias", false);
    fixed(value, "use_value_norm_no_gamma", true);
    fixed(value, "attn_res_normalize_values", true);
    fixed(value, "n_shared_experts", 0);
    fixed(value, "use_gate_bias", true);
    fixed(value, "scoring_func", "sigmoid");
    equal_dimension(value, "swa_head_dim", out.head_dim);
    equal_dimension(value, "swa_num_attention_heads", out.num_attention_heads);
    equal_dimension(value, "swa_num_key_value_heads", out.num_key_value_heads);

    if (out.unpadded_vocab_size > out.vocab_size) {
        throw ArtifactError("unpadded_vocab_size exceeds vocab_size");
    }
    if (out.num_attention_heads % out.num_key_value_heads) {
        throw ArtifactError("attention query heads must be divisible by KV heads");
    }
    if (out.experts_per_token > out.num_experts) {
        throw ArtifactError("selected experts exceed expert count");
    }
    if (out.core_start % out.block_layers || out.core_end % out.block_layers ||
        out.core_start >= out.core_end || out.core_end > out.num_hidden_layers) {
        throw ArtifactError("the looped core must be whole attention-residual blocks inside the "
                            "layers, after the first block");
    }

    Json local_ids = Json::array(), layer_names = Json::array();
    for (std::uint32_t layer = 0; layer < out.num_hidden_layers; ++layer) {
        const bool global = (layer + 1) % kGlobalLayerPeriod == 0;
        out.layer_types.push_back(global ? LayerKind::Global : LayerKind::Local);
        layer_names.push_back(global ? "hybrid" : "hybrid_sliding");
        if (!global) { local_ids.push_back(layer); }
    }
    fixed(value, "local_layer_ids", local_ids);
    fixed(value, "layer_types", layer_names);
    out.visits = visit_schedule(out);
    return out;
}

bool is_inkling2(const artifact::Directory& directory) {
    const auto found = directory.components.find("text");
    if (found == directory.components.end()) { return false; }
    const auto& config = found->second.config;
    const auto names   = config.find("architectures");
    const auto type    = config.find("model_type");
    return names != config.end() && names->is_array() && names->size() == 1 &&
           (*names)[0] == std::string(kTextArchitecture) && type != config.end() &&
           *type == std::string(kTextModelType);
}

Config parse_config(const artifact::Directory& directory, const LoadOptions& options) {
    try {
        if (options.vision) {
            throw ArtifactError("only the Text backbone and its drafters are implemented (no "
                                "Vision)");
        }
        if (options.purpose != EnginePurpose::Generation &&
            options.purpose != EnginePurpose::CausalScoring) {
            throw ArtifactError("unknown loading purpose");
        }
        Config out{with_loop_contrast(parse_text_config(directory.component("text").config),
                                      options.loop_contrast),
                   std::nullopt, std::nullopt};
        // An artifact without the selected drafter decodes without it; --spec auto chooses between
        // the two and needs both.
        const bool paired = options.speculative == SpeculativeBackend::Auto;
        if (paired &&
            (!directory.components.contains("mtp") || !directory.components.contains("dflash2"))) {
            throw ArtifactError("--spec auto needs an artifact with both drafters (the mtp and "
                                "dflash2 components)");
        }
        if ((options.speculative == SpeculativeBackend::Mtp || paired) &&
            directory.components.contains("mtp")) {
            out.draft = draft::parse_draft_config(directory.component("mtp").config, out.text);
        }
        if ((options.speculative == SpeculativeBackend::DFlash2 || paired) &&
            directory.components.contains("dflash2")) {
            out.dflash2 =
                draft::parse_dflash2_config(directory.component("dflash2").config, out.text);
        }
        return out;
    } catch (const std::exception& error) {
        throw ArtifactError(std::string("Inkling 2 config: ") + error.what());
    }
}

} // namespace ninfer::models::inkling2
