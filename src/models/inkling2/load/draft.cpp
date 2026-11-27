#include "models/inkling2/load/bindings.h"

#include <string>

namespace ninfer::models::inkling2::loading {

DraftWeights bind_draft(Bindings& b, const draft::DraftConfig& c, const TextConfig& text) {
    const std::uint64_t h = c.hidden_size, d = c.head_dim, f = c.intermediate_size,
                        r = c.markov_rank;
    const auto q = c.query_width(), kv = c.key_width();
    DraftWeights out;
    out.fuse_weight    = b.direct("mtp/fuse_weight", {c.num_layers + 1ULL, draft::kFeatures});
    out.mask_embedding = b.direct("mtp/mask_embedding", {h});
    out.final_norm     = b.direct("mtp/final_norm", {h});
    out.layers.reserve(c.num_layers);
    for (std::uint32_t layer = 0; layer < c.num_layers; ++layer) {
        const auto p = "mtp/layers/" + std::to_string(layer) + "/";
        const auto a = p + "attention/";
        DraftLayerWeights w;
        w.context_norm        = b.direct(p + "context_norm", {h});
        w.input_norm          = b.direct(p + "input_norm", {h});
        w.post_attention_norm = b.direct(p + "post_attention_norm", {h});
        w.query               = b.parameter(a + "query", {q, h}, {a + "input"}, QType::BF16);
        w.key                 = b.parameter(a + "key", {kv, h}, {a + "input"}, QType::BF16);
        w.value               = b.parameter(a + "value", {kv, h}, {a + "input"}, QType::BF16);
        w.output        = b.parameter(a + "output", {h, q}, {a + "output_input"}, QType::BF16);
        w.query_norm    = b.direct(a + "query_norm", {d});
        w.key_norm      = b.direct(a + "key_norm", {d});
        w.context_key   = b.parameter(a + "context_key", {kv, h}, {p + "context"}, QType::BF16);
        w.context_value = b.parameter(a + "context_value", {kv, h}, {p + "context"}, QType::BF16);
        w.context_key_norm = b.direct(a + "context_key_norm", {d});
        w.gate             = b.parameter(p + "mlp/gate", {f, h}, {p + "mlp/input"}, QType::BF16);
        w.up               = b.parameter(p + "mlp/up", {f, h}, {p + "mlp/input"}, QType::BF16);
        w.down             = b.parameter(p + "mlp/down", {h, f}, {p + "mlp/product"}, QType::BF16);
        out.layers.push_back(w);
    }
    out.markov_context_norm    = b.direct("mtp/markov/context_norm", {h});
    out.markov_token_embedding = b.direct("mtp/markov/token_embedding", {text.vocab_size, r});
    out.recurrence             = b.direct("mtp/markov/recurrence", {3 * r, 2 * r + 2 * h});
    out.recurrence_bias        = b.direct("mtp/markov/recurrence_bias", {3 * r});
    out.folded_output = b.direct("mtp/markov/folded_output", {text.unpadded_vocab_size, r});
    out.folded_bias   = b.direct("mtp/markov/folded_bias", {text.unpadded_vocab_size});
    return out;
}

DFlash2Weights bind_dflash2(Bindings& b, const draft::DFlash2Config& c, const TextConfig& text) {
    const std::uint64_t h = c.hidden_size, d = c.head_dim, f = c.intermediate_size,
                        r = c.selector_rank, taps = c.conv_taps;
    const auto q = c.query_width(), kv = c.key_width(), coefficients = c.conv_coefficients();
    DFlash2Weights out;
    out.feature_projection = b.parameter("dflash2/feature_projection", {h, draft::kFeatures * h},
                                         {"dflash2/target_features"}, QType::Q8_G32_FP16);
    out.context_norm       = b.direct("dflash2/context_norm", {h});
    out.final_norm         = b.direct("dflash2/final_norm", {h});
    out.layers.reserve(c.num_layers);
    for (std::uint32_t layer = 0; layer < c.num_layers; ++layer) {
        const auto p    = "dflash2/layers/" + std::to_string(layer) + "/";
        const auto a    = p + "attention/";
        const auto conv = [&](const std::string& branch) {
            return DFlash2ConvWeights{b.direct(p + branch + "/base_kernel", {2, taps, h}),
                                      b.parameter(p + branch + "/kernel_projection",
                                                  {coefficients, h}, {p + branch + "/input"},
                                                  QType::BF16)};
        };
        DFlash2LayerWeights w;
        w.input_norm          = b.direct(p + "input_norm", {h});
        w.post_attention_norm = b.direct(p + "post_attention_norm", {h});
        w.query               = b.parameter(a + "query", {q, h}, {a + "input"}, QType::Q8_G32_FP16);
        w.key =
            b.parameter(a + "key", {kv, h}, {a + "input", "dflash2/context"}, QType::Q8_G32_FP16);
        w.value =
            b.parameter(a + "value", {kv, h}, {a + "input", "dflash2/context"}, QType::Q8_G32_FP16);
        w.query_norm = b.direct(a + "query_norm", {d});
        w.key_norm   = b.direct(a + "key_norm", {d});
        w.output     = b.parameter(a + "output", {h, q}, {a + "output_input"}, QType::Q8_G32_FP16);
        w.gate       = b.parameter(p + "mlp/gate", {f, h}, {p + "mlp/input"}, QType::Q8_G32_FP16);
        w.up         = b.parameter(p + "mlp/up", {f, h}, {p + "mlp/input"}, QType::Q8_G32_FP16);
        w.down       = b.parameter(p + "mlp/down", {h, f}, {p + "mlp/product"}, QType::Q8_G32_FP16);
        w.attention_conv = conv("attention_conv");
        w.mlp_conv       = conv("mlp_conv");
        out.layers.push_back(w);
    }
    const std::string s = "dflash2/candidate_selector/";
    out.hidden_projection =
        b.parameter(s + "hidden_projection", {r, h}, {"dflash2/final_hidden"}, QType::BF16);
    out.predecessor_codebook = b.direct(s + "predecessor_codebook", {text.vocab_size, r});
    out.successor_codebook   = b.direct(s + "successor_codebook", {text.vocab_size, r});
    return out;
}

} // namespace ninfer::models::inkling2::loading
