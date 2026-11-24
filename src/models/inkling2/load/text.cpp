#include "models/inkling2/load/bindings.h"

#include <array>
#include <string>

namespace ninfer::models::inkling2::loading {
namespace {

AttentionWeights bind_attention(Bindings& b, const TextConfig& c, std::uint32_t layer,
                                const std::string& p) {
    const std::uint64_t h = c.hidden_size, nh = c.num_attention_heads, nkv = c.num_key_value_heads,
                        taps = c.conv_kernel;
    const auto q = c.query_width(), kv = c.key_width();
    const auto a = p + "attention/";
    AttentionWeights out;
    out.query = b.parameter(a + "query", {q, h}, {a + "query_input"}, QType::BF16);
    out.gate  = b.parameter(a + "gate", {q, h}, {a + "query_input"}, QType::BF16);
    out.relative =
        b.parameter(a + "relative", {c.relative_width(), h}, {a + "query_input"}, QType::BF16);
    out.sink           = b.parameter(a + "sink", {nh, h}, {a + "query_input"}, QType::BF16);
    out.key            = b.parameter(a + "key", {kv, h}, {a + "key_input"}, QType::BF16);
    out.value          = b.parameter(a + "value", {kv, h}, {a + "value_input"}, QType::BF16);
    out.output         = b.parameter(a + "output", {h, q}, {a + "gated_output"}, QType::BF16);
    out.query_scale    = b.direct(a + "query_scale", {nh, c.head_dim});
    out.relative_scale = b.direct(a + "relative_scale", {nh, c.relative_dim});
    out.relative_profile =
        b.direct(a + "relative_profile", {c.relative_dim, c.relative_bias_extent(layer)});
    if (c.global(layer)) {
        DynamicKvConvWeights conv;
        conv.key_static = b.direct(a + "key_conv_static", {taps, kv}, QType::FP32);
        conv.key_projection =
            b.parameter(a + "key_conv_projection", {nkv * taps, h}, {a + "key_input"}, QType::BF16);
        conv.value_static     = b.direct(a + "value_conv_static", {taps, kv}, QType::FP32);
        conv.value_projection = b.parameter(a + "value_conv_projection", {nkv * taps, h},
                                            {a + "value_input"}, QType::BF16);
        out.kv_conv           = conv;
    } else {
        out.kv_conv = StaticKvConvWeights{b.direct(a + "key_conv", {kv, taps}, QType::FP32),
                                          b.direct(a + "value_conv", {kv, taps}, QType::FP32)};
    }
    return out;
}

MoeWeights bind_moe(Bindings& b, const TextConfig& c, const std::string& p) {
    const std::uint64_t h = c.hidden_size, f = c.expert_intermediate;
    MoeWeights out;
    out.router      = b.direct(p + "moe/router", {c.num_experts, h}, QType::FP32);
    out.router_bias = b.direct(p + "moe/router_bias", {c.num_experts}, QType::FP32);
    out.experts.reserve(c.num_experts);
    for (std::uint32_t e = 0; e < c.num_experts; ++e) {
        const auto ep = p + "moe/experts/" + std::to_string(e) + "/";
        // NVFP4 or MXFP8, as the checkpoint stored the layer's experts (one format per bank),
        // mapped from the file and cut into the ranks' banks at load (load/expert_banks.h).
        constexpr auto mapped = artifact::Residency::HostFile;
        out.experts.push_back(
            {b.parameter(ep + "gate_up", {2ULL * f, h}, {p + "ffn_input"}, {}, mapped),
             b.parameter(ep + "down", {h, f}, {ep + "product"}, {}, mapped)});
    }
    return out;
}

CoreWeights bind_core(Bindings& b, const TextConfig& c, const std::string& p) {
    const std::uint64_t h = c.hidden_size, rank = c.gate_rank;
    CoreWeights out;
    out.message_queries = b.direct(p + "message_queries", {4, h}, QType::FP32);
    constexpr std::array<const char*, 4> roles{"query", "key", "value", "ffn"};
    for (std::size_t i = 0; i < roles.size(); ++i) {
        const auto mp = p + "message_gate/" + roles[i] + "/";
        out.gates[i]  = {b.parameter(mp + "down", {rank, 2 * h}, {mp + "input"}, QType::BF16),
                         b.direct(mp + "forget", {h, rank}, QType::FP32),
                         b.direct(mp + "gain", {h, rank}, QType::FP32)};
    }
    return out;
}

LayerWeights bind_layer(Bindings& b, const TextConfig& c, std::uint32_t layer) {
    const std::uint64_t h = c.hidden_size, taps = c.conv_kernel;
    const auto p = "text/layers/" + std::to_string(layer) + "/";
    LayerWeights out;
    if (const auto queries = c.attention_queries(layer)) {
        out.attention_queries = b.direct(p + "attention_queries", {queries, h}, QType::FP32);
    }
    if (c.global(layer)) {
        for (const auto* role : {"query", "key", "value"}) {
            out.input_norms.push_back(b.direct(p + role + "_input_norm", {h}));
        }
    } else {
        out.input_norms.push_back(b.direct(p + "input_norm", {h}));
    }
    out.attention           = bind_attention(b, c, layer, p);
    out.post_attention_conv = b.direct(p + "post_attention_conv", {h, taps}, QType::FP32);
    out.ffn_queries         = b.direct(p + "ffn_queries", {1, h}, QType::FP32);
    out.ffn_norm            = b.direct(p + "ffn_norm", {h});
    out.moe                 = bind_moe(b, c, p);
    out.post_ffn_conv       = b.direct(p + "post_ffn_conv", {h, taps}, QType::FP32);
    if (c.core(layer)) { out.core = bind_core(b, c, p); }
    return out;
}

} // namespace

TextWeights bind_text(Bindings& b, const TextConfig& c) {
    const std::uint64_t h = c.hidden_size;
    TextWeights out;
    out.token_embedding = b.direct("text/token_embedding", {c.vocab_size, h});
    out.embedding_norm  = b.direct("text/embedding_norm", {h});
    out.layers.reserve(c.num_hidden_layers);
    for (std::uint32_t layer = 0; layer < c.num_hidden_layers; ++layer) {
        out.layers.push_back(bind_layer(b, c, layer));
    }
    out.final_queries   = b.direct("text/final_queries", {1, h}, QType::FP32);
    out.final_norm      = b.direct("text/final_norm", {h});
    out.output_head     = b.parameter("text/output_head", {c.unpadded_vocab_size, h},
                                      {"text/final_hidden"}, QType::BF16);
    out.output_head_use = b.use(out.output_head, "text/final_hidden");
    return out;
}

} // namespace ninfer::models::inkling2::loading
