// Inkling 2 tensor-parallel placement (models/inkling2/load/sharding.h), host only.
//
// 1. Every logical parameter of the released and the synthetic geometry (the converter's name
//    list, written out here from the config) at tp 2: axis, holder and per-rank ranges against
//    the name table of expected_placement.h; whole and Replicated at tp 1.
// 2. The rank config (12/4 heads, half the output rows) and the rejections: a parameter without
//    a rule, a layer outside the model, tensor parallelism 3, indivisible heads, an expert width
//    that does not halve into 64-column parts.
// 3. With embedding_host, text/token_embedding alone among the text parameters is HostMapped, at
//    tp 1 and tp 2, and so are the DFlash2 selector's two codebooks.

#include "artifact/fixture.h"
#include "models/inkling2/config.h"
#include "models/inkling2/execution/tp.h"
#include "models/inkling2/load/sharding.h"
#include "models/inkling2/expected_placement.h"

#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace loading = ninfer::models::inkling2::loading;
using artifact::Json;
using artifact::ShardAxis;
using ninfer::test::artifact_fixture::rejects;
using ninfer::test::artifact_fixture::require;
using Named = std::vector<std::pair<std::string, artifact::Shape>>;

inkling2::TextConfig config(const char* file) {
    std::ifstream stream(std::string(NINFER_SOURCE_DIR "/tests/models/inkling2/") + file);
    auto text             = Json::parse(stream).at("text_config");
    text["architectures"] = Json::array({"Inkling2LoopForCausalLM"});
    return inkling2::parse_text_config(text);
}

// The logical parameters tools/convert/inkling2.py writes, with their shapes.
Named parameters(const inkling2::TextConfig& c) {
    const std::uint64_t h = c.hidden_size, nh = c.num_attention_heads, nkv = c.num_key_value_heads,
                        d = c.head_dim, dr = c.relative_dim, k = c.conv_kernel,
                        f = c.expert_intermediate, r = c.gate_rank;
    Named out{{"text/token_embedding", {c.vocab_size, h}},
              {"text/embedding_norm", {h}},
              {"text/final_queries", {1, h}},
              {"text/final_norm", {h}},
              {"text/output_head", {c.unpadded_vocab_size, h}}};
    for (std::uint32_t layer = 0; layer < c.num_hidden_layers; ++layer) {
        const auto p      = "text/layers/" + std::to_string(layer) + "/";
        const bool global = (layer + 1) % 6 == 0;
        const auto add    = [&](const std::string& leaf, artifact::Shape shape) {
            out.emplace_back(p + leaf, std::move(shape));
        };
        if (layer > 0) { add("attention_queries", {global ? 3U : 1U, h}); }
        if (global) {
            for (const auto* norm : {"query_input_norm", "key_input_norm", "value_input_norm"}) {
                add(norm, {h});
            }
            for (const auto* role : {"key", "value"}) {
                add(std::string("attention/") + role + "_conv_static", {k, nkv * d});
                add(std::string("attention/") + role + "_conv_projection", {nkv * k, h});
            }
        } else {
            add("input_norm", {h});
            add("attention/key_conv", {nkv * d, k});
            add("attention/value_conv", {nkv * d, k});
        }
        add("attention/query", {nh * d, h});
        add("attention/gate", {nh * d, h});
        add("attention/relative", {nh * dr, h});
        add("attention/sink", {nh, h});
        add("attention/key", {nkv * d, h});
        add("attention/value", {nkv * d, h});
        add("attention/output", {h, nh * d});
        add("attention/query_scale", {nh, d});
        add("attention/relative_scale", {nh, dr});
        add("attention/relative_profile", {dr, global ? c.relative_extent : c.sliding_window});
        add("post_attention_conv", {h, k});
        add("ffn_queries", {1, h});
        add("ffn_norm", {h});
        add("moe/router", {c.num_experts, h});
        add("moe/router_bias", {c.num_experts});
        for (std::uint32_t e = 0; e < c.num_experts; ++e) {
            add("moe/experts/" + std::to_string(e) + "/gate_up", {2 * f, h});
            add("moe/experts/" + std::to_string(e) + "/down", {h, f});
        }
        add("post_ffn_conv", {h, k});
        if (layer >= 6 && layer < 18) {
            add("message_queries", {4, h});
            for (const auto* role : {"query", "key", "value", "ffn"}) {
                const auto m = std::string("message_gate/") + role + "/";
                add(m + "down", {r, 2 * h});
                add(m + "forget", {h, r});
                add(m + "gain", {h, r});
            }
        }
    }
    return out;
}

std::size_t check_geometry(const inkling2::TextConfig& c) {
    const LoadOptions tp2{.tp = 2}, tp1{};
    std::map<std::string, std::size_t> kinds;
    const auto named = parameters(c);
    for (const auto& [name, shape] : named) {
        const auto expect = ninfer::test::inkling2::expected_placement(name, c.num_experts);
        const auto shard  = loading::logical_shard(name, shape, c, tp2);
        bool ok           = shard.axis == expect.axis && shard.device == expect.holder;
        if (name.find("/moe/experts/") != std::string::npos && expect.axis == ShardAxis::Rows) {
            // gate_up [2F, H]: rank r's gate rows [r F/2, (r+1) F/2) and the matching up rows.
            const auto f = shape.front() / 2;
            ok           = ok &&
                 shard.ranges[0] == std::vector<artifact::SliceRange>{{0, f / 2}, {f, f / 2}} &&
                 shard.ranges[1] ==
                     std::vector<artifact::SliceRange>{{f / 2, f / 2}, {f + f / 2, f / 2}};
        } else if (artifact::is_sharded(expect.axis)) {
            const auto extent = expect.axis == ShardAxis::Rows ? shape.front() : shape.back();
            ok = ok && shard.ranges[0] == std::vector<artifact::SliceRange>{{0, extent / 2}} &&
                 shard.ranges[1] == std::vector<artifact::SliceRange>{{extent / 2, extent / 2}};
        } else {
            ok = ok && shard.ranges[0].empty() && shard.ranges[1].empty();
        }
        if (!ok) { throw std::runtime_error(name + ": unexpected tensor-parallel placement"); }
        const auto whole = loading::logical_shard(name, shape, c, tp1);
        require(whole.axis == ShardAxis::Replicated && whole.ranges[0].empty(), "tp 1 is whole");
        ++kinds[expect.kind];
    }
    for (const auto& [kind, count] : kinds) {
        std::printf("  %-22s %6zu parameters\n", kind.c_str(), count);
    }
    return named.size();
}

void released() {
    const auto c = config("config_real.json");
    std::printf("released geometry, tp 2:\n");
    const auto count = check_geometry(c);
    std::printf("  %zu parameters\n", count);
    for (const int tp : {1, 2}) {
        const LoadOptions host{.tp = tp, .embedding_host = true};
        const LoadOptions device{.tp = tp};
        for (const auto& [name, shape] : parameters(c)) {
            const auto shard = loading::logical_shard(name, shape, c, host);
            const auto plain = loading::logical_shard(name, shape, c, device);
            const auto want  = name == "text/token_embedding" ? ShardAxis::HostMapped : plain.axis;
            if (shard.axis != want || shard.ranges != plain.ranges) {
                throw std::runtime_error(name + ": unexpected placement with embedding_host");
            }
        }
        for (const char* codebook : {"dflash2/candidate_selector/predecessor_codebook",
                                     "dflash2/candidate_selector/successor_codebook"}) {
            const artifact::Shape shape{c.vocab_size, 256};
            require(loading::logical_shard(codebook, shape, c, host).axis == ShardAxis::HostMapped,
                    "the DFlash2 codebooks are HostMapped with embedding_host");
        }
    }
    std::printf(
        "  embedding_host: text/token_embedding and the DFlash2 codebooks HostMapped at tp 1 "
        "and 2\n");
    const auto rank = inkling2::execution::shard_text_config(c, 2);
    require(rank.num_attention_heads == 12 && rank.num_key_value_heads == 4 &&
                rank.unpadded_vocab_size == 100029 && rank.num_experts == 128 &&
                rank.hidden_size == 2304 && rank.visits.size() == 48,
            "rank config");
    const LoadOptions tp2{.tp = 2};
    rejects<std::invalid_argument>(
        [&] { (void)loading::logical_shard("text/layers/0/attention/unknown", {1}, c, tp2); },
        "a parameter without a rule");
    rejects<std::invalid_argument>(
        [&] { (void)loading::logical_shard("text/layers/24/ffn_norm", {2304}, c, tp2); },
        "a layer outside the model");
    rejects<std::invalid_argument>(
        [&] { (void)loading::logical_shard("text/final_norm", {2304}, c, LoadOptions{.tp = 3}); },
        "tensor parallelism 3");
    auto odd                = c;
    odd.num_key_value_heads = 3;
    rejects<std::invalid_argument>(
        [&] { (void)loading::logical_shard("text/layers/0/attention/key", {384, 2304}, odd, tp2); },
        "indivisible KV heads");
    rejects<std::invalid_argument>([&] { (void)inkling2::execution::shard_text_config(odd, 2); },
                                   "indivisible rank config");
    auto narrow                = c;
    narrow.expert_intermediate = 1120;
    rejects<std::invalid_argument>(
        [&] {
            (void)loading::logical_shard("text/layers/0/moe/experts/0/down", {2304, 1120}, narrow,
                                         tp2);
        },
        "an expert width without 64-column halves");
}

void synthetic() {
    const auto c = config("config_synthetic.json");
    std::printf("synthetic geometry, tp 2:\n");
    const auto count = check_geometry(c);
    std::printf("  %zu parameters\n", count);
    const auto rank = inkling2::execution::shard_text_config(c, 2);
    require(rank.num_attention_heads == 3 && rank.num_key_value_heads == 1, "synthetic rank");
}

} // namespace

int main() {
    try {
        released();
        synthetic();
        std::cout << "OK inkling2 shard map\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
