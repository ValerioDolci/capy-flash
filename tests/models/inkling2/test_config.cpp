// Inkling 2 Text config (models/inkling2/config.h), host only.
//
// 1. The released checkpoint's config.json and the synthetic reference checkpoint's, as the
//    converter stores them (text_config plus the Text architecture): every dimension, and the
//    48-visit schedule against an independent table (layer, pass, state slot, kind, messages).
// 2. Rejections: unknown members, unsupported fixed values, a layer schedule or looped core this
//    implementation does not run, options other than the Text backbone; the DFlash2 drafter's
//    component config and its rejections.
// 3. The Qwen path does not accept the Inkling Text component.
// 4. The loop contrast (with_loop_contrast): a zero strength is the model's own schedule, Hidden
//    keeps it, Logits appends the reference coda on six slots after the others; options outside
//    the domain are refused.

#include "artifact/fixture.h"
#include "models/inkling2/config.h"
#include "models/qwen3_5/config.h"

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

namespace {

using namespace ninfer;
using namespace ninfer::models;
using artifact::Json;
using ninfer::test::artifact_fixture::rejects;
using ninfer::test::artifact_fixture::require;

Json component(const char* file) {
    std::ifstream stream(std::string(NINFER_SOURCE_DIR "/tests/models/inkling2/") + file);
    if (!stream) { throw std::runtime_error(std::string("cannot read ") + file); }
    auto text             = Json::parse(stream).at("text_config");
    text["architectures"] = Json::array({"Inkling2LoopForCausalLM"});
    return text;
}

// The reference's visit order (prelude, core once per pass, coda) written out by hand for the
// 24-layer, core [6, 18) x 3 schedule: {layer, pass, slot}.
struct Expected {
    std::uint32_t layer, pass, slot;
};

std::vector<Expected> expected_visits() {
    std::vector<Expected> out;
    for (std::uint32_t i = 0; i < 18; ++i) { out.push_back({i, 1, i}); }
    for (std::uint32_t i = 0; i < 12; ++i) { out.push_back({6 + i, 2, 24 + i}); }
    for (std::uint32_t i = 0; i < 12; ++i) { out.push_back({6 + i, 3, 36 + i}); }
    for (std::uint32_t i = 0; i < 6; ++i) { out.push_back({18 + i, 1, 18 + i}); }
    return out;
}

void check_schedule(const inkling2::TextConfig& c) {
    require(c.num_hidden_layers == 24 && c.layer_types.size() == 24, "layer count");
    const std::set<std::uint32_t> globals{5, 11, 17, 23};
    for (std::uint32_t layer = 0; layer < 24; ++layer) {
        require(c.global(layer) == globals.contains(layer), "layer kind");
        require(c.core(layer) == (layer >= 6 && layer < 18), "core layers");
        require(c.attention_queries(layer) == (layer == 0                ? 0U
                                               : globals.contains(layer) ? 3U
                                                                         : 1U),
                "attention-input queries");
    }
    require(c.block_layers == 6 && c.core_start == 6 && c.core_end == 18 && c.num_passes == 3,
            "loop geometry");
    const auto expected = expected_visits();
    require(c.visits.size() == 48 && expected.size() == 48, "48 visits");
    std::set<std::uint32_t> slots, global_slots;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto& v = c.visits[i];
        const auto& e = expected[i];
        if (v.layer != e.layer || v.pass != e.pass || v.slot != e.slot) {
            throw std::runtime_error("visit " + std::to_string(i) + " differs from the schedule");
        }
        require((v.kind == inkling2::LayerKind::Global) == globals.contains(v.layer), "visit kind");
        require(v.messages == (i >= 18 && i < 42), "visit messages");
        require(v.block_boundary == (v.layer % 6 == 0), "visit block boundary");
        slots.insert(v.slot);
        if (v.kind == inkling2::LayerKind::Global) { global_slots.insert(v.slot); }
    }
    require(slots.size() == 48 && *slots.rbegin() == 47, "slots are a permutation of 0..47");
    require(global_slots == std::set<std::uint32_t>{5, 11, 17, 23, 29, 35, 41, 47},
            "global visit slots");
}

void released() {
    const auto c = inkling2::parse_text_config(component("config_real.json"));
    require(c.hidden_size == 2304 && c.vocab_size == 201024 && c.unpadded_vocab_size == 200058,
            "vocabulary and hidden size");
    require(c.max_position_embeddings == 1048576, "positions");
    require(c.num_attention_heads == 24 && c.num_key_value_heads == 8 && c.head_dim == 128,
            "attention heads");
    require(c.query_width() == 3072 && c.key_width() == 1024 && c.relative_width() == 384,
            "projection widths");
    require(c.relative_dim == 16 && c.relative_extent == 1024 && c.sliding_window == 1024,
            "relative bias and window");
    require(c.relative_bias_extent(0) == 1024 && c.relative_bias_extent(5) == 1024,
            "relative extent per layer kind");
    require(c.conv_kernel == 4 && c.gate_rank == 64, "convolution taps and gate rank");
    require(c.num_experts == 128 && c.experts_per_token == 8 && c.expert_intermediate == 1152,
            "experts");
    require(c.route_scale == 8.0F && c.gate_softcap == 4.0F && c.up_softcap == 25.0F &&
                c.final_logit_softcap == 30.0F,
            "routing scale and softcaps");
    require(c.qk_clip_threshold == static_cast<float>(22.627416997969522) &&
                c.relative_clip_threshold == 8.0F,
            "clip thresholds");
    require(c.log_scaling_floor == 8192 && c.log_scaling_alpha == 0.1F, "log scaling");
    require(c.rms_norm_eps == 1e-6F, "norm epsilon");
    check_schedule(c);
}

void synthetic() {
    const auto c = inkling2::parse_text_config(component("config_synthetic.json"));
    require(c.hidden_size == 512 && c.num_attention_heads == 6 && c.num_key_value_heads == 2 &&
                c.head_dim == 128,
            "synthetic attention");
    require(c.sliding_window == 64 && c.relative_extent == 96, "synthetic window and extent");
    require(c.relative_bias_extent(4) == 64 && c.relative_bias_extent(11) == 96,
            "synthetic relative extent per layer kind");
    require(c.log_scaling_floor == 128, "synthetic log floor");
    require(c.num_experts == 16 && c.experts_per_token == 16 && c.expert_intermediate == 256,
            "synthetic experts");
    require(c.vocab_size == 201024 && c.unpadded_vocab_size == 200058, "synthetic vocabulary");
    check_schedule(c);
}

void rejections() {
    const auto base   = component("config_real.json");
    const auto broken = [&](auto edit, const char* message) {
        auto value = base;
        edit(value);
        rejects([&] { (void)inkling2::parse_text_config(value); }, message);
    };
    broken([](Json& v) { v["mtp_config"] = Json::object(); }, "unknown member accepted");
    broken([](Json& v) { v["architectures"] = {"Qwen3_5ForCausalLM"}; }, "foreign architecture");
    broken([](Json& v) { v["model_type"] = "inkling2_text"; }, "foreign model_type");
    broken([](Json& v) { v["q_bias"] = true; }, "query bias accepted");
    broken([](Json& v) { v["n_shared_experts"] = 1; }, "shared expert accepted");
    broken([](Json& v) { v["swa_num_key_value_heads"] = 4; }, "different window heads accepted");
    broken([](Json& v) { v["local_layer_ids"] = {0, 1, 2}; }, "local layers differ");
    broken([](Json& v) { v["layer_types"] = Json::array(); }, "layer types differ");
    broken([](Json& v) { v["loop_config"]["core_start"] = 4; }, "core inside a block accepted");
    broken([](Json& v) { v["loop_config"]["core_end"] = 30; }, "core beyond the layers accepted");
    broken([](Json& v) { v["loop_config"]["is_varloop"] = true; }, "variable loop accepted");
    broken([](Json& v) { v["num_key_value_heads"] = 7; }, "indivisible KV heads accepted");
    broken([](Json& v) { v["unpadded_vocab_size"] = 300000; }, "unpadded beyond padded");
    broken([](Json& v) { v.erase("rel_extent"); }, "missing member accepted");

    artifact::Directory directory;
    directory.components["text"].config = base;
    require(inkling2::is_inkling2(directory), "the Inkling component is recognized");
    require(inkling2::parse_config(directory, {}).text.visits.size() == 48, "Text backbone");
    rejects([&] { (void)inkling2::parse_config(directory, {.vision = true}); }, "Vision accepted");
    require(!inkling2::parse_config(directory, {.speculative = SpeculativeBackend::Mtp}).draft,
            "MTP without the mtp component: no drafter");
    require(
        !inkling2::parse_config(directory, {.speculative = SpeculativeBackend::DFlash2}).dflash2,
        "DFlash2 without the dflash2 component: no drafter");
    // The DFlash2 component (incoai/Inkling-2-Mini-DFlash2 as the converter stores it).
    const Json dflash2                   = {{"architectures", {"DFlash2DraftModel"}},
                                            {"model_type", "inkling2_dflash2"},
                                            {"hidden_size", 2304},
                                            {"intermediate_size", 6912},
                                            {"num_attention_heads", 32},
                                            {"num_key_value_heads", 8},
                                            {"head_dim", 128},
                                            {"num_hidden_layers", 5},
                                            {"sliding_window", 2048},
                                            {"rms_norm_eps", 1e-6},
                                            {"vocab_size", 201024},
                                            {"rope_theta", 10000.0},
                                            {"dflash_config",
                                             {{"block_size", 8},
                                              {"conv_group_size", 16},
                                              {"conv_kernel_size", 2},
                                              {"final_logit_softcapping", 30.0},
                                              {"mask_token_id", 200056},
                                              {"selector_rank", 256},
                                              {"selector_softcap_mode", "after_correction"},
                                              {"selector_top_k", 16},
                                              {"target_layer_ids", {5, 23, 41, 48}}}}};
    artifact::Directory drafted          = directory;
    drafted.components["dflash2"].config = dflash2;
    const auto read = inkling2::parse_config(drafted, {.speculative = SpeculativeBackend::DFlash2});
    require(read.dflash2 && read.dflash2->proposals() == 7 && read.dflash2->num_layers == 5 &&
                read.dflash2->conv_coefficients() == 576 && read.dflash2->mask_token_id == 200056 &&
                read.dflash2->feature_visits == std::array<std::uint32_t, 3>{5, 23, 41},
            "DFlash2 config");
    const auto broken_dflash2 = [&](auto edit, const char* label) {
        artifact::Directory d = drafted;
        edit(d.components["dflash2"].config);
        rejects(
            [&] { (void)inkling2::parse_config(d, {.speculative = SpeculativeBackend::DFlash2}); },
            label);
    };
    broken_dflash2(
        [](Json& v) { v["dflash_config"]["selector_softcap_mode"] = "before_correction"; },
        "softcap before the correction accepted");
    broken_dflash2([](Json& v) { v["dflash_config"]["conv_kernel_size"] = 3; },
                   "three taps accepted");
    broken_dflash2([](Json& v) { v["dflash_config"]["target_layer_ids"] = {5, 23, 41, 47}; },
                   "no head-input sentinel accepted");
    broken_dflash2([](Json& v) { v["hidden_size"] = 2048; }, "foreign hidden size accepted");
    // The Qwen path refuses the component: it is never read as a Qwen model.
    try {
        (void)qwen3_5::parse_config(directory, {});
        throw std::runtime_error("the Qwen parser accepted an Inkling 2 text component");
    } catch (const artifact::ArtifactError& error) {
        std::cout << "Qwen path rejects the Inkling component: " << error.what() << '\n';
    }
    artifact::Directory qwen;
    qwen.components["text"].config = {{"architectures", {"Qwen3_5ForCausalLM"}},
                                      {"model_type", "qwen3_5_text"}};
    require(!inkling2::is_inkling2(qwen), "a Qwen component is not Inkling");
}

void loop_contrast() {
    const auto base = inkling2::parse_text_config(component("config_real.json"));
    const auto with = [&](const LoopContrastOptions& options) {
        return inkling2::with_loop_contrast(base, options);
    };
    const auto zero = with({.mode = LoopContrast::Logits, .strength = 0.0F, .adaptive = true});
    require(zero.contrast == LoopContrastOptions{}, "a zero strength is not off");
    check_schedule(zero);
    const auto hidden = with({.mode = LoopContrast::Hidden, .strength = 0.3F});
    require(hidden.contrast.mode == LoopContrast::Hidden, "Hidden not applied");
    check_schedule(hidden);

    const auto logits = with({.mode = LoopContrast::Logits, .strength = 0.5F, .adaptive = true});
    require(logits.contrast.adaptive && logits.visits.size() == 54, "54 visits with Logits");
    for (std::size_t i = 0; i < 48; ++i) {
        const auto& v = logits.visits[i];
        require(!v.reference && v.layer == base.visits[i].layer && v.pass == base.visits[i].pass &&
                    v.slot == base.visits[i].slot,
                "the model's visits change with Logits");
    }
    for (std::uint32_t i = 0; i < 6; ++i) {
        const auto& v = logits.visits[48 + i];
        require(v.reference && v.layer == 18 + i && v.pass == 1 && v.slot == 48 + i &&
                    !v.messages && v.block_boundary == (i == 0) &&
                    (v.kind == inkling2::LayerKind::Global) == (i == 5),
                "reference coda visit");
    }

    rejects(
        [&] { (void)with({.mode = LoopContrast::Logits, .strength = 0.5F, .reference_pass = 3}); },
        "the last pass accepted as the reference");
    rejects(
        [&] { (void)with({.mode = LoopContrast::Logits, .strength = 0.5F, .reference_pass = 0}); },
        "pass 0 accepted as the reference");
    rejects([&] { (void)with({.mode = LoopContrast::Hidden, .strength = -0.3F}); },
            "a negative strength accepted");
    rejects([&] { (void)with({.mode = LoopContrast::Hidden, .strength = std::nanf("")}); },
            "a NaN strength accepted");
    rejects([&] { (void)with({.mode = LoopContrast::Hidden, .strength = 0.3F, .adaptive = true}); },
            "adaptive Hidden accepted");
}

} // namespace

int main() {
    try {
        released();
        synthetic();
        rejections();
        loop_contrast();
        std::cout << "OK inkling2 config\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
