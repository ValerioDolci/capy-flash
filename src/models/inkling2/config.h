#pragma once

#include "artifact/schema.h"
#include "models/inkling2/draft/config.h"
#include "models/load_options.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace ninfer::models::inkling2 {

// Text component names an Inkling 2 artifact carries (tools/convert/inkling2.py). Nothing else
// selects this implementation; the Qwen registry does not know them.
inline constexpr std::string_view kTextArchitecture = "Inkling2LoopForCausalLM";
inline constexpr std::string_view kTextModelType    = "inkling2_loop_text";

// Sliding-window attention with static K/V convolutions (Local), or full attention with dynamic
// K/V convolutions and three attention-input queries (Global).
enum class LayerKind : std::uint8_t { Local, Global };

// One execution of a physical layer. The forward runs the visits in order: the prelude, the core
// layers once per pass, then the coda. Passes after the first reuse the core layers' weights with
// cross-pass messages and their own state slots; `slot` indexes the per-visit state (K/V and the
// four convolution tails) and does not follow the execution order.
struct Visit {
    std::uint32_t layer = 0;
    std::uint32_t pass  = 1; // 1-based; layers outside the core run in pass 1
    std::uint32_t slot  = 0;
    LayerKind kind      = LayerKind::Local;
    bool messages       = false; // core layer in a pass after the first
    bool block_boundary = false; // layer % block_layers == 0: the input becomes a block source
};

struct TextConfig {
    std::uint32_t hidden_size             = 0;
    std::uint32_t vocab_size              = 0; // embedding rows (padded)
    std::uint32_t unpadded_vocab_size     = 0; // output head rows = logits
    std::uint32_t num_hidden_layers       = 0;
    std::uint32_t max_position_embeddings = 0;
    float rms_norm_eps                    = 0;

    // Attention: the same head geometry in every layer.
    std::uint32_t num_attention_heads = 0;
    std::uint32_t num_key_value_heads = 0;
    std::uint32_t head_dim            = 0;
    std::uint32_t relative_dim        = 0; // d_rel: relative-state width per query head
    std::uint32_t relative_extent     = 0; // global layers' relative-bias distance
    std::uint32_t sliding_window      = 0; // local window and local relative-bias distance
    float qk_clip_threshold           = 0;
    float relative_clip_threshold     = 0;
    std::uint32_t log_scaling_floor   = 0;
    float log_scaling_alpha           = 0;
    std::uint32_t conv_kernel         = 0; // short convolution taps

    // Mixture of experts: sigmoid router with a selection bias, no shared expert.
    std::uint32_t num_experts         = 0;
    std::uint32_t experts_per_token   = 0;
    std::uint32_t expert_intermediate = 0;
    float route_scale                 = 0;
    float gate_softcap                = 0;
    float up_softcap                  = 0;
    float final_logit_softcap         = 0;

    // Block attention residuals and the looped core [core_start, core_end).
    std::uint32_t block_layers = 0;
    std::uint32_t core_start   = 0;
    std::uint32_t core_end     = 0;
    std::uint32_t num_passes   = 0;
    std::uint32_t gate_rank    = 0; // message gate rank

    std::vector<LayerKind> layer_types; // per physical layer
    std::vector<Visit> visits;          // execution order

    [[nodiscard]] std::uint64_t query_width() const noexcept {
        return std::uint64_t(num_attention_heads) * head_dim;
    }

    [[nodiscard]] std::uint64_t key_width() const noexcept {
        return std::uint64_t(num_key_value_heads) * head_dim;
    }

    [[nodiscard]] std::uint64_t relative_width() const noexcept {
        return std::uint64_t(num_attention_heads) * relative_dim;
    }

    [[nodiscard]] bool global(std::uint32_t layer) const {
        return layer_types.at(layer) == LayerKind::Global;
    }

    [[nodiscard]] bool core(std::uint32_t layer) const noexcept {
        return layer >= core_start && layer < core_end;
    }

    [[nodiscard]] std::uint32_t core_layers() const noexcept { return core_end - core_start; }

    // Relative-bias distance of a layer: the window for local layers, rel_extent for global ones.
    [[nodiscard]] std::uint32_t relative_bias_extent(std::uint32_t layer) const {
        return global(layer) ? relative_extent : sliding_window;
    }

    // Queries of the attention-input block residual: none for layer 0, 3 (Q, K, V) for a global
    // layer, 1 otherwise.
    [[nodiscard]] std::uint32_t attention_queries(std::uint32_t layer) const {
        return layer == 0 ? 0 : global(layer) ? 3 : 1;
    }
};

struct Config {
    TextConfig text;
    std::optional<draft::DraftConfig> draft; // the `mtp` component, with the speculative backend
};

// The Text component config: the checkpoint's text_config verbatim plus `architectures`
// (kTextArchitecture). Every member is read or checked against the one value this implementation
// supports; an unknown member is an error.
[[nodiscard]] TextConfig parse_text_config(const artifact::Json& value);

// The text component of an Inkling 2 artifact and, with options.speculative = Mtp, its `mtp`
// component (the DSpark drafter). Vision and the other speculative backends are rejected.
[[nodiscard]] Config parse_config(const artifact::Directory& directory, const LoadOptions& options);

// Whether the artifact's Text component names this implementation (architecture and model_type).
[[nodiscard]] bool is_inkling2(const artifact::Directory& directory);

} // namespace ninfer::models::inkling2
