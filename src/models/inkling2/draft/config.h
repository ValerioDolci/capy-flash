#pragma once

// Configuration of the DSpark drafter of Inkling 2 (the artifact's `mtp` component, the
// checkpoint's `mtp_config`). The drafter reads four states of the Text forward at every committed
// position (the inputs of three visits and the head input), mixes them into its context K/V, and
// proposes a block of `block_size - 1` tokens from one forward of three pre-norm layers over the
// block [anchor, mask x (block_size - 1)] and a recurrent head chained over the proposals.

#include "artifact/schema.h"

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

namespace ninfer::models::inkling2 {
struct TextConfig;
} // namespace ninfer::models::inkling2

namespace ninfer::models::inkling2::draft {

inline constexpr std::string_view kArchitecture = "Inkling2DSpark";
inline constexpr std::string_view kModelType    = "inkling2_dspark";

// The states the drafter reads (target_layer_ids): three visit inputs, then the head input.
inline constexpr std::size_t kFeatures = 4;

struct DraftConfig {
    std::uint32_t hidden_size         = 0;
    std::uint32_t num_layers          = 0;
    std::uint32_t num_attention_heads = 0;
    std::uint32_t num_key_value_heads = 0;
    std::uint32_t head_dim            = 0;
    std::uint32_t intermediate_size   = 0;
    std::uint32_t sliding_window      = 0; // local layers' keys: [P - window, P + block)
    std::uint32_t block_size          = 0; // anchor + proposals
    std::uint32_t markov_rank         = 0;
    float rms_norm_eps                = 0;
    float rope_theta                  = 0;
    std::vector<bool> local;                       // per layer: windowed keys
    std::array<std::uint32_t, 3> feature_visits{}; // execution indices whose inputs are read

    [[nodiscard]] std::uint32_t proposals() const noexcept { return block_size - 1; }

    [[nodiscard]] std::uint64_t query_width() const noexcept {
        return std::uint64_t(num_attention_heads) * head_dim;
    }

    [[nodiscard]] std::uint64_t key_width() const noexcept {
        return std::uint64_t(num_key_value_heads) * head_dim;
    }
};

// The `mtp` component config: the checkpoint's mtp_config plus `architectures` (kArchitecture) and
// `model_type` (kModelType). Every member is read or checked against the one supported value; the
// hidden size must be the Text component's, target_layer_ids three visits of its schedule followed
// by the head-input sentinel (the visit count).
[[nodiscard]] DraftConfig parse_draft_config(const artifact::Json& value, const TextConfig& text);

// The DFlash2 block drafter (the artifact's `dflash2` component, the drafter checkpoint's
// config.json). It reads the same four states as the DSpark drafter and projects them into one
// context vector per position (feature projection, context norm), whose keys and values every layer
// keeps over a sliding window; it proposes `block_size - 1` tokens from one non-causal forward of
// `num_layers` layers (grouped dynamic convolutions around attention and MLP) over the block
// [anchor, mask x (block_size - 1)], whose last hidden states score the output head's top
// candidates and walk them with the candidate selector.
inline constexpr std::string_view kDFlash2Architecture = "DFlash2DraftModel";
inline constexpr std::string_view kDFlash2ModelType    = "inkling2_dflash2";

struct DFlash2Config {
    std::uint32_t hidden_size         = 0;
    std::uint32_t num_layers          = 0;
    std::uint32_t num_attention_heads = 0;
    std::uint32_t num_key_value_heads = 0;
    std::uint32_t head_dim            = 0;
    std::uint32_t intermediate_size   = 0;
    std::uint32_t sliding_window      = 0; // a query at q sees the keys at p with q - p < window
    std::uint32_t block_size          = 0; // anchor + proposals
    std::uint32_t mask_token_id       = 0;
    std::uint32_t conv_taps           = 0;
    std::uint32_t conv_group_size     = 0;
    std::uint32_t selector_rank       = 0;
    std::uint32_t selector_top_k      = 0;
    float rms_norm_eps                = 0;
    float rope_theta                  = 0;
    // The selector's scores are soft-capped after its correction (selector_softcap_mode
    // after_correction): a monotone map, so the greedy walk does not depend on it.
    float selector_softcap = 0;
    std::array<std::uint32_t, 3> feature_visits{}; // execution indices whose inputs are read

    [[nodiscard]] std::uint32_t proposals() const noexcept { return block_size - 1; }

    [[nodiscard]] std::uint64_t query_width() const noexcept {
        return std::uint64_t(num_attention_heads) * head_dim;
    }

    [[nodiscard]] std::uint64_t key_width() const noexcept {
        return std::uint64_t(num_key_value_heads) * head_dim;
    }

    // Rows of a convolution's kernel projection: (side, tap, group) for the input and output sides.
    [[nodiscard]] std::uint64_t conv_coefficients() const noexcept {
        return std::uint64_t(2) * conv_taps * (hidden_size / conv_group_size);
    }
};

// The `dflash2` component config. Every member is read or checked against the one supported
// value: hidden size and vocabulary the Text component's, target_layer_ids as for the DSpark
// drafter, two convolution taps in groups of 16, a selector of rank 256 over 16 candidates with the
// Text component's soft cap after the correction, sliding attention on every layer.
[[nodiscard]] DFlash2Config parse_dflash2_config(const artifact::Json& value,
                                                 const TextConfig& text);

} // namespace ninfer::models::inkling2::draft
