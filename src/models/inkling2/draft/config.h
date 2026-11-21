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

} // namespace ninfer::models::inkling2::draft
