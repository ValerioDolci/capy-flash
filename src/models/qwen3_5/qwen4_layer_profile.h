#pragma once

// Per-layer routed-expert usage profile of Qwen3.8-Flash-Next (48 Text layers + the MTP layer),
// shared by the offload backend's slot profile and the load plan's pinned-fallback order.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

namespace ninfer::models::qwen3_5 {

// One GPU's base slots per layer at 66 slots per layer + 21 elastic columns
// (Flash-Next production, tp 2, MTP layer last), fitted greedily on the replayed routing of 30
// prompts through the expert cache policy (marginal CPU experts per slot, at least 29 base
// slots a layer); it lowers the CPU experts per decode round on held-out prompts.
// A higher weight is a layer whose routing spreads over more experts.
inline constexpr std::array<std::int16_t, 49> kQwen4LayerSlotWeights = {
    149, 119, 114, 59, 74, 89, 99, 99, 64, 49, 79, 59, 89, 84, 59, 29, 29,
    59,  94,  59,  59, 79, 99, 44, 60, 49, 69, 54, 84, 69, 59, 29, 29, 54,
    84,  54,  44,  66, 89, 29, 29, 39, 59, 34, 49, 44, 54, 59, 109};

// Text layers the prefill may lend (streaming ring + borrowed prompt workspace: the last Text
// layers, Qwen4OffloadExperts::ring_layers / layers_below), kept page-locked first.
inline constexpr std::int32_t kQwen4PinnedTopLayers = 12;

// Exclusive tier (GGUF banks): the first N expert layers keep the experts the GPUs hold at
// start-up in VRAM only, without a page-locked host copy. 0 = off. N is `option`
// (RoutedExpertOptions::exclusive_layers) when set, else NINFER_QWEN4_EXCLUSIVE_LAYERS, read once
// per process (any text but a non-negative integer is refused), else 0.
[[nodiscard]] inline std::int32_t qwen4_exclusive_layers(std::optional<std::uint32_t> option) {
    if (option) {
        if (*option > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument("exclusive expert layers: too many layers");
        }
        return static_cast<std::int32_t>(*option);
    }
    static const std::int32_t layers = [] {
        const char* v = std::getenv("NINFER_QWEN4_EXCLUSIVE_LAYERS");
        if (v == nullptr || *v == 0) { return 0; }
        char* end        = nullptr;
        const long value = std::strtol(v, &end, 10);
        if (*end != 0 || value < 0 || value > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument(
                "NINFER_QWEN4_EXCLUSIVE_LAYERS must be a number of layers (0 = off)");
        }
        return static_cast<std::int32_t>(value);
    }();
    return layers;
}

// Exclusive tier: the experts of a layer without a host copy, as a multiple of its slot weight.
// The base slots run above the fitted weights (mean 79-84 per GPU against 66), so the tail can too;
// the slot profile raises a layer whose base slots fall short (Qwen4OffloadExperts::
// layer_slot_profile), and the backend refuses at start where they still do.
inline constexpr double kQwen4ExclusiveScale = 1.4;

// Exclusive tier: host slots per layer beyond the experts that keep a host copy, i.e. the
// write-backs one layer can have in flight.
inline constexpr std::int32_t kQwen4ExclusiveSpare = 2;

// Experts of a layer without a host slot when the first `exclusive_layers` layers are exclusive
// (qwen4_exclusive_layers()): the highest ids, twice one GPU's base slots at the fitted profile
// times kQwen4ExclusiveScale (half on each GPU at tp 2). The base slots at load are not known yet
// (startup.cpp sizes them); the backend refuses at start if they are fewer.
[[nodiscard]] inline std::int32_t qwen4_exclusive_tail(std::int32_t layer,
                                                       std::int32_t exclusive_layers) noexcept {
    return layer >= 0 && layer < exclusive_layers &&
                   layer < static_cast<std::int32_t>(kQwen4LayerSlotWeights.size())
               ? 2 * (static_cast<std::int32_t>(
                          kQwen4LayerSlotWeights[static_cast<std::size_t>(layer)] *
                          kQwen4ExclusiveScale) &
                      ~1)
               : 0;
}

// Pinned fallback: the order in which the expert banks of the expert layers [0, layers)
// (Text layers [0, text_layers), then the MTP layer(s)) stay page-locked when not all fit, most
// important first: (1) the last kQwen4PinnedTopLayers Text layers, top down (a lent layer has no
// resident experts during a prompt and its slots are the streaming ring: it must stream);
// (2) the MTP layer(s) (called several times per decode round); (3) the other Text layers by
// kQwen4LayerSlotWeights descending (the layers with the most cache insertions), ties low first.
[[nodiscard]] inline std::vector<std::int32_t> qwen4_host_pin_order(std::int32_t layers,
                                                                    std::int32_t text_layers) {
    std::vector<std::int32_t> order;
    if (layers <= 0 || text_layers < 0 || text_layers > layers) { return order; }
    const std::int32_t top = std::min(kQwen4PinnedTopLayers, text_layers);
    for (std::int32_t l = text_layers - 1; l >= text_layers - top; --l) { order.push_back(l); }
    for (std::int32_t l = text_layers; l < layers; ++l) { order.push_back(l); }
    std::vector<std::int32_t> rest(static_cast<std::size_t>(text_layers - top));
    std::iota(rest.begin(), rest.end(), 0);
    const auto weight = [](std::int32_t l) {
        return l < static_cast<std::int32_t>(kQwen4LayerSlotWeights.size())
                   ? kQwen4LayerSlotWeights[static_cast<std::size_t>(l)]
                   : std::int16_t{0};
    };
    std::stable_sort(rest.begin(), rest.end(),
                     [&](std::int32_t a, std::int32_t b) { return weight(a) > weight(b); });
    order.insert(order.end(), rest.begin(), rest.end());
    return order;
}

} // namespace ninfer::models::qwen3_5
