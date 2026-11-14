#pragma once

// Per-layer routed-expert usage profile of Qwen3.8-Flash-Next (48 Text layers + the MTP layer),
// shared by the offload backend's slot profile and the load plan's pinned-fallback order.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace ninfer::models::qwen3_5 {

// prod/cache (17/10): one GPU's base slots per layer at 66 slots per layer + 21 elastic columns
// (Flash-Next production, tp 2, MTP layer last), fitted greedily on the replayed routing of 30
// T186 prompts through the expert cache policy (marginal CPU experts per slot, at least 29 base
// slots a layer). Held-out (32 T194 prompts): 130.1 -> 112.9 CPU experts per decode round, 74.1 ->
// 69.1 insertions. A higher weight is a layer whose routing spreads over more experts.
inline constexpr std::array<std::int16_t, 49> kQwen4LayerSlotWeights = {
    149, 119, 114, 59, 74, 89, 99, 99, 64, 49, 79, 59, 89, 84, 59, 29, 29,
    59,  94,  59,  59, 79, 99, 44, 60, 49, 69, 54, 84, 69, 59, 29, 29, 54,
    84,  54,  44,  66, 89, 29, 29, 39, 59, 34, 49, 44, 54, 59, 109};

// Text layers the prefill may lend (streaming ring + borrowed prompt workspace: the last Text
// layers, Qwen4OffloadExperts::ring_layers / layers_below), kept page-locked first.
inline constexpr std::int32_t kQwen4PinnedTopLayers = 12;

// Exclusive tier: the first NINFER_QWEN4_EXCLUSIVE_LAYERS expert layers keep the experts the GPUs
// hold at start-up in VRAM only, without a page-locked host copy. Off (0) by default. GGUF banks.
[[nodiscard]] inline std::int32_t qwen4_exclusive_layers() noexcept {
    const char* v = std::getenv("NINFER_QWEN4_EXCLUSIVE_LAYERS");
    return v != nullptr ? std::max(std::atoi(v), 0) : 0;
}

// The base slots run above the fitted weights here (mean 79-84 against 66);
// NINFER_QWEN4_EXCLUSIVE_SCALE lets the tail follow them. Too high and the backend refuses at
// start.
[[nodiscard]] inline double qwen4_exclusive_scale() noexcept {
    const char* v = std::getenv("NINFER_QWEN4_EXCLUSIVE_SCALE");
    return v != nullptr ? std::atof(v) : 1.0;
}

// Experts of a layer without a host slot, split evenly between two GPUs: the highest ids, as many
// as one GPU's base slots at the fitted profile, on both. The base slots at load are not known yet
// (startup.cpp sizes them); the backend refuses at start if they are fewer.
[[nodiscard]] inline std::int32_t qwen4_exclusive_tail(std::int32_t layer) noexcept {
    return layer >= 0 && layer < qwen4_exclusive_layers() &&
                   layer < static_cast<std::int32_t>(kQwen4LayerSlotWeights.size())
               ? 2 * (static_cast<std::int32_t>(
                          kQwen4LayerSlotWeights[static_cast<std::size_t>(layer)] *
                          qwen4_exclusive_scale()) &
                      ~1)
               : 0;
}

// Host slots beyond the non-exclusive experts: the swaps one layer can have in flight.
[[nodiscard]] inline std::int32_t qwen4_exclusive_spare() noexcept {
    const char* v = std::getenv("NINFER_QWEN4_EXCLUSIVE_SPARE");
    return v != nullptr ? std::max(std::atoi(v), 1) : 2;
}

// v0.7 pinned fallback: the order in which the expert banks of the expert layers [0, layers)
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
