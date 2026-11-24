#pragma once

#include "artifact/materializer.h"
#include "core/device.h"
#include "models/inkling2/config.h"
#include "models/inkling2/expert_bank.h"
#include "models/inkling2/load/bindings.h"
#include "models/inkling2/weights.h"

#include <span>

namespace ninfer::models::inkling2::loading {

// Tensor parallelism divides the experts' intermediate width: rank r holds half r of every expert
// of every layer, so both ranks compute every routed expert of a token on their half and the FP32
// partials meet in the MoE exchange (load/sharding.h). At tp 1 device 0 holds both halves as two
// banks. The experts are bound Residency::HostFile (load/text.cpp) and cut here from their mapped
// artifact banks; `devices` has one context per rank. Requires F/2 to be a multiple of 64 and,
// for NVFP4, one weight divisor per expert and matrix.
[[nodiscard]] ExpertBanks cut_expert_banks(std::span<const PendingWeight> pending,
                                           const TextWeights& weights, const TextConfig& config,
                                           const artifact::MaterializedArtifact& materialized,
                                           std::span<const DeviceContext* const> devices);

} // namespace ninfer::models::inkling2::loading
