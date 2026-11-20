#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/program.h"

#include <cstdint>

namespace ninfer::models::qwen3_5 {

inline constexpr std::uint32_t kPrefillChunkAlignment = 128;
// The widest verified MTP window minus its anchor: --draft-tokens (at most 5) or a prompt-lookup
// window (the host experts and expert slots take at most 8 tokens per call).
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 7;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 15;

} // namespace ninfer::models::qwen3_5

namespace ninfer::models::qwen3_5::detail {
using ContractAccess = RuntimeContractAccess;

[[nodiscard]] inline std::uint32_t backend_frontier_at(SpeculativeBackend backend,
                                                       std::uint32_t main_frontier) noexcept {
    if (backend == SpeculativeBackend::Mtp) { return main_frontier == 0 ? 0U : main_frontier - 1U; }
    return 0U;
}

} // namespace ninfer::models::qwen3_5::detail
