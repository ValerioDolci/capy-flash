#pragma once

#include "ninfer/types.h"
#include "runtime/contract/resources.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace ninfer::models::inkling2 {

// Context-cache identity of a prepared prompt, in the shape the common ResourceManager reads
// (RequestBasePlan::context_cache). The Inkling 2 Program resumes only from its one resident
// sequence, matched by token identity, so it plans with the empty value: no session key, no
// opportunity.

inline constexpr std::size_t kPreparedSessionKeyCapacity = kMaximumContextCacheSessionKeyBytes;

struct PreparedSessionKey {
    std::uint16_t size = 0;
    std::array<char, kPreparedSessionKeyCapacity> bytes{};

    [[nodiscard]] std::string_view view() const noexcept { return {bytes.data(), size}; }

    [[nodiscard]] friend bool operator==(const PreparedSessionKey&,
                                         const PreparedSessionKey&) noexcept = default;
};

struct PreparedCacheOpportunity {
    PromptCacheMarkerKind kind       = PromptCacheMarkerKind::SharedStablePrefix;
    SharedCandidateEvidence evidence = SharedCandidateEvidence::None;
    std::uint32_t frontier           = 0;
    std::uint32_t input_order        = 0;

    [[nodiscard]] friend bool operator==(PreparedCacheOpportunity,
                                         PreparedCacheOpportunity) noexcept = default;
};

struct PreparedContextCache {
    std::optional<PreparedSessionKey> session_key;
    runtime::RetentionClass retention = runtime::RetentionClass::RecentPrivate;
    std::vector<PreparedCacheOpportunity> opportunities;
    bool update_session_index = true;
};

} // namespace ninfer::models::inkling2
