#pragma once

#include "ninfer/types.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash2") { return SpeculativeBackend::DFlash2; }
    if (value == "off" || value == "none") { return SpeculativeBackend::None; }
    throw std::invalid_argument("invalid speculative backend: " + std::string(value));
}

// `--spec`. off also drops a draft window and proposal head set before it (by a profile).
inline void set_speculative_backend(SpeculativeOptions& options, std::string_view value) {
    options.backend = parse_speculative_backend(value);
    if (options.backend == SpeculativeBackend::None) {
        options.draft_tokens  = 0;
        options.proposal_head = ProposalHead::Full;
    }
}

[[nodiscard]] inline const char* speculative_backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash2:
        return "dflash2";
    }
    return "unknown";
}

inline void validate_speculative_cli_options(const SpeculativeOptions& options) {
    switch (options.backend) {
    case SpeculativeBackend::None:
        if (options.draft_tokens != 0 || options.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "--draft-tokens and --lm-head-draft require --spec mtp|dflash2");
        }
        if (options.prompt_lookup) {
            throw std::invalid_argument("--prompt-lookup requires --spec mtp");
        }
        if (options.adaptive_drafts != 0) {
            throw std::invalid_argument("--adaptive-draft requires --spec mtp or --spec dflash2");
        }
        return;
    case SpeculativeBackend::Mtp:
        if (options.draft_tokens == 0 || options.draft_tokens > 5) {
            throw std::invalid_argument("--spec mtp requires --draft-tokens in [1,5]");
        }
        if (options.adaptive_drafts > 7) {
            throw std::invalid_argument("--adaptive-draft takes 1..7 drafts");
        }
        if (options.adaptive_drafts != 0 && options.adaptive_drafts < options.draft_tokens) {
            throw std::invalid_argument("--adaptive-draft must be at least --draft-tokens");
        }
        return;
    case SpeculativeBackend::DFlash2:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash2 requires --draft-tokens in [1,15]");
        }
        if (options.prompt_lookup) {
            throw std::invalid_argument("--prompt-lookup requires --spec mtp");
        }
        if (options.adaptive_drafts > 15) {
            throw std::invalid_argument("--adaptive-draft with --spec dflash2 takes 1..15 drafts");
        }
        if (options.adaptive_drafts != 0 && options.adaptive_drafts < options.draft_tokens) {
            throw std::invalid_argument("--adaptive-draft must be at least --draft-tokens");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}

} // namespace ninfer::product
