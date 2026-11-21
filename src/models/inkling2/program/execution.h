#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer {
struct ExecutionContext;
}

namespace ninfer::models::inkling2 {

struct TpTransportStatus;

namespace execution {
class Parameters;
}

namespace detail {

struct SequenceLayout;

// The model execution the Program drives: per-lane visit state, the forward over prompt chunks and
// single decode tokens on every rank, and token selection from the rank-0 logits. Lanes are
// Program lanes; positions are absolute token positions of the lane's sequence.
class ProgramExecution {
public:
    virtual ~ProgramExecution() = default;

    // Clears lane `lane`'s visit state (KV, rings, convolution tails) for a new sequence.
    virtual void reset(std::uint32_t lane) = 0;
    // The anchor image of lane `lane`: the state that a position-ordered resume cannot rebuild
    // from the global K/V (the local rings and the convolution tails), copied on every rank.
    // restore_image puts it back; the global K/V stay where they are, since positions at or after
    // the image's frontier are written before they are read.
    virtual void save_image(std::uint32_t lane)    = 0;
    virtual void restore_image(std::uint32_t lane) = 0;
    // Runs tokens [position, position + tokens.size()) of the lane through the forward. With
    // `select`, returns the token chosen from the logits of the last one.
    [[nodiscard]] virtual std::optional<TokenId>
    prefill(std::uint32_t lane, std::span<const TokenId> tokens, std::uint32_t position,
            const ResolvedSamplingParameters* select) = 0;
    // Runs the lane's committed tokens not yet executed (the last generated one, after forced
    // control tokens all of them) from `position` and returns the token chosen from the logits of
    // the last; `generated` are the lane's committed output tokens (the penalties' counts).
    [[nodiscard]] virtual TokenId decode(std::uint32_t lane, std::span<const TokenId> tokens,
                                         std::uint32_t position,
                                         const ResolvedSamplingParameters& select,
                                         std::span<const TokenId> generated) = 0;
    // The drafts per speculative round, 0 without a drafter.
    [[nodiscard]] virtual std::uint32_t draft_tokens() const noexcept = 0;
    // One speculative round: `anchor` (the last committed token, not yet executed) at position
    // `frontier` and `extent` (0 .. draft_tokens()) of the drafter's proposals verified in one
    // call; returns the licensed tokens (accepted proposals, then the correction or bonus), all
    // executed except the last, which is the next anchor. The drafter's context follows. A greedy
    // `select` accepts on the device (the argmax prefix); otherwise every column j in turn takes
    // the token the decode step's Host sampler draws at position frontier + j, with `generated`
    // and the tokens taken before it as the penalties' history, and the round stops at the first
    // token that is not the column's proposal: each token is the one an ordinary decode step
    // draws there (the same logits, history and draw).
    [[nodiscard]] virtual std::vector<TokenId>
    speculate(std::uint32_t lane, TokenId anchor, std::uint32_t frontier, std::uint32_t extent,
              const ResolvedSamplingParameters& select, std::span<const TokenId> generated) = 0;
    // After speculate(), when only the first `committed` licensed tokens are kept: the state
    // holds frontier + committed positions instead.
    virtual void settle(std::uint32_t lane, std::uint32_t committed) = 0;
    // log p(tokens[i] | tokens[0..i)) for i in [first_target, tokens.size()), on lane 0.
    [[nodiscard]] virtual std::vector<float> score(std::span<const TokenId> tokens,
                                                   std::uint32_t first_target) = 0;
    [[nodiscard]] virtual MemorySummary memory_summary() const                 = 0;
    virtual void reset_memory_peaks() noexcept                                 = 0;
    // The tp 2 all-reduces' transport (captured decode and eager calls); empty at tp 1.
    [[nodiscard]] virtual TpTransportStatus tp_transport() const = 0;
    // Lane `lane`'s visit state on every rank followed by the FP32 soft-capped logits of the last
    // column of the last call that wrote them, as bytes: two executions of the same calls compare
    // byte for byte (tests/models/inkling2/test_decode_graph.cpp).
    [[nodiscard]] virtual std::vector<std::uint8_t> snapshot(std::uint32_t lane) const = 0;
};

// The execution of `layout` over `parameters` (rank 0) and `peer_parameters` (rank 1 at tp 2,
// otherwise null). It warms every kernel up and, with CUDA Graphs, captures the decode step
// before it returns.
[[nodiscard]] std::unique_ptr<ProgramExecution>
make_program_execution(const execution::Parameters& parameters,
                       const execution::Parameters* peer_parameters, const SequenceLayout& layout,
                       ExecutionContext& execution);

// Per-rank startup layout of `max_context` and `max_concurrency` lanes over `parameters` (rank 0)
// and `peer_parameters` (rank 1 at tp 2, otherwise null).
[[nodiscard]] SequenceLayout plan_sequence_layout(const execution::Parameters& parameters,
                                                  const execution::Parameters* peer_parameters,
                                                  const EngineOptions& options);

} // namespace detail
} // namespace ninfer::models::inkling2
