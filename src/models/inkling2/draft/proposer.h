#pragma once

// What the speculative round and the Program's execution ask of a block drafter of Inkling 2 (the
// DSpark drafter, draft/draft.h, the DFlash2 drafter, draft/dflash2.h, and the pair of both,
// draft/paired.h): its context is in arrears (the features of the columns a call licensed become
// context at the start of the next call), it proposes one block's drafts inside the round's graph,
// and it keeps an anchor image.

#include "core/tensor.h"
#include "models/inkling2/execution/forward.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::models::inkling2::draft {

class Proposer {
public:
    virtual ~Proposer()                  = default;
    Proposer()                           = default;
    Proposer(const Proposer&)            = delete;
    Proposer& operator=(const Proposer&) = delete;

    // The drafts of one block (block size - 1).
    [[nodiscard]] virtual std::uint32_t proposals() const noexcept = 0;

    // Rank r's sink for a target call of `width` columns: the call writes its four states into the
    // pending features, which the next materialization consumes.
    [[nodiscard]] virtual const execution::FeatureSink* sink(int rank, std::int32_t width) = 0;

    // Eager: the pending features of the last call (`width` columns at positions first..) become
    // context; then nothing is pending.
    virtual void append(std::int32_t first, std::int32_t width) = 0;

    // In a round (stream-ordered, capturable), every rank from its own device scalars: the pending
    // features of the last call (count = the last commit()) become context at positions
    // frontier - count .., then the block at the anchor's position `frontier` writes the first
    // `proposals` (1 .. block size - 1) proposals into drafts (I32 [block size - 1] per rank).
    // `width` is the width of the call whose features are pending.
    virtual void propose(const std::array<Tensor, 2>& anchor, const std::array<Tensor, 2>& frontier,
                         const std::array<Tensor, 2>& drafts, std::int32_t width,
                         std::int32_t proposals) = 0;

    // In a round, instead of propose() (stream-ordered, capturable): the pending features of the
    // last call become context as propose() would take them, without proposing.
    virtual void follow(const std::array<Tensor, 2>& frontier, std::int32_t width) = 0;

    // In a round after acceptance: the last call licensed `count` (I32 [1] per rank) columns.
    virtual void commit(const std::array<Tensor, 2>& count) = 0;

    // The anchor image of the context a position-ordered resume cannot rebuild, on every rank;
    // restore_image leaves nothing pending.
    virtual void save_image()    = 0;
    virtual void restore_image() = 0;
    // Nothing pending: the next round materializes no context (a new sequence or a resume).
    virtual void clear_pending() = 0;
    // Eager: the pending features of the last round (a call of `width` columns, the device's
    // pending count of them, ending at position `frontier`) become context; then none is pending.
    virtual void catch_up(std::int32_t frontier, std::int32_t width) = 0;

    // Device bytes the drafter's state and workspace hold on each rank.
    [[nodiscard]] virtual std::size_t state_bytes() const noexcept     = 0;
    [[nodiscard]] virtual std::size_t workspace_bytes() const noexcept = 0;
};

} // namespace ninfer::models::inkling2::draft
