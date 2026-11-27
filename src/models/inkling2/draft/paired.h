#pragma once

// The artifact's two drafters over one sequence (--spec auto): one proposes, the other's context
// follows every call, so that either can propose the next round. Both read the same target
// states from one pending buffer (the DFlash2 drafter shares the DSpark drafter's,
// draft/dflash2.h), which the forward writes once. A PairedProposer is one assignment of the roles:
// the execution holds one per drafter (and one verification round each) and the Program picks which
// runs.

#include "models/inkling2/draft/proposer.h"

namespace ninfer::models::inkling2::draft {

class PairedProposer final : public Proposer {
public:
    // `proposer` proposes the drafts; `follower`'s context follows. Both share their pending
    // features and serve the same sequence; neither is owned.
    PairedProposer(Proposer& proposer, Proposer& follower);

    [[nodiscard]] std::uint32_t proposals() const noexcept override {
        return proposer_.proposals();
    }

    [[nodiscard]] const execution::FeatureSink* sink(int rank, std::int32_t width) override;
    void append(std::int32_t first, std::int32_t width) override;
    // The proposer proposes, then the follower takes the same pending features as context.
    void propose(const std::array<Tensor, 2>& anchor, const std::array<Tensor, 2>& frontier,
                 const std::array<Tensor, 2>& drafts, std::int32_t width,
                 std::int32_t proposals) override;
    void follow(const std::array<Tensor, 2>& frontier, std::int32_t width) override;
    void commit(const std::array<Tensor, 2>& count) override;
    void save_image() override;
    void restore_image() override;
    void clear_pending() override;
    void catch_up(std::int32_t frontier, std::int32_t width) override;

    [[nodiscard]] std::size_t state_bytes() const noexcept override {
        return proposer_.state_bytes() + follower_.state_bytes();
    }

    [[nodiscard]] std::size_t workspace_bytes() const noexcept override {
        return proposer_.workspace_bytes() + follower_.workspace_bytes();
    }

private:
    Proposer& proposer_;
    Proposer& follower_;
};

} // namespace ninfer::models::inkling2::draft
