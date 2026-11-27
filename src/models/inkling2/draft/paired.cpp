#include "models/inkling2/draft/paired.h"

#include <stdexcept>

namespace ninfer::models::inkling2::draft {

PairedProposer::PairedProposer(Proposer& proposer, Proposer& follower)
    : proposer_(proposer), follower_(follower) {
    if (&proposer == &follower || proposer.proposals() != follower.proposals()) {
        throw std::invalid_argument("Inkling 2 paired drafters: two drafters of one block size");
    }
}

const execution::FeatureSink* PairedProposer::sink(int rank, std::int32_t width) {
    const execution::FeatureSink* own   = proposer_.sink(rank, width);
    const execution::FeatureSink* other = follower_.sink(rank, width);
    if (own->features.data != other->features.data || own->visits != other->visits) {
        throw std::logic_error("Inkling 2 paired drafters do not share their target states");
    }
    return own;
}

void PairedProposer::append(std::int32_t first, std::int32_t width) {
    proposer_.append(first, width);
    follower_.append(first, width);
}

void PairedProposer::propose(const std::array<Tensor, 2>& anchor,
                             const std::array<Tensor, 2>& frontier,
                             const std::array<Tensor, 2>& drafts, std::int32_t width,
                             std::int32_t proposals) {
    proposer_.propose(anchor, frontier, drafts, width, proposals);
    follower_.follow(frontier, width);
}

void PairedProposer::follow(const std::array<Tensor, 2>& frontier, std::int32_t width) {
    proposer_.follow(frontier, width);
    follower_.follow(frontier, width);
}

void PairedProposer::commit(const std::array<Tensor, 2>& count) {
    proposer_.commit(count);
    follower_.commit(count);
}

void PairedProposer::save_image() {
    proposer_.save_image();
    follower_.save_image();
}

void PairedProposer::restore_image() {
    proposer_.restore_image();
    follower_.restore_image();
}

void PairedProposer::clear_pending() {
    proposer_.clear_pending();
    follower_.clear_pending();
}

void PairedProposer::catch_up(std::int32_t frontier, std::int32_t width) {
    proposer_.catch_up(frontier, width);
    follower_.catch_up(frontier, width);
}

} // namespace ninfer::models::inkling2::draft
