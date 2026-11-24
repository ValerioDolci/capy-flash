#pragma once

// One speculative verification round of Inkling 2 over one sequence, at one device or at
// tensor-parallel width 2, replayed from one CUDA Graph that spans both devices (the form of the
// Qwen DFlash2 round without its proposal):
//
//   1. one host-to-device copy of the ingress record per rank (anchor, frontier, extent P, drafts)
//   2. speculative_prepare_verify_inputs over the physical width K: ids [anchor, d1..dK] at the
//      consecutive positions frontier .. frontier + K (relative_sink_attention needs them
//      consecutive; columns past P carry drafts that acceptance never reads)
//   3. the forward at T = K + 1 with the logits of every column, its short convolutions writing
//      the rank's ConvRecord instead of advancing their tails
//   4. rank 0: the argmax of every column, then speculative_accept_greedy_drafts over the true
//      extent P: the longest draft prefix equal to the argmax, then the correction (or the bonus)
//   5. rank 1 pulls the licensed count; every rank folds its convolution tails after it
//   6. one device-to-host copy of the egress record (licensed tokens and count) on rank 0
//
// With a drafter (draft/draft.h) the round proposes its own drafts in the same graph: after the
// upload every rank's drafter materializes the context the last call licensed and writes the
// proposals over the record's drafts, the forward fills the drafter's pending features, and the
// licensed count becomes the drafter's next pending count. Without one, the record's drafts are
// verified as given.
//
// After a round the state holds exactly the committed tokens: the tails by the fold, the local
// rings because they hold window + kVisitStateMaxDrafts positions (state/visit_state.h), the
// global K/V because positions past the frontier are rewritten before they are read. The
// correction is the next round's anchor, not yet executed, as after an ordinary decode step.
// Every column's logits are those of a decode step at its position, bit for bit: the forward's
// Ops keep the route of one column up to 16 columns.
//
// The round owns its device buffers, records and graph; the caller owns the forward's operands
// (Parameters, state, workspace, streams), the transport (PeerEvents, the mailbox wide enough for
// the round's exchanges, the bridge), and launches only after the streams are idle.

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/tensor.h"
#include "models/inkling2/config.h"
#include "models/inkling2/draft/draft.h"
#include "models/inkling2/execution/forward.h"
#include "models/inkling2/state/conv_record.h"
#include "models/inkling2/state/visit_state.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/types.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>

namespace ninfer::models::inkling2::speculative {

inline constexpr std::uint32_t kMaxDrafts = kVisitStateMaxDrafts;
inline constexpr std::uint32_t kMaxWidth  = kMaxDrafts + 1;

// Host-to-device record of one round, copied whole to every rank.
struct VerifyIngress {
    TokenId anchor        = 0; // the last committed token, not yet executed
    std::int32_t frontier = 0; // its position: the tokens the state holds
    std::int32_t extent   = 0; // drafts to verify, 0 .. K
    std::array<TokenId, kMaxDrafts> drafts{};
};

// Device-to-host record of one round (rank 0).
struct VerifyEgress {
    std::array<TokenId, kMaxWidth> licensed{}; // accepted drafts, then the correction or bonus
    std::int32_t count    = 0;                 // licensed tokens: accepted + 1
    std::int32_t accepted = 0;
    std::array<TokenId, kMaxDrafts> drafts{}; // the drafts the round verified (or proposed)
};

// The bytes of the largest exchange of a round at K drafts: the MoE's FP32 all-reduce of the
// hidden state over K + 1 columns. A tp 2 mailbox carries the round's exchanges when its slots
// (wide or ordinary) are at least this wide.
[[nodiscard]] std::size_t round_exchange_bytes(const TextConfig& config, std::uint32_t drafts);

// One rank's operands of the round's forward (forward.h ForwardRank, without the per-call ones).
struct RoundRank {
    const TextConfig* config                = nullptr;
    const execution::Parameters* parameters = nullptr;
    WorkspaceArena* work                    = nullptr;
    cudaStream_t stream                     = nullptr;
    int device                              = 0;
    const VisitState* state                 = nullptr;
    const execution::PrefetchLane* prefetch = nullptr; // optional: execution/forward/prefetch.h
};

class VerifyRound {
public:
    // A round of `drafts` (1 .. kMaxDrafts) over `width` ranks. `max_tokens` and
    // `max_logit_columns` are the bounds the ranks' workspace was planned for
    // (forward_workspace_bytes); both must be at least drafts + 1. At tp 2 `bridge` orders the
    // launches of the two-device graph. `drafter` (optional) proposes the drafts in the round.
    VerifyRound(const TextConfig& config, std::array<RoundRank, 2> ranks, int width,
                execution::ForwardCollectives collectives, const DecodeGraphPeerBridge* bridge,
                std::uint32_t drafts, std::int32_t max_tokens, std::int32_t max_logit_columns,
                draft::Drafter* drafter = nullptr);
    ~VerifyRound();

    VerifyRound(const VerifyRound&)            = delete;
    VerifyRound& operator=(const VerifyRound&) = delete;

    [[nodiscard]] std::uint32_t drafts() const noexcept { return drafts_; }

    // The record the next round reads (pinned host memory).
    [[nodiscard]] VerifyIngress& ingress() noexcept;
    // The record the last round wrote, valid after launch() returns.
    [[nodiscard]] const VerifyEgress& egress() const noexcept;

    // Rank 0's BF16 logits [Vu, K + 1] of the last round (column j at position frontier + j), and
    // their FP32 soft cap (the values a decode step's Host sampler reads).
    [[nodiscard]] const Tensor& logits() const noexcept { return logits_; }

    [[nodiscard]] const Tensor& capped() const noexcept { return capped_; }

    // Runs one round from ingress() and waits for it. `graph`: replays the round's graph,
    // capturing it on first use (the positions are read on the device, so it replays at any
    // frontier); otherwise runs the same work eagerly.
    void launch(bool graph);

    // After a launch, when the Host keeps another count than the round licensed (a terminal row,
    // or the Host sampler's acceptance): the state takes the first `count` (1 .. K + 1) columns
    // instead, the convolution tails folded again from the record and the drafter's pending count
    // set to `count`.
    void recommit(std::int32_t count);

    // Nodes of the captured graph over both devices (0 before the first capture).
    [[nodiscard]] std::size_t graph_nodes() const;

private:
    struct Rank;

    void body();
    void capture();
    void synchronize() const;

    [[nodiscard]] cudaStream_t stream(int r) const { return ranks_[std::size_t(r)].stream; }

    const TextConfig& config_;
    std::array<RoundRank, 2> ranks_;
    int width_;
    execution::ForwardCollectives collectives_;
    const DecodeGraphPeerBridge* bridge_;
    std::uint32_t drafts_;
    std::int32_t max_tokens_, max_logit_columns_;
    draft::Drafter* drafter_;
    std::array<std::unique_ptr<Rank>, 2> rank_;
    std::unique_ptr<PinnedHostBuffer> host_;
    std::unique_ptr<DeviceBuffer> logits_mem_, capped_mem_, accept_mem_;
    std::unique_ptr<WorkspaceArena> accept_work_;
    Tensor logits_, capped_;
    DecodeGraphDefinition definition_;
    DecodeGraphExecutable graph_;
};

} // namespace ninfer::models::inkling2::speculative
