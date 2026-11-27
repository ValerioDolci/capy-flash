#pragma once

// The DSpark drafter of Inkling 2 over one sequence, at one device or split over two ranks (the
// form of the Qwen DFlash2 drafter): context in arrears, the block in one forward of B = 8 columns,
// fused norms and RoPE, the recurrent head over the vocabulary halves with one top-1 exchange per
// proposal. Mathematics: docs/maintainer/inkling2-model.md; Ops: ninfer/ops/dspark_draft.h.
//
//   Context (materialize): the target's four states of the columns a call licensed (the pending
//   features, written by the forward through sink()) are mixed per layer (feature_mix_rmsnorm),
//   projected by the layer's context key|value rows and normed, rotated and stored into the layer's
//   cache at their positions (context_kv_store); the head's mixed state of the last licensed
//   column becomes f_rnn. Local layers keep a ring of `window` positions, the global layer every
//   position up to max_context.
//   Block (propose): X = [bf16(rms(embedding[anchor], embedding_norm)), mask_embedding x 7] at
//   positions P..P+7; three pre-norm layers (query|key|value, head norm + RoPE, block attention
//   over the cache [lo, P) and the block, output projection, add + norm, gate|up, SiLU product,
//   down projection, add + norm), the final norm Hd; base logits = output head over Hd[:, 1:8];
//   seven chained steps of the recurrent head, each the top-1 over the vocabulary.
//
// Tensor parallelism (width 2): every rank holds its heads and MLP half and the folded head over
// its output-head vocabulary rows, and both ranks keep the identical residual. A layer exchanges
// twice (the output and down projections, row-parallel BF16 partials summed by one all-reduce),
// each proposal once (the two ranks' top-1 slots, an FP32 all-reduce of 16 bytes); both ranks then
// take the same token, so both write the same drafts and neither waits for the other's drafts. The
// drafts are not bit-identical to one device's (the BF16 partials); the target verifies them.
//
// The drafter owns its device state (caches, pending features, f_rnn) and its workspace; the
// caller owns the operands, the transport, the ordering with the target's forward and graphs.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "models/inkling2/config.h"
#include "models/inkling2/draft/config.h"
#include "models/inkling2/draft/parameters.h"
#include "models/inkling2/draft/proposer.h"
#include "models/inkling2/execution/forward.h"
#include "models/inkling2/execution/parameters.h"
#include "ninfer/ops/dspark_draft.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace ninfer::models::inkling2::draft {

// One rank's operands: its drafter geometry (shard_draft_config) and Parameters, the target's
// Parameters of the same rank (token embedding, embedding norm, output head), stream and device.
struct DraftRank {
    const DraftConfig* config           = nullptr;
    const Parameters* parameters        = nullptr;
    const execution::Parameters* target = nullptr;
    cudaStream_t stream                 = nullptr;
    int device                          = 0;
};

class Drafter final : public Proposer {
public:
    // `config` is the logical drafter config; `max_width` the widest call whose features the
    // drafter consumes (prefill blocks and verification calls); `max_context` the sequence
    // capacity.
    Drafter(const DraftConfig& config, const TextConfig& text, std::array<DraftRank, 2> ranks,
            int width, execution::ForwardCollectives collectives, std::int32_t max_width,
            std::uint32_t max_context, bool with_image = false);
    ~Drafter() override;

    [[nodiscard]] std::uint32_t proposals() const noexcept override { return config_.proposals(); }

    // Rank r's sink for a target call of `width` columns: the call writes its four states into the
    // pending features, which the next materialization consumes.
    [[nodiscard]] const execution::FeatureSink* sink(int rank, std::int32_t width) override;

    // Eager: the pending features of the last call (`width` columns at positions first..) become
    // context; then nothing is pending.
    void append(std::int32_t first, std::int32_t width) override;

    // In a round (stream-ordered, capturable), every rank from its own device scalars: the pending
    // features of the last call (count = the last commit()) become context at positions
    // frontier - count .., then the block at the anchor's position `frontier` writes the
    // first `proposals` (1 .. B - 1) proposals into drafts (I32 [B - 1] per rank): the block always
    // has B columns (the width the drafter is trained on), the recurrent head stops after the
    // proposals the round verifies (its chain is causal, so they do not depend on the rest).
    // `width` is the width of the call whose features are pending (a round's verification calls).
    void propose(const std::array<Tensor, 2>& anchor, const std::array<Tensor, 2>& frontier,
                 const std::array<Tensor, 2>& drafts, std::int32_t width,
                 std::int32_t proposals) override;

    // In a round after acceptance: the last call licensed `count` (I32 [1] per rank) columns.
    void follow(const std::array<Tensor, 2>& frontier, std::int32_t width) override;
    void commit(const std::array<Tensor, 2>& count) override;

    // The anchor image of the drafter's context that a position-ordered resume cannot rebuild:
    // the local layers' rings and f_rnn, on every rank (the global layer's cache is per position,
    // like the target's global K/V). save_image keeps them, restore_image puts them back and
    // leaves nothing pending. Requires an image (with_image at construction).
    void save_image() override;
    void restore_image() override;
    // Nothing pending: the next round materializes no context (a new sequence or a resume).
    void clear_pending() override;
    // Eager: the pending features of the last round (a call of `width` columns, the device's
    // pending count of them, ending at position `frontier`) become context; then none is pending.
    void catch_up(std::int32_t frontier, std::int32_t width) override;

    // Device bytes per rank of the state (with the image when `image`) and of the workspace of a
    // drafter of rank geometry `rank` (shard_draft_config of `config`) over `rows` folded rows.
    [[nodiscard]] static std::size_t planned_bytes(const DraftConfig& config,
                                                   const DraftConfig& rank, std::int32_t rows,
                                                   std::int32_t attention_input_rows,
                                                   std::int32_t max_width,
                                                   std::uint32_t max_context, bool image);

    // Device bytes the drafter's state holds on each rank (caches, pending features, buffers).
    [[nodiscard]] std::size_t state_bytes() const noexcept override { return state_bytes_; }

    [[nodiscard]] std::size_t workspace_bytes() const noexcept override { return workspace_bytes_; }

private:
    struct Rank;

    template <class Body>
    void each(Body&& body);

    void materialize(const std::array<Tensor, 2>& frontier, std::int32_t width);
    void copy_image(bool save);
    // The row-parallel projection of `x` by every rank's `weight` into `out` (BF16 [H, B]).
    void project_rows(const std::array<Tensor, 2>& x, std::array<const LinearParameters*, 2> weight,
                      std::array<Tensor, 2>& out);

    DraftConfig config_;
    const TextConfig& text_;
    std::array<DraftRank, 2> ranks_;
    int width_;
    execution::ForwardCollectives tp_;
    std::int32_t max_width_;
    std::uint32_t max_context_;
    std::array<std::unique_ptr<Rank>, 2> rank_;
    std::size_t state_bytes_ = 0, workspace_bytes_ = 0;
};

} // namespace ninfer::models::inkling2::draft
