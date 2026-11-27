#pragma once

// The DFlash2 block drafter of Inkling 2 over one sequence, at one device or split over two ranks
// (the form of the DSpark drafter, draft/draft.h): context in arrears, the block in one forward of
// B = 8 columns, the output head's top candidates over the vocabulary halves with one exchange,
// the candidate selector's greedy walk. Mathematics: the drafter checkpoint's (SGLang
// DFlash2DraftModel with the Inkling 2 selector soft cap after the correction); Ops:
// ninfer/ops/dflash2_draft.h, dspark_draft.h (norm + RoPE, context store, sliding block attention),
// candidate_selector.h.
//
//   Context (materialize): the target's four states of the columns a call licensed (the pending
//   features, written by the forward through sink()) are concatenated per column [s0 | s1 | s2 |
//   s3], projected (feature_projection) and normed (context_norm) into one context vector per
//   position; every layer projects it by its key|value rows, norms and rotates the keys and stores
//   them into its ring of `sliding_window` positions.
//   Block (propose): X = [bf16(rms(embedding[anchor], embedding_norm)), the same of the mask token
//   x 7] at positions P..P+7; per layer, around attention and MLP, the input norm, the kernel
//   projection of the normed input, its grouped dynamic convolution, the sublayer (query|key|value,
//   head norm + RoPE, attention of each column over its window of the ring and the whole block,
//   output projection; gate|up, SiLU product, down projection) and the output-side convolution
//   added to the residual; then the final norm of columns 1..7, their 16 best output-head rows
//   (unary scores = the BF16 logits) and the selector's walk from the anchor (scores = unary +
//   predecessor/successor correction; argmax per step).
//
// Tensor parallelism (width 2): every rank holds its heads and MLP half (row-parallel output and
// down projections, one all-reduce each), its ring of its KV heads, and the output head's
// vocabulary half; both ranks keep the identical residual. The candidates of the two halves meet
// in one FP32 all-reduce of the 16 best (value, id) pairs per column; both ranks then walk the same
// candidates and write the same drafts.
//
// The drafter owns its device state (rings, pending features, buffers) and its workspace; the
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

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <memory>

namespace ninfer::models::inkling2::draft {

// One rank's operands: its drafter geometry (shard_dflash2_config) and Parameters, the target's
// Parameters of the same rank (token embedding, embedding norm, output head), stream and device.
struct DFlash2Rank {
    const DFlash2Config* config         = nullptr;
    const DFlash2Parameters* parameters = nullptr;
    const execution::Parameters* target = nullptr;
    cudaStream_t stream                 = nullptr;
    int device                          = 0;
};

class DFlash2Drafter final : public Proposer {
public:
    // `config` is the logical drafter config; `max_width` the widest call whose features the
    // drafter consumes (prefill blocks and verification calls).
    // `features`: every rank's pending features shared with another drafter of the same target
    // states and `max_width` (BF16 [H, max_width, 4], the DSpark drafter's sink(r, max_width)),
    // which the forward then writes once for both; empty, the drafter holds its own.
    DFlash2Drafter(const DFlash2Config& config, std::array<DFlash2Rank, 2> ranks, int width,
                   execution::ForwardCollectives collectives, std::int32_t max_width,
                   bool with_image = false, std::array<Tensor, 2> features = {});
    ~DFlash2Drafter() override;

    [[nodiscard]] std::uint32_t proposals() const noexcept override { return config_.proposals(); }

    [[nodiscard]] const execution::FeatureSink* sink(int rank, std::int32_t width) override;
    void append(std::int32_t first, std::int32_t width) override;
    void propose(const std::array<Tensor, 2>& anchor, const std::array<Tensor, 2>& frontier,
                 const std::array<Tensor, 2>& drafts, std::int32_t width,
                 std::int32_t proposals) override;
    void follow(const std::array<Tensor, 2>& frontier, std::int32_t width) override;
    void commit(const std::array<Tensor, 2>& count) override;
    // The image is every layer's ring.
    void save_image() override;
    void restore_image() override;
    void clear_pending() override;
    void catch_up(std::int32_t frontier, std::int32_t width) override;

    // Device bytes per rank of the state (with the image when `image`, with its own pending
    // features when `own_features`) and of the workspace of a drafter of rank geometry `rank`
    // (shard_dflash2_config of `config`) over `rows` output-head rows.
    [[nodiscard]] static std::size_t planned_bytes(const DFlash2Config& config,
                                                   const DFlash2Config& rank, std::int32_t rows,
                                                   std::int32_t max_width, bool image,
                                                   bool own_features = true);

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

    DFlash2Config config_;
    std::array<DFlash2Rank, 2> ranks_;
    int width_;
    execution::ForwardCollectives tp_;
    std::int32_t max_width_;
    std::array<std::unique_ptr<Rank>, 2> rank_;
    std::size_t state_bytes_ = 0, workspace_bytes_ = 0;
};

} // namespace ninfer::models::inkling2::draft
