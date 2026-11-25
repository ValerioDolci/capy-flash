#pragma once

// The Text forward of Inkling 2, at one device or at tensor-parallel width 2. Mathematics:
// the 48-visit schedule (config.h Visit), block attention residuals over the completed blocks,
// cross-pass messages in the core's later passes, one visit = attention + short convolution +
// FP32 residual + MoE + short convolution, and the output head with its logit soft cap.
//
// One call processes T >= 1 consecutive tokens of one sequence (a prefill block, or T = 1 for a
// decode step) against that sequence's VisitState, which it advances by T positions. Every visit
// reads and writes its own state slot; the block states, messages and residuals live only inside
// the call (they are per token).
//
// One template serves execution (Arena = WorkspaceArena: allocates and launches) and workspace
// sizing (Arena = WorkspaceLayoutBuilder: the same allocations in the same scopes, each Op
// replaced by its declared scratch), so the planned workspace is the executed one by construction
// (forward_workspace_bytes).
//
// Tensor parallelism (width 2, load/sharding.h): each rank holds the whole residual and issues the
// replicated work (block residuals, message gates, norms, router, post-attention and post-FFN
// convolutions) on its own stream; the per-rank shards are the attention heads (12/4 of 24/8), the
// expert bank of its ID range and the vocabulary half of the output head. A visit exchanges twice:
// the attention output projection (row-parallel, one BF16 all-reduce) and the two banks' FP32 MoE
// partials (allreduce_sum_f32, one BF16 rounding). The logits are gathered on rank 0.
//
// This header declares the forward. Its members are defined by responsibility in forward/ (core,
// visit, residual, attention, moe, head) and instantiated once for execution and once for planning
// in forward.cpp; callers include this header only.

#include "core/arena.h"
#include "core/device.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "models/inkling2/config.h"
#include "models/inkling2/execution/parameters.h"
#include "models/inkling2/state/conv_record.h"
#include "models/inkling2/state/visit_state.h"
#include "ninfer/ops/allreduce.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace ninfer::models::inkling2::execution {

// Teacher-forcing taps of an executing forward (a test driver's dumps). Points: `visit` is the
// execution index 0..visits-1 with part "" (the visit's output, BF16 [H,T]), "attn" (attention
// output after the exchange, BF16), "attn_conv" (after the post-attention convolution, BF16),
// "ain" (attention input after the block residual, FP32 ne {H,T,1|3}), "msg" (the messages, FP32
// ne {H,T,4}: q, k, v, ffn), "min" (FFN input after the block residual, FP32), "moe_in" (the
// normed FFN input, BF16), "moe" (the MoE output after the exchange, BF16), "route_ids" (I32 {k,T},
// in decreasing key order) and "route_weights" (FP32 {k,T}); `visit` = kHeadTap
// with part "embed" (the normed embedding), "final" (the output head's input, BF16 [H,T]) or
// "reference" (the Logits loop contrast's reference head input, BF16 [H,T]).
// Column t is sequence position first_position + t. capture() receives rank 0's copy with rank
// 0's device current; at tp 2 compare_ranks() also receives both copies of the replicated value.
class ForwardTap {
public:
    static constexpr std::int32_t kHeadTap = -1;

    virtual ~ForwardTap()                                                             = default;
    [[nodiscard]] virtual bool wants(std::int32_t visit, std::string_view part) const = 0;
    virtual void capture(std::int32_t visit, std::string_view part, const Tensor& value,
                         std::int32_t first_position, cudaStream_t stream)            = 0;

    virtual void compare_ranks(std::int32_t visit, std::string_view part, const Tensor& rank0,
                               const Tensor& rank1, cudaStream_t stream0, cudaStream_t stream1) {
        (void)visit, (void)part, (void)rank0, (void)rank1, (void)stream0, (void)stream1;
    }
};

// The states a speculative drafter reads from every column of a call: the inputs of three visits
// (the BF16 residual `x` passed to visit(index), which restarts at block boundaries) and the
// output head's normed input. `features` is the rank's BF16 [H, T, 4] destination (each block a
// contiguous [H, T], blocks at any larger stride): block j < 3
// holds the input of visit `visits[j]`, block 3 the head input; column t is position
// first_position + t. The forward computes these values directly into the destination (a visit
// whose input is a temporary writes its output there; other inputs are copied), so a call with a
// sink computes the same bits as one without. At tp 2 each rank fills its own destination from
// its own copy of the replicated residual.
struct FeatureSink {
    std::array<std::uint32_t, 3> visits{};
    Tensor features;
};

class PrefetchLane; // prefetch_lane.h

// One rank's operands. `config` is the rank's geometry (shard_text_config at tp 2, the logical
// config at one device); `device` is the CUDA device (read at tp 2); `state`, `positions` and
// `features` are read only when executing. With `record` (a speculative verification call) every
// short convolution writes its input history into the record and leaves its tail as it was; the
// caller folds the tails after the committed tokens (ConvRecord::fold). With `prefetch`, calls of
// up to kPrefetchMaxTokens columns preload their dense projection weights into L2 on that lane
// (forward/prefetch.h).
template <class Arena>
struct ForwardRank {
    const TextConfig* config     = nullptr;
    const Parameters* parameters = nullptr;
    Arena* work                  = nullptr;
    cudaStream_t stream          = nullptr;
    int device                   = 0;
    const VisitState* state      = nullptr;
    const Tensor* positions      = nullptr; // device I32 [T]: first_position + t
    const ConvRecord* record     = nullptr;
    const FeatureSink* features  = nullptr;
    const PrefetchLane* prefetch = nullptr;
};

// The two-device transport of an executing tp 2 forward.
struct ForwardCollectives {
    const ExecutionContext* execution = nullptr;
    const ops::PeerEvents* events     = nullptr;
};

// The tokens of one call: T columns of one sequence starting at absolute position
// `first_position` (the positions the state already holds). `logit_columns` = the number of last
// columns whose logits the head computes (0: none, the head is skipped).
struct ForwardColumns {
    std::int32_t width          = 0;
    std::int32_t first_position = 0;
    std::int32_t logit_columns  = 0;
};

// Rank 0's logits of the last `logit_columns` columns: `logits` BF16 [Vu, n] as the output head
// rounds them (argmax reads these), `capped` FP32 [Vu, n] = softcap(logits) (null: not written).
// With the Logits loop contrast (TextConfig::contrast) `capped` holds the guided logits and
// `logits` their BF16 rounding, and `reference` (null: not written) receives the reference head's
// soft-capped logits FP32 [Vu, n].
struct ForwardLogits {
    Tensor* logits    = nullptr;
    Tensor* capped    = nullptr;
    Tensor* reference = nullptr;
};

template <class Arena>
class Forward {
public:
    static constexpr bool kExecute = std::is_same_v<Arena, WorkspaceArena>;
    using Rank                     = ForwardRank<Arena>;
    using RankTensors              = std::array<Tensor, 2>;

    // `width` ranks of `ranks` (rank 0 first). `config` is the logical Text config; at width 2
    // `collectives` must be complete when executing (planning ignores it). Planning sizes for
    // calls of up to `max_tokens` columns and `max_logit_columns` logit columns.
    Forward(const TextConfig& config, std::array<Rank, 2> ranks, int width,
            ForwardCollectives collectives, std::int32_t max_tokens,
            std::int32_t max_logit_columns);

    // ids: per rank I32 [T] (the same tokens). Executes the 48 visits and, when
    // columns.logit_columns > 0, the head into `out` (rank 0). Advances every rank's state.
    // Planning: the arguments' tensors are not read; T = max_tokens.
    void run(const RankTensors& ids, const ForwardColumns& columns, const ForwardLogits& out,
             ForwardTap* taps = nullptr);

private:
    struct Block; // forward/visit.h: the call's block states, messages and buffers

    // ---- Operands and arena (forward/core.h) ----------------------------------------------------
    [[nodiscard]] static std::size_t at(int r) { return static_cast<std::size_t>(r); }

    [[nodiscard]] const TextParameters& text(int r) const { return r_[at(r)].parameters->text; }

    [[nodiscard]] const LayerParameters& layer(int r, std::uint32_t l) const {
        return text(r).layers[l];
    }

    [[nodiscard]] const TextConfig& rc() const { return *r_[0].config; }

    [[nodiscard]] Arena& work(int r) const { return *r_[at(r)].work; }

    [[nodiscard]] cudaStream_t s(int r) const { return r_[at(r)].stream; }

    [[nodiscard]] const VisitState& state(int r) const { return *r_[at(r)].state; }

    // Rank r's records of slot `slot` for this call, when the call records its convolutions.
    [[nodiscard]] std::optional<VisitState::ConvTails> records(int r, std::uint32_t slot) const;

    [[nodiscard]] std::int32_t hidden() const;

    // Runs body(r) for each rank, with that rank's device current at tp 2 when executing.
    template <class Body>
    void each(Body&& body);

    struct Scopes {
        std::array<std::optional<typename Arena::Scope>, 2> held;
    };

    [[nodiscard]] Scopes scopes();
    [[nodiscard]] RankTensors alloc(DType dtype, std::initializer_list<std::int32_t> shape);
    void reserve(std::size_t bytes, int rank = -1);

    // Rank r's Linear; in planning its scratch over [1, max_tokens].
    void linear(int r, const Tensor& x, const LinearParameters& p, Tensor& out);

    // Hands the tap rank 0's copy (and at tp 2 both copies) of `value` when it wants the point.
    void tap(std::int32_t visit, std::string_view part, const RankTensors& value);

    // Every rank's [H, T] view of feature block `block` of its sink, when the call has sinks.
    [[nodiscard]] std::optional<RankTensors> sink(std::int32_t block) const;

    // ---- The schedule (forward/visit.h) ---------------------------------------------------------
    void embed(const RankTensors& ids, RankTensors& x);
    // The visits in execution order from the embedding, whole or (split > 0) as two halves of
    // `split` and T - split columns (forward/overlap.h); returns the last visit's output.
    const RankTensors* visits(Block& b, std::int32_t split);
    void visit(std::uint32_t index, Block& b, const RankTensors& x, RankTensors& out);

    // ---- The half-chunk overlap of a wide tp 2 call (forward/overlap.h) -------------------------
    struct Half; // one half's columns, exchange buffers and open exchange
    // The width of the first half of a T-column call, or 0 to run it in one piece.
    [[nodiscard]] std::int32_t overlap_split(std::int32_t T) const;
    [[nodiscard]] std::array<Half, 2> halves(std::int32_t split);
    // Visit `index` with its attention inputs, output projections and FFN by halves around the
    // attention of the whole call; `captured` (or null) receives the visit input's columns once
    // they are complete.
    void visit_halves(std::uint32_t index, Block& b, const RankTensors& x, RankTensors& out,
                      RankTensors* captured, std::array<Half, 2>& halves);
    void half_inputs(std::uint32_t index, Block& b, const RankTensors& x, RankTensors* captured,
                     std::array<RankTensors, 3>& xin, Half& half);
    void half_ffn(std::uint32_t index, Block& b, const RankTensors& x, RankTensors& out,
                  Half& half);
    // Finishes the half's open exchange: the attention output's, or the MoE partials' with the
    // close of their visit (sum, post-FFN convolution, BF16 output). Called on the half's columns.
    void close_half(Half& half);
    void abandon_halves(std::array<Half, 2>& halves) noexcept;
    // Runs body() with the call narrowed to the half's columns (T_ and the taps' first position).
    template <class Body>
    void on_half(Half& half, Body&& body);
    [[nodiscard]] RankTensors window(const RankTensors& t, const Half& half) const;

    // ---- Block residuals, messages, norms, convolutions (forward/residual.h)
    // --------------------- depth_attention of `queries` over the block states and `current` on
    // every rank: FP32 ne {H, T, Q}. At decode and verification widths with a BF16 current, the
    // depth_attention_rmsnorm kernel (residual_mix.h) writes it.
    void depth(const std::vector<const RankTensors*>& states, const RankTensors& current,
               std::array<const Tensor*, 2> queries, RankTensors& out);

    // The attention inputs xq, xk, xv (BF16 [H,T], the caller's; one tensor for all three outside
    // the message passes of a local visit: xin[1] and xin[2] then become views of xin[0]) from the
    // visit input.
    void attention_inputs(std::uint32_t index, Block& b, const RankTensors& x,
                          std::array<RankTensors, 3>& xin);

    // Whether the visit's residual steps run fused (residual_mix.h): decode and verification
    // widths, the conv state advanced in place only for one token, no tap of the steps the
    // fusion does not write (attn_conv, ain, min).
    [[nodiscard]] bool fused_residual(std::uint32_t index) const;

    // The static short convolution with `state` as its tail, or into `record` (not null) without
    // advancing the tail.
    void short_conv(int r, const Tensor& x, const Tensor& weight, const Tensor& state,
                    const Tensor* record, Tensor& out);

    // ---- Attention (forward/attention.h)
    // --------------------------------------------------------- The attention of a visit from its
    // inputs to the exchanged output y BF16 [H,T].
    void attention(std::uint32_t index, const std::array<RankTensors, 3>& xin, RankTensors& y);
    void attention_rank(std::uint32_t index, int r, const std::array<RankTensors, 3>& xin,
                        Tensor& o);
    // One Linear of x over stacked rows; block i receives its `rows` rows ([rows, T]).
    void project(int r, const Tensor& x, const LinearParameters& stacked,
                 std::initializer_list<std::pair<Tensor*, std::int32_t>> blocks);

    // ---- MoE (forward/moe.h)
    // --------------------------------------------------------------------- Router (replicated),
    // the rank's bank(s) into an FP32 partial, one BF16 rounding into z.
    void moe(std::uint32_t index, const RankTensors& u, RankTensors& z);
    // Router and the rank's bank(s) into the rank's FP32 partial [H, T].
    void moe_partial(std::uint32_t index, const RankTensors& u, RankTensors& partial);

    // ---- Head (forward/head.h)
    // -------------------------------------------------------------------
    void head(Block& b, const RankTensors& x, const ForwardColumns& columns,
              const ForwardLogits& out);
    // The head input bf16(final_norm(depth attention of `states` and x)) [H, T] into `hn`.
    void head_input(const std::vector<const RankTensors*>& states, const RankTensors& x,
                    RankTensors& hn);
    // The output head over the columns `last` [H, n], gathered on rank 0 into `logits`.
    void head_logits(const RankTensors& last, Tensor& logits);

    // ---- Loop contrast (forward/visit.h)
    // --------------------------------------------------------- Hidden: before the coda, x and the
    // block states produced by the last pass become their extrapolation away from the reference
    // pass's (into b.guided); states shared by every pass stay.
    void extrapolate(Block& b, const RankTensors*& x,
                     const std::vector<const RankTensors*>& reference_states,
                     const RankTensors& reference_output);

    // ---- L2 preload of the projection weights (forward/prefetch.h) ------------------------------
    enum class Prefetch { Early, Inputs, Output };
    // Rank r (its device current): forks its lane and preloads into L2 there visit `index`'s input
    // projections (Early: their leading part, at one column; Inputs: the rest) or its output
    // projection (at one column followed by the next visit's Early). No-op without a lane, past
    // the last visit or above kPrefetchMaxTokens columns.
    void prefetch(int r, std::uint32_t index, Prefetch what);
    // Joins every lane forked in this call back into its rank's stream.
    void join_prefetch();

    const TextConfig& config_;
    std::array<Rank, 2> r_;
    int width_ = 1;
    ForwardCollectives tp_;
    std::int32_t max_tokens_        = 1;
    std::int32_t max_logit_columns_ = 0;
    // The token embedding table of each rank as the embedding Op's Weight.
    std::array<Weight, 2> embedding_{};
    // The executing call's columns.
    std::int32_t T_              = 1;
    std::int32_t first_position_ = 0;
    ForwardTap* tap_             = nullptr;
    std::array<bool, 2> forked_{};
};

extern template class Forward<WorkspaceArena>;
extern template class Forward<WorkspaceLayoutBuilder>;

// Workspace bytes each rank's arena needs for every call of up to `max_tokens` columns and up to
// `max_logit_columns` logit columns (the same Forward in planning mode). `ranks[r]` supplies rank
// r's config and Parameters (work/state/positions unused).
[[nodiscard]] std::size_t forward_workspace_bytes(const TextConfig& config,
                                                  std::array<ForwardRank<WorkspaceArena>, 2> ranks,
                                                  int width, std::int32_t max_tokens,
                                                  std::int32_t max_logit_columns);

} // namespace ninfer::models::inkling2::execution
