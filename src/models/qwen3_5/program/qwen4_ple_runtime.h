#pragma once

#include "core/arena.h"
#include "models/qwen3_5/execution/ple_gather.h"
#include "models/qwen3_5/execution/forward_runtime.h"
#include "models/qwen3_5/program/qwen4_ple_stager.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

class ProgramImpl;
struct SequencePlanImpl;

// Qwen4Exp PLE of a Program (one device or tp 2): the n-gram table gather and its pinned staging,
// the deferred decode rows and the commit fold of the convolution history. Owned by value by the
// Program, which it reads (sequences, egress, runtime records) through the `program` arguments.
// Empty (no gather) when the model has no PLE.
class Qwen4PleRuntime {
public:
    // The PLE part of ProgramImpl::construct_qwen4: binds the gather, the staging and the
    // convolution states into each rank's runtime.
    void construct(ProgramImpl& program, const SequencePlanImpl& plan, DeviceSpan backing,
                   execution::ForwardRuntime& runtime, execution::ForwardRuntime& peer_runtime);

    // Qwen4Exp PLE n-gram rows of a decode window into the pinned round buffer: columns
    // [first_column, first_column + tokens.size()) are the tokens `tokens` at positions
    // frontier.., with `ledger` (the committed tokens before `frontier`) as their history.
    // Page prefetch of the next round's n-gram rows (an MTP round's anchor and drafts, or
    // an ordinary round's sampled token, are in the egress once this round has synchronized), so
    // the storage reads overlap the commit and the next round's host preparation.
    void prefetch_next_round(const ProgramImpl& program, std::span<const std::uint32_t> lanes,
                             bool mtp);
    void stage_rows(const ProgramImpl& program, std::span<const TokenId> ledger,
                    std::uint32_t frontier, std::span<const TokenId> tokens,
                    std::size_t first_column);
    // Deferred decode rows. A round: begin_round() (no stream waiting: lowers the
    // words), queue_rows() per row while preparing, the launch, then
    // stage_round() gathers the queued rows into the pinned staging and raises the
    // words (also on failure). raise_staged() alone releases an execution whose rows
    // do not matter (warm-up, graph preparation) or that failed after its launch.
    void begin_round();
    void queue_rows(std::uint32_t lane, std::uint32_t frontier, std::span<const TokenId> tokens,
                    std::size_t first_column);
    void stage_round(const ProgramImpl& program);
    void raise_staged() noexcept;
    // Qwen4Exp speculative commit: advances each row's PLE convolution history by its committed
    // prefix of the verified window (the PLE half of the ReplaySSM fold).
    static void fold(ProgramImpl& program, std::span<const ops::GdnReplayFoldRow> rows);
    static void fold_rank(ProgramImpl& program, std::span<const ops::GdnReplayFoldRow> rows,
                          int rank);

    std::unique_ptr<execution::PleGather> gather;
    std::optional<PinnedHostBuffer> host;
    std::optional<PinnedHostBuffer> decode_buffer;
    std::uint16_t* decode_host = nullptr;
    // Deferred decode rows (ForwardRuntime::decode_ple_staged): the mapped words of rank 0 and rank
    // 1, and the round's queued gathers (prepared before the launch, run after it).
    std::optional<PinnedHostBuffer> staged_words;
    std::array<std::uint32_t*, 2> staged{};

    struct Stage {
        std::uint32_t lane       = 0;
        std::uint32_t frontier   = 0;
        std::uint32_t count      = 0;
        std::size_t first_column = 0;
        std::array<TokenId, qwen3_5::kMtpDecodeMaximumWidth> tokens{};
    };

    std::vector<Stage> queue;
    // Runs stage_round() of a graph round after its launch (created with the words).
    std::unique_ptr<Qwen4PleStager> stager;
};

} // namespace ninfer::models::qwen3_5::detail
