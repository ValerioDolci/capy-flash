#pragma once

// The Qwen4Exp MTP proposal head over a draft token shortlist.
//
// The verification head stays the full-vocabulary Text output head, so greedy acceptance (and the
// output of a greedy MTP run) does not depend on the drafts; only which tokens are proposed does.
// A shortlist S of token IDs is executed as two row blocks of the Text head's weights (a Q6
// RowSplit head, or a BF16 head as in the FP4-Local checkpoint):
//   * prefix: rows [0, P) of the output head itself, a view (no copy), where [0, P) is the longest
//     ID prefix contained in S (Qwen BPE IDs are roughly ordered by merge frequency, so a large
//     prefix is most of a good shortlist); at tp 2, P is at most rank 0's vocabulary half;
//   * extra: the other M = |S| - P IDs, sorted ascending, whose head rows are copied (byte-exact)
//     into Program-owned planes at startup (Q6: code, high-bit and scale planes; BF16: one plane).
// One proposal: logits[0, P) = prefix * x, logits[P, P+M) = extra * x into the caller's proposal
// logits buffer reinterpreted as contiguous [P+M, B], argmax over the P+M rows (lowest row on ties
// = lowest token ID, as the full head's argmax), optionally the softmax log-probability of the
// argmax over the shortlist rows, then row -> token ID through `token_ids`. A token outside S is
// never proposed. Rows keep the head's stored values, so a shortlist row's logit is the full head's
// row logit up to the GEMV route's summation order (which may depend on the block's row count).

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "models/qwen3_5/execution/parameters.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

struct Qwen4DraftHead {
    LinearParameters prefix; // view of output-head rows [0, prefix_rows)
    LinearParameters extra;  // gathered rows (token IDs token_ids[prefix_rows..])
    std::int32_t prefix_rows = 0;
    std::int32_t extra_rows  = 0;
    // tp 2, split shortlist: rank 0 scores `prefix` (shortlist rows [0, R), a view of its
    // vocabulary half) and rank 1 `peer` (shortlist rows [R, 2R), gathered on rank 1 from both
    // halves); the two argmax candidates are combined as the split optimized proposal head's
    // (execution::proposal_argmax_split). extra_rows is 0 then.
    bool split = false;
    LinearParameters peer;
    std::int32_t peer_rows        = 0;
    const std::int32_t* token_ids = nullptr; // device I32 [rows()]: shortlist row -> token ID

    [[nodiscard]] std::int32_t rows() const noexcept {
        return prefix_rows + extra_rows + peer_rows;
    }
};

// The split of a shortlist: the longest ID prefix [0, prefix) inside it (capped at prefix_limit,
// rounded down to `multiple`) and the remaining IDs ascending. A shortlist whose size is not a
// multiple of `multiple` is first completed with the lowest IDs it lacks (the proposal writes the
// two blocks' logits back to back per column; ops::linear needs 16-byte aligned outputs, so
// `multiple` = 8 BF16 rows). Throws on duplicates, negative IDs or IDs >= vocabulary.
struct DraftShortlistSplit {
    std::int32_t prefix = 0;
    std::vector<std::int32_t> extras;
};

[[nodiscard]] DraftShortlistSplit split_draft_shortlist(std::span<const std::int32_t> ids,
                                                        std::int32_t vocabulary,
                                                        std::int32_t prefix_limit,
                                                        std::int32_t multiple = 8);

// Reads a shortlist file: whitespace-separated decimal token IDs in priority order ('#' starts a
// comment line). `rows` > 0 keeps the first `rows` IDs.
[[nodiscard]] std::vector<std::int32_t> read_draft_shortlist(const char* path, std::uint32_t rows);

// Per-row bytes of the output head's planes: Q6 RowSplit {codes, high bits, scales}; BF16
// {2 K, 0, 0} (row-major [N, K], one plane).
[[nodiscard]] std::array<std::size_t, 3> draft_row_plane_bytes(const Weight& head);
// Whether `head` is an output head the draft head can view and gather: Q6 RowSplit, or BF16
// contiguous with 16-byte aligned rows (K % 8 == 0).
[[nodiscard]] bool draft_head_supported(const Weight& head) noexcept;
// A view of rows [first, first + rows) of the output head.
[[nodiscard]] LinearParameters draft_row_view(const LinearParameters& head, std::int32_t first,
                                              std::int32_t rows);
// A weight with `head`'s geometry over `rows` rows stored in `planes` (the planes of
// draft_row_plane_bytes; a zero-byte plane is ignored and may be null).
[[nodiscard]] LinearParameters draft_rows_weight(const LinearParameters& head,
                                                 const std::array<std::byte*, 3>& planes,
                                                 std::int32_t rows);
// Copies head rows `source_rows[i]` (row indices of `head`) to rows destination_row0 + i of
// `planes`, byte-exact, as asynchronous D2D copies on `stream` (cross-device over UVA when the
// planes live on another device: the caller orders and synchronizes).
void gather_draft_rows(const Weight& head, std::span<const std::int32_t> source_rows,
                       const std::array<std::byte*, 3>& planes, std::int32_t destination_row0,
                       cudaStream_t stream);

// One proposal over B columns: hidden x BF16 [H, B] -> tokens I32 [B] (token IDs). `logits` is the
// proposal logits buffer (BF16, contiguous, at least rows() * B elements); `logprobs` (FP32 [B])
// optionally receives log softmax over the shortlist rows at the chosen row. B = 1 runs one GEMV
// per block; B > 1 one per column (the T = 1 route).
void draft_head_propose(const Qwen4DraftHead& head, const Tensor& x, const Tensor& logits,
                        Tensor& tokens, Tensor* logprobs, WorkspaceArena& work,
                        cudaStream_t stream);

} // namespace ninfer::models::qwen3_5::execution
