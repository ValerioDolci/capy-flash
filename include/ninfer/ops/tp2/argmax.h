#pragma once

// Two-device (tp 2) forms of argmax.

#include "ninfer/ops/argmax.h"
#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Vocabulary-split argmax over two tensor-parallel ranks, in three steps: each rank packs its own
 * candidate (argmax_split_pack), one allreduce_sum of the packed candidates gives both ranks both
 * candidates, and a rank selects the complete argmax (argmax_split_select). Rank r holds the
 * contiguous row block [r R, (r + 1) R) of the complete [2R,T] logits.
 *
 * The largest row count one rank may hold: a candidate's row travels as two 8-bit digits.
 */
inline constexpr std::int32_t kArgmaxSplitMaxRowsPerRank = 65536;

// BF16 elements per column of the packed candidates: four digits per rank.
inline constexpr std::int32_t kArgmaxSplitCandidateRows = 8;

/**
 * Rank `rank`'s half of the vocabulary-split argmax. `logits` is its contiguous BF16 block [R,T]
 * with 1 <= R <= kArgmaxSplitMaxRowsPerRank, and `local` the contiguous I32 [T] argmax() of that
 * block over all R rows. Writes all of the contiguous BF16 `candidates` [8,T]: column t is zero
 * except elements [4 rank, 4 rank + 4), which hold the base-256 digits
 * (b >> 8, b & 255, l >> 8, l & 255) of the BF16 bits b of logits[l,t] and of the row l = local[t].
 * Each digit is an integer in [0,255], exactly representable in BF16, and x + 0 is exact, so the
 * allreduce_sum of both ranks' candidates is their exact union on both ranks. `rank` is 0 or 1;
 * `candidates` must not overlap `logits` or `local`. No workspace or other state.
 */
void argmax_split_pack(const Tensor& logits, const Tensor& local, int rank, Tensor& candidates,
                       cudaStream_t stream);

/**
 * The complete argmax from the summed candidates [8,T] of argmax_split_pack over blocks of
 * `rows_per_rank` rows: out[t] = rows_per_rank + l1 when v1 > v0, else l0, where (v_r, l_r) is
 * rank r's candidate. Equal maxima therefore select rank 0's, the lower row, and the result equals
 * argmax() over the complete [2R,T] logits, NaN included, except when row R holds a NaN: argmax()
 * skips a NaN except in its row 0, which row R is in rank 1's block. `out` is contiguous I32 [T]
 * and does not overlap `candidates`. No workspace or other state.
 */
void argmax_split_select(const Tensor& candidates, std::int32_t rows_per_rank, Tensor& out,
                         cudaStream_t stream);

/**
 * Log-probability of the vocabulary-split argmax, in the same three steps: each rank packs the
 * log-probability of its own candidate within its block (logprob_split_pack of
 * target_logprobs(logits, local)), one allreduce_sum of the packed values gives both ranks both,
 * and a rank computes the log-probability of the complete argmax over both blocks
 * (argmax_split_logprob).
 *
 * logprob_split_pack: `local_logprob` is the contiguous FP32 [T] log-probability of rank `rank`'s
 * candidate within its block, p_r = v_r - lse_r. Writes all of the contiguous BF16 `packed` [8,T]:
 * column t is zero except elements [4 rank, 4 rank + 4), the base-256 digits of the FP32 bits of
 * p_r[t]; as for argmax_split_pack, the allreduce_sum of both ranks' packed values is their exact
 * union.
 *
 * argmax_split_logprob: from the summed `candidates` of argmax_split_pack (v_r) and the summed
 * `packed` of logprob_split_pack (p_r), writes the contiguous FP32 `out` [T]:
 *   out[t] = max(v_0, v_1) - log(exp(v_0 - p_0) + exp(v_1 - p_1)),
 * the log-probability of the complete argmax over both blocks, evaluated in FP32 about the
 * larger log-normalizer. No workspace or other state; outputs do not overlap inputs.
 */
void logprob_split_pack(const Tensor& local_logprob, int rank, Tensor& packed, cudaStream_t stream);
void argmax_split_logprob(const Tensor& candidates, const Tensor& packed, Tensor& out,
                          cudaStream_t stream);

} // namespace ninfer::ops
