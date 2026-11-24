#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

namespace ninfer::ops {

// One block of rows of a stacked projection and the [rows, T] BF16 buffer it is written to.
struct Inkling2ProjectionBlock {
    void* data        = nullptr;
    std::int32_t rows = 0;
};

inline constexpr int kInkling2ProjectionMaxBlocks = 8;

/**
 * Op: Inkling 2 Mini stacked BF16 projection at verification widths, written block by block
 *
 * out = W x for x [K, T] (2 <= T <= 16) and one of Inkling 2 Mini's stacked attention input
 * projections W [n, K] (the tp 2 shards: queries, all, keys/values), where the rows of W are the
 * `blocks` in order (their rows add up to n): block b's rows go to its own [rows_b, T] buffer
 * instead of one [n, T] output that row copies then split.
 *
 * Effects:
 *   The kernels and arithmetic of ops::linear on these shapes at these widths (the generic GEMV's
 *   per-row chains): every element has the bits ops::linear gives it, and every column the bits
 *   of the same projection at T = 1.
 *
 * Returns false and launches nothing when the operands are not such a projection (other shape,
 * dtype, width, alignment or blocks); the caller then takes ops::linear and the row copies.
 *
 * Execution:
 *   One kernel, stream ordered on `stream`, CUDA Graph capturable, no workspace.
 */
bool inkling2_projection_blocks(const Tensor& x, const Weight& w,
                                std::span<const Inkling2ProjectionBlock> blocks,
                                cudaStream_t stream);

} // namespace ninfer::ops
