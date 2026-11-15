#pragma once

// ninfer::ops::detail - private launch of the grouped slot-pool NVFP4 MoE (moe_grouped_nvfp4).

#include "ninfer/ops/moe_grouped.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// One job is one 64-column tile of one expert's grouped assignments.
inline constexpr std::int32_t kMoeGroupedJobColumns = 64;

// Jobs of `columns` tokens: whole tiles of every assignment plus one partial tile per expert.
[[nodiscard]] constexpr std::int32_t moe_grouped_max_jobs(std::int32_t columns) {
    return columns * kMoeSlotsTopK / kMoeGroupedJobColumns + 512;
}

struct MoeGroupedLayout {
    std::size_t input_codes, input_scales, counts, offsets, rank, packed_token, packed_index,
        job_experts, job_columns, job_slots, job_count, middle_codes, middle_scales, grouped, total;
};

[[nodiscard]] MoeGroupedLayout moe_grouped_layout(std::int32_t columns);

// One piece of <= kMoeGroupedChunkColumns columns with its workspace carved out.
struct MoeGroupedPiece {
    const __nv_bfloat16* x;  // [columns, H]
    const std::int32_t* ids; // [columns, K]
    const float* weights;    // [columns, K]
    float* out;              // [columns, H]
    std::int32_t columns;
    const std::int32_t* table; // [E]
    std::int32_t slots;
    const float* gate_up_divisors; // [2E]
    const float* down_divisors;    // [E]
    float gate_up_input_divisor;
    float down_input_divisor;
    const std::uint8_t* planes[4]; // gate/up codes, gate/up scales, down codes, down scales
    std::uint8_t* input_codes;     // [columns, H/2]
    std::uint8_t* input_scales;    // [columns, H/16]
    std::int32_t* counts;          // [E]
    std::int32_t* offsets;         // [E+1]
    std::int32_t* rank;            // [columns*K]
    std::int32_t* packed_token;    // [columns*K]
    std::int32_t* packed_index;    // [columns*K]
    std::int32_t* job_experts;     // [max_jobs]
    std::int32_t* job_columns;     // [max_jobs]
    std::int32_t* job_slots;       // [max_jobs]
    std::int32_t* job_count;       // [1]
    std::uint8_t* middle_codes;    // [columns*K, I/2]
    std::uint8_t* middle_scales;   // [columns*K, I/16]
    float* grouped;                // [columns*K, H] FP32 per-assignment expert outputs
};

void moe_grouped_launch(const MoeGroupedPiece& piece, cudaStream_t stream);

// A call wider than one piece. gate/up runs ONCE over the whole call, grouped by expert
// (each expert's weights read once per call instead of once per piece); the down projection and
// the merge run per piece of <= kMoeGroupedChunkColumns columns (their FP32 per-assignment
// outputs bound the workspace), reading the call's grouped h through a per-piece row map. Every
// output row is computed exactly as by the pieces (same tiles, same K order): bit-identical.
struct MoeGroupedWideLayout {
    std::size_t input_codes, input_scales, counts, offsets, rank, packed_token, packed_index,
        job_experts, job_columns, job_slots, job_count, middle_codes, middle_scales, piece_counts,
        piece_offsets, piece_rank, piece_index, piece_middle_row, piece_job_experts,
        piece_job_columns, piece_job_slots, piece_job_count, grouped, total;
};

[[nodiscard]] MoeGroupedWideLayout moe_grouped_wide_layout(std::int32_t columns);

struct MoeGroupedWide {
    MoeGroupedPiece call;           // the whole call: x/ids/weights/out of all columns, the global
                                    // grouping arrays, middle_* for every assignment
    std::int32_t* piece_counts;     // [E]
    std::int32_t* piece_offsets;    // [E+1]
    std::int32_t* piece_rank;       // [piece*K]
    std::int32_t* piece_index;      // [piece*K] position of an assignment in the piece grouping
    std::int32_t* piece_middle_row; // [piece*K] piece position -> row of the call's middle
    std::int32_t* piece_job_experts;
    std::int32_t* piece_job_columns;
    std::int32_t* piece_job_slots;
    std::int32_t* piece_job_count;
};

void moe_grouped_wide_launch(const MoeGroupedWide& wide, cudaStream_t stream);

} // namespace ninfer::ops::detail
