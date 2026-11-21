#pragma once

// ninfer::ops::detail - private launch of moe_bank_capped_swiglu and moe_partial_to_bf16.

#include "core/tensor.h"
#include "ninfer/ops/moe_bank.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// One weight plane of a bank, resolved from its view: logical row n of the view is row
// row_begin + n of the parent at `data`.
struct MoeBankPlane {
    const std::byte* data         = nullptr;
    bool nvfp4                    = false;
    bool mxfp8                    = false;
    std::int64_t row_begin        = 0;
    std::int32_t columns          = 0;
    std::uint64_t scale_offset    = 0; // nvfp4: swizzled scale plane; mxfp8: row-major UE8M0
    std::uint64_t divisor_offset  = 0; // nvfp4: FP32 divisor words
    std::int64_t rows_per_divisor = 0; // nvfp4
};

struct MoeBankShape {
    std::int32_t hidden       = 0; // H
    std::int32_t intermediate = 0; // F
    std::int32_t experts      = 0; // E'
    std::int32_t first_expert = 0;
    std::int32_t top_k        = 0;
    std::int32_t tokens       = 0;
};

// Workspace of one call: h of every (token, position) of a chunk (BF16 [chunk * k][F]) and, on
// the wide route, the down outputs o (BF16 [chunk * k][H]) and the chunk's plan of columns per
// expert (I32 [E' + 1 + chunk * k]); the call runs in chunks of
// chunk_tokens tokens (the narrow route always in one).
struct MoeBankWorkspace {
    std::uint16_t* hidden = nullptr;
    std::uint16_t* out    = nullptr;
    std::int32_t* plan    = nullptr; // wide route: offsets [E' + 1], then the sorted columns
    int chunk_tokens      = 0;
};

// Whether the call takes the wide (tensor-core, grouped-by-expert) route.
[[nodiscard]] bool moe_bank_wide_route(const MoeBankShape& shape);

void moe_bank_capped_swiglu_launch(const Tensor& x, const Tensor& ids, const Tensor& weights,
                                   const MoeBankPlane& gate_up, const MoeBankPlane& down,
                                   const MoeBankShape& shape, CappedSwiglu activation,
                                   bool accumulate, Tensor& partial, const MoeBankWorkspace& work,
                                   cudaStream_t stream);

void moe_partial_to_bf16_launch(const Tensor& partial, Tensor& y, cudaStream_t stream);

} // namespace ninfer::ops::detail
