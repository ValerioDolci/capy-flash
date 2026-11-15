#pragma once

// ninfer::ops::detail - private launch of the GGUF routed-expert Ops (moe_slots_gguf,
// moe_grouped_gguf).

#include "ninfer/ops/moe_slots_gguf.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kGgufMaxPairs = kMoeSlotsMaxTokens * kMoeSlotsTopK; // 80

// Where one call finds its experts (MoeGgufLayer with the row sizes resolved).
struct GgufExperts {
    const std::int32_t* table        = nullptr; // [E]
    std::int32_t experts             = 0;
    std::int32_t slots               = 0;
    const std::int64_t* slot_offset  = nullptr; // blob form
    const std::uint8_t* base         = nullptr;
    const std::uint8_t* gate_up_base = nullptr; // planar form
    const std::uint8_t* down_base    = nullptr;
    const std::int32_t* host_table   = nullptr; // grouped, blob form: zero-copy host experts
    std::int64_t gate_up_bytes       = 0;       // 1280 gate|up rows of one expert
    std::int64_t down_bytes          = 0;       // 2560 down rows of one expert
    MoeGgufFormat gate_up_format     = MoeGgufFormat::IQ3_XXS;
    MoeGgufFormat down_format        = MoeGgufFormat::IQ4_NL;
};

// ---- decode window (moe_slots_gguf) -----------------------------------------------------------

// Work list of one call, written by the plan kernel. An item is up to TMAX pairs (token,
// position) of one slot: its gate|up and down rows and its pair range; pairs are grouped by item.
struct GgufDecodeWork {
    std::int32_t n_items;
    std::int32_t item_pair_begin[kGgufMaxPairs + 1];
    const std::uint8_t* item_gate_up[kGgufMaxPairs];
    const std::uint8_t* item_down[kGgufMaxPairs];
    std::int32_t pair_token[kGgufMaxPairs];
    std::int32_t token_pair[kGgufMaxPairs]; // [t * 10 + k] -> pair, or -1 (no slot)
};

struct GgufDecodeWorkspace {
    GgufDecodeWork* work = nullptr;
    std::int8_t* xq      = nullptr; // [T][2560] q8 codes of x
    float* xd            = nullptr; // [T][80] their scales
    float* h             = nullptr; // [10T][640] FP32 h per pair
    float* partial       = nullptr; // [10T][2560] FP32 down output per pair (unweighted)
};

std::size_t gguf_decode_workspace_layout(std::int32_t tokens, std::size_t offsets[5]);

struct GgufDecodeCall {
    const __nv_bfloat16* x  = nullptr; // [T][2560]
    const std::int32_t* ids = nullptr; // [T][10]
    const float* weights    = nullptr; // [T][10]
    std::int32_t tokens     = 0;
    GgufExperts experts;
    bool accumulate         = false;
    float* out_f32          = nullptr;
    __nv_bfloat16* out_bf16 = nullptr;
};

void gguf_decode_launch(const GgufDecodeCall& call, const GgufDecodeWorkspace& ws,
                        cudaStream_t stream);

// ---- prefill widths (moe_grouped_gguf) --------------------------------------------------------

// Experts of the grouped path are counted by id: E <= kGgufGroupedMaxExperts.
inline constexpr std::int32_t kGgufGroupedMaxExperts = 1024;
// Columns of one piece (the FP32 per-assignment outputs bound the workspace).
inline constexpr std::int32_t kGgufGroupedChunkColumns = 2048;

// NINFER_GGUF_GROUPED_COLUMNS sets the piece width (a multiple of 2048 up to 16384; the
// default 2048). Every expert's weights are dequantized once per piece and tile, so a wider piece
// spreads that fixed cost over more tokens (real routing, RTX 5080 / 5070 Ti: 4096 -20 %, 8192 -32
// % of the expert kernels' time, the same bits) for kGgufGroupedChunkColumns-proportional workspace
// (FP32 per assignment: 210 MB per 2048). With the T62 exchange the call width is its group: set
// NINFER_TP_AR_GROUP to the same value.
inline std::int32_t gguf_grouped_chunk_columns() {
    static const std::int32_t value = [] {
        const char* v = std::getenv("NINFER_GGUF_GROUPED_COLUMNS");
        const int n   = v != nullptr ? std::atoi(v) : 0;
        return n >= kGgufGroupedChunkColumns && n <= 16384 && n % kGgufGroupedChunkColumns == 0
                   ? std::int32_t(n)
                   : kGgufGroupedChunkColumns;
    }();
    return value;
}

inline constexpr std::int32_t kGgufGroupedTileTokens = 64;

[[nodiscard]] constexpr std::int32_t gguf_grouped_max_tiles(std::int32_t columns,
                                                            std::int32_t experts) {
    const std::int32_t pairs = columns * kMoeSlotsTopK;
    return pairs / kGgufGroupedTileTokens + (experts < pairs ? experts : pairs) + 1;
}

struct GgufGroupedLayout {
    std::size_t counts, offsets, rank, packed_token, packed_index, tiles, tile_count,
        expert_gate_up, expert_down, h, grouped, total;
};

[[nodiscard]] GgufGroupedLayout gguf_grouped_layout(std::int32_t columns);

struct GgufGroupedPiece {
    const __nv_bfloat16* x  = nullptr; // [columns][2560]
    const std::int32_t* ids = nullptr; // [columns][10]
    const float* weights    = nullptr; // [columns][10]
    float* out              = nullptr; // [columns][2560]
    std::int32_t columns    = 0;
    GgufExperts experts;
    std::int32_t* counts                = nullptr; // [kGgufGroupedMaxExperts]
    std::int32_t* offsets               = nullptr; // [kGgufGroupedMaxExperts + 1]
    std::int32_t* rank                  = nullptr; // [columns * 10]
    std::int32_t* packed_token          = nullptr; // [columns * 10]
    std::int32_t* packed_index          = nullptr; // [columns * 10]
    int4* tiles                         = nullptr; // [max_tiles] {begin, end, expert, 0}
    std::int32_t* tile_count            = nullptr; // [1]
    const std::uint8_t** expert_gate_up = nullptr; // [kGgufGroupedMaxExperts]
    const std::uint8_t** expert_down    = nullptr; // [kGgufGroupedMaxExperts]
    __nv_bfloat16* h                    = nullptr; // [columns * 10][640] grouped order
    float* grouped                      = nullptr; // [columns * 10][2560] grouped order, unweighted
};

void gguf_grouped_launch(const GgufGroupedPiece& piece, cudaStream_t stream);

} // namespace ninfer::ops::detail
