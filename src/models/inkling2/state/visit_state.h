#pragma once

// The per-sequence state of the Inkling 2 Text forward on one rank: everything a token leaves for
// the tokens after it, per visit slot (config.h Visit::slot, 48 for the released schedule).
//
//   local visits   K/V on a cyclic ring of window + kVisitStateMaxDrafts positions
//                  (CyclicKVCache, one layer per local slot, slot x mod capacity holds position x)
//   global visits  K/V of every position in page-major planes [D, 64, Hkv_r, pages] read through
//                  a PagedKVBatchLayerView (the BFloat16 storage profile: K BF16, V FP16) with an
//                  identity block table of one row
//   every visit    the four short-convolution tails, BF16 [C, 3] (column 0 = three tokens back):
//                  K and V over the rank's KV channels, post-attention and post-FFN over H
//
// The convolution tails and the batch-row index are the only state that must be zero at the
// start of a sequence (reset()); K/V rows are written before any call reads them. At tp 2 each
// rank holds its own VisitState: K/V and the K/V tails over its KV heads, the post-attention and
// post-FFN tails replicated (both ranks update them identically). The layout is planned into a
// caller-owned backing (LayoutBuilder), like the engine's other fixed state.

#include "core/cyclic_kv_cache.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "models/inkling2/config.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ninfer::models::inkling2 {

// Positions per page of the global visits' K/V planes.
inline constexpr std::int32_t kVisitStatePageTokens = 64;

// The most draft columns a speculative verification call carries after its anchor. A
// verification from frontier L writes positions up to L + kVisitStateMaxDrafts and may commit
// only L + 1; with rings of window + kVisitStateMaxDrafts positions, every slot such a rejected
// position overwrites held a position no query at or after the new frontier sees, so the rings
// need no rollback (the next call rewrites the rejected positions before reading them).
inline constexpr std::uint32_t kVisitStateMaxDrafts = 7;

struct VisitStateLayout {
    std::uint32_t max_context = 0; // positions a sequence may reach
    std::uint32_t pages       = 0; // per global visit
    // Per slot: the local ring layer or the global plane index.
    std::vector<LayerKind> kinds;
    std::vector<std::uint32_t> index;
    CyclicKVCacheLayout ring;
    std::vector<TensorRegion> global_k, global_v; // per global visit
    TensorRegion table;                           // I32 [pages, 1]: identity
    // Per slot: key, value, attention, ffn tails; then the I32 [1] batch row (= 0). One
    // contiguous range [zero_begin, zero_end) that reset() clears.
    std::vector<std::array<TensorRegion, 4>> tails;
    TensorRegion row;
    std::size_t zero_begin = 0;
    std::size_t zero_end   = 0;
    std::size_t bytes      = 0;
};

// Plans the state of one rank for sequences of up to `max_context` positions. `config` is the
// rank's geometry (execution/tp.h shard_text_config at tp 2).
[[nodiscard]] VisitStateLayout plan_visit_state(LayoutBuilder& builder, const TextConfig& config,
                                                std::uint32_t max_context);

class VisitState {
public:
    // The four convolution tails of one visit slot, BF16 ne = {C, 3}.
    struct ConvTails {
        Tensor key, value, attention, ffn;
    };

    // Binds `layout` to `backing` (at least layout.bytes, on the current device) and writes the
    // identity block table. Call reset() before the first token.
    VisitState(DeviceSpan backing, const VisitStateLayout& layout);
    ~VisitState();
    VisitState(const VisitState&)            = delete;
    VisitState& operator=(const VisitState&) = delete;

    [[nodiscard]] std::uint32_t max_context() const noexcept { return layout_.max_context; }

    // Starts a new sequence: zeroes every convolution tail (stream ordered on `stream`).
    void reset(cudaStream_t stream) const;

    [[nodiscard]] bool global(std::uint32_t slot) const;
    [[nodiscard]] ConvTails tails(std::uint32_t slot) const;
    // The K/V of a local slot (ring) or of a global slot (pages); the other kind throws.
    [[nodiscard]] CyclicKVCacheLayerView ring(std::uint32_t slot) const;
    [[nodiscard]] PagedKVBatchLayerView pages(std::uint32_t slot) const;

    // Device I32 [1] = {0}: the batch row's ring lane and block-table row.
    [[nodiscard]] const Tensor& row() const noexcept { return row_; }

private:
    VisitStateLayout layout_;
    DeviceSpan backing_;
    std::unique_ptr<CyclicKVCache> ring_;
    Tensor table_, row_;
};

} // namespace ninfer::models::inkling2
