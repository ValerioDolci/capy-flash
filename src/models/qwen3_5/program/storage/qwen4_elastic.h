#pragma once

// Elastic VRAM between the KV cache and the routed-expert slot cache of Qwen4Exp
// (EngineOptions::elastic_kv).
//
// Every rank holds an ElasticArena (CUDA VMM) with the KV page planes and the four expert slot
// planes (PersistentLayout::elastic_bytes; the rest of the persistent state is an ordinary
// allocation, so tp 2 cross-device copies never touch VMM memory). Its elastic ranges are the KV
// page planes of the Text and MTP pools (pages outermost: a plane's first P pages are its first
// P * page bytes) and the elastic tail of each expert slot plane (column-major: the first A
// columns are the tail's first A * column bytes). The base slots are always backed.
//
// The manager keeps, on every rank, the backed KV pages of each pool (rank 1's pools mirror rank
// 0's page indices, so both ranks back the same pages) and the active expert columns A (the same
// on every GPU), under one invariant: each arena's physical bytes stay within its budget, the
// bytes it had at startup (all columns active, one grow step of KV pages backed). A pool that
// hands out a page past its backed pages calls grow() first (DeviceKVPagePool::set_backing): the
// pool's backing rises to the next multiple of the grow step, and while the arenas would exceed
// their budgets the cache gives up its last active column (Qwen4OffloadExperts::
// set_active_columns: each layer keeps its hottest experts, the device is idle and its tables no
// longer name the column when it returns), whose granules are released before the KV granules
// are backed. rebalance(), at the start of a request, shrinks each pool's backing to its live
// extent (the pages still allocated, e.g. a reused prefix) or the new prompt's pages, whichever
// is larger, rounded up to the grow step, and gives the freed bytes back as columns: their slots
// are empty and the cache fills them with later insertions. With a Host KV tier the KV of
// idle conversations leaves the Device while a request decodes and the live pages are packed low,
// so shrink_to_live() returns their bytes to the experts at once instead of at the next prompt.
// A request that needs more KV than fits (or than kv_budget_bytes) fails in grow() before any
// page is handed out.
//
// Execution thread only, between windows (the KV materialization points: before a prompt, before
// a decode round). Not thread-safe.

#include "core/elastic_arena.h"
#include "core/paged_kv_cache.h"
#include "models/qwen3_5/execution/qwen4_experts.h"
#include "models/qwen3_5/program/planning/startup.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// The elastic ranges of one rank's persistent layout, in the order the manager indexes them: the
// Text KV planes, the MTP KV planes, then the four expert planes' elastic tails. KV planes start
// with `step_pages` pages backed (at most the pool's pages), expert tails fully backed.
[[nodiscard]] std::vector<ElasticRangeSpec> qwen4_elastic_ranges(const PersistentLayout& layout,
                                                                 std::uint32_t step_pages);

class Qwen4ElasticVram {
public:
    struct Rank {
        ElasticArena* arena            = nullptr;
        cudaStream_t stream            = nullptr;
        const PersistentLayout* layout = nullptr;
        // Rank 0: the pools that hand out pages; rank 1: their mirrors.
        DeviceKVPagePool* text = nullptr;
        DeviceKVPagePool* mtp  = nullptr;
    };

    struct Stats {
        std::uint32_t text_pages = 0; // backed
        std::uint32_t mtp_pages  = 0;
        std::int32_t columns     = 0; // active expert columns
        std::int32_t base_slots  = 0; // per layer and GPU
        std::int64_t grows       = 0;
        std::int64_t shrinks     = 0; // column releases
        std::int64_t returns     = 0; // column returns
        std::size_t kv_bytes     = 0; // per GPU, backed KV pages
    };

    Qwen4ElasticVram(std::vector<Rank> ranks, execution::Qwen4OffloadExperts& experts,
                     std::uint32_t step_pages, std::size_t kv_budget_bytes);
    Qwen4ElasticVram(const Qwen4ElasticVram&)            = delete;
    Qwen4ElasticVram& operator=(const Qwen4ElasticVram&) = delete;

    // Before a request's prompt: `text_tokens` / `mtp_tokens` = the KV it is about to map.
    void rebalance(std::uint32_t text_tokens, std::uint32_t mtp_tokens);
    // Idle KV to the Host, between execution units with the devices idle: once the idle
    // owners' pages have left the Device and the live pages were packed low
    // (LogicalKVPageStore::compact_device_replicas), shrinks each pool's backing to its live
    // extent rounded up to the grow step and gives the freed bytes back as expert columns.
    // shrinkable() tells whether that would release anything.
    [[nodiscard]] bool shrinkable() const;
    void shrink_to_live();

    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

    [[nodiscard]] std::uint32_t step_tokens() const noexcept {
        return step_pages_ * static_cast<std::uint32_t>(kPagedKVPageSize);
    }

private:
    struct Plane {
        std::size_t range      = 0;
        std::size_t page_bytes = 0;
    };

    struct RankState {
        Rank memory;
        std::vector<Plane> text, mtp;
        std::array<std::size_t, 4> expert{};
        std::array<std::size_t, 4> column_bytes{};
        std::size_t budget = 0;
    };

    struct Backing {
        std::uint32_t text   = 0;
        std::uint32_t mtp    = 0;
        std::int32_t columns = 0;
    };

    // The backing rebalance(text_tokens, mtp_tokens) would apply.
    [[nodiscard]] Backing plan(std::uint32_t text_tokens, std::uint32_t mtp_tokens) const;
    void grow(bool mtp_pool, std::uint32_t pages);
    [[nodiscard]] std::vector<std::size_t> targets(const RankState& r, std::uint32_t text_pages,
                                                   std::uint32_t mtp_pages,
                                                   std::int32_t columns) const;
    [[nodiscard]] bool fits(std::uint32_t text_pages, std::uint32_t mtp_pages,
                            std::int32_t columns) const;
    [[nodiscard]] std::size_t kv_bytes(std::uint32_t text_pages, std::uint32_t mtp_pages) const;
    void apply(std::uint32_t text_pages, std::uint32_t mtp_pages, std::int32_t columns,
               const char* why);
    [[nodiscard]] std::uint32_t step_up(std::uint32_t pages, std::uint32_t capacity) const noexcept;

    std::vector<RankState> ranks_;
    execution::Qwen4OffloadExperts& experts_;
    std::uint32_t step_pages_    = 0;
    std::size_t kv_budget_       = 0;
    std::uint32_t text_capacity_ = 0;
    std::uint32_t mtp_capacity_  = 0;
    Stats stats_;
};

} // namespace ninfer::models::qwen3_5::detail
