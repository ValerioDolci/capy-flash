#pragma once

// Pinned / pageable split of the HostMapped parents (pinned fallback).
//
// The HostMapped parents (Qwen4Exp routed-expert banks, `--embedding-host`) normally share one
// mapped page-locked allocation. Where the page-locked memory a process may hold is capped below
// their size (Windows WDDM, also under WSL2: about half of the physical RAM), the parents that may
// live in pageable memory (HostMappedPlacement::pageable_rank > 0) are moved there, the lowest
// ranks staying pinned first. Readers of a pageable parent: the CPU reads it in place; device
// copies go through pinned staging; no kernel reads it in place.
//
// Pure planning, no CUDA: the materializer allocates what this returns.

#include "artifact/framing.h"
#include "artifact/materializer.h"
#include "core/arena.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::artifact {

struct HostResidency {
    // Per placement, in plan order: 1 = pageable allocation, 0 = pinned allocation.
    std::vector<std::uint8_t> pageable;
    // Per placement: offset in its own allocation (alignment kept, plan order kept).
    std::vector<std::uint64_t> offset;
    std::uint64_t pinned_bytes   = 0; // size of the pinned allocation (0: none)
    std::uint64_t pageable_bytes = 0; // size of the pageable allocation (0: none)
    // The pinned bytes of the parents that must stay pinned (pageable_rank 0) alone.
    std::uint64_t required_pinned_bytes = 0;
    std::size_t pageable_objects        = 0;

    [[nodiscard]] bool split() const noexcept { return pageable_objects != 0; }
};

// Unload and load inside one process (serve --unload-endpoints): the HostMapped parents' bytes
// (Flash-Next's expert banks, --embedding-host), page-locked, kept across an Engine rebuild.
// MaterializedArtifact::release_host_weights() hands them out before the Engine is freed; a later
// materialization of the same plan (MaterializationPlan::retained_host) adopts them instead of
// allocating, page-locking and reading them again. Nothing writes these bytes after the load but
// the exclusive tier, which is refused.
struct RetainedHostWeights {
    std::unique_ptr<PinnedHostBuffer> pinned;
    std::unique_ptr<PageableHostBuffer> pageable;
    // What they were placed for: the plan's host_mapped_objects and its total, the residency the
    // load chose, and the artifact.
    std::vector<HostMappedPlacement> placements;
    std::uint64_t bytes = 0;
    HostResidency residency;
    ArtifactId artifact_id{};
    int device_count                      = 0;
    std::uint64_t host_pinned_limit_bytes = 0;
    bool host_pinned_alloc_failed         = false;

    // Whether `plan` places exactly these parents (same objects, offsets and sizes, same artifact
    // and device count) and the bytes are still held.
    [[nodiscard]] bool matches(const MaterializationPlan& plan, const ArtifactId& artifact) const;
};

// Splits `placements` (a plan's host_mapped_objects, `total_bytes` = its host_mapped_bytes) so
// that the pinned allocation holds at most `pinned_budget` bytes where possible: every
// pageable_rank 0 parent stays pinned; then, rank by rank ascending, all parents of one rank are
// pinned together when they still fit (a rank that does not fit is skipped, a later one may fit).
// pinned_budget >= total_bytes returns the plan's own layout: nothing pageable, the plan's offsets,
// pinned_bytes == total_bytes. When the rank 0 parents alone exceed the budget the result keeps
// them pinned anyway (required_pinned_bytes > pinned_budget): the caller refuses.
[[nodiscard]] HostResidency plan_host_residency(std::span<const HostMappedPlacement> placements,
                                                std::uint64_t total_bytes,
                                                std::uint64_t pinned_budget);

// True when some placement may be pageable (pageable_rank > 0).
[[nodiscard]] bool any_pageable_candidate(std::span<const HostMappedPlacement> placements) noexcept;

// The page-locked limit that free CMA memory implies. Linux counts free CMA pages in
// MemAvailable (Ubuntu 26.04's kernel 7.0 reserves its kexec-handover scratch as CMA: 9.2 GiB on
// a 64 GiB machine) and ordinary pageable pages may use them, but cudaHostRegister pins with
// FOLL_LONGTERM, which must first migrate every page out of CMA. A page-locked buffer larger than
// the RAM outside CMA never finishes registering: reclaim thrashes until systemd-oomd kills the
// process. Returns the limit for the weights and the page-locked bytes reserved after them
// (`reserved_bytes`, MaterializationPlan::host_pinned_reserved_bytes) together: the available RAM
// outside CMA minus max(2 GiB, 5 % of MemTotal) for the system and `engine_bytes` for the Engine's
// other page-locked buffers; at least 1. Returns 0 (no limit) without CMA or when the weights fit.
[[nodiscard]] std::uint64_t
cma_pinned_limit(std::uint64_t total_bytes, std::uint64_t available_bytes,
                 std::uint64_t cma_free_bytes, std::uint64_t host_mapped_bytes,
                 std::uint64_t reserved_bytes, std::uint64_t engine_bytes) noexcept;

// MemTotal, MemAvailable and CmaFree of /proc/meminfo in bytes (0 when unreadable or absent).
struct HostMemInfo {
    std::uint64_t total     = 0;
    std::uint64_t available = 0;
    std::uint64_t cma_free  = 0;
};

[[nodiscard]] HostMemInfo read_host_meminfo();

// `bytes` as "x.y GiB".
[[nodiscard]] std::string gib_text(std::uint64_t bytes);

} // namespace ninfer::artifact
