#pragma once

// Prefill expert streaming of the Qwen4Exp offload backend (Qwen4OffloadExperts).
//
// A prefill chunk of a few thousand tokens routes to nearly every expert of every layer, so per
// layer the experts the device slot cache does not hold are copied from the pinned host bank into
// a ring of device slots over PCIe while the previous layer computes, and the whole layer then
// runs on the GPU (ops::moe_grouped_nvfp4 through a per-layer table: resident slot, ring slot, or
// -1 for an expert left to the host executor). The ring is a contiguous run of slot-pool slots
// the backend took from the ExpertCache for the duration (the base slots of its last few layers,
// which then stream all their experts, or the slots of elastic columns), given per window, so
// streaming costs no device memory beyond the pool.
//
// Ordering (ninfer-ext's double-buffered staging, generalized to a ring): one copy stream per
// device. Layer l's copies are issued, in id order and in runs of consecutive experts of this
// device's partition (one 2D copy per plane and run when the pool or the partition strides), when
// the compute stream reaches the MoE of layer l-1 (the host enqueues far ahead of the device, so
// the copy engine never idles between layers; at tp 2 in two parts, set_defer_to_exchange); a run
// that overwrites ring slots of an earlier layer first waits for that layer's expert kernel (its
// `done` event). The layer's table follows its
// copies on the copy stream, then `ready` is recorded; the compute stream waits for `ready`
// before the layer's expert kernel. The ring holds at least one layer (512 slots) plus a margin,
// so the copies of layer l+1 into free slots overlap the expert kernel of layer l; a ring of two
// layers or more never makes a layer's copies wait for the layer before it.

#include "core/expert_cache.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

class Qwen4ExpertStream {
public:
    struct Config {
        int device           = 0;
        std::int32_t layers  = 0; // streamable layers [0, layers): the Text layers
        std::int32_t experts = 512;
        std::array<std::int64_t, 4> plane_bytes{};
        std::vector<const std::byte*> bank_base; // [layers * 4] pinned host planes
        std::array<std::byte*, 4> pool_base{};   // device pool planes
        // Bytes between consecutive slots of plane q: 0 = plane_bytes[q] (planar pool); the slot
        // size for a slot-major pool.
        std::array<std::int64_t, 4> pool_stride{};
        std::int32_t* tables = nullptr; // device I32 [layers * experts]
        // Experts e with e % partitions == partition are this device's (tp 2: one stream per
        // GPU, each over its own link); the others are -1 in its tables and never copied.
        std::int32_t partitions = 1;
        std::int32_t partition  = 0;
        // Pinned fallback: [layers] 0 = the layer's bank is pageable and never streamed
        // (its prefill calls take the backend's CPU path); empty = every layer streams. release()
        // issues the next streamed layer's copies; exchanged(l) only when l + 1 streams.
        std::vector<std::uint8_t> streamable;
    };

    [[nodiscard]] bool streams(std::int32_t layer) const noexcept {
        return layer >= 0 && layer < config_.layers &&
               (config_.streamable.empty() ||
                config_.streamable[static_cast<std::size_t>(layer)] != 0);
    }

    // Per-window counters.
    struct Stats {
        std::int64_t layers   = 0;
        std::int64_t streamed = 0; // experts copied
        std::int64_t bytes    = 0;
        std::int64_t host     = 0; // experts left to the host executor
        std::int64_t resident = 0; // experts served by the slot cache
    };

    explicit Qwen4ExpertStream(Config config);
    ~Qwen4ExpertStream();
    Qwen4ExpertStream(const Qwen4ExpertStream&)            = delete;
    Qwen4ExpertStream& operator=(const Qwen4ExpertStream&) = delete;

    // Starts a window. `cache` (may be null: nothing resident) supplies the published residency
    // of this device's partition (its lent layers count as empty); `wait_first` (may be null) is
    // waited for before the window's first copy (the ExpertCache's lend event). The first copy
    // also waits for the last expert kernel of the previous window (its ring slots). The window's
    // ring is the pool slots [ring_first, ring_first + ring_slots), at least one layer of this
    // device's partition, which nothing else reads or writes until end_window.
    void begin_window(const expert_cache::ExpertCache* cache, cudaEvent_t wait_first,
                      std::int32_t ring_first, std::int32_t ring_slots);
    // `stream_expert[e]` != 0: expert e of `layer` is copied when not resident, else left to the
    // host (-1 in the table). Must be set before the layer's copies are issued (acquire of the
    // layer, or release of the layer before it); null = stream every non-resident expert.
    void set_selection(std::int32_t layer, const std::uint8_t* stream_expert);
    // The compute stream waits for layer's copies and table (issued now if not yet).
    void acquire(std::int32_t layer, cudaStream_t compute);
    // After the layer's expert kernel on `compute`: records its done event and issues the next
    // layer's copies (unless deferred, below).
    void release(std::int32_t layer, cudaStream_t compute);

    // At tp 2: the copies of layer l+1 are issued by the exchanges of layer l instead of
    // release(l), each part behind an event recorded on `compute`. Without P2P an exchange crosses
    // the same host link in the H2D direction as the expert copies, and a link's copies run in
    // submission order: issued at release(l), the next layer's copies (needed only a whole dense
    // block later) would delay layer l's MoE exchange on the critical path by their full duration.
    // mixer_exchanged(l), after layer l's mixer all-reduce, issues the copies that land in ring
    // slots of layers whose expert kernel is already enqueued (the ring's margin over one layer),
    // so they run under layer l's MoE; exchanged(l), after layer l's MoE all-reduce, issues the
    // rest. A layer whose exchanges never come is issued by its acquire().
    void set_defer_to_exchange(bool defer) noexcept { defer_ = defer; }

    void mixer_exchanged(std::int32_t layer, cudaStream_t compute);
    void exchanged(std::int32_t layer, cudaStream_t compute);
    // Waits for every copy issued (window end or failure).
    void end_window() noexcept;

    // Device table of `layer` (valid on the compute stream after acquire()).
    [[nodiscard]] const std::int32_t* table(std::int32_t layer) const {
        return config_.tables + static_cast<std::size_t>(layer) * config_.experts;
    }

    // Host copy of the table of `layer` as issued (slot, or -1).
    [[nodiscard]] const std::int32_t* host_table(std::int32_t layer) const {
        return host_tables_ + static_cast<std::size_t>(layer) * config_.experts;
    }

    // Some of the layer's copies were issued.
    [[nodiscard]] bool issued(std::int32_t layer) const {
        return issued_[static_cast<std::size_t>(layer)] != 0;
    }

    // Every copy of the layer and its table were issued.
    [[nodiscard]] bool complete(std::int32_t layer) const {
        return next_[static_cast<std::size_t>(layer)] >= config_.experts;
    }

    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

    [[nodiscard]] const Config& config() const noexcept { return config_; }

private:
    // Issues the layer's copies from its next expert on; with `enqueued_owners_only`, stops before
    // the first ring slot whose layer has not recorded its done event (the rest is issued later).
    void issue(std::int32_t layer, bool enqueued_owners_only = false);
    void wait_for(std::int32_t owner);

    Config config_;
    std::int64_t expert_bytes_ = 0;
    cudaStream_t copy_         = nullptr;
    std::vector<cudaEvent_t> ready_;     // [layers]
    std::vector<cudaEvent_t> done_;      // [layers]
    std::vector<cudaEvent_t> mixed_;     // [layers] after the layer's mixer all-reduce
    std::vector<cudaEvent_t> exchanged_; // [layers] after the layer's MoE all-reduce
    bool defer_                = false;
    cudaEvent_t wait_next_     = nullptr; // waited before the next issue()
    std::int32_t* host_tables_ = nullptr; // pinned [layers * experts]

    // Window state.
    const expert_cache::ExpertCache* cache_ = nullptr;
    cudaEvent_t wait_first_                 = nullptr;
    std::vector<std::uint8_t> issued_;           // [layers]
    std::vector<std::int32_t> next_;             // [layers] next expert to issue
    std::vector<std::uint8_t> recorded_;         // [layers] done event recorded in this window
    std::vector<const std::uint8_t*> selection_; // [layers]
    std::int32_t ring_first_ = 0; // the window's ring: first pool slot and slot count
    std::int32_t ring_slots_ = 0;
    std::vector<std::int32_t> owner_; // [ring_slots_] layer of this window using it, or -1
    std::int32_t cursor_        = 0;
    std::int32_t waited_        = -1; // highest layer whose done event the copy stream waits for
    bool open_                  = false;
    std::int32_t last_recorded_ = -1; // last layer of the previous window that recorded done
    Stats stats_;
};

} // namespace ninfer::models::qwen3_5::execution
