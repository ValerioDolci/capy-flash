#pragma once

// ninfer::ops - pinned-host mailbox transport for the two-device TP2 collectives.
//
// A PeerMailbox owns the pinned host memory and the per-device words the mailbox exchange (see
// src/ops/kernel/peer_exchange.cuh) runs on. Attached to a PeerEvents instance
// (PeerEvents::attach_mailbox), it becomes the transport of that stream pair's CAPTURED
// allreduce_sum calls whose payload fits a slot: one exchange kernel per device instead of the
// staged path's two event-ordered cross-device copies and two combines. Eager calls and oversized
// payloads stay on the staged path, whose results the mailbox reproduces bit for bit.
//
// LIFECYCLE. The owner (a Program) creates the mailbox once, before capturing any graph that
// uses it (cudaHostAlloc and cudaMalloc are neither stream-ordered nor capturable), attaches it
// to its PeerEvents, and destroys it only after every graph executable captured with it: the
// captured kernels bake in the slab and word addresses.
//
// SLOTS. Captured exchanges take slots round robin (take_capture_slot()). The epoch protocol
// makes two slots sufficient for any sequence of captured exchanges and graph launches (see
// peer_exchange.cuh, SLOT REUSE), so the slab never runs out and repeated or abandoned captures
// leak nothing. The only preconditions are those of every captured decode graph: its launches
// are issued on one stream, and every launch executes each captured exchange on both devices.
//
// WIDE SLOTS. A mailbox can carry a second set of slots, wider, in a pinned allocation of its own
// (`wide_slot_bytes`): an exchange that fits the ordinary slot takes an ordinary one, a wider one a
// wide slot, each set with its own round robin, flags and epochs (the SLOT REUSE argument holds per
// set: its consecutive exchanges alternate its two slots, and every exchange in between only adds
// ordering). The Program sizes the ordinary slot for one request and the wide one for a full
// batch: kept apart, the one-request exchanges run from the same small slab as without the wide
// slots. Measured in the engine: 80 KiB exchanges from a 320 KiB-slot slab cost ~2 us more each
// (+1.7 % per DFlash2 K=7 round at one request), which a slab of their own does not.
//
// FAULTS. A poller that waits too long for its peer (kPeerSpinLimit, qualified below the 2 s
// display watchdog by test_allreduce) gives up, skips its combine and sets a sticky hang word;
// hang_reported() exposes it to the owner, which fails that round and every later one.
//
// ENTRY. A rank may reach its first captured exchange only after the peer has finished every
// cross-device copy still owed from it: under WSL2 an outstanding copy stalls behind the peer's
// spinning exchange kernel (issue #1) and the pair waits on each other until the spin limit. The
// MTP draft phase ends its staged collectives with such a rendezvous; the DFlash2 draft share adds
// its own (execution/tp2/draft_tp2.inc).

#include "core/device.h" // ExecutionContext

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops {

// One byte range of device memory for an exchange to prefetch into L2 (PeerMailbox::
// set_l2_prefetch).
struct PeerL2Range {
    const void* data  = nullptr;
    std::size_t bytes = 0;
};

class PeerMailbox {
public:
    // Allocates the pinned host slab (`slots` payload slots of `slot_bytes`, rounded up to 256
    // bytes, per rank; per slot and rank one 64-byte release line per warp lane; the hang word)
    // and each device's epoch words, and loads the exchange kernels on both devices so that no
    // capture has to. A `wide_slot_bytes` above `slot_bytes` adds the wide slots (WIDE SLOTS
    // above) in a second slab laid out the same way, sharing the hang word. Throws
    // std::invalid_argument for a context without two distinct devices, fewer than two slots or
    // an empty slot, and std::runtime_error on allocation failure.
    PeerMailbox(const ExecutionContext& ec, std::size_t slot_bytes, int slots = 2,
                std::size_t wide_slot_bytes = 0);
    ~PeerMailbox();

    PeerMailbox(const PeerMailbox&)            = delete;
    PeerMailbox& operator=(const PeerMailbox&) = delete;
    PeerMailbox(PeerMailbox&&)                 = delete;
    PeerMailbox& operator=(PeerMailbox&&)      = delete;

    // True when this mailbox was created for `ec`'s device pair, in rank order.
    [[nodiscard]] bool serves(const ExecutionContext& ec) const noexcept;

    // The largest payload an ordinary slot carries.
    [[nodiscard]] std::size_t slot_bytes() const noexcept { return sets_[0].slot_bytes; }

    // The largest payload a wide slot carries; 0 without wide slots.
    [[nodiscard]] std::size_t wide_slot_bytes() const noexcept { return sets_[1].slot_bytes; }

    // The largest payload one exchange carries.
    [[nodiscard]] std::size_t max_bytes() const noexcept {
        return sets_[1].slot_bytes > sets_[0].slot_bytes ? sets_[1].slot_bytes
                                                         : sets_[0].slot_bytes;
    }

    // The slot of the next captured exchange of `bytes`, round robin within the set that carries
    // it (the ordinary slots when they fit, else the wide ones). Called once per captured call
    // site, for both ranks together.
    [[nodiscard]] int take_capture_slot(std::size_t bytes = 0) noexcept;

    // The exchanges this mailbox has carried: one per take_capture_slot() call, in either set.
    // Read before and after a capture, the difference is the number of the capture's all-reduces
    // that took the mailbox (a call on the staged path takes no slot): the transport
    // discriminator, independent of how many graph nodes an exchange or its neighbours add.
    [[nodiscard]] std::uint64_t captured_exchanges() const noexcept { return captured_exchanges_; }

    // Enqueues rank `rank`'s half of the summing exchange of `bytes` (a multiple of 16, at most
    // max_bytes()) at the 16-byte aligned `data`, resident on that rank's device, onto `stream`,
    // in the set that carries `bytes`. The caller issues both ranks' halves for one slot taken by
    // take_capture_slot(bytes), with the same `bytes`, inside one capture, and makes `rank`'s
    // device current.
    void enqueue_exchange_sum(int rank, int slot, void* data, std::size_t bytes,
                              cudaStream_t stream) const;

    // The FP32 form of enqueue_exchange_sum(): `data` holds FP32 partials (`bytes` / 4 of them),
    // replaced in place by fl32(partial_0 + partial_1), and, when `bf16_out` is non-null (16-byte
    // aligned, `bytes` / 4 BF16 elements on the same device), that sum rounded once to BF16 is
    // stored there too. Same slots, sets and protocol as the BF16 exchange, so the two interleave
    // on one mailbox.
    void enqueue_exchange_sum_f32(int rank, int slot, void* data, std::size_t bytes, void* bf16_out,
                                  cudaStream_t stream) const;

    // True once an exchange's poller gave up waiting for its peer. Sticky: the two ranks' results
    // diverged, so the owner must not trust any later round.
    [[nodiscard]] bool hang_reported() const noexcept;

    // When true, the pipelined exchange kernels (BF16 and FP32) release the next kernel of their
    // stream at entry (griddepcontrol.launch_dependents): a programmatic dependent may then start
    // while the exchange waits on the peer, so it must not read the exchanged data before its
    // dependency wait. Off by default (kernels such as the Q4/Q5 GEMVs launched as dependents read
    // their input before waiting); Qwen4Exp turns it on (its dependents are the hyper-connection
    // mixers). Applies to exchanges enqueued afterwards.
    void set_release_dependents(bool release) noexcept { release_dependents_ = release; }

    [[nodiscard]] bool release_dependents() const noexcept { return release_dependents_; }

    // L2 PREFETCH (peer_exchange.cuh). The next pipelined exchange enqueued for `rank`
    // (BF16 or FP32) also prefetches `ranges` -- rank `rank`'s device memory, at most
    // kL2PrefetchRanges of them, e.g. the leading rows of the weight the kernel after the exchange
    // streams -- into that device's L2 from blocks of its own, while the exchange waits on PCIe.
    // The ranges are consumed by that one launch (a later exchange prefetches nothing unless set
    // again); each is shrunk to whole 16-byte units, empty ones dropped. A hint never changes a
    // result bit.
    static constexpr int kL2PrefetchRanges = 8; // a Q8 weight takes two (codes, scales)
    void set_l2_prefetch(int rank, std::span<const PeerL2Range> ranges);

    // Bytes per exchange the owner's hints should cover (0: the owner sets none). The Program sets
    // it once (NINFER_TP_EXCHANGE_PREFETCH_KIB); the mailbox only stores it for the forward code.
    void set_l2_prefetch_budget(std::size_t bytes) noexcept { l2_prefetch_budget_ = bytes; }

    [[nodiscard]] std::size_t l2_prefetch_budget() const noexcept { return l2_prefetch_budget_; }

    // Test aid: sets the sticky hang word from the host, as a poller that gave up would, so the
    // owner's recovery can be exercised on hardware where no exchange ever hangs.
    void report_hang() noexcept;

private:
    // One set of slots: its pinned slab (payloads, then flag lines; the ordinary set's also holds
    // the hang word) and each device's epoch words.
    struct SlotSet {
        void* slab               = nullptr; // pinned host allocation, UVA-mapped
        std::uint8_t* payload[2] = {nullptr, nullptr};
        std::uint32_t* flags[2]  = {nullptr, nullptr}; // [slot][lane * kPeerFlagStride]
        // [slot][lane] epochs.
        std::uint32_t* device_words[2] = {nullptr, nullptr};
        std::size_t slot_bytes         = 0;
        int lanes                      = 0; // pipelined warp lanes per slot (flag lines per slot)
        int next_slot                  = 0;
    };

    // The set that carries `bytes`: the ordinary slots when they fit, else the wide ones.
    [[nodiscard]] int set_for(std::size_t bytes) const noexcept {
        return bytes <= sets_[0].slot_bytes || sets_[1].slot_bytes == 0 ? 0 : 1;
    }

    void release() noexcept;

    SlotSet sets_[2]; // ordinary, wide (slot_bytes 0: absent)
    std::uint32_t* hang_            = nullptr;
    int slots_                      = 0;
    int devices_[2]                 = {0, 0};
    bool release_dependents_        = false;
    std::size_t l2_prefetch_budget_ = 0;

    std::uint64_t captured_exchanges_ = 0; // take_capture_slot() calls

    // Per rank: the hint the next enqueue consumes. Mutable: enqueueing is const (it changes no
    // transport state), the hint is a one-shot argument of that launch.
    struct L2Hint {
        PeerL2Range ranges[kL2PrefetchRanges];
        int count = 0;
    };

    mutable L2Hint l2_hint_[2];
};

// Prefetches `ranges` (each shrunk to whole 16-byte units, at most kL2PrefetchKernelRanges, each
// at most 1 GiB) into the L2 of the stream's device with one small kernel that issues bulk
// prefetches and exits. Its blocks retire about as fast as DRAM serves the lines, so a caller that
// must not wait for them issues it on a stream of its own. Timings only, never a result bit.
inline constexpr int kL2PrefetchKernelRanges = 12;
void l2_prefetch(std::span<const PeerL2Range> ranges, cudaStream_t stream);

} // namespace ninfer::ops
