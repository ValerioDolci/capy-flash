// Implements: include/ninfer/ops/peer_mailbox.h
//
// The pinned host slab layout of each slot set, in one allocation per set (the wide set has no
// hang word):
//
//   [ payload rank0 slot0 .. slotN-1 ][ payload rank1 slot0 .. slotN-1 ]
//   [ flags rank0 slot0 lane0..L-1 .. slotN-1 ][ flags rank1 ... ][ hang word ]
//
// Payload slots are 256-byte aligned (the vectorized exchange reads and writes 16-byte units;
// the headroom also keeps a slot's stores on distinct cache lines). Every release flag has a
// 64-byte line of its own (kPeerFlagStride): L per slot and rank, one per warp lane of the
// exchange kernel at the widest payload. Flag words sit after the payload so a payload overflow
// from a mis-sized slot cannot reach them without being loudly out of contract, and the hang word
// is last, on its own line.
//
// Each rank's epochs live in that rank's device memory: only that rank's kernels touch them, and
// the epoch is read at the start of every exchange, where a pinned host read would add a PCIe round
// trip.

#include "ninfer/ops/peer_mailbox.h"

#include "ops/kernel/peer_exchange.cuh" // detail::peer_exchange_pipelined_kernel

#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

namespace {

constexpr std::size_t kSlotAlignment = 256;

std::size_t aligned(std::size_t bytes) {
    return (bytes + kSlotAlignment - 1) & ~(kSlotAlignment - 1);
}

// Current-device save/restore for the phases that need a rank's device current.
class ScopedDevice {
public:
    ScopedDevice() { CUDA_CHECK(cudaGetDevice(&previous_)); }

    ~ScopedDevice() { (void)cudaSetDevice(previous_); }

    ScopedDevice(const ScopedDevice&)            = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;

    static void set(int device) { CUDA_CHECK(cudaSetDevice(device)); }

private:
    int previous_ = 0;
};

void require_two_devices(const ExecutionContext& ec, const char* message) {
    if (ec.tp != 2 || !ec.dev[0].has_value() || !ec.dev[1].has_value() ||
        ec.dev[0]->device == ec.dev[1]->device) {
        throw std::invalid_argument(message);
    }
}

void throw_on_error(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("PeerMailbox: ") + what + " failed: " +
                                 cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
    }
}

} // namespace

PeerMailbox::PeerMailbox(const ExecutionContext& ec, std::size_t slot_bytes, int slots,
                         std::size_t wide_slot_bytes)
    : slots_(slots) {
    require_two_devices(ec, "PeerMailbox: requires an ExecutionContext with two distinct devices");
    if (slots < 2) { throw std::invalid_argument("PeerMailbox: requires at least two slots"); }
    if (slot_bytes == 0) { throw std::invalid_argument("PeerMailbox: slot bytes must be nonzero"); }

    const int pair[2] = {ec.dev[0]->device, ec.dev[1]->device};
    devices_[0]       = pair[0];
    devices_[1]       = pair[1];

    const std::size_t set_bytes[2] = {
        aligned(slot_bytes),
        aligned(wide_slot_bytes) > aligned(slot_bytes) ? aligned(wide_slot_bytes) : 0};
    // Everything allocates into the sets first and the destructor's release() frees whatever was
    // allocated: a throwing constructor does not run the destructor, so it releases itself.
    const ScopedDevice scope;
    try {
        for (int index = 0; index < 2; ++index) {
            if (set_bytes[index] == 0) { continue; }
            SlotSet& set   = sets_[index];
            set.slot_bytes = set_bytes[index];
            set.lanes      = detail::peer_pipelined_warps(set.slot_bytes / sizeof(detail::PeerVec),
                                                          detail::kPeerPipelinedVecsPerLane);
            const std::size_t payload_bytes = static_cast<std::size_t>(slots_) * set.slot_bytes * 2;
            const std::size_t flag_words    = static_cast<std::size_t>(slots_) *
                                           static_cast<std::size_t>(set.lanes) *
                                           detail::kPeerFlagStride;
            // Both ranks' flag lines, then (ordinary set) the hang word's line.
            const std::size_t words_bytes =
                (2 * flag_words + (index == 0 ? detail::kPeerFlagStride : 0)) *
                sizeof(std::uint32_t);
            const std::size_t device_words_bytes =
                static_cast<std::size_t>(slots_) * set.lanes * sizeof(std::uint32_t);
            throw_on_error(cudaHostAlloc(&set.slab, payload_bytes + words_bytes,
                                         cudaHostAllocMapped | cudaHostAllocPortable),
                           "cudaHostAlloc");
            std::memset(set.slab, 0, payload_bytes + words_bytes);
            auto* base     = static_cast<std::uint8_t*>(set.slab);
            set.payload[0] = base;
            set.payload[1] = base + static_cast<std::size_t>(slots_) * set.slot_bytes;
            auto* flags    = reinterpret_cast<std::uint32_t*>(base + payload_bytes);
            set.flags[0]   = flags;
            set.flags[1]   = flags + flag_words;
            if (index == 0) { hang_ = flags + 2 * flag_words; }
            for (int rank = 0; rank < 2; ++rank) {
                ScopedDevice::set(pair[rank]);
                void* words = nullptr;
                throw_on_error(cudaMalloc(&words, device_words_bytes), "cudaMalloc");
                set.device_words[rank] = static_cast<std::uint32_t*>(words);
                // The legacy default stream does not order the Program's non-blocking streams, so
                // the zeroing is retired here rather than left for the first captured exchange to
                // race.
                throw_on_error(cudaMemsetAsync(words, 0, device_words_bytes, nullptr),
                               "cudaMemsetAsync");
                throw_on_error(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize");
            }
        }
        for (int rank = 0; rank < 2; ++rank) {
            ScopedDevice::set(pair[rank]);
            // Loads the exchange kernels on this device now: a module cannot be loaded inside the
            // capture that first launches it.
            cudaFuncAttributes attributes{};
            constexpr int kVecsPerLane = detail::kPeerPipelinedVecsPerLane;
            throw_on_error(cudaFuncGetAttributes(
                               &attributes, detail::peer_exchange_pipelined_kernel<kVecsPerLane>),
                           "cudaFuncGetAttributes");
            throw_on_error(
                cudaFuncGetAttributes(&attributes,
                                      detail::peer_exchange_pipelined_f32_kernel<kVecsPerLane>),
                "cudaFuncGetAttributes");
        }
    } catch (...) {
        release();
        throw;
    }
}

void PeerMailbox::release() noexcept {
    for (SlotSet& set : sets_) {
        for (int rank = 0; rank < 2; ++rank) {
            if (set.device_words[rank] == nullptr) { continue; }
            (void)cudaSetDevice(devices_[rank]);
            (void)cudaFree(set.device_words[rank]);
            set.device_words[rank] = nullptr;
        }
        if (set.slab != nullptr) { (void)cudaFreeHost(set.slab); }
        set.slab = nullptr;
    }
}

PeerMailbox::~PeerMailbox() {
    const ScopedDevice scope;
    // The owner destroys every graph executable first, but a launch may still be in flight: the
    // pinned slabs and the counters are released only once both devices retired it.
    for (int rank = 0; rank < 2; ++rank) {
        if (cudaSetDevice(devices_[rank]) != cudaSuccess) { continue; }
        const cudaError_t status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            std::fprintf(stderr, "CUDA cleanup failed during cudaDeviceSynchronize: %s: %s\n",
                         cudaGetErrorName(status), cudaGetErrorString(status));
        }
    }
    release();
}

bool PeerMailbox::serves(const ExecutionContext& ec) const noexcept {
    if (ec.tp != 2 || !ec.dev[0].has_value() || !ec.dev[1].has_value()) { return false; }
    return ec.dev[0]->device == devices_[0] && ec.dev[1]->device == devices_[1];
}

int PeerMailbox::take_capture_slot(std::size_t bytes) noexcept {
    SlotSet& set   = sets_[set_for(bytes)];
    const int slot = set.next_slot;
    set.next_slot  = (set.next_slot + 1) % slots_;
    return slot;
}

namespace {
detail::PeerPipelinedTuning pipelined_tuning(bool release_dependents) {
    detail::PeerPipelinedTuning tuning{};
    tuning.release_dependents = release_dependents;
    return tuning;
}
} // namespace

void PeerMailbox::set_l2_prefetch(int rank, std::span<const PeerL2Range> ranges) {
    if (rank < 0 || rank > 1 || ranges.size() > static_cast<std::size_t>(kL2PrefetchRanges)) {
        throw std::invalid_argument("PeerMailbox: rank or L2 prefetch range count out of range");
    }
    L2Hint hint{};
    for (const PeerL2Range& range : ranges) {
        const auto begin           = reinterpret_cast<std::uintptr_t>(range.data);
        const std::uintptr_t first = (begin + 15u) & ~std::uintptr_t{15};
        const std::uintptr_t last  = (begin + range.bytes) & ~std::uintptr_t{15};
        if (range.data == nullptr || last <= first) { continue; }
        const std::size_t bytes   = std::min<std::size_t>(last - first, std::size_t{1} << 30);
        hint.ranges[hint.count++] = {reinterpret_cast<const void*>(first), bytes};
    }
    l2_hint_[rank] = hint;
}

namespace {
// Takes rank `rank`'s pending hint into `tuning` behind `blocks` exchange blocks; returns the
// prefetching blocks to add to the grid.
template <class Hint>
int take_l2_hint(Hint& hint, int blocks, int threads, detail::PeerPipelinedTuning& tuning) {
    detail::PeerL2Prefetch& prefetch = tuning.prefetch;
    for (int r = 0; r < hint.count; ++r) {
        prefetch.data[r]  = hint.ranges[r].data;
        prefetch.bytes[r] = static_cast<std::uint32_t>(hint.ranges[r].bytes);
    }
    prefetch.ranges      = hint.count;
    prefetch.first_block = blocks;
    hint.count           = 0;
    const int extra      = detail::peer_prefetch_blocks(prefetch, threads);
    if (extra == 0) { prefetch.ranges = 0; }
    return extra;
}
} // namespace

void PeerMailbox::enqueue_exchange_sum(int rank, int slot, void* data, std::size_t bytes,
                                       cudaStream_t stream) const {
    if (rank < 0 || rank > 1 || slot < 0 || slot >= slots_) {
        throw std::invalid_argument("PeerMailbox: rank or slot out of range");
    }
    const SlotSet& set = sets_[set_for(bytes)];
    if (bytes == 0 || bytes > set.slot_bytes || (bytes % sizeof(detail::PeerVec)) != 0 ||
        (reinterpret_cast<std::uintptr_t>(data) % sizeof(detail::PeerVec)) != 0) {
        throw std::invalid_argument(
            "PeerMailbox: an exchange carries whole, aligned 16-byte vectors within one slot");
    }
    const std::size_t offset = static_cast<std::size_t>(slot) * set.slot_bytes;
    const std::size_t lane_base =
        static_cast<std::size_t>(slot) * static_cast<std::size_t>(set.lanes);
    const std::size_t flag_slot = lane_base * detail::kPeerFlagStride;
    const std::size_t vecs      = bytes / sizeof(detail::PeerVec);
    std::uint32_t* words        = set.device_words[rank];
    auto* partial               = static_cast<detail::PeerVecBf16*>(data);
    auto* mine_payload = reinterpret_cast<detail::PeerVecBf16*>(set.payload[rank] + offset);
    const auto* peer_payload =
        reinterpret_cast<const detail::PeerVecBf16*>(set.payload[1 - rank] + offset);
    constexpr int kVecsPerLane = detail::kPeerPipelinedVecsPerLane;
    constexpr int kThreads     = detail::kPeerPipelinedThreads;
    const int blocks           = detail::peer_pipelined_blocks(vecs, kVecsPerLane, kThreads);
    auto tuning                = pipelined_tuning(release_dependents_);
    const int prefetch_blocks  = take_l2_hint(l2_hint_[rank], blocks, kThreads, tuning);
    detail::peer_exchange_pipelined_kernel<kVecsPerLane>
        <<<blocks + prefetch_blocks, kThreads, 0, stream>>>(
            partial, mine_payload, set.flags[rank] + flag_slot, peer_payload,
            set.flags[1 - rank] + flag_slot, words + lane_base, hang_, static_cast<int>(vecs),
            tuning);
    CUDA_CHECK(cudaGetLastError());
}

void PeerMailbox::enqueue_exchange_sum_f32(int rank, int slot, void* data, std::size_t bytes,
                                           void* bf16_out, cudaStream_t stream) const {
    if (rank < 0 || rank > 1 || slot < 0 || slot >= slots_) {
        throw std::invalid_argument("PeerMailbox: rank or slot out of range");
    }
    const SlotSet& set = sets_[set_for(bytes)];
    if (bytes == 0 || bytes > set.slot_bytes || (bytes % sizeof(detail::PeerVec)) != 0 ||
        (reinterpret_cast<std::uintptr_t>(data) % sizeof(detail::PeerVec)) != 0 ||
        (reinterpret_cast<std::uintptr_t>(bf16_out) % sizeof(detail::PeerVec)) != 0) {
        throw std::invalid_argument(
            "PeerMailbox: an exchange carries whole, aligned 16-byte vectors within one slot");
    }
    const std::size_t offset = static_cast<std::size_t>(slot) * set.slot_bytes;
    const std::size_t lane_base =
        static_cast<std::size_t>(slot) * static_cast<std::size_t>(set.lanes);
    const std::size_t flag_slot = lane_base * detail::kPeerFlagStride;
    const std::size_t vecs      = bytes / sizeof(detail::PeerVec);
    constexpr int kVecsPerLane  = detail::kPeerPipelinedVecsPerLane;
    constexpr int kThreads      = detail::kPeerPipelinedThreads;
    const int blocks            = detail::peer_pipelined_blocks(vecs, kVecsPerLane, kThreads);
    auto tuning                 = pipelined_tuning(release_dependents_);
    const int prefetch_blocks   = take_l2_hint(l2_hint_[rank], blocks, kThreads, tuning);
    detail::peer_exchange_pipelined_f32_kernel<kVecsPerLane>
        <<<blocks + prefetch_blocks, kThreads, 0, stream>>>(
            static_cast<detail::PeerVec*>(data),
            reinterpret_cast<detail::PeerVec*>(set.payload[rank] + offset),
            set.flags[rank] + flag_slot,
            reinterpret_cast<const detail::PeerVec*>(set.payload[1 - rank] + offset),
            set.flags[1 - rank] + flag_slot, set.device_words[rank] + lane_base, hang_,
            static_cast<int>(vecs), static_cast<__nv_bfloat16*>(bf16_out), tuning);
    CUDA_CHECK(cudaGetLastError());
}

bool PeerMailbox::hang_reported() const noexcept {
    return *static_cast<volatile const std::uint32_t*>(hang_) != 0;
}

void PeerMailbox::report_hang() noexcept { *static_cast<volatile std::uint32_t*>(hang_) = 1u; }

} // namespace ninfer::ops
