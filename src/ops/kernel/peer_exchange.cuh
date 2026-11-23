#pragma once

// ninfer::ops::detail - the kernels behind the pinned-host mailbox transport for TP2 collectives.
//
// TRANSPORT CONTRACT (why this exists). Without peer access a cross-device cudaMemcpyAsync is
// staged by the driver through host memory, and the event chain that orders the staged copy costs
// far more than the payload. An exchange kernel pair replaces the whole choreography: both ranks
// run one kernel CONCURRENTLY, publish their operand into their own pinned host slot, release a
// flag, spin on the peer's flag, then sum locally -- no events, no copy engine, no driver round
// trip inside the exchange.
//
// EPOCH PROTOCOL (per slot, per warp lane and rank; see PeerMailbox for how slots are assigned).
// Each WARP is an independent mailbox lane. Warp w of the grid owns the vectors w*32*G + j*32 +
// lane, j in [0, G) -- 512*G contiguous bytes, every warp access 512 contiguous bytes -- plus one
// release flag of its own in the slot (mine_flags[w * kPeerFlagStride], a 64-byte host line each)
// and one epoch of its own (epochs[w], this device's memory, this rank's count of completed
// publishes of that lane):
//
//   publish: lane 0 reads target = epochs[w] + 1; every lane loads its G partial vectors (all in
//            flight), then stores them to the rank's host slot; __syncwarp() orders the warp's
//            stores before lane 0's release, and lane 0 stores mine_flags[w] = target with release
//            semantics at system scope (one fence.acq_rel.sys, cumulative over the warp barrier).
//   consume: lane 0 polls peer_flags[w] with ACQUIRE system loads (ld.acquire.sys) until it
//            reaches target, with __nanosleep between probes, and stores epochs[w] = target;
//            __syncwarp() hands the acquire to the warp, and every lane reads its G peer vectors
//            (ld.global.cv, all in flight) and combines them with the registers it published from.
//
// Both ranks execute the same exchanges with the same geometry, so each (slot, warp) epoch advances
// in lockstep: execution n of a slot publishes n on both ranks and waits for n from the peer, and a
// warp lane that an exchange leaves unused is skipped by both ranks, its epoch and flags staying
// equal on both. A flag left at n - 1 by the previous execution never satisfies the wait, so the
// protocol needs no host reset between graph replays and no host synchronization between launches.
// (A 0/1 flag would: without a reset the second replay would see the first replay's 1 and read a
// payload the peer has not republished.) Comparisons are wrap-safe. No warp waits on another warp
// or block of its own rank, so the kernel needs no co-residency of its blocks on either device, and
// the peer's reads of one warp's chunk start while later chunks are still being published.
//
// SLOT REUSE. A rank starts exchange k + 2 only after its exchange k + 1 observed at least one of
// the peer's k + 1 flags, which the peer releases from its k + 1 kernel, launched after its whole
// exchange k kernel (every warp's read) finished. Consecutive exchanges on two alternating slots
// therefore never overwrite a payload or flag the peer still reads, and exchanges of successive
// graph launches are separated by the launches themselves (a graph launch completes on both
// devices before the next launch on the same stream starts).
//
// HANG GUARD. A warp's poller gives up after kPeerSpinLimit probes, or earlier once another
// exchange already reported a hang, sets the pinned hang word and skips its own combine. The
// Program reads the word after every round's device synchronization and fails the round
// (PeerMailbox::hang_reported()); the word is sticky, because the ranks' results have diverged.
//
// ARITHMETIC. The combine is the qualified residual_add body: for every element float(mine) +
// float(peer) in FP32 and one __floats2bfloat162_rn, `mine` being exactly the value this rank
// published. This matches the staged path's local combine bit for bit (the same two partials, the
// same order, the same rounding), so the mailbox transport cannot change a single output value.

#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "core/pdl.cuh"

namespace ninfer::ops::detail {

// Probes of __nanosleep(100) polling before the poller gives up and reports. One probe costs about
// 0.8 us (4M probes measured 3.25 s on two RTX 5070 Ti), so the limit is ~0.8 s: long enough for
// any legitimate skew between the two ranks' schedules, short enough to surface a missing peer
// before a 2 s display watchdog would reset the device (test_allreduce checks the bound).
inline constexpr std::uint32_t kPeerSpinLimit = 1000000u;

// A poller re-reads the pinned hang word once every this many probes, so a round that already
// reported a hang does not spin the full limit again at every later exchange.
inline constexpr std::uint32_t kPeerHangProbe = 1024u;

using PeerVec = uint4; // 16 bytes = 8 BF16 elements, one PCIe transaction per access

// Words between two warps' release flags: a 64-byte host line each.
inline constexpr int kPeerFlagStride = 16;

// Sleep between two polls of the peer's flag; kPeerSpinLimit is calibrated with it.
inline constexpr std::uint32_t kPeerPollSleepNs = 100u;

// 16-byte vectors per lane of the pipelined geometry (one warp lane = 1 KiB of payload).
inline constexpr int kPeerPipelinedVecsPerLane = 2;

union PeerVecF32 {
    PeerVec raw;
    float4 value;
};

// One warp lane of the pipelined protocol, for any kernel that exchanges FP32 data with the peer
// at the pipelined geometry (warp `warp` owns vectors warp*32*G + j*32 + lane, j < G, and one
// release line / epoch of the slot): publishes `mine` (the warp's vectors, already in registers),
// releases, polls the peer's line with acquire loads (HANG GUARD as above), records the epoch and
// loads the peer's vectors into `peer`. Returns false, warp-uniformly, when the poller gave up;
// `peer` is then unspecified. The caller must call it with the whole warp converged and only for
// warps with warp*32*G < vecs.
template <int G>
__device__ __forceinline__ bool
peer_exchange_lane(int warp, int lane, const PeerVec (&mine)[G], PeerVec (&peer)[G],
                   PeerVec* mine_payload, std::uint32_t* mine_flags, const PeerVec* peer_payload,
                   std::uint32_t* peer_flags, std::uint32_t* epochs, std::uint32_t* hang, int vecs,
                   std::uint32_t spin_limit, std::uint32_t sleep_ns) {
    using SystemWord = cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system>;
    const int base   = warp * (32 * G) + lane;
#pragma unroll
    for (int j = 0; j < G; ++j) {
        const int v = base + j * 32;
        if (v < vecs) { mine_payload[v] = mine[j]; }
    }
    __syncwarp();
    int published = 1;
    if (lane == 0) {
        const std::uint32_t target = epochs[warp] + 1u;
        SystemWord(mine_flags[warp * kPeerFlagStride]).store(target, cuda::memory_order_release);
        const SystemWord flag(peer_flags[warp * kPeerFlagStride]);
        const SystemWord fault(*hang);
        std::uint32_t seen  = flag.load(cuda::memory_order_acquire);
        std::uint32_t spins = 0;
        while (static_cast<std::int32_t>(seen - target) < 0) {
            __nanosleep(sleep_ns);
            ++spins;
            if (spins % kPeerHangProbe == 0u &&
                (spins > spin_limit || fault.load(cuda::memory_order_relaxed) != 0u)) {
                fault.store(1u, cuda::memory_order_relaxed);
                published = 0;
                break;
            }
            seen = flag.load(cuda::memory_order_acquire);
        }
        epochs[warp] = target;
    }
    __syncwarp();
    if (__shfl_sync(0xffffffffu, published, 0) == 0) { return false; }
#pragma unroll
    for (int j = 0; j < G; ++j) {
        const int v = base + j * 32;
        if (v < vecs) { peer[j] = __ldcv(&peer_payload[v]); }
    }
    return true;
}

__device__ __forceinline__ PeerVecF32 peer_sum_f32(const PeerVec& a, const PeerVec& b) {
    PeerVecF32 x;
    PeerVecF32 y;
    x.raw = a;
    y.raw = b;
    PeerVecF32 sum;
    sum.value.x = __fadd_rn(x.value.x, y.value.x);
    sum.value.y = __fadd_rn(x.value.y, y.value.y);
    sum.value.z = __fadd_rn(x.value.z, y.value.z);
    sum.value.w = __fadd_rn(x.value.w, y.value.w);
    return sum;
}

// Four floats rounded once to BF16 (RNE), as 8 bytes.
__device__ __forceinline__ uint2 peer_pack_bf16x4(const float4& v) {
    const __nv_bfloat162 low  = __floats2bfloat162_rn(v.x, v.y);
    const __nv_bfloat162 high = __floats2bfloat162_rn(v.z, v.w);
    uint2 packed;
    packed.x = *reinterpret_cast<const std::uint32_t*>(&low);
    packed.y = *reinterpret_cast<const std::uint32_t*>(&high);
    return packed;
}

union PeerVecBf16 {
    PeerVec raw;
    __nv_bfloat162 pair[4];
};

// The production geometry: 16-byte vectors per lane (kPeerPipelinedVecsPerLane) and threads per
// block. Two warps per block
// spread the 40 KiB MTP-3 verify payload over 20 SMs, which keeps more PCIe reads in flight than
// 5 blocks of 256: 6.2 against 6.8 us per 40 KiB exchange, 3.8 against 4.4 at 10 KiB, on two
// RTX 5070 Ti over PCIe 5.0 x8 (tools/tp2/mailbox_probe.cu --sweep; 1, 4 or 8 vectors per lane
// are within 0.4 us, the early relaxed probe plus fence 0.6-1 us slower, the poll sleep neutral).
inline constexpr int kPeerPipelinedThreads = 64;

// L2 PREFETCH. A decode exchange keeps the DRAM of its device idle for its whole PCIe
// round trip (~5 us once both ranks arrived, 140 per 27B decode round), and the kernel that follows
// it (after an RMSNorm) is a weight-streaming GEMM whose weights depend on nothing the exchange
// produces. The owner may therefore name up to kPeerPrefetchRanges byte ranges of those weights
// (PeerMailbox::set_l2_prefetch); the exchange launch then carries extra blocks after its own,
// which do nothing but issue bulk L2 prefetches of the ranges (cp.async.bulk.prefetch.L2, one
// kPeerPrefetchChunk per thread and pass) and exit. They touch no exchange state: no epoch, flag,
// payload or partial, so the protocol, its lockstep and its arithmetic are unchanged and the
// prefetch cannot change a result bit -- only which weight lines the next GEMM finds in L2.
// The widest hint (the hyper-connection block's: a Q8 weight takes two ranges) and the plain
// block's widest. A hint of up to kPeerPrefetchRangesNarrow ranges launches the exchange with the
// narrow argument, so the plain block (all of the 27B) launches with the four-range argument.
inline constexpr int kPeerPrefetchRanges          = 8;
inline constexpr int kPeerPrefetchRangesNarrow    = 4;
inline constexpr std::uint32_t kPeerPrefetchChunk = 4096u;
inline constexpr int kPeerPrefetchMaxBlocks       = 32;

template <int R>
struct PeerL2Prefetch {
    static constexpr int kRanges = R;
    const void* data[R]          = {};
    std::uint32_t bytes[R]       = {}; // multiples of 16, data 16-byte aligned
    int ranges                   = 0;
    int first_block = 0; // blocks [first_block, gridDim.x) prefetch; the exchange's come first
};

// The prefetching blocks' whole body: worker w issues chunks w, w + workers, ... of every range.
template <int R>
__device__ __forceinline__ void peer_prefetch_l2(const PeerL2Prefetch<R>& prefetch) {
    const auto first   = static_cast<std::uint64_t>(prefetch.first_block);
    const auto worker  = (blockIdx.x - first) * blockDim.x + threadIdx.x;
    const auto workers = (gridDim.x - first) * blockDim.x;
    for (int r = 0; r < prefetch.ranges; ++r) {
        const auto* base = static_cast<const std::uint8_t*>(prefetch.data[r]);
        for (std::uint64_t offset = worker * kPeerPrefetchChunk; offset < prefetch.bytes[r];
             offset += workers * kPeerPrefetchChunk) {
            const auto bytes = static_cast<std::uint32_t>(
                min(static_cast<std::uint64_t>(kPeerPrefetchChunk), prefetch.bytes[r] - offset));
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(base + offset),
                         "r"(bytes)
                         : "memory");
        }
    }
}

// Blocks of `threads` that prefetch `prefetch`'s ranges: one chunk per thread, at most
// kPeerPrefetchMaxBlocks (each thread then loops). 0 without ranges.
template <int R>
inline int peer_prefetch_blocks(const PeerL2Prefetch<R>& prefetch, int threads) {
    std::uint64_t chunks = 0;
    for (int r = 0; r < prefetch.ranges; ++r) {
        chunks += (prefetch.bytes[r] + kPeerPrefetchChunk - 1) / kPeerPrefetchChunk;
    }
    const std::uint64_t blocks = (chunks + threads - 1) / threads;
    return static_cast<int>(blocks < kPeerPrefetchMaxBlocks ? blocks : kPeerPrefetchMaxBlocks);
}

// Launch knobs of one pipelined exchange. Production passes the defaults; the standalone probe
// varies them and, with kTimed, collects every warp's phase timestamps.
template <int R>
struct PeerPipelinedTuningT {
    std::uint32_t spin_limit   = kPeerSpinLimit;
    std::uint32_t sleep_ns     = kPeerPollSleepNs;
    unsigned long long* stamps = nullptr; // kTimed only: kPeerStampCount words per warp
    // Release the next kernel in the stream at entry (PeerMailbox::set_release_dependents).
    bool release_dependents = false;
    // Weight ranges the blocks after the exchange's own prefetch into L2 (L2 PREFETCH above).
    PeerL2Prefetch<R> prefetch{};
};

using PeerPipelinedTuning = PeerPipelinedTuningT<kPeerPrefetchRangesNarrow>;

// kTimed stamps, per warp (lane 0): globaltimer at entry, clock64 at entry, after the publish
// stores issued (the partial loads returned), after the release store issued (its system fence
// completed), when the peer's flag was seen (acquired), at the end (the peer loads returned and
// the combine stored), globaltimer at the end, and the number of polls that missed.
inline constexpr int kPeerStampCount = 8;

// Poll flavours of the pipelined kernel. Production polls with acquire loads after its release;
// the probe also measures an early relaxed first probe (issued before the publish, so its PCIe
// round trip overlaps it) followed by relaxed polls and one acquire fence.
inline constexpr int kPeerPollAcquire    = 0;
inline constexpr int kPeerPollEarlyFence = 1;

__device__ __forceinline__ unsigned long long peer_globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// One rank's half of the pipelined two-rank allreduce exchange.
//
//   partial         this rank's operand, replaced in place by the sum (this device's VRAM)
//   mine_payload    this rank's pinned host publish slot; only this kernel writes it
//   mine_flags      the slot's first release line of this rank (warp w's at w * kPeerFlagStride)
//   peer_payload    the peer's publish slot, written by the OTHER GPU while this kernel runs,
//                   hence neither const-restrict nor read through the non-coherent path
//   peer_flags      the slot's first release line of the peer
//   epochs          the slot's first warp epoch of this rank (warp w's at w; this device's VRAM)
//   hang            the pinned aggregate hang word (PeerMailbox::hang_reported())
//   vecs            payload length in 16-byte units
//
// Launch geometry: peer_pipelined_blocks(vecs, G, threads) blocks of `threads` (a multiple of 32,
// at most 256), identical on both ranks for one slot execution.
template <int G, int kPoll = kPeerPollAcquire, bool kTimed = false,
          int R = kPeerPrefetchRangesNarrow>
__global__ __launch_bounds__(256) void peer_exchange_pipelined_kernel(
    PeerVecBf16* partial, PeerVecBf16* mine_payload, std::uint32_t* mine_flags,
    const PeerVecBf16* peer_payload, std::uint32_t* peer_flags, std::uint32_t* epochs,
    std::uint32_t* hang, int vecs, PeerPipelinedTuningT<R> tuning) {
    using SystemWord = cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system>;
    // With tuning.release_dependents a programmatic dependent may start while the exchange waits
    // on the peer (the next hyper-connection mixer requests its weights then); it must not
    // read the exchanged data before its dependency wait.
    if (tuning.release_dependents && threadIdx.x == 0) { pdl::trigger_dependents(); }
    if (tuning.prefetch.ranges > 0 && static_cast<int>(blockIdx.x) >= tuning.prefetch.first_block) {
        peer_prefetch_l2(tuning.prefetch); // block-uniform: a prefetching block exchanges nothing
        return;
    }

    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int base = warp * (32 * G) + lane;
    if (warp * (32 * G) >= vecs) { return; } // warp-uniform: the grid's last warps may be idle

    unsigned long long stamp[5] = {};
    const bool timed            = kTimed && lane == 0 && tuning.stamps != nullptr;
    if (timed) {
        stamp[0] = peer_globaltimer();
        stamp[1] = clock64();
    }

    std::uint32_t target = 0;
    std::uint32_t seen   = 0;
    if (lane == 0) {
        target = epochs[warp] + 1u;
        if constexpr (kPoll == kPeerPollEarlyFence) {
            seen = SystemWord(peer_flags[warp * kPeerFlagStride]).load(cuda::memory_order_relaxed);
        }
    }

    // Publish: every load in flight before the first store. Copies go through the trivial `raw`
    // member: the union's BF16 members have non-trivial special members, which would make the
    // whole union non-copyable in device code.
    PeerVec mine[G] = {};
#pragma unroll
    for (int j = 0; j < G; ++j) {
        const int v = base + j * 32;
        if (v < vecs) { mine[j] = partial[v].raw; }
    }
#pragma unroll
    for (int j = 0; j < G; ++j) {
        const int v = base + j * 32;
        if (v < vecs) { mine_payload[v].raw = mine[j]; }
    }
    __syncwarp();
    if (timed) { stamp[2] = clock64(); }

    int published       = 1;
    std::uint32_t spins = 0;
    if (lane == 0) {
        SystemWord(mine_flags[warp * kPeerFlagStride]).store(target, cuda::memory_order_release);
        if (timed) { stamp[3] = clock64(); }
        const SystemWord peer(peer_flags[warp * kPeerFlagStride]);
        const SystemWord fault(*hang);
        if constexpr (kPoll == kPeerPollAcquire) { seen = peer.load(cuda::memory_order_acquire); }
        while (static_cast<std::int32_t>(seen - target) < 0) {
            __nanosleep(tuning.sleep_ns);
            ++spins;
            if (spins % kPeerHangProbe == 0u &&
                (spins > tuning.spin_limit || fault.load(cuda::memory_order_relaxed) != 0u)) {
                fault.store(1u, cuda::memory_order_relaxed);
                published = 0;
                break;
            }
            seen = peer.load(kPoll == kPeerPollAcquire ? cuda::memory_order_acquire
                                                       : cuda::memory_order_relaxed);
        }
        if constexpr (kPoll == kPeerPollEarlyFence) {
            // Acquire: the peer payload reads below are ordered after the observed release.
            cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
        }
        epochs[warp] = target;
        if (timed) { stamp[4] = clock64(); }
    }
    __syncwarp();
    if (__shfl_sync(0xffffffffu, published, 0) == 0) { return; }

    // Consume: every peer load in flight, then combine in place with the published registers.
    PeerVec peer[G] = {};
#pragma unroll
    for (int j = 0; j < G; ++j) {
        const int v = base + j * 32;
        if (v < vecs) { peer[j] = __ldcv(&peer_payload[v].raw); }
    }
#pragma unroll
    for (int j = 0; j < G; ++j) {
        const int v = base + j * 32;
        if (v >= vecs) { continue; }
        PeerVecBf16 sum;
        PeerVecBf16 other;
        sum.raw   = mine[j];
        other.raw = peer[j];
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float a0 = __low2float(sum.pair[pair]);
            const float b0 = __high2float(sum.pair[pair]);
            const float a1 = __low2float(other.pair[pair]);
            const float b1 = __high2float(other.pair[pair]);
            sum.pair[pair] = __floats2bfloat162_rn(a0 + a1, b0 + b1);
        }
        partial[v].raw = sum.raw;
    }
    if (timed) {
        const unsigned long long end = clock64();
        unsigned long long* out = tuning.stamps + static_cast<std::size_t>(warp) * kPeerStampCount;
        out[0]                  = stamp[0];
        out[1]                  = stamp[1];
        out[2]                  = stamp[2];
        out[3]                  = stamp[3];
        out[4]                  = stamp[4];
        out[5]                  = end;
        out[6]                  = peer_globaltimer();
        out[7]                  = spins;
    }
}

// ---- pipelined FP32 partial-sum kernel
// ------------------------------------------------------------
//
// The same pipelined protocol (per slot, per warp lane epochs and 64-byte release lines, all loads
// in flight, the HANG GUARD per warp) over FP32 partials: a 16-byte vector is four floats. The
// combine is one IEEE FP32 add per element, fl(mine + peer), which is commutative, so both ranks
// compute the same bits from the same two partials; the sum replaces the partial in place, and when
// `bf16_out` is non-null it is also stored there rounded once to BF16 (round to nearest even). This
// is the exchange of row-parallel partials that are themselves FP32 (the Qwen4Exp MoE:
// resident-expert, CPU-path and shared-expert contributions accumulated in FP32 per rank), so the
// block output is rounded once, after the cross-rank sum. A slot set mixes BF16 and FP32 exchanges
// freely: both kernels move 16-byte vectors with the same lane geometry and advance the same
// per-lane epochs.

template <int G, int R = kPeerPrefetchRangesNarrow>
__global__ __launch_bounds__(256) void peer_exchange_pipelined_f32_kernel(
    PeerVec* partial, PeerVec* mine_payload, std::uint32_t* mine_flags, const PeerVec* peer_payload,
    std::uint32_t* peer_flags, std::uint32_t* epochs, std::uint32_t* hang, int vecs,
    __nv_bfloat16* bf16_out, PeerPipelinedTuningT<R> tuning) {
    if (tuning.release_dependents && threadIdx.x == 0) { pdl::trigger_dependents(); } // as above
    if (tuning.prefetch.ranges > 0 && static_cast<int>(blockIdx.x) >= tuning.prefetch.first_block) {
        peer_prefetch_l2(tuning.prefetch);
        return;
    }
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int base = warp * (32 * G) + lane;
    if (warp * (32 * G) >= vecs) { return; } // warp-uniform: the grid's last warps may be idle

    PeerVec mine[G] = {};
#pragma unroll
    for (int j = 0; j < G; ++j) {
        const int v = base + j * 32;
        if (v < vecs) { mine[j] = partial[v]; }
    }
    PeerVec peer[G] = {};
    if (!peer_exchange_lane<G>(warp, lane, mine, peer, mine_payload, mine_flags, peer_payload,
                               peer_flags, epochs, hang, vecs, tuning.spin_limit,
                               tuning.sleep_ns)) {
        return;
    }
#pragma unroll
    for (int j = 0; j < G; ++j) {
        const int v = base + j * 32;
        if (v >= vecs) { continue; }
        const PeerVecF32 sum = peer_sum_f32(mine[j], peer[j]);
        partial[v]           = sum.raw;
        if (bf16_out != nullptr) {
            reinterpret_cast<uint2*>(bf16_out)[v] = peer_pack_bf16x4(sum.value);
        }
    }
}

// Warp lanes one pipelined exchange of `vecs` vectors uses, at `vecs_per_lane` vectors per lane.
inline int peer_pipelined_warps(std::size_t vecs, int vecs_per_lane) {
    const std::size_t per_warp = 32u * static_cast<std::size_t>(vecs_per_lane);
    const std::size_t warps    = (vecs + per_warp - 1) / per_warp;
    return warps < 1 ? 1 : static_cast<int>(warps);
}

// Blocks of `threads` for one pipelined exchange of `vecs` vectors.
inline int peer_pipelined_blocks(std::size_t vecs, int vecs_per_lane, int threads) {
    const int warps_per_block = threads / 32;
    return (peer_pipelined_warps(vecs, vecs_per_lane) + warps_per_block - 1) / warps_per_block;
}

} // namespace ninfer::ops::detail
