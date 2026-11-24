#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <span>

namespace ninfer::ops {

// One byte range of device memory for l2_preload.
struct L2PreloadRange {
    const void* data  = nullptr;
    std::size_t bytes = 0;
};

inline constexpr int kL2PreloadMaxRanges = 4;
// Loads one block of l2_preload keeps in flight.
inline constexpr std::size_t kL2PreloadBlockInFlightBytes = std::size_t{32} << 10;

/**
 * Op: L2 preload of device memory
 *
 * Reads every 16-byte unit of `ranges` (at most kL2PreloadMaxRanges, each shrunk to whole 16-byte
 * units, empty ones dropped) once through L2, in range order, so that when the kernel ends their
 * lines are in the L2 of the stream's device. Unlike the bulk prefetch hints of l2_prefetch
 * (peer_mailbox.h), whose kernel retires before DRAM has served more than a few MB, the preload's
 * blocks retire only after their loads have returned.
 *
 * `blocks` (> 0) sets the rate: each block keeps kL2PreloadBlockInFlightBytes of loads in flight,
 * so the rate grows with the blocks up to DRAM bandwidth; a preload beside latency-bound kernels
 * keeps fewer in flight than saturating DRAM would need, or it stretches their DRAM latency.
 *
 * Effects:
 *   Writes nothing: no result bit of any Op changes, only which lines later kernels find in L2.
 *
 * Execution:
 *   Stream ordered on `stream`, CUDA Graph capturable; runs for as long as DRAM takes to serve the
 *   ranges at that rate. A caller that must not wait for it issues it on a stream of its own.
 */
void l2_preload(std::span<const L2PreloadRange> ranges, int blocks, cudaStream_t stream);

} // namespace ninfer::ops
