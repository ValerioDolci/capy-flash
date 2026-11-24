#pragma once

// A rank's prefetch lane: a second stream of the rank's device and the two events that fork it from
// the rank's stream and join it back. A decode or verification forward issues the L2 prefetches
// of its dense projection weights there (forward/prefetch.h), so that they stream from DRAM
// beside the control kernels instead of in line with them. The owner (the caller that owns the
// rank's stream) keeps the lane for every forward call that uses it.

#include <cuda_runtime.h>

namespace ninfer::models::inkling2::execution {

class PrefetchLane {
public:
    explicit PrefetchLane(int device);
    ~PrefetchLane();

    PrefetchLane(const PrefetchLane&)            = delete;
    PrefetchLane& operator=(const PrefetchLane&) = delete;
    PrefetchLane(PrefetchLane&&)                 = delete;
    PrefetchLane& operator=(PrefetchLane&&)      = delete;

    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }

    [[nodiscard]] cudaEvent_t fork() const noexcept { return fork_; }

    [[nodiscard]] cudaEvent_t join() const noexcept { return join_; }

private:
    int device_          = 0;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t fork_    = nullptr;
    cudaEvent_t join_    = nullptr;
};

} // namespace ninfer::models::inkling2::execution
