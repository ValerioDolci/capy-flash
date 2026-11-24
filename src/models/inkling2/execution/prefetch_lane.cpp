#include "models/inkling2/execution/prefetch_lane.h"

#include "core/device.h"
#include "core/device_scope.h"

namespace ninfer::models::inkling2::execution {

PrefetchLane::PrefetchLane(int device) : device_(device) {
    const ScopedCurrentDevice scope(device);
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&fork_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&join_, cudaEventDisableTiming));
}

PrefetchLane::~PrefetchLane() {
    int previous = 0;
    (void)cudaGetDevice(&previous);
    (void)cudaSetDevice(device_);
    (void)cudaStreamSynchronize(stream_);
    (void)cudaEventDestroy(join_);
    (void)cudaEventDestroy(fork_);
    (void)cudaStreamDestroy(stream_);
    (void)cudaSetDevice(previous);
}

} // namespace ninfer::models::inkling2::execution
