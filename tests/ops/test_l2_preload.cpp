// ops::l2_preload writes nothing: after preloads of aligned, misaligned, tiny, empty and null
// ranges at several block counts, eagerly and replayed from a captured graph, the source and a
// guarded neighbour keep every byte. Rejects more than kL2PreloadMaxRanges ranges and no blocks.
// Whether the lines land in L2 is a timing property: bench/ops/inkling2_l2_bench.cu measures it.
#include "ninfer/ops/l2_preload.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::size_t kBytes = std::size_t{40} << 20; // more than half the L2, several ranges

std::vector<ops::L2PreloadRange> case_ranges(const std::uint8_t* base, int which) {
    switch (which) {
    case 0:
        return {{base, kBytes}};
    case 1:
        return {{base + 3, kBytes - 7}, {base + 1, 12}, {nullptr, 4096}, {base + 64, 0}};
    case 2:
        return {{base, 4096}, {base + 8192, 1 << 20}, {base + (8 << 20), 19 << 20}};
    default:
        return {};
    }
}

int run(int device) {
    cuda_check(cudaSetDevice(device), "cudaSetDevice");
    std::vector<std::uint8_t> host(kBytes);
    for (std::size_t i = 0; i < host.size(); ++i) {
        host[i] = static_cast<std::uint8_t>((i * 2654435761u) >> 13);
    }
    DeviceBuffer source(kBytes);
    source.copy_from_host(host.data(), host.size());
    GuardedDeviceBuffer neighbour(1 << 20);
    neighbour.fill(0x3c);
    const auto* base    = static_cast<const std::uint8_t*>(source.p);
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    int failures = 0;
    for (int which = 0; which < 3; ++which) {
        const auto ranges = case_ranges(base, which);
        for (const int blocks : {1, 8, 12, 64}) { ops::l2_preload(ranges, blocks, stream); }
        cudaGraph_t graph    = nullptr;
        cudaGraphExec_t exec = nullptr;
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
        ops::l2_preload(ranges, 12, stream);
        cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
        cuda_check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
        cuda_check(cudaGraphLaunch(exec, stream), "graph launch");
        cuda_check(cudaGraphLaunch(exec, stream), "graph launch");
        cuda_synchronize(stream);
        cuda_check(cudaGraphExecDestroy(exec), "destroy exec");
        cuda_check(cudaGraphDestroy(graph), "destroy graph");
    }
    const std::vector<std::uint8_t> after = from_device<std::uint8_t>(source.p, kBytes);
    if (std::memcmp(after.data(), host.data(), kBytes) != 0) {
        std::cerr << "l2_preload changed its source on device " << device << '\n';
        ++failures;
    }
    if (neighbour.verify_guards("l2_preload") != 0) { ++failures; }
    const std::vector<std::uint8_t> next = from_device<std::uint8_t>(neighbour.data(), 1 << 20);
    for (const std::uint8_t byte : next) {
        if (byte != 0x3c) {
            std::cerr << "l2_preload wrote a neighbouring buffer on device " << device << '\n';
            ++failures;
            break;
        }
    }
    const std::vector<ops::L2PreloadRange> many(ops::kL2PreloadMaxRanges + 1, {base, 64});
    try {
        ops::l2_preload(many, 8, stream);
        std::cerr << "l2_preload accepted too many ranges\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    try {
        ops::l2_preload(case_ranges(base, 0), 0, stream);
        std::cerr << "l2_preload accepted no blocks\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    ops::l2_preload({}, 8, stream); // nothing to read: no launch
    cuda_synchronize(stream);
    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    std::cout << (failures ? "FAIL" : "OK") << " l2_preload device " << device << '\n';
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int devices = 0;
    cuda_check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    int failures = 0;
    try {
        for (int d = 0; d < devices && d < 2; ++d) { failures += run(d); }
    } catch (const std::exception& error) {
        std::cerr << "l2_preload: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " l2_preload\n";
    return failures ? 1 : 0;
}
