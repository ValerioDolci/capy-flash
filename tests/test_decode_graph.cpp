#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect_value(void* device, std::uint32_t expected, const char* label) {
    std::uint32_t actual  = 0;
    const cudaError_t err = cudaMemcpy(&actual, device, sizeof(actual), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " copy failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    if (actual == expected) { return 0; }
    std::cerr << label << " expected 0x" << std::hex << expected << ", got 0x" << actual << std::dec
              << '\n';
    return 1;
}

int expect(bool condition, const char* label) {
    if (condition) { return 0; }
    std::cerr << label << '\n';
    return 1;
}

bool capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    return status != cudaStreamCaptureStatusNone;
}

// One graph holding both devices' nodes: the peer stream is forked into the origin's capture and
// joined back, so a single launch on the origin stream runs both devices' work.
int exercise_dual_device_capture() {
    int failures = 0;
    ninfer::DeviceContext origin(0);
    ninfer::DeviceArena origin_storage(sizeof(std::uint32_t));
    ninfer::DeviceContext peer(1);
    ninfer::DeviceArena peer_storage(sizeof(std::uint32_t));
    origin.bind_to_current_thread();

    const ninfer::DecodeGraphPeerBridge bridge(origin.device, peer.device);
    const ninfer::DecodeGraphPeerCapture peer_capture{.bridge = &bridge, .stream = peer.stream};
    const auto write = [&](int origin_value, int peer_value) {
        CUDA_CHECK(cudaMemsetAsync(origin_storage.base(), origin_value, sizeof(std::uint32_t),
                                   origin.stream));
        peer.bind_to_current_thread();
        CUDA_CHECK(
            cudaMemsetAsync(peer_storage.base(), peer_value, sizeof(std::uint32_t), peer.stream));
        origin.bind_to_current_thread();
    };

    ninfer::DecodeGraphDefinition first;
    first.capture(origin.stream, [&] { write(0x11, 0x44); }, peer_capture);
    ninfer::DecodeGraphDefinition second;
    second.capture(origin.stream, [&] { write(0x22, 0x55); }, peer_capture);
    int current = -1;
    CUDA_CHECK(cudaGetDevice(&current));
    failures += expect(current == origin.device, "dual-device capture changed the current device");
    failures +=
        expect(first.node_count() == 2, "dual-device graph does not hold both devices' nodes");

    ninfer::DecodeGraphExecutable executable;
    executable.instantiate(first);
    executable.launch(origin.stream);
    origin.synchronize();
    failures += expect_value(origin_storage.base(), 0x11111111U, "dual-device origin launch");
    failures += expect_value(peer_storage.base(), 0x44444444U, "dual-device peer launch");

    executable.update(second);
    peer.bind_to_current_thread();
    CUDA_CHECK(cudaMemsetAsync(peer_storage.base(), 0x66, sizeof(std::uint32_t), peer.stream));
    origin.bind_to_current_thread();
    bridge.gate_launch(peer.stream, origin.stream);
    executable.launch(origin.stream);
    origin.synchronize();
    failures += expect_value(origin_storage.base(), 0x22222222U, "updated dual-device launch");
    failures += expect_value(peer_storage.base(), 0x55555555U, "gated dual-device peer launch");

    // The converse ordering. The graph's peer nodes do not run on the peer's own stream, so work
    // issued there after a launch is ordered behind them only by gate_peer_after_launch(). A bulk
    // memset ahead of the graph's peer write keeps that write late: without the converse gate the
    // marker issued after the launch usually lands first and is overwritten with 0x55.
    constexpr std::size_t kBulkBytes = std::size_t{256} << 20;
    peer.bind_to_current_thread();
    ninfer::DeviceArena peer_bulk(kBulkBytes);
    origin.bind_to_current_thread();
    ninfer::DecodeGraphDefinition slow_peer;
    slow_peer.capture(
        origin.stream,
        [&] {
            CUDA_CHECK(
                cudaMemsetAsync(origin_storage.base(), 0x22, sizeof(std::uint32_t), origin.stream));
            peer.bind_to_current_thread();
            CUDA_CHECK(cudaMemsetAsync(peer_bulk.base(), 0, kBulkBytes, peer.stream));
            CUDA_CHECK(
                cudaMemsetAsync(peer_storage.base(), 0x55, sizeof(std::uint32_t), peer.stream));
            origin.bind_to_current_thread();
        },
        peer_capture);
    ninfer::DecodeGraphExecutable slow_executable;
    slow_executable.instantiate(slow_peer);
    bridge.gate_launch(peer.stream, origin.stream);
    slow_executable.launch(origin.stream);
    bridge.gate_peer_after_launch(peer.stream, origin.stream);
    peer.bind_to_current_thread();
    CUDA_CHECK(cudaMemsetAsync(peer_storage.base(), 0x99, sizeof(std::uint32_t), peer.stream));
    peer.synchronize();
    origin.bind_to_current_thread();
    origin.synchronize();
    failures += expect_value(peer_storage.base(), 0x99999999U,
                             "peer work issued after a launch overtook the graph's peer nodes");

    bool body_error = false;
    ninfer::DecodeGraphDefinition discarded;
    try {
        discarded.capture(
            origin.stream,
            [&] {
                write(0x33, 0x77);
                throw std::runtime_error("capture body failure");
            },
            peer_capture);
    } catch (const std::runtime_error&) { body_error = true; }
    failures += expect(body_error && !discarded.ready(), "failed dual-device capture was kept");
    failures += expect(!capturing(origin.stream) && !capturing(peer.stream),
                       "failed dual-device capture left a stream capturing");

    // A memcpy whose source moves from the origin device to the peer keeps the topology but not
    // the node's context: the driver refuses the update and the executable is re-instantiated,
    // so the copy reads the peer's word (0x99999999).
    ninfer::DeviceArena origin_source(sizeof(std::uint32_t));
    ninfer::DecodeGraphDefinition from_origin;
    from_origin.capture(origin.stream, [&] {
        CUDA_CHECK(cudaMemcpyAsync(origin_storage.base(), origin_source.base(),
                                   sizeof(std::uint32_t), cudaMemcpyDefault, origin.stream));
    });
    ninfer::DecodeGraphDefinition from_peer;
    from_peer.capture(origin.stream, [&] {
        CUDA_CHECK(cudaMemcpyAsync(origin_storage.base(), peer_storage.base(),
                                   sizeof(std::uint32_t), cudaMemcpyDefault, origin.stream));
    });
    ninfer::DecodeGraphExecutable copies;
    copies.instantiate(from_origin);
    bool copy_recovered = true;
    try {
        copies.update(from_peer);
    } catch (const std::exception& error) {
        std::cerr << "rejected memcpy update threw: " << error.what() << '\n';
        copy_recovered = false;
    }
    failures += expect(copy_recovered, "a rejected memcpy update was not re-instantiated");
    origin.bind_to_current_thread();
    copies.launch(origin.stream);
    origin.synchronize();
    failures += expect_value(origin_storage.base(), 0x99999999U,
                             "the re-instantiated copy did not read the peer's word");

    bool missing_bridge = false;
    try {
        discarded.capture(origin.stream, [] {}, ninfer::DecodeGraphPeerCapture{});
    } catch (const std::invalid_argument&) { missing_bridge = true; }
    failures += expect(missing_bridge, "dual-device capture accepted a missing peer bridge");
    return failures;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    try {
        ninfer::DeviceContext device(0);
        ninfer::DeviceArena storage(sizeof(std::uint32_t));
        CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x33, sizeof(std::uint32_t), device.stream));
        device.synchronize();

        ninfer::DecodeGraphDefinition first;
        first.capture(device.stream, [&] {
            CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x11, sizeof(std::uint32_t), device.stream));
        });
        ninfer::DecodeGraphDefinition second;
        second.capture(device.stream, [&] {
            CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x22, sizeof(std::uint32_t), device.stream));
        });

        int failures = 0;
        ninfer::DecodeGraphExecutable executable;
        executable.instantiate(first);
        executable.upload(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x33333333U, "initial graph upload");

        executable.launch(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x11111111U, "first graph launch");

        executable.update(second);
        executable.upload(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x11111111U, "updated graph upload");

        executable.launch(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x22222222U, "updated graph launch");

        // A definition the executable cannot take in place (two nodes against one): a rejected
        // update re-instantiates the executable from the new definition instead of throwing, so
        // the next launch runs the new work (last memset: 0x55).
        ninfer::DecodeGraphDefinition reshaped;
        reshaped.capture(device.stream, [&] {
            CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x44, sizeof(std::uint32_t), device.stream));
            CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x55, sizeof(std::uint32_t), device.stream));
        });
        bool recovered = true;
        try {
            executable.update(reshaped);
        } catch (const std::exception& error) {
            std::cerr << "rejected update threw: " << error.what() << '\n';
            recovered = false;
        }
        failures += expect(recovered, "a rejected graph update was not re-instantiated");
        executable.launch(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x55555555U,
                                 "a re-instantiated graph did not run the new definition");

        if (count >= 2) {
            failures += exercise_dual_device_capture();
        } else {
            std::cout << "dual-device capture checks skipped: fewer than two CUDA devices\n";
        }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "decode graph test failed: " << error.what() << '\n';
        return 1;
    }
}
