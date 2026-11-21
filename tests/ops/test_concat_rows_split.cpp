// Two-device gather of row blocks on rank 0 (ninfer/ops/tp2/concat_rows.h): an exact copy, checked
// bit for bit against the host concatenation, eagerly over the UVA copy and over an attached
// HostExchange.
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/tp2/concat_rows.h"

#include "core/device.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

void set_device(const ExecutionContext& ec, int rank) {
    cuda_check(cudaSetDevice(ec.dev[rank]->device), "cudaSetDevice");
}

void synchronize_both(const ExecutionContext& ec) {
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        cuda_check(cudaStreamSynchronize(ec.dev[rank]->stream), "cudaStreamSynchronize");
    }
}

// Uploads run on each device's legacy stream, which DeviceContext::stream does not wait for.
void retire_staging(const ExecutionContext& ec) {
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
}

// Distinct finite BF16 bit patterns per rank and element.
std::vector<std::uint16_t> pattern(std::size_t n, std::uint32_t salt) {
    std::vector<std::uint16_t> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t h = (static_cast<std::uint32_t>(i) + salt) * 2654435761U;
        v[i]                  = static_cast<std::uint16_t>((h >> 16) & 0x7f7fU);
    }
    return v;
}

int run_case(const ExecutionContext& ec, const ops::PeerEvents& events, std::int32_t rows0,
             std::int32_t rows1, std::int32_t columns, const std::string& transport) {
    const std::string id = "concat_rows_rank0 R0=" + std::to_string(rows0) +
                           " R1=" + std::to_string(rows1) + " T=" + std::to_string(columns) + " " +
                           transport;
    const std::size_t n0 = static_cast<std::size_t>(rows0) * columns;
    const std::size_t n1 = static_cast<std::size_t>(rows1) * columns;
    const auto host0     = pattern(n0, 1U);
    const auto host1     = pattern(n1, 7U);
    std::vector<std::uint16_t> expected;
    for (std::int32_t t = 0; t < columns; ++t) {
        expected.insert(expected.end(), host0.begin() + static_cast<std::ptrdiff_t>(t) * rows0,
                        host0.begin() + static_cast<std::ptrdiff_t>(t + 1) * rows0);
        expected.insert(expected.end(), host1.begin() + static_cast<std::ptrdiff_t>(t) * rows1,
                        host1.begin() + static_cast<std::ptrdiff_t>(t + 1) * rows1);
    }

    set_device(ec, 1);
    DeviceBuffer part1 = to_device(host1);
    set_device(ec, 0);
    DeviceBuffer part0 = to_device(host0);
    GuardedDeviceBuffer staging(n1 * sizeof(std::uint16_t));
    GuardedDeviceBuffer out((n0 + n1) * sizeof(std::uint16_t));
    out.fill(0xff);
    const std::array<Tensor, 2> part{Tensor(part0.p, DType::BF16, {rows0, columns}),
                                     Tensor(part1.p, DType::BF16, {rows1, columns})};
    const Tensor tstaging(staging.data(), DType::BF16, {rows1, columns});
    Tensor tout(out.data(), DType::BF16, {rows0 + rows1, columns});
    retire_staging(ec);
    set_device(ec, 1);
    ops::concat_rows_rank0(part, tstaging, tout, ec, events);
    int current = -1;
    cuda_check(cudaGetDevice(&current), "cudaGetDevice");
    synchronize_both(ec);

    int failures = 0;
    if (current != ec.dev[1]->device) {
        std::cerr << id << ": the current device was not preserved\n";
        ++failures;
    }
    set_device(ec, 0);
    failures += verify_exact(id.c_str(), from_device<std::uint16_t>(out.data(), n0 + n1), expected);
    failures += out.verify_guards(id) + staging.verify_guards(id + " staging");
    failures += verify_exact((id + " part 0 unchanged").c_str(),
                             from_device<std::uint16_t>(part0.p, n0), host0);
    set_device(ec, 1);
    failures += verify_exact((id + " part 1 unchanged").c_str(),
                             from_device<std::uint16_t>(part1.p, n1), host1);
    set_device(ec, 0);
    std::cout << (failures ? "FAIL " : "OK ") << id << '\n';
    return failures;
}

int run_all(const ExecutionContext& ec, const ops::PeerEvents& events,
            const std::string& transport) {
    int failures = 0;
    // The two halves of the 201024-row vocabulary (decode and a short prompt), and odd row counts
    // that leave the vectorized concatenation.
    failures += run_case(ec, events, 100512, 100512, 1, transport);
    failures += run_case(ec, events, 100512, 100512, 3, transport);
    failures += run_case(ec, events, 13, 7, 5, transport);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) {
        std::cout << "SKIP: concat_rows_rank0 requires two CUDA devices, found " << device_count
                  << '\n';
        return 77;
    }
    int failures = 0;
    try {
        const ExecutionContext ec({0, 1});
        std::cout << "peer access: " << (ops::enable_peer_access(ec) ? "direct" : "host-staged")
                  << '\n';
        ops::PeerEvents events(ec);
        failures += run_all(ec, events, "uva copy");
        ops::HostExchange exchange(ec);
        events.attach_host_exchange(&exchange);
        failures += run_all(ec, events, "host exchange");
        events.attach_host_exchange(nullptr);

        set_device(ec, 0);
        DeviceBuffer a(64), b(64);
        try {
            const std::array<Tensor, 2> part{Tensor(a.p, DType::BF16, {8, 2}),
                                             Tensor(b.p, DType::BF16, {8, 2})};
            Tensor out(a.p, DType::BF16, {16, 2});
            ops::concat_rows_rank0(part, Tensor(b.p, DType::BF16, {8, 2}), out, ec, events);
            std::cerr << "concat_rows_rank0 accepted an output over part 0\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    } catch (const std::exception& error) {
        std::cerr << "concat_rows_rank0: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " concat_rows_rank0\n";
    return failures ? 1 : 0;
}
