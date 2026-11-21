// ninfer::ops - two-device row gather on rank 0: validates, pulls rank 1's block, concatenates.
#include "ninfer/ops/tp2/concat_rows.h"

#include "core/device.h"
#include "ops/common/validate.h"
#include "ops/launcher/concat_rows.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::invalid_argument("concat_rows_rank0: " + message); }
}

bool is_bf16_matrix(const Tensor& t, std::int32_t columns) {
    return t.dtype == DType::BF16 && t.data != nullptr && t.is_contiguous() && t.ne[0] >= 1 &&
           t.ne[1] == columns && t.ne[2] == 1 && t.ne[3] == 1;
}

} // namespace

void concat_rows_rank0(const std::array<Tensor, 2>& part, const Tensor& staging, Tensor& out,
                       const ExecutionContext& ec, const PeerEvents& events) {
    require(ec.tp == 2 && ec.dev[0].has_value() && ec.dev[1].has_value(), "requires tp == 2");
    require(events.live(), "dead events");
    const std::int32_t columns = part[0].ne[1];
    require(columns >= 1, "at least one column");
    require(is_bf16_matrix(part[0], columns) && is_bf16_matrix(part[1], columns),
            "part blocks must be contiguous BF16 [R_r,T] with one column count");
    const std::int32_t rows1 = part[1].ne[0];
    require(is_bf16_matrix(staging, columns) && staging.ne[0] == rows1,
            "staging must be contiguous BF16 [R1,T]");
    require(is_bf16_matrix(out, columns) && out.ne[0] == part[0].ne[0] + rows1,
            "out must be contiguous BF16 [R0+R1,T]");
    require(!detail::overlaps(out, staging) && !detail::overlaps(out, part[0]) &&
                !detail::overlaps(staging, part[0]),
            "out, staging and part[0] must not overlap");

    const std::size_t bytes    = part[1].bytes();
    const DeviceContext& rank0 = *ec.dev[0];
    const DeviceContext& rank1 = *ec.dev[1];
    int previous               = 0;
    CUDA_CHECK(cudaGetDevice(&previous));
    CUDA_CHECK(cudaSetDevice(rank1.device));
    CUDA_CHECK(cudaEventRecord(events.inputs_ready(1), rank1.stream));
    CUDA_CHECK(cudaSetDevice(rank0.device));
    CUDA_CHECK(cudaStreamWaitEvent(rank0.stream, events.inputs_ready(1), 0));
    // Eagerly with a HostExchange attached the pull goes through its pinned host buffers (rank 1's
    // block may sit in memory rank 0 cannot address without P2P); otherwise one D2D copy over
    // UVA, the capturable form the collectives use.
    if (HostExchange* exchange = eager_exchange(ec, events)) {
        exchange->copy(1, part[1].data, staging.data, bytes, ec, events);
        CUDA_CHECK(cudaSetDevice(rank0.device));
    } else {
        CUDA_CHECK(::ninfer::memcpy_device_async(staging.data, part[1].data, bytes, rank0.stream));
    }
    CUDA_CHECK(cudaEventRecord(events.pull_done(0), rank0.stream));
    // A kernel rather than two pitched copies: a captured 2D memcpy node cannot take a change of
    // extent or pointers in place, a kernel node can.
    detail::concat_rows_bf16_launch(part[0], staging, out, rank0.stream);
    // Rank 1 may overwrite part[1] only after rank 0's pull has read it.
    CUDA_CHECK(cudaSetDevice(rank1.device));
    CUDA_CHECK(cudaStreamWaitEvent(rank1.stream, events.pull_done(0), 0));
    CUDA_CHECK(cudaSetDevice(previous));
}

} // namespace ninfer::ops
