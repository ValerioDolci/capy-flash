#include "ops/linear/q8/q8_geometry.h"
#include "core/weight.h"
#include "ops/attn_input_proj/q8/q8_attn_input_kernels.h"
#include "ops/attn_input_proj/tp2/q8_dflash2_attn_input_shard.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q8/q8_schedule.cuh"
#include "ops/linear/q8/q8_mma_launch.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

// One launcher source for the DFlash2 drafter's [6144,5120] query|key|value projection and its
// two-device [3072,5120] shard: they share K and every schedule; the section output, a template
// argument of the kernels, is each problem's own.

namespace ninfer::ops::detail {
namespace {

using Geometry                          = Q8LinearGeometry<6144, 5120>;
constexpr std::int32_t kQueryRows       = 4096;
constexpr std::int32_t kKvRows          = 1024;
constexpr std::int32_t kLastSmallTokens = 48;
using ParentOutput                      = LinearBf16SegmentedOutput<kQueryRows, kKvRows, kKvRows>;
// The shard's: a rank's 16 of the 32 query heads and 4 of the 8 KV heads of D128, in the parent's
// section order.
using ShardOutput = LinearBf16SegmentedOutput<kQ8DFlash2ShardQueryRows, kQ8DFlash2ShardKvRows,
                                              kQ8DFlash2ShardKvRows>;
using Launch      = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, cudaStream_t);

// Exact input-load unrolling matters in the first two MMA column tiles. Wider blocks use
// runtime live columns with one specialization per accumulator capacity.
template <class Output, int Columns, bool Exact>
void launch_small(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                  cudaStream_t stream) {
    constexpr int Capacity = (Columns + 7) / 8 * 8;
    constexpr int Warps    = Columns <= 4 ? 16 : Columns <= 16 ? 8 : 4;
    constexpr auto Scales  = Columns <= 4 || (Columns > 8 && Columns <= 16) ? Q8ScaleAccess::Direct
                                                                            : Q8ScaleAccess::Shared;
    using Schedule         = Q8A16SlicedKMmaSchedule<Capacity, Warps, 1, 2, Scales>;
    static_assert((kQueryRows % Schedule::kBlockRows) == 0);
    static_assert((kKvRows % Schedule::kBlockRows) == 0);
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);

    const Output output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    launch_q8_a16_sliced_k_mma<
        typename Schedule::template with_problem<Geometry::kInputRows, Columns, Exact>,
        Q8SlicedKIdentityRows>(q8_linear_operands(x, weight), output, LinearIdentityEpilogue{},
                               stream);
    CUDA_CHECK(cudaGetLastError());
}

template <class Output, bool Exact, std::size_t... I>
constexpr auto make_small_launchers(std::index_sequence<I...>) {
    return std::array<Launch, sizeof...(I)>{
        &launch_small<Output, (Exact ? 1 + static_cast<int>(I) : 24 + 8 * static_cast<int>(I)),
                      Exact>...};
}

template <class Output>
constexpr auto kExactLaunchers = make_small_launchers<Output, true>(std::make_index_sequence<16>{});
template <class Output>
constexpr auto kTileLaunchers = make_small_launchers<Output, false>(std::make_index_sequence<4>{});

template <class Output>
void launch_small_t(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                    cudaStream_t stream) {
    if (x.ne[1] < 1 || x.ne[1] > kLastSmallTokens) {
        throw std::invalid_argument("Q8 DFlash2 attention input small-T: unsupported T");
    }
    if (x.ne[1] <= 16)
        kExactLaunchers<Output>[x.ne[1] - 1](x, weight, q, k, v, stream);
    else
        kTileLaunchers<Output>[(x.ne[1] - 17) / 8](x, weight, q, k, v, stream);
}

template <class Output, class Schedule>
void launch_mma(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                cudaStream_t stream) {
    const Output output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    launch_q8_a16_mma<Schedule>(q8_linear_operands(x, weight), output, LinearIdentityEpilogue{},
                                stream);
}

using MmaR32C64  = Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>;
using MmaR64C128 = Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>;
// This route owns only the partial 49..63-column tile.
using MmaR16C64K128 = Q8A16MmaSchedule<16, 64, 128, 16, 16, 1, 1>;
using MmaR32C32K128 = Q8A16MmaSchedule<32, 32, 128, 16, 16, 1, 1>;
// Three-block launch bounds reduce register usage and keep all 384 decode CTAs in one wave.
using MmaR32C64K128 = Q8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>;

} // namespace

void q8_dflash2_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                          Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_small_t<ParentOutput>(x, weight, q, k, v, stream);
}

void q8_dflash2_attn_input_mma_r32_c64_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                              Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_mma<ParentOutput, MmaR32C64>(x, weight, q, k, v, stream);
}

void q8_dflash2_attn_input_mma_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                               Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_mma<ParentOutput, MmaR64C128>(x, weight, q, k, v, stream);
}

void q8_dflash2_attn_input_mma_r16_c64_k128_launch(const Tensor& x, const Weight& w, Tensor& q,
                                                   Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_mma<ParentOutput, MmaR16C64K128>(x, w, q, k, v, stream);
}

void q8_dflash2_attn_input_mma_r32_c32_k128_launch(const Tensor& x, const Weight& w, Tensor& q,
                                                   Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_mma<ParentOutput, MmaR32C32K128>(x, w, q, k, v, stream);
}

void q8_dflash2_attn_input_mma_r32_c64_k128_launch(const Tensor& x, const Weight& w, Tensor& q,
                                                   Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_mma<ParentOutput, MmaR32C64K128>(x, w, q, k, v, stream);
}

void q8_dflash2_attn_input_shard_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_small_t<ShardOutput>(x, weight, q, k, v, stream);
}

void q8_dflash2_attn_input_shard_mma_r32_c64_launch(const Tensor& x, const Weight& weight,
                                                    Tensor& q, Tensor& k, Tensor& v,
                                                    cudaStream_t stream) {
    launch_mma<ShardOutput, MmaR32C64>(x, weight, q, k, v, stream);
}

void q8_dflash2_attn_input_shard_mma_r64_c128_launch(const Tensor& x, const Weight& weight,
                                                     Tensor& q, Tensor& k, Tensor& v,
                                                     cudaStream_t stream) {
    launch_mma<ShardOutput, MmaR64C128>(x, weight, q, k, v, stream);
}

void q8_dflash2_attn_input_shard_mma_r16_c64_k128_launch(const Tensor& x, const Weight& w,
                                                         Tensor& q, Tensor& k, Tensor& v,
                                                         cudaStream_t stream) {
    launch_mma<ShardOutput, MmaR16C64K128>(x, w, q, k, v, stream);
}

void q8_dflash2_attn_input_shard_mma_r32_c32_k128_launch(const Tensor& x, const Weight& w,
                                                         Tensor& q, Tensor& k, Tensor& v,
                                                         cudaStream_t stream) {
    launch_mma<ShardOutput, MmaR32C32K128>(x, w, q, k, v, stream);
}

void q8_dflash2_attn_input_shard_mma_r32_c64_k128_launch(const Tensor& x, const Weight& w,
                                                         Tensor& q, Tensor& k, Tensor& v,
                                                         cudaStream_t stream) {
    launch_mma<ShardOutput, MmaR32C64K128>(x, w, q, k, v, stream);
}

} // namespace ninfer::ops::detail
