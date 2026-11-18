#include "core/weight.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_tiles.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_a4_tma_launch.h"
#include "ops/gdn_input_proj/tp2/nvfp4_gdn_input_shard.h"

// One A4 launcher source for the [16384,5120] parent and its [8192,5120] two-device shard: they
// share K, so the routes and schedules are the same; only the qkv|z section output and the TMA
// launcher (one per output, in the non-RDC archive) differ.

namespace ninfer::ops::detail {
namespace {

using Geometry = Nvfp4N16384K5120;

using TmaLaunch = void (*)(const Nvfp4A4Operands&, __nv_bfloat16*, __nv_bfloat16*, cudaStream_t);

using A4Route = Nvfp4GdnInputA4Route;

template <class Output, class Schedule>
void launch_gemm(const Weight& weight, Tensor& qkv, Tensor& z, Nvfp4A4Workspace workspace,
                 std::int32_t tokens, cudaStream_t stream) {
    launch_nvfp4_a4_mma<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a4_operands(weight, workspace, tokens, Nvfp4ScaleLayout::RowMajor),
        Output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)},
        LinearIdentityEpilogue{}, stream);
}

template <TmaLaunch Tma>
void launch_tma(const Weight& weight, Tensor& qkv, Tensor& z, Nvfp4A4Workspace workspace,
                std::int32_t tokens, cudaStream_t stream) {
    Tma(nvfp4_a4_operands(weight, workspace, tokens, Nvfp4ScaleLayout::Tiled256),
        static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data), stream);
}

template <class Output, class Schedule>
constexpr A4Route mma_route() {
    return {launch_gemm<Output, Schedule>, Nvfp4ScaleLayout::RowMajor};
}

template <class Output, TmaLaunch Tma>
A4Route select_a4(std::int32_t tokens) {
    if (tokens >= kNvfp4GdnInputFamilyFirstTmaTokens)
        return {launch_tma<Tma>, Nvfp4ScaleLayout::Tiled256};
    if (tokens <= 64) return mma_route<Output, Nvfp4A4T32R64>();
    if (tokens <= 96) return mma_route<Output, Nvfp4A4T32R128>();
    if (tokens <= 128) return mma_route<Output, Nvfp4A4T128R128Pipelined>();
    if (tokens <= 192) return mma_route<Output, Nvfp4A4T64R128>();
    return mma_route<Output, Nvfp4A4T128R128Resident>();
}

template <class Output, TmaLaunch Tma>
void launch_a4(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
               Nvfp4A4Workspace workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    const A4Route route       = select_a4<Output, Tma>(tokens);
    launch_nvfp4_a4_quantize(x, weight, workspace, route.scales, stream);
    route.launch(weight, qkv, z, workspace, tokens, stream);
}

} // namespace

Nvfp4GdnInputA4Route nvfp4_gdn_input_select_a4(std::int32_t tokens) {
    return select_a4<Nvfp4GdnInputOutput, launch_nvfp4_a4_tma_gdn>(tokens);
}

Nvfp4GdnInputA4Route nvfp4_gdn_input_shard_select_a4(std::int32_t tokens) {
    return select_a4<Nvfp4GdnInputShardOutput, launch_nvfp4_a4_tma_gdn_shard>(tokens);
}

void nvfp4_gdn_input_a4_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                               Nvfp4A4Workspace workspace, cudaStream_t stream) {
    launch_a4<Nvfp4GdnInputOutput, launch_nvfp4_a4_tma_gdn>(x, weight, qkv, z, workspace, stream);
}

void nvfp4_gdn_input_shard_a4_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                     Nvfp4A4Workspace workspace, cudaStream_t stream) {
    launch_a4<Nvfp4GdnInputShardOutput, launch_nvfp4_a4_tma_gdn_shard>(x, weight, qkv, z, workspace,
                                                                       stream);
}

} // namespace ninfer::ops::detail
