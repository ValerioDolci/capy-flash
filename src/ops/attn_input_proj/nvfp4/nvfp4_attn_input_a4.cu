#include "core/weight.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_tiles.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_a4_tma_launch.h"
#include "ops/attn_input_proj/tp2/nvfp4_attn_input_shard.h"

#include <cuda_bf16.h>

#include <cstdint>

// One A4 launcher source for the [14336,5120] parent and its [7168,5120] two-device shard: they
// share K, so the routes and schedules are the same; only the section output and the TMA launcher
// (one per output, in the non-RDC archive) differ. The shard's own TMA choice at T=1024 wraps this
// launcher (tp2/nvfp4_attn_input_shard_a4.cu).

namespace ninfer::ops::detail {
namespace {

using Geometry = Nvfp4N14336K5120;

using TmaLaunch = void (*)(const Nvfp4A4Operands&, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
                           __nv_bfloat16*, cudaStream_t);

using A4Route = Nvfp4AttnInputA4Route;

template <class Output, class Schedule>
void launch_gemm(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                 Nvfp4A4Workspace workspace, std::int32_t tokens, cudaStream_t stream) {
    launch_nvfp4_a4_mma<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a4_operands(weight, workspace, tokens, Nvfp4ScaleLayout::RowMajor),
        Output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
               static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)},
        LinearIdentityEpilogue{}, stream);
}

template <TmaLaunch Tma, Nvfp4ScaleLayout Layout>
void launch_tma(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                Nvfp4A4Workspace workspace, std::int32_t tokens, cudaStream_t stream) {
    Tma(nvfp4_a4_operands(weight, workspace, tokens, Layout), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(v.data), stream);
}

template <class Output, class Schedule>
constexpr A4Route mma_route() {
    return {launch_gemm<Output, Schedule>, Nvfp4ScaleLayout::RowMajor};
}

template <TmaLaunch Tma, Nvfp4ScaleLayout Layout>
constexpr A4Route tma_route() {
    return {launch_tma<Tma, Layout>, Layout};
}

template <class Output, TmaLaunch Tma>
A4Route select_a4(std::int32_t tokens) {
    if (tokens >= 1024) return tma_route<Tma, Nvfp4ScaleLayout::Tiled256>();
    if (tokens >= kNvfp4AttnInputFamilyFirstTmaTokens)
        return tma_route<Tma, Nvfp4ScaleLayout::Tiled128>();
    if (tokens <= 64) return mma_route<Output, Nvfp4A4T32R64>();
    if (tokens <= 96) return mma_route<Output, Nvfp4A4T32R128>();
    if (tokens <= 128) return mma_route<Output, Nvfp4A4T128R128Pipelined>();
    if (tokens <= 192) return mma_route<Output, Nvfp4A4T64R128>();
    if (tokens <= 384) return mma_route<Output, Nvfp4A4T128R128Resident>();
    return mma_route<Output, Nvfp4A4T128R128Pipelined>();
}

template <class Output, TmaLaunch Tma>
void launch_a4(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
               Nvfp4A4Workspace workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    const A4Route route       = select_a4<Output, Tma>(tokens);
    launch_nvfp4_a4_quantize(x, weight, workspace, route.scales, stream);
    route.launch(weight, q, gate, k, v, workspace, tokens, stream);
}

} // namespace

Nvfp4AttnInputA4Route nvfp4_attn_input_select_a4(std::int32_t tokens) {
    return select_a4<Nvfp4AttentionInputOutput, launch_nvfp4_a4_tma_attention>(tokens);
}

Nvfp4AttnInputA4Route nvfp4_attn_input_shard_select_a4(std::int32_t tokens) {
    return select_a4<Nvfp4AttnInputShardOutput, launch_nvfp4_a4_tma_attention_shard>(tokens);
}

void nvfp4_attn_input_a4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                Tensor& k, Tensor& v, Nvfp4A4Workspace workspace,
                                cudaStream_t stream) {
    launch_a4<Nvfp4AttentionInputOutput, launch_nvfp4_a4_tma_attention>(x, weight, q, gate, k, v,
                                                                        workspace, stream);
}

void nvfp4_attn_input_shard_a4_launch_upstream(const Tensor& x, const Weight& weight, Tensor& q,
                                               Tensor& gate, Tensor& k, Tensor& v,
                                               Nvfp4A4Workspace workspace, cudaStream_t stream) {
    launch_a4<Nvfp4AttnInputShardOutput, launch_nvfp4_a4_tma_attention_shard>(x, weight, q, gate, k,
                                                                              v, workspace, stream);
}

} // namespace ninfer::ops::detail
