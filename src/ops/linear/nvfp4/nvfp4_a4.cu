#include "core/weight.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_a4_quantize.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_tma_launch.h"
#include "ninfer/ops/nvfp4_activation.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

template <class ActivationGeometry>
void launch_quantize_exact(const Tensor& x, const Weight& weight, Nvfp4A4Workspace workspace,
                           Nvfp4ScaleLayout layout, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    constexpr int kThreads    = 256;
    const auto* input         = static_cast<const __nv_bfloat16*>(x.data);
    // The tiled plane is allocated and addressed over whole tiles, so its launch has to reach the
    // padding of the last one; the row-major plane stops at the real token count.
    const std::int32_t written_tokens =
        layout != Nvfp4ScaleLayout::RowMajor ? nvfp4_a4_padded_tokens(tokens, layout) : tokens;
    const std::int32_t tasks = written_tokens * ActivationGeometry::kGroupsPerRow;
    const int blocks         = (tasks + kThreads - 1) / kThreads;
    // The tiled plane exists only for a K whose group count divides into whole tiles. Registered
    // Ks all do; gating the instantiation keeps a future K that does not out of a hard compile
    // error and into a runtime message.
    if constexpr (ActivationGeometry::kGroupsPerRow % kNvfp4ScaleTileGroups == 0) {
        if (layout != Nvfp4ScaleLayout::RowMajor) {
            if (layout == Nvfp4ScaleLayout::Tiled128) {
                nvfp4_a4_quantize_kernel<ActivationGeometry, kThreads, Nvfp4ScaleLayout::Tiled128>
                    <<<blocks, kThreads, 0, stream>>>(input, workspace.codes, workspace.scales,
                                                      tokens, written_tokens,
                                                      weight.input_scale_divisor);
            } else {
                nvfp4_a4_quantize_kernel<ActivationGeometry, kThreads, Nvfp4ScaleLayout::Tiled256>
                    <<<blocks, kThreads, 0, stream>>>(input, workspace.codes, workspace.scales,
                                                      tokens, written_tokens,
                                                      weight.input_scale_divisor);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    } else if (layout != Nvfp4ScaleLayout::RowMajor) {
        throw std::invalid_argument("nvfp4 A4 tiled scales need K groups in whole tiles");
    }
    nvfp4_a4_quantize_kernel<ActivationGeometry, kThreads, Nvfp4ScaleLayout::RowMajor>
        <<<blocks, kThreads, 0, stream>>>(input, workspace.codes, workspace.scales, tokens,
                                          written_tokens, weight.input_scale_divisor);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_nvfp4_a4_quantize(const Tensor& x, const Weight& weight, Nvfp4A4Workspace& workspace,
                              Nvfp4ScaleLayout layout, cudaStream_t stream) {
    if (ScopedNvfp4Activation* scope = ScopedNvfp4Activation::find(x.data, x.ne[1])) {
        // Sequence-parallel prefill hand-off (ninfer/ops/nvfp4_activation.h).
        Nvfp4ActivationPlane& plane = scope->plane();
        const std::int32_t tokens   = x.ne[1];
        const std::size_t code_bytes =
            static_cast<std::size_t>(tokens) * (static_cast<std::size_t>(weight.k) / 2);
        const std::size_t scale_bytes =
            static_cast<std::size_t>(layout != Nvfp4ScaleLayout::RowMajor
                                         ? nvfp4_a4_padded_tokens(tokens, layout)
                                         : tokens) *
            (static_cast<std::size_t>(weight.k) / 16);
        if (plane.codes == nullptr || plane.scales == nullptr || plane.code_capacity < code_bytes ||
            plane.scale_capacity < scale_bytes) {
            throw std::invalid_argument("NVFP4 activation hand-off: the plane is too small");
        }
        workspace = {static_cast<std::uint8_t*>(plane.codes),
                     static_cast<std::uint8_t*>(plane.scales), plane.scale_capacity};
        scope->mark_used();
        if (scope->mode() == ScopedNvfp4Activation::Mode::Consume) {
            if (plane.layout != static_cast<int>(layout) || plane.k != weight.k ||
                plane.code_bytes != code_bytes || plane.scale_bytes != scale_bytes) {
                throw std::invalid_argument(
                    "NVFP4 activation hand-off: the consuming projection reads another layout "
                    "than the capture wrote");
            }
            return;
        }
        plane.layout      = static_cast<int>(layout);
        plane.k           = weight.k;
        plane.code_bytes  = code_bytes;
        plane.scale_bytes = scale_bytes;
    }
    if (workspace.codes == nullptr || workspace.scales == nullptr) {
        throw std::invalid_argument("nvfp4 A4 requires caller workspace");
    }
    // The tiled layout writes the padding of the last tile, so it needs a plane allocated over the
    // padded token count. Check that here, where the layout is acted on, rather than trust each
    // caller to have sized it that way.
    if (layout != Nvfp4ScaleLayout::RowMajor) {
        const std::size_t needed =
            static_cast<std::size_t>(nvfp4_a4_padded_tokens(x.ne[1], layout)) *
            (static_cast<std::size_t>(weight.k) / 16);
        if (workspace.scale_bytes < needed) {
            throw std::invalid_argument(
                "nvfp4 A4 tiled scales need a plane allocated over whole token tiles");
        }
    }
    switch (weight.k) {
    case Nvfp4Activation5120Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation5120Geometry>(x, weight, workspace, layout, stream);
        return;
    case Nvfp4Activation6144Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation6144Geometry>(x, weight, workspace, layout, stream);
        return;
    case Nvfp4Activation17408Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation17408Geometry>(x, weight, workspace, layout, stream);
        return;
    case Nvfp4Activation8704Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation8704Geometry>(x, weight, workspace, layout, stream);
        return;
    case Nvfp4Activation3072Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation3072Geometry>(x, weight, workspace, layout, stream);
        return;
    case Nvfp4Activation14336Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation14336Geometry>(x, weight, workspace, layout, stream);
        return;
    case Nvfp4Activation7168Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation7168Geometry>(x, weight, workspace, layout, stream);
        return;
    default:
        throw std::invalid_argument("nvfp4 A4 quantize: unsupported K");
    }
}

} // namespace ninfer::ops::detail

namespace ninfer::ops {
namespace {
thread_local ScopedNvfp4Activation* activation_scopes = nullptr;
} // namespace

std::size_t nvfp4_activation_code_capacity(std::int32_t tokens, std::int32_t k) {
    return detail::nvfp4_a4_checked_bytes(tokens, static_cast<std::size_t>(k) / 2);
}

std::size_t nvfp4_activation_scale_capacity(std::int32_t tokens, std::int32_t k) {
    return detail::nvfp4_a4_checked_bytes(detail::nvfp4_a4_padded_tokens(tokens),
                                          static_cast<std::size_t>(k) / 16);
}

ScopedNvfp4Activation::ScopedNvfp4Activation(Mode mode, const void* input, std::int32_t tokens,
                                             Nvfp4ActivationPlane& plane)
    : mode_(mode), input_(input), tokens_(tokens), plane_(&plane), previous_(activation_scopes) {
    if (input == nullptr || tokens <= 0) {
        throw std::invalid_argument("NVFP4 activation hand-off: invalid input or token count");
    }
    activation_scopes = this;
}

ScopedNvfp4Activation::~ScopedNvfp4Activation() {
    // Scopes nest; unlink this one wherever it sits so an exception unwinding several stays sound.
    for (ScopedNvfp4Activation** link = &activation_scopes; *link != nullptr;
         link                         = &(*link)->previous_) {
        if (*link == this) {
            *link = previous_;
            break;
        }
    }
}

ScopedNvfp4Activation* ScopedNvfp4Activation::find(const void* input,
                                                   std::int32_t tokens) noexcept {
    for (ScopedNvfp4Activation* scope = activation_scopes; scope != nullptr;
         scope                        = scope->previous_) {
        if (scope->input_ == input && scope->tokens_ == tokens) { return scope; }
    }
    return nullptr;
}
} // namespace ninfer::ops
