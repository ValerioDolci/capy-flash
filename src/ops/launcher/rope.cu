// ninfer::ops - rope launcher: private token-count tuning and generic fallback.
#include "ops/launcher/rope.h"

#include "core/device.h" // CUDA_CHECK
#include "ninfer/ops/rope.h"
#include "ops/kernel/rope.cuh"
#include "ops/kernel/yarn.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops {
namespace {

// Per device: written by rope_install_text_yarn, read by text_yarn_sincos.
__device__ __constant__ detail::TextRopeYarn kTextRopeYarn;

// The table last installed, its geometry and the devices holding it; no devices: no YaRN.
struct TextYarnState {
    detail::TextRopeYarn table{};
    int rotary_dim = 0;
    float theta    = 0.0F;
    std::vector<int> devices;
};

TextYarnState& text_yarn_state() {
    static TextYarnState state;
    return state;
}

} // namespace

__device__ void text_yarn_sincos(std::int32_t position, int pair, float* sine, float* cosine) {
    const float angle = static_cast<float>(position) * kTextRopeYarn.inv_freq[pair];
    sincosf(angle, sine, cosine);
    *sine *= kTextRopeYarn.attention_scale;
    *cosine *= kTextRopeYarn.attention_scale;
}

void rope_install_text_yarn(const std::vector<int>& devices, float factor,
                            std::uint32_t original_max_position, int rotary_dim, float theta) {
    TextYarnState& state = text_yarn_state();
    state.devices.clear();
    if (!(factor > 1.0F)) { return; }
    constexpr int kPairs = sizeof(detail::TextRopeYarn::inv_freq) / sizeof(float);
    if (!std::isfinite(factor) || rotary_dim != 2 * kPairs || !(theta > 0.0F) ||
        !std::isfinite(theta) || original_max_position == 0 ||
        original_max_position > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "rope_install_text_yarn: needs a finite factor, rotary_dim 64, a positive theta and "
            "original_max_position");
    }
    rope::YarnParams params;
    params.factor                     = factor;
    params.original_max_position      = static_cast<int>(original_max_position);
    params.theta                      = theta;
    params.rotary_dim                 = rotary_dim;
    const std::vector<float> inv_freq = rope::yarn_inv_freq(params);
    std::copy(inv_freq.begin(), inv_freq.end(), state.table.inv_freq);
    state.table.attention_scale = rope::yarn_attention_scale(params);
    state.rotary_dim            = rotary_dim;
    state.theta                 = theta;
    int previous                = 0;
    CUDA_CHECK(cudaGetDevice(&previous));
    for (const int device : devices) {
        cudaError_t status = cudaSetDevice(device);
        if (status == cudaSuccess) {
            status = cudaMemcpyToSymbol(kTextRopeYarn, &state.table, sizeof(state.table));
        }
        if (status != cudaSuccess) {
            (void)cudaSetDevice(previous);
            CUDA_CHECK(status);
        }
        state.devices.push_back(device);
    }
    CUDA_CHECK(cudaSetDevice(previous));
}

} // namespace ninfer::ops

namespace ninfer::ops::detail {
namespace {

constexpr int kLargeBlock               = 256;
constexpr int kFullChunkBlock           = 192;
constexpr int kSmallBlock               = 128;
constexpr int kDefaultChunkTargetTokens = 1024;
// RTX 5090 has 170 SMs and admits six of these 256-thread CTAs per SM.
constexpr int kLargeBlockWaveCapacity = 1020;

template <RopeKernelMode Mode>
inline constexpr bool kTextMode =
    Mode == RopeKernelMode::Text1D || Mode == RopeKernelMode::TextMrope ||
    Mode == RopeKernelMode::DflashText1D;

std::int64_t token_stride(const Tensor* tensor) {
    return tensor == nullptr ? 0 : tensor->nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
}

bool bf16x2_aligned(const Tensor& tensor) {
    return (reinterpret_cast<std::uintptr_t>(tensor.data) & (alignof(__nv_bfloat162) - 1)) == 0 &&
           tensor.nb[2] % static_cast<std::int64_t>(alignof(__nv_bfloat162)) == 0;
}

template <RopeKernelMode Mode, int QHeads, int KHeads>
void launch_fixed_block(const Tensor& positions, Tensor* q, Tensor* k, int block, bool yarn,
                        cudaStream_t stream) {
    const int tokens = positions.ne[0];
    auto* kernel     = rope_fixed_kernel<Mode, QHeads, KHeads>;
    if constexpr (Mode == RopeKernelMode::Text1D || Mode == RopeKernelMode::TextMrope) {
        if (yarn) { kernel = rope_fixed_kernel<Mode, QHeads, KHeads, true>; }
    }
    kernel<<<tokens, block, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data),
        q == nullptr ? nullptr : static_cast<__nv_bfloat16*>(q->data),
        k == nullptr ? nullptr : static_cast<__nv_bfloat16*>(k->data), tokens, token_stride(q),
        token_stride(k));
}

template <RopeKernelMode Mode, int QHeads, int KHeads>
void launch_fixed(const Tensor& positions, Tensor* q, Tensor* k, bool yarn, cudaStream_t stream) {
    const int tokens = positions.ne[0];
    int block        = kSmallBlock;
    if constexpr (kTextMode<Mode>) {
        if (tokens <= 6) {
            block = (QHeads + KHeads) * 32;
        } else if (tokens <= kLargeBlockWaveCapacity) {
            block = kLargeBlock;
        } else if (tokens <= kDefaultChunkTargetTokens) {
            block = kFullChunkBlock;
        }
        const int head_warps = (QHeads + KHeads) * 32;
        if (block > head_warps) { block = head_warps; }
        if (block > 1024) { block = 1024; }
    }
    launch_fixed_block<Mode, QHeads, KHeads>(positions, q, k, block, yarn, stream);
}

template <int HeadsPerBlock, int QHeads, int KHeads>
void launch_dflash_split(const Tensor& positions, Tensor* q, Tensor* k, cudaStream_t stream) {
    constexpr int kGroups = (QHeads + KHeads + HeadsPerBlock - 1) / HeadsPerBlock;
    constexpr int kBlock  = HeadsPerBlock <= 2 ? 64 : HeadsPerBlock * 32;
    const int tokens      = positions.ne[0];
    rope_fixed_split_kernel<RopeKernelMode::DflashText1D, QHeads, KHeads, HeadsPerBlock>
        <<<tokens * kGroups, kBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(positions.data),
            q == nullptr ? nullptr : static_cast<__nv_bfloat16*>(q->data),
            k == nullptr ? nullptr : static_cast<__nv_bfloat16*>(k->data), tokens, token_stride(q),
            token_stride(k));
}

bool launch_fixed_pair(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                       bool yarn, cudaStream_t stream) {
    if (!bf16x2_aligned(q) || !bf16x2_aligned(k)) { return false; }
    const int axes = positions.ne[1];
    if (axes == 1 && q.ne[0] == 128 && rotary_dim == 128 && theta == 1.0e7F && q.ne[1] == 32 &&
        k.ne[1] == 8) {
        const int tokens = positions.ne[0];
        if (tokens <= 16) {
            launch_dflash_split<5, 32, 8>(positions, &q, &k, stream);
        } else if (tokens <= 400) {
            launch_dflash_split<8, 32, 8>(positions, &q, &k, stream);
        } else {
            launch_fixed_block<RopeKernelMode::DflashText1D, 32, 8>(positions, &q, &k, 160, false,
                                                                    stream);
        }
        return true;
    }
    if (rotary_dim == 64 && theta == 1.0e7F) {
        if (q.ne[1] == 24 && k.ne[1] == 4) {
            if (axes == 1) {
                launch_fixed<RopeKernelMode::Text1D, 24, 4>(positions, &q, &k, yarn, stream);
                return true;
            }
            if (axes == 3) {
                launch_fixed<RopeKernelMode::TextMrope, 24, 4>(positions, &q, &k, yarn, stream);
                return true;
            }
        }
        if (q.ne[1] == 16 && k.ne[1] == 2) {
            if (axes == 1) {
                launch_fixed<RopeKernelMode::Text1D, 16, 2>(positions, &q, &k, yarn, stream);
                return true;
            }
            if (axes == 3) {
                launch_fixed<RopeKernelMode::TextMrope, 16, 2>(positions, &q, &k, yarn, stream);
                return true;
            }
        }
    }
    if (axes == 2 && rotary_dim == 72 && theta == 10'000.0F && q.ne[1] == 16 && k.ne[1] == 16) {
        launch_fixed<RopeKernelMode::Vision2D, 16, 16>(positions, &q, &k, false, stream);
        return true;
    }
    return false;
}

template <RopeKernelMode Mode, int Heads>
void launch_fixed_single(const Tensor& positions, Tensor& x, bool yarn, cudaStream_t stream) {
    launch_fixed<Mode, Heads, 0>(positions, &x, nullptr, yarn, stream);
}

template <int Heads>
bool launch_text_single(const Tensor& positions, int axes, Tensor& x, bool yarn,
                        cudaStream_t stream) {
    if (x.ne[1] != Heads) { return false; }
    if (axes == 1) {
        launch_fixed_single<RopeKernelMode::Text1D, Heads>(positions, x, yarn, stream);
        return true;
    }
    if (axes == 3) {
        launch_fixed_single<RopeKernelMode::TextMrope, Heads>(positions, x, yarn, stream);
        return true;
    }
    return false;
}

bool launch_fixed_single_dispatch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                                  bool yarn, cudaStream_t stream) {
    if (!bf16x2_aligned(x)) { return false; }
    const int axes = positions.ne[1];
    if (axes == 1 && x.ne[0] == 128 && rotary_dim == 128 && theta == 1.0e7F) {
        if (x.ne[1] == 32) {
            launch_fixed_single<RopeKernelMode::DflashText1D, 32>(positions, x, false, stream);
            return true;
        }
        if (x.ne[1] == 8) {
            launch_fixed_single<RopeKernelMode::DflashText1D, 8>(positions, x, false, stream);
            return true;
        }
    }
    if (rotary_dim == 64 && theta == 1.0e7F) {
        if (launch_text_single<24>(positions, axes, x, yarn, stream) ||
            launch_text_single<4>(positions, axes, x, yarn, stream) ||
            launch_text_single<16>(positions, axes, x, yarn, stream) ||
            launch_text_single<2>(positions, axes, x, yarn, stream)) {
            return true;
        }
    }
    if (axes == 2 && rotary_dim == 72 && theta == 10'000.0F && x.ne[1] == 16) {
        launch_fixed_single<RopeKernelMode::Vision2D, 16>(positions, x, false, stream);
        return true;
    }
    return false;
}

void launch_generic(const Tensor& positions, int rotary_dim, float theta, Tensor* q, Tensor* k,
                    bool yarn, cudaStream_t stream) {
    constexpr int block = 128;
    Tensor& sample      = q != nullptr ? *q : *k;
    const int tokens    = sample.ne[2];
    auto* kernel        = yarn ? rope_generic_kernel<true> : rope_generic_kernel<false>;
    kernel<<<tokens, block, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data), positions.ne[1],
        q == nullptr ? nullptr : static_cast<__nv_bfloat16*>(q->data),
        k == nullptr ? nullptr : static_cast<__nv_bfloat16*>(k->data), sample.ne[0], rotary_dim,
        theta, q == nullptr ? 0 : q->ne[1], k == nullptr ? 0 : k->ne[1], tokens, token_stride(q),
        token_stride(k));
}

} // namespace

const TextRopeYarn* text_rope_yarn(int rotary_dim, float theta) {
    const TextYarnState& state = text_yarn_state();
    if (state.devices.empty() || rotary_dim != state.rotary_dim || theta != state.theta) {
        return nullptr;
    }
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (std::find(state.devices.begin(), state.devices.end(), device) == state.devices.end()) {
        throw std::logic_error("YaRN text RoPE is not installed on CUDA device " +
                               std::to_string(device));
    }
    return &state.table;
}

void rope_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                 cudaStream_t stream) {
    const bool yarn = text_rope_yarn(rotary_dim, theta) != nullptr;
    if (!launch_fixed_pair(positions, rotary_dim, theta, q, k, yarn, stream)) {
        launch_generic(positions, rotary_dim, theta, &q, &k, yarn, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

void rope_single_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                        cudaStream_t stream) {
    const bool yarn = text_rope_yarn(rotary_dim, theta) != nullptr;
    if (!launch_fixed_single_dispatch(positions, rotary_dim, theta, x, yarn, stream)) {
        launch_generic(positions, rotary_dim, theta, &x, nullptr, yarn, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
