#include "core/weight.h"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h"

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_gemm_mma.cuh"
#include "ops/launcher/kernel_attr_once.h"

#include "core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

constexpr int kN                  = 48;
constexpr int kK                  = 5120;
constexpr int kThreads            = 256;
constexpr int kLogicalRows        = 2 * kN;
constexpr int kSmallTMax          = 8;
constexpr int kSmallTKSlice       = 512;
constexpr int kSmallTSplits       = kK / kSmallTKSlice;
constexpr int kSmallTRowsPerBlock = 4;
constexpr int kSmallTThreads      = kSmallTRowsPerBlock * 32;
static_assert(kK % kSmallTKSlice == 0, "small-T K split must divide K");

// Two-device column shard: each rank owns 24 of the 48 value heads, rows [24r,24r+24) of both A
// and B. The MMA route tiles rows by kBf16GdnBlockM = 16, which does not divide 24, so the shard
// uses the row-generic GEMV kernel at T=1 and the split-10 small-T kernel, which has no upper T
// bound, at every T>=2.
constexpr int kShardN = 24;
// Through this T the shard runs the single-launch split-10 kernel (verify and draft blocks); above
// it, prefill keeps the two-kernel form, whose partial grid has one CTA per four rows and slice.
constexpr int kShardFusedMaxTokens = 32;

template <int TokenTile, int KSlice, int RowsPerBlock, int N = kN>
__global__ void bf16_gdn_gating_proj_small_t_partial_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ a_weight,
    const __nv_bfloat16* __restrict__ b_weight, float* __restrict__ partial, std::int32_t t) {
    static_assert(TokenTile == kSmallTMax, "small-T token tile is fixed to 8");
    static_assert(KSlice == kSmallTKSlice, "small-T K split is fixed to 512");
    static_assert(RowsPerBlock == kSmallTRowsPerBlock, "small-T rows/block mismatch");
    constexpr int kLocalLogicalRows = 2 * N;
    constexpr int kVecsPerCol       = KSlice / 8;

    __shared__ __align__(16) __nv_bfloat16 x_sh[TokenTile][KSlice];

    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int token0 = static_cast<int>(blockIdx.z) * TokenTile;
    if (token0 >= t) { return; }
    const int ncols = min(TokenTile, t - token0);
    const int split = static_cast<int>(blockIdx.y);
    const int k0    = split * KSlice;

    auto* x_sh_v = reinterpret_cast<uint4*>(x_sh);
    for (int i = static_cast<int>(threadIdx.x); i < ncols * kVecsPerCol;
         i += static_cast<int>(blockDim.x)) {
        const int col = i / kVecsPerCol;
        const int vec = i - col * kVecsPerCol;
        x_sh_v[col * kVecsPerCol + vec] =
            load_vec<uint4>(x + static_cast<std::int64_t>(token0 + col) * kK + k0 + vec * 8);
    }
    __syncthreads();

    const int logical_row = static_cast<int>(blockIdx.x) * RowsPerBlock + warp;
    if (logical_row >= kLocalLogicalRows) { return; }
    const bool is_b = logical_row >= N;
    const int row   = is_b ? logical_row - N : logical_row;
    const __nv_bfloat16* wrow =
        (is_b ? b_weight : a_weight) + static_cast<std::int64_t>(row) * kK + k0;

    float acc[TokenTile];
#pragma unroll
    for (int tt = 0; tt < TokenTile; ++tt) { acc[tt] = 0.0f; }

    for (int vec = lane; vec < kVecsPerCol; vec += 32) {
        const uint4 wv   = load_vec<uint4>(wrow + vec * 8);
        const float2 wf0 = bf16x2_bits_to_float2(wv.x);
        const float2 wf1 = bf16x2_bits_to_float2(wv.y);
        const float2 wf2 = bf16x2_bits_to_float2(wv.z);
        const float2 wf3 = bf16x2_bits_to_float2(wv.w);

#pragma unroll
        for (int tt = 0; tt < TokenTile; ++tt) {
            if (tt < ncols) {
                const uint4 xv   = x_sh_v[tt * kVecsPerCol + vec];
                const float2 xf0 = bf16x2_bits_to_float2(xv.x);
                const float2 xf1 = bf16x2_bits_to_float2(xv.y);
                const float2 xf2 = bf16x2_bits_to_float2(xv.z);
                const float2 xf3 = bf16x2_bits_to_float2(xv.w);
                acc[tt]          = fmaf(wf0.x, xf0.x, acc[tt]);
                acc[tt]          = fmaf(wf0.y, xf0.y, acc[tt]);
                acc[tt]          = fmaf(wf1.x, xf1.x, acc[tt]);
                acc[tt]          = fmaf(wf1.y, xf1.y, acc[tt]);
                acc[tt]          = fmaf(wf2.x, xf2.x, acc[tt]);
                acc[tt]          = fmaf(wf2.y, xf2.y, acc[tt]);
                acc[tt]          = fmaf(wf3.x, xf3.x, acc[tt]);
                acc[tt]          = fmaf(wf3.y, xf3.y, acc[tt]);
            }
        }
    }

#pragma unroll
    for (int tt = 0; tt < TokenTile; ++tt) {
        if (tt < ncols) {
            float sum = warp_reduce_sum(acc[tt]);
            if (lane == 0) {
                const int token      = token0 + tt;
                partial[(static_cast<std::int64_t>(split) * t + token) * kLocalLogicalRows +
                        logical_row] = sum;
            }
        }
    }
}

template <int N = kN>
__global__ void
bf16_gdn_gating_proj_small_t_reduce_kernel(const float* __restrict__ partial,
                                           const float* __restrict__ A_log,
                                           const float* __restrict__ dt_bias, float* __restrict__ g,
                                           float* __restrict__ beta, std::int32_t t) {
    constexpr int kLocalLogicalRows = 2 * N;
    const int i =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int elems = N * t;
    if (i >= elems) { return; }

    const int row   = i % N;
    const int token = i / N;
    float acc_a     = 0.0f;
    float acc_b     = 0.0f;
#pragma unroll
    for (int split = 0; split < kSmallTSplits; ++split) {
        const std::int64_t base =
            (static_cast<std::int64_t>(split) * t + token) * kLocalLogicalRows;
        acc_a += partial[base + row];
        acc_b += partial[base + N + row];
    }

    const std::int64_t out_index = static_cast<std::int64_t>(token) * N + row;
    const float sp               = softplus(acc_a + dt_bias[row]);
    g[out_index]                 = -expf(A_log[row]) * sp;
    beta[out_index]              = sigmoid(acc_b);
}

// Single-launch form of the split-10 partial and reduce kernels above for one token tile. Each warp
// owns one K slice of one logical row, exactly as a partial-kernel warp does: the same lane-strided
// vectors in the same order, the same FMA chain per token and the same warp reduction. The ten
// slice sums meet in shared memory instead of the global partial plane, and the reduction then adds
// them in split order with the reduce kernel's expressions, so g and beta are bit-identical to the
// two-kernel route while the second launch and the plane's round trip disappear. Rows are spread
// over CTAs (RowsPerCta each) so the small-T weight read keeps as many SMs busy as the partial
// grid.
template <int RowsPerCta, int N>
__global__ __launch_bounds__(kSmallTSplits * 32) void bf16_gdn_gating_proj_small_t_fused_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ a_weight,
    const __nv_bfloat16* __restrict__ b_weight, const float* __restrict__ A_log,
    const float* __restrict__ dt_bias, float* __restrict__ g, float* __restrict__ beta,
    std::int32_t t) {
    constexpr int kTokenTile        = kSmallTMax;
    constexpr int kLocalLogicalRows = 2 * N;
    constexpr int kVecsPerCol       = kSmallTKSlice / 8;
    constexpr int kVecsPerLane      = kVecsPerCol / 32;
    static_assert(kLocalLogicalRows % RowsPerCta == 0 && N % RowsPerCta == 0,
                  "a CTA's rows must not straddle the A/B boundary");
    static_assert(kVecsPerCol % 32 == 0);

    __shared__ float slice_sums[RowsPerCta][kSmallTSplits][kTokenTile];

    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int split  = static_cast<int>(threadIdx.x) >> 5;
    const int token0 = static_cast<int>(blockIdx.y) * kTokenTile;
    const int ncols  = min(kTokenTile, t - token0);
    const int k0     = split * kSmallTKSlice;

    uint4 xv[kVecsPerLane][kTokenTile];
#pragma unroll
    for (int j = 0; j < kVecsPerLane; ++j) {
#pragma unroll
        for (int tt = 0; tt < kTokenTile; ++tt) {
            if (tt < ncols) {
                xv[j][tt] = load_vec<uint4>(x + static_cast<std::int64_t>(token0 + tt) * kK + k0 +
                                            (lane + j * 32) * 8);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < RowsPerCta; ++r) {
        const int logical_row = static_cast<int>(blockIdx.x) * RowsPerCta + r;
        const bool is_b       = logical_row >= N;
        const int row         = is_b ? logical_row - N : logical_row;
        const __nv_bfloat16* wrow =
            (is_b ? b_weight : a_weight) + static_cast<std::int64_t>(row) * kK + k0;

        float acc[kTokenTile];
#pragma unroll
        for (int tt = 0; tt < kTokenTile; ++tt) { acc[tt] = 0.0f; }

#pragma unroll
        for (int j = 0; j < kVecsPerLane; ++j) {
            const uint4 wv   = load_vec<uint4>(wrow + (lane + j * 32) * 8);
            const float2 wf0 = bf16x2_bits_to_float2(wv.x);
            const float2 wf1 = bf16x2_bits_to_float2(wv.y);
            const float2 wf2 = bf16x2_bits_to_float2(wv.z);
            const float2 wf3 = bf16x2_bits_to_float2(wv.w);
#pragma unroll
            for (int tt = 0; tt < kTokenTile; ++tt) {
                if (tt < ncols) {
                    const float2 xf0 = bf16x2_bits_to_float2(xv[j][tt].x);
                    const float2 xf1 = bf16x2_bits_to_float2(xv[j][tt].y);
                    const float2 xf2 = bf16x2_bits_to_float2(xv[j][tt].z);
                    const float2 xf3 = bf16x2_bits_to_float2(xv[j][tt].w);
                    acc[tt]          = fmaf(wf0.x, xf0.x, acc[tt]);
                    acc[tt]          = fmaf(wf0.y, xf0.y, acc[tt]);
                    acc[tt]          = fmaf(wf1.x, xf1.x, acc[tt]);
                    acc[tt]          = fmaf(wf1.y, xf1.y, acc[tt]);
                    acc[tt]          = fmaf(wf2.x, xf2.x, acc[tt]);
                    acc[tt]          = fmaf(wf2.y, xf2.y, acc[tt]);
                    acc[tt]          = fmaf(wf3.x, xf3.x, acc[tt]);
                    acc[tt]          = fmaf(wf3.y, xf3.y, acc[tt]);
                }
            }
        }
#pragma unroll
        for (int tt = 0; tt < kTokenTile; ++tt) {
            if (tt < ncols) {
                const float sum = warp_reduce_sum(acc[tt]);
                if (lane == 0) { slice_sums[r][split][tt] = sum; }
            }
        }
    }
    __syncthreads();

    const int i = static_cast<int>(threadIdx.x);
    if (i >= RowsPerCta * kTokenTile) { return; }
    const int r  = i / kTokenTile;
    const int tt = i - r * kTokenTile;
    if (tt >= ncols) { return; }
    float acc = 0.0f;
#pragma unroll
    for (int s = 0; s < kSmallTSplits; ++s) { acc += slice_sums[r][s][tt]; }
    const int logical_row        = static_cast<int>(blockIdx.x) * RowsPerCta + r;
    const std::int64_t out_index = static_cast<std::int64_t>(token0 + tt) * N;
    if (logical_row >= N) {
        beta[out_index + logical_row - N] = sigmoid(acc);
    } else {
        const float sp             = softplus(acc + dt_bias[logical_row]);
        g[out_index + logical_row] = -expf(A_log[logical_row]) * sp;
    }
}

template <int N = kN>
__global__ void bf16_gdn_gating_proj_gemv_kernel(const __nv_bfloat16* x,
                                                 const __nv_bfloat16* a_weight,
                                                 const __nv_bfloat16* b_weight, const float* A_log,
                                                 const float* dt_bias, float* g, float* beta) {
    const int global_row = static_cast<int>(blockIdx.x);
    const bool is_b      = global_row >= N;
    const int row        = is_b ? global_row - N : global_row;
    const auto* weight   = is_b ? b_weight : a_weight;
    __shared__ float warp_sums[kThreads / kWarpSize];

    float acc                   = 0.0f;
    constexpr int kPairs        = kK / 2;
    const std::int64_t row_base = static_cast<std::int64_t>(row) * kK;
    const auto* x2              = reinterpret_cast<const __nv_bfloat162*>(x);
    const auto* w2 =
        reinterpret_cast<const __nv_bfloat162*>(weight + static_cast<std::int64_t>(row_base));
    for (int p = static_cast<int>(threadIdx.x); p < kPairs; p += static_cast<int>(blockDim.x)) {
        const float2 wf = bf16x2_to_float2(w2[p]);
        const float2 xf = bf16x2_to_float2(x2[p]);
        acc             = fmaf(wf.x, xf.x, acc);
        acc             = fmaf(wf.y, xf.y, acc);
    }

    acc = block_reduce_sum<kThreads>(acc, warp_sums);
    if (threadIdx.x == 0) {
        if (is_b) {
            beta[row] = sigmoid(acc);
        } else {
            const float sp = softplus(acc + dt_bias[row]);
            g[row]         = -expf(A_log[row]) * sp;
        }
    }
}

void require_shape(const Weight& w, const char* name) {
    if (w.n != kN || w.k != kK || w.shape[0] != kN || w.shape[1] != kK) {
        throw std::invalid_argument(std::string("gdn_gating_proj: ") + name +
                                    " requires contiguous BF16 [48,5120]");
    }
}

// The tp2 column shard's own shape (24 of the parent's 48 rows, K unchanged).
void require_shard_shape(const Weight& w, const char* name) {
    if (w.n != kShardN || w.k != kK || w.shape[0] != kShardN || w.shape[1] != kK) {
        throw std::invalid_argument(std::string("gdn_gating_proj column-parallel: ") + name +
                                    " requires contiguous BF16 [24,5120]");
    }
}

template <class Geometry, int SplitK>
constexpr std::int32_t cooperative_resident_ctas_per_sm() noexcept {
    static_assert(SplitK > 1);
    static_assert(std::is_same_v<Geometry, Bf16Gdn27Geometry>);
    static_assert(SplitK == 8 || SplitK == 4 || SplitK == 2);
    // Qualified on the sm_120a build: BN128 split-8 uses 256 threads and split-4/2 use
    // 512 threads; registers and 40-KiB shared memory admit two resident CTAs per SM.
    return 2;
}

template <class Geometry, int SplitK, int Warps = kBf16GdnWarps>
bool launch_bf16_prefill_mma(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                             const Weight& a_weight, const Weight& b_weight, const Tensor& A_log,
                             const Tensor& dt_bias, void* workspace, Tensor& g, Tensor& beta,
                             cudaStream_t stream, std::int32_t multiprocessor_count = 0) {
    constexpr int kBlockN    = Geometry::kBlockN;
    constexpr int kSmemBytes = kBf16GdnSmemBytes<kBlockN>;
    const dim3 block(Warps * 32);
    auto launch_problem = [&](Bf16GdnGatingTokenVariant launch_variant, const Tensor& launch_x,
                              Tensor& launch_g, Tensor& launch_beta) {
        const std::int32_t launch_t = launch_x.ne[1];
        const dim3 grid(static_cast<unsigned>(div_up(launch_t, kBlockN)),
                        static_cast<unsigned>(Geometry::kHeads / kBf16GdnBlockM),
                        static_cast<unsigned>(SplitK));
        auto launch = [&](auto full_tokens) {
            constexpr bool FullTokens = decltype(full_tokens)::value;
            static FuncAttrPerDevice attr;
            attr.ensure(bf16_gdn_gating_proj_gemm_mma_kernel<Geometry, SplitK, FullTokens, Warps>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
            if constexpr (SplitK > 1) {
                cudaLaunchConfig_t config{};
                config.gridDim          = grid;
                config.blockDim         = block;
                config.dynamicSmemBytes = kSmemBytes;
                config.stream           = stream;
                cudaLaunchAttribute cooperative{};
                cooperative.id              = cudaLaunchAttributeCooperative;
                cooperative.val.cooperative = 1;
                config.attrs                = &cooperative;
                config.numAttrs             = 1;
                CUDA_CHECK(cudaLaunchKernelEx(
                    &config,
                    bf16_gdn_gating_proj_gemm_mma_kernel<Geometry, SplitK, FullTokens, Warps>,
                    static_cast<const __nv_bfloat16*>(launch_x.data),
                    static_cast<const __nv_bfloat16*>(a_weight.qdata),
                    static_cast<const __nv_bfloat16*>(b_weight.qdata),
                    static_cast<const float*>(A_log.data), static_cast<const float*>(dt_bias.data),
                    static_cast<float*>(workspace), static_cast<float*>(launch_g.data),
                    static_cast<float*>(launch_beta.data), launch_t));
            } else {
                bf16_gdn_gating_proj_gemm_mma_kernel<Geometry, SplitK, FullTokens, Warps>
                    <<<grid, block, kSmemBytes, stream>>>(
                        static_cast<const __nv_bfloat16*>(launch_x.data),
                        static_cast<const __nv_bfloat16*>(a_weight.qdata),
                        static_cast<const __nv_bfloat16*>(b_weight.qdata),
                        static_cast<const float*>(A_log.data),
                        static_cast<const float*>(dt_bias.data), static_cast<float*>(workspace),
                        static_cast<float*>(launch_g.data), static_cast<float*>(launch_beta.data),
                        launch_t);
            }
        };
        if (launch_variant == Bf16GdnGatingTokenVariant::Full) {
            launch(std::true_type{});
        } else if (launch_variant == Bf16GdnGatingTokenVariant::Predicated) {
            launch(std::false_type{});
        } else {
            throw std::invalid_argument(
                "BF16 GDN gating MMA requires Full or Predicated token variant");
        }
    };

    if constexpr (SplitK == 1) {
        launch_problem(variant, x, g, beta);
    } else {
        constexpr std::int64_t kCtasPerTokenTile =
            static_cast<std::int64_t>(Geometry::kHeads / kBf16GdnBlockM) * SplitK;
        constexpr std::int32_t kResidentCtasPerSm =
            cooperative_resident_ctas_per_sm<Geometry, SplitK>();
        const std::int64_t resident_ctas =
            static_cast<std::int64_t>(multiprocessor_count) * kResidentCtasPerSm;
        const std::int64_t max_token_tiles = resident_ctas / kCtasPerTokenTile;
        if (max_token_tiles < 1) { return false; }

        const std::int32_t t = x.ne[1];
        const std::int64_t total_token_tiles =
            div_up(static_cast<std::int64_t>(t), static_cast<std::int64_t>(kBlockN));
        if (total_token_tiles <= max_token_tiles) {
            launch_problem(variant, x, g, beta);
        } else {
            // Token tiles have no cross-tile reduction. Rebase each tensor to a disjoint token
            // interval and reuse the call-scoped partial workspace in stream order.
            const std::int64_t launch_token_capacity = max_token_tiles * kBlockN;
            for (std::int32_t token_begin = 0; token_begin < t;) {
                const std::int32_t launch_t = static_cast<std::int32_t>(std::min<std::int64_t>(
                    static_cast<std::int64_t>(t) - token_begin, launch_token_capacity));
                Tensor launch_x             = x.slice(1, token_begin, launch_t);
                Tensor launch_g             = g.slice(1, token_begin, launch_t);
                Tensor launch_beta          = beta.slice(1, token_begin, launch_t);
                const Bf16GdnGatingTokenVariant launch_variant =
                    launch_t % kBlockN == 0 ? Bf16GdnGatingTokenVariant::Full
                                            : Bf16GdnGatingTokenVariant::Predicated;
                launch_problem(launch_variant, launch_x, launch_g, launch_beta);
                token_begin += launch_t;
            }
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

} // namespace

void bf16_gdn_gating_proj_gemv_launch(const Tensor& x, const Weight& a_weight,
                                      const Weight& b_weight, const Tensor& A_log,
                                      const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                      cudaStream_t stream) {
    require_shape(a_weight, "a_weight");
    require_shape(b_weight, "b_weight");
    bf16_gdn_gating_proj_gemv_kernel<<<2 * kN, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(a_weight.qdata),
        static_cast<const __nv_bfloat16*>(b_weight.qdata), static_cast<const float*>(A_log.data),
        static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data));
    CUDA_CHECK(cudaGetLastError());
}

void bf16_gdn_gating_proj_small_t_split10_launch(const Tensor& x, const Weight& a_weight,
                                                 const Weight& b_weight, const Tensor& A_log,
                                                 const Tensor& dt_bias, void* workspace,
                                                 std::size_t workspace_bytes, Tensor& g,
                                                 Tensor& beta, cudaStream_t stream) {
    require_shape(a_weight, "a_weight");
    require_shape(b_weight, "b_weight");
    const std::int32_t t       = x.ne[1];
    const std::size_t required = static_cast<std::size_t>(kSmallTSplits) *
                                 static_cast<std::size_t>(t) *
                                 static_cast<std::size_t>(kLogicalRows) * sizeof(float);
    if (workspace == nullptr || workspace_bytes < required) {
        throw std::invalid_argument("gdn_gating_proj: small-T workspace is too small");
    }

    dim3 partial_block(kSmallTThreads);
    dim3 partial_grid(div_up(kLogicalRows, kSmallTRowsPerBlock), kSmallTSplits,
                      div_up(t, kSmallTMax));
    bf16_gdn_gating_proj_small_t_partial_kernel<kSmallTMax, kSmallTKSlice, kSmallTRowsPerBlock>
        <<<partial_grid, partial_block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(a_weight.qdata),
            static_cast<const __nv_bfloat16*>(b_weight.qdata), static_cast<float*>(workspace), t);
    CUDA_CHECK(cudaGetLastError());

    constexpr int kReduceThreads = 128;
    const int reduce_elems       = kN * t;
    const int reduce_blocks      = div_up(reduce_elems, kReduceThreads);
    bf16_gdn_gating_proj_small_t_reduce_kernel<<<reduce_blocks, kReduceThreads, 0, stream>>>(
        static_cast<const float*>(workspace), static_cast<const float*>(A_log.data),
        static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data), t);
    CUDA_CHECK(cudaGetLastError());
}

void bf16_gdn_gating_proj_gemv_shard_launch(const Tensor& x, const Weight& a_weight,
                                            const Weight& b_weight, const Tensor& A_log,
                                            const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                            cudaStream_t stream) {
    require_shard_shape(a_weight, "a_weight");
    require_shard_shape(b_weight, "b_weight");
    bf16_gdn_gating_proj_gemv_kernel<kShardN><<<2 * kShardN, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(a_weight.qdata),
        static_cast<const __nv_bfloat16*>(b_weight.qdata), static_cast<const float*>(A_log.data),
        static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data));
    CUDA_CHECK(cudaGetLastError());
}

void bf16_gdn_gating_proj_small_t_split10_shard_launch(const Tensor& x, const Weight& a_weight,
                                                       const Weight& b_weight, const Tensor& A_log,
                                                       const Tensor& dt_bias, void* workspace,
                                                       std::size_t workspace_bytes, Tensor& g,
                                                       Tensor& beta, cudaStream_t stream) {
    require_shard_shape(a_weight, "a_weight");
    require_shard_shape(b_weight, "b_weight");
    constexpr int kShardLogicalRows = 2 * kShardN;
    const std::int32_t t            = x.ne[1];
    const std::size_t required      = static_cast<std::size_t>(kSmallTSplits) *
                                 static_cast<std::size_t>(t) *
                                 static_cast<std::size_t>(kShardLogicalRows) * sizeof(float);
    if (workspace == nullptr || workspace_bytes < required) {
        throw std::invalid_argument(
            "gdn_gating_proj column-parallel: small-T workspace is too small");
    }
    dim3 partial_grid(div_up(kShardLogicalRows, kSmallTRowsPerBlock), kSmallTSplits,
                      div_up(t, kSmallTMax));
    bf16_gdn_gating_proj_small_t_partial_kernel<kSmallTMax, kSmallTKSlice, kSmallTRowsPerBlock,
                                                kShardN>
        <<<partial_grid, kSmallTThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(a_weight.qdata),
            static_cast<const __nv_bfloat16*>(b_weight.qdata), static_cast<float*>(workspace), t);
    constexpr int kReduceThreads = 128;
    const int reduce_blocks      = div_up(kShardN * t, kReduceThreads);
    bf16_gdn_gating_proj_small_t_reduce_kernel<kShardN>
        <<<reduce_blocks, kReduceThreads, 0, stream>>>(
            static_cast<const float*>(workspace), static_cast<const float*>(A_log.data),
            static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
            static_cast<float*>(beta.data), t);
    CUDA_CHECK(cudaGetLastError());
}

void bf16_gdn_gating_proj_small_t_fused_shard_launch(const Tensor& x, const Weight& a_weight,
                                                     const Weight& b_weight, const Tensor& A_log,
                                                     const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                                     cudaStream_t stream) {
    require_shard_shape(a_weight, "a_weight");
    require_shard_shape(b_weight, "b_weight");
    const std::int32_t t = x.ne[1];
    if (t < 2 || t > kShardFusedMaxTokens) {
        throw std::invalid_argument("gdn_gating_proj column-parallel: fused small-T needs T=2..32");
    }
    constexpr int kRowsPerCta = 2;
    const dim3 grid(2 * kShardN / kRowsPerCta, div_up(t, kSmallTMax));
    bf16_gdn_gating_proj_small_t_fused_kernel<kRowsPerCta, kShardN>
        <<<grid, kSmallTSplits * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(a_weight.qdata),
            static_cast<const __nv_bfloat16*>(b_weight.qdata),
            static_cast<const float*>(A_log.data), static_cast<const float*>(dt_bias.data),
            static_cast<float*>(g.data), static_cast<float*>(beta.data), t);
    CUDA_CHECK(cudaGetLastError());
}

void bf16_gdn_gating_dispatch_shard(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                                    const Tensor& A_log, const Tensor& dt_bias, void* workspace,
                                    std::size_t workspace_bytes, Tensor& g, Tensor& beta,
                                    cudaStream_t stream) {
    if (x.ne[1] == 1) {
        bf16_gdn_gating_proj_gemv_shard_launch(x, a_weight, b_weight, A_log, dt_bias, g, beta,
                                               stream);
        return;
    }
    if (x.ne[1] <= kShardFusedMaxTokens) {
        bf16_gdn_gating_proj_small_t_fused_shard_launch(x, a_weight, b_weight, A_log, dt_bias, g,
                                                        beta, stream);
        return;
    }
    bf16_gdn_gating_proj_small_t_split10_shard_launch(x, a_weight, b_weight, A_log, dt_bias,
                                                      workspace, workspace_bytes, g, beta, stream);
}

std::size_t bf16_gdn_gating_shard_workspace_bytes(std::int32_t tokens) {
    if (tokens <= kShardFusedMaxTokens) { return 0; }
    return static_cast<std::size_t>(kSmallTSplits) * static_cast<std::size_t>(tokens) *
           static_cast<std::size_t>(2 * kShardN) * sizeof(float);
}

bool bf16_gdn_gating_proj_mma_split8_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                            const Weight& a_weight, const Weight& b_weight,
                                            const Tensor& A_log, const Tensor& dt_bias,
                                            void* workspace, Tensor& g, Tensor& beta,
                                            std::int32_t multiprocessor_count,
                                            cudaStream_t stream) {
    return launch_bf16_prefill_mma<Bf16Gdn27Geometry, 8, 8>(variant, x, a_weight, b_weight, A_log,
                                                            dt_bias, workspace, g, beta, stream,
                                                            multiprocessor_count);
}

bool bf16_gdn_gating_proj_mma_split4_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                            const Weight& a_weight, const Weight& b_weight,
                                            const Tensor& A_log, const Tensor& dt_bias,
                                            void* workspace, Tensor& g, Tensor& beta,
                                            std::int32_t multiprocessor_count,
                                            cudaStream_t stream) {
    return launch_bf16_prefill_mma<Bf16Gdn27Geometry, 4>(variant, x, a_weight, b_weight, A_log,
                                                         dt_bias, workspace, g, beta, stream,
                                                         multiprocessor_count);
}

bool bf16_gdn_gating_proj_mma_split2_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                            const Weight& a_weight, const Weight& b_weight,
                                            const Tensor& A_log, const Tensor& dt_bias,
                                            void* workspace, Tensor& g, Tensor& beta,
                                            std::int32_t multiprocessor_count,
                                            cudaStream_t stream) {
    return launch_bf16_prefill_mma<Bf16Gdn27Geometry, 2>(variant, x, a_weight, b_weight, A_log,
                                                         dt_bias, workspace, g, beta, stream,
                                                         multiprocessor_count);
}

void bf16_gdn_gating_proj_mma_unsplit_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                             const Weight& a_weight, const Weight& b_weight,
                                             const Tensor& A_log, const Tensor& dt_bias, Tensor& g,
                                             Tensor& beta, cudaStream_t stream) {
    (void)launch_bf16_prefill_mma<Bf16Gdn27Geometry, 1, 8>(variant, x, a_weight, b_weight, A_log,
                                                           dt_bias, nullptr, g, beta, stream);
}

} // namespace ninfer::ops::detail
