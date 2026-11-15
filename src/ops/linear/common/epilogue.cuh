#pragma once

#include "ops/linear/common/output.cuh"

#include <concepts>

namespace ninfer::ops::detail {

// Thread-local operations over a fully reduced accumulator. The contraction
// owns predicates and synchronization; epilogues only receive valid coordinates.
struct LinearIdentityEpilogue {
    __device__ __forceinline__ float apply(int, int, float value) const { return value; }

    __device__ __forceinline__ float apply_scaled(int, int, float value, float scale) const {
        return value * scale;
    }
};

struct LinearResidualAddEpilogue {
    LinearBf16InputView residual;

    __device__ __forceinline__ float apply(int row, int token, float value) const {
        return value + residual.load(row, token);
    }

    // Scaling belongs to the fused residual update, rounded as a product and then a sum in every
    // tile. A caller's live-column predicate must not decide whether the compiler contracts the
    // two into one FMA, or the widths that fill every token tile would round differently from all
    // the others.
    __device__ __forceinline__ float apply_scaled(int row, int token, float value,
                                                  float scale) const {
        return __fadd_rn(__fmul_rn(value, scale), residual.load(row, token));
    }

    // apply(row, token, value) split in two, so a kernel can issue the residual read before its
    // reduction instead of after it: apply_loaded(value, load(row, token)) == apply(row, token,
    // value), bit for bit.
    __device__ __forceinline__ float load(int row, int token) const {
        return residual.load(row, token);
    }

    __device__ __forceinline__ float apply_loaded(float value, float loaded) const {
        return value + loaded;
    }
};

// A one-row-per-thread-group kernel spreads its rows' residual reads over the lanes of the warp
// that owns them, one value per lane, before the K loop: the read then overlaps the weight stream
// instead of adding a dependent memory round trip after the reduction (on the tp 2 rank-0 down half
// [5120,8704] at T=4 the residual is cold in L2, and that round trip cost ~2.5 us of ~37).
template <class Epilogue>
concept LinearPreloadedEpilogue = requires(const Epilogue& epilogue) {
    { epilogue.load(0, 0) } -> std::same_as<float>;
    { epilogue.apply_loaded(0.0f, 0.0f) } -> std::same_as<float>;
};

// Row consumers run in the single thread owning the fully reduced row. They
// may publish fused Op state, but do not introduce warp/CTA synchronization.
template <class Output, class Epilogue, int Tokens>
__device__ __forceinline__ void
linear_finish_row(const Output& output, const Epilogue& epilogue, int row, int token_begin,
                  const float (&values)[Tokens], int active_tokens) {
    if constexpr (requires {
                      epilogue.apply_row(output, row, token_begin, values, active_tokens);
                  }) {
        epilogue.apply_row(output, row, token_begin, values, active_tokens);
    } else {
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            if (token < active_tokens) {
                const int global_token = token_begin + token;
                output.store(row, global_token, epilogue.apply(row, global_token, values[token]));
            }
        }
    }
}

} // namespace ninfer::ops::detail
