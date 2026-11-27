// Kernels of the DFlash2 drafter Ops (ninfer/ops/dflash2_draft.h).
#include "core/device.h"
#include "ops/dflash2_draft/launch.h"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using Key = unsigned long long;

constexpr int kGroup = 16;

__device__ __forceinline__ float load(const __nv_bfloat16* p) { return __bfloat162float(*p); }

// ---- grouped dynamic convolution
// ----------------------------------------------------------------- One thread per (channel,
// column); the side's two taps read their coefficients from the projection's column and the base
// kernel's side.
__global__ void dynamic_conv_kernel(const __nv_bfloat16* __restrict__ x,
                                    const __nv_bfloat16* __restrict__ coefficients,
                                    const __nv_bfloat16* __restrict__ base, __nv_bfloat16* out,
                                    int hidden, int width, int side) {
    const std::int64_t index = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= std::int64_t(hidden) * width) return;
    const int h = int(index % hidden), i = int(index / hidden);
    const int groups = hidden / kGroup, g = h / kGroup;
    const __nv_bfloat16* c = coefficients + std::size_t(i) * 4 * groups;
    const float current = (load(base + h + 2 * hidden * side) + load(c + 2 * side * groups + g)) *
                          load(x + std::size_t(i) * hidden + h);
    float value = current;
    if (i > 0) {
        value = fmaf(load(base + h + hidden + 2 * hidden * side) +
                         load(c + (2 * side + 1) * groups + g),
                     load(x + std::size_t(i - 1) * hidden + h), current);
    }
    __nv_bfloat16* dst = out + std::size_t(i) * hidden + h;
    *dst               = __float2bfloat16_rn(side == 0 ? value : load(dst) + value);
}

// ---- vocabulary top 16
// --------------------------------------------------------------------------- A candidate is one
// 64-bit key: the value's order-preserving bits above, the complement of its id below, so that the
// larger key is the larger value and, among equal values, the lower id. Key 0 is below every
// candidate (an empty place).
__device__ __forceinline__ Key top_key(float value, int id) {
    unsigned u = __float_as_uint(value);
    u          = (u & 0x80000000U) != 0 ? ~u : (u | 0x80000000U);
    return (Key(u) << 32) | Key(0xFFFFFFFFU - unsigned(id));
}

__device__ __forceinline__ float key_value(Key key) {
    const auto u = unsigned(key >> 32);
    return __uint_as_float((u & 0x80000000U) != 0 ? (u & 0x7FFFFFFFU) : ~u);
}

__device__ __forceinline__ int key_id(Key key) {
    return int(0xFFFFFFFFU - unsigned(key & 0xFFFFFFFFULL));
}

// A lane's descending list: the new key replaces the last place and bubbles up (static indices,
// in registers).
template <int N>
__device__ __forceinline__ void insert(Key (&list)[N], Key key) {
    if (key <= list[N - 1]) return;
    list[N - 1] = key;
#pragma unroll
    for (int k = N - 1; k > 0; --k) {
        const Key above = list[k - 1], below = list[k];
        list[k - 1] = above > below ? above : below;
        list[k]     = above > below ? below : above;
    }
}

// The warp's largest key over the lanes' list heads; the lane holding it drops it (keys are
// distinct, ids being distinct).
template <int N>
__device__ __forceinline__ Key warp_pop_max(Key (&list)[N]) {
    Key best = list[0];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const Key other = __shfl_xor_sync(0xFFFFFFFFU, best, offset);
        best            = other > best ? other : best;
    }
    if (best != 0 && list[0] == best) {
#pragma unroll
        for (int k = 0; k + 1 < N; ++k) list[k] = list[k + 1];
        list[N - 1] = 0;
    }
    return best;
}

// Grid (kVocabTopChunks, C), one warp: the 16 best of the chunk's rows of one column.
__global__ void __launch_bounds__(32)
    vocab_top16_chunk_kernel(const __nv_bfloat16* __restrict__ logits, int rows, int first,
                             Key* __restrict__ chunk_keys) {
    const int chunk = blockIdx.x, column = blockIdx.y, lane = threadIdx.x;
    const int per   = (rows + kVocabTopChunks - 1) / kVocabTopChunks;
    const int begin = chunk * per, end = min(rows, begin + per);
    const __nv_bfloat16* values = logits + std::size_t(column) * rows;
    Key list[kVocabTop]         = {};
    for (int r = begin + lane; r < end; r += 32) insert(list, top_key(load(values + r), first + r));
    Key* out = chunk_keys + (std::size_t(column) * kVocabTopChunks + chunk) * kVocabTop;
#pragma unroll 1
    for (int k = 0; k < kVocabTop; ++k) {
        const Key best = warp_pop_max(list);
        if (lane == 0) out[k] = best;
    }
}

// Grid (C), one warp: the 16 best over the column's chunks, into slot `slot` (other slots zero).
__global__ void __launch_bounds__(32)
    vocab_top16_slot_kernel(const Key* __restrict__ chunk_keys, int columns, int slot, int slots,
                            float* __restrict__ pairs) {
    const int column = blockIdx.x, lane = threadIdx.x;
    const Key* keys     = chunk_keys + std::size_t(column) * kVocabTopChunks * kVocabTop;
    Key list[kVocabTop] = {};
    for (int k = lane; k < kVocabTopChunks * kVocabTop; k += 32) insert(list, keys[k]);
#pragma unroll 1
    for (int k = 0; k < kVocabTop; ++k) {
        const Key best = warp_pop_max(list);
        if (lane < slots) {
            float* p = pairs + ((std::size_t(lane) * columns + column) * kVocabTop + k) * 2;
            p[0]     = lane == slot ? key_value(best) : 0.0F;
            p[1]     = lane == slot ? float(key_id(best)) : 0.0F;
        }
    }
}

// Grid (C), one warp: the 16 best of the column's slots (one candidate per lane).
__global__ void __launch_bounds__(32)
    vocab_top16_merge_kernel(const float* __restrict__ pairs, int columns, int slots,
                             int* __restrict__ ids, float* __restrict__ values) {
    const int column = blockIdx.x, lane = threadIdx.x;
    Key list[1] = {};
    if (lane < slots * kVocabTop) {
        const int s = lane / kVocabTop, k = lane % kVocabTop;
        const float* p = pairs + ((std::size_t(s) * columns + column) * kVocabTop + k) * 2;
        list[0]        = top_key(p[0], int(p[1]));
    }
#pragma unroll 1
    for (int k = 0; k < kVocabTop; ++k) {
        const Key best = warp_pop_max(list);
        if (lane == 0) {
            ids[std::size_t(column) * kVocabTop + k]    = key_id(best);
            values[std::size_t(column) * kVocabTop + k] = key_value(best);
        }
    }
}

} // namespace

void dynamic_conv_launch(const void* x, const void* coefficients, const void* base, void* out,
                         int hidden, int width, bool output_side, cudaStream_t stream) {
    const std::int64_t total = std::int64_t(hidden) * width;
    dynamic_conv_kernel<<<unsigned((total + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(coefficients),
        static_cast<const __nv_bfloat16*>(base), static_cast<__nv_bfloat16*>(out), hidden, width,
        output_side ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());
}

void vocab_top16_slots_launch(const void* logits, int rows, int columns, int first, int slot,
                              int slots, unsigned long long* chunk_keys, float* pairs,
                              cudaStream_t stream) {
    vocab_top16_chunk_kernel<<<dim3(unsigned(kVocabTopChunks), unsigned(columns)), 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits), rows, first, chunk_keys);
    CUDA_CHECK(cudaGetLastError());
    vocab_top16_slot_kernel<<<unsigned(columns), 32, 0, stream>>>(chunk_keys, columns, slot, slots,
                                                                  pairs);
    CUDA_CHECK(cudaGetLastError());
}

void vocab_top16_merge_launch(const float* pairs, int columns, int slots, int* ids, float* values,
                              cudaStream_t stream) {
    vocab_top16_merge_kernel<<<unsigned(columns), 32, 0, stream>>>(pairs, columns, slots, ids,
                                                                   values);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
