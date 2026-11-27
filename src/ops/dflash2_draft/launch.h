#pragma once

// ninfer::ops::detail - private launch prototypes of the DFlash2 drafter Ops (dflash2_draft.h).

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Vocabulary chunks per column of the first top-16 pass; each chunk keeps its own 16 candidates.
inline constexpr int kVocabTopChunks = 64;
inline constexpr int kVocabTop       = 16;

void dynamic_conv_launch(const void* x, const void* coefficients, const void* base, void* out,
                         int hidden, int width, bool output_side, cudaStream_t stream);
void vocab_top16_slots_launch(const void* logits, int rows, int columns, int first, int slot,
                              int slots, unsigned long long* chunk_keys, float* pairs,
                              cudaStream_t stream);
void vocab_top16_merge_launch(const float* pairs, int columns, int slots, int* ids, float* values,
                              cudaStream_t stream);

} // namespace ninfer::ops::detail
