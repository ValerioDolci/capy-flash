#pragma once

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Strided row block copy: for every column c in [0, columns),
 *
 *   destination[c * row_bytes + b] = source[c * source_pitch + b],   b in [0, row_bytes),
 *
 * i.e. cudaMemcpy2DAsync(destination, row_bytes, source, source_pitch, row_bytes, columns) as one
 * kernel, so that a CUDA Graph node built from it is a kernel node (whose pointer arguments
 * cudaGraphExecUpdate always accepts) instead of a memcpy node. Device memory, non-overlapping.
 * 16-byte vectors when both pointers, the pitch and the row size allow it, else bytes.
 */
void copy_rows(void* destination, const void* source, std::size_t source_pitch,
               std::size_t row_bytes, std::size_t columns, cudaStream_t stream);

} // namespace ninfer::ops
