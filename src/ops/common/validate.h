#pragma once

#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// True when the byte ranges [a, a + a_bytes) and [b, b + b_bytes) share at least one byte.
inline bool overlaps(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) {
    const auto pa = reinterpret_cast<std::uintptr_t>(a);
    const auto pb = reinterpret_cast<std::uintptr_t>(b);
    return pa < pb + b_bytes && pb < pa + a_bytes;
}

inline bool overlaps(const Tensor& a, const Tensor& b) {
    return overlaps(a.data, a.bytes(), b.data, b.bytes());
}

} // namespace ninfer::ops::detail
