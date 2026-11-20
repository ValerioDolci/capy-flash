#pragma once

// 64 x 64 -> 128-bit products for compilers without unsigned __int128 (MSVC).
// GCC and Clang keep the native type, so their code is unchanged.

#include <cstdint>
#include <limits>

#if !defined(__SIZEOF_INT128__) && defined(_MSC_VER)
#    include <intrin.h>
#endif

namespace ninfer {

#if defined(__SIZEOF_INT128__)

using WideProduct = unsigned __int128;

inline WideProduct wide_mul(std::uint64_t a, std::uint64_t b) noexcept {
    return static_cast<WideProduct>(a) * b;
}

#else

// The exact product as (high, low) words; compares like the 128-bit integer it stands for.
struct WideProduct {
    std::uint64_t high = 0;
    std::uint64_t low  = 0;

    friend bool operator==(const WideProduct& x, const WideProduct& y) noexcept {
        return x.high == y.high && x.low == y.low;
    }

    friend bool operator!=(const WideProduct& x, const WideProduct& y) noexcept {
        return !(x == y);
    }

    friend bool operator<(const WideProduct& x, const WideProduct& y) noexcept {
        return x.high != y.high ? x.high < y.high : x.low < y.low;
    }
};

inline WideProduct wide_mul(std::uint64_t a, std::uint64_t b) noexcept {
    WideProduct p;
    p.low = _umul128(a, b, &p.high);
    return p;
}

#endif

// a * b, or UINT64_MAX when the product does not fit.
inline std::uint64_t saturating_mul(std::uint64_t a, std::uint64_t b) noexcept {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return a * b;
}

// a + b, or UINT64_MAX when the sum does not fit.
inline std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) noexcept {
    return a > std::numeric_limits<std::uint64_t>::max() - b
               ? std::numeric_limits<std::uint64_t>::max()
               : a + b;
}

} // namespace ninfer
