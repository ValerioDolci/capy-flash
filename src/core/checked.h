#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace ninfer {

// Size arithmetic that throws std::overflow_error(label) instead of wrapping.
[[nodiscard]] inline std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

[[nodiscard]] inline std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

} // namespace ninfer
