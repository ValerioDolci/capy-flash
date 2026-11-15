#pragma once

// Response and message IDs from the kernel's CSPRNG. resp_ IDs are the lookup key of
// GET /v1/responses/{id} and previous_response_id: with a seeded mt19937_64 a client that saw 312
// outputs (156 IDs) could predict every later one.

#include <array>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <string_view>

#if defined(__linux__)
#    include <sys/random.h>
#endif

namespace ninfer::serve {

inline std::uint64_t random_id_bits() {
    std::uint64_t value = 0;
#if defined(__linux__)
    if (::getrandom(&value, sizeof(value), 0) == static_cast<ssize_t>(sizeof(value))) {
        return value;
    }
#endif
    std::random_device device; // also non-deterministic, on every supported platform
    return (static_cast<std::uint64_t>(device()) << 32U) ^ device();
}

// `prefix` followed by 16 random hex digits.
inline std::string random_identifier(std::string_view prefix) {
    std::array<char, 32> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%016llx",
                  static_cast<unsigned long long>(random_id_bits()));
    return std::string(prefix) + buffer.data();
}

} // namespace ninfer::serve
