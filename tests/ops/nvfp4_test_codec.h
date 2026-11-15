#pragma once

// Host NVFP4 codec of the MoE expert tests (moe_slots, moe_grouped, Flash-Next tp2 MoE): the E2M1
// values, the E4M3FN decode, and the round-to-nearest-even encoders of the GPU quantizers.

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ninfer::test {

inline constexpr float kE2M1[16] = {0.F,  0.5F,  1.F,  1.5F,  2.F,  3.F,  4.F,  6.F,
                                    -0.F, -0.5F, -1.F, -1.5F, -2.F, -3.F, -4.F, -6.F};

inline float e4m3_decode(std::uint8_t c) {
    const int sign = c >> 7, e = (c >> 3) & 15, m = c & 7;
    float v = 0.F;
    if (e == 0)
        v = static_cast<float>(m) * std::ldexp(1.0F, -9);
    else
        v = (1.0F + static_cast<float>(m) / 8.0F) * std::ldexp(1.0F, e - 7);
    return sign ? -v : v;
}

// E4M3FN round-to-nearest-even, satfinite, of a non-negative value (the activation scale).
inline float e4m3_rn(float v) {
    if (!(v > 0.0F)) return 0.0F;
    if (v >= 448.0F) return 448.0F;
    int ex = 0;
    std::frexp(v, &ex);                        // v = f * 2^ex, f in [0.5,1)
    const int exponent = std::max(ex - 1, -6); // subnormal step below 2^-6
    const float step   = std::ldexp(1.0F, exponent - 3);
    return std::min(std::nearbyint(v / step) * step, 448.0F);
}

// E2M1 round-to-nearest-even (ties to the even code), satfinite.
inline float e2m1_rn(float v) {
    static const float grid[8] = {0.F, 0.5F, 1.F, 1.5F, 2.F, 3.F, 4.F, 6.F};
    const float a              = std::fabs(v);
    float r                    = 6.0F;
    for (int i = 0; i < 7; ++i) {
        if (a > grid[i + 1]) continue;
        const float mid = 0.5F * (grid[i] + grid[i + 1]);
        r = a < mid ? grid[i] : a > mid ? grid[i + 1] : (i % 2 == 0 ? grid[i] : grid[i + 1]);
        break;
    }
    return std::copysign(r, v);
}

} // namespace ninfer::test
