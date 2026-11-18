#pragma once

// Synthetic Qwen4Exp routed-expert bank in block_scale_k16_m128x4_v1 (storage-layouts section 4)
// and an independent decoder for it. Weights are Gaussian values quantized to NVFP4 the way a
// checkpoint is (per-16 E4M3 scale = amax / 6 * S, E2M1 round-to-nearest-even, one FP32 divisor
// per matrix), so codes, scales and divisors have realistic distributions.

#include "ops/host_moe/host_moe_executor.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace ninfer::test::host_moe {

inline double e2m1_value(unsigned code) {
    static constexpr double kGrid[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    return (code & 8U ? -1.0 : 1.0) * kGrid[code & 7U];
}

inline double e4m3_value(unsigned code) {
    const unsigned e = (code >> 3) & 15U, m = code & 7U;
    const double mag =
        e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + m / 8.0, static_cast<int>(e) - 7);
    return code & 0x80U ? -mag : mag;
}

// Nearest positive E4M3FN code to v >= 0 (ties to even mantissa), saturating at 448.
inline std::uint8_t e4m3_encode_positive(double v) {
    std::uint8_t best = 0;
    double best_err   = std::abs(v);
    for (unsigned c = 1; c < 0x7F; ++c) {
        const double err = std::abs(e4m3_value(c) - v);
        if (err < best_err || (err == best_err && (c & 1U) == 0))
            best = static_cast<std::uint8_t>(c), best_err = err;
    }
    return best;
}

// E2M1 code of v, round to nearest even, saturating at +-6.
inline std::uint8_t e2m1_encode(double v) {
    const double a  = std::abs(v);
    unsigned best   = 0;
    double best_err = a;
    for (unsigned c = 1; c < 8; ++c) {
        const double err = std::abs(e2m1_value(c) - a);
        if (err < best_err || (err == best_err && (c & 1U) == 0)) best = c, best_err = err;
    }
    return static_cast<std::uint8_t>(best | (v < 0 && best ? 8U : 0U));
}

inline std::size_t scale_offset(std::int64_t row, std::int64_t group, std::int32_t k) {
    const std::int64_t k_tiles = k / 64;
    return static_cast<std::size_t>(((row / 128) * k_tiles + group / 4) * 512 +
                                    (row % 128 % 32) * 16 + (row % 128 / 32) * 4 + group % 4);
}

// One NVFP4 plane payload built from Gaussian weights; `sigma[i]` = weight scale of source i.
struct PlaneData {
    std::vector<std::uint8_t> payload;
    std::int64_t rows = 0, divisors = 0;
    std::int32_t k         = 0;
    std::size_t scale_base = 0;

    PlaneData(std::int64_t rows_, std::int32_t k_, std::int64_t divisors_, std::mt19937_64& rng)
        : rows(rows_), divisors(divisors_), k(k_) {
        payload.assign(ops::host::Nvfp4Plane::payload_bytes(rows, k, divisors), 0);
        const std::size_t codes = static_cast<std::size_t>(rows) * k / 2;
        scale_base              = (codes + 255) / 256 * 256;
        auto* div               = reinterpret_cast<float*>(payload.data() + scale_base + codes / 8);
        constexpr double S      = 64.0; // keeps the E4M3 scales in their normal range
        std::normal_distribution<double> normal(0.0, 1.0);
        std::uniform_real_distribution<double> sig(0.01, 0.04);
        for (std::int64_t d = 0; d < divisors; ++d) div[d] = static_cast<float>(S / sig(rng));
        double v[16];
        for (std::int64_t r = 0; r < rows; ++r) {
            for (std::int64_t g = 0; g < k / 16; ++g) {
                double amax = 0;
                for (double& x : v) x = normal(rng), amax = std::max(amax, std::abs(x));
                const std::uint8_t sc                       = e4m3_encode_positive(amax / 6.0 * S);
                payload[scale_base + scale_offset(r, g, k)] = sc;
                const double sv                             = e4m3_value(sc);
                for (int j = 0; j < 16; ++j) {
                    const std::uint8_t c   = e2m1_encode(v[j] * S / sv);
                    const std::size_t byte = static_cast<std::size_t>(r) * (k / 2) + g * 8 + j / 2;
                    payload[byte] |= static_cast<std::uint8_t>(j % 2 ? c << 4 : c);
                }
            }
        }
    }

    [[nodiscard]] ops::host::Nvfp4Plane plane() const {
        return ops::host::Nvfp4Plane::from_payload(payload.data(), rows, k, divisors);
    }

    // Independent decode of w[row, col].
    [[nodiscard]] double weight(std::int64_t row, std::int32_t col) const {
        const std::uint8_t byte = payload[static_cast<std::size_t>(row) * (k / 2) + col / 2];
        const unsigned code     = col % 2 ? byte >> 4 : byte & 15U;
        const double sc         = e4m3_value(payload[scale_base + scale_offset(row, col / 16, k)]);
        const std::size_t codes = static_cast<std::size_t>(rows) * k / 2;
        float d;
        std::memcpy(&d, payload.data() + scale_base + codes / 8 + 4 * (row / (rows / divisors)), 4);
        return e2m1_value(code) * sc / static_cast<double>(d);
    }

    void decode_row(std::int64_t row, std::vector<double>& out) const {
        out.resize(static_cast<std::size_t>(k));
        for (std::int32_t c = 0; c < k; ++c) out[static_cast<std::size_t>(c)] = weight(row, c);
    }
};

// Routed experts of one layer: gate_up [E*2I, H] (divisor per gate / up matrix), down [E*H, I].
struct LayerData {
    std::int32_t experts, hidden, intermediate;
    PlaneData gate_up, down;

    LayerData(std::int32_t e, std::int32_t h, std::int32_t i, std::mt19937_64& rng)
        : experts(e), hidden(h), intermediate(i), gate_up(std::int64_t{e} * 2 * i, h, 2 * e, rng),
          down(std::int64_t{e} * h, i, e, rng) {}

    [[nodiscard]] ops::host::HostExpertLayer layer() const {
        return {gate_up.plane(), down.plane(), experts, hidden, intermediate};
    }
};

inline std::uint16_t to_bf16(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7FFFU + ((u >> 16) & 1U);
    return static_cast<std::uint16_t>(u >> 16);
}

inline float from_bf16(std::uint16_t h) {
    const std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

} // namespace ninfer::test::host_moe
