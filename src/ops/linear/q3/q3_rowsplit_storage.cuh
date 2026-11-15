#pragma once

// q3_g128_fp16 row-split storage (docs/maintainer/storage-layouts.md §3).
//
// One group = 128 signed codes in [-4, 3] and one binary16 multiplier. The base plane stores each
// group as one little-endian bit stream: lane i occupies bits 3i..3i+2, so bytes 3j..3j+2 hold the
// 24-bit word u[8j] | u[8j+1] << 3 | ... | u[8j+7] << 21 with u = code mod 8 (two's complement).
// A group is 48 bytes; a row is (K_pad / 128) * 48 bytes; there is no high plane. The layout is
// byte-identical to roofkid/ninfer-4080 Q3G128_F16S (Apache-2.0), so its Q3 matrices interchange.
//
// Everything here is plain integer arithmetic usable from host code, so the CPU tests exercise the
// exact decode the kernels run.

#include <cstdint>

#if defined(__CUDACC__)
#    define NINFER_Q3_HD __host__ __device__ __forceinline__
#else
#    define NINFER_Q3_HD inline
#endif

namespace ninfer::ops::detail {

struct Q3RowSplitStorage {
    static constexpr int kGroupK             = 128;
    static constexpr int kCodeBytesPerGroup  = 48;
    static constexpr int kScaleBytesPerGroup = 2;
    // A decode unit: 32 consecutive codes of one group = 12 bytes = three 32-bit words.
    static constexpr int kUnitCodes  = 32;
    static constexpr int kUnitBytes  = 12;
    static constexpr int kUnitsGroup = kGroupK / kUnitCodes;
};

// XOR with this mask turns eight two's-complement 3-bit fields of a 24-bit word into their offset
// form u' = code + 4 (bit 2 of every field flips).
inline constexpr std::uint32_t kQ3SignFlip24 = 0x924924u;

// Signed code of field `lane` (0..7) of a 24-bit word.
NINFER_Q3_HD int q3_code(std::uint32_t word24, int lane) {
    return static_cast<int>(((word24 ^ kQ3SignFlip24) >> (3 * lane)) & 7u) - 4;
}

// The four 24-bit words of one 12-byte unit (32 codes) held as three little-endian 32-bit words.
NINFER_Q3_HD void q3_unit_words(std::uint32_t w0, std::uint32_t w1, std::uint32_t w2,
                                std::uint32_t (&word24)[4]) {
    word24[0] = w0 & 0xffffffu;
    word24[1] = (w0 >> 24) | ((w1 & 0xffffu) << 8);
    word24[2] = (w1 >> 16) | ((w2 & 0xffu) << 16);
    word24[3] = w2 >> 8;
}

// 32 signed codes of one unit, as exact floats.
NINFER_Q3_HD void q3_decode_unit(std::uint32_t w0, std::uint32_t w1, std::uint32_t w2,
                                 float (&codes)[32]) {
    std::uint32_t word24[4];
    q3_unit_words(w0, w1, w2, word24);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const std::uint32_t u = word24[j] ^ kQ3SignFlip24;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            codes[8 * j + i] = static_cast<float>(static_cast<int>((u >> (3 * i)) & 7u) - 4);
        }
    }
}

// Host/device scalar reference: code of logical column `k` of a row whose base plane starts at
// `row_codes`.
NINFER_Q3_HD int q3_row_code(const std::uint8_t* row_codes, int k) {
    const int byte             = (k / 8) * 3;
    const std::uint32_t word24 = static_cast<std::uint32_t>(row_codes[byte]) |
                                 (static_cast<std::uint32_t>(row_codes[byte + 1]) << 8) |
                                 (static_cast<std::uint32_t>(row_codes[byte + 2]) << 16);
    return q3_code(word24, k % 8);
}

} // namespace ninfer::ops::detail
