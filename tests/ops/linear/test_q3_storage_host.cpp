// CPU-only checks of the q3_g128_fp16 storage decode the GPU kernels run
// (src/ops/linear/q3/q3_rowsplit_storage.cuh). No CUDA device or runtime is used.
//
// Pre-registered criteria (all exact):
//  1. the reference vector codes -4..3 packs to bytes ac 8f 68 (storage-layouts.md §3.3, and the
//     same bytes as roofkid/ninfer-4080's Q3G128_F16S layout test);
//  2. for 4096 random groups, q3_row_code and q3_decode_unit return every code an independent
//     spec packer (one bit at a time) wrote;
//  3. the compressed-tensors stream of the GSQ checkpoints (unsigned u' = code + 4, same bit
//     positions) XOR the repeating mask 24 49 92 equals the stored base plane, byte for byte
//     (the converter decodes and re-packs; this pins the identity it relies on);
//  4. a kernel-order FP32 dot product (per 32-code unit: sum code*x, then scale*sum) stays within
//     2^-16 * sum|w*x| of an FP64 reference on 64 rows of K = 17408, i.e. accumulation order is
//     not a meaningful error source next to the BF16 output rounding (2^-9 relative).
#include "ops/linear/q3/q3_rowsplit_storage.cuh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using ninfer::ops::detail::q3_decode_unit;
using ninfer::ops::detail::q3_row_code;
using ninfer::ops::detail::Q3RowSplitStorage;

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what);
    }
}

// Independent packer straight from the specification: lane i, bit b -> stream bit 3i + b.
std::vector<std::uint8_t> pack_group(const std::vector<int>& codes, bool offset_form) {
    std::vector<std::uint8_t> out(codes.size() * 3 / 8, 0);
    for (std::size_t i = 0; i < codes.size(); ++i) {
        const unsigned u = offset_form ? static_cast<unsigned>(codes[i] + 4)
                                       : static_cast<unsigned>(codes[i]) & 7u;
        for (int b = 0; b < 3; ++b) {
            const std::size_t bit = 3 * i + b;
            if ((u >> b) & 1u) out[bit / 8] |= static_cast<std::uint8_t>(1u << (bit % 8));
        }
    }
    return out;
}

std::uint32_t le32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

} // namespace

int main() {
    // 1. Reference vector.
    {
        std::vector<int> codes{-4, -3, -2, -1, 0, 1, 2, 3};
        const auto bytes = pack_group(codes, false);
        check(bytes.size() == 3 && bytes[0] == 0xac && bytes[1] == 0x8f && bytes[2] == 0x68,
              "reference vector -4..3 -> ac 8f 68");
        for (int i = 0; i < 8; ++i)
            check(q3_row_code(bytes.data(), i) == codes[i], "reference decode");
    }

    std::mt19937 rng(20261019u);
    std::uniform_int_distribution<int> code_dist(-4, 3);

    // 2. Random groups through both decoders.
    {
        constexpr int kGroups = 4096;
        std::vector<int> codes(kGroups * 128);
        for (auto& c : codes) c = code_dist(rng);
        std::vector<std::uint8_t> plane;
        plane.reserve(kGroups * 48);
        for (int g = 0; g < kGroups; ++g) {
            std::vector<int> group(codes.begin() + g * 128, codes.begin() + (g + 1) * 128);
            const auto bytes = pack_group(group, false);
            plane.insert(plane.end(), bytes.begin(), bytes.end());
        }
        int scalar_bad = 0, unit_bad = 0;
        for (int i = 0; i < kGroups * 128; ++i) {
            if (q3_row_code(plane.data(), i) != codes[i]) ++scalar_bad;
        }
        for (int u = 0; u < kGroups * Q3RowSplitStorage::kUnitsGroup; ++u) {
            const std::uint8_t* p = plane.data() + u * Q3RowSplitStorage::kUnitBytes;
            float decoded[32];
            q3_decode_unit(le32(p), le32(p + 4), le32(p + 8), decoded);
            for (int j = 0; j < 32; ++j) {
                if (decoded[j] != static_cast<float>(codes[u * 32 + j])) ++unit_bad;
            }
        }
        check(scalar_bad == 0, "q3_row_code matches the spec packer");
        check(unit_bad == 0, "q3_decode_unit matches the spec packer");
        std::printf("random groups: %d codes, scalar mismatches %d, unit mismatches %d\n",
                    kGroups * 128, scalar_bad, unit_bad);

        // 3. compressed-tensors offset stream XOR 24 49 92 == stored plane.
        int xor_bad                            = 0;
        static constexpr std::uint8_t kMask[3] = {0x24, 0x49, 0x92};
        for (int g = 0; g < kGroups; ++g) {
            std::vector<int> group(codes.begin() + g * 128, codes.begin() + (g + 1) * 128);
            const auto offset = pack_group(group, true);
            for (int b = 0; b < 48; ++b) {
                if (static_cast<std::uint8_t>(offset[b] ^ kMask[b % 3]) != plane[g * 48 + b])
                    ++xor_bad;
            }
        }
        check(xor_bad == 0, "offset stream xor 24 49 92 equals the stored plane");
    }

    // 4. Kernel-order FP32 accumulation against FP64.
    {
        constexpr int kRows = 64, kK = 17408, kGroupsRow = kK / 128;
        std::normal_distribution<float> xdist(0.0F, 1.0F);
        std::uniform_real_distribution<float> sdist(0.002F, 0.03F);
        double worst = 0.0;
        for (int r = 0; r < kRows; ++r) {
            std::vector<int> codes(kK);
            for (auto& c : codes) c = code_dist(rng);
            std::vector<float> x(kK), scale(kGroupsRow);
            for (auto& v : x) v = xdist(rng);
            for (auto& s : scale) s = sdist(rng);
            std::vector<std::uint8_t> row;
            for (int g = 0; g < kGroupsRow; ++g) {
                std::vector<int> group(codes.begin() + g * 128, codes.begin() + (g + 1) * 128);
                const auto bytes = pack_group(group, false);
                row.insert(row.end(), bytes.begin(), bytes.end());
            }
            double ref = 0.0, magnitude = 0.0;
            for (int k = 0; k < kK; ++k) {
                const double w = static_cast<double>(codes[k]) * scale[k / 128];
                ref += w * x[k];
                magnitude += std::fabs(w * x[k]);
            }
            float acc = 0.0F;
            for (int u = 0; u < kK / 32; ++u) {
                const std::uint8_t* p = row.data() + u * 12;
                float decoded[32];
                q3_decode_unit(le32(p), le32(p + 4), le32(p + 8), decoded);
                float sum = 0.0F;
                for (int j = 0; j < 32; ++j) sum = std::fma(decoded[j], x[u * 32 + j], sum);
                acc = std::fma(scale[u / 4], sum, acc);
            }
            worst = std::fmax(worst, std::fabs(acc - ref) / magnitude);
        }
        std::printf("kernel-order FP32 vs FP64: worst |err| / sum|w*x| = %.3e (limit %.3e)\n",
                    worst, std::ldexp(1.0, -16));
        check(worst <= std::ldexp(1.0, -16), "kernel-order accumulation error");
    }

    std::printf("%s (%d failures)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
