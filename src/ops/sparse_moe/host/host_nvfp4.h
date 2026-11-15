#pragma once

// NVFP4 W4A8 row dots on the host CPU over block_scale_k16_m128x4_v1 planes, in two instruction
// sets chosen at run time (HostIsa): AVX-512 F/BW/VL/VNNI and AVX2 + FMA + F16C (Zen 2/3, Intel
// client cores without AVX-512). Both compute the same integer block sums and apply the same FP32
// operations to the same 16 lanes in the same order (the AVX2 kernels keep each 512-bit lane set as
// two 256-bit halves and reduce them as _mm512_reduce_add_ps does), so their results are identical
// bit for bit (tests/ops/test_host_nvfp4_isa.cpp).
//
// Weights: E2M1 codes with one E4M3FN scale per 16 K elements and one FP32 divisor per stacked
// source matrix (docs/maintainer/storage-layouts.md section 4): w = e2m1 * e4m3 / divisor.
// Activations: int8 per 16-element block, q = rint(x / s) with s = absmax / 127 (s = 1 for an
// all-zero block). The integer block sums are exact (vpdpbusd); the per-block float product
// e4m3 * s and the FP32 accumulation are the only roundings. This is FreeToken's W4A8 recipe
// (python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp @555efd8, Apache-2.0, FlashML-org /
// FreeToken authors), read here from the swizzled scale plane used by the GPU kernels.
//
// The dot decodes every 64-weight group once for all the tokens of the call, one vpdpbusd per
// token. The result of a token does not depend on how many tokens share the call (bit-identical
// for T = 1..8).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace ninfer::ops::host {

inline constexpr std::int32_t kNvfp4Group      = 16;  // K elements per E4M3 scale
inline constexpr std::int32_t kNvfp4Quad       = 64;  // K elements per scale word quad
inline constexpr std::int32_t kNvfp4RowTile    = 128; // rows per swizzled scale tile
inline constexpr std::int32_t kScaleQuadStride = 512; // bytes between quads of one row
inline constexpr std::int32_t kHostMaxTokens   = 8;   // tokens per dot call

// One NVFP4 matrix [rows, k] in block_scale_k16_m128x4_v1. Divisor i serves rows
// [i * rows_per_divisor, (i + 1) * rows_per_divisor).
struct Nvfp4Plane {
    const std::uint8_t* codes     = nullptr;
    const std::uint8_t* scales    = nullptr;
    const float* divisors         = nullptr;
    std::int64_t rows             = 0;
    std::int32_t k                = 0;
    std::int64_t rows_per_divisor = 0;

    // Views the stored payload of a [rows, k] plane with `divisor_count` divisor words.
    [[nodiscard]] static Nvfp4Plane from_payload(const void* payload, std::int64_t rows,
                                                 std::int32_t k, std::int64_t divisor_count);
    // Payload bytes of such a plane (codes, padding to 256, scales, divisors).
    [[nodiscard]] static std::size_t payload_bytes(std::int64_t rows, std::int32_t k,
                                                   std::int64_t divisor_count);

    [[nodiscard]] const std::uint8_t* row_codes(std::int64_t row) const noexcept {
        return codes + static_cast<std::size_t>(row) * static_cast<std::size_t>(k / 2);
    }

    // Scale word of (row, quad 0); quad q of the row is at + q * kScaleQuadStride.
    [[nodiscard]] const std::uint8_t* row_scales(std::int64_t row) const noexcept {
        const std::int64_t quads = k / kNvfp4Quad;
        return scales +
               static_cast<std::size_t>((row / kNvfp4RowTile) * quads * kScaleQuadStride +
                                        (row % 32) * 16 + ((row % kNvfp4RowTile) / 32) * 4);
    }

    [[nodiscard]] float divisor(std::int64_t row) const noexcept {
        return divisors[row / rows_per_divisor];
    }
};

// One token's prepared activation of length k (k % 64 == 0). Per 16-block b: q[16b .. 16b+8) =
// the even elements, q[16b+8 .. 16b+16) = the odd ones (the nibble order of a code byte);
// asb[b] = block scale; corr[d] = -12 * (q[4d] + ... + q[4d+3]); sb16[d] = asb[d / 4];
// qu = q + 128 (u8).
struct W4A8Activation {
    std::int8_t* q     = nullptr;
    float* asb         = nullptr;
    std::int32_t* corr = nullptr;
    float* sb16        = nullptr;
    std::uint8_t* qu   = nullptr;
};

// Owns `count` prepared activations of length k, 64-byte aligned.
class W4A8ActivationBuffer {
public:
    W4A8ActivationBuffer() = default;
    W4A8ActivationBuffer(std::int32_t k, std::int32_t count);
    [[nodiscard]] W4A8Activation at(std::int32_t i) const noexcept;

    [[nodiscard]] std::int32_t k() const noexcept { return k_; }

    [[nodiscard]] std::int32_t count() const noexcept { return count_; }

private:
    struct Free {
        void operator()(void* p) const noexcept;
    };

    std::unique_ptr<std::byte, Free> storage_;
    std::int32_t k_ = 0, count_ = 0;
    std::size_t q_stride_ = 0, asb_stride_ = 0, d_stride_ = 0;
    std::byte* q_    = nullptr;
    std::byte* asb_  = nullptr;
    std::byte* corr_ = nullptr;
    std::byte* sb16_ = nullptr;
    std::byte* qu_   = nullptr;
};

// Instruction set of the host expert kernels.
enum class HostIsa : std::uint8_t {
    Avx2,       // AVX2 + FMA + F16C (Haswell / Zen 1 and later)
    Avx512Vnni, // AVX-512 F/BW/VL/VNNI + AVX2 + FMA (Ice Lake / Zen 4 and later)
};

// True when the CPU (and the OS: cpuid + xgetbv) runs `isa`.
[[nodiscard]] bool host_isa_supported(HostIsa isa) noexcept;
// True when the CPU runs one of the ISAs (AVX2 + FMA + F16C at least).
[[nodiscard]] bool host_experts_supported() noexcept;
// True when the CPU runs AVX-512 F/BW/VL/VNNI.
[[nodiscard]] bool host_avx512_vnni_supported() noexcept;
// The ISA of the functions below and of the executor: the widest the CPU runs, or the one named by
// NINFER_HOST_ISA (avx2 | avx512 | auto), read once. Throws std::runtime_error when the CPU runs
// none, when NINFER_HOST_ISA names one it lacks, or names none of these.
[[nodiscard]] HostIsa host_isa();
[[nodiscard]] const char* host_isa_name(HostIsa isa) noexcept;

// The kernels of one ISA (the free functions below are those of host_isa()).
struct HostNvfp4Kernels {
    HostIsa isa;
    void (*quantize_blocks)(const float* x, std::int32_t block0, std::int32_t blocks,
                            W4A8Activation a);
    void (*quantize_bf16)(const std::uint16_t* x, std::int32_t k, W4A8Activation a);
    void (*quantize_blocks_a4)(const float* x, std::int32_t block0, std::int32_t blocks, float gs,
                               W4A8Activation a);
    void (*quantize_bf16_a4)(const std::uint16_t* x, std::int32_t k, float gs, W4A8Activation a);
    void (*rows_dot)(const Nvfp4Plane& plane, std::int64_t r0, std::int32_t n,
                     const W4A8Activation* acts, std::int32_t tokens, float* out, std::size_t ldo);
};

// Throws std::runtime_error when the CPU does not run `isa`.
[[nodiscard]] const HostNvfp4Kernels& host_nvfp4_kernels(HostIsa isa);
// host_nvfp4_kernels(isa ? *isa : host_isa()).
[[nodiscard]] const HostNvfp4Kernels& host_nvfp4_kernels(std::optional<HostIsa> isa = {});

// Quantizes blocks [block0, block0 + blocks) of `a` from FP32 x (x[0] is element 16*block0).
void quantize_blocks(const float* x, std::int32_t block0, std::int32_t blocks, W4A8Activation a);
// Quantizes a whole BF16 row of length k (raw BF16 bits).
void quantize_bf16(const std::uint16_t* x, std::int32_t k, W4A8Activation a);
// A4 (default; NINFER_HOST_MOE_A4=0 turns it off): the GPU's NVFP4 activation codec in the same
// prepared form, so the W4A8 dots compute W4A4: per 16 values sf = e4m3_rn(gs * amax / 6), code =
// e2m1_rn(x * gs / sf) (moe_slots_nvfp4 / test_moe_slots quant16), stored as q = 2 * code (|q| <=
// 12) with block scale sf / (2 gs); `gs` = the layer's input divisor of this activation.
void quantize_blocks_a4(const float* x, std::int32_t block0, std::int32_t blocks, float gs,
                        W4A8Activation a);
void quantize_bf16_a4(const std::uint16_t* x, std::int32_t k, float gs, W4A8Activation a);

// out[t * ldo + i] = sum_k w[r0 + i, k] * a_t[k] for i < n, t < tokens (1..kHostMaxTokens).
void rows_dot(const Nvfp4Plane& plane, std::int64_t r0, std::int32_t n, const W4A8Activation* acts,
              std::int32_t tokens, float* out, std::size_t ldo);

// Decoded weight w[row, k] (exact; the test oracle uses its own decoder).
[[nodiscard]] float e4m3_value(std::uint8_t code) noexcept;

} // namespace ninfer::ops::host
