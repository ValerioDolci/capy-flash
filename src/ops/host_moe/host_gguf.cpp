// Modified for fngine from llama.cpp/ggml:ggml/src/ggml-cpu/arch/x86/quants.c @
// 3cf03257f219afbe7334045ff7c6a06ac68c627d. See NOTICE.
// SPDX-License-Identifier: MIT
//
// Modified for capy-flash from fngine:src/cpu/iq_cpu.cpp @ c20a885 (satellitedown/fngine; this
// file is SPDX MIT because it derives from ggml). See NOTICE.
//
// The baseline (x86-64, no AVX) part of the GGUF host kernels: formats and sizes, the
// run-time choice of the instruction set and the dispatch of the public functions of host_gguf.h
// and host_q8.h to gguf::detail::avx512 (host_gguf_avx512.cpp, host_q8.cpp) or gguf::detail::avx2
// (host_gguf_avx2.cpp, host_q8_avx2.cpp). Nothing here executes AVX instructions.
#include "ops/host_moe/host_gguf.h"

#include "ops/host_moe/host_gguf_isa.h"
#include "ops/host_moe/host_q8.h"

#include <algorithm>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops::host::gguf {

// ------------------------------------------------------------------ formats
FormatInfo format_info(WeightFormat fmt) {
    switch (fmt) {
    case WeightFormat::IQ2_S:
        return {QK_K, int32_t(sizeof(block_iq2_s)), ActFormat::Q8_K};
    case WeightFormat::IQ3_XXS:
        return {QK_K, int32_t(sizeof(block_iq3_xxs)), ActFormat::Q8_K};
    case WeightFormat::IQ3_S:
        return {QK_K, int32_t(sizeof(block_iq3_s)), ActFormat::Q8_K};
    case WeightFormat::IQ4_XS:
        return {QK_K, int32_t(sizeof(block_iq4_xs)), ActFormat::Q8_K};
    case WeightFormat::IQ4_NL:
        return {QK4_NL, int32_t(sizeof(block_iq4_nl)), ActFormat::Q8_0};
    case WeightFormat::Q2_0:
        return {QK2_0, int32_t(sizeof(block_q2_0)), ActFormat::Q8_0};
    }
    throw std::invalid_argument("unknown weight format " + std::to_string(int32_t(fmt)));
}

const char* format_name(WeightFormat fmt) {
    switch (fmt) {
    case WeightFormat::IQ2_S:
        return "IQ2_S";
    case WeightFormat::IQ3_XXS:
        return "IQ3_XXS";
    case WeightFormat::IQ3_S:
        return "IQ3_S";
    case WeightFormat::IQ4_XS:
        return "IQ4_XS";
    case WeightFormat::IQ4_NL:
        return "IQ4_NL";
    case WeightFormat::Q2_0:
        return "Q2_0";
    }
    return "?";
}

bool parse_weight_format(std::string_view name, WeightFormat* out) {
    static constexpr WeightFormat kAll[] = {WeightFormat::IQ2_S,  WeightFormat::IQ3_XXS,
                                            WeightFormat::IQ3_S,  WeightFormat::IQ4_XS,
                                            WeightFormat::IQ4_NL, WeightFormat::Q2_0};
    for (WeightFormat f : kAll) {
        if (name == format_name(f)) {
            *out = f;
            return true;
        }
    }
    return false;
}

bool is_weight_format(int32_t id) {
    switch (WeightFormat(id)) {
    case WeightFormat::IQ2_S:
    case WeightFormat::IQ3_XXS:
    case WeightFormat::IQ3_S:
    case WeightFormat::IQ4_XS:
    case WeightFormat::IQ4_NL:
    case WeightFormat::Q2_0:
        return true;
    }
    return false;
}

void detail::check_k(WeightFormat fmt, int32_t k) {
    const FormatInfo fi = format_info(fmt);
    if (k <= 0 || k % fi.block_elems != 0)
        throw std::invalid_argument(std::string("row length not a multiple of the ") +
                                    format_name(fmt) + " block size");
}

std::size_t weight_row_bytes(WeightFormat fmt, int32_t k) {
    detail::check_k(fmt, k);
    const FormatInfo fi = format_info(fmt);
    return std::size_t(k / fi.block_elems) * std::size_t(fi.block_bytes);
}

std::size_t act_row_bytes(ActFormat act, int32_t k) {
    if (act == ActFormat::Q8_K) {
        if (k % QK_K != 0) throw std::invalid_argument("Q8_K row length not a multiple of 256");
        return std::size_t(k / QK_K) * sizeof(block_q8_K);
    }
    if (k % QK8_0 != 0) throw std::invalid_argument("Q8_0 row length not a multiple of 32");
    return std::size_t(k / QK8_0) * sizeof(block_q8_0);
}

std::size_t decoded_row_bytes(WeightFormat fmt, int32_t k) {
    detail::check_k(fmt, k);
    const ActFormat a = format_info(fmt).act;
    return std::size_t(detail::groups_of(a, k)) * detail::wgroup_bytes(detail::chunks_of(a));
}

std::size_t packed_act_bytes(ActFormat act, int32_t k) {
    act_row_bytes(act, k); // validates k
    return std::size_t(detail::groups_of(act, k)) * detail::agroup_bytes(detail::chunks_of(act));
}

// ------------------------------------------------------------------ instruction sets
// __builtin_cpu_supports reads cpuid and, for the AVX/AVX-512 features, the OS-enabled state.
bool host_gguf_avx512_supported() noexcept {
    return host_isa_supported(HostIsa::Avx2) && __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl") &&
           __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vnni") &&
           __builtin_cpu_supports("avx512vbmi") && __builtin_cpu_supports("avx512bitalg") &&
           __builtin_cpu_supports("avx512vpopcntdq") && __builtin_cpu_supports("avx512bf16") &&
           __builtin_cpu_supports("bmi2");
}

bool host_gguf_avxvnni_supported() noexcept {
    return host_isa_supported(HostIsa::Avx2) && __builtin_cpu_supports("avxvnni");
}

bool host_gguf_isa_supported(HostIsa isa) noexcept {
    switch (isa) {
    case HostIsa::Avx2:
        return host_isa_supported(HostIsa::Avx2);
    case HostIsa::Avx512Vnni:
        return host_gguf_avx512_supported();
    }
    return false;
}

HostIsa host_gguf_isa(std::optional<HostIsa> isa) {
    const HostIsa want = isa ? *isa : host_isa();
    if (want == HostIsa::Avx512Vnni && host_gguf_avx512_supported()) return HostIsa::Avx512Vnni;
    if (!host_isa_supported(HostIsa::Avx2))
        throw std::runtime_error("GGUF host kernels: this CPU does not run AVX2 + FMA + F16C");
    return HostIsa::Avx2;
}

namespace {

#define NINFER_GGUF_KERNELS(ISA, NS)                                                               \
    HostGgufKernels {                                                                              \
        ISA, detail::NS::decode_rows, detail::NS::decode_rows_big, detail::NS::pack_act_q8_K,      \
            detail::NS::pack_act_q8_0, detail::NS::quantize_pack_bf16,                             \
            detail::NS::quantize_pack_f32, detail::NS::quantize_pack_bf16_q8k_g32,                 \
            detail::NS::quantize_pack_bf16_q8f32, detail::NS::quantize_pack_f32_q8f32,             \
            detail::NS::quantize_pack_q8_0_half_q8f32, detail::NS::quantize_pack_q8_0_half,        \
            detail::NS::gemm_decoded, detail::NS::decode_rows_bf16, detail::NS::gemm_bf16,         \
            detail::NS::silu_mul, detail::NS::quantize_row_q8_K_bf16,                              \
            detail::NS::quantize_row_q8_K_f32, detail::NS::quantize_row_q8_0_bf16,                 \
            detail::NS::quantize_row_q8_0_f32                                                      \
    }

constexpr HostGgufKernels kAvx512Kernels   = NINFER_GGUF_KERNELS(HostIsa::Avx512Vnni, avx512);
constexpr HostGgufKernels kAvx2Kernels     = NINFER_GGUF_KERNELS(HostIsa::Avx2, avx2);
constexpr HostGgufKernels kAvx2VnniKernels = [] {
    HostGgufKernels k = kAvx2Kernels;
    k.gemm_decoded    = detail::avx2::gemm_decoded_vnni;
    return k;
}();

#undef NINFER_GGUF_KERNELS

const HostGgufKernels& active() {
    static const HostGgufKernels& k = host_gguf_kernels(host_gguf_isa());
    return k;
}

} // namespace

const HostGgufKernels& host_gguf_kernels(HostIsa isa) {
    if (!host_gguf_isa_supported(isa)) {
        throw std::runtime_error(std::string("GGUF host kernels: this CPU does not run ") +
                                 (isa == HostIsa::Avx2
                                      ? "AVX2 + FMA + F16C"
                                      : "AVX-512 F/BW/VL/DQ/VNNI/VBMI/BITALG/VPOPCNTDQ/BF16"));
    }
    if (isa == HostIsa::Avx512Vnni) return kAvx512Kernels;
    static const bool vnni = host_gguf_avxvnni_supported();
    return vnni ? kAvx2VnniKernels : kAvx2Kernels;
}

// ------------------------------------------------------------------ dispatched entry points
void decode_rows(WeightFormat fmt, const void* rows, std::size_t row_bytes, int32_t nrows,
                 int32_t k, void* dst) {
    active().decode_rows(fmt, rows, row_bytes, nrows, k, dst);
}

void pack_act_q8_K(const block_q8_K* x, int32_t k, void* dst) { active().pack_act_q8_K(x, k, dst); }

void pack_act_q8_0(const block_q8_0* x, int32_t k, void* dst) { active().pack_act_q8_0(x, k, dst); }

void quantize_pack_bf16(ActFormat act, const uint16_t* x, int32_t k, void* dst) {
    active().quantize_pack_bf16(act, x, k, dst);
}

void quantize_pack_f32(ActFormat act, const float* x, int32_t k, void* dst) {
    active().quantize_pack_f32(act, x, k, dst);
}

void quantize_pack_bf16_q8k_g32(const uint16_t* x, int32_t k, void* dst) {
    active().quantize_pack_bf16_q8k_g32(x, k, dst);
}

void quantize_pack_bf16_q8f32(ActFormat act, const uint16_t* x, int32_t k, void* dst) {
    active().quantize_pack_bf16_q8f32(act, x, k, dst);
}

void quantize_pack_f32_q8f32(ActFormat act, const float* x, int32_t k, void* dst) {
    active().quantize_pack_f32_q8f32(act, x, k, dst);
}

void quantize_pack_q8_0_half_q8f32(const float* x32, int32_t group, int32_t half, void* dst) {
    active().quantize_pack_q8_0_half_q8f32(x32, group, half, dst);
}

void quantize_pack_q8_0_half(const float* x32, int32_t group, int32_t half, void* dst) {
    active().quantize_pack_q8_0_half(x32, group, half, dst);
}

void gemm_decoded(WeightFormat fmt, const void* wdec, int32_t nrows, int32_t k,
                  const void* const* acts, int32_t T, float* out, std::size_t ldo) {
    active().gemm_decoded(fmt, wdec, nrows, k, acts, T, out, ldo);
}

void decode_rows_bf16(WeightFormat fmt, const void* rows, std::size_t row_bytes, int32_t nrows,
                      int32_t k, uint16_t* dst) {
    active().decode_rows_bf16(fmt, rows, row_bytes, nrows, k, dst);
}

void gemm_bf16(const uint16_t* weights, int32_t nrows, int32_t k, const void* const* acts,
               int32_t T, float* out, std::size_t ldo) {
    active().gemm_bf16(weights, nrows, k, acts, T, out, ldo);
}

void silu_mul(const float* g, const float* u, float* h, int32_t n) {
    active().silu_mul(g, u, h, n);
}

void quantize_row_q8_K_bf16(const uint16_t* x, block_q8_K* y, int64_t k) {
    active().quantize_row_q8_K_bf16(x, y, k);
}

void quantize_row_q8_K_f32(const float* x, block_q8_K* y, int64_t k) {
    active().quantize_row_q8_K_f32(x, y, k);
}

void quantize_row_q8_0_bf16(const uint16_t* x, block_q8_0* y, int64_t k) {
    active().quantize_row_q8_0_bf16(x, y, k);
}

void quantize_row_q8_0_f32(const float* x, block_q8_0* y, int64_t k) {
    active().quantize_row_q8_0_f32(x, y, k);
}

// ------------------------------------------------------------------ cpu_dot (tests, probes)
namespace {

struct Scratch {
    uint8_t* p      = nullptr;
    std::size_t cap = 0;

    ~Scratch() { std::free(p); }

    uint8_t* get(std::size_t bytes) {
        if (bytes > cap) {
            std::free(p);
            cap = (bytes + 4095) & ~std::size_t{4095};
            p   = static_cast<uint8_t*>(std::aligned_alloc(64, cap));
            if (!p) throw std::bad_alloc();
        }
        return p;
    }
};

void dot_impl(WeightFormat fmt, const void* row, int nblk, int T,
              const uint8_t* (*act_at)(const void*, std::size_t, int), const void* acts,
              std::size_t stride, float* out) {
    const HostGgufKernels& K = active();
    const FormatInfo fi      = format_info(fmt);
    const int32_t k          = nblk * fi.block_elems;
    thread_local Scratch wbuf, abuf;
    thread_local std::vector<const void*> ptrs;
    const std::size_t pb = packed_act_bytes(fi.act, k);
    uint8_t* ap          = abuf.get(pb * std::size_t(std::max(T, 1)));
    ptrs.resize(std::size_t(std::max(T, 1)));
    for (int t = 0; t < T; ++t) {
        const uint8_t* src = act_at(acts, stride, t);
        if (fi.act == ActFormat::Q8_K)
            K.pack_act_q8_K(reinterpret_cast<const block_q8_K*>(src), k, ap + pb * t);
        else
            K.pack_act_q8_0(reinterpret_cast<const block_q8_0*>(src), k, ap + pb * t);
        ptrs[std::size_t(t)] = ap + pb * t;
    }
    uint8_t* wp = wbuf.get(decoded_row_bytes(fmt, k));
    K.decode_rows(fmt, row, weight_row_bytes(fmt, k), 1, k, wp);
    K.gemm_decoded(fmt, wp, 1, k, ptrs.data(), T, out, std::size_t(T));
}

} // namespace

void cpu_dot(WeightFormat fmt, const void* row, int nblk, const void* acts, int T,
             std::size_t act_stride_bytes, float* out) {
    dot_impl(
        fmt, row, nblk, T,
        [](const void* a, std::size_t s, int t) {
            return static_cast<const uint8_t*>(a) + std::size_t(t) * s;
        },
        acts, act_stride_bytes, out);
}

void cpu_dot_ptr(WeightFormat fmt, const void* row, int nblk, const void* const* acts, int T,
                 float* out) {
    dot_impl(
        fmt, row, nblk, T,
        [](const void* a, std::size_t, int t) {
            return static_cast<const uint8_t*>(static_cast<const void* const*>(a)[t]);
        },
        acts, 0, out);
}

} // namespace ninfer::ops::host::gguf
