#pragma once

#include "core/dtype.h"

#include <cstdint>

namespace ninfer {

enum class QType : std::uint16_t {
    Q4_G64_FP16            = 0,
    Q5_G64_FP16            = 1,
    Q6_G64_FP16            = 2,
    Q8_G32_FP16            = 3,
    BF16                   = 4,
    FP32                   = 5,
    INT32                  = 6,
    NVFP4                  = 7,
    FP8_E4M3FN_ROW_BF16    = 8,
    MXFP8_E4M3FN_G32_UE8M0 = 9,
    NVFP4_ROW_E4M3_G16     = 10,
    // ggml block types of the GSQ-RCO routed-expert banks, stored as whole ggml blocks per row
    // (layout GgufBlocks): ggml-common.h's block_iq3_xxs, block_iq4_nl, block_iq3_s, block_iq2_s,
    // block_iq4_xs and block_q2_0.
    GGUF_IQ3_XXS = 11,
    GGUF_IQ4_NL  = 12,
    GGUF_IQ3_S   = 13,
    GGUF_IQ2_S   = 14,
    GGUF_IQ4_XS  = 15,
    GGUF_Q2_0    = 16,
    Q3_G128_FP16 = 17, // 27b-16gb: 3-bit GSQ body, 48 code bytes per 128 (artifacts store the name)
};

enum class QuantLayout : std::uint16_t {
    RowSplit            = 0,
    Contiguous          = 1,
    BlockScaleK16M128x4 = 2,
    RowScale            = 3,
    BlockScaleK32       = 4,
    RowInterleaved      = 5,
    GgufBlocks          = 6,
};

struct Weight {
    const void* payload            = nullptr;
    std::uint64_t payload_bytes    = 0;
    std::uint64_t high_plane_bytes = 0;
    QType qtype                    = QType::Q4_G64_FP16;
    std::uint32_t group_size       = 0;
    std::int32_t shape[4]          = {1, 1, 1, 1};
    std::int32_t padded_shape[4]   = {1, 1, 1, 1};
    std::uint32_t ndim             = 0;

    const void* qdata          = nullptr;
    const void* qhigh          = nullptr;
    const void* scales         = nullptr;
    std::int32_t n             = 0;
    std::int32_t k             = 0;
    std::int32_t group         = 0;
    QuantLayout layout         = QuantLayout::RowSplit;
    DType scale_dtype          = DType::FP32;
    std::int32_t scale_ne[4]   = {1, 1, 1, 1};
    std::int64_t scale_nb[4]   = {0, 0, 0, 0};
    float weight_scale_divisor = 0.0F;
    float input_scale_divisor  = 0.0F;
};

} // namespace ninfer
