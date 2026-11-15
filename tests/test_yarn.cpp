// test_yarn.cpp - the YaRN table (ops/kernel/yarn.h) against Hugging Face transformers.
//
// The expected bits are inv_freq and attention_scaling of transformers 5.16.1
// Qwen4ExpTextRotaryEmbedding with head_dim 256 and rope_parameters {rope_type "yarn", factor F,
// original_max_position_embeddings 262144, rope_theta 1e7, partial_rotary_factor 0.25}: rotary_dim
// 64, beta_fast 32, beta_slow 1, as that build computed them on the reference machine (another
// libm or torch build may round a pow differently in the last place). A wrong table does not
// crash; it moves every rotated position.
#include "ops/kernel/yarn.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

struct Expected {
    float factor;
    std::uint32_t attention_scale;
    std::uint32_t inv_freq[32];
};

const Expected kTransformers[] = {
    {2.0F,
     0x3f88df4eU,
     {0x3f800000U, 0x3f1ab32bU, 0x3ebaf81bU, 0x3e61f835U, 0x3e088d77U, 0x3da50956U, 0x3d47763fU,
      0x3cf11177U, 0x3c91ad39U, 0x3c301052U, 0x3bd4ca15U, 0x3b80967dU, 0x3b1b690dU, 0x3abbd3edU,
      0x3a6301e2U, 0x3a009b22U, 0x399111f4U, 0x3922ce9dU, 0x38b5a1aaU, 0x384939aeU, 0x37dd1727U,
      0x37707cc9U, 0x37012dabU, 0x369c1fc4U, 0x363cb0c1U, 0x35e40cc6U, 0x3589cf4aU, 0x35268e4cU,
      0x34c94c57U, 0x3473499dU, 0x3413048eU, 0x33b1af44U}},
    {4.0F,
     0x3f91be9cU,
     {0x3f800000U, 0x3f1ab32bU, 0x3ebaf81bU, 0x3e61f835U, 0x3e088d77U, 0x3da50956U, 0x3d47763fU,
      0x3cf11177U, 0x3c91ad39U, 0x3c301052U, 0x3bd4ca15U, 0x3b80967dU, 0x3b1b690dU, 0x3abbd3edU,
      0x3a6301e2U, 0x39f8a364U, 0x3986b53eU, 0x3910058bU, 0x38975c0dU, 0x381b7e06U, 0x379ac368U,
      0x3712f6edU, 0x36812dabU, 0x361c1fc4U, 0x35bcb0c1U, 0x35640cc6U, 0x3509cf4aU, 0x34a68e4cU,
      0x34494c57U, 0x33f3499dU, 0x3393048eU, 0x3331af44U}},
};

std::uint32_t bits(float value) {
    std::uint32_t out = 0;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

} // namespace

int main() {
    int failures = 0;
    for (const Expected& expected : kTransformers) {
        ninfer::rope::YarnParams params;
        params.factor     = expected.factor;
        const auto table  = ninfer::rope::yarn_inv_freq(params);
        const float scale = ninfer::rope::yarn_attention_scale(params);
        int wrong         = table.size() == 32 ? 0 : 1;
        for (std::size_t i = 0; i < table.size() && i < 32; ++i) {
            if (bits(table[i]) != expected.inv_freq[i]) {
                std::printf("  factor %g pair %zu: %08x, transformers %08x\n",
                            static_cast<double>(expected.factor), i, bits(table[i]),
                            expected.inv_freq[i]);
                ++wrong;
            }
        }
        if (bits(scale) != expected.attention_scale) {
            std::printf("  factor %g attention factor: %08x, transformers %08x\n",
                        static_cast<double>(expected.factor), bits(scale),
                        expected.attention_scale);
            ++wrong;
        }
        std::printf("[%s] factor %g: table and attention factor equal transformers bit for bit\n",
                    wrong == 0 ? "PASS" : "FAIL", static_cast<double>(expected.factor));
        failures += wrong;
    }
    // transformers' get_mscale: no attention factor up to factor 1.
    const bool unit = ninfer::rope::yarn_attention_scale(ninfer::rope::YarnParams{}) == 1.0F;
    std::printf("[%s] factor 1: attention factor 1\n", unit ? "PASS" : "FAIL");
    failures += unit ? 0 : 1;
    return failures != 0 ? 1 : 0;
}
