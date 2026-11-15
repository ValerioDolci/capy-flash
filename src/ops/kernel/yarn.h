#pragma once
// YaRN (Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models") for
// text RoPE, host side: the inverse-frequency table and the attention factor of Hugging Face
// transformers' rope_type "yarn" (_compute_yarn_parameters in modeling_rope_utils.py).
//
// Rotary pairs whose wavelength completes more than beta_fast turns across the original window
// keep their frequency (extrapolation); pairs completing fewer than beta_slow turns are divided
// by the factor (interpolation); a linear ramp in the pair index blends the two in between. The
// attention factor multiplies cos and sin, compensating the flatter attention of the stretched
// positions.
//
// The correction range is computed in FP64 and every per-pair value in FP32, operation by
// operation as transformers does. On the reference build (x86-64, no FMA contraction) the table
// equals the bits transformers 5.16.1 gave (ninfer_yarn_test); another libm or torch build may
// round a pow differently in the last place.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ninfer::rope {

struct YarnParams {
    float factor              = 1.0F;
    int original_max_position = 262144;
    float beta_fast           = 32.0F;
    float beta_slow           = 1.0F;
    float theta               = 1.0e7F;
    int rotary_dim            = 64;
};

// The (fractional) pair index whose wavelength completes `rotations` turns across the original
// window.
inline double correction_dim(double rotations, int dim, double base, int original_max_position) {
    const double kPi = 3.141592653589793;
    return (static_cast<double>(dim) *
            std::log(static_cast<double>(original_max_position) / (rotations * 2.0 * kPi))) /
           (2.0 * std::log(base));
}

// rotary_dim / 2 inverse frequencies, pair i first.
inline std::vector<float> yarn_inv_freq(const YarnParams& p) {
    const int dim     = p.rotary_dim;
    const double base = p.theta;
    double low        = std::floor(correction_dim(p.beta_fast, dim, base, p.original_max_position));
    double high       = std::ceil(correction_dim(p.beta_slow, dim, base, p.original_max_position));
    low               = std::max(low, 0.0);
    high              = std::min(high, static_cast<double>(dim - 1));
    if (low == high) { high += 0.001; }
    const float ramp_low  = static_cast<float>(low);
    const float ramp_span = static_cast<float>(high - low);

    std::vector<float> out(static_cast<std::size_t>(dim / 2));
    for (int i = 0; i < dim / 2; ++i) {
        const float exponent      = static_cast<float>(2 * i) / static_cast<float>(dim);
        const float pos_freq      = std::pow(p.theta, exponent);
        const float extrapolation = 1.0F / pos_freq;
        const float interpolation = 1.0F / (p.factor * pos_freq);
        const float ramp = std::clamp((static_cast<float>(i) - ramp_low) / ramp_span, 0.0F, 1.0F);
        const float extrapolation_weight = 1.0F - ramp;
        const float interpolated         = interpolation * (1.0F - extrapolation_weight);
        const float extrapolated         = extrapolation * extrapolation_weight;
        out[static_cast<std::size_t>(i)] = interpolated + extrapolated;
    }
    return out;
}

// 0.1 ln(factor) + 1 in FP64, rounded once to FP32; 1 for factor <= 1.
inline float yarn_attention_scale(const YarnParams& p) {
    if (p.factor <= 1.0F) { return 1.0F; }
    return static_cast<float>(0.1 * std::log(static_cast<double>(p.factor)) + 1.0);
}

} // namespace ninfer::rope
