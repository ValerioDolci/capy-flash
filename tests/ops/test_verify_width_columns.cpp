// Under AllowA4Decode, the policy of the columns after the prefill, the two-device projections of
// the NVFP4 27B target give a token column the same bits at every width from 1 to 7 as at width 4.
//
// Speculative decoding verifies W = K+1 columns per round and decoding without a drafter runs one,
// so a column's bits decide the greedy text only if they do not depend on how many columns share
// the pass. Width 4 (MTP K=3) is the anchor: for each rank-local projection under AllowA4Decode,
// every column of every width 1..7 must equal, bit for bit, the column computed from the same
// input at width 4. The columns are distinct and dense (random NVFP4 codes, random BF16
// activations), so a different reduction order over K, a different activation precision or a
// different K partition changes the rounding of many outputs.
//
// The dense Q8 projections of Flash-Next (A16Only before and after the prefill) are held to the
// same rule at every registered Qwen4Exp shape, the two-device shards and the one-device problems.
#include "core/device.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/attn_input_proj/tp2/nvfp4_attn_input_shard.h"
#include "ops/direct_bf16_weight.h"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h"
#include "ops/gdn_input_proj/tp2/nvfp4_gdn_input_shard.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr ops::LinearPolicy kPolicy = ops::LinearPolicy::AllowA4Decode;
constexpr std::int32_t kAnchorWidth = 4;
constexpr std::int32_t kMaxWidth    = 7;
// The widest call of the Flash-Next dense Q8 cases (A16Only keeps the T=4 bits up to T=16).
constexpr std::int32_t kQ8MaxWidth = 16;
// Input columns 0..15: width w reads [0,w), the anchor passes of width 4 read [0,4), [4,8), ...
// and, when the widest call is not a multiple of 4, [max_width-4, max_width).
constexpr std::int32_t kColumns = kQ8MaxWidth;
// Independent input draws. A reduction-order change moves an A16 output across a BF16 rounding
// boundary on only a few rows of 5120 per draw, so each width is compared over several draws.
constexpr std::int32_t kDraws = 8;

std::uint32_t mix(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

std::vector<std::uint16_t> random_bf16(std::size_t count, std::uint32_t seed, float scale) {
    std::vector<std::uint16_t> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint32_t h = mix(static_cast<std::uint32_t>(i) * 0x9e3779b9U ^ seed);
        const float unit      = static_cast<float>(h >> 8) / static_cast<float>(1U << 24);
        values[i]             = f32_to_bf16((2.0F * unit - 1.0F) * scale);
    }
    return values;
}

// A registered NVFP4 problem with the fixture's scales and pseudo-random E2M1 codes.
quantized_weight::PackedWeight random_nvfp4(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor    = 0.13F;
    options.input_scale_divisor     = 3.5F;
    options.decorrelate_coordinates = true;
    auto packed = quantized_weight::make_patterned_weight(QType::NVFP4, n, k, seed, options);
    const std::size_t code_bytes = static_cast<std::size_t>(n) * k / 2;
    for (std::size_t i = 0; i < code_bytes; ++i) {
        packed.payload[i] = static_cast<std::uint8_t>(mix(static_cast<std::uint32_t>(i) ^ seed));
    }
    return packed;
}

struct DeviceWeight {
    explicit DeviceWeight(const quantized_weight::PackedWeight& packed)
        : buffer(packed.payload.size()) {
        buffer.copy_from_host(packed.payload.data(), packed.payload.size());
        weight = packed.device_weight(buffer.data());
    }

    GuardedDeviceBuffer buffer;
    Weight weight{};
};

// A registered Q8 problem with pseudo-random codes and scales.
quantized_weight::PackedWeight random_q8(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    return quantized_weight::make_patterned_weight(QType::Q8_G32_FP16, n, k, seed,
                                                   {quantized_weight::RowSplitScalePattern::Small,
                                                    quantized_weight::RowSplitCodePattern::Hashed});
}

// kDraws sets of kColumns input columns [k] on the device; view(first, width) is
// x[:, first:first+width] of the current draw.
struct DeviceColumns {
    DeviceColumns(std::int32_t k, std::uint32_t seed)
        : k(k), host(random_bf16(static_cast<std::size_t>(k) * kColumns * kDraws, seed, 2.0F)),
          buffer(host.size() * sizeof(std::uint16_t)) {
        buffer.copy_from_host(host.data(), buffer.bytes());
    }

    Tensor view(std::int32_t first, std::int32_t width) {
        auto* base = static_cast<std::uint16_t*>(buffer.data()) +
                     (static_cast<std::size_t>(draw) * kColumns + first) * k;
        return Tensor(base, DType::BF16, {k, width});
    }

    std::int32_t k;
    std::int32_t draw = 0;
    std::vector<std::uint16_t> host;
    GuardedDeviceBuffer buffer;
};

// One output of an Op: `rows` elements of `element_bytes` per token column.
struct OutputSpec {
    const char* name;
    std::int32_t rows;
    std::int32_t element_bytes;
};

// Device outputs of one pass, `width` columns each, optionally seeded from host columns.
struct Outputs {
    Outputs(const std::vector<OutputSpec>& specs, std::int32_t width) : specs(specs), width(width) {
        for (const OutputSpec& spec : specs) {
            buffers.emplace_back(static_cast<std::size_t>(spec.rows) * spec.element_bytes * width);
            buffers.back().fill(0);
        }
    }

    Tensor tensor(std::size_t output, DType dtype) {
        return Tensor(buffers[output].data(), dtype, {specs[output].rows, width});
    }

    std::vector<OutputSpec> specs;
    std::int32_t width;
    std::vector<GuardedDeviceBuffer> buffers;
};

// Enqueues the Op over input columns [first, first+width) into `out` on the legacy stream.
using Pass = std::function<void(std::int32_t first, std::int32_t width, Outputs& out)>;

struct Case {
    std::string name;
    std::vector<OutputSpec> outputs;
    DeviceColumns* x;
    Pass pass;
    std::int32_t max_width = kMaxWidth;
};

// Host bytes of every output, [output][column][row bytes].
using Columns = std::vector<std::vector<std::vector<std::uint8_t>>>;

Columns run(const Case& c, std::int32_t first, std::int32_t width, int& failures) {
    Outputs out(c.outputs, width);
    c.pass(first, width, out);
    cuda_check(cudaDeviceSynchronize(), "synchronize verify-width pass");
    Columns columns(c.outputs.size());
    for (std::size_t o = 0; o < c.outputs.size(); ++o) {
        const std::size_t column_bytes =
            static_cast<std::size_t>(c.outputs[o].rows) * c.outputs[o].element_bytes;
        std::vector<std::uint8_t> host(column_bytes * width);
        out.buffers[o].copy_to_host(host.data(), host.size());
        failures += out.buffers[o].verify_guards(c.name + " " + c.outputs[o].name +
                                                 " T=" + std::to_string(width));
        for (std::int32_t j = 0; j < width; ++j) {
            columns[o].emplace_back(host.begin() + static_cast<std::ptrdiff_t>(column_bytes * j),
                                    host.begin() +
                                        static_cast<std::ptrdiff_t>(column_bytes * (j + 1)));
        }
    }
    return columns;
}

std::size_t differing_elements(const std::vector<std::uint8_t>& a,
                               const std::vector<std::uint8_t>& b, std::int32_t element_bytes) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < a.size(); i += static_cast<std::size_t>(element_bytes)) {
        if (std::memcmp(a.data() + i, b.data() + i, static_cast<std::size_t>(element_bytes)) != 0)
            ++count;
    }
    return count;
}

int check(const Case& c) {
    int failures                 = 0;
    const std::int32_t max_width = c.max_width;
    std::vector<std::int32_t> firsts;
    for (std::int32_t first = 0; first + kAnchorWidth <= max_width; first += kAnchorWidth)
        firsts.push_back(first);
    const auto aligned = static_cast<std::int32_t>(firsts.size());
    if (max_width % kAnchorWidth != 0) firsts.push_back(max_width - kAnchorWidth);
    std::vector<std::size_t> differing(max_width + 1, 0), values(max_width + 1, 0);
    for (std::int32_t draw = 0; draw < kDraws; ++draw) {
        c.x->draw = draw;
        // Column j's anchor: the width-4 pass that holds it, [4*(j/4), 4*(j/4)+4) while that
        // fits, the last pass otherwise. A column computed by two passes (at max_width 7, column 3
        // at token 3 and at token 0) must agree.
        std::vector<Columns> passes;
        for (const std::int32_t first : firsts)
            passes.push_back(run(c, first, kAnchorWidth, failures));
        Columns anchor(c.outputs.size());
        for (std::size_t o = 0; o < c.outputs.size(); ++o) {
            for (std::int32_t j = 0; j < max_width; ++j) {
                const std::int32_t p =
                    std::min(j / kAnchorWidth, static_cast<std::int32_t>(firsts.size()) - 1);
                anchor[o].push_back(passes[p][o][j - firsts[p]]);
            }
            if (aligned < static_cast<std::int32_t>(firsts.size())) {
                const std::int32_t last = firsts.back();
                for (std::int32_t j = last; j < aligned * kAnchorWidth; ++j) {
                    if (passes.back()[o][j - last] != anchor[o][j]) {
                        std::cerr << c.name << " " << c.outputs[o].name << ": T=4 column " << j
                                  << " changes with its token position\n";
                        ++failures;
                    }
                }
            }
            const std::size_t nonzero = differing_elements(
                passes[0][o][0], std::vector<std::uint8_t>(passes[0][o][0].size(), 0),
                c.outputs[o].element_bytes);
            if (nonzero < static_cast<std::size_t>(c.outputs[o].rows) / 2) {
                std::cerr << c.name << " " << c.outputs[o].name << ": T=4 output is mostly zero\n";
                ++failures;
            }
        }
        for (std::int32_t width = 1; width <= max_width; ++width) {
            if (width == kAnchorWidth) continue;
            const Columns got = run(c, 0, width, failures);
            for (std::size_t o = 0; o < c.outputs.size(); ++o) {
                for (std::int32_t j = 0; j < width; ++j) {
                    differing[width] +=
                        differing_elements(got[o][j], anchor[o][j], c.outputs[o].element_bytes);
                    values[width] += static_cast<std::size_t>(c.outputs[o].rows);
                }
            }
        }
    }
    std::string changed;
    for (std::int32_t width = 1; width <= max_width; ++width) {
        if (differing[width] == 0) continue;
        std::cerr << c.name << " T=" << width << ": " << differing[width] << " of " << values[width]
                  << " values differ from the T=4 columns\n";
        changed += (changed.empty() ? "" : ",") + std::to_string(width);
        ++failures;
    }
    std::cout << c.name << ": "
              << (changed.empty()
                      ? "every column of T=1.." + std::to_string(max_width) + " has the T=4 bits"
                      : "T=" + changed + " differ from T=4")
              << " (" << kDraws << " input draws)\n";
    return failures;
}

// Rank 0 of a row-parallel projection: linear_add over the input-column half, residual included.
Case linear_add_half(std::int32_t k, DeviceWeight& weight, DeviceColumns& x,
                     const std::vector<std::uint16_t>& residual) {
    Case c;
    c.name    = "linear_add NVFP4 [5120," + std::to_string(k) + "]";
    c.outputs = {{"residual", 5120, 2}};
    c.x       = &x;
    c.pass    = [&weight, &x, &residual, k](std::int32_t first, std::int32_t width, Outputs& out) {
        out.buffers[0].copy_from_host(
            residual.data() + (static_cast<std::size_t>(x.draw) * kColumns + first) * 5120,
            static_cast<std::size_t>(width) * 5120 * 2);
        Tensor y = out.tensor(0, DType::BF16);
        const std::size_t capacity =
            ops::linear_add_workspace_capacity_bytes(QType::NVFP4, 5120, k, kPolicy, width, width);
        WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
        ops::linear_add(x.view(first, width), weight.weight, y, kPolicy, workspace, nullptr);
    };
    return c;
}

// Rank 1 of the same projection: linear() over the other half.
Case linear_half(std::int32_t k, DeviceWeight& weight, DeviceColumns& x) {
    Case c;
    c.name    = "linear NVFP4 [5120," + std::to_string(k) + "]";
    c.outputs = {{"y", 5120, 2}};
    c.x       = &x;
    c.pass    = [&weight, &x, k](std::int32_t first, std::int32_t width, Outputs& out) {
        Tensor y = out.tensor(0, DType::BF16);
        const std::size_t capacity =
            ops::linear_workspace_capacity_bytes(QType::NVFP4, 5120, k, kPolicy, width, width);
        WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
        ops::linear(x.view(first, width), weight.weight, y, kPolicy, workspace, nullptr);
    };
    return c;
}

// A Flash-Next dense Q8 projection [n,k] through linear() under A16Only, FP32 output (BF16
// rounding would hide most reduction-order changes on the narrow a|b rows).
Case linear_q8(std::int32_t n, std::int32_t k, DeviceWeight& weight, DeviceColumns& x) {
    Case c;
    c.name      = "linear Q8 [" + std::to_string(n) + "," + std::to_string(k) + "]";
    c.outputs   = {{"y", n, 4}};
    c.x         = &x;
    c.max_width = kQ8MaxWidth;
    c.pass      = [&weight, &x, n, k](std::int32_t first, std::int32_t width, Outputs& out) {
        constexpr ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
        Tensor y                           = out.tensor(0, DType::FP32);
        const std::size_t capacity =
            ops::linear_workspace_capacity_bytes(QType::Q8_G32_FP16, n, k, policy, width, width);
        WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
        ops::linear(x.view(first, width), weight.weight, y, policy, workspace, nullptr);
    };
    return c;
}

// The attention q|k|gate|v projection: one rank's head-local shard [7168,5120].
Case attn_input_shard(DeviceWeight& weight, DeviceColumns& x) {
    Case c;
    c.name    = "attn_input_proj NVFP4 shard [7168,5120]";
    c.outputs = {{"q", 3072, 2}, {"gate", 3072, 2}, {"k", 512, 2}, {"v", 512, 2}};
    c.x       = &x;
    c.pass    = [&weight, &x](std::int32_t first, std::int32_t width, Outputs& out) {
        Tensor q    = out.tensor(0, DType::BF16);
        Tensor gate = out.tensor(1, DType::BF16);
        Tensor k    = out.tensor(2, DType::BF16);
        Tensor v    = out.tensor(3, DType::BF16);
        const std::size_t capacity =
            ops::detail::nvfp4_attn_input_shard_workspace_capacity_bytes(kPolicy, width, width);
        WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
        ops::detail::nvfp4_attn_input_shard_dispatch(x.view(first, width), weight.weight, q, gate,
                                                        k, v, kPolicy, &workspace, nullptr);
    };
    return c;
}

// The GDN q|k|v|z projection of one rank's shard [8192,5120] as the conv snapshot and record forms
// run it (ops/wrapper/gdn_input_proj.cpp, project_conv_shard): quantized exactly where the
// single-parent snapshot form is, through the shard dispatch.
Case gdn_input_conv_shard(DeviceWeight& weight, DeviceColumns& x) {
    Case c;
    c.name    = "gdn_input_proj conv NVFP4 shard [8192,5120]";
    c.outputs = {{"qkv", 5120, 2}, {"z", 3072, 2}};
    c.x       = &x;
    c.pass    = [&weight, &x](std::int32_t first, std::int32_t width, Outputs& out) {
        Tensor qkv                     = out.tensor(0, DType::BF16);
        Tensor z                       = out.tensor(1, DType::BF16);
        const ops::LinearPolicy policy = ops::detail::nvfp4_gdn_conv_uses_a4(kPolicy, width, 1)
                                                ? ops::LinearPolicy::AllowA4
                                                : ops::LinearPolicy::A16Only;
        const std::size_t capacity =
            ops::detail::nvfp4_gdn_input_shard_workspace_capacity_bytes(policy, width, width);
        WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
        ops::detail::nvfp4_gdn_input_shard_dispatch(x.view(first, width), weight.weight, qkv, z,
                                                       policy, &workspace, nullptr);
    };
    return c;
}

// The GDN a|b gating of one rank's 24 value heads (BF16 weights, FP32 g and beta).
struct GatingOperands {
    GatingOperands()
        : a(direct_bf16_weight::make_patterned(24, 5120, 61U)),
          b(direct_bf16_weight::make_patterned(24, 5120, 67U)), a_buffer(a.bits.size() * 2),
          b_buffer(b.bits.size() * 2), log_a_buffer(24 * 4), bias_buffer(24 * 4) {
        a_buffer.copy_from_host(a.bits.data(), a_buffer.bytes());
        b_buffer.copy_from_host(b.bits.data(), b_buffer.bytes());
        std::vector<float> log_a(24), bias(24);
        for (int h = 0; h < 24; ++h) {
            log_a[h] = -1.0F + 0.07F * static_cast<float>(h);
            bias[h]  = 0.3F - 0.025F * static_cast<float>(h);
        }
        log_a_buffer.copy_from_host(log_a.data(), log_a_buffer.bytes());
        bias_buffer.copy_from_host(bias.data(), bias_buffer.bytes());
    }

    direct_bf16_weight::HostWeight a, b;
    GuardedDeviceBuffer a_buffer, b_buffer, log_a_buffer, bias_buffer;
};

Case gdn_gating_shard(GatingOperands& operands, DeviceColumns& x) {
    Case c;
    c.name    = "gdn_gating_proj BF16 shard [24,5120]";
    c.outputs = {{"g", 24, 4}, {"beta", 24, 4}};
    c.x       = &x;
    c.pass    = [&operands, &x](std::int32_t first, std::int32_t width, Outputs& out) {
        Tensor g       = out.tensor(0, DType::FP32);
        Tensor beta    = out.tensor(1, DType::FP32);
        const Weight a = operands.a.device_weight(operands.a_buffer.data());
        const Weight b = operands.b.device_weight(operands.b_buffer.data());
        const Tensor log_a(operands.log_a_buffer.data(), DType::FP32, {24});
        const Tensor bias(operands.bias_buffer.data(), DType::FP32, {24});
        const std::size_t bytes = ops::detail::bf16_gdn_gating_shard_workspace_bytes(width);
        GuardedDeviceBuffer workspace(std::max<std::size_t>(bytes, 256));
        ops::detail::bf16_gdn_gating_dispatch_shard(x.view(first, width), a, b, log_a, bias,
                                                    bytes == 0 ? nullptr : workspace.data(), bytes,
                                                       g, beta, kPolicy, nullptr);
    };
    return c;
}

// The MLP gate|up projection with SwiGLU: one rank's output-row half [17408,5120].
Case swiglu_half(DeviceWeight& weight, DeviceColumns& x) {
    Case c;
    c.name    = "linear_swiglu NVFP4 half [17408,5120]";
    c.outputs = {{"out", 8704, 2}};
    c.x       = &x;
    c.pass    = [&weight, &x](std::int32_t first, std::int32_t width, Outputs& out) {
        Tensor y                   = out.tensor(0, DType::BF16);
        const std::size_t capacity = ops::linear_swiglu_workspace_capacity_bytes(
            QType::NVFP4, 17408, 5120, kPolicy, width, width);
        WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
        ops::linear_swiglu(x.view(first, width), weight.weight, y, kPolicy, workspace, nullptr);
    };
    return c;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    // The attention and GDN output projection [5120,6144] and the MLP down projection [5120,17408]:
    // rank 0's linear_add and rank 1's linear() over their input-column halves.
    for (const std::int32_t k : {3072, 8704}) {
        DeviceWeight weight(random_nvfp4(5120, k, 4099U + static_cast<std::uint32_t>(k)));
        DeviceColumns x(k, 7919U + static_cast<std::uint32_t>(k));
        const std::vector<std::uint16_t> residual =
            random_bf16(static_cast<std::size_t>(5120) * kColumns * kDraws,
                        131U + static_cast<std::uint32_t>(k), 4.0F);
        failures += check(linear_add_half(k, weight, x, residual));
        failures += check(linear_half(k, weight, x));
    }
    {
        DeviceColumns x(5120, 7457U);
        DeviceWeight attn(random_nvfp4(7168, 5120, 5003U));
        failures += check(attn_input_shard(attn, x));
        DeviceWeight gdn(random_nvfp4(8192, 5120, 5011U));
        failures += check(gdn_input_conv_shard(gdn, x));
        GatingOperands gating;
        failures += check(gdn_gating_shard(gating, x));
        DeviceWeight swiglu(random_nvfp4(17408, 5120, 5021U));
        failures += check(swiglu_half(swiglu, x));
    }

    // Flash-Next dense Q8 (qwen3_8_flash_next_nvfp4 recipe): attention q|k|gate|v, GDN q|k|v|z and
    // a|b, the attention and GDN output, the shared expert, the PLE key/value parent and the MTP
    // stem, at tp 1 and as one rank's tp 2 shard.
    struct Q8Shape {
        std::int32_t n;
        std::int32_t k;
    };

    for (const Q8Shape shape :
         {Q8Shape{6656, 2560}, Q8Shape{8192, 2560}, Q8Shape{48, 2560}, Q8Shape{2560, 3072},
          Q8Shape{13312, 2560}, Q8Shape{16384, 2560}, Q8Shape{96, 2560}, Q8Shape{2560, 6144},
          Q8Shape{640, 2560}, Q8Shape{1280, 2560}, Q8Shape{2560, 640}, Q8Shape{12800, 2560},
          Q8Shape{2560, 2560}}) {
        const auto seed = static_cast<std::uint32_t>(shape.n * 31 + shape.k);
        DeviceWeight weight(random_q8(shape.n, shape.k, 6007U + seed));
        DeviceColumns x(shape.k, 7001U + seed);
        failures += check(linear_q8(shape.n, shape.k, weight, x));
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " verify-width columns\n";
    return failures == 0 ? 0 : 1;
}
