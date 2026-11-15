#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>
#include <vector>

namespace {
using namespace ninfer::test::linear;

struct Geometry {
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
};

constexpr std::array kGeometries{
    Geometry{1024, 5120, 223U}, Geometry{4608, 4608, 277U}, Geometry{5120, 4608, 281U},
    Geometry{5120, 6144, 239U}, Geometry{5120, 10240, 211U}, Geometry{5120, 17408, 241U},
    Geometry{5120, 25600, 293U}, Geometry{6144, 5120, 227U}, Geometry{14336, 5120, 229U},
    Geometry{34816, 5120, 233U}, Geometry{248320, 5120, 197U},
    // Two-device halves of the MTP projections.
    Geometry{5120, 3072, 307U}, Geometry{5120, 5120, 311U}, Geometry{5120, 8704, 313U},
    Geometry{7168, 5120, 317U}, Geometry{17408, 5120, 331U},
    // Qwen4Exp hyper-connection Down(+inject)/Up and the PLE key/value parent.
    Geometry{324, 10240, 337U}, Geometry{320, 10240, 347U}, Geometry{10240, 320, 349U},
    Geometry{12800, 2560, 353U},
    // Hyper-connection Down(+inject)/Up at H = 5120 and at low rank 640 (H = 2560 and 5120).
    Geometry{324, 20480, 479U}, Geometry{320, 20480, 487U}, Geometry{20480, 320, 491U},
    Geometry{644, 10240, 499U}, Geometry{640, 10240, 503U}, Geometry{10240, 640, 509U},
    Geometry{644, 20480, 521U}, Geometry{640, 20480, 523U}, Geometry{20480, 640, 541U},
    // Qwen4Exp dense projections of the Q8 recipe.
    Geometry{13312, 2560, 359U}, Geometry{640, 2560, 367U}, Geometry{2560, 6144, 373U},
    Geometry{16384, 2560, 379U}, Geometry{96, 2560, 383U}, Geometry{1280, 2560, 389U},
    Geometry{2560, 640, 397U}, Geometry{2560, 2560, 401U},
    // Qwen4Exp Vision merger output [2560, 4608].
    Geometry{2560, 4608, 443U},
    // Qwen3.5 Vision tower stored Q8.
    Geometry{3456, 1152, 449U}, Geometry{1152, 1152, 457U}, Geometry{4304, 1152, 461U},
    Geometry{1152, 4304, 463U}, Geometry{1152, 1536, 467U},
    // Qwen4Exp tp 2 shards of the QSA input and output (sliced-K decode routes).
    Geometry{6656, 2560, 409U}, Geometry{2560, 3072, 419U},
    // ... and of the GDN q|k|v|z and a|b projections (warp-row decode routes).
    Geometry{8192, 2560, 433U}, Geometry{48, 2560, 439U},
    // Two-device vocabulary-row half of a Q8 output head.
    Geometry{124160, 5120, 421U}};

int q8_a16_conformance() {
    int failures = 0;
    for (const auto& shape : kGeometries) {
        std::vector<Invocation> calls;
        // Cover live-column tails and the transitions from K-split to tiled contractions.
        for (int t : {1,  2,  3,  4,  5,  7,  8,  9,  15,  16,  17,  23,  24,  25,
                      31, 32, 33, 39, 40, 41, 44, 47, 48,  49,  55,  56,  57,  63,
                      64, 65, 79, 80, 81, 95, 96, 97, 127, 128, 129, 256, 1024}) {
            calls.push_back({t});
        }
        if (shape.n == 6144 && shape.k == 5120) {
            for (int t : {191, 192, 193}) calls.push_back({t});
        }
        if (shape.k == 4608) {
            for (int t : {6, 11, 12, 13, 14, 19, 20, 21, 27, 28, 29}) calls.push_back({t});
            if (shape.n == 4608) {
                for (int t : {255, 257}) calls.push_back({t});
            }
        }
        if (shape.n == 248320 || shape.n == 124160) calls.push_back({34});
        for (int t : {1, 8, 16, 32, 48, 64, 65, 128}) {
            calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
        }
        for (int t : {1, 16, 64, 128}) calls.push_back({t, CallForm::A16Convenience});
        calls.push_back({17, CallForm::Policy, ninfer::ops::LinearPolicy::AllowA8});
        calls.push_back({64, CallForm::Policy, ninfer::ops::LinearPolicy::AllowA4});
        failures += run_shape("Q8_A16", ActivationCompute::A16, make_q8_g32_fp16_weight,
                              {shape.n, shape.k, shape.seed, Comparison::Sampled, true, calls});
    }
    return failures;
}
} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = q8_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q8_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q8_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
