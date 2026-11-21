// logprob_split_pack + argmax_split_logprob (ninfer/ops/tp2/argmax.h): the log-probability of the
// vocabulary-split argmax over two blocks equals the log-probability of the argmax over the
// complete vocabulary. Both ranks are simulated on one device and the summing exchange is the
// host's sum of their packed digits (exact: the digits are disjoint integers). Checked against an
// FP64 evaluation over the complete BF16 logits, for random columns, a column whose maximum is in
// rank 1's block, and blocks of very different log-normalizers.
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/target_logprobs.h"
#include "ninfer/ops/tp2/argmax.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

float bf16_round(float x) {
    std::uint32_t u = 0;
    std::memcpy(&u, &x, 4);
    u += 0x7fffU + ((u >> 16U) & 1U);
    u &= 0xffff0000U;
    float y = 0;
    std::memcpy(&y, &u, 4);
    return y;
}

std::vector<float> sum_digits(const std::vector<std::uint16_t>& a,
                              const std::vector<std::uint16_t>& b) {
    std::vector<float> out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        float x = 0, y = 0;
        const std::uint32_t ua = static_cast<std::uint32_t>(a[i]) << 16U;
        const std::uint32_t ub = static_cast<std::uint32_t>(b[i]) << 16U;
        std::memcpy(&x, &ua, 4);
        std::memcpy(&y, &ub, 4);
        out[i] = x + y;
    }
    return out;
}

int run_case(const std::string& label, int rows, int columns, float scale1, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> n(0.0F, 3.0F);
    // Complete logits [2R,T] column-major per column t: rows [0,R) rank 0, [R,2R) rank 1.
    std::vector<float> full(static_cast<std::size_t>(2 * rows * columns));
    for (int t = 0; t < columns; ++t) {
        for (int r = 0; r < 2 * rows; ++r) {
            const float v = n(rng) * (r < rows ? 1.0F : scale1);
            full[static_cast<std::size_t>(t) * 2 * rows + r] = bf16_round(v);
        }
    }
    if (columns > 1) { full[static_cast<std::size_t>(1) * 2 * rows + rows + 3] = 40.0F; }
    std::vector<float> half[2];
    for (int k = 0; k < 2; ++k) {
        half[k].resize(static_cast<std::size_t>(rows * columns));
        for (int t = 0; t < columns; ++t) {
            for (int r = 0; r < rows; ++r) {
                half[k][static_cast<std::size_t>(t) * rows + r] =
                    full[static_cast<std::size_t>(t) * 2 * rows + k * rows + r];
            }
        }
    }
    std::vector<std::uint16_t> cand[2], pack[2];
    for (int k = 0; k < 2; ++k) {
        DeviceBuffer logits = to_device_bf16(half[k]);
        DeviceBuffer local(static_cast<std::size_t>(columns) * 4);
        DeviceBuffer lp(static_cast<std::size_t>(columns) * 4);
        DeviceBuffer c(static_cast<std::size_t>(8 * columns) * 2);
        DeviceBuffer p(static_cast<std::size_t>(8 * columns) * 2);
        Tensor lt(logits.p, DType::BF16, {rows, columns});
        Tensor loc(local.p, DType::I32, {columns});
        Tensor lpt(lp.p, DType::FP32, {columns});
        Tensor ct(c.p, DType::BF16, {8, columns});
        Tensor pt(p.p, DType::BF16, {8, columns});
        ops::argmax(lt, loc, rows, nullptr);
        ops::argmax_split_pack(lt, loc, k, ct, nullptr);
        ops::target_logprobs(lt, loc, rows, lpt, nullptr);
        ops::logprob_split_pack(lpt, k, pt, nullptr);
        cand[k] = from_device<std::uint16_t>(c, static_cast<std::size_t>(8 * columns));
        pack[k] = from_device<std::uint16_t>(p, static_cast<std::size_t>(8 * columns));
    }
    DeviceBuffer cs = to_device_bf16(sum_digits(cand[0], cand[1]));
    DeviceBuffer ps = to_device_bf16(sum_digits(pack[0], pack[1]));
    DeviceBuffer out(static_cast<std::size_t>(columns) * 4);
    Tensor cst(cs.p, DType::BF16, {8, columns});
    Tensor pst(ps.p, DType::BF16, {8, columns});
    Tensor ot(out.p, DType::FP32, {columns});
    ops::argmax_split_logprob(cst, pst, ot, nullptr);
    const std::vector<float> got = from_device<float>(out, static_cast<std::size_t>(columns));
    int failures = 0;
    for (int t = 0; t < columns; ++t) {
        const float* col = &full[static_cast<std::size_t>(t) * 2 * rows];
        double m = col[0];
        for (int r = 1; r < 2 * rows; ++r) { m = std::max(m, static_cast<double>(col[r])); }
        double s = 0;
        for (int r = 0; r < 2 * rows; ++r) { s += std::exp(static_cast<double>(col[r]) - m); }
        const double want = -std::log(s);
        if (std::fabs(got[static_cast<std::size_t>(t)] - want) > 1e-4 + 1e-5 * std::fabs(want)) {
            std::cerr << "logprob_split " << label << " column " << t << ": got "
                      << got[static_cast<std::size_t>(t)] << " want " << want << '\n';
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "logprob_split: SKIP (CUDA unavailable)\n";
        return 77;
    }
    int failures = 0;
    failures += run_case("small", 64, 5, 1.0F, 1);
    failures += run_case("shortlist half", 8192, 7, 1.0F, 2);
    failures += run_case("sharp rank 1", 4096, 4, 4.0F, 3);
    failures += run_case("flat rank 1", 4096, 4, 0.05F, 4);
    if (failures != 0) {
        std::cerr << "logprob_split failures=" << failures << '\n';
        return 1;
    }
    std::cout << "logprob_split: PASS\n";
    return 0;
}
