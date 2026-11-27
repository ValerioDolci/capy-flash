// ninfer::ops - DFlash2 drafter Op wrappers: validate the public contract, then launch.
#include "ninfer/ops/dflash2_draft.h"

#include "ops/common/validate.h"
#include "ops/dflash2_draft/launch.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* op, const std::string& message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

void require_dense(const Tensor& t, DType dtype, const char* op, const char* label) {
    require(t.data != nullptr && t.dtype == dtype && t.is_contiguous() && t.numel() > 0, op,
            std::string(label) + " must be a contiguous non-empty tensor of the expected dtype");
}

void check_conv(const char* op, const Tensor& x, const Tensor& coefficients, const Tensor& base,
                const Tensor& out) {
    for (const Tensor* t : {&x, &coefficients, &base, &out}) {
        require_dense(*t, DType::BF16, op, "operand");
    }
    const std::int32_t H = x.ne[0], W = x.ne[1];
    require(H > 0 && H % 16 == 0 && W >= 1 && x.ne[2] == 1 && x.ne[3] == 1, op,
            "x must be [H, W] with H a multiple of 16");
    require(out.ne[0] == H && out.ne[1] == W && out.numel() == x.numel(), op, "out must be [H, W]");
    require(coefficients.ne[0] == 4 * (H / 16) && coefficients.ne[1] == W &&
                coefficients.numel() == std::int64_t(4) * (H / 16) * W,
            op, "coefficients must be [4 H / 16, W]");
    require(base.ne[0] == H && base.ne[1] == 2 && base.ne[2] == 2 && base.numel() == 4 * H, op,
            "base must be [H, 2, 2]");
    for (const Tensor* t : {&x, &coefficients, &base}) {
        require(!detail::overlaps(*t, out), op, "out must not overlap an input");
    }
}

} // namespace

void dynamic_conv_input(const Tensor& x, const Tensor& coefficients, const Tensor& base,
                        Tensor& out, cudaStream_t stream) {
    check_conv("dynamic_conv_input", x, coefficients, base, out);
    detail::dynamic_conv_launch(x.data, coefficients.data, base.data, out.data, x.ne[0], x.ne[1],
                                false, stream);
}

void dynamic_conv_output_add(const Tensor& z, const Tensor& coefficients, const Tensor& base,
                             Tensor& residual, cudaStream_t stream) {
    check_conv("dynamic_conv_output_add", z, coefficients, base, residual);
    detail::dynamic_conv_launch(z.data, coefficients.data, base.data, residual.data, z.ne[0],
                                z.ne[1], true, stream);
}

std::size_t vocab_top16_workspace_bytes(std::int32_t columns) {
    if (columns < 1 || columns > 16) {
        throw std::invalid_argument("vocab_top16_workspace_bytes: columns must be 1..16");
    }
    return std::size_t(columns) * detail::kVocabTopChunks * detail::kVocabTop *
           sizeof(unsigned long long);
}

void vocab_top16_slots(const Tensor& logits, std::int32_t first, std::int32_t slot, Tensor& pairs,
                       WorkspaceArena& workspace, cudaStream_t stream) {
    constexpr const char* op = "vocab_top16_slots";
    require_dense(logits, DType::BF16, op, "logits");
    require_dense(pairs, DType::FP32, op, "pairs");
    const std::int32_t rows = logits.ne[0], columns = logits.ne[1], slots = pairs.ne[3];
    require(rows >= detail::kVocabTop && columns >= 1 && columns <= 16 && logits.ne[2] == 1 &&
                logits.ne[3] == 1,
            op, "logits must be [V_r >= 16, C <= 16]");
    require(first >= 0 && std::int64_t(first) + rows <= (std::int64_t(1) << 24), op,
            "the global ids must lie below 2^24");
    require(pairs.ne[0] == 2 && pairs.ne[1] == detail::kVocabTop && pairs.ne[2] == columns &&
                (slots == 1 || slots == 2) && slot >= 0 && slot < slots,
            op, "pairs must be [2, 16, C, 1 or 2] with slot inside");
    require(!detail::overlaps(logits, pairs), op, "pairs must not overlap logits");
    auto scope = workspace.scope();
    auto* keys = static_cast<unsigned long long*>(
        workspace.alloc_bytes(vocab_top16_workspace_bytes(columns)).data);
    detail::vocab_top16_slots_launch(logits.data, rows, columns, first, slot, slots, keys,
                                     static_cast<float*>(pairs.data), stream);
}

void vocab_top16_merge(const Tensor& pairs, Tensor& ids, Tensor& values, cudaStream_t stream) {
    constexpr const char* op = "vocab_top16_merge";
    require_dense(pairs, DType::FP32, op, "pairs");
    require_dense(ids, DType::I32, op, "ids");
    require_dense(values, DType::FP32, op, "values");
    const std::int32_t columns = pairs.ne[2], slots = pairs.ne[3];
    require(pairs.ne[0] == 2 && pairs.ne[1] == detail::kVocabTop && columns >= 1 && columns <= 16 &&
                (slots == 1 || slots == 2),
            op, "pairs must be [2, 16, C, 1 or 2]");
    require(ids.ne[0] == detail::kVocabTop && ids.numel() == std::int64_t(16) * columns &&
                values.ne[0] == detail::kVocabTop && values.numel() == ids.numel(),
            op, "ids and values must be [16, C]");
    require(!detail::overlaps(pairs, ids) && !detail::overlaps(pairs, values) &&
                !detail::overlaps(ids, values),
            op, "operands must not overlap");
    detail::vocab_top16_merge_launch(static_cast<const float*>(pairs.data), columns, slots,
                                     static_cast<int*>(ids.data), static_cast<float*>(values.data),
                                     stream);
}

} // namespace ninfer::ops
