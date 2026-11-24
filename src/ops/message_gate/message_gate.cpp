// ninfer::ops - message_gate wrapper: validates the public contract, then launches.
#include "ninfer/ops/message_gate.h"

#include "ops/common/validate.h"
#include "ninfer/ops/rmsnorm_f32.h"
#include "ops/message_gate/launch.h"
#include "ops/message_gate/wide.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::invalid_argument("message_gate: " + message); }
}

bool is_dense(const Tensor& t, DType dtype) {
    return t.dtype == dtype && t.data != nullptr && t.is_contiguous() && t.numel() > 0;
}

bool is_matrix(const Tensor& t, std::int32_t rows, std::int32_t columns) {
    return t.ne[0] == rows && t.ne[1] == columns && t.ne[2] == 1 && t.ne[3] == 1;
}

void check_gate(const Tensor& hidden, const Tensor& message, const Tensor& down,
                const Tensor& forget, const Tensor& gain, float eps, const Tensor& out) {
    require(is_dense(hidden, DType::FP32), "hidden must be contiguous non-empty FP32");
    const std::int32_t width  = hidden.ne[0];
    const std::int32_t tokens = hidden.ne[1];
    require(is_matrix(hidden, width, tokens), "hidden must be [H,T]");
    for (const Tensor* t : {&message, static_cast<const Tensor*>(&out)}) {
        require(is_dense(*t, DType::FP32) && is_matrix(*t, width, tokens),
                "message and out must be FP32 [H,T] like hidden");
    }
    require(is_dense(down, DType::BF16), "down must be contiguous non-empty BF16");
    const std::int32_t rank = down.ne[1];
    require(down.ne[0] == 2 * width && is_matrix(down, 2 * width, rank) && rank >= 1 &&
                rank <= kMessageGateMaxRank,
            "down must be ne={2H,R} with 1 <= R <= kMessageGateMaxRank");
    for (const Tensor* t : {&forget, &gain}) {
        require(is_dense(*t, DType::FP32) && is_matrix(*t, rank, width),
                "forget and gain must be FP32 ne={R,H}");
    }
    require(std::isfinite(eps) && eps > 0.0F, "eps must be finite and positive");
    for (const Tensor* input : {&hidden, &message}) {
        require(input->data == out.data || !detail::overlaps(*input, out),
                "out must be disjoint from hidden and message or exactly one of them");
    }
    for (const Tensor* input : {&down, &forget, &gain}) {
        require(!detail::overlaps(*input, out), "out must not overlap a weight");
    }
}

} // namespace

void message_gate(const Tensor& hidden, const Tensor& message, const Tensor& down,
                  const Tensor& forget, const Tensor& gain, float eps, Tensor& out,
                  cudaStream_t stream) {
    check_gate(hidden, message, down, forget, gain, eps, out);
    if (hidden.ne[1] >= detail::kMessageGateWideMinTokens &&
        detail::message_gate_wide(hidden, message, down, forget, gain, eps, out, stream)) {
        return;
    }
    detail::message_gate_launch(hidden, message, down, forget, gain, eps, out, stream);
}

void message_gates_rmsnorm(std::span<const MessageGateNormed> gates, float eps, float norm_eps,
                           cudaStream_t stream) {
    require(!gates.empty() && gates.size() <= static_cast<std::size_t>(kMessageGateMaxBatch),
            "1 to kMessageGateMaxBatch gates");
    require(std::isfinite(norm_eps) && norm_eps > 0.0F, "norm eps must be finite and positive");
    const Tensor& first       = gates.front().hidden;
    const DType first_weights = gates.front().forget.dtype;
    bool aligned              = true;
    detail::MessageGateBatch batch;
    for (std::size_t g = 0; g < gates.size(); ++g) {
        const MessageGateNormed& x = gates[g];
        require(is_dense(x.hidden, DType::FP32) && is_matrix(x.hidden, first.ne[0], first.ne[1]) &&
                    x.down.ne[1] == gates.front().down.ne[1],
                "every gate must have the same H, R and T");
        require(is_dense(x.norm, DType::BF16) && x.norm.numel() == first.ne[0],
                "norm must be contiguous BF16 with H elements");
        require(is_dense(x.normed, DType::BF16) && is_matrix(x.normed, first.ne[0], first.ne[1]),
                "normed must be contiguous BF16 [H,T]");
        for (const MessageGateNormed& y : gates) {
            for (const Tensor* t :
                 {&y.hidden, &y.message, &y.down, &y.forget, &y.gain, &y.out, &y.norm}) {
                require(!detail::overlaps(x.normed, *t), "normed must not overlap an operand");
                if (&y != &x) {
                    require(!detail::overlaps(x.out, *t),
                            "a gate's out must not overlap another gate's operands");
                }
            }
        }
        for (const Tensor* t : {&x.hidden, &x.message, &x.down, &x.forget, &x.gain, &x.out}) {
            aligned = aligned && (reinterpret_cast<std::uintptr_t>(t->data) & 15U) == 0;
        }
        require(x.forget.dtype == first_weights && x.gain.dtype == first_weights,
                "forget and gain must have one dtype in a batch");
        batch.gates[g] = {static_cast<const float*>(x.hidden.data),
                          static_cast<const float*>(x.message.data),
                          static_cast<const __nv_bfloat16*>(x.down.data),
                          static_cast<const float*>(x.forget.data),
                          static_cast<const float*>(x.gain.data),
                          static_cast<float*>(x.out.data),
                          static_cast<const __nv_bfloat16*>(x.norm.data),
                          static_cast<__nv_bfloat16*>(x.normed.data)};
    }
    // Every gate's own contract (shapes, eps, in place) as message_gate checks it, without
    // launching.
    const auto& g0 = gates.front();
    if (first_weights == DType::BF16) {
        // BF16 forget/gain (the checkpoint's values): the batch route only.
        require(aligned, "BF16 forget/gain need 16-byte aligned operands");
        for (const MessageGateNormed& x : gates) {
            const std::int32_t rank = x.down.ne[1], width = x.hidden.ne[0];
            for (const Tensor* t : {&x.forget, &x.gain}) {
                require(is_dense(*t, DType::BF16) && is_matrix(*t, rank, width) &&
                            !detail::overlaps(*t, x.out),
                        "BF16 forget and gain must be ne={R,H} and not overlap out");
            }
            // The rest of the gate's contract, with FP32 stand-ins of the right shape for the
            // weights (their storage is the down matrix's: never written, never overlapping out).
            Tensor out = x.out;
            const Tensor standin(x.down.data, DType::FP32, {rank, width});
            require(static_cast<std::size_t>(x.down.bytes()) >= standin.bytes(),
                    "down smaller than a forget stand-in");
            check_gate(x.hidden, x.message, x.down, standin, standin, eps, out);
        }
        require(detail::message_gate_batch_launch(batch, static_cast<int>(gates.size()), g0.hidden,
                                                  g0.message, g0.down, g0.forget, g0.gain, g0.out,
                                                  eps, norm_eps, stream),
                "BF16 forget/gain outside the batch route");
        return;
    }
    if (aligned) {
        for (const MessageGateNormed& x : gates) {
            Tensor out = x.out;
            check_gate(x.hidden, x.message, x.down, x.forget, x.gain, eps, out);
        }
        if (detail::message_gate_batch_launch(batch, static_cast<int>(gates.size()), g0.hidden,
                                              g0.message, g0.down, g0.forget, g0.gain, g0.out, eps,
                                              norm_eps, stream)) {
            return;
        }
    }
    for (const MessageGateNormed& x : gates) {
        Tensor out = x.out, normed = x.normed;
        message_gate(x.hidden, x.message, x.down, x.forget, x.gain, eps, out, stream);
        rmsnorm_f32(out, x.norm, norm_eps, normed, stream);
    }
}

} // namespace ninfer::ops
