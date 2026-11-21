// ninfer::ops - DSpark drafter Op wrappers: validate the public contract, then launch.
#include "ninfer/ops/dspark_draft.h"

#include "ops/common/validate.h"
#include "ops/dspark_draft/launch.h"

#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim = 128;

void require(bool condition, const char* op, const std::string& message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

void require_dense(const Tensor& t, DType dtype, const char* op, const char* label) {
    require(t.data != nullptr && t.dtype == dtype && t.is_contiguous() && t.numel() > 0, op,
            std::string(label) + " must be a contiguous non-empty tensor of the expected dtype");
}

void require_scalar(const Tensor& t, const char* op, const char* label) {
    require_dense(t, DType::I32, op, label);
    require(t.numel() == 1, op, std::string(label) + " must be I32 [1]");
}

detail::InvFrequencies inverse_frequencies(float theta) {
    detail::InvFrequencies out{};
    for (int i = 0; i < kHeadDim / 2; ++i) {
        const float power = std::pow(theta, float(2 * i) / float(kHeadDim));
        out.v[i]          = 1.0F / power;
    }
    return out;
}

} // namespace

void feature_mix_rmsnorm(const Tensor& features, const Tensor& logits,
                         const std::array<const Tensor*, 8>& norms, std::int32_t rows, float eps,
                         const Tensor* count, Tensor& out, Tensor& last, cudaStream_t stream) {
    constexpr const char* op = "feature_mix_rmsnorm";
    require(features.data != nullptr && features.dtype == DType::BF16 && features.nb[0] == 2 &&
                features.nb[1] == 2 * std::int64_t(features.ne[0]) &&
                features.nb[2] >= features.nb[1] * features.ne[1],
            op, "features must be BF16 blocks of contiguous [H, W]");
    require_dense(logits, DType::BF16, op, "logits");
    const std::int32_t H = features.ne[0], W = features.ne[1], S = features.ne[2];
    require(S == 4 && features.ne[3] == 1 && H % 8 == 0, op, "features must be BF16 [H, W, 4]");
    require(rows >= 2 && rows <= 8 && logits.ne[0] == S && logits.ne[1] == rows, op,
            "logits must be [rows, 4] with 2 <= rows <= 8");
    detail::MixNorms pointers{};
    for (std::int32_t r = 0; r < rows; ++r) {
        require(norms[r] != nullptr, op, "a norm is missing");
        require_dense(*norms[r], DType::BF16, op, "norm");
        require(norms[r]->numel() == H, op, "norms must be [H]");
        pointers.p[r] = norms[r]->data;
    }
    require_dense(out, DType::BF16, op, "out");
    require(out.ne[0] == H && out.ne[1] == W && out.ne[2] == rows - 1, op,
            "out must be [H, W, rows - 1]");
    require_dense(last, DType::BF16, op, "last");
    require(last.numel() == H, op, "last must be [H]");
    if (count != nullptr) { require_scalar(*count, op, "count"); }
    require(!detail::overlaps(features, out) && !detail::overlaps(features, last) &&
                !detail::overlaps(out, last),
            op, "outputs must not overlap the features or each other");
    detail::feature_mix_rmsnorm_launch(
        features.data, logits.data, pointers, H, W, std::size_t(features.nb[2] / 2), S, rows, eps,
        count != nullptr ? static_cast<const int*>(count->data) : nullptr, out.data, last.data,
        stream);
}

void consecutive_positions(const Tensor& start, const Tensor* minus, Tensor& out,
                           cudaStream_t stream) {
    constexpr const char* op = "consecutive_positions";
    require_scalar(start, op, "start");
    if (minus != nullptr) { require_scalar(*minus, op, "minus"); }
    require_dense(out, DType::I32, op, "out");
    detail::consecutive_positions_launch(
        static_cast<const int*>(start.data),
        minus != nullptr ? static_cast<const int*>(minus->data) : nullptr,
        static_cast<int*>(out.data), static_cast<int>(out.numel()), stream);
}

void head_rmsnorm_rope(Tensor& x, std::int32_t query_heads, std::int32_t key_heads,
                       const Tensor& q_norm, const Tensor& k_norm, const Tensor& positions,
                       float theta, float eps, cudaStream_t stream) {
    constexpr const char* op = "head_rmsnorm_rope";
    require_dense(x, DType::BF16, op, "x");
    require(query_heads >= 0 && key_heads >= 0 && query_heads + key_heads > 0 &&
                x.ne[0] >= (query_heads + key_heads) * kHeadDim && x.ne[2] == 1 && x.ne[3] == 1,
            op, "x must be [rows, T] with rows >= (nq + nk) * 128");
    require_dense(q_norm, DType::BF16, op, "q_norm");
    require_dense(k_norm, DType::BF16, op, "k_norm");
    require(q_norm.numel() == kHeadDim && k_norm.numel() == kHeadDim, op, "norms must be [128]");
    require_dense(positions, DType::I32, op, "positions");
    require(positions.numel() == x.ne[1], op, "positions must be I32 [T]");
    require(std::isfinite(theta) && theta > 0, op, "theta must be positive");
    detail::head_rmsnorm_rope_launch(x.data, x.ne[0], x.ne[1], query_heads, key_heads, q_norm.data,
                                     k_norm.data, static_cast<const int*>(positions.data),
                                     inverse_frequencies(theta), eps, stream);
}

void context_kv_store(const std::array<const Tensor*, 8>& raw,
                      const std::array<const Tensor*, 8>& k_norms,
                      const std::array<DraftKVCacheView, 8>& caches, std::int32_t layers,
                      const Tensor& positions, const Tensor& count, float theta, float eps,
                      cudaStream_t stream) {
    constexpr const char* op = "context_kv_store";
    require(layers >= 1 && layers <= 8, op, "1..8 layers");
    require_dense(positions, DType::I32, op, "positions");
    require_scalar(count, op, "count");
    const std::int32_t W = static_cast<std::int32_t>(positions.numel());
    detail::StoreLayers store{};
    std::int32_t kv_heads = 0;
    for (std::int32_t l = 0; l < layers; ++l) {
        require(raw[l] != nullptr && k_norms[l] != nullptr, op, "a layer operand is missing");
        require_dense(*raw[l], DType::BF16, op, "raw");
        require_dense(*k_norms[l], DType::BF16, op, "k_norm");
        const DraftKVCacheView& cache = caches[l];
        require_dense(cache.k, DType::BF16, op, "cache k");
        require_dense(cache.v, DType::BF16, op, "cache v");
        const std::int32_t heads = cache.k.ne[1];
        require(raw[l]->ne[0] == 2 * heads * kHeadDim && raw[l]->ne[1] == W &&
                    cache.k.ne[0] == kHeadDim && cache.k.ne[2] >= cache.capacity &&
                    cache.capacity > 0 && cache.v.numel() == cache.k.numel() &&
                    k_norms[l]->numel() == kHeadDim && (l == 0 || heads == kv_heads),
                op, "raw must be [2 Hkv 128, W] and caches [128, Hkv, capacity]");
        kv_heads   = heads;
        store.l[l] = {raw[l]->data, k_norms[l]->data, cache.k.data, cache.v.data, cache.capacity};
    }
    require(std::isfinite(theta) && theta > 0, op, "theta must be positive");
    detail::context_kv_store_launch(
        store, layers, kv_heads, W, static_cast<const int*>(positions.data),
        static_cast<const int*>(count.data), inverse_frequencies(theta), eps, stream);
}

std::size_t block_attention_workspace_bytes(std::int32_t query_heads, std::int32_t block) {
    if (query_heads <= 0 || block <= 0 || block > detail::kBlockAttentionMaxBlock) {
        throw std::invalid_argument("block_attention_workspace_bytes: invalid geometry");
    }
    // Partials of the padded query rows of every KV head and chunk: (max, sum, 128 values).
    return std::size_t(detail::kBlockAttentionChunks) * std::size_t(query_heads) *
           detail::kBlockAttentionMaxBlock * (kHeadDim + 2) * sizeof(float);
}

void block_attention(const Tensor& qkv, std::int32_t query_heads, std::int32_t kv_heads,
                     const DraftKVCacheView& cache, const Tensor& frontier, std::int32_t window,
                     float scale, WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "block_attention";
    require_dense(qkv, DType::BF16, op, "qkv");
    const std::int32_t B = qkv.ne[1];
    require(kv_heads > 0 && query_heads % kv_heads == 0 && query_heads / kv_heads <= 4 && B >= 1 &&
                B <= detail::kBlockAttentionMaxBlock &&
                qkv.ne[0] >= (query_heads + 2 * kv_heads) * kHeadDim,
            op, "qkv must be [rows, B <= 8] with Hq / Hkv <= 4");
    require_dense(cache.k, DType::BF16, op, "cache k");
    require_dense(cache.v, DType::BF16, op, "cache v");
    require(cache.k.ne[0] == kHeadDim && cache.k.ne[1] == kv_heads &&
                cache.k.ne[2] >= cache.capacity && cache.capacity > 0 &&
                (window == 0 || cache.capacity >= window),
            op, "cache must be [128, Hkv, capacity] and hold the window");
    require_scalar(frontier, op, "frontier");
    require_dense(out, DType::BF16, op, "out");
    require(out.ne[0] == query_heads * kHeadDim && out.ne[1] == B, op, "out must be [Hq 128, B]");
    require(!detail::overlaps(qkv, out), op, "out must not overlap qkv");
    auto scope    = workspace.scope();
    auto* partial = static_cast<float*>(
        workspace.alloc_bytes(block_attention_workspace_bytes(query_heads, B)).data);
    detail::block_attention_launch(
        qkv.data, qkv.ne[0], query_heads, kv_heads, B, cache.k.data, cache.v.data, cache.capacity,
        static_cast<const int*>(frontier.data), window, scale, partial, out.data, stream);
}

void add_rmsnorm(Tensor& residual, const Tensor& y, const Tensor& weight, float eps, Tensor& out,
                 cudaStream_t stream) {
    constexpr const char* op = "add_rmsnorm";
    for (const Tensor* t : std::initializer_list<const Tensor*>{&residual, &y, &out}) {
        require_dense(*t, DType::BF16, op, "operand");
    }
    require_dense(weight, DType::BF16, op, "weight");
    const std::int32_t H = residual.ne[0];
    require(y.ne[0] == H && out.ne[0] == H && y.numel() == residual.numel() &&
                out.numel() == residual.numel() && weight.numel() == H,
            op, "operands must be [H, T] and weight [H]");
    require(!detail::overlaps(out, residual) && !detail::overlaps(out, y), op,
            "out must not overlap an input");
    detail::add_rmsnorm_launch(residual.data, y.data, weight.data, H,
                               static_cast<int>(residual.numel() / H), eps, out.data, stream);
}

void recurrent_head_context(const Tensor& weight, const Tensor& bias, const Tensor& hidden,
                            const Tensor& context, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "recurrent_head_context";
    require_dense(weight, DType::BF16, op, "weight");
    require_dense(bias, DType::BF16, op, "bias");
    require_dense(hidden, DType::BF16, op, "hidden");
    require_dense(context, DType::BF16, op, "context");
    require_dense(out, DType::FP32, op, "out");
    const std::int32_t H = hidden.ne[0], G = hidden.ne[1];
    const std::int32_t outputs = weight.ne[1], in = weight.ne[0];
    const std::int32_t rank = outputs / 3;
    require(outputs % 3 == 0 && in == 2 * rank + 2 * H && bias.numel() == outputs &&
                context.numel() == H && G >= 1 && G <= detail::kRecurrentMaxSteps &&
                out.ne[0] == outputs && out.ne[1] == G,
            op, "weight [3r, 2r + 2H], bias [3r], hidden [H, G], context [H], out [3r, G]");
    detail::recurrent_head_context_launch(weight.data, in, rank, bias.data, hidden.data, H, G,
                                          context.data, static_cast<float*>(out.data), stream);
}

std::size_t recurrent_head_step_workspace_bytes() {
    return 4 * 256 + 3 * detail::kRecurrentRank * sizeof(float) +
           std::size_t(detail::kRecurrentBlocks) * (sizeof(float) + sizeof(int));
}

void recurrent_head_step(const RecurrentHeadOperands& head, std::int32_t step, const Tensor& anchor,
                         const Tensor& state_in, Tensor& state_out, Tensor& drafts, Tensor& choice,
                         WorkspaceArena& workspace, cudaStream_t stream) {
    constexpr const char* op = "recurrent_head_step";
    constexpr std::int32_t R = detail::kRecurrentRank;
    require_dense(head.weight, DType::BF16, op, "weight");
    require_dense(head.embedding, DType::BF16, op, "embedding");
    require_dense(head.folded, DType::BF16, op, "folded");
    require_dense(head.folded_bias, DType::BF16, op, "folded_bias");
    require_dense(head.base, DType::BF16, op, "base");
    require_dense(head.context, DType::FP32, op, "context");
    require_scalar(anchor, op, "anchor");
    require_dense(state_in, DType::BF16, op, "state_in");
    require_dense(state_out, DType::BF16, op, "state_out");
    require_dense(drafts, DType::I32, op, "drafts");
    require_dense(choice, DType::FP32, op, "choice");
    const std::int32_t rows = head.folded.ne[1], G = head.base.ne[1];
    require(head.weight.ne[1] == 3 * R && head.embedding.ne[0] == R && head.folded.ne[0] == R &&
                head.folded_bias.numel() == rows && head.base.ne[0] == rows &&
                head.context.ne[0] == 3 * R && head.context.ne[1] == G && state_in.numel() == R &&
                state_out.numel() == R && drafts.numel() == G && choice.numel() == 4 && step >= 0 &&
                step < G && (head.ranks == 1 || head.ranks == 2) && head.rank >= 0 &&
                head.rank < head.ranks && head.cap > 0,
            op, "operands outside the r = 256 contract");
    require(!detail::overlaps(state_in, state_out), op, "state_in and state_out must differ");
    auto scope = workspace.scope();
    detail::RecurrentLaunch a{};
    a.raw        = static_cast<float*>(workspace.alloc_bytes(3 * R * sizeof(float)).data);
    a.best_value = static_cast<float*>(
        workspace.alloc_bytes(std::size_t(detail::kRecurrentBlocks) * sizeof(float)).data);
    a.best_id = static_cast<int*>(
        workspace.alloc_bytes(std::size_t(detail::kRecurrentBlocks) * sizeof(int)).data);
    a.done        = static_cast<unsigned*>(workspace.alloc_bytes(sizeof(unsigned)).data);
    a.weight      = head.weight.data;
    a.embedding   = head.embedding.data;
    a.state_in    = state_in.data;
    a.state_out   = state_out.data;
    a.folded      = head.folded.data;
    a.folded_bias = head.folded_bias.data;
    a.base        = head.base.data;
    a.context     = static_cast<const float*>(head.context.data);
    a.anchor      = static_cast<const int*>(anchor.data);
    a.drafts      = static_cast<int*>(drafts.data);
    a.choice      = static_cast<float*>(choice.data);
    a.in          = head.weight.ne[0];
    a.step        = step;
    a.rows        = rows;
    a.first       = head.first;
    a.rank        = head.rank;
    a.ranks       = head.ranks;
    a.cap         = head.cap;
    detail::recurrent_head_step_launch(a, stream);
}

void recurrent_head_select(const Tensor& choice, Tensor& drafts, cudaStream_t stream) {
    constexpr const char* op = "recurrent_head_select";
    require_dense(choice, DType::FP32, op, "choice");
    require_dense(drafts, DType::I32, op, "drafts");
    require(choice.numel() == 4, op, "choice must be FP32 [4]");
    detail::recurrent_head_select_launch(static_cast<const float*>(choice.data),
                                         static_cast<int*>(drafts.data),
                                         static_cast<int>(drafts.numel()) - 1, stream);
}

} // namespace ninfer::ops
