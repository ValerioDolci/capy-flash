#include "models/inkling2/execution/parameters.h"

#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

namespace ninfer::models::inkling2::execution {
namespace {

std::int32_t extent(std::uint64_t value) {
    if (value > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("model dimension exceeds the Tensor integer domain");
    }
    return static_cast<std::int32_t>(value);
}

class Prepare {
public:
    Prepare(const Model& model, int device) : model_(model), device_(device) {}

    TextParameters text() const {
        const auto& w = model_.weights().text;
        TextParameters out;
        out.token_embedding = tensor(w.token_embedding);
        out.embedding_norm  = tensor(w.embedding_norm);
        out.layers.reserve(w.layers.size());
        for (const auto& layer : w.layers) { out.layers.push_back(this->layer(layer)); }
        out.final_queries = tensor(w.final_queries);
        out.final_norm    = tensor(w.final_norm);
        out.output_head   = ops::prepare_linear_weight(model_.input(w.output_head_use, device_));
        return out;
    }

private:
    // A contiguous BF16/FP32 parameter as a Tensor with ne = the logical shape reversed.
    Tensor tensor(WeightId id) const {
        const auto& bound = model_.weight(id, device_);
        const auto& shape = bound.view.shape;
        if (shape.empty() || shape.size() > 4) {
            throw std::invalid_argument(bound.name + ": direct parameter rank is not 1-4");
        }
        std::array<std::int32_t, 4> ne{1, 1, 1, 1};
        for (std::size_t i = 0; i < shape.size(); ++i) {
            ne[i] = extent(shape[shape.size() - 1 - i]);
        }
        return weight_tensor(bound.view, {ne[0], ne[1], ne[2], ne[3]});
    }

    LinearParameters linear(WeightId id) const {
        return ops::prepare_linear_weight(model_.input(id, device_));
    }

    // Consecutive rows of one parent as one Linear operand (throws for separate parents).
    static LinearParameters stacked(std::initializer_list<ops::WeightInput> rows) {
        return ops::prepare_linear_weight(
            std::span<const ops::WeightInput>(rows.begin(), rows.size()));
    }

    AttentionParameters attention(const AttentionWeights& w) const {
        AttentionParameters out;
        out.query            = linear(w.query);
        out.gate             = linear(w.gate);
        out.relative         = linear(w.relative);
        out.sink             = linear(w.sink);
        out.key              = linear(w.key);
        out.value            = linear(w.value);
        out.output           = linear(w.output);
        const auto in        = [&](WeightId id) { return model_.input(id, device_); };
        out.queries          = stacked({in(w.query), in(w.gate), in(w.relative), in(w.sink)});
        out.query_scale      = tensor(w.query_scale);
        out.relative_scale   = tensor(w.relative_scale);
        out.relative_profile = tensor(w.relative_profile);
        if (const auto* conv = std::get_if<StaticKvConvWeights>(&w.kv_conv)) {
            out.kv_conv = StaticKvConvParameters{tensor(conv->key), tensor(conv->value)};
            out.all     = stacked(
                {in(w.query), in(w.gate), in(w.relative), in(w.sink), in(w.key), in(w.value)});
        } else {
            const auto& dynamic = std::get<DynamicKvConvWeights>(w.kv_conv);
            out.kv_conv         = DynamicKvConvParameters{
                tensor(dynamic.key_static), tensor(dynamic.value_static),
                linear(dynamic.key_projection), linear(dynamic.value_projection)};
            out.keys   = stacked({in(w.key), in(dynamic.key_projection)});
            out.values = stacked({in(w.value), in(dynamic.value_projection)});
        }
        return out;
    }

    // The whole parent one bank's expert matrices are consecutive row blocks of.
    static const WeightParent& bank_parent(const BoundWeight& first) {
        if (first.view.parts.size() != 1 || first.view.parts.front().begin != 0) {
            throw std::invalid_argument(first.name + ": an expert bank starts at its first row");
        }
        return *first.view.parts.front().parent;
    }

    static WeightView whole_view(const WeightParent& parent) {
        return WeightView{parent.geometry.shape, {{&parent, 0, parent.geometry.elements}}};
    }

    ExpertBankParameters bank(const MoeWeights& w, std::uint32_t first, std::uint32_t count) const {
        const auto& gate_up = bank_parent(model_.weight(w.experts.at(first).gate_up, device_));
        const auto& down    = bank_parent(model_.weight(w.experts.at(first).down, device_));
        for (std::uint32_t i = 0; i < count; ++i) {
            for (const auto& [id, parent] : {std::pair{w.experts[first + i].gate_up, &gate_up},
                                             std::pair{w.experts[first + i].down, &down}}) {
                const auto& bound = model_.weight(id, device_);
                const auto rows   = weight_element_count(bound.view.shape);
                if (bound.view.parts.size() != 1 || bound.view.parts[0].parent != parent ||
                    bound.view.parts[0].begin != i * rows ||
                    parent->geometry.elements != count * rows) {
                    throw std::invalid_argument(bound.name +
                                                ": experts are not the consecutive row blocks "
                                                "of one bank parent");
                }
            }
        }
        const auto format = gate_up.geometry.format;
        const bool nvfp4 =
            format == QType::NVFP4 && gate_up.geometry.layout == QuantLayout::BlockScaleK16M128x4;
        const bool mxfp8 = format == QType::MXFP8_E4M3FN_G32_UE8M0 &&
                           gate_up.geometry.layout == QuantLayout::BlockScaleK32;
        const bool bf16 =
            format == QType::BF16 && gate_up.geometry.layout == QuantLayout::Contiguous;
        if ((!nvfp4 && !mxfp8 && !bf16) || down.geometry.format != format ||
            down.geometry.layout != gate_up.geometry.layout) {
            throw std::invalid_argument("expert banks are block-scaled NVFP4, MXFP8 or BF16, one "
                                        "format for both matrices");
        }
        return {whole_view(gate_up), whole_view(down), static_cast<std::int32_t>(first),
                static_cast<std::int32_t>(count)};
    }

    MoeParameters moe(const MoeWeights& w) const {
        MoeParameters out;
        out.router      = tensor(w.router);
        out.router_bias = tensor(w.router_bias);
        // Consecutive resident experts that share one gate_up parent form a bank.
        const auto experts  = static_cast<std::uint32_t>(w.experts.size());
        std::uint32_t first = 0;
        while (first < experts) {
            if (!model_.resident(w.experts[first].gate_up, device_)) {
                ++first;
                continue;
            }
            const auto* parent =
                model_.weight(w.experts[first].gate_up, device_).view.parts.front().parent;
            std::uint32_t last = first + 1;
            while (last < experts && model_.resident(w.experts[last].gate_up, device_) &&
                   model_.weight(w.experts[last].gate_up, device_).view.parts.front().parent ==
                       parent) {
                ++last;
            }
            out.banks.push_back(bank(w, first, last - first));
            first = last;
        }
        if (out.banks.empty()) {
            throw std::invalid_argument("a rank holds no routed-expert bank");
        }
        return out;
    }

    CoreParameters core(const CoreWeights& w) const {
        CoreParameters out;
        out.message_queries = tensor(w.message_queries);
        for (std::size_t i = 0; i < w.gates.size(); ++i) {
            out.gates[i] = {tensor(w.gates[i].down), tensor(w.gates[i].forget),
                            tensor(w.gates[i].gain)};
        }
        return out;
    }

    LayerParameters layer(const LayerWeights& w) const {
        LayerParameters out;
        if (w.attention_queries) { out.attention_queries = tensor(*w.attention_queries); }
        for (const auto id : w.input_norms) { out.input_norms.push_back(tensor(id)); }
        out.attention           = attention(w.attention);
        out.post_attention_conv = tensor(w.post_attention_conv);
        out.ffn_queries         = tensor(w.ffn_queries);
        out.ffn_norm            = tensor(w.ffn_norm);
        out.moe                 = moe(w.moe);
        out.post_ffn_conv       = tensor(w.post_ffn_conv);
        if (w.core) { out.core = core(*w.core); }
        return out;
    }

    const Model& model_;
    int device_;
};

} // namespace

Parameters::Parameters(const Model& source, int device)
    : model(source), device(device), text(Prepare(source, device).text()) {}

} // namespace ninfer::models::inkling2::execution
