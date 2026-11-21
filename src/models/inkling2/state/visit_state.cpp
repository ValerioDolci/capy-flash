#include "models/inkling2/state/visit_state.h"

#include "core/device.h"
#include "ninfer/ops/short_conv.h"

#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace ninfer::models::inkling2 {
namespace {

constexpr std::size_t kAlign = 256;

std::int32_t extent(std::uint64_t value, const char* what) {
    if (value == 0 || value > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(std::string("Inkling 2 visit state: ") + what +
                                    " is out of range");
    }
    return static_cast<std::int32_t>(value);
}

} // namespace

VisitStateLayout plan_visit_state(LayoutBuilder& builder, const TextConfig& config,
                                  std::uint32_t max_context) {
    if (max_context == 0 || config.visits.empty()) {
        throw std::invalid_argument("Inkling 2 visit state needs a schedule and a context");
    }
    const std::size_t slots = config.visits.size();
    VisitStateLayout out;
    out.max_context = max_context;
    out.pages       = (max_context + kVisitStatePageTokens - 1) / kVisitStatePageTokens;
    out.kinds.assign(slots, LayerKind::Local);
    out.index.assign(slots, 0);
    std::vector<bool> seen(slots, false);
    for (const Visit& visit : config.visits) {
        if (visit.slot >= slots || seen[visit.slot]) {
            throw std::invalid_argument("Inkling 2 visit slots are not a permutation");
        }
        seen[visit.slot]      = true;
        out.kinds[visit.slot] = visit.kind;
    }
    std::uint32_t locals = 0, globals = 0;
    for (std::size_t slot = 0; slot < slots; ++slot) {
        out.index[slot] = out.kinds[slot] == LayerKind::Global ? globals++ : locals++;
    }

    const std::int32_t D   = extent(config.head_dim, "head dimension");
    const std::int32_t Hkv = extent(config.num_key_value_heads, "KV heads");
    const std::int32_t C   = extent(config.key_width(), "KV channels");
    const std::int32_t H   = extent(config.hidden_size, "hidden size");
    const std::int32_t P   = extent(out.pages, "page count");
    if (locals > 0) {
        out.ring = plan_cyclic_kv_cache(builder, locals,
                                        config.sliding_window + kVisitStateMaxDrafts, Hkv, D, 1);
    }
    for (std::uint32_t g = 0; g < globals; ++g) {
        const std::string label = "Inkling 2 global visit " + std::to_string(g);
        out.global_k.push_back(builder.add_tensor(DType::BF16, {D, kVisitStatePageTokens, Hkv, P},
                                                  kAlign, label + " K pages"));
        out.global_v.push_back(builder.add_tensor(DType::FP16, {D, kVisitStatePageTokens, Hkv, P},
                                                  kAlign, label + " V pages"));
    }
    out.table = builder.add_tensor(DType::I32, {P, 1}, kAlign, "Inkling 2 block table");
    out.tails.reserve(slots);
    for (std::size_t slot = 0; slot < slots; ++slot) {
        const std::string label = "Inkling 2 visit slot " + std::to_string(slot);
        const std::int32_t taps = ops::kShortConvTaps - 1;
        out.tails.push_back(
            {builder.add_tensor(DType::BF16, {C, taps}, kAlign, label + " K tail"),
             builder.add_tensor(DType::BF16, {C, taps}, kAlign, label + " V tail"),
             builder.add_tensor(DType::BF16, {H, taps}, kAlign, label + " attention tail"),
             builder.add_tensor(DType::BF16, {H, taps}, kAlign, label + " ffn tail")});
    }
    out.row        = builder.add_tensor(DType::I32, {1}, kAlign, "Inkling 2 batch row");
    out.zero_begin = out.tails.front()[0].region.offset;
    out.zero_end   = out.row.region.offset + out.row.region.bytes;
    out.bytes      = builder.finish(kAlign, "Inkling 2 visit state");
    return out;
}

VisitState::VisitState(DeviceSpan backing, const VisitStateLayout& layout)
    : layout_(layout), backing_(backing) {
    if (backing.data == nullptr || backing.bytes < layout.bytes) {
        throw std::invalid_argument("Inkling 2 visit state backing is smaller than its layout");
    }
    if (!layout_.ring.k.empty()) { ring_ = std::make_unique<CyclicKVCache>(backing, layout_.ring); }
    table_ = layout_.table.bind(backing);
    row_   = layout_.row.bind(backing);
    std::vector<std::int32_t> identity(layout_.pages);
    std::iota(identity.begin(), identity.end(), 0);
    CUDA_CHECK(cudaMemcpy(table_.data, identity.data(), identity.size() * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
}

VisitState::~VisitState() = default;

void VisitState::reset(cudaStream_t stream) const {
    CUDA_CHECK(cudaMemsetAsync(static_cast<unsigned char*>(backing_.data) + layout_.zero_begin, 0,
                               layout_.zero_end - layout_.zero_begin, stream));
}

bool VisitState::global(std::uint32_t slot) const {
    return layout_.kinds.at(slot) == LayerKind::Global;
}

VisitState::ConvTails VisitState::tails(std::uint32_t slot) const {
    const auto& t = layout_.tails.at(slot);
    return {t[0].bind(backing_), t[1].bind(backing_), t[2].bind(backing_), t[3].bind(backing_)};
}

CyclicKVCacheLayerView VisitState::ring(std::uint32_t slot) const {
    if (global(slot)) {
        throw std::logic_error("Inkling 2 visit state: a global slot has no ring");
    }
    return ring_->layer_view(layout_.index[slot]);
}

PagedKVBatchLayerView VisitState::pages(std::uint32_t slot) const {
    if (!global(slot)) {
        throw std::logic_error("Inkling 2 visit state: a local slot has no pages");
    }
    const std::uint32_t g = layout_.index[slot];
    PagedKVBatchLayerView view;
    view.k_pages      = layout_.global_k.at(g).bind(backing_);
    view.v_pages      = layout_.global_v.at(g).bind(backing_);
    view.block_tables = table_;
    view.head_dim     = view.k_pages.ne[0];
    view.num_kv_heads = view.k_pages.ne[2];
    view.storage      = KvCacheStorage::BFloat16;
    return view;
}

} // namespace ninfer::models::inkling2
