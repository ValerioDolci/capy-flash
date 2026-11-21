#include "models/inkling2/state/conv_record.h"

#include "ninfer/ops/short_conv_record.h"

#include <array>
#include <stdexcept>

namespace ninfer::models::inkling2 {
namespace {

constexpr std::size_t kAlign        = 256;
constexpr std::int32_t kHistory     = ops::kShortConvTaps - 1;
constexpr std::size_t kTailsPerSlot = 4;

std::array<Tensor, kTailsPerSlot> tails_of(const VisitState::ConvTails& t) {
    return {t.key, t.value, t.attention, t.ffn};
}

} // namespace

ConvRecord::ConvRecord(const VisitState& state, std::uint32_t slots, std::int32_t max_width)
    : max_width_(max_width) {
    if (slots == 0 || max_width < 1) {
        throw std::invalid_argument("Inkling 2 convolution record needs slots and a width");
    }
    std::size_t bytes = 0;
    for (std::uint32_t slot = 0; slot < slots; ++slot) {
        for (const Tensor& tail : tails_of(state.tails(slot))) {
            const std::int32_t C = tail.ne[0];
            entries_.push_back({bytes, C});
            bytes += (std::size_t(C) * std::size_t(max_width + kHistory) * 2 + kAlign - 1) /
                     kAlign * kAlign;
        }
    }
    storage_ = DeviceBuffer(bytes);
    std::vector<ops::ConvTailFoldEntry> table;
    table.reserve(entries_.size());
    for (std::uint32_t slot = 0; slot < slots; ++slot) {
        const auto tails = tails_of(state.tails(slot));
        for (std::size_t part = 0; part < kTailsPerSlot; ++part) {
            const Entry& e = entries_[slot * kTailsPerSlot + part];
            table.push_back({static_cast<const std::byte*>(storage_.p) + e.offset, tails[part].data,
                             e.channels});
        }
    }
    table_ = DeviceBuffer(table.size() * sizeof(ops::ConvTailFoldEntry));
    table_.copy_from_host(table.data(), table_.bytes);
}

VisitState::ConvTails ConvRecord::records(std::uint32_t slot, std::int32_t width) const {
    if (width < 1 || width > max_width_ || (slot + 1) * kTailsPerSlot > entries_.size()) {
        throw std::invalid_argument("Inkling 2 convolution record: slot or width out of range");
    }
    std::array<Tensor, kTailsPerSlot> out;
    for (std::size_t part = 0; part < kTailsPerSlot; ++part) {
        const Entry& e = entries_[slot * kTailsPerSlot + part];
        out[part]      = Tensor(static_cast<std::byte*>(storage_.p) + e.offset, DType::BF16,
                                {e.channels, width + kHistory});
    }
    return {out[0], out[1], out[2], out[3]};
}

void ConvRecord::fold(const Tensor& committed, std::int32_t width, cudaStream_t stream) const {
    if (width < 1 || width > max_width_) {
        throw std::invalid_argument("Inkling 2 convolution record: fold width out of range");
    }
    ops::conv_tail_fold(static_cast<const ops::ConvTailFoldEntry*>(table_.p),
                        static_cast<std::int32_t>(entries_.size()), width, committed, stream);
}

} // namespace ninfer::models::inkling2
