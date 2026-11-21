#pragma once

// The convolution record of a speculative verification call on one rank: for every visit slot of
// a VisitState and each of its four short-convolution tails, the call's input history
// [tail ; x] (ops/short_conv_record.h), BF16 [C, T + 3] for calls of up to `max_width` tokens.
//
// A verification call runs the forward with the record attached (ForwardRank::record): every
// convolution writes its record and leaves its tail untouched. fold() then writes every tail from
// its record after the committed count, read on the device (one conv_tail_fold over all tails),
// so the tails equal those of a forward over the committed tokens alone.

#include "core/arena.h"
#include "core/tensor.h"
#include "models/inkling2/state/visit_state.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace ninfer::models::inkling2 {

class ConvRecord {
public:
    // The record of `state`'s tails, allocated on the current device (the state's).
    ConvRecord(const VisitState& state, std::uint32_t slots, std::int32_t max_width);

    ConvRecord(const ConvRecord&)            = delete;
    ConvRecord& operator=(const ConvRecord&) = delete;

    [[nodiscard]] std::int32_t max_width() const noexcept { return max_width_; }

    // The records of slot `slot`'s four tails for a call of `width` tokens: BF16 [C, width + 3].
    [[nodiscard]] VisitState::ConvTails records(std::uint32_t slot, std::int32_t width) const;

    // Every tail := its record's columns n .. n + 2, n = committed[0] (device I32 [1]) of the
    // recorded call of `width` tokens.
    void fold(const Tensor& committed, std::int32_t width, cudaStream_t stream) const;

private:
    struct Entry {
        std::size_t offset    = 0;
        std::int32_t channels = 0;
    };

    std::int32_t max_width_ = 0;
    std::vector<Entry> entries_; // slot-major, then key, value, attention, ffn
    DeviceBuffer storage_;
    DeviceBuffer table_; // ops::ConvTailFoldEntry per entry
};

} // namespace ninfer::models::inkling2
