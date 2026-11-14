#pragma once

// Host side of the exclusive expert tier: which pool slot of its
// layer holds an expert's host copy, or none while the expert lives only in a device slot.
//
// Pure host code without CUDA. Slot release is two-phase: release() parks a slot until the window
// counter passes the last launched window, because a launched window may still read it. The caller
// reports window ends with retire().

#include <cstddef>
#include <cstdint>
#include <deque>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::expert_cache {

class ExpertHostPool {
public:
    static constexpr std::int32_t kNoSlot = -1;

    // slots[l] = host slots of layer l; every expert starts without one (assign them with
    // place()). A layer with slots[l] == experts can hold every expert (an inclusive layer).
    ExpertHostPool(std::int32_t layers, std::int32_t experts, std::span<const std::int32_t> slots)
        : layers_(layers), experts_(experts),
          slot_of_(static_cast<std::size_t>(layers) * static_cast<std::size_t>(experts), kNoSlot),
          free_(static_cast<std::size_t>(layers)), parked_(static_cast<std::size_t>(layers)) {
        if (layers <= 0 || experts <= 0 || slots.size() != static_cast<std::size_t>(layers)) {
            throw std::invalid_argument("ExpertHostPool: bad geometry");
        }
        for (std::int32_t l = 0; l < layers; ++l) {
            const std::int32_t n = slots[static_cast<std::size_t>(l)];
            if (n < 0 || n > experts) { throw std::invalid_argument("ExpertHostPool: bad slots"); }
            auto& list = free_[static_cast<std::size_t>(l)];
            list.reserve(static_cast<std::size_t>(n));
            for (std::int32_t s = n - 1; s >= 0; --s) { list.push_back(s); }
        }
    }

    [[nodiscard]] std::int32_t slot(std::int32_t layer, std::int32_t expert) const {
        return slot_of_[cell(layer, expert)];
    }

    // slot(layer, e) of every expert of a layer: host readers map an id to its pool slot with it.
    // Stable storage; an entry a window reads does not change while that window runs.
    [[nodiscard]] const std::int32_t* layer_slots(std::int32_t layer) const {
        return slot_of_.data() + cell(layer, 0);
    }

    [[nodiscard]] std::size_t free_slots(std::int32_t layer) const {
        return free_[static_cast<std::size_t>(layer)].size();
    }

    [[nodiscard]] std::size_t parked_slots(std::int32_t layer) const {
        return parked_[static_cast<std::size_t>(layer)].size();
    }

    // Takes a free slot for `expert`; kNoSlot when the layer has none (the swap must wait). The
    // slot is the expert's at once for accounting; its bytes are valid when the caller's copy
    // completes, and only then may readers be pointed at it.
    [[nodiscard]] std::int32_t reserve(std::int32_t layer, std::int32_t expert) {
        auto& entry = slot_of_[cell(layer, expert)];
        if (entry != kNoSlot) {
            throw std::logic_error("ExpertHostPool: expert already has a slot");
        }
        auto& list = free_[static_cast<std::size_t>(layer)];
        if (list.empty()) { return kNoSlot; }
        entry = list.back();
        list.pop_back();
        return entry;
    }

    // Gives up `expert`'s slot. Windows launched so far may read it: it is reusable once the window
    // counter passed `last_launched` (retire()).
    void release(std::int32_t layer, std::int32_t expert, std::uint64_t last_launched) {
        auto& entry = slot_of_[cell(layer, expert)];
        if (entry == kNoSlot) { throw std::logic_error("ExpertHostPool: expert has no slot"); }
        parked_[static_cast<std::size_t>(layer)].push_back({entry, last_launched});
        entry = kNoSlot;
    }

    // Abandons a reservation no reader was pointed at (a dropped swap): free at once.
    void cancel(std::int32_t layer, std::int32_t expert) {
        auto& entry = slot_of_[cell(layer, expert)];
        if (entry == kNoSlot) { throw std::logic_error("ExpertHostPool: expert has no slot"); }
        free_[static_cast<std::size_t>(layer)].push_back(entry);
        entry = kNoSlot;
    }

    // Hands `from`'s slot to `to` at once (both of the same layer; the caller copies the bytes and
    // no window reads `from`'s host copy any more).
    void transfer(std::int32_t layer, std::int32_t from, std::int32_t to) {
        auto& source = slot_of_[cell(layer, from)];
        auto& target = slot_of_[cell(layer, to)];
        if (source == kNoSlot || target != kNoSlot) {
            throw std::logic_error("ExpertHostPool: bad transfer");
        }
        target = source;
        source = kNoSlot;
    }

    // Every window up to and including `finished` has ended on every device.
    // Returns the number of slots freed.
    std::size_t retire(std::uint64_t finished) {
        std::size_t freed = 0;
        for (std::size_t l = 0; l < parked_.size(); ++l) {
            auto& queue = parked_[l];
            while (!queue.empty() && queue.front().window <= finished) {
                free_[l].push_back(queue.front().slot);
                queue.pop_front();
                ++freed;
            }
        }
        return freed;
    }

private:
    struct Parked {
        std::int32_t slot    = 0;
        std::uint64_t window = 0;
    };

    [[nodiscard]] std::size_t cell(std::int32_t layer, std::int32_t expert) const {
        if (layer < 0 || layer >= layers_ || expert < 0 || expert >= experts_) {
            throw std::out_of_range("ExpertHostPool: cell");
        }
        return static_cast<std::size_t>(layer) * static_cast<std::size_t>(experts_) +
               static_cast<std::size_t>(expert);
    }

    std::int32_t layers_  = 0;
    std::int32_t experts_ = 0;
    std::vector<std::int32_t> slot_of_;
    std::vector<std::vector<std::int32_t>> free_;
    std::vector<std::deque<Parked>> parked_;
};

} // namespace ninfer::expert_cache
