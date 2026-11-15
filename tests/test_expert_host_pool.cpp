#include "core/expert_host_pool.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

namespace {

using ninfer::expert_cache::ExpertHostPool;

int failures = 0;

void check(bool ok, const char* what, int line) {
    if (!ok) {
        std::fprintf(stderr, "FAIL line %d: %s\n", line, what);
        ++failures;
    }
}

#define EXPECT_EQ(a, b) check((a) == (b), #a " == " #b, __LINE__)
#define EXPECT_NE(a, b) check((a) != (b), #a " != " #b, __LINE__)
#define ASSERT_EQ(a, b) EXPECT_EQ(a, b)
#define ASSERT_NE(a, b) EXPECT_NE(a, b)
#define EXPECT_THROW(expr, type)                                                                   \
    do {                                                                                           \
        bool thrown = false;                                                                       \
        try {                                                                                      \
            expr;                                                                                  \
        } catch (const type&) { thrown = true; }                                                   \
        check(thrown, #expr " throws", __LINE__);                                                  \
    } while (false)

void ReservesUntilTheLayerIsFull() {
    const std::array<std::int32_t, 2> slots{2, 1};
    ExpertHostPool pool(2, 8, slots);
    const std::int32_t a = pool.reserve(0, 1);
    const std::int32_t b = pool.reserve(0, 2);
    EXPECT_NE(a, ExpertHostPool::kNoSlot);
    EXPECT_NE(b, ExpertHostPool::kNoSlot);
    EXPECT_NE(a, b);
    EXPECT_EQ(pool.reserve(0, 3), ExpertHostPool::kNoSlot);
    EXPECT_EQ(pool.slot(0, 3), ExpertHostPool::kNoSlot);
    EXPECT_NE(pool.reserve(1, 3), ExpertHostPool::kNoSlot); // the other layer has its own slots
}

void ReleasedSlotWaitsForTheLaunchedWindows() {
    const std::array<std::int32_t, 1> slots{1};
    ExpertHostPool pool(1, 8, slots);
    const std::int32_t slot = pool.reserve(0, 1);
    pool.release(0, 1, 5);
    EXPECT_EQ(pool.slot(0, 1), ExpertHostPool::kNoSlot);
    EXPECT_EQ(pool.parked_slots(0), 1U);
    EXPECT_EQ(pool.reserve(0, 2), ExpertHostPool::kNoSlot);
    pool.retire(4);
    EXPECT_EQ(pool.reserve(0, 2), ExpertHostPool::kNoSlot);
    pool.retire(5);
    EXPECT_EQ(pool.parked_slots(0), 0U);
    EXPECT_EQ(pool.reserve(0, 2), slot);
}

void RejectsDoubleOwnership() {
    const std::array<std::int32_t, 1> slots{2};
    ExpertHostPool pool(1, 8, slots);
    EXPECT_NE(pool.reserve(0, 1), ExpertHostPool::kNoSlot);
    EXPECT_THROW((void)pool.reserve(0, 1), std::logic_error);
    EXPECT_THROW(pool.release(0, 2, 0), std::logic_error);
}

void TransferHandsTheSlotOverAtOnce() {
    const std::array<std::int32_t, 1> slots{2};
    ExpertHostPool pool(1, 8, slots);
    const std::int32_t slot = pool.reserve(0, 1);
    pool.transfer(0, 1, 2);
    EXPECT_EQ(pool.slot(0, 2), slot);
    EXPECT_EQ(pool.slot(0, 1), ExpertHostPool::kNoSlot);
    EXPECT_EQ(pool.parked_slots(0), 0U); // not parked: no window waits on it
    EXPECT_EQ(pool.free_slots(0), 1U);
    EXPECT_THROW(pool.transfer(0, 1, 3), std::logic_error); // the source has no slot
    EXPECT_NE(pool.reserve(0, 3), ExpertHostPool::kNoSlot);
    EXPECT_THROW(pool.transfer(0, 2, 3), std::logic_error); // the target has one
}

void LayerSlotsFollowsEveryChange() {
    const std::array<std::int32_t, 2> slots{2, 2};
    ExpertHostPool pool(2, 4, slots);
    const std::int32_t* row = pool.layer_slots(1);
    const std::int32_t slot = pool.reserve(1, 3);
    EXPECT_EQ(row[3], slot);
    pool.transfer(1, 3, 0);
    EXPECT_EQ(row[0], slot);
    EXPECT_EQ(row[3], ExpertHostPool::kNoSlot);
    pool.release(1, 0, 0);
    for (std::int32_t e = 0; e < 4; ++e) {
        EXPECT_EQ(row[e], pool.slot(1, e));
        EXPECT_EQ(pool.layer_slots(0)[e], ExpertHostPool::kNoSlot);
    }
}

void RejectsBadGeometry() {
    const std::array<std::int32_t, 1> too_many{9};
    EXPECT_THROW(ExpertHostPool(1, 8, too_many), std::invalid_argument);
    const std::array<std::int32_t, 2> wrong_count{1, 1};
    EXPECT_THROW(ExpertHostPool(1, 8, wrong_count), std::invalid_argument);
    EXPECT_THROW(ExpertHostPool(-1, 8, {}), std::invalid_argument); // checked before any sizing
    const std::array<std::int32_t, 1> one{1};
    EXPECT_THROW(ExpertHostPool(1, -8, one), std::invalid_argument);
}

// No two experts ever share a slot, and no slot is lost, over random swaps.
void RandomSwapsKeepEverySlotOwnedOnce() {
    constexpr std::int32_t kExperts = 64;
    constexpr std::int32_t kSlots   = 48;
    const std::array<std::int32_t, 1> slots{kSlots};
    ExpertHostPool pool(1, kExperts, slots);
    std::vector<std::int32_t> with_slot;
    std::vector<std::int32_t> without;
    for (std::int32_t e = 0; e < kExperts; ++e) {
        if (pool.reserve(0, e) != ExpertHostPool::kNoSlot) {
            with_slot.push_back(e);
        } else {
            without.push_back(e);
        }
    }
    ASSERT_EQ(with_slot.size(), static_cast<std::size_t>(kSlots));
    std::mt19937 random(7);
    std::uint64_t window = 0;
    for (int step = 0; step < 20000; ++step) {
        ++window;
        const std::size_t a = random() % with_slot.size();
        pool.release(0, with_slot[a], window);
        pool.retire(window);
        const std::size_t b     = random() % without.size();
        const std::int32_t next = without[b];
        ASSERT_NE(pool.reserve(0, next), ExpertHostPool::kNoSlot);
        without[b]   = with_slot[a];
        with_slot[a] = next;
    }
    std::set<std::int32_t> owned;
    for (const std::int32_t e : with_slot) { owned.insert(pool.slot(0, e)); }
    EXPECT_EQ(owned.size(), static_cast<std::size_t>(kSlots));
    EXPECT_EQ(pool.free_slots(0), 0U);
}

} // namespace

int main() {
    ReservesUntilTheLayerIsFull();
    ReleasedSlotWaitsForTheLaunchedWindows();
    RejectsDoubleOwnership();
    TransferHandsTheSlotOverAtOnce();
    LayerSlotsFollowsEveryChange();
    RejectsBadGeometry();
    RandomSwapsKeepEverySlotOwnedOnce();
    if (failures != 0) { return 1; }
    std::puts("ok");
    return 0;
}
