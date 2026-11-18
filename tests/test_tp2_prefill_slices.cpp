// The column-slice plan of the sliced tp 2 prefill
// (models/qwen3_5/execution/tp2/prefill_slices.h), host only. The properties the schedule relies
// on: the slices tile the chunk exactly, every boundary is a multiple of 128, no slice is narrower
// than 64 columns, at most eight slices (the split exchange's lanes), and every chunk of at least
// 256 columns gets a plan.

#include "models/qwen3_5/execution/tp2/prefill_slices.h"

#include <iostream>
#include <string>

namespace {

using namespace ninfer::models::qwen3_5::execution;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check_plan(std::int32_t len, const PrefillSlicePlan& plan) {
    const std::string label = "len=" + std::to_string(len);
    if (!plan.usable()) {
        return check(plan.count == 0 && plan.width == 0, label + ": an empty plan is all zero");
    }
    int failures = 0;
    failures += check(plan.count >= 2 && plan.count <= kPrefillSliceMaxCount, label + ": count");
    failures += check(plan.width % kPrefillSliceAlign == 0, label + ": width alignment");
    std::int32_t next = 0;
    for (std::int32_t s = 0; s < plan.count; ++s) {
        const std::int32_t first   = plan.first(s);
        const std::int32_t columns = plan.columns(s, len);
        failures += check(first == next, label + ": slices must be consecutive");
        failures += check(first % kPrefillSliceAlign == 0, label + ": boundary alignment");
        failures += check(columns >= kPrefillSliceMinTail, label + ": narrow slice");
        failures += check(s + 1 == plan.count || columns == plan.width, label + ": inner width");
        failures += check(columns < 2 * plan.width, label + ": last slice too wide");
        next = first + columns;
    }
    failures += check(next == len, label + ": slices must cover the chunk");
    return failures;
}

int check_example(std::int32_t len, std::int32_t width, std::int32_t count) {
    const PrefillSlicePlan plan = prefill_slice_plan(len);
    return check(plan.width == width && plan.count == count,
                 "auto len=" + std::to_string(len) + ": got " + std::to_string(plan.count) + "x" +
                     std::to_string(plan.width) + ", want " + std::to_string(count) + "x" +
                     std::to_string(width));
}

} // namespace

int main() {
    int failures = 0;
    for (std::int32_t len = 1; len <= 70000; ++len) {
        const PrefillSlicePlan plan = prefill_slice_plan(len);
        failures += check_plan(len, plan);
        failures += check(plan.usable() == (len >= 256),
                          "len=" + std::to_string(len) + ": usable iff len >= 256");
    }
    // Worked examples of the plan.
    failures += check_example(255, 0, 0);
    failures += check_example(256, 128, 2);
    failures += check_example(300, 128, 2);  // 128 + 172
    failures += check_example(500, 128, 4);  // 3x128 + 116
    failures += check_example(513, 128, 4);  // 3x128 + 129
    failures += check_example(700, 128, 5);  // 4x128 + 188
    failures += check_example(1500, 256, 6); // 5x256 + 220
    failures += check_example(2048, 512, 4);
    failures += check_example(4096, 1024, 4);
    failures += check_example(16384, 4096, 4);
    static_assert(prefill_slice_plan(700).count == 5);
    static_assert(!prefill_slice_plan(200).usable());
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "OK tp2_prefill_slices\n";
    return 0;
}
