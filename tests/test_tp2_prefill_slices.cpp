// The column-slice plan of the sliced tp 2 prefill
// (models/qwen3_5/execution/tp2/prefill_slices.h), host only. The properties the schedule relies
// on: the slices tile the chunk exactly, every boundary is a multiple of 128, no slice is narrower
// than 64 columns, at most eight slices (the split exchange's lanes), and every chunk of at least
// 256 columns gets a plan in auto mode.

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

int check_plan(std::int32_t len, std::int32_t width, const PrefillSlicePlan& plan) {
    const std::string label = "len=" + std::to_string(len) + " override=" + std::to_string(width);
    if (!plan.usable()) {
        return check(plan.count == 0 && plan.width == 0, label + ": an empty plan is all zero");
    }
    int failures = 0;
    failures += check(plan.count >= 2 && plan.count <= kPrefillSliceMaxCount, label + ": count");
    failures += check(plan.width % kPrefillSliceAlign == 0, label + ": width alignment");
    failures += check(width == 0 || plan.width >= width, label + ": width below the override");
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
    const PrefillSlicePlan plan = prefill_slice_plan(len, 0);
    return check(plan.width == width && plan.count == count,
                 "auto len=" + std::to_string(len) + ": got " + std::to_string(plan.count) + "x" +
                     std::to_string(plan.width) + ", want " + std::to_string(count) + "x" +
                     std::to_string(width));
}

} // namespace

int main() {
    int failures = 0;
    for (std::int32_t len = 1; len <= 70000; ++len) {
        for (const std::int32_t width : {0, 128, 256, 384, 1024}) {
            const PrefillSlicePlan plan = prefill_slice_plan(len, width);
            failures += check_plan(len, width, plan);
            if (width == 0) {
                failures += check(plan.usable() == (len >= 256),
                                  "auto len=" + std::to_string(len) + ": usable iff len >= 256");
            }
        }
    }
    // Invalid overrides give no plan (the caller rejects them at parse time).
    for (const std::int32_t width : {-128, 1, 64, 200, 1000}) {
        failures += check(!prefill_slice_plan(1024, width).usable(),
                          "invalid override " + std::to_string(width));
        failures +=
            check(!prefill_slice_width_valid(width), "valid(" + std::to_string(width) + ")");
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
    failures += check(prefill_slice_plan(1024, 128).count == 8, "1024 / 128 = 8 slices");
    failures += check(prefill_slice_plan(4096, 128).width == 512, "4096 / 128 capped at 8 slices");
    // Mode parsing.
    failures += check(prefill_slices_mode("") == PrefillSlicesMode::Auto, "mode ''");
    failures += check(prefill_slices_mode("1") == PrefillSlicesMode::Auto, "mode '1'");
    failures += check(prefill_slices_mode("0") == PrefillSlicesMode::Off, "mode '0'");
    failures += check(prefill_slices_mode("force") == PrefillSlicesMode::Force, "mode 'force'");
    static_assert(prefill_slice_plan(700, 0).count == 5);
    static_assert(!prefill_slice_plan(200, 0).usable());
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "OK tp2_prefill_slices\n";
    return 0;
}
