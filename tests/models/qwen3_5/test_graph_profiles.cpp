#include "models/qwen3_5/program/planning/graph_profiles.h"

#include "ninfer/ops/sparse_attention.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using ninfer::models::qwen3_5::GraphExecutionProfile;
namespace detail = ninfer::models::qwen3_5::detail;

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

// The profiles past the QSA split top-k boundary launch the split (another node topology): only
// they are marked, and the topology classes stay as they were.
void check_split(const std::vector<GraphExecutionProfile>& profiles, std::uint32_t offset,
                 bool split, const char* label) {
    bool marked = false;
    for (const GraphExecutionProfile& p : profiles) {
        const bool past = static_cast<std::uint64_t>(p.min) + offset >
                          static_cast<std::uint64_t>(ninfer::ops::kQsaSplitTopkMinKeys);
        expect(p.split_topk == (split && past), label);
        expect(p.topology_class == 0, label);
        marked = marked || p.split_topk;
    }
    expect(marked == split, label);
}

} // namespace

int main() {
    const std::uint32_t capacity = 262144;
    check_split(detail::ordinary_graph_profiles(capacity, true), 1, true, "ordinary, Qwen4Exp QSA");
    check_split(detail::ordinary_graph_profiles(capacity, false), 1, false, "ordinary, no QSA");
    check_split(detail::mtp_graph_profiles(capacity, 3, true), 6, true, "MTP K=3, Qwen4Exp QSA");
    check_split(detail::mtp_graph_profiles(capacity, 5, true), 10, true, "MTP K=5, Qwen4Exp QSA");
    check_split(detail::mtp_graph_profiles(capacity, 3, false), 6, false, "MTP K=3, no QSA");
    // A capacity that never reaches the boundary has no split profile.
    check_split(detail::mtp_graph_profiles(65536, 3, true), 6, false, "MTP below the boundary");
    if (failures != 0) { return EXIT_FAILURE; }
    std::printf("graph profiles: ok\n");
    return EXIT_SUCCESS;
}
