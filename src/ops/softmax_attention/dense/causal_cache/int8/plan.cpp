#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/instances.h"
#include "ops/softmax_attention/dense/causal_cache/int8/operands.h"
#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kGroupedPrefillMaxWidth = 256;

struct GroupedShape {
    int resident_ctas; // CTAs per SM guaranteed by the schedule's launch bound
    int key_shift;     // log2 of the schedule's key tile, the smallest split
};

template <class G>
GroupedShape grouped_shape(int tokens) {
    const auto of = []<int Tokens>() {
        using S = typename Int8KvGroupedInstance<G, Tokens>::Schedule;
        static_assert(S::kKeyRows == 32 || S::kKeyRows == 64);
        return GroupedShape{S::kMinBlocks, S::kKeyRows == 32 ? 5 : 6};
    };
    switch (tokens) {
    case 1:
        return of.template operator()<1>();
    case 2:
        return of.template operator()<2>();
    case 3:
        return of.template operator()<3>();
    case 4:
        return of.template operator()<4>();
    case 5:
        return of.template operator()<5>();
    case 6:
        return of.template operator()<6>();
    case 7:
        return of.template operator()<7>();
    default:
        return of.template operator()<8>();
    }
}

GroupedShape grouped_shape(int heads, int tokens) {
    if (heads == 24) return grouped_shape<CausalD256H24Kv4>(tokens);
    if (heads == 12) return grouped_shape<CausalD256H12Kv2>(tokens);
    return grouped_shape<CausalD256H16Kv2>(tokens);
}
} // namespace

namespace {
// int8_kv_dec_pipe_routes_for_test; -1 = default.
int test_dec_pipe               = -1;
int test_dec_pipe_resident_ctas = -1;
} // namespace

bool int8_kv_dec_pipe_enabled(int heads, int width) {
    static const bool env = [] {
        const char* e = std::getenv("NINFER_ATTN_DEC_PIPE");
        return e != nullptr && e[0] == '1';
    }();
    const bool enabled = test_dec_pipe >= 0 ? test_dec_pipe != 0 : env;
    return enabled && (heads == 12 || heads == 24) && width <= Int8KvCausalPlan::kTokenTile &&
           (width * 6 + 15) / 16 == 3;
}

Int8KvCausalPlan make_int8_kv_causal_plan(int heads, int width, int batch,
                                          CausalAttentionExecutionEnvelope envelope, int sms) {
    if ((heads != 24 && heads != 16 && heads != 12) || width < 1 || batch < 1 || batch > 8 ||
        sms < 1 || (batch > 1 && width > 16) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys)
        throw std::invalid_argument("INT8 attention: invalid plan inputs");
    constexpr int grouped_limit = Int8KvCausalPlan::kTokenTile;
    const auto family           = width <= grouped_limit             ? Int8KvFamily::Grouped
                                  : width <= kGroupedPrefillMaxWidth ? Int8KvFamily::ParallelGrouped
                                                                     : Int8KvFamily::Tiled;
    const int tiles =
        family == Int8KvFamily::ParallelGrouped ? (width + grouped_limit - 1) / grouped_limit : 1;
    const int independent_tiles = batch * (heads == 24 ? 4 : 2) * tiles;
    auto shape = grouped_shape(heads, family == Int8KvFamily::Grouped ? width : grouped_limit);
    if (family == Int8KvFamily::Grouped && int8_kv_dec_pipe_enabled(heads, width)) {
        // One 15-warp CTA per SM (a test may assume the original kernel's partition instead).
        shape.resident_ctas = test_dec_pipe_resident_ctas > 0 ? test_dec_pipe_resident_ctas : 1;
    }
    // Split rule: a row takes as many splits as it has key tiles, up to one wave of the launching
    // device (the schedule's resident CTAs per SM times its SMs, shared with the other rows, KV
    // heads and query tiles). Past one wave the partition is balanced: the fewest splits that
    // keep the longest one as short as a full wave would. A short row thus runs one key tile per
    // CTA instead of queueing behind the 128/256-key floor this replaces (4-8 tiles per split,
    // tuned for a 170-SM wave); a long row fills exactly one wave. More than one wave never
    // shortens the critical path, since 2 * ceil(n / 2w) >= ceil(n / w) key tiles.
    CausalKvPartition partition{1, std::clamp(shape.resident_ctas * sms / independent_tiles, 1,
                                              CausalKvPartition::kMaxSplits)};
    partition.key_shift = shape.key_shift;
    partition.balanced  = true;
    partition.capacity  = partition.bound(envelope.max_visible_keys);
    return {family, heads, width, batch, envelope, partition};
}

std::size_t int8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                    CausalAttentionExecutionEnvelope envelope) {
    std::size_t maximum = 0;
    for (int width = min_width; width <= std::min(max_width, kGroupedPrefillMaxWidth); ++width) {
        for (const int sms : causal_attention_device_sm_counts()) {
            const auto plan = make_int8_kv_causal_plan(heads, width, batch, envelope, sms);
            if (plan.family == Int8KvFamily::Tiled) continue;
            const int splits = plan.partition.capacity;
            WorkspaceLayoutBuilder layout;
            (void)allocate_causal_partials(layout, heads, width, splits, batch);
            maximum = std::max(maximum, layout.peak_bytes(1));
        }
    }
    return maximum;
}

} // namespace ninfer::ops::detail

namespace ninfer::ops {

void int8_kv_dec_pipe_routes_for_test(int dec_pipe, int resident_ctas) noexcept {
    detail::test_dec_pipe               = dec_pipe;
    detail::test_dec_pipe_resident_ctas = resident_ctas;
}

} // namespace ninfer::ops
