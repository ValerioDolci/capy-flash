// The offload backend's slot arena for ggml block banks (Qwen4OffloadExperts::gguf_pool) against
// the expert cache's slot numbering (ExpertCachePolicy::pool_slot), on the Flash-Next production
// shape (49 cache layers at tp 2, per-layer base slot profile, elastic columns) with the real
// per-layer expert bytes of the GSQ-RCO artifact. CPU only. Criteria: every (partition, layer,
// local slot) maps to a distinct pool slot whose byte range [offset, offset + slot bytes of its
// layer) lies in the arena and overlaps no other slot's; base slots end at base_bytes; elastic
// column c of every layer lies in [base_bytes + c * column_bytes, base_bytes + (c + 1) *
// column_bytes) (one contiguous tail).
#include "core/expert_cache_policy.h"
#include "models/qwen3_5/execution/qwen4_experts.h"

#include <algorithm>
#include <cstdio>
#include <vector>

using namespace ninfer;
using execution_t = models::qwen3_5::execution::Qwen4OffloadExperts;

int main() {
    int failures     = 0;
    const auto check = [&](bool ok, const char* what) {
        if (!ok) {
            ++failures;
            std::fprintf(stderr, "FAIL: %s\n", what);
        }
    };
    // Per-expert bytes of the 48 GSQ-RCO layers + the IQ4_NL MTP layer, rounded to 256.
    const std::int64_t kIq2q2 = 1510400, kIq2nl = 1971200, kXxsq2 = 1715200, kXxsnl = 2176000,
                       kSq2 = 1868800, kSnl = 2329600, kXsnl = 2662400, kMtp = 2764800;
    const std::vector<std::int64_t> raw = {
        kXxsnl, kIq2q2, kIq2nl, kIq2nl, kIq2nl, kIq2nl, kIq2nl, kXxsnl, kIq2q2, kIq2nl,
        kIq2q2, kIq2nl, kXxsq2, kIq2q2, kXxsnl, kIq2nl, kXxsnl, kSnl,   kXxsnl, kIq2nl,
        kXxsq2, kSq2,   kXxsnl, kIq2nl, kIq2nl, kSnl,   kIq2nl, kSnl,   kSnl,   kSnl,
        kXxsq2, kXxsnl, kIq2nl, kXxsnl, kIq2nl, kSnl,   kIq2q2, kXxsnl, kXxsnl, kIq2nl,
        kSnl,   kXxsnl, kXxsnl, kXxsnl, kSnl,   kSnl,   kXxsnl, kXsnl,  kMtp};
    std::vector<std::int64_t> bytes;
    for (const auto b : raw) bytes.push_back((b + 255) / 256 * 256);
    for (const auto& [spl, elastic] : {std::pair{66, 21}, std::pair{90, 0}, std::pair{40, 60}}) {
        const std::int32_t L = 49, P = 2;
        const auto base = execution_t::layer_slot_profile(L, 48, spl, elastic, P, 0);
        const auto pool = execution_t::gguf_pool(bytes, base, elastic);
        expert_cache::PolicyConfig cfg;
        cfg.layers          = L;
        cfg.partitions      = P;
        cfg.slots_per_layer = spl * P;
        for (const auto b : base) cfg.layer_slots.push_back(b * P);
        cfg.elastic_columns = elastic;
        const expert_cache::ExpertCachePolicy policy(cfg);
        check(static_cast<std::int64_t>(pool.slot_offset.size()) ==
                  expert_cache::ExpertCachePolicy::pool_slot_count(cfg, 0),
              "pool slot count = the cache's");
        for (std::int32_t p = 0; p < P; ++p) {
            std::vector<std::pair<std::int64_t, std::int64_t>> ranges;
            for (std::int32_t l = 0; l < L; ++l) {
                for (std::int32_t local = 0; local < base[l] + elastic; ++local) {
                    const std::int32_t s = policy.pool_slot(p, l, local);
                    if (s < 0 || s >= static_cast<std::int32_t>(pool.slot_offset.size())) {
                        check(false, "pool slot in range");
                        continue;
                    }
                    const std::int64_t off = pool.slot_offset[static_cast<std::size_t>(s)];
                    ranges.emplace_back(off, off + bytes[l]);
                    if (local < base[l]) {
                        check(off + bytes[l] <= pool.base_bytes, "base slot below base_bytes");
                    } else {
                        const std::int64_t c = local - base[l];
                        check(off >= pool.base_bytes + c * pool.column_bytes &&
                                  off + bytes[l] <= pool.base_bytes + (c + 1) * pool.column_bytes,
                              "elastic slot inside its column");
                    }
                    check(off % 256 == 0, "slot offset 256-aligned");
                }
            }
            std::sort(ranges.begin(), ranges.end());
            for (std::size_t i = 1; i < ranges.size(); ++i) {
                check(ranges[i].first >= ranges[i - 1].second, "slots do not overlap");
            }
            check(ranges.back().second <= pool.total_bytes, "slots inside the arena");
        }
        std::printf("spl %d elastic %d: %zu slots, %.2f GiB (base %.2f, column %.1f MB)\n", spl,
                    elastic, pool.slot_offset.size(), pool.total_bytes / 1073741824.0,
                    pool.base_bytes / 1073741824.0, pool.column_bytes / 1e6);
    }
    if (failures) {
        std::printf("FAILED: %d\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
