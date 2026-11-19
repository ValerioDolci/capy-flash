// The QSA top-k stage (qsa_topk_launch: qsa_select's routes and scratch) against an exact CPU
// selection on synthetic score rows. Per column: no row (an invalid column) selects nothing; a row
// of at most block_topk blocks selects every token; otherwise the block_topk highest scores, ties
// to the lower block, in ascending block order, each block's ratio tokens, then the tokens past the
// last whole block. Rows below and above the split threshold in one call, envelopes below and
// above it (a long envelope with short rows takes the one-CTA kernel per column), 1-16 columns, one
// or two lanes, valid-column masks, heavy ties and continuous scores.
#include "ops/sparse_attention/launch.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

namespace {

constexpr std::int32_t kRatio  = 4;
constexpr std::int32_t kBudget = 2048;
constexpr std::int32_t kKeep   = kBudget / kRatio;
constexpr std::int32_t kSlots  = 8; // pass slots per tile (kTileColumns)

int failures = 0;

bool ok(cudaError_t e, const char* what) {
    if (e == cudaSuccess) { return true; }
    std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(e));
    ++failures;
    return false;
}

std::uint32_t order_key(float x) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// The exact selection of one column holding `visible` keys over `row` (its block scores).
std::vector<std::int32_t> reference(const float* row, std::int32_t visible) {
    const std::int32_t blocks = visible / kRatio;
    std::vector<std::int32_t> out;
    if (blocks <= kKeep) {
        out.resize(static_cast<std::size_t>(visible));
        std::iota(out.begin(), out.end(), 0);
        return out;
    }
    std::vector<std::int32_t> order(static_cast<std::size_t>(blocks));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::int32_t a, std::int32_t b) {
        return order_key(row[a]) > order_key(row[b]);
    });
    order.resize(kKeep);
    std::sort(order.begin(), order.end());
    for (const std::int32_t b : order) {
        for (std::int32_t t = 0; t < kRatio; ++t) { out.push_back(b * kRatio + t); }
    }
    for (std::int32_t t = blocks * kRatio; t < visible; ++t) { out.push_back(t); }
    return out;
}

// One call: `lanes` lanes of `width` columns, lane l's column w at position rows[l] - 1 + w,
// lane 0 with `live` valid columns (< 0: no mask), envelope `envelope` keys.
void run_case(std::uint32_t envelope, std::int32_t width, std::int32_t lanes,
              const std::vector<std::int32_t>& rows, std::int32_t live, std::int32_t ties,
              std::uint32_t seed) {
    using namespace ninfer::ops::detail;
    const QsaSelectPlan plan = qsa_select_plan(envelope, kRatio, width, lanes);
    const std::int32_t tiles = plan.tiles_per_lane * lanes;
    if (tiles > plan.pass_tiles) {
        std::fprintf(stderr, "FAIL: the test expects one pass\n");
        ++failures;
        return;
    }
    const std::size_t scratch_bytes = qsa_select_scratch_bytes(plan);
    const std::size_t row_floats    = static_cast<std::size_t>(plan.max_blocks);
    std::vector<float> scores(static_cast<std::size_t>(plan.pass_tiles) * kSlots * row_floats);
    std::uint32_t x = seed * 2654435761u + 1u;
    for (float& v : scores) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        v = ties > 0 ? static_cast<float>(x % static_cast<std::uint32_t>(ties))
                     : static_cast<float>(x & 0xFFFFFF) / 16777216.0f;
    }
    const std::int32_t columns = lanes * width;
    std::vector<std::int32_t> cache(static_cast<std::size_t>(columns));
    for (std::int32_t l = 0; l < lanes; ++l) {
        for (std::int32_t w = 0; w < width; ++w) {
            cache[static_cast<std::size_t>(l * width + w)] =
                rows[static_cast<std::size_t>(l)] - 1 + w;
        }
    }
    std::vector<std::int32_t> valid(static_cast<std::size_t>(lanes), width);
    if (live >= 0) { valid[0] = live; }
    const std::int32_t max_selected = kBudget + kRatio - 1;
    void* d_scratch                 = nullptr;
    std::int32_t *d_cache = nullptr, *d_valid = nullptr, *d_sel = nullptr, *d_cnt = nullptr;
    const std::size_t sel_count = static_cast<std::size_t>(columns) * max_selected;
    if (!ok(cudaMalloc(&d_scratch, scratch_bytes), "scratch") ||
        !ok(cudaMalloc(&d_cache, cache.size() * 4), "cache") ||
        !ok(cudaMalloc(&d_valid, valid.size() * 4), "valid") ||
        !ok(cudaMalloc(&d_sel, sel_count * 4), "selected") ||
        !ok(cudaMalloc(&d_cnt, static_cast<std::size_t>(columns) * 4), "counts")) {
        return;
    }
    ok(cudaMemcpy(d_scratch, scores.data(), scores.size() * 4, cudaMemcpyHostToDevice), "scores");
    ok(cudaMemcpy(d_cache, cache.data(), cache.size() * 4, cudaMemcpyHostToDevice), "cache");
    ok(cudaMemcpy(d_valid, valid.data(), valid.size() * 4, cudaMemcpyHostToDevice), "valid");
    ok(cudaMemset(d_sel, 0xFF, sel_count * 4), "selected");
    ok(cudaMemset(d_cnt, 0x7F, static_cast<std::size_t>(columns) * 4), "counts");
    qsa_topk_launch(plan, kRatio, kBudget, d_cache, live >= 0 ? d_valid : nullptr, width, lanes,
                    static_cast<float*>(d_scratch), d_sel, max_selected, d_cnt, nullptr);
    ok(cudaDeviceSynchronize(), "qsa_topk_launch");
    std::vector<std::int32_t> sel(sel_count), cnt(static_cast<std::size_t>(columns));
    ok(cudaMemcpy(sel.data(), d_sel, sel_count * 4, cudaMemcpyDeviceToHost), "selected");
    ok(cudaMemcpy(cnt.data(), d_cnt, cnt.size() * 4, cudaMemcpyDeviceToHost), "counts");
    int bad = 0;
    for (std::int32_t l = 0; l < lanes; ++l) {
        for (std::int32_t w = 0; w < width; ++w) {
            const std::int32_t column  = l * width + w;
            const bool live_column     = w < valid[static_cast<std::size_t>(l)];
            const std::int32_t visible = cache[static_cast<std::size_t>(column)] + 1;
            const std::int32_t tile    = l * plan.tiles_per_lane + w / plan.tile_columns;
            const std::int32_t slot    = tile * kSlots + w % plan.tile_columns;
            std::vector<std::int32_t> want;
            if (live_column) {
                want =
                    reference(scores.data() + static_cast<std::size_t>(slot) * row_floats, visible);
            }
            const std::int32_t got = cnt[static_cast<std::size_t>(column)];
            bool same              = got == static_cast<std::int32_t>(want.size());
            for (std::size_t i = 0; same && i < want.size(); ++i) {
                same = sel[static_cast<std::size_t>(column) * max_selected + i] == want[i];
            }
            bad += same ? 0 : 1;
        }
    }
    if (bad != 0) {
        std::fprintf(stderr,
                     "FAIL: envelope %u width %d lanes %d rows %d/%d live %d ties %d: %d columns "
                     "differ\n",
                     envelope, width, lanes, rows[0], lanes > 1 ? rows[1] : 0, live, ties, bad);
        ++failures;
    }
    cudaFree(d_scratch);
    cudaFree(d_cache);
    cudaFree(d_valid);
    cudaFree(d_sel);
    cudaFree(d_cnt);
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    // Rows in blocks against the split threshold (20480 blocks): short, staged, at the threshold,
    // past it.
    const std::int32_t rows[]   = {900, 5000, 20480 * 4, 20480 * 4 + 37, 114414, 262000};
    const std::int32_t widths[] = {1, 2, 4, 8, 16};
    const std::int32_t ties[]   = {0, 3, 50};
    std::uint32_t seed          = 1;
    for (const std::int32_t r : rows) {
        for (const std::int32_t w : widths) {
            for (const std::int32_t t : ties) {
                // Envelope of the rows themselves, and the top tier's (262144) over them.
                for (const std::uint32_t envelope :
                     {static_cast<std::uint32_t>(r + w), std::uint32_t{262144}}) {
                    run_case(envelope, w, 1, {r}, -1, t, seed++);
                    run_case(envelope, w, 1, {r}, (w + 1) / 2, t, seed++);
                    // Two lanes, one past the threshold and one well below it, in one pass.
                    run_case(std::max<std::uint32_t>(envelope, 114414 + w), w, 2, {r, 30000}, -1, t,
                             seed++);
                    run_case(std::max<std::uint32_t>(envelope, 114414 + w), w, 2, {114414, r}, 0, t,
                             seed++);
                }
            }
        }
    }
    if (failures != 0) { return 1; }
    std::printf("qsa top-k: every column matches the exact selection\n");
    return 0;
}
