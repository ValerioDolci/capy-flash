#include "core/expert_cache_policy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::expert_cache {

namespace {

// Frequencies decayed below this and not routed in the window are dropped to zero (as the F0
// simulators delete them), so a long-idle expert restarts from its window count alone.
constexpr double kFrequencyFloor = 1e-3;

} // namespace

ExpertCachePolicy::ExpertCachePolicy(const PolicyConfig& config) : config_(config) {
    const PolicyConfig& c = config_;
    if (c.layers <= 0 || c.experts <= 0 || c.partitions <= 0 || c.partitions > 32) {
        throw std::invalid_argument(
            "expert cache policy: layers and experts must be > 0, partitions in [1, 32]");
    }
    if (c.slots_per_layer < c.partitions) {
        throw std::invalid_argument("expert cache policy: every partition needs at least one slot");
    }
    if (c.insert_cap < 0) {
        throw std::invalid_argument("expert cache policy: insert_cap must be >= 0");
    }
    if (c.admission && !(c.half_life_tokens > 0.0 && c.admit_threshold >= 0.0)) {
        throw std::invalid_argument("expert cache policy: admission needs half_life_tokens > 0");
    }
    if (!(c.evict_tau_windows >= 0.0) || !std::isfinite(c.evict_tau_windows) ||
        (c.evict_tau_windows > 0.0 && (!(c.half_life_tokens > 0.0) || c.rule != Rule::kEngine))) {
        throw std::invalid_argument(
            "expert cache policy: evict_tau_windows must be >= 0; > 0 needs half_life_tokens > 0 "
            "and the engine rule");
    }
    if (c.rule == Rule::kF0KSinglePass && (c.admission || c.insert_cap != 0)) {
        throw std::invalid_argument("expert cache policy: kF0KSinglePass has no admission or cap");
    }
    if (c.elastic_columns < 0) {
        throw std::invalid_argument("expert cache policy: elastic columns must be >= 0");
    }
    const std::int32_t P = c.partitions;
    if (!c.layer_slots.empty()) {
        if (c.layer_slots.size() != static_cast<std::size_t>(c.layers)) {
            throw std::invalid_argument(
                "expert cache policy: layer_slots needs one entry per layer");
        }
        for (const std::int32_t k : c.layer_slots) {
            if (k < P) {
                throw std::invalid_argument(
                    "expert cache policy: every partition needs at least one slot in every layer");
            }
        }
    }
    partition_slots_.resize(static_cast<std::size_t>(P));
    partition_base_.resize(static_cast<std::size_t>(P));
    layer_base_.resize(static_cast<std::size_t>(P * c.layers));
    base_offset_.resize(static_cast<std::size_t>(P * (c.layers + 1)));
    std::size_t total = 0;
    for (std::int32_t p = 0; p < P; ++p) {
        const std::int32_t slots                      = partition_slot_count(c, p);
        partition_slots_[static_cast<std::size_t>(p)] = slots;
        partition_base_[static_cast<std::size_t>(p)]  = total;
        std::int64_t offset                           = 0;
        for (std::int32_t l = 0; l < c.layers; ++l) {
            const std::int32_t base                                 = layer_slot_count(c, l, p);
            layer_base_[static_cast<std::size_t>(p * c.layers + l)] = base;
            base_offset_[static_cast<std::size_t>(p * (c.layers + 1) + l)] =
                static_cast<std::int32_t>(offset);
            offset += base;
        }
        if (offset != static_cast<std::int64_t>(slots) * c.layers) {
            throw std::invalid_argument(
                "expert cache policy: layer_slots must give every partition layers * "
                "partition_slot_count base slots");
        }
        base_offset_[static_cast<std::size_t>(p * (c.layers + 1) + c.layers)] =
            static_cast<std::int32_t>(offset);
        total += static_cast<std::size_t>(slots + c.elastic_columns) *
                 static_cast<std::size_t>(c.layers);
    }
    active_columns_ = c.elastic_columns;
    const std::size_t cells =
        static_cast<std::size_t>(c.layers) * static_cast<std::size_t>(c.experts);
    slot_of_.assign(cells, -1);
    frequency_.assign(cells, 0.0);
    expert_of_.assign(total, -1);
    prev_.assign(total, -1);
    next_.assign(total, -1);
    last_use_.assign(total, 0);
    window_.assign(static_cast<std::size_t>(c.layers), 0);
    lanes_.assign(static_cast<std::size_t>(c.layers * P), Lane{});
    routed_mark_.assign(static_cast<std::size_t>(c.experts), 0);
}

void ExpertCachePolicy::set_insert_cap(std::int32_t cap) {
    if (cap < 0) { throw std::invalid_argument("expert cache policy: insert_cap must be >= 0"); }
    config_.insert_cap = cap;
}

std::int32_t ExpertCachePolicy::pool_slot(std::int32_t partition, std::int32_t layer,
                                          std::int32_t local) const noexcept {
    const std::int32_t base = base_slots(partition, layer);
    if (local < base) { return base_offset(partition, layer) + local; }
    return base_offset(partition, config_.layers) + (local - base) * config_.layers + layer;
}

void ExpertCachePolicy::locate(std::int32_t partition, std::int32_t slot, std::int32_t& layer,
                               std::int32_t& local) const noexcept {
    const std::int32_t L = config_.layers;
    const auto offsets   = base_offset_.begin() + partition * (L + 1);
    if (slot < offsets[L]) {
        // The last layer whose base region begins at or before `slot` (layers with one slot or
        // more, so the region is not empty).
        layer =
            static_cast<std::int32_t>(std::upper_bound(offsets, offsets + L, slot) - offsets) - 1;
        local = slot - offsets[layer];
        return;
    }
    const std::int32_t e = slot - offsets[L];
    layer                = e % L;
    local                = base_slots(partition, layer) + e / L;
}

std::int32_t ExpertCachePolicy::expert_in(std::int32_t partition,
                                          std::int32_t slot) const noexcept {
    std::int32_t layer = 0;
    std::int32_t local = 0;
    locate(partition, slot, layer, local);
    return expert_of_[lane_base(layer, partition) + static_cast<std::size_t>(local)];
}

std::int32_t ExpertCachePolicy::occupied(std::int32_t layer,
                                         std::int32_t partition) const noexcept {
    return lanes_[static_cast<std::size_t>(layer * config_.partitions + partition)].count;
}

std::int32_t ExpertCachePolicy::capacity(std::int32_t layer,
                                         std::int32_t partition) const noexcept {
    const Lane& ln = lanes_[static_cast<std::size_t>(layer * config_.partitions + partition)];
    return ln.blocked ? active_columns_ : slots_in_partition(partition, layer);
}

bool ExpertCachePolicy::full(std::int32_t layer, std::int32_t partition) const noexcept {
    return occupied(layer, partition) >= capacity(layer, partition);
}

std::int32_t ExpertCachePolicy::take_free(std::int32_t layer, std::int32_t partition) {
    Lane& ln           = lane(layer, partition);
    std::int32_t local = 0;
    if (!ln.free.empty()) {
        local = ln.free.back();
        ln.free.pop_back();
    } else {
        local = ln.used++;
    }
    ++ln.count;
    return local;
}

std::vector<std::int32_t> ExpertCachePolicy::recency(std::int32_t layer,
                                                     std::int32_t partition) const {
    const Lane& ln = lanes_[static_cast<std::size_t>(layer * config_.partitions + partition)];
    const std::size_t base = lane_base(layer, partition);
    std::vector<std::int32_t> order;
    for (std::int32_t s = ln.head; s >= 0; s = next_[base + static_cast<std::size_t>(s)]) {
        order.push_back(expert_of_[base + static_cast<std::size_t>(s)]);
    }
    return order;
}

void ExpertCachePolicy::unlink(Lane& ln, std::size_t base, std::int32_t local) noexcept {
    const std::size_t i  = base + static_cast<std::size_t>(local);
    const std::int32_t p = prev_[i];
    const std::int32_t n = next_[i];
    if (p >= 0) {
        next_[base + static_cast<std::size_t>(p)] = n;
    } else {
        ln.head = n;
    }
    if (n >= 0) {
        prev_[base + static_cast<std::size_t>(n)] = p;
    } else {
        ln.tail = p;
    }
    prev_[i] = -1;
    next_[i] = -1;
}

void ExpertCachePolicy::push_mru(Lane& ln, std::size_t base, std::int32_t local) noexcept {
    const std::size_t i = base + static_cast<std::size_t>(local);
    prev_[i]            = ln.tail;
    next_[i]            = -1;
    if (ln.tail >= 0) {
        next_[base + static_cast<std::size_t>(ln.tail)] = local;
    } else {
        ln.head = local;
    }
    ln.tail = local;
}

void ExpertCachePolicy::touch(std::int32_t layer, std::int32_t partition,
                              std::int32_t slot) noexcept {
    Lane& ln             = lane(layer, partition);
    std::int32_t located = 0;
    std::int32_t local   = 0;
    locate(partition, slot, located, local);
    const std::size_t base                            = lane_base(layer, partition);
    last_use_[base + static_cast<std::size_t>(local)] = window_[static_cast<std::size_t>(layer)];
    if (ln.tail == local) { return; }
    unlink(ln, base, local);
    push_mru(ln, base, local);
}

std::int32_t ExpertCachePolicy::victim_local(std::int32_t layer, std::int32_t partition,
                                             bool scored) const noexcept {
    const Lane& ln = lanes_[static_cast<std::size_t>(layer * config_.partitions + partition)];
    if (!scored || config_.evict_tau_windows <= 0.0) { return ln.head; }
    const std::size_t base = lane_base(layer, partition);
    const std::int64_t now = window_[static_cast<std::size_t>(layer)];
    const double* f        = frequency_.data() + index(layer, 0);
    const double inv_tau   = 1.0 / config_.evict_tau_windows;
    std::int32_t best      = -1;
    double best_score      = 0.0;
    // Least recently used first, strict improvement: ties keep the least recently used.
    for (std::int32_t s = ln.head; s >= 0; s = next_[base + static_cast<std::size_t>(s)]) {
        const std::size_t i = base + static_cast<std::size_t>(s);
        if (last_use_[i] == now) { continue; } // used or inserted in this window
        const double age   = static_cast<double>(now - last_use_[i]);
        const double score = f[expert_of_[i]] * std::exp2(-age * inv_tau);
        if (best < 0 || score < best_score) {
            best       = s;
            best_score = score;
        }
    }
    return best >= 0 ? best : ln.head;
}

Insert ExpertCachePolicy::insert(std::int32_t layer, std::int32_t expert, bool scored) {
    const std::int32_t p   = partition_of(expert);
    Lane& ln               = lane(layer, p);
    const std::size_t base = lane_base(layer, p);
    Insert ins{expert, p, 0, -1};
    std::int32_t local = 0;
    if (ln.count < capacity(layer, p)) {
        local = take_free(layer, p);
    } else {
        local      = victim_local(layer, p, scored);
        ins.victim = expert_of_[base + static_cast<std::size_t>(local)];
        unlink(ln, base, local);
        slot_of_[index(layer, ins.victim)] = -1;
    }
    ins.slot                                           = pool_slot(p, layer, local);
    expert_of_[base + static_cast<std::size_t>(local)] = expert;
    slot_of_[index(layer, expert)]                     = ins.slot;
    last_use_[base + static_cast<std::size_t>(local)]  = window_[static_cast<std::size_t>(layer)];
    push_mru(ln, base, local);
    return ins;
}

void ExpertCachePolicy::decay(std::int32_t layer, std::span<const Routed> routed,
                              std::int32_t tokens, double weight) {
    const double factor = std::pow(2.0, static_cast<double>(-tokens) / config_.half_life_tokens);
    for (const Routed& r : routed) { routed_mark_[static_cast<std::size_t>(r.expert)] = 1; }
    double* f = frequency_.data() + index(layer, 0);
    for (std::int32_t e = 0; e < config_.experts; ++e) {
        const double v = f[e] * factor;
        f[e] = (v < kFrequencyFloor && routed_mark_[static_cast<std::size_t>(e)] == 0) ? 0.0 : v;
    }
    for (const Routed& r : routed) {
        f[r.expert] += weight * static_cast<double>(r.count);
        routed_mark_[static_cast<std::size_t>(r.expert)] = 0;
    }
}

void ExpertCachePolicy::decide(std::int32_t layer, std::span<const Routed> routed,
                               std::int32_t tokens, LayerDecision& out) {
    if (layer < 0 || layer >= config_.layers) {
        throw std::out_of_range("expert cache policy: layer " + std::to_string(layer));
    }
    validate_window(routed, tokens);
    out.accesses = 0;
    out.hits     = 0;
    out.distinct = static_cast<std::int32_t>(routed.size());
    out.missing.clear();
    out.inserts.clear();
    ++window_[static_cast<std::size_t>(layer)];

    if (tracks_frequency()) { decay(layer, routed, tokens); }
    for (const Routed& r : routed) {
        out.accesses += r.count;
        if (slot_of(layer, r.expert) >= 0) {
            out.hits += r.count;
        } else {
            out.missing.push_back(r.expert);
        }
    }

    if (config_.rule == Rule::kF0KSinglePass) {
        for (const Routed& r : routed) {
            const std::int32_t s = slot_of(layer, r.expert);
            if (s >= 0) {
                touch(layer, partition_of(r.expert), s);
            } else if (capacity(layer, partition_of(r.expert)) > 0) {
                out.inserts.push_back(insert(layer, r.expert));
            }
        }
        return;
    }

    for (const Routed& r : routed) {
        const std::int32_t s = slot_of(layer, r.expert);
        if (s >= 0) { touch(layer, partition_of(r.expert), s); }
    }
    if (out.missing.empty()) { return; }

    // Partitions with a free slot at window start (kF0MWindowStartFree judges admission on it).
    std::uint32_t free_at_start = 0;
    for (std::int32_t p = 0; p < config_.partitions; ++p) {
        if (!full(layer, p)) { free_at_start |= 1U << p; }
    }
    const double* f = frequency_.data() + index(layer, 0);
    auto admissible = [&](std::int32_t e, bool dynamic) {
        if (!config_.admission || f[e] >= config_.admit_threshold) { return true; }
        const std::int32_t p = partition_of(e);
        return dynamic ? !full(layer, p) : ((free_at_start >> p) & 1U) != 0;
    };

    candidates_.clear();
    if (config_.rule == Rule::kF0MWindowStartFree) {
        for (const std::int32_t e : out.missing) {
            if (admissible(e, false)) { candidates_.push_back(e); }
        }
    } else {
        candidates_.assign(out.missing.begin(), out.missing.end());
    }
    const std::int32_t cap = config_.insert_cap;
    if (cap > 0) {
        // Hottest first: decayed frequency with admission, the window count without. Ties keep
        // ascending id order.
        auto count_of = [&](std::int32_t e) {
            const auto it =
                std::lower_bound(routed.begin(), routed.end(), e,
                                 [](const Routed& r, std::int32_t v) { return r.expert < v; });
            return static_cast<double>(it->count);
        };
        auto heat = [&](std::int32_t e) { return config_.admission ? f[e] : count_of(e); };
        std::stable_sort(candidates_.begin(), candidates_.end(),
                         [&](std::int32_t a, std::int32_t b) { return heat(a) > heat(b); });
    }
    std::int32_t inserted = 0;
    for (const std::int32_t e : candidates_) {
        if (cap > 0 && inserted >= cap) { break; }
        if (config_.rule == Rule::kEngine && !admissible(e, true)) { continue; }
        // A lane whose base slots are blocked with no active column holds nothing.
        if (capacity(layer, partition_of(e)) == 0) { continue; }
        if (config_.hysteresis_ratio > 0.0 && config_.rule == Rule::kEngine) {
            const std::int32_t p = partition_of(e);
            if (full(layer, p)) {
                const std::int32_t local = victim_local(layer, p, /*scored=*/true);
                const std::size_t i      = lane_base(layer, p) + static_cast<std::size_t>(local);
                const double age =
                    static_cast<double>(window_[static_cast<std::size_t>(layer)] - last_use_[i]);
                const double recency = config_.evict_tau_windows > 0.0
                                           ? std::exp2(-age / config_.evict_tau_windows)
                                           : 1.0;
                const double victim  = f[expert_of_[i]] * recency;
                if (f[e] < config_.hysteresis_ratio * victim + config_.hysteresis_gain) {
                    continue;
                }
            }
        }
        out.inserts.push_back(insert(layer, e, /*scored=*/true));
        ++inserted;
    }
}

void ExpertCachePolicy::validate_window(std::span<const Routed> routed, std::int32_t tokens) const {
    if (tokens < 0) { throw std::invalid_argument("expert cache policy: tokens must be >= 0"); }
    for (std::size_t i = 0; i < routed.size(); ++i) {
        const Routed& r = routed[i];
        if (r.expert < 0 || r.expert >= config_.experts || r.count <= 0 ||
            (i > 0 && routed[i - 1].expert >= r.expert)) {
            throw std::invalid_argument(
                "expert cache policy: routed experts must be distinct, ascending, in range, "
                "with positive counts");
        }
    }
}

void ExpertCachePolicy::classify_prompt(std::int32_t layer, std::span<const Routed> routed,
                                        std::int32_t tokens, bool frequencies, LayerDecision& out) {
    if (layer < 0 || layer >= config_.layers) {
        throw std::out_of_range("expert cache policy: layer " + std::to_string(layer));
    }
    if (config_.rule != Rule::kEngine) {
        throw std::logic_error("expert cache policy: prompt windows follow the engine rule");
    }
    validate_window(routed, tokens);
    out.accesses = 0;
    out.hits     = 0;
    out.distinct = static_cast<std::int32_t>(routed.size());
    out.missing.clear();
    out.inserts.clear();
    ++window_[static_cast<std::size_t>(layer)];
    if (frequencies && tracks_frequency()) {
        double weight = 1.0;
        if (config_.recency_weighted_prompts && tokens > 0) {
            const double x = static_cast<double>(tokens) / config_.half_life_tokens;
            weight         = (1.0 - std::exp2(-x)) / (x * std::log(2.0));
        }
        decay(layer, routed, tokens, weight);
    }
    for (const Routed& r : routed) {
        out.accesses += r.count;
        if (slot_of(layer, r.expert) >= 0) {
            out.hits += r.count;
        } else {
            out.missing.push_back(r.expert);
        }
    }
}

void ExpertCachePolicy::count_prompt(std::int32_t layer, std::span<const Routed> routed,
                                     std::int32_t tokens, LayerDecision& out, bool frequencies) {
    classify_prompt(layer, routed, tokens, frequencies, out);
}

void ExpertCachePolicy::decide_prompt(std::int32_t layer, std::span<const Routed> routed,
                                      std::int32_t tokens, LayerDecision& out) {
    decide_prompt(layer, routed, tokens, std::numeric_limits<std::int32_t>::max(), out);
}

void ExpertCachePolicy::decide_prompt(std::int32_t layer, std::span<const Routed> routed,
                                      std::int32_t tokens, std::int32_t target_cap,
                                      LayerDecision& out) {
    if (target_cap <= 0) {
        throw std::invalid_argument("expert cache policy: prompt target cap must be positive");
    }
    classify_prompt(layer, routed, tokens, /*frequencies=*/true, out);
    // Targets: per partition its slot count of the most routed experts (count descending, ties
    // to the lower id), coldest first.
    candidates_.clear();
    for (const Routed& r : routed) { candidates_.push_back(r.expert); }
    auto count_of = [&](std::int32_t e) {
        const auto it =
            std::lower_bound(routed.begin(), routed.end(), e,
                             [](const Routed& r, std::int32_t v) { return r.expert < v; });
        return it->count;
    };
    std::stable_sort(candidates_.begin(), candidates_.end(),
                     [&](std::int32_t a, std::int32_t b) { return count_of(a) > count_of(b); });
    std::vector<std::int32_t> taken(static_cast<std::size_t>(config_.partitions), 0);
    std::vector<std::int32_t> targets;
    for (const std::int32_t e : candidates_) {
        auto& n = taken[static_cast<std::size_t>(partition_of(e))];
        if (n < std::min(capacity(layer, partition_of(e)), target_cap)) {
            targets.push_back(e);
            ++n;
        }
    }
    std::reverse(targets.begin(), targets.end());
    // Resident targets first, so that every insertion below evicts a non-target (a partition
    // holds at most its slot count of targets).
    for (const std::int32_t e : targets) {
        const std::int32_t s = slot_of(layer, e);
        if (s >= 0) { touch(layer, partition_of(e), s); }
    }
    for (const std::int32_t e : targets) {
        if (slot_of(layer, e) < 0) { out.inserts.push_back(insert(layer, e)); }
    }
}

void ExpertCachePolicy::prefill(std::int32_t layer, std::span<const std::int32_t> hottest_first,
                                std::vector<Insert>& out) {
    if (layer < 0 || layer >= config_.layers) {
        throw std::out_of_range("expert cache policy: layer " + std::to_string(layer));
    }
    for (std::int32_t p = 0; p < config_.partitions; ++p) {
        if (occupied(layer, p) != 0) {
            throw std::logic_error("expert cache policy: prefill needs an empty layer");
        }
    }
    std::vector<std::vector<std::int32_t>> chosen(static_cast<std::size_t>(config_.partitions));
    std::vector<std::uint8_t> seen(static_cast<std::size_t>(config_.experts), 0);
    for (const std::int32_t e : hottest_first) {
        if (e < 0 || e >= config_.experts || seen[static_cast<std::size_t>(e)] != 0) {
            throw std::invalid_argument(
                "expert cache policy: profile experts must be distinct and in range");
        }
        seen[static_cast<std::size_t>(e)] = 1;
        auto& list                        = chosen[static_cast<std::size_t>(partition_of(e))];
        if (static_cast<std::int32_t>(list.size()) < capacity(layer, partition_of(e))) {
            list.push_back(e);
        }
    }
    for (const auto& list : chosen) {
        for (auto it = list.rbegin(); it != list.rend(); ++it) {
            out.push_back(insert(layer, *it));
        }
    }
}

void ExpertCachePolicy::top_up(std::int32_t layer, std::span<const std::int32_t> hottest_first,
                               std::vector<Insert>& out) {
    if (layer < 0 || layer >= config_.layers) {
        throw std::out_of_range("expert cache policy: layer " + std::to_string(layer));
    }
    std::vector<std::vector<std::int32_t>> chosen(static_cast<std::size_t>(config_.partitions));
    std::vector<std::uint8_t> seen(static_cast<std::size_t>(config_.experts), 0);
    for (const std::int32_t e : hottest_first) {
        if (e < 0 || e >= config_.experts || seen[static_cast<std::size_t>(e)] != 0) {
            throw std::invalid_argument(
                "expert cache policy: profile experts must be distinct and in range");
        }
        seen[static_cast<std::size_t>(e)] = 1;
        if (slot_of(layer, e) >= 0) { continue; }
        const std::int32_t p = partition_of(e);
        auto& list           = chosen[static_cast<std::size_t>(p)];
        if (static_cast<std::int32_t>(list.size()) < capacity(layer, p) - occupied(layer, p)) {
            list.push_back(e);
        }
    }
    for (const auto& list : chosen) {
        for (auto it = list.rbegin(); it != list.rend(); ++it) {
            out.push_back(insert(layer, *it));
        }
    }
}

void ExpertCachePolicy::block_base(std::int32_t layer, std::vector<SlotEviction>& evicted) {
    if (layer < 0 || layer >= config_.layers) {
        throw std::out_of_range("expert cache policy: layer " + std::to_string(layer));
    }
    for (std::int32_t p = 0; p < config_.partitions; ++p) {
        Lane& ln = lane(layer, p);
        if (ln.blocked) { continue; }
        const std::size_t base   = lane_base(layer, p);
        const std::int32_t slots = base_slots(p, layer);
        for (std::int32_t local = 0; local < std::min(ln.used, slots); ++local) {
            const std::size_t i  = base + static_cast<std::size_t>(local);
            const std::int32_t e = expert_of_[i];
            if (e < 0) { continue; }
            evicted.push_back(SlotEviction{layer, e, p, pool_slot(p, layer, local)});
            unlink(ln, base, local);
            slot_of_[index(layer, e)] = -1;
            expert_of_[i]             = -1;
            --ln.count;
        }
        std::erase_if(ln.free, [&](std::int32_t local) { return local < slots; });
        ln.used    = std::max(ln.used, slots);
        ln.blocked = true;
    }
}

void ExpertCachePolicy::unblock_base(std::int32_t layer) {
    if (layer < 0 || layer >= config_.layers) {
        throw std::out_of_range("expert cache policy: layer " + std::to_string(layer));
    }
    for (std::int32_t p = 0; p < config_.partitions; ++p) {
        Lane& ln = lane(layer, p);
        if (!ln.blocked) { continue; }
        ln.blocked             = false;
        const std::size_t base = lane_base(layer, p);
        // Highest first: take_free hands out the lowest base slot first.
        for (std::int32_t local = base_slots(p, layer) - 1; local >= 0; --local) {
            if (expert_of_[base + static_cast<std::size_t>(local)] < 0) {
                ln.free.push_back(local);
            }
        }
    }
}

void ExpertCachePolicy::clear_layer(std::int32_t layer, std::vector<std::int32_t>& evicted) {
    if (layer < 0 || layer >= config_.layers) {
        throw std::out_of_range("expert cache policy: layer " + std::to_string(layer));
    }
    for (std::int32_t p = 0; p < config_.partitions; ++p) {
        const std::size_t base = lane_base(layer, p);
        Lane& ln               = lane(layer, p);
        for (std::int32_t s = 0; s < ln.used; ++s) {
            const std::size_t i  = base + static_cast<std::size_t>(s);
            const std::int32_t e = expert_of_[i];
            if (e >= 0) {
                slot_of_[index(layer, e)] = -1;
                evicted.push_back(e);
            }
            expert_of_[i] = -1;
            prev_[i]      = -1;
            next_[i]      = -1;
        }
        ln = Lane{};
    }
}

void ExpertCachePolicy::set_active_columns(std::int32_t columns, std::vector<SlotEviction>& evicted,
                                           std::vector<SlotMove>& moves) {
    if (columns < 0 || columns > config_.elastic_columns) {
        throw std::out_of_range("expert cache policy: active columns out of range");
    }
    if (columns >= active_columns_) {
        active_columns_ = columns;
        return;
    }
    std::vector<std::int32_t> order;
    for (std::int32_t layer = 0; layer < config_.layers; ++layer) {
        for (std::int32_t p = 0; p < config_.partitions; ++p) {
            Lane& ln               = lane(layer, p);
            const std::int32_t cap = base_slots(p, layer) + columns;
            if (ln.used <= cap) { continue; }
            const std::size_t base = lane_base(layer, p);
            // A lane may have free slots below `used` and, while its base slots are blocked,
            // keeps only local slots [lo, cap).
            const std::int32_t lo = ln.blocked ? base_slots(p, layer) : 0;
            // Coldest first: the LRU order, stably re-ranked by decayed frequency with admission.
            order.clear();
            for (std::int32_t s = ln.head; s >= 0; s = next_[base + static_cast<std::size_t>(s)]) {
                order.push_back(s);
            }
            if (config_.admission) {
                const double* f = frequency_.data() + index(layer, 0);
                std::stable_sort(order.begin(), order.end(), [&](std::int32_t a, std::int32_t b) {
                    return f[expert_of_[base + static_cast<std::size_t>(a)]] <
                           f[expert_of_[base + static_cast<std::size_t>(b)]];
                });
            }
            const std::int32_t excess = std::max(0, ln.count - (cap - lo));
            for (std::int32_t i = 0; i < excess; ++i) {
                const std::int32_t local = order[static_cast<std::size_t>(i)];
                const std::size_t at     = base + static_cast<std::size_t>(local);
                const std::int32_t e     = expert_of_[at];
                evicted.push_back(SlotEviction{layer, e, p, pool_slot(p, layer, local)});
                unlink(ln, base, local);
                slot_of_[index(layer, e)] = -1;
                expert_of_[at]            = -1;
                --ln.count;
            }
            // Survivors above the new capacity fill the empty slots of [lo, cap) in ascending
            // order, keeping their place in the recency list (prev/next are carried over and the
            // neighbours relinked).
            std::int32_t hole = lo;
            for (std::int32_t local = cap; local < ln.used; ++local) {
                if (expert_of_[base + static_cast<std::size_t>(local)] < 0) { continue; }
                while (expert_of_[base + static_cast<std::size_t>(hole)] >= 0) { ++hole; }
                const std::size_t from = base + static_cast<std::size_t>(local);
                const std::size_t to   = base + static_cast<std::size_t>(hole);
                const std::int32_t e   = expert_of_[from];
                const std::int32_t pv  = prev_[from];
                const std::int32_t nx  = next_[from];
                expert_of_[to]         = e;
                prev_[to]              = pv;
                next_[to]              = nx;
                last_use_[to]          = last_use_[from];
                if (pv >= 0) {
                    next_[base + static_cast<std::size_t>(pv)] = hole;
                } else {
                    ln.head = hole;
                }
                if (nx >= 0) {
                    prev_[base + static_cast<std::size_t>(nx)] = hole;
                } else {
                    ln.tail = hole;
                }
                expert_of_[from]           = -1;
                prev_[from]                = -1;
                next_[from]                = -1;
                const std::int32_t to_slot = pool_slot(p, layer, hole);
                moves.push_back(SlotMove{layer, e, p, pool_slot(p, layer, local), to_slot});
                slot_of_[index(layer, e)] = to_slot;
                ++hole;
            }
            ln.free.clear();
            for (std::int32_t local = cap - 1; local >= lo; --local) {
                if (expert_of_[base + static_cast<std::size_t>(local)] < 0) {
                    ln.free.push_back(local);
                }
            }
            ln.used = cap;
        }
    }
    active_columns_ = columns;
}

void count_routed(std::span<const std::int32_t> ids, std::int32_t experts,
                  std::vector<Routed>& out) {
    out.clear();
    std::vector<std::int32_t> sorted;
    sorted.reserve(ids.size());
    for (const std::int32_t e : ids) {
        if (e >= 0 && e < experts) { sorted.push_back(e); }
    }
    std::sort(sorted.begin(), sorted.end());
    for (const std::int32_t e : sorted) {
        if (!out.empty() && out.back().expert == e) {
            ++out.back().count;
        } else {
            out.push_back(Routed{e, 1});
        }
    }
}

} // namespace ninfer::expert_cache
