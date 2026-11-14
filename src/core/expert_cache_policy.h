#pragma once

// Host-side replacement policy of the routed-expert cache: per layer and per GPU partition, which
// experts occupy which slots. Pure host code without CUDA; ExpertCache (core/expert_cache.h) drives
// it and moves the bytes.

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::expert_cache {

// One distinct expert routed in one layer during one window and the number of (token, top-k)
// assignments that chose it.
struct Routed {
    std::int32_t expert = 0;
    std::int32_t count  = 0;
};

// Rules that reproduce the offline cache simulators used to size the cache (an LRU simulator and a
// window cost model, not part of this repository) exactly. kEngine is the engine rule; the other
// two exist so a replay can check the policy window by window against those simulators.
enum class Rule : std::uint8_t {
    // Touch every hit first, then insert. A miss is a candidate when it passes admission or a slot
    // of its partition is free at that moment; with an insertion cap the candidates are taken
    // hottest first (decayed frequency with admission, window count without, ties to the lower
    // id) and inserted in that order, otherwise in ascending id order.
    kEngine,
    // lru_sim.py `lru`/`lru_id2`: one pass in ascending id order that touches hits and inserts
    // misses as it goes, so an insert may evict a hit of the same window not yet touched (which is
    // then fetched again). Requires no admission and no cap.
    kF0KSinglePass,
    // cost_model.py trace_stats: as kEngine, but "a slot is free" is judged once at window start
    // for all misses of the layer.
    kF0MWindowStartFree,
};

struct PolicyConfig {
    std::int32_t layers  = 0;
    std::int32_t experts = 512;
    // Expert e of every layer belongs to partition (GPU) e % partitions.
    std::int32_t partitions = 1; // 1..32
    // Slots per layer summed over partitions: partition p holds K/P slots, the last K%P
    // partitions one more (P = 2: K/2 and K - K/2).
    std::int32_t slots_per_layer = 0;
    // Per-layer base slots summed over partitions (each layer split between the partitions as
    // slots_per_layer is); empty = slots_per_layer for every layer. Otherwise one entry per layer,
    // and every partition must hold as many base slots over all layers as with slots_per_layer
    // (layers * partition_slot_count), so the pool and its elastic columns keep their size and
    // place: only the base region is divided differently between the layers (pool_slot).
    std::vector<std::int32_t> layer_slots;
    // Admission: a miss enters a full partition only if its decayed frequency f >= threshold, with
    // f <- f * 2^(-tokens/half_life) once per window, then f += count. Disabled when false.
    bool admission          = false;
    double admit_threshold  = 0.0;
    double half_life_tokens = 0.0;
    // Most insertions per layer per window over all partitions; 0 = unlimited.
    std::int32_t insert_cap = 0;
    // Victim of an insertion into a full partition (engine rule, decode windows; T61):
    // 0 = the least recently used expert. > 0 = frequency x recency: the expert with the lowest
    // f * 2^(-age / evict_tau_windows), f = its decayed frequency (half_life_tokens, as admission;
    // tracked also without admission), age = windows of this layer since its last use; ties go to
    // the least recently used, and experts used or inserted in the current window are not
    // candidates (if all are, the least recently used goes, as with 0). Prompt windows keep LRU.
    double evict_tau_windows = 0.0;
    // Hysteresis of a decode insertion into a full lane (radiance's heat engine): the candidate's
    // decayed frequency must reach hysteresis_ratio * the victim's score + hysteresis_gain, so the
    // two ends of the ranking do not trade places on noise. A declined swap costs a miss; a swap
    // costs link traffic in both directions with the exclusive tier. 0 ratio = off.
    double hysteresis_ratio = 0.0;
    double hysteresis_gain  = 0.0;
    Rule rule               = Rule::kEngine;
    // T69: a prompt window (decide_prompt) of T tokens adds its counts to the decayed frequency
    // weighted by the window's mean recency, hl * (1 - 2^(-T/hl)) / (T ln 2) (hl =
    // half_life_tokens; ~1 for a decode window, ~hl/(T ln 2) for a wide prompt chunk), as if its
    // tokens arrived one by one, instead of at full weight: a 16k-token chunk would otherwise
    // dominate the frequencies (and the frequency x recency victim) for thousands of decode tokens.
    bool recency_weighted_prompts = false;
    // Elastic columns (T64): every partition has `elastic_columns` slots per layer beyond its
    // base slots (partition_slot_count), of which only the first `active` columns may hold
    // experts (set_active_columns; all of them initially). They are numbered column-major after
    // the base slots and `elastic_gap` further pool slots, so the last columns are one contiguous
    // tail of the pool: see ExpertCachePolicy::pool_slot. 0 = no elastic slots.
    std::int32_t elastic_columns = 0;
    std::int32_t elastic_gap     = 0;
};

// One slot insertion: `expert` of the layer goes to `slot` of `partition`, replacing `victim`
// (-1 when the slot was free). `slot` is a pool slot of the partition
// (ExpertCachePolicy::pool_slot): base slots layer * base + local, elastic slots after them column
// by column.
struct Insert {
    std::int32_t expert    = 0;
    std::int32_t partition = 0;
    std::int32_t slot      = 0;
    std::int32_t victim    = -1;
};

// set_active_columns() results: an expert dropped from `slot`, or moved from `from` to `to` (pool
// slots of `partition`, same layer).
struct SlotEviction {
    std::int32_t layer     = 0;
    std::int32_t expert    = 0;
    std::int32_t partition = 0;
    std::int32_t slot      = 0;
};

struct SlotMove {
    std::int32_t layer     = 0;
    std::int32_t expert    = 0;
    std::int32_t partition = 0;
    std::int32_t from      = 0;
    std::int32_t to        = 0;
};

struct LayerDecision {
    std::int64_t accesses = 0; // sum of counts
    std::int64_t hits     = 0; // counts of experts resident at window start
    std::int32_t distinct = 0;
    std::vector<std::int32_t> missing; // distinct experts not resident at window start, ascending
    std::vector<Insert> inserts;       // in insertion order
};

class ExpertCachePolicy {
public:
    explicit ExpertCachePolicy(const PolicyConfig& config);

    // Slots per layer of `partition` under `config` (K/P, the last K%P partitions one more): every
    // layer's base slots without PolicyConfig::layer_slots, their mean over the layers with it.
    [[nodiscard]] static std::int32_t partition_slot_count(const PolicyConfig& config,
                                                           std::int32_t partition) noexcept {
        return split_slots(config.slots_per_layer, config.partitions, partition);
    }

    // Base slots of `layer` in `partition` under `config` (layer_slots, or slots_per_layer).
    [[nodiscard]] static std::int32_t layer_slot_count(const PolicyConfig& config,
                                                       std::int32_t layer,
                                                       std::int32_t partition) noexcept {
        const std::int32_t K = config.layer_slots.empty()
                                   ? config.slots_per_layer
                                   : config.layer_slots[static_cast<std::size_t>(layer)];
        return split_slots(K, config.partitions, partition);
    }

    // K slots over P partitions: partition p holds K/P, the last K%P one more.
    [[nodiscard]] static constexpr std::int32_t split_slots(std::int32_t K, std::int32_t P,
                                                            std::int32_t partition) noexcept {
        return K / P + (partition >= P - K % P ? 1 : 0);
    }

    // Pool slots of one partition: base slots of every layer, the gap, every elastic column.
    [[nodiscard]] static std::int64_t pool_slot_count(const PolicyConfig& config,
                                                      std::int32_t partition) noexcept {
        return static_cast<std::int64_t>(config.layers) *
                   (partition_slot_count(config, partition) + config.elastic_columns) +
               (config.elastic_columns > 0 ? config.elastic_gap : 0);
    }

    [[nodiscard]] const PolicyConfig& config() const noexcept { return config_; }

    // T69: changes PolicyConfig::insert_cap (>= 0) for the next windows.
    void set_insert_cap(std::int32_t cap);

    // Usable slots per layer of `partition` now: its base slots + the active elastic columns
    // (the layer-uniform count, partition_slot_count; per layer: the overload with `layer`).
    [[nodiscard]] std::int32_t slots_in_partition(std::int32_t partition) const noexcept {
        return partition_slots_[static_cast<std::size_t>(partition)] + active_columns_;
    }

    [[nodiscard]] std::int32_t slots_in_partition(std::int32_t partition,
                                                  std::int32_t layer) const noexcept {
        return base_slots(partition, layer) + active_columns_;
    }

    // Base slots per layer of `partition` (partition_slot_count: every layer's without
    // PolicyConfig::layer_slots, their mean with it).
    [[nodiscard]] std::int32_t base_slots(std::int32_t partition) const noexcept {
        return partition_slots_[static_cast<std::size_t>(partition)];
    }

    // Base slots of `layer` in `partition`.
    [[nodiscard]] std::int32_t base_slots(std::int32_t partition,
                                          std::int32_t layer) const noexcept {
        return layer_base_[static_cast<std::size_t>(partition * config_.layers + layer)];
    }

    // Pool slot of `layer`'s first base slot in `partition` (layer == layers: the end of the base
    // region, layers * base_slots(partition)). The base slots of layers [a, b) are the pool slots
    // [base_offset(p, a), base_offset(p, b)).
    [[nodiscard]] std::int32_t base_offset(std::int32_t partition,
                                           std::int32_t layer) const noexcept {
        return base_offset_[static_cast<std::size_t>(partition * (config_.layers + 1) + layer)];
    }

    [[nodiscard]] std::int32_t active_columns() const noexcept { return active_columns_; }

    // Pool slot of local slot `local` of `layer` in `partition`: local < base(layer):
    // base_offset(layer) + local (layer * base + local with uniform layers); else layers * base +
    // elastic_gap + (local - base(layer)) * layers + layer.
    [[nodiscard]] std::int32_t pool_slot(std::int32_t partition, std::int32_t layer,
                                         std::int32_t local) const noexcept;

    /**
     * Elastic columns (T64): makes `columns` (<= elastic_columns) elastic columns usable. Growing
     * only raises every layer's capacity: later insertions take the new slots. Shrinking keeps,
     * per layer and partition, its new capacity of experts and drops the coldest others (lowest
     * decayed frequency with admission, ties and the no-admission case by least recent use),
     * appending them to `evicted`; every survivor in a released column moves to a free kept slot
     * (appended to `moves`), so no expert is left in a released column. Frequencies and the
     * recency order of the survivors are unchanged.
     */
    void set_active_columns(std::int32_t columns, std::vector<SlotEviction>& evicted,
                            std::vector<SlotMove>& moves);

    [[nodiscard]] std::int32_t partition_of(std::int32_t expert) const noexcept {
        return expert % config_.partitions;
    }

    /**
     * One window of `layer`: `routed` lists distinct experts in ascending id order (counts >= 1),
     * `tokens` is the number of tokens of the window (drives the frequency decay). Classifies the
     * routed experts against the state at window start, then updates recency, frequency and
     * occupancy. Call it once per layer per window, also with an empty `routed`, so frequencies
     * decay at every window. `out` is overwritten.
     */
    void decide(std::int32_t layer, std::span<const Routed> routed, std::int32_t tokens,
                LayerDecision& out);

    /**
     * One prompt window (a prefill chunk) of `layer`, same input contract as decide(): the
     * experts the window routed most often become resident without an insertion cap. Per
     * partition the target is its slot count of routed experts by count (descending, ties to the
     * lower id); resident targets are touched, then missing targets are inserted coldest first,
     * evicting only non-target experts (least recently used first). Admission decays and counts
     * as in decide() but does not gate the insertions. Engine rule only.
     */
    void decide_prompt(std::int32_t layer, std::span<const Routed> routed, std::int32_t tokens,
                       LayerDecision& out);
    // T211b: decide_prompt with at most `target_cap` targets per partition (> 0; capped by the
    // slot count): only the window's `target_cap` most routed experts of each partition are made
    // resident, the other residents stay (in their recency order).
    void decide_prompt(std::int32_t layer, std::span<const Routed> routed, std::int32_t tokens,
                       std::int32_t target_cap, LayerDecision& out);

    /**
     * T211: a prompt window of `layer` that leaves the decode's state alone: classifies as
     * decide_prompt() and counts as a window of the layer (the recency age), but inserts, evicts
     * and touches nothing and neither decays nor adds to the frequencies (a wide chunk's decay
     * would erase the decode's frequencies and its counts would steer admission and eviction for
     * hundreds of decode windows). With `frequencies` (T211b) the window still decays and counts
     * them as decide_prompt() does. Engine rule only.
     */
    void count_prompt(std::int32_t layer, std::span<const Routed> routed, std::int32_t tokens,
                      LayerDecision& out, bool frequencies = false);

    /**
     * Fills the empty `layer` from a routing profile (experts hottest first): every partition
     * takes its hottest experts up to its slot count, the hottest one most recently used.
     * Frequencies are unchanged. Appends the insertions to `out`.
     */
    void prefill(std::int32_t layer, std::span<const std::int32_t> hottest_first,
                 std::vector<Insert>& out);

    /**
     * Evicts every expert of `layer` (all partitions): the layer is empty afterwards, as before
     * its first insertion. Frequencies are unchanged. Appends the evicted experts to `evicted`.
     */
    void clear_layer(std::int32_t layer, std::vector<std::int32_t>& evicted);

    /**
     * T69 lending of a layer's base slots only: evicts the experts of `layer`'s base slots (every
     * partition; appended to `evicted`) and blocks those slots until unblock_base(): the layer
     * keeps the experts of its elastic slots and may insert into its free elastic slots (its
     * capacity is the active columns). Frequencies and the survivors' recency are unchanged.
     */
    void block_base(std::int32_t layer, std::vector<SlotEviction>& evicted);
    // Ends block_base(): the layer's empty base slots become free slots again.
    void unblock_base(std::int32_t layer);

    [[nodiscard]] bool base_blocked(std::int32_t layer) const noexcept {
        return lanes_[static_cast<std::size_t>(layer * config_.partitions)].blocked;
    }

    // Slots `layer` of `partition` may hold now: slots_in_partition, minus the base slots while
    // they are blocked.
    [[nodiscard]] std::int32_t capacity(std::int32_t layer, std::int32_t partition) const noexcept;

    /**
     * T69: completes `layer` from a routing profile (experts hottest first) without evicting:
     * every partition takes its hottest absent experts up to its free capacity; on an empty
     * layer the same insertions as prefill(). Each partition's insertions go coldest first (the
     * hottest one most recently used, as prefill). Frequencies are unchanged. Appends the
     * insertions to `out`.
     */
    void top_up(std::int32_t layer, std::span<const std::int32_t> hottest_first,
                std::vector<Insert>& out);

    // Pool slot of `expert` in its partition, or -1 when absent.
    [[nodiscard]] std::int32_t slot_of(std::int32_t layer, std::int32_t expert) const noexcept {
        return slot_of_[index(layer, expert)];
    }

    // Expert in pool slot `slot` of `partition`, or -1 when the slot is free.
    [[nodiscard]] std::int32_t expert_in(std::int32_t partition, std::int32_t slot) const noexcept;
    [[nodiscard]] std::int32_t occupied(std::int32_t layer, std::int32_t partition) const noexcept;
    // Experts of `layer`/`partition` from least to most recently used.
    [[nodiscard]] std::vector<std::int32_t> recency(std::int32_t layer,
                                                    std::int32_t partition) const;

    [[nodiscard]] double frequency(std::int32_t layer, std::int32_t expert) const noexcept {
        return frequency_[index(layer, expert)];
    }

private:
    // decide_prompt / count_prompt: window count, hits and missing of a prompt window, and with
    // `frequencies` its decay and counts.
    void classify_prompt(std::int32_t layer, std::span<const Routed> routed, std::int32_t tokens,
                         bool frequencies, LayerDecision& out);

    // Per layer and partition: an intrusive LRU list over local slots and a bump allocator of
    // never-used slots (slots are only ever reassigned, never freed).
    // T69: local slots [0, used) are occupied, free (in `free`, reused before `used` grows) or,
    // while `blocked`, empty base slots; slots from `used` on are free and unlisted.
    struct Lane {
        std::int32_t head  = -1;        // least recently used local slot
        std::int32_t tail  = -1;        // most recently used local slot
        std::int32_t used  = 0;         // bump allocator: local slots ever handed out
        std::int32_t count = 0;         // occupants
        bool blocked       = false;     // base slots lent (block_base)
        std::vector<std::int32_t> free; // free local slots below `used`
    };

    [[nodiscard]] std::size_t index(std::int32_t layer, std::int32_t expert) const noexcept {
        return static_cast<std::size_t>(layer) * static_cast<std::size_t>(config_.experts) +
               static_cast<std::size_t>(expert);
    }

    [[nodiscard]] Lane& lane(std::int32_t layer, std::int32_t partition) noexcept {
        return lanes_[static_cast<std::size_t>(layer * config_.partitions + partition)];
    }

    // Index of local slot 0 of (layer, partition) in expert_of_ / prev_ / next_.
    [[nodiscard]] std::size_t lane_base(std::int32_t layer, std::int32_t partition) const noexcept {
        return partition_base_[static_cast<std::size_t>(partition)] +
               static_cast<std::size_t>(base_offset(partition, layer)) +
               static_cast<std::size_t>(layer) * static_cast<std::size_t>(config_.elastic_columns);
    }

    // (layer, local) of a pool slot of `partition`.
    void locate(std::int32_t partition, std::int32_t slot, std::int32_t& layer,
                std::int32_t& local) const noexcept;
    [[nodiscard]] bool full(std::int32_t layer, std::int32_t partition) const noexcept;
    void validate_window(std::span<const Routed> routed, std::int32_t tokens) const;
    void touch(std::int32_t layer, std::int32_t partition, std::int32_t slot) noexcept;
    // Local slot an insertion into the full lane (layer, partition) replaces: the LRU head, or the
    // frequency x recency choice (PolicyConfig::evict_tau_windows) when `scored`.
    [[nodiscard]] std::int32_t victim_local(std::int32_t layer, std::int32_t partition,
                                            bool scored) const noexcept;

    [[nodiscard]] bool tracks_frequency() const noexcept {
        return config_.admission || config_.evict_tau_windows > 0.0;
    }

    void unlink(Lane& lane, std::size_t base, std::int32_t local) noexcept;
    void push_mru(Lane& lane, std::size_t base, std::int32_t local) noexcept;
    // A free local slot of the lane below its capacity (free list first, then the bump allocator).
    [[nodiscard]] std::int32_t take_free(std::int32_t layer, std::int32_t partition);
    Insert insert(std::int32_t layer, std::int32_t expert, bool scored = false);
    void decay(std::int32_t layer, std::span<const Routed> routed, std::int32_t tokens,
               double weight = 1.0);

    PolicyConfig config_;
    std::int32_t active_columns_ = 0;
    std::vector<std::int32_t> partition_slots_; // base slots per layer of each partition (mean)
    std::vector<std::int32_t> layer_base_;      // [partitions * layers] base slots of each lane
    std::vector<std::int32_t> base_offset_;     // [partitions * (layers + 1)] prefix of layer_base_
    std::vector<std::size_t> partition_base_;   // offset of partition p's lanes in the arrays below
    std::vector<std::int32_t> slot_of_; // [layers*experts] pool slot in owner partition or -1
    // [per partition, per layer: base + elastic_columns local slots] expert or -1
    std::vector<std::int32_t> expert_of_;
    std::vector<std::int32_t> prev_, next_; // same index, LRU links, local slot numbers
    std::vector<Lane> lanes_;               // [layers*partitions]
    std::vector<double> frequency_;         // [layers*experts]
    std::vector<std::int64_t> last_use_;    // [all slots] window index of the occupant's last use
    std::vector<std::int64_t> window_;      // [layers] windows decided so far (decode + prompt)
    std::vector<std::uint8_t> routed_mark_; // [experts] scratch
    std::vector<std::int32_t> candidates_;  // scratch
};

// Distinct experts and their counts from routed ids (any order); ids outside [0, experts) are
// ignored (the -1 "skipped" convention). `out` is overwritten, ascending by expert.
void count_routed(std::span<const std::int32_t> ids, std::int32_t experts,
                  std::vector<Routed>& out);

} // namespace ninfer::expert_cache
