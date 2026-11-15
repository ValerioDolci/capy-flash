// Unit tests of the expert-cache replacement policy (core/expert_cache_policy.h). Expected states
// are derived by hand from the rules documented in the header; the window-by-window replay
// against the F0 simulators lives in tests/expert_cache_replay.cpp.

#include "core/expert_cache_policy.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer::expert_cache;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

template <typename T>
std::string show(const std::vector<T>& v) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size(); ++i) { s += (i ? "," : "") + std::to_string(v[i]); }
    return s + "]";
}

template <typename T>
void check_eq(const std::vector<T>& got, const std::vector<T>& want, const std::string& what) {
    check(got == want, what + ": got " + show(got) + " want " + show(want));
}

std::vector<Routed> routed(std::initializer_list<std::pair<int, int>> list) {
    std::vector<Routed> r;
    for (const auto& [e, c] : list) { r.push_back(Routed{e, c}); }
    return r;
}

std::vector<std::int32_t> experts_of(const std::vector<Insert>& ins) {
    std::vector<std::int32_t> v;
    for (const Insert& i : ins) { v.push_back(i.expert); }
    return v;
}

std::vector<std::int32_t> victims_of(const std::vector<Insert>& ins) {
    std::vector<std::int32_t> v;
    for (const Insert& i : ins) { v.push_back(i.victim); }
    return v;
}

template <typename Exception>
void expect_throw(const std::function<void()>& fn, const std::string& what) {
    try {
        fn();
    } catch (const Exception&) { return; }
    check(false, what + " did not throw");
}

PolicyConfig base(std::int32_t slots, std::int32_t partitions = 1, std::int32_t layers = 1,
                  std::int32_t experts = 32) {
    PolicyConfig c;
    c.layers          = layers;
    c.experts         = experts;
    c.partitions      = partitions;
    c.slots_per_layer = slots;
    return c;
}

void test_lru_basic() {
    ExpertCachePolicy p(base(3));
    LayerDecision d;
    p.decide(0, routed({{1, 2}, {5, 1}}), 4, d);
    check(d.hits == 0 && d.accesses == 3 && d.distinct == 2, "lru w1 counts");
    check_eq(d.missing, {1, 5}, "lru w1 missing");
    check_eq(experts_of(d.inserts), {1, 5}, "lru w1 inserts");
    check(d.inserts[0].slot == 0 && d.inserts[1].slot == 1, "lru w1 slots in allocation order");
    check_eq(victims_of(d.inserts), {-1, -1}, "lru w1 no victims");

    p.decide(0, routed({{1, 1}, {2, 1}, {7, 1}}), 4, d);
    check(d.hits == 1 && d.accesses == 3, "lru w2 hit count is the hit expert's count");
    check_eq(d.missing, {2, 7}, "lru w2 missing");
    check_eq(experts_of(d.inserts), {2, 7}, "lru w2 inserts");
    check_eq(victims_of(d.inserts), {-1, 5}, "lru w2 evicts the least recently used (5)");
    check(d.inserts[1].slot == 1, "lru w2: 7 takes 5's slot");
    check(p.slot_of(0, 5) == -1 && p.slot_of(0, 7) == 1 && p.expert_in(0, 1) == 7,
          "lru w2 maps updated both ways");
    check_eq(p.recency(0, 0), {1, 2, 7}, "lru w2 recency");
}

void test_prompt_window() {
    ExpertCachePolicy p(base(3));
    LayerDecision d;
    p.decide(0, routed({{1, 1}, {2, 1}, {7, 1}}), 1, d); // recency [1,2,7]
    // Targets = the 3 most routed {3,4,9}; inserted coldest first, evicting LRU non-targets.
    p.decide_prompt(0, routed({{1, 1}, {3, 5}, {4, 4}, {7, 1}, {9, 2}}), 64, d);
    check(d.hits == 2 && d.accesses == 13 && d.distinct == 5, "prompt w1 counts");
    check_eq(d.missing, {3, 4, 9}, "prompt w1 missing");
    check_eq(experts_of(d.inserts), {9, 4, 3}, "prompt w1 inserts coldest first, no cap");
    check_eq(victims_of(d.inserts), {1, 2, 7}, "prompt w1 victims in LRU order");
    check_eq(p.recency(0, 0), {9, 4, 3}, "prompt w1 recency");
    // A resident target that is least recently used is touched first, never evicted.
    p.decide_prompt(0, routed({{4, 3}, {9, 3}, {20, 5}, {21, 1}}), 64, d);
    check_eq(experts_of(d.inserts), {20}, "prompt w2 inserts the missing target only");
    check_eq(victims_of(d.inserts), {3}, "prompt w2 evicts the non-target (3), not 9");
    check_eq(p.recency(0, 0), {9, 4, 20}, "prompt w2 recency (ties to the lower id: 4 hotter)");
    // Fewer routed experts than slots.
    p.decide_prompt(0, routed({{30, 1}}), 1, d);
    check_eq(victims_of(d.inserts), {9}, "prompt w3 one insertion");
    // Per partition targets (expert e -> partition e % 2, 2 slots each).
    ExpertCachePolicy q(base(4, 2));
    q.decide_prompt(0, routed({{0, 5}, {1, 4}, {2, 3}, {3, 2}, {4, 6}, {5, 1}}), 64, d);
    check_eq(experts_of(d.inserts), {3, 1, 0, 4}, "prompt partitions: two hottest of each");
    // Engine rule only.
    PolicyConfig single = base(3);
    single.rule         = Rule::kF0KSinglePass;
    ExpertCachePolicy r(single);
    expect_throw<std::logic_error>([&] { r.decide_prompt(0, routed({{1, 1}}), 1, d); },
                                   "prompt window under kF0KSinglePass");
    expect_throw<std::invalid_argument>([&] { p.decide_prompt(0, routed({{5, 1}, {2, 1}}), 1, d); },
                                        "prompt window with unsorted ids");
}

// decide_prompt with a target cap: only the cap's most routed experts per partition.
void test_prompt_target_cap() {
    ExpertCachePolicy p(base(4, 2)); // 2 slots per partition
    LayerDecision d;
    p.decide(0, routed({{0, 1}, {1, 1}, {2, 1}, {3, 1}}), 1, d); // resident 0,1,2,3
    p.decide_prompt(0, routed({{4, 9}, {5, 8}, {6, 7}, {7, 6}, {8, 5}}), 64, 1, d);
    check_eq(experts_of(d.inserts), {5, 4}, "prompt cap 1: the hottest of each partition");
    check(p.slot_of(0, 6) == -1 && p.slot_of(0, 7) == -1, "prompt cap 1: no other target");
    check_eq(victims_of(d.inserts), {1, 0}, "prompt cap 1: LRU non-targets evicted");
    check(p.slot_of(0, 2) >= 0 && p.slot_of(0, 3) >= 0, "prompt cap 1: the other residents stay");
    // A cap above the slot count is the uncapped rule.
    ExpertCachePolicy q(base(4, 2));
    q.decide_prompt(0, routed({{0, 5}, {1, 4}, {2, 3}, {3, 2}, {4, 6}, {5, 1}}), 64, 64, d);
    check_eq(experts_of(d.inserts), {3, 1, 0, 4}, "prompt cap 64 = uncapped");
    expect_throw<std::invalid_argument>([&] { q.decide_prompt(0, routed({{1, 1}}), 1, 0, d); },
                                        "prompt cap 0");
}

// A prompt window that keeps the decode's state: classified as decide_prompt, but no
// insertion, eviction, touch or frequency change.
void test_count_prompt() {
    PolicyConfig c      = base(3);
    c.half_life_tokens  = 256.0;
    c.evict_tau_windows = 4.0;
    ExpertCachePolicy p(c);
    LayerDecision d;
    p.decide(0, routed({{1, 2}, {2, 1}, {7, 1}}), 4, d); // recency [1,2,7]
    const double f1 = p.frequency(0, 1), f3 = p.frequency(0, 3), f7 = p.frequency(0, 7);
    p.count_prompt(0, routed({{1, 1}, {3, 5}, {4, 4}, {7, 1}, {9, 2}}), 4096, d);
    check(d.hits == 2 && d.accesses == 13 && d.distinct == 5, "count prompt: counts");
    check_eq(d.missing, {3, 4, 9}, "count prompt: missing");
    check(d.inserts.empty(), "count prompt: no insertion");
    check_eq(p.recency(0, 0), {1, 2, 7}, "count prompt: recency unchanged (nothing touched)");
    check(p.slot_of(0, 1) >= 0 && p.slot_of(0, 3) == -1, "count prompt: residency unchanged");
    check(p.frequency(0, 1) == f1 && p.frequency(0, 3) == f3 && p.frequency(0, 7) == f7,
          "count prompt: frequencies neither decayed nor counted");
    // The decode after it decides from the decode's state: 4 evicts the least recently used (1
    // is hotter: frequency x recency keeps it, 2 goes).
    p.decide(0, routed({{4, 1}, {7, 1}}), 4, d);
    check_eq(experts_of(d.inserts), {4}, "count prompt: the next decode inserts as usual");
    check_eq(victims_of(d.inserts), {2}, "count prompt: victim from the decode's frequencies");
    PolicyConfig single = base(3);
    single.rule         = Rule::kF0KSinglePass;
    ExpertCachePolicy r(single);
    expect_throw<std::logic_error>([&] { r.count_prompt(0, routed({{1, 1}}), 1, d); },
                                   "count prompt under kF0KSinglePass");
}

void test_touch_first_vs_single_pass() {
    // State [4,3] (4 least recent), then a window routing {1,4}.
    for (const Rule rule : {Rule::kEngine, Rule::kF0KSinglePass}) {
        PolicyConfig c = base(2);
        c.rule         = rule;
        ExpertCachePolicy p(c);
        LayerDecision d;
        p.decide(0, routed({{4, 1}}), 4, d);
        p.decide(0, routed({{3, 1}}), 4, d);
        p.decide(0, routed({{1, 1}, {4, 1}}), 4, d);
        check(d.hits == 1, "touch-first/single-pass: 4 is a hit at window start");
        check_eq(d.missing, {1}, "touch-first/single-pass: missing");
        if (rule == Rule::kEngine) {
            // 4 is touched first, so 1 evicts 3.
            check_eq(experts_of(d.inserts), {1}, "engine inserts");
            check_eq(victims_of(d.inserts), {3}, "engine evicts 3, never the routed 4");
            check_eq(p.recency(0, 0), {4, 1}, "engine recency");
        } else {
            // One ascending pass: 1 evicts the untouched 4, which is then fetched again over 3.
            check_eq(experts_of(d.inserts), {1, 4}, "single-pass inserts");
            check_eq(victims_of(d.inserts), {4, 3}, "single-pass evicts the hit 4, then 3");
            check_eq(p.recency(0, 0), {1, 4}, "single-pass recency");
        }
    }
}

void test_partitions() {
    PolicyConfig c = base(5, 2, 2);
    check(ExpertCachePolicy::partition_slot_count(c, 0) == 2 &&
              ExpertCachePolicy::partition_slot_count(c, 1) == 3,
          "K=5 over 2 partitions: 2 and 3 (lru_sim caps K//2, K-K//2)");
    ExpertCachePolicy p(c);
    LayerDecision d;
    p.decide(1, routed({{2, 1}, {3, 1}, {4, 1}, {6, 1}, {7, 1}}), 4, d);
    // Partition 0 (even) holds 2 slots: 2, 4, then 6 evicts 2. Partition 1: 3, 7.
    check_eq(experts_of(d.inserts), {2, 3, 4, 6, 7}, "partition inserts");
    check_eq(victims_of(d.inserts), {-1, -1, -1, 2, -1}, "even partition full after two");
    check(d.inserts[1].partition == 1 && d.inserts[3].partition == 0, "partition = expert % 2");
    // Layer 1 slots of partition 0 are 2..3, of partition 1 are 3..5.
    check(p.slot_of(1, 4) == 3 && p.slot_of(1, 6) == 2 && p.slot_of(1, 3) == 3 &&
              p.slot_of(1, 7) == 4,
          "slots numbered layer * slots_in_partition + local");
    check(p.occupied(0, 0) == 0 && p.occupied(1, 0) == 2 && p.occupied(1, 1) == 2,
          "layers are independent");
    check_eq(p.recency(1, 0), {4, 6}, "even recency");
}

void test_admission_and_decay() {
    PolicyConfig c     = base(1);
    c.admission        = true;
    c.admit_threshold  = 2.0;
    c.half_life_tokens = 4.0; // 4-token windows halve the frequency
    ExpertCachePolicy p(c);
    LayerDecision d;
    p.decide(0, routed({{10, 1}}), 4, d);
    check_eq(experts_of(d.inserts), {10}, "a free slot admits regardless of frequency");
    check(p.frequency(0, 10) == 1.0, "f = 0 * 0.5 + 1");
    p.decide(0, routed({{11, 1}}), 4, d);
    check(d.inserts.empty() && d.missing == std::vector<std::int32_t>{11},
          "f(11) = 1 < 2 does not enter a full cache");
    check(p.frequency(0, 10) == 0.5 && p.frequency(0, 11) == 1.0, "decay of an idle expert");
    p.decide(0, routed({{11, 2}}), 4, d);
    check(p.frequency(0, 11) == 2.5, "f(11) = 1 * 0.5 + 2");
    check_eq(experts_of(d.inserts), {11}, "f >= threshold admits");
    check_eq(victims_of(d.inserts), {10}, "and evicts the LRU");
    check(p.frequency(0, 10) == 0.25, "evicted experts keep decaying");

    // Floor: an idle frequency below 1e-3 drops to zero; 0.5^9 survives, 0.5^10 does not.
    ExpertCachePolicy q(c);
    q.decide(0, routed({{3, 1}}), 4, d);
    for (int w = 0; w < 9; ++w) { q.decide(0, {}, 4, d); }
    check(q.frequency(0, 3) == std::ldexp(1.0, -9), "0.5^9 kept");
    q.decide(0, {}, 4, d);
    check(q.frequency(0, 3) == 0.0, "0.5^10 < 1e-3 dropped");
    // A routed expert is never dropped: its tiny decayed value survives under the new count.
    ExpertCachePolicy r(c);
    r.decide(0, routed({{3, 1}}), 4, d);
    for (int w = 0; w < 9; ++w) { r.decide(0, {}, 4, d); }
    r.decide(0, routed({{3, 1}}), 4, d);
    check(r.frequency(0, 3) == 1.0 + std::ldexp(1.0, -10), "routed tiny frequency kept");
}

void test_cap() {
    PolicyConfig c = base(4);
    c.insert_cap   = 2;
    ExpertCachePolicy p(c);
    LayerDecision d;
    p.decide(0, routed({{1, 1}, {2, 3}, {3, 2}, {4, 1}}), 4, d);
    check_eq(d.missing, {1, 2, 3, 4}, "cap: every miss is reported");
    check_eq(experts_of(d.inserts), {2, 3}, "cap 2: the two most routed, hottest first");
    PolicyConfig c3 = base(4);
    c3.insert_cap   = 3;
    ExpertCachePolicy p3(c3);
    p3.decide(0, routed({{1, 1}, {2, 3}, {3, 2}, {4, 1}}), 4, d);
    check_eq(experts_of(d.inserts), {2, 3, 1}, "cap 3: ties keep ascending id");

    // With admission the order is the decayed frequency, not the window count.
    // With admission the order is the decayed frequency, not the window count.
    PolicyConfig ca     = base(1);
    ca.insert_cap       = 1;
    ca.admission        = true;
    ca.admit_threshold  = 0.0;
    ca.half_life_tokens = 4.0;
    ExpertCachePolicy pa(ca);
    pa.decide(0, routed({{20, 1}}), 4, d);        // f(20) = 1, fills the only slot
    pa.decide(0, routed({{8, 6}}), 4, d);         // f(8) = 6, admitted, evicts 20
    pa.decide(0, routed({{20, 1}}), 4, d);        // f(20) = 1.25, admitted, evicts 8; f(8) = 3
    pa.decide(0, routed({{8, 1}, {9, 2}}), 4, d); // f(8) = 1.5 + 1 = 2.5 > f(9) = 2
    check(pa.frequency(0, 8) == 2.5 && pa.frequency(0, 9) == 2.0, "cap/admission frequencies");
    check_eq(experts_of(d.inserts), {8},
             "cap with admission: highest decayed frequency first, not count");
}

void test_free_slot_rules() {
    for (const Rule rule : {Rule::kEngine, Rule::kF0MWindowStartFree}) {
        PolicyConfig c     = base(2);
        c.rule             = rule;
        c.admission        = true;
        c.admit_threshold  = 100.0;
        c.half_life_tokens = 1e9;
        ExpertCachePolicy p(c);
        LayerDecision d;
        p.decide(0, routed({{1, 1}}), 4, d);
        p.decide(0, routed({{5, 1}, {6, 1}}), 4, d);
        if (rule == Rule::kEngine) {
            check_eq(experts_of(d.inserts), {5}, "engine: only the free slot is filled");
        } else {
            check_eq(experts_of(d.inserts), {5, 6}, "F0-M: free at window start admits all");
            check_eq(victims_of(d.inserts), {-1, 1}, "F0-M: the second evicts the LRU");
        }
    }
}

void test_prefill() {
    PolicyConfig c = base(4, 2, 1);
    ExpertCachePolicy p(c);
    std::vector<Insert> ins;
    const std::vector<std::int32_t> profile{7, 2, 9, 4, 6, 1};
    p.prefill(0, profile, ins);
    check(ins.size() == 4, "prefill fills both partitions");
    check_eq(p.recency(0, 0), {4, 2}, "even partition: hottest (2) most recent, 6 left out");
    check_eq(p.recency(0, 1), {9, 7}, "odd partition: hottest (7) most recent, 1 left out");
    expect_throw<std::logic_error>([&] { p.prefill(0, profile, ins); }, "prefill of a full layer");
    ExpertCachePolicy q(c);
    expect_throw<std::invalid_argument>([&] { q.prefill(0, std::vector<std::int32_t>{1, 1}, ins); },
                                        "duplicate profile expert");
}

void test_count_routed_and_validation() {
    std::vector<Routed> r;
    count_routed(std::vector<std::int32_t>{5, -1, 3, 5, 600, 3, 5}, 512, r);
    check(r.size() == 2 && r[0].expert == 3 && r[0].count == 2 && r[1].expert == 5 &&
              r[1].count == 3,
          "count_routed skips -1 and out-of-range ids, sorts, counts");
    ExpertCachePolicy p(base(4));
    LayerDecision d;
    expect_throw<std::invalid_argument>([&] { p.decide(0, routed({{5, 1}, {3, 1}}), 4, d); },
                                        "unsorted routed");
    expect_throw<std::invalid_argument>([&] { p.decide(0, routed({{3, 0}}), 4, d); }, "zero count");
    expect_throw<std::out_of_range>([&] { p.decide(1, {}, 4, d); }, "layer out of range");
    PolicyConfig bad = base(4);
    bad.rule         = Rule::kF0KSinglePass;
    bad.insert_cap   = 2;
    expect_throw<std::invalid_argument>([&] { ExpertCachePolicy x(bad); }, "single pass with cap");
    expect_throw<std::invalid_argument>([&] { ExpertCachePolicy x(base(1, 2)); },
                                        "a partition without slots");
}

// Frequency x recency victim (PolicyConfig::evict_tau_windows). Hand-derived: half-life 4
// tokens with 4-token windows halves every frequency per window; tau = 1 window halves the score
// per window of age.
void test_evict_frequency_recency() {
    PolicyConfig c        = base(2);
    c.half_life_tokens    = 4.0;
    c.evict_tau_windows   = 1.0;
    PolicyConfig lru      = c;
    lru.evict_tau_windows = 0.0;
    ExpertCachePolicy p(c), q(lru);
    LayerDecision d, e;
    for (ExpertCachePolicy* x : {&p, &q}) {
        x->decide(0, routed({{1, 8}}), 4, d); // window 1: f1 = 8
        x->decide(0, routed({{2, 1}}), 4, d); // window 2: f1 = 4, f2 = 1
    }
    check(p.frequency(0, 1) == 4.0 && p.frequency(0, 2) == 1.0,
          "frequency tracked without admission when evict_tau > 0");
    check(q.frequency(0, 1) == 0.0, "LRU without admission tracks no frequency");
    // Window 3: f1 = 2 (age 2: score 2 / 4 = 0.5), f2 = 0.5 (age 1: score 0.25) -> 2 goes; LRU: 1.
    p.decide(0, routed({{3, 1}}), 4, d);
    q.decide(0, routed({{3, 1}}), 4, e);
    check_eq(victims_of(d.inserts), {2}, "frequency x recency evicts the lowest score");
    check_eq(victims_of(e.inserts), {1}, "LRU evicts the least recently used");
    check_eq(p.recency(0, 0), {1, 3}, "the inserted expert is the most recent");

    // Equal scores go to the least recently used: f1 = 1 (age 3: 1/8), f3 = 0.5 (age 1: 1/4)...
    // window 4 routes 5 with f5 = 1: scores f1 = 1 * 2^-3 = 0.125, f3 = 0.25 * 2^-1 = 0.125.
    p.decide(0, routed({{5, 1}}), 4, d);
    check_eq(victims_of(d.inserts), {1}, "a tie goes to the least recently used");

    // An expert used in the window is never the victim, however low its score.
    ExpertCachePolicy r(c);
    r.decide(0, routed({{7, 1}}), 4, d);  // f7 = 1, the coldest
    r.decide(0, routed({{8, 16}}), 4, d); // f8 = 16
    r.decide(0, routed({{7, 1}, {9, 1}}), 4, d);
    check_eq(victims_of(d.inserts), {8}, "the hit 7 is protected, 8 goes");
    // All residents used now (K = 1): falls back to the LRU head, as the LRU rule.
    PolicyConfig one    = c;
    one.slots_per_layer = 1;
    ExpertCachePolicy s(one);
    s.decide(0, routed({{4, 1}}), 4, d);
    s.decide(0, routed({{4, 1}, {6, 1}}), 4, d);
    check_eq(victims_of(d.inserts), {4}, "nothing else to evict: the LRU head");

    // Prompt windows keep the LRU replacement of non-targets.
    PolicyConfig c3    = c;
    c3.slots_per_layer = 3;
    ExpertCachePolicy t(c3);
    t.decide(0, routed({{1, 30}}), 4, d);        // hot but old
    t.decide(0, routed({{2, 1}, {3, 1}}), 4, d); // cold, recent
    t.decide_prompt(0, routed({{10, 5}}), 4, d);
    check_eq(victims_of(d.inserts), {1}, "prompt window: LRU non-target, not the score");

    expect_throw<std::invalid_argument>(
        [] {
            PolicyConfig b      = base(4);
            b.half_life_tokens  = 4.0;
            b.evict_tau_windows = -1.0;
            ExpertCachePolicy x(b);
        },
        "negative evict_tau");
    expect_throw<std::invalid_argument>(
        [] {
            PolicyConfig b      = base(4);
            b.evict_tau_windows = 2.0; // half_life_tokens 0
            ExpertCachePolicy x(b);
        },
        "evict_tau without a half-life");
    expect_throw<std::invalid_argument>(
        [] {
            PolicyConfig b      = base(4);
            b.half_life_tokens  = 4.0;
            b.evict_tau_windows = 2.0;
            b.rule              = Rule::kF0MWindowStartFree;
            ExpertCachePolicy x(b);
        },
        "evict_tau with a non-engine rule");
}

// Hysteresis of a decode insertion into a full lane: the candidate's decayed frequency must reach
// hysteresis_ratio x the victim's score + hysteresis_gain.
void test_hysteresis() {
    PolicyConfig c       = base(1);
    c.half_life_tokens   = 4.0; // 4-token windows halve the frequency
    c.evict_tau_windows  = 1.0;
    c.hysteresis_ratio   = 1.25;
    c.hysteresis_gain    = 1.5;
    PolicyConfig off     = c;
    off.hysteresis_ratio = 0.0;
    ExpertCachePolicy p(c), q(off);
    LayerDecision d, e;
    p.decide(0, routed({{1, 8}}), 4, d); // window 1: f1 = 8
    check_eq(experts_of(d.inserts), {1}, "hysteresis: a free slot takes any candidate");
    // Window 2: f1 = 4 (age 1: score 2), f2 = 1 < 1.25 * 2 + 1.5 = 4.
    p.decide(0, routed({{2, 1}}), 4, d);
    check(d.inserts.empty() && d.missing == std::vector<std::int32_t>{2},
          "hysteresis: a candidate below the bar does not enter a full lane");
    q.decide(0, routed({{1, 8}}), 4, e);
    q.decide(0, routed({{2, 1}}), 4, e);
    check_eq(experts_of(e.inserts), {2}, "without hysteresis the same candidate enters");
    // Window 3: f1 = 2 (age 2: score 0.5), f2 = 0.5 + 2 = 2.5 >= 1.25 * 0.5 + 1.5 = 2.125.
    p.decide(0, routed({{2, 2}}), 4, d);
    check_eq(experts_of(d.inserts), {2}, "hysteresis: a candidate over the bar enters");
    check_eq(victims_of(d.inserts), {1}, "hysteresis: and replaces the scored victim");
}

// Randomized invariants for every rule: id <-> slot maps stay a bijection, occupancy never
// exceeds a partition, insertions respect the cap, reported misses are exactly the routed experts
// absent at window start, and the engine rule never evicts an expert routed in the same window
// while the window's distinct experts of that partition fit in it.
void test_random_invariants() {
    std::mt19937 rng(1234);
    for (const Rule rule : {Rule::kEngine, Rule::kF0KSinglePass, Rule::kF0MWindowStartFree}) {
        for (int variant = 0; variant < 4; ++variant) {
            PolicyConfig c = base(9 + variant * 5, 1 + variant % 2, 3, 64);
            if (rule != Rule::kF0KSinglePass) {
                c.insert_cap       = variant;
                c.admission        = variant >= 2;
                c.admit_threshold  = 1.5;
                c.half_life_tokens = 16.0;
                // The frequency x recency victim on odd variants (with and without admission).
                c.evict_tau_windows = rule == Rule::kEngine && variant % 2 == 1 ? 4.0 : 0.0;
            }
            c.rule = rule;
            ExpertCachePolicy p(c);
            LayerDecision d;
            std::vector<Routed> r;
            for (int w = 0; w < 400; ++w) {
                for (std::int32_t l = 0; l < c.layers; ++l) {
                    std::vector<std::int32_t> ids;
                    std::geometric_distribution<int> geo(0.08);
                    for (int t = 0; t < 4 * 3; ++t) { ids.push_back(std::min(63, geo(rng) + l)); }
                    count_routed(ids, c.experts, r);
                    std::vector<std::int32_t> absent;
                    for (const Routed& x : r) {
                        if (p.slot_of(l, x.expert) < 0) { absent.push_back(x.expert); }
                    }
                    p.decide(l, r, 4, d);
                    check(d.missing == absent, "missing = absent at window start");
                    if (c.insert_cap > 0) {
                        check(static_cast<std::int32_t>(d.inserts.size()) <= c.insert_cap,
                              "cap respected");
                    }
                    for (std::int32_t q = 0; q < c.partitions; ++q) {
                        const std::int32_t slots = p.slots_in_partition(q);
                        check(p.occupied(l, q) <= slots, "occupancy bounded");
                        check(static_cast<std::int32_t>(p.recency(l, q).size()) == p.occupied(l, q),
                              "recency lists every occupant");
                        for (std::int32_t s = l * slots; s < l * slots + p.occupied(l, q); ++s) {
                            const std::int32_t e = p.expert_in(q, s);
                            check(e >= 0 && p.slot_of(l, e) == s && p.partition_of(e) == q,
                                  "slot -> expert -> slot");
                        }
                    }
                    if (rule == Rule::kEngine) {
                        std::vector<std::int32_t> per(static_cast<std::size_t>(c.partitions), 0);
                        for (const Routed& x : r) {
                            ++per[static_cast<std::size_t>(p.partition_of(x.expert))];
                        }
                        for (const Insert& ins : d.inserts) {
                            if (ins.victim < 0 || per[static_cast<std::size_t>(ins.partition)] >
                                                      p.slots_in_partition(ins.partition)) {
                                continue;
                            }
                            bool routed_now = false;
                            for (const Routed& x : r) { routed_now |= x.expert == ins.victim; }
                            check(!routed_now, "engine never evicts a routed expert that fits");
                        }
                    }
                }
                if (failures > 20) { return; }
            }
        }
    }
}

// Elastic columns: the pool-slot numbering (base slots layer-major, then the columns
// column-major), and set_active_columns under random windows: every lane keeps occupying
// local slots [0, occupied), no expert stays in a released column, a shrink evicts exactly the
// excess and the coldest (LRU order; ascending decayed frequency with admission, ties by LRU),
// survivors keep their recency order, and growing makes later insertions use the new slots.
// Frequency x recency victim over the elastic pool: one base slot + 2 elastic columns, 2 layers;
// the lane stride is base + elastic columns whatever is active, and a slot moved by a shrink keeps
// its last use. Hand-derived (half-life 4 tokens = 4-token window, tau 1 window), on both layers:
// w1 routes 1, w2 routes 2 (x3), w3 routes 3 (x2): 3 slots full. Shrink to 1 column: 1 (least
// recently used) goes, 3 moves into its slot. w4 routes 4: f2 = 3/4 at age 2 -> 0.1875,
// f3 = 2/2 at age 1 -> 0.5 => 2 goes (with 1's stale last use, 3 would score 1/8 and go).
void test_evict_frequency_recency_elastic() {
    PolicyConfig c      = base(1, 1, 2, 32);
    c.half_life_tokens  = 4.0;
    c.evict_tau_windows = 1.0;
    c.elastic_columns   = 2;
    ExpertCachePolicy p(c);
    LayerDecision d;
    for (std::int32_t l = 0; l < 2; ++l) { p.decide(l, routed({{1, 8}}), 4, d); }
    for (std::int32_t l = 0; l < 2; ++l) { p.decide(l, routed({{2, 3}}), 4, d); }
    for (std::int32_t l = 0; l < 2; ++l) { p.decide(l, routed({{3, 2}}), 4, d); }
    std::vector<SlotEviction> ev;
    std::vector<SlotMove> mv;
    p.set_active_columns(1, ev, mv);
    check(ev.size() == 2 && ev[0].expert == 1 && ev[1].expert == 1,
          "shrink evicts 1 on each layer");
    check(mv.size() == 2 && mv[0].expert == 3 && mv[1].expert == 3, "shrink moves 3 on each layer");
    for (std::int32_t l = 0; l < 2; ++l) {
        p.decide(l, routed({{4, 1}}), 4, d);
        check_eq(victims_of(d.inserts), {2}, "elastic lane: frequency x recency victim");
    }
}

void test_elastic_columns() {
    {
        PolicyConfig c    = base(4, 2, 3, 32);
        c.elastic_columns = 2;
        ExpertCachePolicy p(c);
        // partition 0: base 2, partition 1: base 2; pool slots per partition 3*(2+2) = 12
        check(ExpertCachePolicy::pool_slot_count(c, 0) == 12, "pool slot count");
        check(p.pool_slot(0, 1, 1) == 3 && p.pool_slot(0, 2, 2) == 6 + 2 &&
                  p.pool_slot(1, 0, 3) == 6 + 3,
              "pool slot formula");
        check(p.slots_in_partition(0) == 4 && p.active_columns() == 2, "all columns active");
    }
    std::mt19937 rng(64);
    for (int variant = 0; variant < 4; ++variant) {
        PolicyConfig c     = base(3 + variant, 1 + variant % 2, 4, 64);
        c.elastic_columns  = 4;
        c.insert_cap       = variant % 3;
        c.admission        = variant >= 2;
        c.admit_threshold  = 1.0;
        c.half_life_tokens = 16.0;
        ExpertCachePolicy p(c);
        {
            std::vector<std::uint8_t> seen(
                static_cast<std::size_t>(ExpertCachePolicy::pool_slot_count(c, 0)), 0);
            for (std::int32_t l = 0; l < c.layers; ++l) {
                for (std::int32_t k = 0; k < p.base_slots(0) + c.elastic_columns; ++k) {
                    const std::int32_t s = p.pool_slot(0, l, k);
                    const bool in_range  = s >= 0 && s < static_cast<std::int32_t>(seen.size());
                    check(in_range && seen[static_cast<std::size_t>(s)] == 0,
                          "pool slots distinct");
                    if (in_range) { seen[static_cast<std::size_t>(s)] = 1; }
                }
            }
        }
        LayerDecision d;
        std::vector<Routed> r;
        const auto lane_ok = [&](const std::string& when) {
            for (std::int32_t l = 0; l < c.layers; ++l) {
                for (std::int32_t q = 0; q < c.partitions; ++q) {
                    const std::int32_t used = p.occupied(l, q);
                    check(used <= p.slots_in_partition(q), when + ": occupancy bounded");
                    for (std::int32_t k = 0; k < p.base_slots(q) + c.elastic_columns; ++k) {
                        const std::int32_t s = p.pool_slot(q, l, k);
                        const std::int32_t e = p.expert_in(q, s);
                        if (k < used) {
                            check(e >= 0 && p.slot_of(l, e) == s && p.partition_of(e) == q,
                                  when + ": occupied slot -> expert -> slot");
                        } else {
                            check(e == -1, when + ": slot past the occupancy is free");
                        }
                    }
                }
            }
        };
        for (int w = 0; w < 300; ++w) {
            for (std::int32_t l = 0; l < c.layers; ++l) {
                std::vector<std::int32_t> ids;
                std::geometric_distribution<int> geo(0.1);
                for (int t = 0; t < 12; ++t) { ids.push_back(std::min(63, geo(rng) + 2 * l)); }
                count_routed(ids, c.experts, r);
                p.decide(l, r, 4, d);
            }
            if (w % 7 != 3) { continue; }
            lane_ok("before resize");
            const std::int32_t columns = static_cast<std::int32_t>(rng() % 5);
            std::vector<std::vector<std::int32_t>> before_order;
            std::vector<std::int32_t> before_used;
            for (std::int32_t l = 0; l < c.layers; ++l) {
                for (std::int32_t q = 0; q < c.partitions; ++q) {
                    before_order.push_back(p.recency(l, q));
                    before_used.push_back(p.occupied(l, q));
                }
            }
            std::vector<SlotEviction> ev;
            std::vector<SlotMove> mv;
            const std::int32_t old_columns = p.active_columns();
            p.set_active_columns(columns, ev, mv);
            check(p.active_columns() == columns, "active columns set");
            lane_ok("after resize");
            for (std::int32_t l = 0; l < c.layers; ++l) {
                for (std::int32_t q = 0; q < c.partitions; ++q) {
                    const auto i           = static_cast<std::size_t>(l * c.partitions + q);
                    const std::int32_t cap = p.base_slots(q) + columns;
                    const std::int32_t excess =
                        columns < old_columns ? std::max(0, before_used[i] - cap) : 0;
                    std::vector<std::int32_t> dropped;
                    for (const SlotEviction& v : ev) {
                        if (v.layer == l && v.partition == q) { dropped.push_back(v.expert); }
                    }
                    check(static_cast<std::int32_t>(dropped.size()) == excess, "evicts the excess");
                    std::vector<std::int32_t> coldest = before_order[i];
                    if (c.admission) {
                        std::stable_sort(coldest.begin(), coldest.end(),
                                         [&](std::int32_t a, std::int32_t b) {
                                             return p.frequency(l, a) < p.frequency(l, b);
                                         });
                    }
                    coldest.resize(static_cast<std::size_t>(excess));
                    std::vector<std::int32_t> sorted_dropped = dropped;
                    std::sort(sorted_dropped.begin(), sorted_dropped.end());
                    std::sort(coldest.begin(), coldest.end());
                    check_eq(sorted_dropped, coldest, "evicts the coldest");
                    for (const std::int32_t e : dropped) {
                        check(p.slot_of(l, e) == -1, "evicted is absent");
                    }
                    std::vector<std::int32_t> survivors;
                    for (const std::int32_t e : before_order[i]) {
                        if (std::find(dropped.begin(), dropped.end(), e) == dropped.end()) {
                            survivors.push_back(e);
                        }
                    }
                    check_eq(p.recency(l, q), survivors, "survivors keep their recency order");
                }
            }
            for (const SlotMove& m : mv) {
                check(p.slot_of(m.layer, m.expert) == m.to && m.from != m.to,
                      "moved expert at its new slot");
            }
            if (failures > 20) { return; }
        }
        // Growing: the next insertions of a full layer take the new slots before evicting.
        std::vector<SlotEviction> ev;
        std::vector<SlotMove> mv;
        p.set_active_columns(0, ev, mv);
        for (std::int32_t l = 0; l < c.layers; ++l) {
            p.decide(
                l, routed({{0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}, {6, 1}, {7, 1}, {8, 1}}),
                4, d);
        }
        ev.clear();
        mv.clear();
        p.set_active_columns(c.elastic_columns, ev, mv);
        check(ev.empty() && mv.empty(), "growing moves nothing");
        p.decide(0, routed({{60, 3}}), 4, d);
        check(d.inserts.size() <= 1 && (d.inserts.empty() || d.inserts[0].victim == -1),
              "a grown layer inserts into a free slot");
        lane_ok("after grow");
    }
}

// Lending only the base slots of a layer (block_base): one layer, 2 partitions, base 2 + 2
// elastic columns per partition (pool slots of a partition: base 0..1, elastic columns 2..3).
// Hand-derived step by step in the comments.
void test_block_base() {
    PolicyConfig c    = base(4, 2, 1, 32);
    c.elastic_columns = 2;
    ExpertCachePolicy p(c);
    std::vector<Insert> ins;
    // prefill: partition 0 takes 0 2 4 6 (hottest 0), inserted coldest first: 6 -> local 0,
    // 4 -> 1, 2 -> 2 (elastic), 0 -> 3 (elastic); partition 1 likewise 7 5 | 3 1.
    p.prefill(0, std::vector<std::int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, ins);
    check(p.occupied(0, 0) == 4 && p.occupied(0, 1) == 4, "block: full before");
    check(p.slot_of(0, 6) == 0 && p.slot_of(0, 4) == 1 && p.slot_of(0, 2) == 2 &&
              p.slot_of(0, 0) == 3,
          "block: prefill slot order");
    std::vector<SlotEviction> ev;
    p.block_base(0, ev);
    std::vector<std::int32_t> gone;
    for (const SlotEviction& v : ev) {
        gone.push_back(v.expert);
        check(v.slot < 2, "block: evicted only base slots");
    }
    std::sort(gone.begin(), gone.end());
    check_eq(gone, {4, 5, 6, 7}, "block: base residents evicted");
    check(p.base_blocked(0) && p.capacity(0, 0) == 2 && p.occupied(0, 0) == 2 &&
              p.occupied(0, 1) == 2,
          "block: capacity = elastic columns, elastic residents kept");
    check_eq(p.recency(0, 0), {2, 0}, "block: survivors keep their recency");
    // A prompt window while blocked: targets = capacity (2 per partition), inserted into the
    // elastic slots only, evicting elastic non-targets.
    LayerDecision d;
    p.decide_prompt(0, routed({{0, 1}, {10, 9}, {11, 8}, {12, 7}}), 16, d);
    for (const Insert& i : d.inserts) {
        check(i.slot >= 2, "block: prompt inserts into elastic slots");
    }
    // Partition 0 targets 10 12 (0 routed once loses), evicting 2 then 0; partition 1 takes 11.
    check(p.slot_of(0, 10) >= 2 && p.slot_of(0, 12) >= 2 && p.slot_of(0, 11) >= 2 &&
              p.slot_of(0, 0) == -1,
          "block: prompt targets in the elastic slots");
    // Decode-style decide on a blocked full lane evicts within the elastic slots only.
    p.decide(0, routed({{14, 1}, {16, 1}}), 4, d);
    for (const Insert& i : d.inserts) {
        check(i.slot >= 2, "block: decide inserts into elastic slots");
    }
    check(p.occupied(0, 0) == 2, "block: occupancy stays at the elastic capacity");
    // Unblock + top up from a profile: the base slots take the hottest absent experts, no victim.
    p.unblock_base(0);
    check(!p.base_blocked(0) && p.capacity(0, 0) == 4, "unblock: full capacity back");
    ins.clear();
    const std::vector<std::int32_t> profile{14, 20, 22, 21, 23, 24, 25};
    p.top_up(0, profile, ins);
    for (const Insert& i : ins) {
        check(i.victim == -1 && i.slot < 2, "top-up: free base slots, no victim");
    }
    check(p.slot_of(0, 20) >= 0 && p.slot_of(0, 22) >= 0 && p.slot_of(0, 24) == -1,
          "top-up: partition 0 takes 20 22 (14 resident), not 24");
    check(p.slot_of(0, 21) >= 0 && p.slot_of(0, 23) >= 0 && p.slot_of(0, 25) == -1,
          "top-up: partition 1 takes 21 23");
    check(p.occupied(0, 0) == 4 && p.occupied(0, 1) == 4, "top-up: full");
    // top_up on an empty layer = prefill.
    ExpertCachePolicy a(c), b(c);
    std::vector<Insert> ia, ib;
    a.prefill(0, profile, ia);
    b.top_up(0, profile, ib);
    check(ia.size() == ib.size(), "top-up on an empty layer = prefill (count)");
    for (std::size_t i = 0; i < std::min(ia.size(), ib.size()); ++i) {
        check(ia[i].expert == ib[i].expert && ia[i].slot == ib[i].slot &&
                  ia[i].victim == ib[i].victim,
              "top-up on an empty layer = prefill (insertions)");
    }
    // Blocking a never-filled lane: insertions go to elastic slots.
    ExpertCachePolicy e(c);
    ev.clear();
    e.block_base(0, ev);
    e.decide(0, routed({{0, 1}, {2, 1}, {4, 1}}), 4, d);
    // 0 and 2 take the two elastic slots, 4 evicts 0 (least recently used).
    check(ev.empty() && d.inserts.size() == 3 && d.inserts[2].victim == 0 && e.occupied(0, 0) == 2,
          "block empty: 2 elastic slots for 3 misses");
    for (const Insert& i : d.inserts) { check(i.slot >= 2, "block empty: elastic slots"); }
    e.unblock_base(0);
    e.decide(0, routed({{6, 1}, {8, 1}}), 4, d);
    for (const Insert& i : d.inserts) {
        check(i.victim == -1 && i.slot < 2, "unblocked empty: base slots free");
    }
}

// prod/cache: per-layer base slots. Layer l's base slots follow those of the layers below it, the
// elastic columns keep their place after the whole base region, every lane's capacity is its own
// base + the active columns, and a config that would resize the pool is rejected.
void test_layer_slots() {
    PolicyConfig c    = base(4, 2, 3, 32);
    c.layer_slots     = {2, 4, 6};
    c.elastic_columns = 1;
    ExpertCachePolicy p(c);
    for (std::int32_t q = 0; q < 2; ++q) {
        check(p.base_slots(q, 0) == 1 && p.base_slots(q, 1) == 2 && p.base_slots(q, 2) == 3,
              "layer slots split between the partitions");
        check(p.base_offset(q, 0) == 0 && p.base_offset(q, 1) == 1 && p.base_offset(q, 2) == 3 &&
                  p.base_offset(q, 3) == 6,
              "base regions one after another");
        check(ExpertCachePolicy::pool_slot_count(c, q) == 9, "pool size as with uniform layers");
    }
    check(p.pool_slot(0, 2, 0) == 3 && p.pool_slot(0, 2, 2) == 5, "layer 2 base slots 3..5");
    check(p.pool_slot(0, 0, 1) == 6 && p.pool_slot(0, 2, 3) == 8, "elastic column after the base");
    check(p.capacity(0, 0) == 2 && p.capacity(1, 0) == 3 && p.capacity(2, 1) == 4,
          "capacity = layer base + active columns");
    LayerDecision d;
    // Layer 2, partition 0: 4 slots (3 base + 1 elastic) for 5 even misses: the fifth evicts.
    p.decide(2, routed({{0, 1}, {2, 1}, {4, 1}, {6, 1}, {8, 1}}), 4, d);
    check_eq(victims_of(d.inserts), {-1, -1, -1, -1, 0}, "layer 2 holds 4 per partition");
    for (const Insert& i : d.inserts) {
        check((i.slot >= 3 && i.slot < 6) || i.slot == 8, "layer 2 slots: its base or column");
        if (i.expert != 0) {
            check(p.slot_of(2, i.expert) == i.slot && p.expert_in(0, i.slot) == i.expert,
                  "layer 2: expert <-> slot");
        }
    }
    // Layer 0 holds 2 per partition.
    p.decide(0, routed({{0, 1}, {2, 1}, {4, 1}}), 4, d);
    check_eq(victims_of(d.inserts), {-1, -1, 0}, "layer 0 holds 2 per partition");
    check(p.occupied(0, 0) == 2 && p.occupied(2, 0) == 4, "occupancy per layer");
    std::vector<SlotEviction> ev;
    std::vector<SlotMove> mv;
    p.set_active_columns(0, ev, mv);
    check(ev.size() == 2 && p.occupied(0, 0) == 1 && p.occupied(2, 0) == 3,
          "shrink drops the excess of each layer over its own base");
    for (const SlotMove& m : mv) { check(m.to < 6, "survivors move into base slots"); }
    PolicyConfig bad = c;
    bad.layer_slots  = {2, 4};
    expect_throw<std::invalid_argument>([&] { ExpertCachePolicy x(bad); }, "one entry per layer");
    bad.layer_slots = {1, 5, 6};
    expect_throw<std::invalid_argument>([&] { ExpertCachePolicy x(bad); }, "a slot per partition");
    bad.layer_slots = {2, 4, 8};
    expect_throw<std::invalid_argument>([&] { ExpertCachePolicy x(bad); }, "total must match");
    bad.layer_slots = {3, 3, 6};
    expect_throw<std::invalid_argument>([&] { ExpertCachePolicy x(bad); },
                                        "odd layers move a slot between the partitions");
}

// Random interleaving of windows, block/unblock + top-up, and elastic resizes (shrinks with
// free slots and blocked lanes included): every occupant is consistent both ways, occupancy never
// exceeds the capacity, a blocked lane holds nothing in its base slots, nothing sits in an
// inactive column, recency lists every occupant, a shrink evicts exactly the excess over the
// usable slots and moves survivors only into usable slots.
void test_block_base_random() {
    std::mt19937 rng(69);
    for (int variant = 0; variant < 10; ++variant) {
        PolicyConfig c = base(4 + 2 * (variant % 3), 1 + variant % 2, 3, 64);
        if (variant >= 6) {
            // prod/cache: per-layer base slots (same total; even entries at 2 partitions).
            const std::int32_t K = c.slots_per_layer, P = c.partitions;
            c.layer_slots = {P, 3 * K - P - (K + 2 * P), K + 2 * P};
            if (variant % 4 == 3) { std::swap(c.layer_slots[0], c.layer_slots[2]); }
        }
        c.elastic_columns   = 3;
        c.admission         = variant >= 3;
        c.admit_threshold   = 1.0;
        c.half_life_tokens  = 16.0;
        c.evict_tau_windows = variant % 2 == 1 ? 4.0 : 0.0;
        c.insert_cap        = variant % 3;
        ExpertCachePolicy p(c);
        LayerDecision d;
        std::vector<Routed> r;
        std::vector<Insert> ins;
        const auto lanes_ok = [&](const std::string& when) {
            for (std::int32_t l = 0; l < c.layers; ++l) {
                for (std::int32_t q = 0; q < c.partitions; ++q) {
                    const std::int32_t base = p.base_slots(q, l);
                    std::int32_t seen       = 0;
                    for (std::int32_t k = 0; k < base + c.elastic_columns; ++k) {
                        const std::int32_t s = p.pool_slot(q, l, k);
                        const std::int32_t e = p.expert_in(q, s);
                        if (e < 0) { continue; }
                        ++seen;
                        check(p.slot_of(l, e) == s && p.partition_of(e) == q,
                              when + ": slot -> expert -> slot");
                        check(k < base + p.active_columns(),
                              when + ": nothing in an inactive column");
                        check(!(p.base_blocked(l) && k < base),
                              when + ": blocked base slots hold nothing");
                    }
                    check(seen == p.occupied(l, q), when + ": occupancy counts the occupants");
                    check(p.occupied(l, q) <= p.capacity(l, q), when + ": occupancy <= capacity");
                    check(static_cast<std::int32_t>(p.recency(l, q).size()) == p.occupied(l, q),
                          when + ": recency lists every occupant");
                }
                for (std::int32_t e = 0; e < c.experts; ++e) {
                    const std::int32_t s = p.slot_of(l, e);
                    if (s >= 0) {
                        check(p.expert_in(p.partition_of(e), s) == e,
                              when + ": expert -> slot -> expert");
                    }
                }
            }
        };
        for (int w = 0; w < 400; ++w) {
            const std::int32_t l = static_cast<std::int32_t>(rng() % c.layers);
            const int action     = static_cast<int>(rng() % 10);
            if (action < 6) {
                std::vector<std::int32_t> ids;
                std::geometric_distribution<int> geo(0.1);
                for (int t = 0; t < 12; ++t) { ids.push_back(std::min(63, geo(rng) + 3 * l)); }
                count_routed(ids, c.experts, r);
                if (action < 4) {
                    p.decide(l, r, 4, d);
                } else {
                    p.decide_prompt(l, r, 4, d);
                }
                for (const Insert& i : d.inserts) {
                    std::int32_t layer = 0, local = 0;
                    const std::int32_t q = i.partition;
                    // Recover the local slot from the pool slot (base regions layer after layer,
                    // then the elastic columns).
                    if (i.slot < p.base_offset(q, c.layers)) {
                        while (p.base_offset(q, layer + 1) <= i.slot) { ++layer; }
                        local = i.slot - p.base_offset(q, layer);
                    } else {
                        const std::int32_t x = i.slot - p.base_offset(q, c.layers);
                        layer                = x % c.layers;
                        local                = p.base_slots(q, layer) + x / c.layers;
                    }
                    check(layer == l, "insert into its own layer");
                    check(!(p.base_blocked(l) && local < p.base_slots(q, l)),
                          "blocked: no insert into a base slot");
                }
            } else if (action == 6) {
                std::vector<SlotEviction> ev;
                const bool was = p.base_blocked(l);
                p.block_base(l, ev);
                if (was) { check(ev.empty(), "block twice evicts nothing"); }
                for (const SlotEviction& v : ev) {
                    check(v.layer == l && p.slot_of(l, v.expert) == -1, "block: evicted absent");
                }
            } else if (action == 7) {
                p.unblock_base(l);
                std::vector<std::int32_t> profile(static_cast<std::size_t>(c.experts));
                std::iota(profile.begin(), profile.end(), 0);
                std::shuffle(profile.begin(), profile.end(), rng);
                profile.resize(static_cast<std::size_t>(rng() % 20));
                ins.clear();
                p.top_up(l, profile, ins);
                for (const Insert& i : ins) { check(i.victim == -1, "top-up never evicts"); }
            } else {
                const std::int32_t columns = static_cast<std::int32_t>(rng() % 4);
                std::vector<std::int32_t> occ_before(
                    static_cast<std::size_t>(c.layers * c.partitions));
                for (std::int32_t ll = 0; ll < c.layers; ++ll) {
                    for (std::int32_t q = 0; q < c.partitions; ++q) {
                        occ_before[static_cast<std::size_t>(ll * c.partitions + q)] =
                            p.occupied(ll, q);
                    }
                }
                std::vector<SlotEviction> ev;
                std::vector<SlotMove> mv;
                const std::int32_t old = p.active_columns();
                p.set_active_columns(columns, ev, mv);
                for (std::int32_t ll = 0; ll < c.layers; ++ll) {
                    for (std::int32_t q = 0; q < c.partitions; ++q) {
                        const std::int32_t room =
                            (p.base_blocked(ll) ? 0 : p.base_slots(q, ll)) + columns;
                        const std::int32_t was =
                            occ_before[static_cast<std::size_t>(ll * c.partitions + q)];
                        std::int32_t dropped = 0;
                        for (const SlotEviction& v : ev) {
                            dropped += v.layer == ll && v.partition == q;
                        }
                        check(dropped == (columns < old ? std::max(0, was - room) : 0),
                              "resize evicts the excess");
                    }
                }
                for (const SlotMove& m : mv) {
                    check(p.slot_of(m.layer, m.expert) == m.to, "moved expert at its new slot");
                }
            }
            lanes_ok("step " + std::to_string(w));
            if (failures > 20) { return; }
        }
    }
}

// Recency-weighted prompt counts. half-life 256 tokens: a 16 384-token prompt window adds
// count * 256 (1 - 2^-64) / (16 384 ln 2) = count * 0.022541; a 4-token window (decide) adds the
// full count; without the flag the prompt adds the full count (as before).
void test_recency_weighted_prompts() {
    for (const bool weighted : {false, true}) {
        PolicyConfig c             = base(4, 1, 1, 32);
        c.half_life_tokens         = 256.0;
        c.evict_tau_windows        = 4.0;
        c.recency_weighted_prompts = weighted;
        ExpertCachePolicy p(c);
        LayerDecision d;
        p.decide_prompt(0, routed({{3, 300}, {5, 20}}), 16384, d);
        const double w =
            weighted ? 256.0 * (1.0 - std::exp2(-64.0)) / (16384.0 * std::log(2.0)) : 1.0;
        check(std::abs(p.frequency(0, 3) - 300.0 * w) < 1e-9 &&
                  std::abs(p.frequency(0, 5) - 20.0 * w) < 1e-9,
              std::string("prompt counts weighted by mean recency (") + (weighted ? "on" : "off") +
                  ")");
        const double f3 = p.frequency(0, 3);
        p.decide(0, routed({{3, 2}}), 4, d);
        check(std::abs(p.frequency(0, 3) - (f3 * std::exp2(-4.0 / 256.0) + 2.0)) < 1e-9,
              "decode windows keep full-weight counts");
    }
}

// set_insert_cap changes the cap of the next windows (the post-prompt boost).
void test_set_insert_cap() {
    PolicyConfig c = base(8, 1, 1, 32);
    c.insert_cap   = 1;
    ExpertCachePolicy p(c);
    LayerDecision d;
    p.decide(0, routed({{1, 1}, {2, 1}, {3, 1}}), 4, d);
    check(d.inserts.size() == 1, "cap 1: one insertion");
    p.set_insert_cap(3);
    p.decide(0, routed({{4, 1}, {5, 1}, {6, 1}, {7, 1}}), 4, d);
    check(d.inserts.size() == 3, "cap raised to 3: three insertions");
    p.set_insert_cap(0);
    p.decide(0, routed({{8, 1}, {9, 1}, {10, 1}}), 4, d);
    check(d.inserts.size() == 3, "cap 0: no cap");
    expect_throw<std::invalid_argument>([&] { p.set_insert_cap(-1); }, "negative cap");
}
} // namespace

int main() {
    test_lru_basic();
    test_prompt_window();
    test_count_prompt();
    test_prompt_target_cap();
    test_touch_first_vs_single_pass();
    test_partitions();
    test_admission_and_decay();
    test_cap();
    test_free_slot_rules();
    test_prefill();
    test_count_routed_and_validation();
    test_evict_frequency_recency();
    test_hysteresis();
    test_random_invariants();
    test_elastic_columns();
    test_evict_frequency_recency_elastic();
    test_block_base();
    test_block_base_random();
    test_layer_slots();
    test_recency_weighted_prompts();
    test_set_insert_cap();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "expert cache policy: all checks passed\n";
    return 0;
}
