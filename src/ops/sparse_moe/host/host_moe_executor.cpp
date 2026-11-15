#include "ops/sparse_moe/host/host_moe_executor.h"

#include <immintrin.h>
#include <pthread.h>
#include <utility>
#include <sched.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

namespace ninfer::ops::host {
namespace {

constexpr std::int32_t kSliceRows   = 16; // consecutive rows of a slice (one int8 block of h)
constexpr std::int32_t kMaxItemRows = 4 * kSliceRows; // rows of one Static run

// Static calls: the bytes of its own down range a member prefetches (into L2) while it waits at the
// barrier after gate/up, in the order it will read them. The gate/up tail (up to one 16-row slice,
// ~5 us at T = 4) leaves the RAM bandwidth to the last members; the waiting ones spend it on their
// next phase. About what one core loads in the tail: a larger budget competes with the last
// gate/up slices for RAM (graph bench, 48 layers of 3-5 misses: 32 KiB -1.5 % per window vs none,
// 384 KiB no better than none). Results never change.
constexpr std::size_t kDownPrefetchBytes = std::size_t{32} << 10;

void pin_current_thread(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof set, &set) != 0) {
        throw std::runtime_error("host MoE pool: cannot pin a thread to cpu " +
                                 std::to_string(cpu));
    }
}

std::int64_t steady_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// A4: round to BF16 (RNE), as the GPU recipe does for gate/up and h.
inline float a4_bf16(float v) {
    std::uint32_t u;
    std::memcpy(&u, &v, 4);
    u = (u + 0x7FFFU + ((u >> 16) & 1U)) & 0xFFFF0000U;
    float r;
    std::memcpy(&r, &u, 4);
    return r;
}

bool read_first_int(const std::string& path, int& value) {
    std::ifstream f(path);
    if (!f) return false;
    f >> value;
    return static_cast<bool>(f);
}

// A sysfs cpu list ("0-3,8,10-11") as cpu ids; empty when the file is missing or malformed.
std::vector<int> read_cpu_list(const std::string& path) {
    std::ifstream f(path);
    std::string text;
    if (!f || !std::getline(f, text)) return {};
    try {
        return parse_cpu_list(text);
    } catch (const std::invalid_argument&) { return {}; }
}

struct HostCore {
    int cpu  = 0; // lowest allowed SMT sibling
    int l3   = 0;
    int node = 0;
};

// The allowed physical cores (process affinity), grouped by nothing, in cpu order.
// The affinity of the process as it was at the first call: HostCpuPool pins its calling thread to
// cpus[0], so reading the calling thread's mask later would see a single cpu (and a second pool, or
// a test that builds several, would find "fewer physical cores than requested threads").
const std::pair<bool, cpu_set_t>& process_mask() {
    static const std::pair<bool, cpu_set_t> mask = [] {
        cpu_set_t m;
        CPU_ZERO(&m);
        const bool ok = sched_getaffinity(0, sizeof(m), &m) == 0;
        return std::pair<bool, cpu_set_t>{ok, m};
    }();
    return mask;
}

std::vector<HostCore> allowed_cores(int* logical = nullptr) {
    auto allowed = [](int cpu) { return process_may_run_on(cpu); };
    std::vector<HostCore> cores;
    int n_logical = 0;
    for (int cpu = 0;; ++cpu) {
        const std::string base          = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
        const std::vector<int> siblings = read_cpu_list(base + "/topology/thread_siblings_list");
        if (siblings.empty()) break;
        if (!allowed(cpu)) continue;
        ++n_logical;
        int first = cpu;
        for (const int s : siblings) {
            if (allowed(s)) {
                first = s;
                break;
            }
        }
        if (first != cpu) continue; // an SMT sibling of a core already listed
        HostCore core;
        core.cpu = cpu;
        if (!read_first_int(base + "/cache/index3/id", core.l3)) core.l3 = 0;
        for (int node = 0; node < 64; ++node) {
            if (std::ifstream(base + "/node" + std::to_string(node) + "/cpulist")) {
                core.node = node;
                break;
            }
        }
        cores.push_back(core);
    }
    if (logical != nullptr) *logical = n_logical;
    return cores;
}

std::map<int, std::vector<int>> cores_by_l3() {
    std::map<int, std::vector<int>> by_l3; // L3 id -> allowed primary cpus, ascending
    for (const HostCore& core : allowed_cores()) by_l3[core.l3].push_back(core.cpu);
    return by_l3;
}

} // namespace

bool process_may_run_on(int cpu) {
    const auto& [have_mask, mask] = process_mask();
    return cpu >= 0 && (!have_mask || (cpu < CPU_SETSIZE && CPU_ISSET(cpu, &mask)));
}

std::vector<int> parse_cpu_list(std::string_view text) {
    const auto bad = [&] {
        return std::invalid_argument("not a cpu list (expected e.g. 0-7,16,18-19): \"" +
                                     std::string(text) + "\"");
    };
    const auto number = [&](std::string_view digits) {
        if (digits.empty() || digits.size() > 4) throw bad();
        int value = 0;
        for (const char c : digits) {
            if (c < '0' || c > '9') throw bad();
            value = value * 10 + (c - '0');
        }
        if (value >= CPU_SETSIZE) throw bad();
        return value;
    };
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.remove_suffix(1);
    if (text.empty()) throw bad();
    std::vector<int> out;
    for (std::size_t at = 0; at <= text.size();) {
        std::size_t end = text.find(',', at);
        if (end == std::string_view::npos) end = text.size();
        const std::string_view item = text.substr(at, end - at);
        const std::size_t dash      = item.find('-');
        if (dash == std::string_view::npos) {
            out.push_back(number(item));
        } else {
            const int lo = number(item.substr(0, dash)), hi = number(item.substr(dash + 1));
            if (hi < lo) throw bad();
            for (int c = lo; c <= hi; ++c) out.push_back(c);
        }
        at = end + 1;
    }
    return out;
}

HostCpuTopology host_cpu_topology() {
    HostCpuTopology t;
    const std::vector<HostCore> cores = allowed_cores(&t.logical_cpus);
    std::map<int, int> l3, node;
    for (const HostCore& c : cores) {
        ++l3[c.l3];
        ++node[c.node];
    }
    t.physical_cores = static_cast<std::int32_t>(cores.size());
    t.l3_domains     = static_cast<std::int32_t>(l3.size());
    t.numa_nodes     = static_cast<std::int32_t>(node.size());
    return t;
}

std::int32_t auto_expert_cpu_threads(const HostCpuTopology& topology, bool gguf,
                                     std::string* reason) {
    const std::int32_t cores = std::max<std::int32_t>(1, topology.physical_cores);
    std::int32_t n           = 0;
    std::string why;
    if (gguf) {
        // ggml block banks: the decode-once kernels are compute-bound below one thread per core
        // (9900X: 6 -> 12 threads = +4 % decode; 10 vs 12 within 3 %).
        n   = cores;
        why = "one per physical core (ggml block banks)";
    } else {
        // NVFP4: the dots are bound by the memory bandwidth (9900X: 68 GB/s at 6 threads,
        // per-expert cost flat from 6 to 12): half the physical cores, at least 6, and at least two
        // per L3 domain, since one core cannot fill a CCD's link to memory.
        n   = std::max({std::int32_t{6}, cores / 2, 2 * topology.l3_domains});
        why = "max(6, cores/2, 2 per L3 domain) (NVFP4 banks, memory-bound)";
    }
    n = std::min({n, cores, std::int32_t{64}});
    // The main thread spin-waits on the GPU (NINFER_CUDA_SYNC=spin) and is not pinned. With one
    // allowed logical CPU per core (a cpuset such as taskset -c 0-15, or SMT off) it has no free
    // sibling and shares a worker's core: 16 workers on a 16-CPU cpuset dropped decode from ~144
    // to 80-90 tok/s on an EPYC 7F52 (issue #11), so one core is left to it.
    const bool spare_core =
        cores > 1 && topology.logical_cpus <= topology.physical_cores && n >= cores;
    if (spare_core) n = cores - 1;
    if (reason != nullptr) {
        if (spare_core) why += ", one core left to the main thread (no free SMT sibling)";
        *reason = why + "; " + std::to_string(topology.physical_cores) + " physical cores, " +
                  std::to_string(topology.logical_cpus) + " logical CPUs allowed, " +
                  std::to_string(topology.l3_domains) + " L3 domains, " +
                  std::to_string(topology.numa_nodes) + " NUMA nodes";
        if (topology.numa_nodes > 1) *reason += " (bank pages are not NUMA-placed)";
    }
    return n;
}

std::vector<int> spread_cpus(std::int32_t count) {
    std::map<int, std::vector<int>> by_l3 = cores_by_l3();
    std::vector<int> out;
    for (std::size_t i = 0; out.size() < static_cast<std::size_t>(count); ++i) {
        bool any = false;
        for (auto& [id, cpus] : by_l3) {
            if (i < cpus.size() && out.size() < static_cast<std::size_t>(count)) {
                out.push_back(cpus[i]);
                any = true;
            }
        }
        if (!any) break;
    }
    if (out.size() < static_cast<std::size_t>(count)) {
        throw std::runtime_error("host MoE pool: fewer physical cores than requested threads");
    }
    return out;
}

std::vector<int> packed_cpus(std::int32_t count, std::int32_t first) {
    std::map<int, std::vector<int>> by_l3 = cores_by_l3();
    std::vector<std::vector<int>> domains;
    for (auto& [id, cpus] : by_l3) domains.push_back(cpus);
    std::vector<int> out;
    for (std::size_t d = 0; d < domains.size() && out.size() < static_cast<std::size_t>(count);
         ++d) {
        for (const int cpu : domains[(static_cast<std::size_t>(first) + d) % domains.size()]) {
            if (out.size() < static_cast<std::size_t>(count)) out.push_back(cpu);
        }
    }
    if (out.size() < static_cast<std::size_t>(count)) {
        throw std::runtime_error("host MoE pool: fewer physical cores than requested threads");
    }
    return out;
}

// ------------------------------------------------------------------ pool
HostCpuPool::HostCpuPool(std::span<const int> cpus, double spin_ms)
    : size_(static_cast<std::int32_t>(cpus.size())), spin_ms_(spin_ms) {
    if (cpus.empty()) throw std::invalid_argument("host MoE pool: no cpus");
    pin_current_thread(cpus[0]);
    threads_.reserve(cpus.size() - 1);
    for (std::size_t i = 1; i < cpus.size(); ++i) {
        threads_.emplace_back(
            [this, i, cpu = cpus[i]] { worker(static_cast<std::int32_t>(i), cpu); });
    }
}

HostCpuPool::~HostCpuPool() {
    stop_.store(true, std::memory_order_seq_cst);
    generation_.fetch_add(1, std::memory_order_seq_cst);
    generation_.notify_all();
    for (auto& t : threads_) t.join();
}

void HostCpuPool::worker(std::int32_t member, int cpu) {
    pin_current_thread(cpu);
    using clock = std::chrono::steady_clock;
    // Generation 0 is the constructor's: a run() may bump it before this thread gets here.
    std::uint32_t seen  = 0;
    const auto spin_for = std::chrono::duration<double, std::milli>(spin_ms_);
    for (;;) {
        std::uint32_t g = generation_.load(std::memory_order_acquire);
        if (g == seen) {
            const auto start = clock::now();
            for (std::uint32_t polls = 1; (g = generation_.load(std::memory_order_acquire)) == seen;
                 ++polls) {
                if ((polls & 1023U) == 0 && clock::now() - start > spin_for) {
                    generation_.wait(seen, std::memory_order_acquire);
                } else {
                    _mm_pause();
                }
            }
        }
        seen = g;
        if (stop_.load(std::memory_order_acquire)) return;
        body_(context_, member);
        finished_.fetch_add(1, std::memory_order_acq_rel);
    }
}

void HostCpuPool::run(Body body, void* context) {
    body_    = body;
    context_ = context;
    finished_.store(0, std::memory_order_relaxed);
    arrived_.store(0, std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_seq_cst);
    // Sleeping members only: a waiter registers before re-reading the generation.
    generation_.notify_all();
    body(context, 0);
    while (finished_.load(std::memory_order_acquire) != size_ - 1) _mm_pause();
}

void HostCpuPool::arrive() noexcept { arrived_.fetch_add(1, std::memory_order_acq_rel); }

bool HostCpuPool::released(std::int32_t index) const noexcept {
    return arrived_.load(std::memory_order_acquire) >= size_ * index;
}

void HostCpuPool::barrier(std::int32_t index) noexcept {
    arrive();
    while (!released(index)) _mm_pause();
}

// ------------------------------------------------------------------ executor
HostMoeExecutor::HostMoeExecutor(HostCpuPool& pool, std::int32_t hidden, std::int32_t intermediate,
                                 std::int32_t max_tokens, std::int32_t max_top_k, Options options)
    : pool_(pool), hidden_(hidden), intermediate_(intermediate), max_tokens_(max_tokens),
      max_top_k_(max_top_k), options_(options), kernels_(host_nvfp4_kernels(options.isa)),
      gguf_kernels_(gguf::host_gguf_kernels(gguf::host_gguf_isa(kernels_.isa))) {
    if (hidden <= 0 || hidden % kNvfp4RowTile || intermediate <= 0 ||
        intermediate % kNvfp4RowTile || max_tokens < 1 || max_top_k < 1) {
        throw std::invalid_argument("host MoE executor: unsupported geometry");
    }
    const std::int32_t max_pairs = max_tokens * max_top_k;
    unique_.resize(static_cast<std::size_t>(max_pairs));
    pair_begin_.resize(static_cast<std::size_t>(max_pairs) + 1);
    pair_token_.resize(static_cast<std::size_t>(max_pairs));
    pair_alpha_.resize(static_cast<std::size_t>(max_pairs));
    used_tokens_.reserve(static_cast<std::size_t>(max_tokens));
    if (!options.gguf_only) {
        x_act_ = W4A8ActivationBuffer(hidden, max_tokens);
        h_act_ = W4A8ActivationBuffer(intermediate, max_pairs);
    }
    const auto n = static_cast<std::size_t>(pool.size());
    acc_.assign(n,
                std::vector<float>(static_cast<std::size_t>(std::max(max_tokens, kHostMaxTokens)) *
                                   kMaxItemRows));
    for (std::size_t m = 0; m < n; ++m) member_x_.emplace_back(hidden, kHostMaxTokens);
    members_.resize(n);
    gate_cursor_ = std::make_unique<Cursor[]>(n);
    down_cursor_ = std::make_unique<Cursor[]>(n);
}

std::int64_t HostMoeExecutor::now_rel() const noexcept { return steady_ns() - start_ns_; }

HostMoeCallStats HostMoeExecutor::run(const HostExpertLayer& layer, const HostMoeJob& job) {
    start_ns_            = steady_ns();
    const std::int32_t T = job.tokens;
    if (T < 1 || T > max_tokens_ || job.top_k < 1 || job.top_k > max_top_k_ || !job.x || !job.ids ||
        !job.alpha || !job.out) {
        throw std::invalid_argument("host MoE executor: invalid job");
    }
    const bool gguf_layer = layer.format == HostExpertFormat::Gguf;
    if (layer.hidden != hidden_ || layer.intermediate != intermediate_ ||
        (!gguf_layer && (layer.gate_up.k != hidden_ || layer.down.k != intermediate_ ||
                         layer.gate_up.rows != std::int64_t{layer.experts} * 2 * intermediate_ ||
                         layer.down.rows != std::int64_t{layer.experts} * hidden_))) {
        throw std::invalid_argument("host MoE executor: layer geometry mismatch");
    }
    if (!gguf_layer && options_.gguf_only) {
        throw std::invalid_argument("host MoE executor: an NVFP4 layer on a GGUF-only executor");
    }
    if (expert_slot_.size() < static_cast<std::size_t>(layer.experts)) {
        expert_slot_.assign(static_cast<std::size_t>(layer.experts), -1);
    }
    // Experts in ascending id with their pair counts, then the pairs grouped by expert (tokens
    // ascending within an expert).
    std::int32_t U = 0, pairs = 0;
    used_tokens_.clear();
    for (std::int32_t t = 0; t < T; ++t) {
        bool used = false;
        for (std::int32_t s = 0; s < job.top_k; ++s) {
            const std::int32_t e = job.ids[t * job.top_k + s];
            if (e < 0) continue;
            if (e >= layer.experts) {
                for (std::int32_t u = 0; u < U; ++u)
                    expert_slot_[static_cast<std::size_t>(unique_[u])] = -1;
                throw std::invalid_argument("host MoE executor: expert id out of range");
            }
            std::int32_t& u = expert_slot_[static_cast<std::size_t>(e)];
            if (u < 0) {
                u                                        = U;
                unique_[static_cast<std::size_t>(U)]     = e;
                pair_begin_[static_cast<std::size_t>(U)] = 0;
                ++U;
            }
            ++pair_begin_[static_cast<std::size_t>(u)];
            ++pairs;
            used = true;
        }
        if (used) used_tokens_.push_back(t);
    }
    // Ascending expert id: every token sums its experts in the same order whatever the other
    // tokens of the call route to (an MTP verification window and the same token decoded alone
    // then produce identical bits).
    std::sort(unique_.data(), unique_.data() + U);
    for (std::int32_t u = 0; u < U; ++u) {
        expert_slot_[static_cast<std::size_t>(unique_[static_cast<std::size_t>(u)])] = u;
        pair_begin_[static_cast<std::size_t>(u)]                                     = 0;
    }
    for (std::int32_t t = 0; t < T; ++t) {
        for (std::int32_t s = 0; s < job.top_k; ++s) {
            const std::int32_t e = job.ids[t * job.top_k + s];
            if (e >= 0) {
                ++pair_begin_[static_cast<std::size_t>(expert_slot_[static_cast<std::size_t>(e)])];
            }
        }
    }
    unique_count_ = U;
    if (U == 0) {
        std::memset(job.out, 0, sizeof(float) * static_cast<std::size_t>(T) * hidden_);
        return {};
    }
    std::int32_t offset = 0;
    for (std::int32_t u = 0; u < U; ++u) {
        const std::int32_t n                     = pair_begin_[static_cast<std::size_t>(u)];
        pair_begin_[static_cast<std::size_t>(u)] = offset;
        offset += n;
    }
    pair_begin_[static_cast<std::size_t>(U)] = offset;
    {
        // Fill: a running cursor per expert (pair_begin_ shifted back afterwards).
        for (std::int32_t t = 0; t < T; ++t) {
            for (std::int32_t s = 0; s < job.top_k; ++s) {
                const std::int32_t e = job.ids[t * job.top_k + s];
                if (e < 0) continue;
                const std::int32_t u = expert_slot_[static_cast<std::size_t>(e)];
                std::int32_t& cursor = pair_begin_[static_cast<std::size_t>(u)];
                pair_token_[static_cast<std::size_t>(cursor)] = t;
                pair_alpha_[static_cast<std::size_t>(cursor)] = job.alpha[t * job.top_k + s];
                ++cursor;
            }
        }
        for (std::int32_t u = U; u > 0; --u)
            pair_begin_[static_cast<std::size_t>(u)] = pair_begin_[static_cast<std::size_t>(u - 1)];
        pair_begin_[0] = 0;
    }
    for (std::int32_t u = 0; u < U; ++u) expert_slot_[static_cast<std::size_t>(unique_[u])] = -1;
    if (gguf_layer) {
        layer_ = &layer;
        job_   = &job;
        HostMoeCallStats gs{U, pairs, {}};
        gs.dispatch_ns = now_rel();
        gguf_call();
        gs.end_ns = now_rel();
        for (int i = 0; i < 3; ++i) gs.phase_ns[i] = phase_ns_[i] - start_ns_;
        if (options_.record_members) {
            gs.members      = members_.data();
            gs.member_count = pool_.size();
        }
        return gs;
    }
    // Wide calls (prefill) claim single slices from shared counters: many more slices than threads
    // anyway, and the down phase holds T x 16 rows of sums per thread.
    static_ = T <= kHostMaxTokens;
    if (static_) {
        const std::int32_t n  = pool_.size();
        const std::int32_t Ng = U * (intermediate_ / kSliceRows), Nd = hidden_ / kSliceRows;
        for (std::int32_t m = 0; m < n; ++m) {
            Cursor& g = gate_cursor_[static_cast<std::size_t>(m)];
            Cursor& d = down_cursor_[static_cast<std::size_t>(m)];
            g.next.store(static_cast<std::int32_t>(std::int64_t{Ng} * m / n),
                         std::memory_order_relaxed);
            g.end = static_cast<std::int32_t>(std::int64_t{Ng} * (m + 1) / n);
            d.next.store(Nd * m / n, std::memory_order_relaxed);
            d.end = Nd * (m + 1) / n;
        }
    }
    layer_ = &layer;
    job_   = &job;
    next_token_.store(0, std::memory_order_relaxed);
    next_gate_.store(0, std::memory_order_relaxed);
    next_down_.store(0, std::memory_order_relaxed);
    HostMoeCallStats st{U, pairs, {}};
    st.dispatch_ns = now_rel();
    pool_.run(&HostMoeExecutor::body, this);
    st.end_ns = now_rel();
    for (int i = 0; i < 3; ++i) st.phase_ns[i] = phase_ns_[i] - start_ns_;
    if (options_.record_members) {
        st.members      = members_.data();
        st.member_count = pool_.size();
    }
    return st;
}

std::int32_t HostMoeExecutor::intermediate_count() const noexcept {
    return unique_count_ > 0 ? pair_begin_[static_cast<std::size_t>(unique_count_)] : 0;
}

HostMoeIntermediate HostMoeExecutor::intermediate(std::int32_t i) const {
    if (i < 0 || i >= intermediate_count())
        throw std::out_of_range("host MoE executor: intermediate index");
    std::int32_t u = 0;
    while (pair_begin_[static_cast<std::size_t>(u) + 1] <= i) ++u;
    return {unique_[static_cast<std::size_t>(u)], pair_token_[static_cast<std::size_t>(i)],
            h_act_.at(i)};
}

void HostMoeExecutor::body(void* self, std::int32_t member) {
    auto& ex = *static_cast<HostMoeExecutor*>(self);
    ex.stamp(member, 0);
    if (ex.static_) {
        // No shared token phase: each member quantizes what its own slices read.
        ex.stamp(member, 1);
        ex.stamp(member, 2);
        if (member == 0) ex.phase_ns_[0] = steady_ns();
        ex.static_gate_up(member);
        ex.stamp(member, 3);
        ex.await_down(member);
    } else {
        ex.phase_tokens(member);
        ex.stamp(member, 1);
        ex.pool_.barrier(1);
        ex.stamp(member, 2);
        if (member == 0) ex.phase_ns_[0] = steady_ns();
        ex.phase_gate_up();
        ex.stamp(member, 3);
        ex.pool_.barrier(2);
    }
    ex.stamp(member, 4);
    if (member == 0) ex.phase_ns_[1] = steady_ns();
    if (ex.static_) {
        ex.static_down(member);
    } else {
        ex.phase_down(member);
    }
    ex.stamp(member, 5);
    if (member == 0) ex.phase_ns_[2] = steady_ns();
}

void HostMoeExecutor::await_down(std::int32_t member) {
    pool_.arrive();
    // The member's own down slices [next, end) (nobody claims down work before the barrier
    // opens), in static_down's reading order: runs of up to 4 slices, every host expert in order
    // within a run: its rows' codes (contiguous), then the 128-row scale tiles holding them.
    constexpr std::size_t kLine = 64;
    constexpr int kBurst        = 4; // lines between barrier checks
    const Cursor& cur           = down_cursor_[static_cast<std::size_t>(member)];
    const Nvfp4Plane& down      = layer_->down;
    const std::int32_t H        = hidden_;
    const std::int32_t first    = cur.next.load(std::memory_order_relaxed);
    const std::size_t row_bytes = static_cast<std::size_t>(down.k / 2);
    const std::size_t tile_bytes =
        static_cast<std::size_t>(down.k / kNvfp4Quad) * static_cast<std::size_t>(kScaleQuadStride);
    std::size_t issued = 0;
    for (std::int32_t i = first; i < cur.end && issued < kDownPrefetchBytes;
         i += kMaxItemRows / kSliceRows) {
        const std::int32_t rows = std::min(kMaxItemRows / kSliceRows, cur.end - i) * kSliceRows;
        for (std::int32_t u = 0; u < unique_count_ && issued < kDownPrefetchBytes; ++u) {
            const std::int64_t r0 =
                std::int64_t{unique_[static_cast<std::size_t>(u)]} * H + i * kSliceRows;
            const auto* codes      = reinterpret_cast<const char*>(down.row_codes(r0));
            const std::size_t span = static_cast<std::size_t>(rows) * row_bytes;
            const auto* scales     = reinterpret_cast<const char*>(down.scales) +
                                 static_cast<std::size_t>(r0 / kNvfp4RowTile) * tile_bytes;
            const std::size_t tiles =
                static_cast<std::size_t>((r0 + rows - 1) / kNvfp4RowTile - r0 / kNvfp4RowTile + 1);
            const std::array<std::pair<const char*, std::size_t>, 2> ranges{
                {{codes, span}, {scales, tiles * tile_bytes}}};
            for (const auto& [base, bytes] : ranges) {
                for (std::size_t off = 0; off < bytes; off += kLine * kBurst) {
                    if (pool_.released(1)) return;
                    const std::size_t stop = std::min(bytes, off + kLine * kBurst);
                    for (std::size_t o = off; o < stop; o += kLine) {
                        _mm_prefetch(base + o, _MM_HINT_T1);
                    }
                    issued += stop - off;
                }
            }
        }
    }
    while (!pool_.released(1)) _mm_pause();
}

// Static gate/up: the call's gate/up work is N = U * I/16 slices (expert order, rows ascending);
// a slice = 16 gate rows + the same 16 up rows + SwiGLU + one int8 block of h. Member m owns
// [N m / n, N (m+1) / n) and claims it front to back in runs of up to 4 slices (64 consecutive
// rows, one atomic on its own cache line per run); a member that has finished its range takes
// single slices from the front of the others' remaining ranges (absorbs per-core bandwidth
// differences). Each member quantizes the input of the tokens its own slices need (same codec,
// private buffer), so no shared token phase and barrier precede gate/up.
void HostMoeExecutor::static_gate_up(std::int32_t member) {
    const std::int32_t I   = intermediate_;
    const std::int32_t per = I / kSliceRows;
    const std::int32_t n   = pool_.size();
    alignas(64) float gate[kHostMaxTokens * kMaxItemRows];
    alignas(64) float up[kHostMaxTokens * kMaxItemRows];
    alignas(64) float h[kMaxItemRows];
    W4A8Activation acts[kHostMaxTokens];
    std::uint32_t quantized = 0; // tokens already in member_x_[member]
    // Slices [i, i + c) of one expert.
    auto slices = [&](std::int32_t i, std::int32_t c) {
        const std::int32_t u     = i / per;
        const std::int32_t rows  = c * kSliceRows;
        const std::int32_t r     = (i % per) * kSliceRows;
        const std::int64_t e     = unique_[static_cast<std::size_t>(u)];
        const std::int32_t first = pair_begin_[static_cast<std::size_t>(u)];
        const std::int32_t last  = pair_begin_[static_cast<std::size_t>(u) + 1];
        for (std::int32_t g0 = first; g0 < last; g0 += kHostMaxTokens) {
            const std::int32_t m = std::min(kHostMaxTokens, last - g0);
            for (std::int32_t k = 0; k < m; ++k) {
                const std::int32_t t   = pair_token_[static_cast<std::size_t>(g0 + k)];
                const W4A8Activation a = member_x_[static_cast<std::size_t>(member)].at(t);
                if ((quantized >> t & 1U) == 0) {
                    if (options_.a4) {
                        kernels_.quantize_bf16_a4(job_->x + static_cast<std::size_t>(t) * hidden_,
                                                  hidden_, layer_->gate_up_input_divisor, a);
                    } else
                        kernels_.quantize_bf16(job_->x + static_cast<std::size_t>(t) * hidden_,
                                               hidden_, a);
                    quantized |= 1U << t;
                }
                acts[k] = a;
            }
            kernels_.rows_dot(layer_->gate_up, e * 2 * I + r, rows, acts, m, gate, rows);
            kernels_.rows_dot(layer_->gate_up, e * 2 * I + I + r, rows, acts, m, up, rows);
            for (std::int32_t k = 0; k < m; ++k) {
                if (options_.a4) {
                    for (std::int32_t j = 0; j < rows; ++j) {
                        const float g = a4_bf16(gate[k * rows + j]), u = a4_bf16(up[k * rows + j]);
                        h[j] = a4_bf16(g * (1.0F / (1.0F + std::exp(-g))) * u);
                    }
                    kernels_.quantize_blocks_a4(h, r / kNvfp4Group, c, layer_->down_input_divisor,
                                                h_act_.at(g0 + k));
                    continue;
                }
                for (std::int32_t j = 0; j < rows; ++j) {
                    const float g = gate[k * rows + j];
                    h[j]          = g / (1.0F + std::exp(-g)) * up[k * rows + j];
                }
                kernels_.quantize_blocks(h, r / kNvfp4Group, c, h_act_.at(g0 + k));
            }
        }
    };
    auto claim_run = [&](Cursor& cur, std::int32_t want) {
        const std::int32_t i = cur.next.fetch_add(want, std::memory_order_relaxed);
        if (i >= cur.end) return false;
        for (std::int32_t j = i, c = std::min(want, cur.end - i); c > 0;) {
            const std::int32_t cc = std::min(c, per - j % per); // never across an expert boundary
            slices(j, cc);
            j += cc, c -= cc;
        }
        return true;
    };
    while (claim_run(gate_cursor_[static_cast<std::size_t>(member)], kMaxItemRows / kSliceRows)) {}
    for (std::int32_t k = 1; k < n; ++k) {
        Cursor& victim = gate_cursor_[static_cast<std::size_t>((member + k) % n)];
        while (claim_run(victim, 1)) {}
    }
}

// Static down: N = H/16 output slices; member m owns [N m / n, N (m+1) / n) (claimed in runs of up
// to 4 slices, stealing as in static_gate_up). A run's outputs are the weighted sum of every host
// expert's down rows in expert order (the order of phase_down), written for all T tokens.
void HostMoeExecutor::static_down(std::int32_t member) {
    const std::int32_t H = hidden_;
    const std::int32_t T = job_->tokens;
    const std::int32_t n = pool_.size();
    alignas(64) float partial[kHostMaxTokens * kMaxItemRows];
    float* acc = acc_[static_cast<std::size_t>(member)].data(); // [T][rows]
    W4A8Activation acts[kHostMaxTokens];
    auto claim_run = [&](Cursor& cur, std::int32_t want) {
        const std::int32_t i = cur.next.fetch_add(want, std::memory_order_relaxed);
        if (i >= cur.end) return false;
        const std::int32_t r    = i * kSliceRows;
        const std::int32_t rows = std::min(want, cur.end - i) * kSliceRows;
        std::fill(acc, acc + static_cast<std::size_t>(T) * rows, 0.0F);
        for (std::int32_t u = 0; u < unique_count_; ++u) {
            const std::int64_t e     = unique_[static_cast<std::size_t>(u)];
            const std::int32_t first = pair_begin_[static_cast<std::size_t>(u)];
            const std::int32_t last  = pair_begin_[static_cast<std::size_t>(u) + 1];
            for (std::int32_t g0 = first; g0 < last; g0 += kHostMaxTokens) {
                const std::int32_t m = std::min(kHostMaxTokens, last - g0);
                for (std::int32_t k = 0; k < m; ++k) acts[k] = h_act_.at(g0 + k);
                kernels_.rows_dot(layer_->down, e * H + r, rows, acts, m, partial, rows);
                for (std::int32_t k = 0; k < m; ++k) {
                    const std::size_t p = static_cast<std::size_t>(g0 + k);
                    const float a       = pair_alpha_[p];
                    float* dst          = acc + static_cast<std::size_t>(pair_token_[p]) * rows;
                    for (std::int32_t j = 0; j < rows; ++j) dst[j] += a * partial[k * rows + j];
                }
            }
        }
        for (std::int32_t t = 0; t < T; ++t) {
            std::memcpy(job_->out + static_cast<std::size_t>(t) * H + r,
                        acc + static_cast<std::size_t>(t) * rows,
                        sizeof(float) * static_cast<std::size_t>(rows));
        }
        return true;
    };
    while (claim_run(down_cursor_[static_cast<std::size_t>(member)], kMaxItemRows / kSliceRows)) {}
    for (std::int32_t k = 1; k < n; ++k) {
        Cursor& victim = down_cursor_[static_cast<std::size_t>((member + k) % n)];
        while (claim_run(victim, 1)) {}
    }
}

void HostMoeExecutor::phase_tokens(std::int32_t) {
    const auto n = static_cast<std::int32_t>(used_tokens_.size());
    for (std::int32_t i; (i = next_token_.fetch_add(1, std::memory_order_relaxed)) < n;) {
        const std::int32_t t = used_tokens_[static_cast<std::size_t>(i)];
        if (options_.a4) {
            kernels_.quantize_bf16_a4(job_->x + static_cast<std::size_t>(t) * hidden_, hidden_,
                                      layer_->gate_up_input_divisor, x_act_.at(t));
        } else
            kernels_.quantize_bf16(job_->x + static_cast<std::size_t>(t) * hidden_, hidden_,
                                   x_act_.at(t));
    }
}

// Dynamic gate/up: the call's U * I/16 slices (expert order, rows ascending) claimed one at a time
// from a shared counter; a slice = 16 gate rows + the same 16 up rows + SwiGLU + one int8 block of
// h.
void HostMoeExecutor::phase_gate_up() {
    const std::int32_t I     = intermediate_;
    const std::int32_t per   = I / kSliceRows;
    const std::int32_t items = unique_count_ * per;
    alignas(64) float gate[kHostMaxTokens * kSliceRows];
    alignas(64) float up[kHostMaxTokens * kSliceRows];
    alignas(64) float h[kSliceRows];
    W4A8Activation acts[kHostMaxTokens];
    for (std::int32_t item; (item = next_gate_.fetch_add(1, std::memory_order_relaxed)) < items;) {
        const std::int32_t u     = item / per;
        const std::int32_t base  = (item % per) * kSliceRows;
        const std::int64_t e     = unique_[static_cast<std::size_t>(u)];
        const std::int32_t first = pair_begin_[static_cast<std::size_t>(u)];
        const std::int32_t last  = pair_begin_[static_cast<std::size_t>(u) + 1];
        for (std::int32_t g0 = first; g0 < last; g0 += kHostMaxTokens) {
            const std::int32_t m = std::min(kHostMaxTokens, last - g0);
            for (std::int32_t n = 0; n < m; ++n)
                acts[n] = x_act_.at(pair_token_[static_cast<std::size_t>(g0 + n)]);
            kernels_.rows_dot(layer_->gate_up, e * 2 * I + base, kSliceRows, acts, m, gate,
                              kSliceRows);
            kernels_.rows_dot(layer_->gate_up, e * 2 * I + I + base, kSliceRows, acts, m, up,
                              kSliceRows);
            for (std::int32_t n = 0; n < m; ++n) {
                for (std::int32_t r = 0; r < kSliceRows; ++r) {
                    const float g = gate[n * kSliceRows + r];
                    if (options_.a4) {
                        const float gb = a4_bf16(g), ub = a4_bf16(up[n * kSliceRows + r]);
                        h[r] = a4_bf16(gb * (1.0F / (1.0F + std::exp(-gb))) * ub);
                    } else
                        h[r] = g / (1.0F + std::exp(-g)) * up[n * kSliceRows + r];
                }
                const W4A8Activation dst = h_act_.at(g0 + n);
                if (options_.a4) {
                    kernels_.quantize_blocks_a4(h, base / kNvfp4Group, 1,
                                                layer_->down_input_divisor, dst);
                } else
                    kernels_.quantize_blocks(h, base / kNvfp4Group, 1, dst);
            }
        }
    }
}

// Dynamic down: the H/16 output slices claimed one at a time from a shared counter; a slice's
// outputs are the weighted sum of every host expert's down rows in expert order (the order of
// static_down), written for all T tokens.
void HostMoeExecutor::phase_down(std::int32_t member) {
    const std::int32_t H     = hidden_;
    const std::int32_t T     = job_->tokens;
    const std::int32_t items = H / kSliceRows;
    alignas(64) float partial[kHostMaxTokens * kSliceRows];
    float* acc = acc_[static_cast<std::size_t>(member)].data(); // [T][kSliceRows]
    W4A8Activation acts[kHostMaxTokens];
    for (std::int32_t item; (item = next_down_.fetch_add(1, std::memory_order_relaxed)) < items;) {
        const std::int32_t base = item * kSliceRows;
        std::fill(acc, acc + static_cast<std::size_t>(T) * kSliceRows, 0.0F);
        for (std::int32_t u = 0; u < unique_count_; ++u) {
            const std::int64_t e     = unique_[static_cast<std::size_t>(u)];
            const std::int32_t first = pair_begin_[static_cast<std::size_t>(u)];
            const std::int32_t last  = pair_begin_[static_cast<std::size_t>(u) + 1];
            for (std::int32_t g0 = first; g0 < last; g0 += kHostMaxTokens) {
                const std::int32_t m = std::min(kHostMaxTokens, last - g0);
                for (std::int32_t n = 0; n < m; ++n) acts[n] = h_act_.at(g0 + n);
                kernels_.rows_dot(layer_->down, e * H + base, kSliceRows, acts, m, partial,
                                  kSliceRows);
                for (std::int32_t n = 0; n < m; ++n) {
                    const std::size_t i = static_cast<std::size_t>(g0 + n);
                    const float a       = pair_alpha_[i];
                    float* dst = acc + static_cast<std::size_t>(pair_token_[i]) * kSliceRows;
                    for (std::int32_t r = 0; r < kSliceRows; ++r)
                        dst[r] += a * partial[n * kSliceRows + r];
                }
            }
        }
        for (std::int32_t t = 0; t < T; ++t) {
            std::memcpy(job_->out + static_cast<std::size_t>(t) * H + base,
                        acc + static_cast<std::size_t>(t) * kSliceRows, sizeof(float) * kSliceRows);
        }
    }
}

} // namespace ninfer::ops::host
