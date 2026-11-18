#pragma once

// Weight-major host executor for the routed experts a layer could not find on the GPU.
//
// One call computes, for T <= max_tokens token positions (decode windows T <= 8, prefill chunks
// up to the chunk length),
//   out[t] = sum over host experts e routed by t of alpha[t,e] * W_down[e] (SiLU(W_gate[e] x_t) *
//   (W_up[e] x_t))
// with NVFP4 weights read in place from the pinned host banks (block_scale_k16_m128x4_v1, the
// Qwen4Exp bank layout of models::qwen3_5::routed_expert_spans) and W4A8 activations: x_t and the
// SwiGLU output are quantized to int8 per 16 elements (host_nvfp4.h). Each expert is read once
// per call whatever its number of tokens (weight-major):
//   phase 1  slices of 16 intermediate rows of one expert: gate and up rows of all the expert's
//            tokens, SwiGLU in FP32, int8 quantization of the slice's 16-block
//   phase 2  slices of 16 output rows: down rows of every host expert, weighted sum over experts
//            in ascending id order, written for all T tokens (zeros where no host expert)
// An expert's tokens are taken in groups of up to kHostMaxTokens per dot call (one group in
// decode windows). Two work distributions:
//   Static   decode-size calls (T <= kHostMaxTokens): member m owns the contiguous slices
//            [N m / n, N (m+1) / n) of each phase (gate/up: N = U * I/16 in expert order; down:
//            N = H/16) and claims them in runs of 4 from a counter on its own cache line, then
//            takes single slices from the others' remaining ranges; each member quantizes the
//            input of its own slices (one barrier per call), and prefetches its down range while
//            it waits at that barrier.
//   Dynamic  wider calls (prefill): a shared token phase, then single slices claimed from a shared
//            counter, ordered so that consecutive slices read the same swizzled scale cache lines.
// The output is deterministic and independent of the thread count and the work distribution.
//
// Port of ninfer-offload's HostMoeExecutor (src/ops/sparse_moe/host/host_moe.{h,cpp} @1fd5c39,
// Apache-2.0, 1872183316/ninfer-offload) to NVFP4 W4A8 with a pinned spin pool.

#include "ops/host_moe/host_gguf.h"
#include "ops/host_moe/host_nvfp4.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::ops::host {

enum class HostExpertFormat : std::uint8_t { Nvfp4, Gguf };

// A layer bank of ggml blocks (artifact layout gguf_blocks_v1): gate_up [E*2I] rows of H values
// (expert-major: gate rows, then up rows), down [E*H] rows of I values, whole blocks per row.
struct HostGgufBank {
    gguf::WeightFormat gate_up       = gguf::WeightFormat::IQ3_XXS;
    gguf::WeightFormat down          = gguf::WeightFormat::IQ4_NL;
    const std::uint8_t* gate_up_rows = nullptr;
    const std::uint8_t* down_rows    = nullptr;
    std::size_t gate_up_row_bytes    = 0;
    std::size_t down_row_bytes       = 0;
    // Exclusive tier: expert id -> its host slot in the rows; null = the id itself. An entry a
    // call reads does not change during it (ExpertHostPool).
    const std::int32_t* host_slots = nullptr;
};

// Routed experts of one MoE layer in host memory. Expert e owns gate rows [e*2I, e*2I+I), up rows
// [e*2I+I, (e+1)*2I) of gate_up [E*2I, H] and rows [e*H, (e+1)*H) of down [E*H, I]: NVFP4 planes
// (format Nvfp4) or ggml block rows (format Gguf, `gguf`).
//
// Gguf layers: decode-once kernels of host_gguf.h (AVX-512 or AVX2, the same bits;
// gguf::host_gguf_isa(Options::isa)); x and h quantized per
// 32 values with an FP32 scale (quantize_pack_*_q8f32, the GPU GGUF Ops' codec), h = silu(g) * u in
// FP32; three dynamic phases (tokens, gate/up in (expert, 32-row) items, down in 32-row output
// items summing every host expert in ascending id order), so the output does not depend on the
// thread count or the item order. Options::a4 does not apply.
struct HostExpertLayer {
    Nvfp4Plane gate_up;
    Nvfp4Plane down;
    std::int32_t experts      = 0;
    std::int32_t hidden       = 0;
    std::int32_t intermediate = 0;
    // The layer's NVFP4 activation divisors of x and h (Options::a4).
    float gate_up_input_divisor = 0.0F;
    float down_input_divisor    = 0.0F;
    // Last, so the NVFP4 aggregate initializations keep their order.
    HostExpertFormat format = HostExpertFormat::Nvfp4;
    HostGgufBank gguf;
};

struct HostMoeJob {
    std::int32_t tokens     = 0;
    std::int32_t top_k      = 0;
    const std::uint16_t* x  = nullptr; // [T][H] BF16 bits
    const std::int32_t* ids = nullptr; // [T][top_k] expert id, or negative = computed elsewhere
    const float* alpha      = nullptr; // [T][top_k] routing weight
    float* out              = nullptr; // [T][H] overwritten
};

// One (expert, token) intermediate of the last call: the int8 SwiGLU output (stored order, see
// W4A8Activation) and its block scales, the semantic quantization boundary between the two phases.
struct HostMoeIntermediate {
    std::int32_t expert = 0;
    std::int32_t token  = 0;
    W4A8Activation act;
};

// Steady-clock ns of one pool member during a call, from the start of run(): body entered (woken),
// token phase done, first barrier passed, gate/up done, second barrier passed, down done. In the
// Static distribution the token phase is part of gate/up (stamps 1 and 2 equal stamp 0).
struct alignas(64) HostMoeMemberTimes {
    std::int64_t ns[6] = {0, 0, 0, 0, 0, 0};
};

struct HostMoeCallStats {
    std::int32_t experts = 0; // distinct host experts
    std::int32_t pairs   = 0; // (token, host expert) pairs
    // Steady-clock ns seen by member 0: end of the token phase, of gate/up, of down (from the
    // start of run()).
    std::int64_t phase_ns[3] = {0, 0, 0};
    // From the start of run(): the pool dispatched (routing parsed), the last member finished.
    std::int64_t dispatch_ns = 0;
    std::int64_t end_ns      = 0;
    // record_members: [members] per-member stamps, owned by the executor until the next run().
    const HostMoeMemberTimes* members = nullptr;
    std::int32_t member_count         = 0;
};

// The cores this process may run on (sched_getaffinity), from Linux sysfs: an SMT group counts
// once, represented by its lowest allowed sibling.
struct HostCpuTopology {
    std::int32_t logical_cpus   = 0; // allowed logical CPUs
    std::int32_t physical_cores = 0; // allowed physical cores
    std::int32_t l3_domains     = 0; // distinct L3 caches (CCDs) among them
    std::int32_t numa_nodes     = 0; // distinct NUMA nodes among them
};

[[nodiscard]] HostCpuTopology host_cpu_topology();

// Default number of CPU threads for the missing routed experts when the caller did not set one
// (RoutedExpertOptions cpu_threads / gguf_cpu_threads = 0). `gguf`: ggml block expert banks.
// `reason` (optional) gets a one-line explanation for the startup log.
[[nodiscard]] std::int32_t auto_expert_cpu_threads(const HostCpuTopology& topology, bool gguf,
                                                   std::string* reason = nullptr);

// A Linux cpu list as taskset and sysfs write it ("0-7,16,18-19") as cpu ids, in order. Throws
// std::invalid_argument on anything else (empty items, negative or reversed ranges, ids >= 1024).
[[nodiscard]] std::vector<int> parse_cpu_list(std::string_view text);
// Whether the process affinity (taskset, cgroup cpuset) as read at its first use includes `cpu`.
[[nodiscard]] bool process_may_run_on(int cpu);
// Logical CPUs of the big cores of a hybrid CPU (Intel P-cores: /sys/devices/cpu_core/cpus), read
// once; empty when the CPU is not hybrid or the file is missing.
[[nodiscard]] const std::vector<int>& big_core_cpus();
// 1 for each of `cpus` (pool members in order) that is in `big`, else 0.
[[nodiscard]] std::vector<std::uint8_t> big_core_members(std::span<const int> cpus,
                                                         std::span<const int> big);

// Physical cores spread over the L3 domains (CCDs) in round-robin order, lowest allowed SMT
// sibling of each core, from Linux sysfs. On the 9900X: 0, 6, 1, 7, 2, 8, ...
[[nodiscard]] std::vector<int> spread_cpus(std::int32_t count);
// Physical cores filling one L3 domain (CCD) before the next, starting with domain `first`
// (index in ascending L3 id order): 0, 1, 2, 3, 4, 5, 6, ... on the 9900X with first = 0.
[[nodiscard]] std::vector<int> packed_cpus(std::int32_t count, std::int32_t first = 0);

// Persistent pinned pool. The thread that constructs the pool is member 0 (pinned to cpus[0]) and
// must be the only caller of run(). Idle members spin with pause for `spin_ms`, then sleep on a
// futex until the next run().
class HostCpuPool {
public:
    explicit HostCpuPool(std::span<const int> cpus, double spin_ms = 20.0);
    ~HostCpuPool();
    HostCpuPool(const HostCpuPool&)            = delete;
    HostCpuPool& operator=(const HostCpuPool&) = delete;

    [[nodiscard]] std::int32_t size() const noexcept { return size_; }

    // The cpu of each member (member i is pinned to cpus()[i]).
    [[nodiscard]] std::span<const int> cpus() const noexcept { return cpus_; }

    // Runs body(member, pool) on every member including the caller; returns when all finished.
    using Body = void (*)(void* context, std::int32_t member);
    void run(Body body, void* context);

    // Barrier among the members of the current run(); `index` counts the barriers already passed
    // by the calling member in this run, starting at 1. barrier() = arrive() then spin until
    // released(index): a member may do useful work between the two (prefetching).
    void barrier(std::int32_t index) noexcept;
    void arrive() noexcept;
    [[nodiscard]] bool released(std::int32_t index) const noexcept;

private:
    void worker(std::int32_t member, int cpu);

    std::int32_t size_ = 1;
    double spin_ms_    = 20.0;
    std::vector<int> cpus_;
    std::vector<std::thread> threads_;
    Body body_     = nullptr;
    void* context_ = nullptr;
    alignas(64) std::atomic<std::uint32_t> generation_{0};
    alignas(64) std::atomic<std::int32_t> finished_{0};
    alignas(64) std::atomic<std::int32_t> arrived_{0};
    std::atomic<bool> stop_{false};
};

class HostMoeExecutor {
public:
    struct Options {
        // Per-member timestamps of every call (HostMoeCallStats::members); ~20 ns per stamp.
        bool record_members = false;
        // The same fast dots with the GPU's W4A4 activation codec (quantize_*_a4) and its BF16
        // roundings of gate/up and h, so an expert quantizes its activations the same way whether
        // it runs on the CPU or on the GPU (the sums still run in a different order). The engine
        // always runs it. false (W4A8: int8 activations) serves the tests' exact oracle of the
        // dots and the schedule (test_host_moe, test_host_moe_mailbox) only.
        bool a4 = true;
        // Instruction set of the dots and codecs (empty = host_isa(): the widest the CPU runs, or
        // NINFER_HOST_ISA). Results are identical bit for bit on every ISA.
        std::optional<HostIsa> isa;
        // Every layer is a ggml block bank: the NVFP4 activation buffers are not allocated, and an
        // NVFP4 layer is refused.
        bool gguf_only = false;
        // NINFER_HOST_GGUF_BIG_GATHER=1: members on big cores of a hybrid CPU
        // decode GGUF rows with gguf::HostGgufKernels::decode_rows_big (vpgather IQ3 grids: fast on
        // P-cores, ~2x slower on E-cores, which keep decode_rows). Same bytes either way.
        bool gguf_big_gather = false;
        // With gguf_big_gather and at least one big-core member (NINFER_HOST_GGUF_ECORE_TAIL=N):
        // the other members claim no gate/up or down item once fewer than N remain, so the end of a
        // phase runs on the big cores (an E-core item takes ~3x a P-core one). 0 = off.
        std::int32_t gguf_ecore_tail = 0;
    };

    // `max_tokens` >= 1 bounds the tokens of one call (any value; buffers scale with it).
    HostMoeExecutor(HostCpuPool& pool, std::int32_t hidden, std::int32_t intermediate,
                    std::int32_t max_tokens, std::int32_t max_top_k, Options options);

    // Called by pool member 0 only.
    HostMoeCallStats run(const HostExpertLayer& layer, const HostMoeJob& job);

    // Intermediates of the last run() (qualification of the two phases separately).
    [[nodiscard]] std::int32_t intermediate_count() const noexcept;
    [[nodiscard]] HostMoeIntermediate intermediate(std::int32_t i) const;

private:
    static void body(void* self, std::int32_t member);
    // GGUF layers (host_moe_executor_gguf.cpp).
    void gguf_call();
    static void gguf_body(void* self, std::int32_t member);
    void gguf_member(std::int32_t member);

    struct AlignedFree {
        void operator()(std::uint8_t* p) const noexcept;
    };

    using AlignedBytes = std::unique_ptr<std::uint8_t[], AlignedFree>;

    struct GgufMember {
        AlignedBytes dec;
        std::vector<float> gate, up, part;
        std::vector<const void*> acts;
    };

    AlignedBytes gguf_x_, gguf_h_; // packed activations: [max_tokens], [max pairs]
    std::size_t gguf_x_bytes_ = 0, gguf_h_bytes_ = 0;
    std::vector<GgufMember> gguf_members_;
    alignas(64) std::atomic<std::int32_t> gguf_next_[3]{};
    // Barrier 1 of a Static call, prefetching the member's down range meanwhile.
    void await_down(std::int32_t member);
    void phase_tokens(std::int32_t member);
    void phase_gate_up();
    void phase_down(std::int32_t member);
    void static_gate_up(std::int32_t member);
    void static_down(std::int32_t member);

    void stamp(std::int32_t member, int index) noexcept {
        if (options_.record_members)
            members_[static_cast<std::size_t>(member)].ns[index] = now_rel();
    }

    [[nodiscard]] std::int64_t now_rel() const noexcept;

    HostCpuPool& pool_;
    std::int32_t hidden_, intermediate_, max_tokens_, max_top_k_;
    Options options_;
    const HostNvfp4Kernels& kernels_;
    const gguf::HostGgufKernels& gguf_kernels_; // gguf::host_gguf_isa(kernels_.isa)
    std::vector<std::uint8_t> gguf_big_; // per member: decode_rows_big (Options::gguf_big_gather)

    // Current call. Pairs (token, host expert) are grouped by expert (CSR): expert u owns pairs
    // [pair_begin_[u], pair_begin_[u+1]), tokens ascending; experts in ascending id order.
    const HostExpertLayer* layer_ = nullptr;
    const HostMoeJob* job_        = nullptr;
    std::int32_t unique_count_    = 0;
    std::vector<std::int32_t> unique_;      // [U] expert id
    std::vector<std::int32_t> pair_begin_;  // [U+1]
    std::vector<std::int32_t> pair_token_;  // [P]
    std::vector<float> pair_alpha_;         // [P]
    std::vector<std::int32_t> expert_slot_; // [experts] u of an expert in this call, or -1
    std::vector<std::int32_t> used_tokens_; // tokens with at least one host expert
    W4A8ActivationBuffer x_act_;            // [max_tokens] by token
    W4A8ActivationBuffer h_act_;            // [max_tokens * max_top_k] by pair
    std::vector<std::vector<float>> acc_;   // per pool member: [max_tokens][64] down sums

    struct alignas(64) Cursor { // Static: a member's remaining slice range
        std::atomic<std::int32_t> next{0};
        std::int32_t end = 0;
    };

    std::unique_ptr<Cursor[]> gate_cursor_, down_cursor_;
    std::vector<W4A8ActivationBuffer> member_x_; // Static: per member [kHostMaxTokens]
    std::vector<HostMoeMemberTimes> members_;    // record_members
    bool static_ = false;                        // work distribution of the current call

    std::int64_t start_ns_    = 0;
    std::int64_t phase_ns_[3] = {0, 0, 0};
    alignas(64) std::atomic<std::int32_t> next_token_{0};
    alignas(64) std::atomic<std::int32_t> next_gate_{0};
    alignas(64) std::atomic<std::int32_t> next_down_{0};
};

} // namespace ninfer::ops::host
