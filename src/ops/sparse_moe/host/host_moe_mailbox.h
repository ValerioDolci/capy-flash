#pragma once

// GPU <-> CPU handshake for the routed experts a layer computes on the host, captured in a CUDA
// graph.
//
// Mapped page-locked memory (cudaHostAllocMapped | Portable): ready/done flags per MoE layer and
// one routing / x / partial buffer shared by all layers, so publish(l') must be stream-ordered
// after accumulate(l) of the layer before it (true whenever l' consumes l's output). Per layer the
// GPU stream runs
//   publish(l)     kernel: x [T, H] BF16, the host-side ids (routed id where the device has no
//                  slot, else -1) and routing weights -> slot l; then done[l] = 0, ready[l] = 1:
//                  for T <= 8 written by the (single-block) kernel itself after a system-wide
//                  fence of every thread's writes, for wider calls by two stream memory writes
//                  (each preceded by a system-wide fence)
//   ...            the device keeps working: resident experts, anything else
//   accumulate(l)  cuStreamWaitValue32(done[l] >= 1), then a kernel y += host partial (FP32,
//                  read once from mapped memory; skipped when the host had no expert)
// The wait is executed by the stream front end: it holds no SM, so cooperative kernels queued on
// other streams (the hyper-connection kernel at T <= 8 needs every SM) are unaffected, and the
// stream is held only where the sum is needed. No kernel ever spins on a host flag.
//
// Host side: one service thread (member 0 of the pinned HostCpuPool) polls ready[l] of the layer it
// expects next and of the layer it served last (a layer called in several slices of <= max_tokens
// positions publishes several times; all slots after a quiet millisecond, so an aborted window
// cannot desynchronize it), clears it,
// runs HostMoeExecutor on the slot, writes active[l] and its FP32 partial, then done[l] = 1. It
// spins while requests are recent and backs off to 50 us sleeps after `idle_ms` without one.
//
// Flags are binary and reset by the GPU before every doorbell (FreeToken's protocol,
// cpu_moe_ext.cpp
// @555efd8), because the values of stream memory operations are fixed at graph capture. release()
// writes kReleased (passes the >= 1 wait) to every done word, completing the waits pending now:
// the shutdown path and the hook for an owner's watchdog (which repeats it until the stuck window's
// end event completes, since each later layer re-arms its own wait). A service failure (executor
// exception) makes failed() true and from then on answers every doorbell at once with active = 0:
// no window can block, and the owner must discard the outputs of windows that saw the failure.

#include "ops/sparse_moe/host/host_moe_executor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace ninfer::ops::host {

struct HostMoeResidency;

// How a call's routed experts that the device does not hold (the misses) are divided between the
// host executor and the device itself, which then reads its share of the expert weights over
// PCIe. Decided per publish, on the device, from the call's routing: per layer and per window.
// The distinct misses are ordered by columns, most first, then route order, and with mode Cost the
// device takes the prefix of length g in [0, min(n, max_reads)] minimizing max(host(n - g),
// device(g)), the smallest on ties, 0 when n <= min_misses, with
//   host(m)   = host_fixed_us + host_slowdown(g) * sum(host_expert_us + host_column_us * (c - 1))
//   device(g) = device_fixed_us + sum(device_expert_us + device_column_us * (c - 1))
// (c = columns routed to the expert, host(0) = device(0) = 0, host_slowdown(g) =
// host_read_slowdown when g > 0, else 1: the device's reads take host RAM bandwidth).
struct HostMoeSplit {
    // The values are written to the expert trace ("split": {"mode"}).
    enum class Mode : std::int32_t { Off = 0, Cost = 3 };
    static constexpr std::int32_t kMaxReads = 16;
    Mode mode                               = Mode::Off;
    std::int32_t min_misses                 = 2;
    std::int32_t max_reads                  = kMaxReads;
    float host_fixed_us                     = 13.0F;
    float host_expert_us                    = 58.0F;
    float host_column_us                    = 1.9F;
    float host_read_slowdown                = 1.2F;
    float device_fixed_us                   = 10.0F;
    float device_expert_us                  = 140.0F;
    float device_column_us                  = 0.0F;
};

// Device outputs of a split publish (device memory). reads [tokens, top_k]: the routed id on the
// routes the device computes, -1 elsewhere. Optional (staged device share): picks [kMaxReads + 1] =
// the device's experts in split order, then their count; stage_table [experts] = k for picks[k],
// -1 for every other expert.
struct HostMoeDeviceShare {
    std::int32_t* reads       = nullptr;
    std::int32_t* picks       = nullptr;
    std::int32_t* stage_table = nullptr;
    // Optional device copy of the split rule: when set the publish reads the rule there instead of
    // its launch argument, so the owner can change it between windows without capturing the
    // graphs again (--miss-split calibrate).
    const HostMoeSplit* live = nullptr;
};

// One served decode call (<= kHostMaxTokens tokens with host experts) for the cost fit: distinct
// host experts, (token, expert) pairs, executor time in microseconds.
struct HostMoeCostSample {
    std::int32_t experts = 0;
    std::int32_t pairs   = 0;
    float us             = 0.0F;
};

// Host cost line of HostMoeSplit fitted on samples: us = fixed + expert * experts + column *
// (pairs - experts), least squares, refitted once without the samples whose residual exceeds 4x
// the median absolute residual (wake-ups after idle, preemption). When the samples barely vary in
// columns per expert (one token per call) `column_us` keeps `default_column_us` and the other two
// are fitted. Coefficients are clamped at >= 0. `samples` = samples kept; 0 = no fit.
struct HostMoeCostFit {
    std::int32_t samples = 0;
    float fixed_us       = 0.0F;
    float expert_us      = 0.0F;
    float column_us      = 0.0F;
    float rms_us         = 0.0F; // residual of the kept samples
};

[[nodiscard]] HostMoeCostFit fit_host_costs(std::span<const HostMoeCostSample> samples,
                                            float default_column_us);

struct HostMoeMailboxConfig {
    int device                = 0;
    std::int32_t layers       = 0;
    std::int32_t max_tokens   = kHostMaxTokens; // tokens of one publish (any; buffers scale)
    std::int32_t top_k        = 0;
    std::int32_t hidden       = 0;
    std::int32_t intermediate = 0;
    std::vector<int> cpus; // pool cpus; cpus[0] runs the service thread
    HostMoeExecutor::Options executor;
    double idle_ms              = 200.0;
    std::int64_t trace_capacity = 0; // timing records kept (0 = no trace)
    // Routing log for the owner's cache policy (publish_table): per layer the full routed ids of
    // up to this many tokens (0 = no log).
    std::int32_t route_log_tokens = 0;
    // %globaltimer stamps per layer (publish_table, stamp, accumulate), for timing breakdowns.
    bool gpu_stamps = false;
};

// One layer's routing log in mapped host memory, written by publish_table. `published` is set to
// 1 by every publish of the layer after its ids are visible (the owner clears it before a window);
// `tokens` = 1 + the last logged token index; ids [tokens, top_k] as routed (all of them, also the
// device-resident ones).
// `split` [kSplitWords]: per publish of the layer since the flags were cleared, at word
// log_offset % kSplitWords, (device experts) | (device routes) << 16 of a split publish.
struct HostMoeRouteLog {
    static constexpr std::int32_t kSplitWords = 12;
    const volatile std::uint32_t* published   = nullptr;
    const volatile std::uint32_t* tokens      = nullptr;
    const std::int32_t* ids                   = nullptr;
    const volatile std::uint32_t* split       = nullptr;
};

// Stamp indices of HostMoeMailbox::gpu_stamps (nanoseconds of the device %globaltimer).
enum HostMoeStamp : std::int32_t {
    kStampPublish    = 0, // publish kernel start
    kStampDevice     = 1, // stamp() after the device's own expert work (owner-placed)
    kStampAccumulate = 2, // accumulate kernel start = the wait for the host partial released
    kStampCount      = 4,
};

// One served request, on the CLOCK_MONOTONIC nanosecond clock of now_ns().
struct HostMoeTraceRecord {
    std::int32_t layer   = 0;
    std::int32_t tokens  = 0;
    std::int32_t experts = 0;
    std::int32_t pairs   = 0;
    std::int64_t seen_ns = 0; // doorbell observed
    std::int64_t done_ns = 0; // just before done[l] = 1
};

// One served call with the executor's breakdown (NINFER_HOST_MOE_CALL_LOG=<file>: every call is
// kept, up to 4 Mi, and written as CSV when the mailbox is destroyed). Times in ns: seen/done on
// the now_ns() clock, the rest from the start of the executor's run().
struct HostMoeCallRecord {
    std::int32_t layer = 0, tokens = 0, experts = 0, pairs = 0;
    std::int64_t seen_ns = 0, done_ns = 0;
    std::int64_t dispatch_ns = 0, gate_end_ns = 0, down_end_ns = 0, end_ns = 0;
    std::int64_t wake_max_ns = 0, gate_min_ns = 0, gate_max_ns = 0, down_min_ns = 0,
                 down_max_ns = 0;
};

// Capturable. Staged device share: copies the four planes of the experts a split publish picked
// (share.picks: ids, then their count at [HostMoeSplit::kMaxReads]) from the host bank (plane q of
// expert e at bank[q] + e * plane_bytes[q], host-mapped) into scratch slot k of staging[q] (k = the
// position in picks), with a streaming kernel of `blocks` CTAs (16: as fast as 64-128 when
// measured, and leaves the SMs to the resident experts; every byte read once over PCIe, many
// loads in flight; 16-byte units where plane base and size allow, 4-byte otherwise). Returns at
// once when nothing was picked. Planes 4-byte aligned, plane_bytes multiples of 4. Capturable. The
// device share a split publish_table with the same ids, table and split would pick (reads, picks,
// scratch table), without publishing anything: at tp 2 the other GPU runs it on its own copy of the
// residency mirror and computes the share over its own link.
void select_device_share(cudaStream_t stream, std::int32_t tokens, std::int32_t top_k,
                         const std::int32_t* ids, const std::int32_t* slot_table,
                         std::int32_t experts, const HostMoeSplit& split,
                         const HostMoeDeviceShare& share);
void stage_device_share(cudaStream_t stream, const std::int32_t* picks,
                        const std::array<const std::byte*, 4>& bank,
                        const std::array<std::byte*, 4>& staging,
                        const std::array<std::int64_t, 4>& plane_bytes, int blocks = 16);

// A one-shot host-to-stream signal: a 32-bit word of mapped page-locked memory (any device
// reads it at the same address: unified addressing). Capturable: the stream waits until the host
// has raised the word (a stream memory operation: no SM, no kernel spinning on it), then lowers
// it, so every execution of a captured graph consumes one raise. The host lowers it only while no
// stream can be waiting on it, and raises it after the data the stream reads next are written.
void consume_host_flag(cudaStream_t stream, std::uint32_t* word);
void raise_host_flag(std::uint32_t* word) noexcept;
void lower_host_flag(std::uint32_t* word) noexcept;

class HostMoeMailbox {
public:
    static constexpr std::uint32_t kReleased = 0x7fffffffU;

    // `layers[l]` = host banks of MoE layer l; they must outlive the mailbox.
    HostMoeMailbox(HostMoeMailboxConfig config, std::vector<HostExpertLayer> layers);
    ~HostMoeMailbox();
    HostMoeMailbox(const HostMoeMailbox&)            = delete;
    HostMoeMailbox& operator=(const HostMoeMailbox&) = delete;

    // Capturable. x [tokens, hidden], ids / alpha [tokens, top_k] on the device; device_slot
    // [tokens, top_k] = slot of the routed expert on this device or -1 (nullptr: all on the host).
    void publish(cudaStream_t stream, std::int32_t layer, std::int32_t tokens,
                 const __nv_bfloat16* x, const std::int32_t* ids, const float* alpha,
                 const std::int32_t* device_slot) const;
    // Capturable. As publish, for a device that holds an expert iff slot_table[id] >= 0
    // (slot_table: the layer's I32 [experts] table, read on the device at execution), and logs
    // the full routed ids at token offset `log_offset` of the layer's routing log (requires
    // route_log_tokens >= log_offset + tokens). With a split (mode != Off; tokens <= 8,
    // share.reads required) the device's share of the misses is left out of the host's ids and
    // written to `share` (HostMoeSplit, HostMoeDeviceShare), and its size to the routing log.
    void publish_table(cudaStream_t stream, std::int32_t layer, std::int32_t tokens,
                       const __nv_bfloat16* x, const std::int32_t* ids, const float* alpha,
                       const std::int32_t* slot_table, std::int32_t experts,
                       std::int32_t log_offset, const HostMoeSplit& split = {},
                       const HostMoeDeviceShare& share = {}) const;
    // Capturable. Only the routing log of `layer`, as publish_table writes it (ids at token offset
    // `log_offset`, then the published flag): no x, no doorbell, the host executor is not involved
    // (a prefill call whose routed experts all run on the device). Requires a routing log.
    void log_routes(cudaStream_t stream, std::int32_t layer, std::int32_t tokens,
                    const std::int32_t* ids, std::int32_t log_offset) const;
    // Capturable. Writes the device %globaltimer into stamp `index` of `layer` (gpu_stamps only).
    void stamp(cudaStream_t stream, std::int32_t layer, std::int32_t index) const;
    // Capturable. Waits for layer's host partial and adds it: y [tokens, hidden] BF16 (y = bf16(y +
    // p)).
    void accumulate(cudaStream_t stream, std::int32_t layer, std::int32_t tokens,
                    __nv_bfloat16* y) const;
    // Same, into an FP32 accumulator: y += device_partial (FP32 [tokens, hidden] in device memory,
    // when given: e.g. the device's own reads of host experts), then y += the host partial.
    void accumulate(cudaStream_t stream, std::int32_t layer, std::int32_t tokens, float* y,
                    const float* device_partial = nullptr) const;

    void release() noexcept;

    [[nodiscard]] bool failed() const noexcept { return failed_.load(std::memory_order_acquire); }

    [[nodiscard]] std::string error() const;

    [[nodiscard]] HostMoeRouteLog route_log(std::int32_t layer) const noexcept;
    // Device experts / routes of `layer`'s split publishes since the flags were cleared.
    [[nodiscard]] std::int64_t device_share_experts(std::int32_t layer) const noexcept;
    [[nodiscard]] std::int64_t device_share_routes(std::int32_t layer) const noexcept;

    // Host experts / calls served since construction (all layers; any thread).
    [[nodiscard]] std::int64_t host_experts_served() const noexcept {
        return served_experts_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::int64_t host_calls_served() const noexcept {
        return served_calls_.load(std::memory_order_relaxed);
    }

    // Cost fit (--miss-split calibrate): the service records up to `max_samples` decode calls
    // with host experts from now on; samples() copies them (any thread).
    void collect_cost_samples(std::int64_t max_samples);
    [[nodiscard]] std::vector<HostMoeCostSample> cost_samples() const;

    // Clears every layer's `published` word and split words (no window in flight).
    void clear_route_flags() noexcept;
    // Stamps of `layer` (gpu_stamps; nullptr otherwise): kStampCount words, device clock.
    [[nodiscard]] const volatile std::uint64_t* gpu_stamps(std::int32_t layer) const noexcept;
    // Offset to add to a device %globaltimer value to get now_ns() time (min over `rounds` probes,
    // error ~1 us). Uses `stream`; call with no window in flight.
    [[nodiscard]] std::int64_t calibrate_clock(cudaStream_t stream, int rounds = 32) const;

    // Records served so far (in order); call with no window in flight.
    [[nodiscard]] std::vector<HostMoeTraceRecord> trace() const;
    void clear_trace() noexcept;
    [[nodiscard]] static std::int64_t now_ns() noexcept;

    // Calls of `layer` served so far (incremented right after its done flag is written): the
    // host executor is idle for that layer once the count moved past a value read before the
    // layer's doorbell.
    [[nodiscard]] std::uint32_t served(std::int32_t layer) const noexcept {
        return served_[static_cast<std::size_t>(layer)].load(std::memory_order_acquire);
    }

private:
    struct SlotView {
        volatile std::uint32_t* ready  = nullptr;
        volatile std::uint32_t* done   = nullptr;
        volatile std::uint32_t* tokens = nullptr;
        volatile std::uint32_t* active = nullptr;
        std::int32_t* ids              = nullptr;
        float* alpha                   = nullptr;
        std::uint16_t* x               = nullptr;
        float* out                     = nullptr;
    };

    [[nodiscard]] SlotView slot(std::int32_t layer) const noexcept;
    void publish_impl(cudaStream_t stream, std::int32_t layer, std::int32_t tokens,
                      const __nv_bfloat16* x, const std::int32_t* ids, const float* alpha,
                      const HostMoeResidency& resident, std::int32_t log_offset) const;
    [[nodiscard]] unsigned long long* accumulate_stamp(std::int32_t layer) const noexcept;
    void service();
    void serve(HostMoeExecutor& executor, std::int32_t layer);

    HostMoeMailboxConfig config_;
    std::vector<HostExpertLayer> layers_;
    std::size_t ids_off_ = 0, alpha_off_ = 0, x_off_ = 0, out_off_ = 0;
    std::size_t log_off_ = 0, log_stride_ = 0, stamp_off_ = 0, clock_off_ = 0;
    std::byte* host_   = nullptr; // mapped allocation (same address on the device: UVA)
    std::byte* device_ = nullptr;

    std::thread service_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> failed_{false};
    std::atomic<int> started_{0}; // 1 = running, -1 = startup failed
    mutable std::mutex error_mutex_;
    std::string error_;
    std::vector<HostMoeTraceRecord> trace_;
    std::atomic<std::int64_t> trace_size_{0};
    std::unique_ptr<std::atomic<std::uint32_t>[]> served_;
    std::string call_log_path_;
    std::vector<HostMoeCallRecord> call_log_;
    std::atomic<std::int64_t> served_experts_{0};
    std::atomic<std::int64_t> served_calls_{0};
    std::atomic<std::int64_t> cost_capacity_{0};
    mutable std::mutex cost_mutex_;
    std::vector<HostMoeCostSample> cost_samples_;
    // Validation dump (NINFER_HOST_MOE_DUMP=<file>): every served decode call (<= 8 tokens
    // with host experts) as the executor saw it (layer, ids, alpha, x, out) plus the partial the
    // device's accumulate actually read (`echo_`, mapped, written by the accumulate kernel),
    // collected at the next serve. Written at destruction.
    void dump_record(std::int32_t layer, std::int32_t tokens, std::int32_t experts);
    void dump_fill_echo() noexcept;
    std::string dump_path_;
    std::vector<std::byte> dump_;
    std::int64_t dump_max_           = 0;
    std::int64_t dump_records_       = 0;
    std::int64_t dump_serial_        = 0;
    std::int64_t dump_pending_       = -1; // byte offset of the last record's echo block, or -1
    std::int32_t dump_pending_layer_ = 0, dump_pending_tokens_ = 0;
    float* echo_ = nullptr; // mapped [layers][kHostMaxTokens * hidden]
};

} // namespace ninfer::ops::host
