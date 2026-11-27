#pragma once

#include "core/arena.h"
#include "models/qwen3_5/execution/qwen4_experts.h"
#include "models/qwen3_5/execution/forward_runtime.h"
#include "models/qwen3_5/program/storage/qwen4_elastic.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

class ProgramImpl;
struct SequencePlanImpl;

// Qwen4Exp routed experts of a Program (one device or tp 2): every rank's expert backend, the MTP
// layer's banks, the elastic KV <-> expert slot manager and the expert window of an execution.
// Owned by value by the Program, which it reads (devices, plan inputs) through the `program`
// arguments. Empty (no backend) for a dense FFN.
class Qwen4RoutedExpertRuntime {
public:
    // The routed-expert backends of a Qwen4Exp MoE on every rank: the in-place expert tables, the
    // MTP layer's bank and its divisor planes, the offload or mapped backend, bound into each
    // rank's runtime.
    void construct(ProgramImpl& program, const SequencePlanImpl& plan, DeviceSpan backing,
                   execution::ForwardRuntime& runtime, execution::ForwardRuntime& peer_runtime);
    // Expert window of one execution on every rank's backend (qwen4_experts.h window protocol):
    // begin before anything of it is enqueued, observe once it is enqueued, end after the
    // devices completed it (abandon on failure). No-ops without Qwen4Exp experts.
    void begin_window(const ProgramImpl& program, execution::Qwen4Window kind);
    void observe_window(const ProgramImpl& program);
    void end_window(const ProgramImpl& program);
    void abandon_window(const ProgramImpl& program) noexcept;
    // The slot cache's counters summed over both ranks' offload backends, and the elastic KV's
    // (RuntimeStats::expert_cache); not present without an offload backend.
    [[nodiscard]] ninfer::ExpertCacheRuntimeStats stats() const noexcept;

    std::unique_ptr<execution::Qwen4RoutedExperts> experts;
    std::uint32_t expert_slots      = 0;
    std::size_t expert_pool_bytes   = 0;
    std::int32_t expert_publish_lag = 0;
    // The elastic KV <-> expert slot manager (null unless plan.inputs.elastic_kv.enabled).
    std::unique_ptr<Qwen4ElasticVram> elastic;
    std::optional<execution::Qwen4ExpertBank> mtp_bank;
    // Host copy of the MTP bank's expanded weight divisors (gate|up [2E] then down [E]) for the
    // offload backend (CPU path, device divisor table); empty for per-expert banks.
    std::vector<float> mtp_host_divisors;
    std::optional<execution::Qwen4ExpertBank> peer_mtp_bank; // tp 2: rank 1's MTP bank
    std::unique_ptr<execution::Qwen4MoeProbe> mtp_probe;
    // tp 2: rank 1's expert backend (the odd expert IDs).
    std::unique_ptr<execution::Qwen4RoutedExperts> peer_experts;
};

} // namespace ninfer::models::qwen3_5::detail
