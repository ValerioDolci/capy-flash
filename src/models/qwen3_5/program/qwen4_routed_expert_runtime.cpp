#include "models/qwen3_5/program/qwen4_routed_expert_runtime.h"
#include "models/qwen3_5/program/program_impl.h"
#include "core/device.h"
#include "core/device_scope.h"
#include "core/expert_cache.h"
#include "ops/host_moe/host_moe_executor.h"
#include "core/nvtx.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

namespace {

// NINFER_QWEN4_MTP_PROBE=<file>: appends every MTP-layer routed-expert call's inputs (eager only):
// per call I32 T, then x BF16 [2560,T], ids I32 [10,T], weights FP32 [10,T] (for calibration).
class Qwen4MtpProbeFile final : public execution::Qwen4MoeProbe {
public:
    Qwen4MtpProbeFile(const char* path, std::int32_t layer) : layer_(layer) {
        file_ = std::fopen(path, "ab");
        if (file_ == nullptr) { throw std::runtime_error("cannot open NINFER_QWEN4_MTP_PROBE"); }
    }

    ~Qwen4MtpProbeFile() override { std::fclose(file_); }

    [[nodiscard]] bool wants(std::int32_t layer) const override { return layer == layer_; }

    void capture(std::int32_t, const Tensor& x, const Tensor& ids, const Tensor& weights,
                 cudaStream_t stream) override {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::int32_t T = x.ne[1];
        std::fwrite(&T, sizeof(T), 1, file_);
        for (const Tensor* t : {&x, &ids, &weights}) {
            std::vector<std::byte> host(t->bytes());
            CUDA_CHECK(cudaMemcpy(host.data(), t->data, host.size(), cudaMemcpyDeviceToHost));
            std::fwrite(host.data(), 1, host.size(), file_);
        }
        std::fflush(file_);
    }

private:
    std::FILE* file_ = nullptr;
    std::int32_t layer_;
};

} // namespace

// The routed-expert backends of a Qwen4Exp MoE on every rank: the in-place expert tables, the MTP
// layer's bank and its divisor planes, the offload or mapped backend, bound into each rank's
// runtime.
void Qwen4RoutedExpertRuntime::construct(ProgramImpl& program, const SequencePlanImpl& plan,
                                         DeviceSpan backing, execution::ForwardRuntime& runtime,
                                         execution::ForwardRuntime& peer_runtime) {
    const auto& parameters         = program.parameters;
    const auto& peer               = program.peer;
    const auto* peer_parameters    = program.peer_parameters;
    DeviceContext& device          = program.device;
    const auto& elastic_arena      = program.elastic_arena;
    const auto speculative_backend = program.speculative_backend;
    const auto prefill_chunk       = program.prefill_chunk;
    const bool use_cuda_graph      = program.use_cuda_graph;
    const auto& config             = parameters.model.config().text;
    const auto& q                  = *plan.persistent.qwen4;
    if (!q.expert_table || (peer && !peer->qwen4_expert_table.data)) {
        throw std::logic_error("Qwen4Exp routed experts have no planned expert table");
    }
    // The in-place expert tables: rank r reads expert e from the host-mapped bank when it is in
    // its partition (all experts at one device, e % 2 == r at tp 2: the ID-parity partition,
    // the routing being identical on both ranks), -1 otherwise.
    const auto upload_table = [&](Tensor table, int rank, int ranks, cudaStream_t stream) {
        std::vector<std::int32_t> slots(static_cast<std::size_t>(table.numel()));
        for (std::size_t e = 0; e < slots.size(); ++e) {
            slots[e] = static_cast<int>(e % static_cast<std::size_t>(ranks)) == rank
                           ? static_cast<std::int32_t>(e)
                           : -1;
        }
        CUDA_CHECK(cudaMemcpyAsync(table.data, slots.data(), table.bytes(), cudaMemcpyHostToDevice,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };
    const int ranks = peer ? execution::kTensorParallelWidth : 1;
    // MTP layer: its bank, with a bank-wide weight divisor (ninfer-ext's artifact)
    // expanded into the per-expert planes moe_slots indexes (device copy for the zero-copy
    // backend, host copy for the offload backend's CPU path and its device divisor table).
    const bool mtp = speculative_backend == SpeculativeBackend::Mtp;
    std::optional<execution::Qwen4ExpertBank> mtp_host_bank;
    if (mtp) {
        if (!parameters.qwen4_mtp) { throw std::logic_error("Qwen4Exp MTP has no MTP layer"); }
        mtp_bank      = parameters.qwen4_mtp->layer.moe().experts;
        mtp_host_bank = mtp_bank;
        if (parameters.qwen4_mtp->uniform_gate_up_divisor > 0.0F) {
            if (!q.mtp_gate_up_divisors || !q.mtp_down_divisors) {
                throw std::logic_error("Qwen4Exp MTP divisor planes are not planned");
            }
            Tensor gate_up = q.mtp_gate_up_divisors->bind(backing);
            Tensor down    = q.mtp_down_divisors->bind(backing);
            mtp_host_divisors.assign(static_cast<std::size_t>(gate_up.numel()),
                                     parameters.qwen4_mtp->uniform_gate_up_divisor);
            mtp_host_divisors.resize(static_cast<std::size_t>(gate_up.numel() + down.numel()),
                                     parameters.qwen4_mtp->uniform_down_divisor);
            CUDA_CHECK(cudaMemcpy(gate_up.data, mtp_host_divisors.data(), gate_up.bytes(),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(down.data, mtp_host_divisors.data() + gate_up.numel(),
                                  down.bytes(), cudaMemcpyHostToDevice));
            mtp_bank->gate_up_divisors      = static_cast<const float*>(gate_up.data);
            mtp_bank->down_divisors         = static_cast<const float*>(down.data);
            mtp_host_bank->gate_up_divisors = mtp_host_divisors.data();
            mtp_host_bank->down_divisors =
                mtp_host_divisors.data() + static_cast<std::size_t>(gate_up.numel());
        }
    }
    // Routed experts. At one device the table is the identity.
    Tensor table = q.expert_table->bind(backing);
    upload_table(table, 0, ranks, device.stream);
    std::optional<execution::Qwen4OffloadExperts::Peer> offload_peer;
    if (plan.inputs.qwen4_experts.backend == RoutedExpertBackend::Offload) {
        if (peer) {
            // tp 2: one cache over both GPUs (partition by ID parity), the CPU path on rank 0.
            if (!q.cache_mirror || peer->qwen4_cache_table == nullptr ||
                peer->qwen4_cache_divisors == nullptr) {
                throw std::logic_error("Qwen4Exp tp 2 offloaded experts are not planned on rank 1");
            }
            offload_peer.emplace();
            offload_peer->device        = peer->device.device;
            offload_peer->stream        = peer->device.stream;
            offload_peer->pool          = peer->qwen4_cache_pool;
            offload_peer->table         = peer->qwen4_cache_table;
            offload_peer->divisors      = peer->qwen4_cache_divisors;
            offload_peer->stream_tables = peer->qwen4_stream_tables;
            offload_peer->mirror        = peer->qwen4_cache_mirror;
            offload_peer->reads         = peer->qwen4_cache_reads;
            offload_peer->stage_table   = peer->qwen4_stage_table;
            offload_peer->picks         = peer->qwen4_stage_picks;
            offload_peer->staging       = peer->qwen4_staging;
            offload_peer->slot_offsets  = peer->qwen4_slot_offsets;
        }
        if (q.expert_slots <= 0 ||
            q.cache_layers !=
                static_cast<std::int32_t>(parameters.qwen4->layers.size()) + (mtp ? 1 : 0)) {
            throw std::logic_error("Qwen4Exp offloaded experts need a resolved slot count");
        }
        // Cache layers: the Text layers, then the MTP layer (layer num_hidden_layers).
        std::vector<execution::Qwen4ExpertBank> banks;
        banks.reserve(static_cast<std::size_t>(q.cache_layers));
        for (const auto& layer : parameters.qwen4->layers) { banks.push_back(layer.moe().experts); }
        if (mtp) { banks.push_back(*mtp_host_bank); }
        execution::Qwen4OffloadExperts::DeviceMemory memory;
        memory.pool     = q.bind_pool(elastic_arena ? elastic_arena->span() : backing);
        memory.table    = static_cast<std::int32_t*>(q.cache_table->bind(backing).data);
        memory.divisors = static_cast<float*>(q.cache_divisors->bind(backing).data);
        memory.reads    = static_cast<std::int32_t*>(q.cache_reads->bind(backing).data);
        if (q.cache_slot_offsets) {
            memory.slot_offsets =
                static_cast<std::int64_t*>(q.cache_slot_offsets->bind(backing).data);
        }
        if (q.cache_mirror) {
            memory.mirror = static_cast<std::int32_t*>(q.cache_mirror->bind(backing).data);
        }
        memory.stream_tables =
            static_cast<std::int32_t*>(q.cache_stream_tables->bind(backing).data);
        if (q.stage_table) {
            memory.stage_table = static_cast<std::int32_t*>(q.stage_table->bind(backing).data);
            memory.picks       = static_cast<std::int32_t*>(q.stage_picks->bind(backing).data);
            for (std::size_t plane = 0; plane < memory.staging.size(); ++plane) {
                memory.staging[plane] =
                    static_cast<std::byte*>(q.stage_pool[plane].bind(backing).data);
            }
        }
        execution::Qwen4OffloadOptions options(plan.inputs.qwen4_experts);
        options.slots_per_layer = q.expert_slots;
        options.elastic_columns = q.elastic_columns;
        // 0 (auto) is resolved by the Engine at load; callers that plan without it get the same
        // rule.
        if (options.cpu_threads == 0) {
            options.cpu_threads = ops::host::auto_expert_cpu_threads(
                ops::host::host_cpu_topology(),
                parameters.qwen4->layers.front().moe().experts.gguf);
        }
        options.prompt_tokens = static_cast<std::int32_t>(prefill_chunk);
        // Chunks wider than the base chunk run on workspace lent from the slot pool.
        options.base_chunk             = static_cast<std::int32_t>(plan.qwen4_base_chunk);
        options.prompt_workspace_bytes = static_cast<std::int64_t>(plan.workspace.qwen4_prompt);
        for (const auto& [width, bytes] : plan.workspace.qwen4_prompt_steps) {
            options.prompt_workspace_steps.emplace_back(static_cast<std::int32_t>(width),
                                                        static_cast<std::int64_t>(bytes));
        }
        options.text_layers = static_cast<std::int32_t>(parameters.qwen4->layers.size());
        // Validation / measurement aid (off by default): per-window JSON lines.
        if (const char* trace = std::getenv("NINFER_QWEN4_EXPERT_TRACE")) {
            options.trace_path = trace;
        }
        if (const char* dump = std::getenv("NINFER_QWEN4_ROUTING_DUMP")) {
            options.routing_path = dump;
        }
        if (const char* freeze = std::getenv("NINFER_QWEN4_EXPERT_FREEZE")) {
            options.freeze_after_prompt = std::string_view(freeze) == "1";
            // "all": nothing is ever inserted (residency = initialization / snapshot).
            options.freeze_all = std::string_view(freeze) == "all";
        }
        if (const char* sync = std::getenv("NINFER_QWEN4_EXPERT_SYNC")) {
            options.sync_fills = std::string_view(sync) == "1";
        }
        if (const char* snap = std::getenv("NINFER_QWEN4_EXPERT_SNAPSHOT");
            snap != nullptr && *snap != 0) {
            options.snapshot_load = snap;
        }
        if (const char* snap = std::getenv("NINFER_QWEN4_EXPERT_SNAPSHOT_SAVE");
            snap != nullptr && *snap != 0) {
            options.snapshot_save = snap;
        }
        auto offload = std::make_unique<execution::Qwen4OffloadExperts>(
            banks, options, memory, device.device, device.stream,
            offload_peer ? &*offload_peer : nullptr, parameters.model.host_tails());
        expert_publish_lag = offload->publish_lag();
        experts            = std::move(offload);
        expert_slots       = static_cast<std::uint32_t>(q.expert_slots);
        expert_pool_bytes  = q.cache_pool_bytes;
    } else {
        // The mapped backend has every expert kernel read its bank in place: the pageable
        // fallback banks (--host-pinned-max-gib, or a refused page-locked allocation) cannot.
        const auto pinned = [](const execution::Qwen4ExpertBank& bank) {
            const ops::MoeSlotPool& b = bank.bank;
            if (bank.gguf) { // rows outside the NVFP4 planes
                return expert_cache::pinned_host_memory(bank.gguf_gate_up_rows) &&
                       expert_cache::pinned_host_memory(bank.gguf_down_rows);
            }
            return expert_cache::pinned_host_memory(b.gate_up_codes.base) &&
                   expert_cache::pinned_host_memory(b.gate_up_scales.base) &&
                   expert_cache::pinned_host_memory(b.down_codes.base) &&
                   expert_cache::pinned_host_memory(b.down_scales.base);
        };
        bool all_pinned = !mtp || pinned(*mtp_bank);
        for (const auto& layer : parameters.qwen4->layers) {
            all_pinned = all_pinned && pinned(layer.moe().experts);
        }
        if (!all_pinned) {
            throw std::invalid_argument(
                "the mapped expert backend reads every expert bank in place from page-locked "
                "memory, and part of the banks is in pageable memory (page-locked limit); use "
                "the offload backend");
        }
        experts = std::make_unique<execution::Qwen4MappedExperts>(table);
    }
    if (use_cuda_graph && !experts->graph_capturable()) {
        throw std::logic_error("Qwen4Exp expert backend cannot run inside a CUDA Graph");
    }
    runtime.experts = experts.get();
    if (mtp) {
        // The offload backend serves the MTP layer itself (resident slots + CPU, its own cache
        // layer); the zero-copy one reads the MTP bank in place through the same backend.
        runtime.mtp_experts     = experts.get();
        runtime.mtp_expert_bank = &*mtp_bank;
        if (const char* probe = std::getenv("NINFER_QWEN4_MTP_PROBE");
            probe != nullptr && *probe != '\0') {
            if (use_cuda_graph) {
                throw std::invalid_argument("NINFER_QWEN4_MTP_PROBE requires eager execution");
            }
            mtp_probe = std::make_unique<Qwen4MtpProbeFile>(
                probe, static_cast<std::int32_t>(config.num_hidden_layers));
            runtime.moe_probe = mtp_probe.get();
        }
    }
    if (peer) {
        const ScopedCurrentDevice rank1(peer->device.device);
        upload_table(peer->qwen4_expert_table, 1, ranks, peer->device.stream);
        if (offload_peer) {
            peer_experts = std::make_unique<execution::Qwen4OffloadPeerExperts>(
                q.cache_layers, q.expert_slots, *offload_peer,
                static_cast<execution::Qwen4OffloadExperts*>(experts.get()), q.elastic_columns);
        } else {
            peer_experts =
                std::make_unique<execution::Qwen4MappedExperts>(peer->qwen4_expert_table);
        }
        peer_runtime.experts = peer_experts.get();
        if (mtp) {
            // MTP at tp 2: rank 1 runs the MTP layer's experts of its partition through its
            // offload view (layer num_hidden_layers of the shared cache); the bank only supplies
            // the activation divisors there.
            if (!offload_peer || !peer_parameters->qwen4_mtp) {
                throw std::invalid_argument(
                    "Qwen4Exp MTP at tp 2 requires the offload expert backend and both ranks' "
                    "MTP layers");
            }
            peer_mtp_bank                = peer_parameters->qwen4_mtp->layer.moe().experts;
            peer_runtime.mtp_experts     = peer_experts.get();
            peer_runtime.mtp_expert_bank = &*peer_mtp_bank;
        }
    }
}

void Qwen4RoutedExpertRuntime::begin_window(const ProgramImpl& program,
                                            execution::Qwen4Window kind) {
    const DeviceContext& device = program.device;
    const auto& peer            = program.peer;
    nvtx::ScopedRange range(nvtx::Name::Qwen4BeginWindow, nvtx::Category::Moe);
    if (experts) { experts->begin_window(device.stream, kind); }
    if (peer_experts) {
        try {
            const ScopedCurrentDevice rank1(peer->device.device);
            peer_experts->begin_window(peer->device.stream, kind);
        } catch (...) {
            if (experts) { experts->abandon_window(device.stream); }
            throw;
        }
    }
}

void Qwen4RoutedExpertRuntime::observe_window(const ProgramImpl& program) {
    const DeviceContext& device = program.device;
    const auto& peer            = program.peer;
    if (experts) { experts->observe_window(device.stream); }
    if (peer_experts) {
        const ScopedCurrentDevice rank1(peer->device.device);
        peer_experts->observe_window(peer->device.stream);
    }
}

void Qwen4RoutedExpertRuntime::end_window(const ProgramImpl& program) {
    const DeviceContext& device = program.device;
    const auto& peer            = program.peer;
    nvtx::ScopedRange range(nvtx::Name::Qwen4EndWindow, nvtx::Category::Moe);
    if (experts) { experts->end_window(device.stream); }
    if (peer_experts) {
        const ScopedCurrentDevice rank1(peer->device.device);
        peer_experts->end_window(peer->device.stream);
    }
}

void Qwen4RoutedExpertRuntime::abandon_window(const ProgramImpl& program) noexcept {
    const DeviceContext& device = program.device;
    const auto& peer            = program.peer;
    if (experts) { experts->abandon_window(device.stream); }
    if (peer_experts) {
        try {
            const ScopedCurrentDevice rank1(peer->device.device);
            peer_experts->abandon_window(peer->device.stream);
        } catch (...) {}
    }
}


} // namespace ninfer::models::qwen3_5::detail
