#include "models/qwen3_5/program/program_impl.h"
#include "artifact/host_residency.h"
#include "ninfer/ops/ple.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/execution/linear.h"
#include "core/startup.h"
#include "core/device.h"
#include "core/device_scope.h"
#include "core/expert_cache.h"
#include "ops/host_moe/host_moe_executor.h"
#include "core/nvtx.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/target_logprobs.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string_view>
#include <cstdlib>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

static_assert(std::is_nothrow_move_assignable_v<SpeculativeStats>);

namespace {

// The Host tiers are pinned at startup and the kernel cannot page them
// out. Pinning more than the machine can spare does not fail here but later, as an OOM kill of
// the server or of something else (Qwen3.8-Flash-Next already pins its routed experts). Refuse
// to start when the tiers would leave less than max(4 GiB, 5 % of RAM) available, and say how
// much fits. NINFER_HOST_TIER_RAM_GUARD=0 disables the check.
void require_host_tier_fits_ram(std::uint64_t state_bytes, std::uint64_t kv_bytes) {
    const std::uint64_t pinned = state_bytes + kv_bytes;
    if (pinned == 0) { return; }
    if (const char* guard = std::getenv("NINFER_HOST_TIER_RAM_GUARD");
        guard != nullptr && std::string_view(guard) == "0") {
        return;
    }
    // Free CMA counts in MemAvailable but cannot be page-locked.
    const artifact::HostMemInfo memory = artifact::read_host_meminfo();
    const std::uint64_t available =
        memory.available > memory.cma_free ? memory.available - memory.cma_free : 0;
    if (memory.available == 0 || memory.total == 0) { return; }
    const std::uint64_t reserve = std::max<std::uint64_t>(4ULL << 30, memory.total / 20U);
    if (available >= pinned + reserve) { return; }
    const std::uint64_t fits =
        available > reserve + state_bytes ? available - reserve - state_bytes : 0;
    using artifact::gib_text;
    throw std::invalid_argument(
        "the Host context tiers would pin " + gib_text(pinned) + " of RAM (state " +
        gib_text(state_bytes) + ", KV " + gib_text(kv_bytes) + ") but only " + gib_text(available) +
        " are available outside CMA and " + gib_text(reserve) +
        " must stay free: lower --host-kv-mib (at most " + std::to_string(fits >> 20) +
        " MiB with these state slots) or --host-state-slots, or set "
        "NINFER_HOST_TIER_RAM_GUARD=0 to skip this check");
}

std::uint32_t normalized_private_capacity(const ContextCacheOptions& options);

std::uint32_t normalized_private_capacity(const ContextCacheOptions& options) {
    if (!options.max_private_continuations || *options.max_private_continuations == 0) {
        throw std::logic_error("Qwen3.5 context cache private capacity is not normalized");
    }
    return *options.max_private_continuations;
}

// Both ranks of a tensor-parallel Program start from zeroed persistent and workspace arenas, so
// the two ranks' KV pages, StateImages and scratch hold identical bytes before their first
// write. A single-device Program's arenas start undefined, as every owner fills what it reads.
ZeroFill tensor_parallel_zero_fill(const SequencePlanImpl& plan) noexcept {
    return plan.peer_persistent ? ZeroFill::Yes : ZeroFill::No;
}

// The persistent arena of one rank as an ElasticArena (null without elastic KV).
std::unique_ptr<ElasticArena> make_elastic_arena(const SequencePlanImpl& plan,
                                                 const PersistentLayout& layout, int device) {
    if (!plan.inputs.elastic_kv.enabled) { return nullptr; }
    const std::uint32_t step = (plan.inputs.elastic_kv.grow_tokens + kPagedKVPageSize - 1U) /
                               static_cast<std::uint32_t>(kPagedKVPageSize);
    const auto ranges = qwen4_elastic_ranges(layout, step);
    return std::make_unique<ElasticArena>(device, layout.elastic_bytes, ranges);
}

} // namespace

ProgramImpl::ProgramImpl(const execution::Parameters& parameters_in, const SequencePlanImpl& plan,
                         DeviceContext& device_in, const StartupObserver& startup_observer,
                         ExecutionContext* execution_in,
                         const execution::Parameters* peer_parameters_in)
    : parameters(parameters_in), device(device_in), capacity(plan.inputs.capacity),
      kv_capacity(plan.kv_capacity), max_concurrency(plan.inputs.max_concurrency),
      context_cache(plan.inputs.context_cache),
      continuation_capacity(normalized_private_capacity(plan.inputs.context_cache)),
      shared_prefix_capacity(plan.inputs.context_cache.max_shared_prefixes.value_or(0)),
      prefill_chunk(plan.prefill_chunk), qwen4_prompt_base_chunk(plan.qwen4_base_chunk),
      kv_lease_window_tokens(plan.inputs.kv_lease_window_tokens),
      draft_window(plan.inputs.draft_window), speculative_backend(plan.inputs.speculative_backend),
      kv_storage(plan.inputs.kv_storage), proposal_head(plan.inputs.proposal_head),
      vision_enabled(plan.inputs.features.vision), vision_rank(plan.inputs.features.vision_rank),
      use_cuda_graph(plan.inputs.use_cuda_graph), causal_scoring(plan.inputs.causal_scoring),
      kv_payload_bytes(plan.persistent.kv_payload_bytes),
      graph_allowance_bytes(plan.graph_allowance_bytes), workspace_plan(plan.workspace),
      elastic_arena(make_elastic_arena(plan, plan.persistent, device_in.device)),
      persistent(plan.persistent.bytes, tensor_parallel_zero_fill(plan)),
      workspace_storage(plan.workspace.rank_capacity(0, plan.inputs.features.vision_rank),
                        tensor_parallel_zero_fill(plan)),
      work(DeviceSpan{workspace_storage.base(), plan.workspace.general_capacity}),
      continuation_states(continuation_capacity), continuation_slots(continuation_capacity),
      shared_prefix_states(shared_prefix_capacity), shared_prefix_slots(shared_prefix_capacity),
      round_host(plan.inputs.causal_scoring
                     ? std::nullopt
                     : std::make_optional<PinnedHostBuffer>(sizeof(TokenId))),
      score_logprobs_host(plan.inputs.causal_scoring ? std::make_optional<PinnedHostBuffer>(
                                                           kCausalScoreTile * sizeof(float))
                                                     : std::nullopt),
      ordinary_host(
          !plan.inputs.causal_scoring && plan.inputs.speculative_backend == SpeculativeBackend::None
              ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::OrdinaryDecodeIngress) +
                                                     sizeof(qwen3_5::OrdinaryDecodeEgress))
              : std::nullopt),
      mtp_host(plan.inputs.speculative_backend == SpeculativeBackend::Mtp
                   ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::MtpDecodeIngress) +
                                                          sizeof(qwen3_5::MtpDecodeEgress))
                   : std::nullopt),
      dflash_host(plan.inputs.speculative_backend == SpeculativeBackend::DFlash2
                      ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::DFlashDecodeIngress) +
                                                             sizeof(qwen3_5::DFlashDecodeEgress) +
                                                             sizeof(qwen3_5::DFlashPrefillIngress))
                      : std::nullopt),
      context_source_ready_(device_in), context_completion_(device_in),
      context_transfer_timers_{CudaEventTimer(device_in, device_in.transfer_stream),
                               CudaEventTimer(device_in, device_in.transfer_stream),
                               CudaEventTimer(device_in, device_in.transfer_stream)} {
    if (&parameters != plan.inputs.parameters ||
        parameters.model.options() != plan.inputs.features) {
        throw std::invalid_argument("Program parameters do not match the frozen sequence plan");
    }
    if (workspace_plan.general_capacity == 0 ||
        workspace_plan.vision.has_value() != vision_enabled ||
        causal_scoring != plan.persistent.score_hidden.has_value() ||
        causal_scoring != (workspace_plan.causal_score != 0) ||
        (workspace_plan.vision &&
         workspace_plan.vision->general_capacity_bytes != workspace_plan.general_capacity)) {
        throw std::invalid_argument("Qwen3.5 workspace plan does not match startup features");
    }
    construct_tensor_parallel(plan, execution_in, peer_parameters_in);
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    if (!plan.inputs.context_cache.max_private_continuations ||
        !plan.inputs.context_cache.max_shared_prefixes) {
        throw std::logic_error("Qwen3.5 context cache options are not normalized");
    }
    const std::uint64_t address_capacity64 =
        static_cast<std::uint64_t>(*plan.inputs.context_cache.max_private_continuations) +
        *plan.inputs.context_cache.max_shared_prefixes;
    if (address_capacity64 == 0 || address_capacity64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Qwen3.5 KV address-space capacity exceeds uint32");
    }
    // One unpublished descriptor is reserved for the single in-flight active-capture snapshot.
    // Published private/shared address spaces remain bounded by P + S; the transaction slot lets a
    // full shared catalog replace one entry without releasing the old checkpoint before the new
    // snapshot has been prepared.
    if (address_capacity64 == std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Qwen3.5 KV transaction address capacity exceeds uint32");
    }
    const auto address_capacity      = static_cast<std::uint32_t>(address_capacity64 + 1U);
    const auto logical_page_capacity = [&](const DeviceKVPagePool& pool) {
        const HostKVPageLayout host_layout =
            plan_host_kv_page_layout(pool.geometry(), pool.host_replicas());
        const std::uint64_t host_pages =
            plan.inputs.context_cache.host_kv_capacity_bytes / host_layout.page_stride;
        const std::uint64_t total = static_cast<std::uint64_t>(pool.capacity_pages()) + host_pages;
        if (total > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Qwen3.5 logical KV page capacity exceeds uint32");
        }
        return static_cast<std::uint32_t>(total);
    };

    decoder = std::make_unique<qwen3_5::DecoderState>(
        device.device, backing, plan.persistent.decoder,
        elastic_arena ? elastic_arena->span() : DeviceSpan{});
    state_images =
        std::make_unique<qwen3_5::StateImageDevicePool>(backing, plan.persistent.state_images);
    // Mirrors first: a mirrored pool's Host records hold both ranks' halves, so every Host layout
    // and logical capacity below depends on them.
    if (peer) { attach_tensor_parallel_mirrors(); }
    text_host_kv_page_stride =
        plan_host_kv_page_layout(decoder->text_kv.page_pool().geometry(),
                                 decoder->text_kv.page_pool().host_replicas())
            .page_stride;
    text_kv_pages = std::make_unique<LogicalKVPageStore>(
        decoder->text_kv.page_pool(), logical_page_capacity(decoder->text_kv.page_pool()));
    text_kv_addresses = std::make_unique<KVAddressSpaceStore>(
        *text_kv_pages, decoder->text_kv.execution_tables(), address_capacity,
        decoder->text_kv.execution_tables().logical_page_capacity());
    require_host_tier_fits_ram(static_cast<std::uint64_t>(state_images->host_layout().image_bytes) *
                                   plan.inputs.context_cache.host_state_slots,
                               plan.inputs.context_cache.host_kv_capacity_bytes);
    if (plan.inputs.context_cache.host_state_slots != 0) {
        const std::uint64_t host_state_bytes =
            static_cast<std::uint64_t>(state_images->host_layout().image_bytes) *
            plan.inputs.context_cache.host_state_slots;
        StartupPhaseScope host_state_phase(startup_observer, StartupPhase::HostStatePin,
                                           StartupProgressUnit::Bytes, host_state_bytes);
        host_state_images = std::make_unique<qwen3_5::HostStatePool>(
            state_images->host_layout(), plan.inputs.context_cache.host_state_slots);
        host_state_phase.complete(host_state_bytes, host_state_bytes);
    }
    const std::uint64_t logical_state_capacity =
        static_cast<std::uint64_t>(state_images->slot_count()) +
        plan.inputs.context_cache.host_state_slots;
    if (logical_state_capacity > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Qwen3.5 logical StateImage capacity exceeds uint32");
    }
    state_store = std::make_unique<StateImageStore>(
        *state_images, host_state_images.get(), static_cast<std::uint32_t>(logical_state_capacity));
    pressure_private_owner_scratch_.resize(continuation_capacity);
    pressure_shared_owner_scratch_.resize(shared_prefix_capacity);
    pressure_private_drop_scratch_.resize(continuation_capacity);
    const std::size_t pressure_checkpoint_capacity =
        2U + context_cache.max_long_anchors_per_continuation.value_or(0U);
    for (auto& dropped : pressure_private_drop_scratch_) {
        dropped.reserve(pressure_checkpoint_capacity);
    }
    pressure_state_scratch_.reserve(static_cast<std::size_t>(logical_state_capacity));
    if (plan.persistent.replay_records) {
        replay_records.emplace(backing, *plan.persistent.replay_records);
        replay_fold.emplace(*replay_records, state_images->linear().all_layers_view());
    }
    if (replay_records.has_value() != (speculative_backend != SpeculativeBackend::None) ||
        replay_fold.has_value() != replay_records.has_value()) {
        throw std::logic_error("ReplaySSM records do not match the sequence plan");
    }
    if (plan.persistent.dflash) {
        CyclicKVCache* local = state_images->dflash_local();
        if (local == nullptr) {
            throw std::logic_error("DFlash StateImage has no local fixed state");
        }
        dflash.emplace(backing, *plan.persistent.dflash, *local);
    }
    if (dflash.has_value() != plan.inputs.features.dflash2()) {
        throw std::logic_error("DFlash state does not match the frozen sequence plan");
    }
    if (qwen3_5::PagedKVCache* backend = backend_kv_cache()) {
        backend_host_kv_page_stride = plan_host_kv_page_layout(backend->page_pool().geometry(),
                                                               backend->page_pool().host_replicas())
                                          .page_stride;
        backend_kv_pages = std::make_unique<LogicalKVPageStore>(
            backend->page_pool(), logical_page_capacity(backend->page_pool()));
        backend_kv_addresses = std::make_unique<KVAddressSpaceStore>(
            *backend_kv_pages, backend->execution_tables(), address_capacity,
            backend->execution_tables().logical_page_capacity());
    }
    pressure_text_page_scratch_.resize(text_kv_pages->capacity());
    pressure_text_selected_pages_.reserve(text_kv_pages->capacity());
    if (backend_kv_pages) {
        pressure_backend_page_scratch_.resize(backend_kv_pages->capacity());
        pressure_backend_selected_pages_.reserve(backend_kv_pages->capacity());
    }
    if (plan.inputs.context_cache.host_kv_capacity_bytes != 0) {
        std::vector<HostKVPageLayout> layouts;
        layouts.push_back(plan_host_kv_page_layout(decoder->text_kv.page_pool().geometry(),
                                                   decoder->text_kv.page_pool().host_replicas()));
        if (const qwen3_5::PagedKVCache* backend = backend_kv_cache()) {
            HostKVPageLayout backend_layout = plan_host_kv_page_layout(
                backend->page_pool().geometry(), backend->page_pool().host_replicas());
            if (backend_layout != layouts.front()) { layouts.push_back(std::move(backend_layout)); }
        }
        StartupPhaseScope host_kv_phase(
            startup_observer, StartupPhase::HostKvPin, StartupProgressUnit::Bytes,
            static_cast<std::uint64_t>(plan.inputs.context_cache.host_kv_capacity_bytes));
        host_kv_arena = std::make_unique<HostKVArena>(
            plan.inputs.context_cache.host_kv_capacity_bytes,
            std::span<const HostKVPageLayout>(layouts.data(), layouts.size()));
        host_kv_phase.complete(
            static_cast<std::uint64_t>(plan.inputs.context_cache.host_kv_capacity_bytes),
            static_cast<std::uint64_t>(plan.inputs.context_cache.host_kv_capacity_bytes));
        std::size_t minimum_stride = layouts.front().page_stride;
        for (const HostKVPageLayout& layout : layouts) {
            minimum_stride = std::min(minimum_stride, layout.page_stride);
        }
        const std::size_t extent_capacity =
            plan.inputs.context_cache.host_kv_capacity_bytes / minimum_stride;
        if (extent_capacity > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Qwen3.5 Host KV extent capacity exceeds uint32");
        }
        if (extent_capacity != 0) {
            host_kv_extents = std::make_unique<HostKVExtentStore>(
                *host_kv_arena, static_cast<std::uint32_t>(extent_capacity));
        }
    }

    io = qwen3_5::RoundState(backing, plan.persistent.round);
    if (io.mtp.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("round-state MTP extension does not match the sequence plan");
    }
    if (io.mtp_decode.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("MTP decode frame does not match the sequence plan");
    }
    if (io.ordinary.has_value() !=
        (!causal_scoring && speculative_backend == SpeculativeBackend::None)) {
        throw std::logic_error("ordinary decode frame does not match the sequence plan");
    }
    if (io.dflash_prefill.has_value() != (speculative_backend == SpeculativeBackend::DFlash2)) {
        throw std::logic_error("DFlash prefill scratch does not match the sequence plan");
    }
    if (io.dflash_decode.has_value() != (speculative_backend == SpeculativeBackend::DFlash2)) {
        throw std::logic_error("DFlash decode frame does not match the sequence plan");
    }
    prefill_hidden = plan.persistent.prefill_hidden.bind(backing);
    if (plan.persistent.score_hidden) {
        score_hidden = plan.persistent.score_hidden->bind(backing);
    }
    if (plan.persistent.token_counts) {
        token_counts = plan.persistent.token_counts->bind(backing);
    }
    if (plan.persistent.sampling_config) {
        sampling_config = plan.persistent.sampling_config->bind(backing);
    }
    if (plan.persistent.grammar_masks) {
        // Structured output: the pinned twin of the lane slots, then one staged prefill config
        // per lane (program/grammar_masks.cpp).
        grammar_masks_device = plan.persistent.grammar_masks->bind(backing);
        grammar_masks_host.emplace(grammar_staged_config_offset(grammar_masks_device.bytes()) +
                                   kMaximumConcurrency * sizeof(ops::SamplingConfig));
    }
    active_continuations.fill(continuation_capacity);
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) { lane_epochs[lane] = 1; }
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        SequenceState& sequence = continuation_states[index];
        sequence.ledger.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.prefix_identity.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.prefix_digests.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.long_anchors.reserve(context_cache.max_long_anchors_per_continuation.value_or(0));
        // One retained shared resume source can coexist with every fixed per-request candidate.
        sequence.shared_prefix_references.reserve(8U);
    }
    materialization_ledger_.reserve(static_cast<std::size_t>(capacity) + 1ULL);
    materialization_identity_.reserve(static_cast<std::size_t>(capacity) + 1ULL);
    materialization_prefix_digests_.reserve(static_cast<std::size_t>(capacity) + 1ULL);

    set_device_i32(io.text_kv_table_row, 0);
    if (!causal_scoring) { set_device_i32(io.backend_kv_table_row, 0); }
    if (peer) { bind_tensor_parallel_execution(); }

    host_tokens = round_host ? static_cast<TokenId*>(round_host->data()) : nullptr;
    if (ordinary_host) {
        ordinary_host_ingress = static_cast<qwen3_5::OrdinaryDecodeIngress*>(ordinary_host->data());
        ordinary_host_egress  = reinterpret_cast<qwen3_5::OrdinaryDecodeEgress*>(
            static_cast<unsigned char*>(ordinary_host->data()) +
            sizeof(qwen3_5::OrdinaryDecodeIngress));
        *ordinary_host_ingress = {};
        *ordinary_host_egress  = {};
    }
    if (mtp_host) {
        mtp_host_ingress = static_cast<qwen3_5::MtpDecodeIngress*>(mtp_host->data());
        mtp_host_egress  = reinterpret_cast<qwen3_5::MtpDecodeEgress*>(
            static_cast<unsigned char*>(mtp_host->data()) + sizeof(qwen3_5::MtpDecodeIngress));
        *mtp_host_ingress = {};
        *mtp_host_egress  = {};
    }
    if (dflash_host) {
        dflash_host_ingress = static_cast<qwen3_5::DFlashDecodeIngress*>(dflash_host->data());
        dflash_host_egress  = reinterpret_cast<qwen3_5::DFlashDecodeEgress*>(
            static_cast<unsigned char*>(dflash_host->data()) +
            sizeof(qwen3_5::DFlashDecodeIngress));
        *dflash_host_ingress        = {};
        *dflash_host_egress         = {};
        dflash_prefill_host_ingress = reinterpret_cast<qwen3_5::DFlashPrefillIngress*>(
            static_cast<unsigned char*>(dflash_host->data()) +
            sizeof(qwen3_5::DFlashDecodeIngress) + sizeof(qwen3_5::DFlashDecodeEgress));
        *dflash_prefill_host_ingress = {};
    }
    CUDA_CHECK(cudaMemsetAsync(io.rope_delta.data, 0, io.rope_delta.bytes(), device.stream));
    if (io.mtp) {
        CUDA_CHECK(
            cudaMemsetAsync(io.mtp->position.data, 0, io.mtp->position.bytes(), device.stream));
    }
    if (!causal_scoring) {
        CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
        CUDA_CHECK(
            cudaMemsetAsync(sampling_config.data, 0, sampling_config.bytes(), device.stream));
    }
    if (plan.persistent.qwen4) { construct_qwen4(plan, backing); }
    if (elastic_arena) {
        auto* offload = dynamic_cast<execution::Qwen4OffloadExperts*>(qwen4_routed.experts.get());
        if (offload == nullptr || (peer && !peer->elastic_arena)) {
            throw std::logic_error("elastic KV needs the offload expert backend on every rank");
        }
        std::vector<Qwen4ElasticVram::Rank> ranks;
        ranks.push_back(Qwen4ElasticVram::Rank{
            elastic_arena.get(), device.stream, &plan.persistent, &decoder->text_kv.page_pool(),
            decoder->mtp_cache() != nullptr ? &decoder->mtp_cache()->page_pool() : nullptr});
        if (peer) {
            ranks.push_back(Qwen4ElasticVram::Rank{
                peer->elastic_arena.get(), peer->device.stream, &*plan.peer_persistent,
                &peer->decoder->text_kv.page_pool(),
                peer->decoder->mtp_cache() != nullptr ? &peer->decoder->mtp_cache()->page_pool()
                                                      : nullptr});
        }
        const std::uint32_t step = (plan.inputs.elastic_kv.grow_tokens + kPagedKVPageSize - 1U) /
                                   static_cast<std::uint32_t>(kPagedKVPageSize);
        qwen4_routed.elastic = std::make_unique<Qwen4ElasticVram>(
            std::move(ranks), *offload, step, plan.inputs.elastic_kv.kv_budget_bytes);
    }
    synchronize_devices();
    if (use_cuda_graph) {
        StartupPhaseScope graph_phase(startup_observer, StartupPhase::CudaGraphPrepare);
        prepare_graphs();
        graph_phase.complete();
    }
    work.reset();
    work.reset_peak();
    if (peer) { peer->work.reset(); }
    workspace_logical_peak_bytes = 0;
    initialize_disk_cache();
}

void ProgramImpl::construct_qwen4(const SequencePlanImpl& plan, DeviceSpan backing) {
    const auto& config = parameters.model.config().text;
    const auto& q      = *plan.persistent.qwen4;
    if (!parameters.qwen4 ||
        (peer && (!peer_parameters || !peer_parameters->qwen4 || !tp_execution))) {
        throw std::logic_error("Text forward Program state requires its blocks on every rank");
    }
    execution::ForwardRuntime runtime;
    execution::ForwardRuntime peer_runtime;
    // Routed experts (MoE); a dense FFN binds no expert backend (runtime.experts stays null).
    if (config.qwen4_routed_experts()) {
        qwen4_routed.construct(*this, plan, backing, runtime, peer_runtime);
    }
    // PLE: the n-gram table gather (mapped, with prefetch), its staging and the convolution state.
    // At tp 2 rank 0 gathers once and both ranks upload the same staging (replicated PLE); each
    // rank keeps its own convolution states (its StateImage, mirrored slot operations).
    if (config.ple) { qwen4_ple.construct(*this, plan, backing, runtime, peer_runtime); }
    validation = ValidationHooks::from_environment(
        speculative_backend == SpeculativeBackend::Mtp,
        static_cast<std::int32_t>(config.num_hidden_layers),
        static_cast<std::size_t>(parameters.model.resources().public_token_count),
        peer ? "capy-flash engine, tp 2" : "capy-flash engine, tp 1", parameters.model.info().name);
    if (validation && validation->taps() != nullptr && !use_cuda_graph) {
        runtime.taps = validation->taps();
    }
    // The draft shortlist head. The prefix is a view of rank 0's output head; the extra rows
    // are copied byte-exact from the output head (at tp 2 from either rank's vocabulary half,
    // rank 1's over UVA) into rank 0's planes, in ascending token ID order.
    if (q.draft_ids && plan.inputs.draft_split) {
        // Split over the ranks: rank 0 scores shortlist rows [0, R) through a view of its half,
        // rank 1 rows [R, 2R), gathered (ascending token IDs) from rank 0's half over UVA and
        // from its own half.
        const auto& head          = parameters.text.output_head;
        const std::int32_t prefix = q.draft_prefix;
        const auto rows = prefix + static_cast<std::int32_t>(plan.inputs.draft_extras.size());
        const std::int32_t half = rows / 2;
        if (!peer || !peer_parameters || !plan.peer_persistent || !plan.peer_persistent->qwen4 ||
            plan.peer_persistent->qwen4->draft_extra != half || half > prefix ||
            peer->qwen4_draft_planes[0] == nullptr) {
            throw std::logic_error("Qwen4Exp split draft shortlist is not planned on both ranks");
        }
        std::vector<std::int32_t> ids(static_cast<std::size_t>(rows));
        for (std::int32_t i = 0; i < prefix; ++i) { ids[static_cast<std::size_t>(i)] = i; }
        std::copy(plan.inputs.draft_extras.begin(), plan.inputs.draft_extras.end(),
                  ids.begin() + prefix);
        Tensor device_ids = q.draft_ids->bind(backing);
        CUDA_CHECK(cudaMemcpyAsync(device_ids.data, ids.data(), device_ids.bytes(),
                                   cudaMemcpyHostToDevice, device.stream));
        const auto& peer_head    = peer_parameters->text.output_head;
        const std::int32_t rows0 = head.weight.n;
        const std::span<const std::int32_t> theirs(ids.data() + half,
                                                   static_cast<std::size_t>(half));
        const auto cut = std::lower_bound(theirs.begin(), theirs.end(), rows0);
        const std::vector<std::int32_t> from0(theirs.begin(), cut);
        std::vector<std::int32_t> from1(cut, theirs.end());
        for (auto& row : from1) { row -= rows0; }
        execution::gather_draft_rows(head.weight, from0, peer->qwen4_draft_planes, 0,
                                     device.stream);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        {
            const ScopedCurrentDevice rank1(peer->device.device);
            execution::gather_draft_rows(peer_head.weight, from1, peer->qwen4_draft_planes,
                                         static_cast<std::int32_t>(from0.size()),
                                         peer->device.stream);
            CUDA_CHECK(cudaStreamSynchronize(peer->device.stream));
        }
        execution::Qwen4DraftHead draft;
        draft.split       = true;
        draft.prefix_rows = half;
        draft.prefix      = execution::draft_row_view(head, 0, half);
        draft.peer_rows   = half;
        draft.peer        = execution::draft_rows_weight(peer_head, peer->qwen4_draft_planes, half);
        draft.token_ids   = static_cast<const std::int32_t*>(device_ids.data);
        qwen4_draft_head  = draft;
        runtime.draft_head = &*qwen4_draft_head;
    } else if (q.draft_ids) {
        const auto& head          = parameters.text.output_head;
        const std::int32_t prefix = q.draft_prefix;
        const std::int32_t extra  = q.draft_extra;
        if (static_cast<std::size_t>(extra) != plan.inputs.draft_extras.size() ||
            prefix > head.weight.n) {
            throw std::logic_error("Qwen4Exp draft shortlist plan does not match its layout");
        }
        execution::Qwen4DraftHead draft;
        draft.prefix_rows = prefix;
        draft.extra_rows  = extra;
        if (prefix > 0) { draft.prefix = execution::draft_row_view(head, 0, prefix); }
        Tensor ids = q.draft_ids->bind(backing);
        std::vector<std::int32_t> host(static_cast<std::size_t>(prefix + extra));
        for (std::int32_t i = 0; i < prefix; ++i) { host[static_cast<std::size_t>(i)] = i; }
        std::copy(plan.inputs.draft_extras.begin(), plan.inputs.draft_extras.end(),
                  host.begin() + prefix);
        CUDA_CHECK(cudaMemcpyAsync(ids.data, host.data(), ids.bytes(), cudaMemcpyHostToDevice,
                                   device.stream));
        draft.token_ids = static_cast<const std::int32_t*>(ids.data);
        if (extra > 0) {
            std::array<std::byte*, 3> planes{};
            for (std::size_t p = 0; p < planes.size(); ++p) {
                planes[p] = static_cast<std::byte*>(q.draft_planes[p].bind(backing).data);
            }
            draft.extra = execution::draft_rows_weight(head, planes, extra);
            // Rank 0 owns head rows [0, rows0): all of them at one device.
            const std::int32_t rows0 = head.weight.n;
            const auto split         = std::lower_bound(plan.inputs.draft_extras.begin(),
                                                        plan.inputs.draft_extras.end(), rows0);
            const std::vector<std::int32_t> own(plan.inputs.draft_extras.begin(), split);
            execution::gather_draft_rows(head.weight, own, planes, 0, device.stream);
            if (split != plan.inputs.draft_extras.end()) {
                if (!peer || !peer_parameters) {
                    throw std::logic_error("Qwen4Exp draft shortlist rows outside the output head");
                }
                std::vector<std::int32_t> theirs(split, plan.inputs.draft_extras.end());
                for (auto& row : theirs) { row -= rows0; }
                CUDA_CHECK(cudaStreamSynchronize(device.stream));
                const ScopedCurrentDevice rank1(peer->device.device);
                execution::gather_draft_rows(peer_parameters->text.output_head.weight, theirs,
                                             planes, static_cast<std::int32_t>(own.size()),
                                             peer->device.stream);
                CUDA_CHECK(cudaStreamSynchronize(peer->device.stream));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        qwen4_draft_head   = draft;
        runtime.draft_head = &*qwen4_draft_head;
    }
    forward_runtime_record = runtime;
    if (peer) {
        qwen4_peer_runtime_record = peer_runtime;
        tp_execution->qwen4       = &*qwen4_peer_runtime_record;
    }
}

ProgramImpl::~ProgramImpl() noexcept {
    if (peer) {
        if (peer->device.transfer_stream != nullptr) {
            (void)cudaStreamSynchronize(peer->device.transfer_stream);
        }
        if (peer->device.stream != nullptr) { (void)cudaStreamSynchronize(peer->device.stream); }
    }
    if (device.transfer_stream != nullptr) { (void)cudaStreamSynchronize(device.transfer_stream); }
    if (device.stream != nullptr) { (void)cudaStreamSynchronize(device.stream); }
}

std::vector<float> ProgramImpl::causal_score(PreparedPromptData&& prompt,
                                             std::uint32_t first_target) {
    if (!causal_scoring || !score_hidden || !score_logprobs_host ||
        workspace_plan.causal_score == 0) {
        throw std::logic_error("Program was not constructed for causal scoring");
    }
    if (speculative_backend != SpeculativeBackend::None || vision_enabled || use_cuda_graph ||
        context_cache.enabled) {
        throw std::logic_error("causal scoring Program has generation-only startup features");
    }
    const std::size_t token_count_size = prompt.token_ids.size();
    if (token_count_size < 2 || token_count_size > capacity) {
        throw std::invalid_argument("causal score token count must be in [2,capacity]");
    }
    if (first_target == 0 || first_target >= token_count_size) {
        throw std::invalid_argument("causal score first_target is outside the token window");
    }
    if (prompt.has_media()) {
        throw std::invalid_argument("causal scoring accepts text tokens only");
    }

    const auto token_count                     = static_cast<std::uint32_t>(token_count_size);
    const std::uint32_t predictor_count        = token_count - 1U;
    const std::uint32_t scored_predictor_begin = first_target - 1U;
    const std::uint32_t entitlement            = kv_pages_for_frontier(predictor_count);
    if (entitlement == 0) { throw std::logic_error("causal score has no KV entitlement"); }

    std::optional<StateImageHandle> state;
    std::optional<KVAddressSpaceHandle> address;
    const auto cleanup = [&] {
        bool released = true;
        if (address) {
            if (text_kv_addresses->active(*address)) { text_kv_addresses->deactivate(*address); }
            released = text_kv_addresses->release(*address) && released;
            address.reset();
        }
        if (state) {
            released = state_store->release(*state) && released;
            state.reset();
        }
        if (!released) { throw std::logic_error("causal score resources could not be released"); }
    };

    std::vector<float> output;
    output.reserve(token_count_size - first_target);
    std::vector<TokenId> staged_targets;
    staged_targets.reserve(kCausalScoreTile);
    std::vector<std::int32_t> staged_positions;
    staged_positions.reserve(kCausalScoreTile);
    std::uint32_t staged_columns = 0;
    if (validation) {
        validation->on_score_prompt(
            std::span<const std::int32_t>(prompt.token_ids.data(), prompt.token_ids.size()));
    }

    try {
        state = state_store->reserve_reset(device.stream);
        if (!state) { throw std::bad_alloc(); }
        address = text_kv_addresses->create_active(entitlement, 0, device.stream);
        if (!address) { throw std::bad_alloc(); }
        if (text_kv_addresses->bound_row(*address) != 0) {
            throw std::logic_error("causal score did not bind the unique Main KV row");
        }
        text_kv_addresses->ensure_mapped_to_tokens(*address, predictor_count, device.stream);

        const std::int32_t state_slot = state_store->physical_slot(*state);
        const auto flush              = [&] {
            if (staged_columns == 0) { return; }
            if (staged_columns != staged_targets.size() || staged_columns > kCausalScoreTile) {
                throw std::logic_error("causal score staging has an invalid shape");
            }
            work.reset();
            if (peer) { peer->work.reset(); }
            mark_workspace_usage(workspace_plan.causal_score);
            const auto columns = static_cast<std::int32_t>(staged_columns);
            Tensor logits      = work.alloc(
                DType::BF16, {dimension(parameters.model.config().text.vocab_size), columns});
            Tensor target_ids = work.alloc(DType::I32, {columns});
            Tensor logprobs   = work.alloc(DType::FP32, {columns});
            Tensor hidden     = score_hidden->slice(1, 0, columns);
            if (parameters.qwen4 && parameters.model.config().text.hyper_connection) {
                // The carried hidden is the wide residual: the Text output mixer (replicated, on
                // rank 0), then the head. (Without hyper-connections it is the final-normed one.)
                Tensor mixed = work.alloc(
                    DType::BF16, {dimension(parameters.model.config().text.hidden_size), columns});
                ops::hyper_connection_mix(hidden, parameters.qwen4->head.weights,
                                                       parameters.model.config().text.rms_norm_eps, mixed,
                                                       nullptr, work, device.stream);
                hidden = mixed;
            }
            if (peer) {
                project_score_tile_split(hidden, logits);
            } else {
                execution::project(hidden, parameters.text.output_head, logits, work,
                                                device.stream);
            }
            CUDA_CHECK(cudaMemcpyAsync(target_ids.data, staged_targets.data(), target_ids.bytes(),
                                                    cudaMemcpyHostToDevice, device.stream));
            ops::target_logprobs(logits, target_ids,
                                              dimension(parameters.model.resources().public_token_count),
                                              logprobs, device.stream);
            CUDA_CHECK(cudaMemcpyAsync(score_logprobs_host->data(), logprobs.data, logprobs.bytes(),
                                                    cudaMemcpyDeviceToHost, device.stream));
            synchronize_devices();
            if (validation) { validation->on_score_logits(logits, staged_positions); }
            staged_positions.clear();
            const auto* host = static_cast<const float*>(score_logprobs_host->data());
            output.insert(output.end(), host, host + staged_columns);
            staged_targets.clear();
            staged_columns = 0;
            work.reset();
            if (peer) { peer->work.reset(); }
        };

        std::uint32_t cursor = 0;
        while (cursor < predictor_count) {
            const std::uint32_t nominal = std::min(prefill_chunk, predictor_count - cursor);
            // At tp 2 the chunk runs the Text forward on both ranks; rank 1's KV row and StateImage
            // slot are rank 0's, replayed by the mirrors.
            execution::PrefillContext schedule_state{
                {device, parameters, work, state_images->linear(), nullptr, io, prefill_hidden,
                 prefill_chunk, proposal_head, tp_binding(), nullptr, forward_runtime()},
                decoder->text_kv.execution_view(text_kv_addresses->execution_row(*address)),
                {},
                decoder->text_kv,
                nullptr,
                nullptr,
                cursor,
                nullptr,
                nullptr,
                state_slot,
                state_slot,
                0,
                nullptr};
            mark_workspace_usage(workspace_plan.text_prefill);
            const execution::PrefillChunkResult result = execution::prefill_text_chunk(
                schedule_state, std::span<const TokenId>(prompt.token_ids), nominal, std::nullopt,
                false);
            if (result.finalized || result.processed_tokens == 0 ||
                result.processed_tokens > nominal) {
                throw std::logic_error("causal score Prefill made invalid progress");
            }
            const std::uint32_t chunk_begin = cursor;
            cursor += result.processed_tokens;
            text_kv_addresses->commit_frontier(*address, cursor);

            std::uint32_t selected = std::max(chunk_begin, scored_predictor_begin);
            while (selected < cursor) {
                const std::uint32_t available = cursor - selected;
                const std::uint32_t room      = kCausalScoreTile - staged_columns;
                const std::uint32_t count     = std::min(available, room);
                Tensor source =
                    prefill_hidden.slice(1, static_cast<std::int32_t>(selected - chunk_begin),
                                         static_cast<std::int32_t>(count));
                Tensor destination = score_hidden->slice(
                    1, static_cast<std::int32_t>(staged_columns), static_cast<std::int32_t>(count));
                CUDA_CHECK(::ninfer::memcpy_device_async(destination.data, source.data,
                                                         source.bytes(), device.stream));
                for (std::uint32_t column = 0; column < count; ++column) {
                    staged_targets.push_back(prompt.token_ids[selected + column + 1U]);
                    staged_positions.push_back(static_cast<std::int32_t>(selected + column));
                }
                selected += count;
                staged_columns += count;
                if (staged_columns == kCausalScoreTile) { flush(); }
            }
        }
        flush();
        if (output.size() != token_count_size - first_target) {
            throw std::logic_error("causal score produced the wrong number of logprobs");
        }
        cleanup();
        return output;
    } catch (...) {
        try {
            synchronize_devices();
        } catch (...) {}
        work.reset();
        if (peer) { peer->work.reset(); }
        try {
            cleanup();
        } catch (...) {}
        throw;
    }
}

void ProgramImpl::start_context_transfer_timer(runtime::ContextResourceClass resource) {
    context_transfer_timers_[context_resource_index(resource)].start();
}

void ProgramImpl::stop_context_transfer_timer(runtime::ContextResourceClass resource) {
    context_transfer_timers_[context_resource_index(resource)].record_stop();
}

runtime::ContextTransferObservation ProgramImpl::context_transfer_observation(
    runtime::ContextResourceClass resource, runtime::ContextTransferDirection direction,
    TransferWork work, std::uint32_t page_count, std::uint64_t state_images) const {
    const double elapsed_ns =
        static_cast<double>(
            context_transfer_timers_[context_resource_index(resource)].elapsed_ms()) *
        1'000'000.0;
    const std::uint64_t measured_ns =
        elapsed_ns >= static_cast<double>(std::numeric_limits<std::uint64_t>::max())
            ? std::numeric_limits<std::uint64_t>::max()
            : std::max<std::uint64_t>(1, static_cast<std::uint64_t>(elapsed_ns + 0.5));
    return runtime::ContextTransferObservation{
        .resource  = resource,
        .direction = direction,
        .units =
            resource == runtime::ContextResourceClass::State ? state_images : work.payload_bytes,
        .page_count = page_count,
        .work       = work,
        .elapsed_ns = measured_ns,
    };
}

MemorySummary ProgramImpl::memory_summary() const noexcept {
    MemorySummary out;
    out.device          = device.device;
    out.max_context     = capacity;
    out.kv_capacity     = kv_capacity;
    out.kv_cache        = kv_storage;
    const auto& weights = parameters.model.storage_stats();
    // The summary describes rank 0's device; at tp 2 the model's total spans both ranks.
    const std::size_t weight_bytes =
        peer ? weights.per_device_capacity_bytes[0] : weights.device_capacity_bytes;
    out.weights = ArenaMemorySummary{weight_bytes, weight_bytes, weight_bytes};
    out.sequence =
        ArenaMemorySummary{persistent.capacity(), persistent.used(), persistent.peak_used()};
    std::size_t active_handoff_bytes = 0;
    for (const RequestControl& request : requests) {
        if (request.prefill && request.prefill->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.prefill->vision->active_handoff_bytes());
        }
    }
    // Rank 0's workspace; at tp 2 its Vision plan is the receiver's unless rank 0 holds the tower.
    const VisionWorkspacePlan* vision_plan = workspace_plan.rank_vision(0, vision_rank);
    std::size_t active_workspace_bytes     = work.used();
    if (vision_plan != nullptr && active_handoff_bytes != 0) {
        active_workspace_bytes = std::max(active_workspace_bytes,
                                          vision_plan->handoff_offset_bytes + active_handoff_bytes);
    }
    out.workspace = ArenaMemorySummary{workspace_storage.capacity(), active_workspace_bytes,
                                       std::max(work.peak_used(), workspace_logical_peak_bytes)};
    if (vision_plan != nullptr) {
        out.vision_workspace = VisionWorkspaceMemorySummary{
            .aggregate_prompt_tokens = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(capacity, kMaximumPromptVisionTokens)),
            .max_item_tokens        = vision_plan->max_merged_tokens,
            .general_capacity_bytes = vision_plan->general_capacity_bytes,
            .encode_peak_bytes      = vision_plan->encode_peak_bytes,
            .handoff_offset_bytes   = vision_plan->handoff_offset_bytes,
            .handoff_capacity_bytes = vision_plan->handoff_capacity_bytes,
            .handoff_active_bytes   = active_handoff_bytes,
            .handoff_peak_bytes     = vision_handoff_peak_bytes,
        };
    }
    out.workspace_logical_peak_bytes = workspace_logical_peak_bytes;
    out.cuda_graph_allowance_bytes   = graph_allowance_bytes;
    out.cuda_graph_observed_bytes    = graph_observed_bytes;
    out.kv_payload_bytes             = kv_payload_bytes;
    out.routed_expert_slots =
        qwen4_routed.expert_slots +
        (qwen4_routed.elastic ? static_cast<std::uint32_t>(qwen4_routed.elastic->stats().columns)
                              : 0U);
    out.routed_expert_pool_bytes  = qwen4_routed.expert_pool_bytes;
    out.routed_expert_publish_lag = qwen4_routed.expert_publish_lag;
    out.prefill_chunk             = prefill_chunk;
    out.routed_expert_prompt_workspace_bytes =
        qwen4_prompt_base_chunk > 0 ? workspace_plan.qwen4_prompt : 0;
    if (host_state_images) {
        out.host_state_capacity_slots = host_state_images->capacity();
        out.host_state_occupied_slots = host_state_images->occupied();
    }
    if (host_kv_arena) {
        out.host_kv_capacity_bytes = host_kv_arena->capacity_bytes();
        out.host_kv_occupied_bytes = host_kv_arena->occupied_bytes();
    }
    return out;
}

void ProgramImpl::reset_memory_peaks() noexcept {
    persistent.reset_peak();
    work.reset_peak();
    std::size_t active_handoff_bytes = 0;
    for (const RequestControl& request : requests) {
        if (request.prefill && request.prefill->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.prefill->vision->active_handoff_bytes());
        }
    }
    vision_handoff_peak_bytes              = active_handoff_bytes;
    workspace_logical_peak_bytes           = work.used();
    const VisionWorkspacePlan* vision_plan = workspace_plan.rank_vision(0, vision_rank);
    if (vision_plan != nullptr && active_handoff_bytes != 0) {
        workspace_logical_peak_bytes = std::max(
            workspace_logical_peak_bytes, vision_plan->handoff_offset_bytes + active_handoff_bytes);
    }
}


} // namespace ninfer::models::qwen3_5::detail

#include "models/qwen3_5/program/tp2/program_impl_tp2.inc"
