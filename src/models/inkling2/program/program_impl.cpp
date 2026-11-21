#include "models/inkling2/program/program_impl.h"

#include "models/inkling2/frontend/prepared_prompt.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::inkling2::detail {
namespace {

using Access = RuntimeContractAccess;

[[nodiscard]] double seconds_between(std::chrono::steady_clock::time_point begin,
                                     std::chrono::steady_clock::time_point end) noexcept {
    return std::chrono::duration<double>(end - begin).count();
}

[[nodiscard]] std::uint64_t
nanoseconds_between(std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point end) noexcept {
    const auto count = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    return count > 0 ? static_cast<std::uint64_t>(count) : 0;
}

void require_unconstrained(runtime::TokenMaskProvider* masks, std::size_t rows) {
    if (masks == nullptr) { return; }
    for (std::size_t row = 0; row < rows; ++row) {
        if (masks->constrained(row)) {
            throw RequestError(RequestErrorKind::UnsupportedOutputCombination,
                               "Inkling 2 does not support output constraints");
        }
    }
}

} // namespace

ProgramImpl::ProgramImpl(SequencePlanImpl plan, std::unique_ptr<ProgramExecution> execution)
    : plan_(plan), execution_(std::move(execution)), lanes_(plan.layout.max_concurrency) {
    if (execution_ == nullptr || plan_.layout.max_concurrency == 0 ||
        plan_.layout.max_concurrency > kMaximumConcurrency || plan_.layout.prefill_chunk == 0) {
        throw std::logic_error("Inkling 2 Program plan is invalid");
    }
    transport_ = execution_->tp_transport();
}

std::vector<PrefixDigest> prefix_digests(std::span<const TokenId> tokens) {
    std::vector<PrefixDigest> out(tokens.size() + 1U);
    PrefixDigest digest{0x6a09e667f3bcc909ULL, 0xbb67ae8584caa73bULL};
    out[0] = digest;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const auto token = static_cast<std::uint64_t>(static_cast<std::uint32_t>(tokens[i]));
        digest[0]        = digest[0] * 0x100000001b3ULL + token + 1U;
        digest[1]        = (digest[1] ^ (token + 0x9e3779b97f4a7c15ULL)) * 0xff51afd7ed558ccdULL;
        out[i + 1]       = digest;
    }
    return out;
}

std::uint64_t ProgramImpl::service_work(const runtime::RequestPlanSummary& summary) const {
    // One scheduler quantum per prefill chunk of the suffix and per decode round after the
    // prefill's token.
    const std::uint64_t suffix = summary.prompt_tokens - summary.reusable_prompt_tokens;
    const std::uint64_t chunks = 1ULL + (suffix - 1ULL) / plan_.layout.prefill_chunk;
    return chunks +
           (summary.effective_output_tokens == 0 ? 0ULL : summary.effective_output_tokens - 1ULL);
}

RequestBasePlan ProgramImpl::plan_request(const PreparedPrompt& prompt,
                                          const runtime::ResolvedExecutionOptions& options) {
    const std::span<const TokenId> tokens = prompt.token_ids();
    if (tokens.empty()) { throw std::invalid_argument("prompt must contain tokens"); }
    if (tokens.size() > plan_.layout.capacity) {
        throw std::invalid_argument("prompt exceeds configured context capacity");
    }
    auto base                           = std::make_unique<RequestBasePlanImpl>();
    runtime::RequestPlanSummary& out    = base->summary;
    out.prompt_tokens                   = static_cast<std::uint32_t>(tokens.size());
    out.reusable_prompt_tokens          = 0;
    out.requested_output_tokens         = options.requested_output_tokens;
    const std::uint32_t capacity_output = plan_.layout.capacity - out.prompt_tokens + 1U;
    out.effective_output_tokens = std::min(options.requested_output_tokens, capacity_output);
    out.effective_limit_reason  = options.requested_output_tokens <= capacity_output
                                      ? FinishReason::OutputLimit
                                      : FinishReason::ContextCapacity;
    out.prefix_reuse_path       = PrefixReusePath::Root;
    out.publish_continuation    = plan_.layout.context_cache && options.allow_prefix_reuse &&
                               PreparedPromptAccess::view(prompt).identity.reusable;
    out.service_work_quanta = service_work(out);
    base->sampling          = options.sampling;
    if (plan_.layout.context_cache && options.allow_prefix_reuse &&
        PreparedPromptAccess::view(prompt).identity.reusable) {
        base->digests = prefix_digests(tokens);
    }
    return RequestBasePlan(std::move(base));
}

std::vector<float> ProgramImpl::causal_score(PreparedPrompt&& prompt, std::uint32_t first_target) {
    const std::span<const TokenId> tokens = prompt.token_ids();
    if (tokens.size() < 2 || tokens.size() > plan_.layout.capacity || first_target == 0 ||
        first_target >= tokens.size()) {
        throw std::invalid_argument("causal score window is invalid");
    }
    if (transaction_ ||
        std::any_of(lanes_.begin(), lanes_.end(), [](const Lane& lane) { return lane.occupied; })) {
        throw std::logic_error("causal scoring overlaps an active sequence");
    }
    resident_.reset();
    execution_->reset(0);
    return execution_->score(tokens, first_target);
}

runtime::IdentityMaterializationAssessment
ProgramImpl::assessment(const runtime::RequestPlanSummary& summary) const {
    runtime::IdentityMaterializationAssessment identity;
    identity.physical_status                     = runtime::MaterializationPhysicalStatus::Feasible;
    identity.source_mode                         = runtime::PrivateSourceMode::ConsumeToActive;
    identity.machine_work.remaining_prefill_work = runtime::make_prefill_work(
        summary.reusable_prompt_tokens, summary.prompt_tokens - summary.reusable_prompt_tokens, 0,
        0, plan_.layout.prefill_chunk);
    identity.machine_work.reused_prompt_tokens = summary.reusable_prompt_tokens;
    identity.pressure_may_change_machine_work  = false;
    identity.expandable                        = false;
    return identity;
}

bool ProgramImpl::resident_matches(const ContinuationHandle& handle) const noexcept {
    return resident_ && Access::owner(handle) == this && Access::index(handle) == 0 &&
           Access::epoch(handle) == resident_->generation;
}

CheckpointSummary ProgramImpl::checkpoint(runtime::CheckpointKind kind, std::uint32_t ordinal,
                                          std::uint32_t frontier) const {
    const std::vector<PrefixDigest> digests =
        prefix_digests(std::span<const TokenId>(resident_->tokens).first(frontier));
    CheckpointSummary out;
    out.ref   = runtime::CheckpointRef{.kind = kind, .frontier = frontier, .ordinal = ordinal};
    out.scope = runtime::CheckpointScope::Private;
    out.shortlist_key = PrefixShortlistKey{
        .digests = digests.back(), .frontier = frontier, .identity_tag = kPrefixIdentityTag};
    out.state_residency = runtime::ReplicaResidency::DeviceOnly;
    out.required_kv     = TargetKVRequirement{.main_frontier = frontier,
                                              .main_pages = (frontier + plan_.layout.page_tokens - 1U) /
                                                        plan_.layout.page_tokens};
    out.rebuild_work    = runtime::make_prefill_work(0, frontier, 0, 0, plan_.layout.prefill_chunk);
    return out;
}

ContinuationSummary ProgramImpl::resident_summary() const {
    ContinuationSummary out;
    out.endpoint = checkpoint(runtime::CheckpointKind::SessionEndpoint, 0, resident_->executed);
    if (resident_->anchor != 0 && resident_->anchor < resident_->executed) {
        out.long_anchors.push_back(
            checkpoint(runtime::CheckpointKind::LongAnchor, 1, resident_->anchor));
    }
    return out;
}

std::optional<AdmissionCandidate>
ProgramImpl::inspect_admission(const PreparedPrompt& prompt, const RequestBasePlan& base,
                               runtime::LaneId destination, const ContinuationHandle* source,
                               std::optional<runtime::CheckpointRef> checkpoint, bool from_shared,
                               bool retain_source) {
    if (base.impl_ == nullptr || prompt.token_ids().size() != base.summary().prompt_tokens) {
        throw std::logic_error("admission plan does not match its prompt");
    }
    if (destination.value >= lanes_.size()) {
        throw std::logic_error("admission destination lane is out of range");
    }
    auto candidate         = std::make_unique<AdmissionCandidateImpl>();
    candidate->summary     = base.summary();
    candidate->destination = destination;
    candidate->sampling    = base.impl_->sampling;
    const bool lane_free   = !lanes_[destination.value].occupied;
    if (source == nullptr && !from_shared && !checkpoint) {
        candidate->identity = assessment(candidate->summary);
        // The root needs the one sequence state: free only while no continuation holds it.
        if (!lane_free || resident_) {
            candidate->identity.physical_status =
                runtime::MaterializationPhysicalStatus::Infeasible;
        }
        return AdmissionCandidate(std::move(candidate));
    }
    // One resident sequence, consumed by the request that continues it.
    if (from_shared || retain_source || source == nullptr || !checkpoint ||
        base.impl_->digests.empty() || !resident_matches(*source) || !lane_free) {
        return std::nullopt;
    }
    const std::uint32_t frontier = checkpoint->frontier;
    AdmissionSource kind         = AdmissionSource::Root;
    if (checkpoint->kind == runtime::CheckpointKind::SessionEndpoint &&
        frontier == resident_->executed) {
        kind = AdmissionSource::Endpoint;
    } else if (checkpoint->kind == runtime::CheckpointKind::LongAnchor &&
               checkpoint->ordinal == 1 && resident_->anchor != 0 &&
               frontier == resident_->anchor) {
        kind = AdmissionSource::Anchor;
    } else {
        return std::nullopt;
    }
    const std::span<const TokenId> tokens = prompt.token_ids();
    if (frontier == 0 || frontier >= tokens.size() ||
        !std::equal(tokens.begin(), tokens.begin() + frontier, resident_->tokens.begin())) {
        return std::nullopt;
    }
    candidate->summary.reusable_prompt_tokens = frontier;
    candidate->summary.prefix_reuse_path      = kind == AdmissionSource::Endpoint
                                                    ? PrefixReusePath::PrivateEndpoint
                                                    : PrefixReusePath::PrivateLongAnchor;
    candidate->summary.service_work_quanta    = service_work(candidate->summary);
    candidate->identity                       = assessment(candidate->summary);
    candidate->source                         = kind;
    candidate->resident_generation            = resident_->generation;
    return AdmissionCandidate(std::move(candidate));
}

std::unique_ptr<PressurePlanningSessionImpl>
ProgramImpl::begin_pressure_planning(std::span<const AdmissionCandidate* const> candidates,
                                     std::span<const runtime::PlanningCandidateId> candidate_ids,
                                     std::span<const ContinuationHandle* const> private_owners,
                                     std::span<const runtime::PlanningOwnerId> private_owner_ids,
                                     std::span<const SharedPrefixHandle* const> shared_owners) {
    if (candidates.size() != candidate_ids.size() ||
        private_owners.size() != private_owner_ids.size() || !shared_owners.empty()) {
        throw std::logic_error("Inkling 2 pressure planning inputs are invalid");
    }
    auto session      = std::make_unique<PressurePlanningSessionImpl>();
    session->revision = revision_;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const AdmissionCandidateImpl* candidate = candidates[i]->impl_.get();
        if (candidate != nullptr && candidate->source == AdmissionSource::Root) {
            session->root    = *candidate;
            session->root_id = candidate_ids[i];
        }
    }
    const ContinuationSummary owned = resident_ ? resident_summary() : ContinuationSummary{};
    std::vector<runtime::CheckpointRef> checkpoints;
    if (owned.endpoint) { checkpoints.push_back(owned.endpoint->ref); }
    for (const CheckpointSummary& anchor : owned.long_anchors) {
        checkpoints.push_back(anchor.ref);
    }
    session->recovery.reserve(private_owners.size() * checkpoints.size());
    for (std::size_t i = 0; i < private_owners.size(); ++i) {
        if (!resident_matches(*private_owners[i])) {
            throw std::logic_error("pressure owner is not the resident sequence");
        }
        session->outcomes.push_back(
            {.owner               = private_owner_ids[i],
             .disposition         = runtime::VictimDisposition::Evicted,
             .dropped_checkpoints = static_cast<std::uint32_t>(checkpoints.size())});
        for (const runtime::CheckpointRef& ref : checkpoints) {
            session->recovery.push_back(checkpoint_recovery_work(*private_owners[i], ref));
            session->impacts.push_back({.owner                = private_owner_ids[i],
                                        .checkpoint           = ref,
                                        .target_recovery_work = session->recovery.back(),
                                        .survives             = false});
        }
        if (session->root) {
            session->root->evicted_owner       = private_owner_ids[i];
            session->root->resident_generation = resident_->generation;
        }
    }
    if (session->root) {
        const bool lane_free = !lanes_.at(session->root->destination.value).occupied;
        session->root->identity.physical_status =
            lane_free ? runtime::MaterializationPhysicalStatus::Feasible
                      : runtime::MaterializationPhysicalStatus::Infeasible;
    }
    return session;
}

std::vector<runtime::CheckpointRecoveryAlternativeWork>
ProgramImpl::checkpoint_recovery_work(const ContinuationHandle& owner,
                                      runtime::CheckpointRef checkpoint) const {
    if (!resident_matches(owner)) { throw std::logic_error("checkpoint owner is stale"); }
    return {runtime::CheckpointRecoveryAlternativeWork{
        .prefill =
            runtime::make_prefill_work(0, checkpoint.frontier, 0, 0, plan_.layout.prefill_chunk)}};
}

ReleaseResult ProgramImpl::release_continuation(ContinuationHandle&& continuation) noexcept {
    if (!resident_matches(continuation)) { return {}; }
    resident_.reset();
    Access::consume(continuation);
    bump_revision();
    return {.status = runtime::ConsumeStatus::Consumed};
}

std::optional<AdmissionCandidate> ProgramImpl::seal_identity(const AdmissionCandidate& candidate,
                                                             runtime::FinalScheduleIntent intent) {
    if (candidate.impl_ == nullptr) { throw std::logic_error("admission candidate is empty"); }
    if (!intent.shared_capture_frontiers.empty()) {
        throw std::logic_error("Inkling 2 Program does not capture shared prefixes");
    }
    if (candidate.impl_->identity.physical_status !=
            runtime::MaterializationPhysicalStatus::Feasible ||
        lanes_.at(candidate.impl_->destination.value).occupied) {
        return std::nullopt;
    }
    return AdmissionCandidate(std::make_unique<AdmissionCandidateImpl>(*candidate.impl_));
}

runtime::ContextTransactionReserveStatus
ProgramImpl::start_resource_transaction(AdmissionCandidate&& admission, PreparedPrompt&& prompt,
                                        runtime::CancellationFlagView cancellation) {
    if (transaction_) { throw std::logic_error("Inkling 2 Program already has a transaction"); }
    if (admission.impl_ == nullptr || !prompt ||
        prompt.token_ids().size() != admission.impl_->summary.prompt_tokens) {
        throw std::logic_error("resource plan does not match its prompt");
    }
    if (cancellation.requested()) { return runtime::ContextTransactionReserveStatus::Aborted; }
    const AdmissionCandidateImpl& plan = *admission.impl_;
    const runtime::LaneId lane         = plan.destination;
    const bool uses_resident           = plan.source != AdmissionSource::Root || plan.evicted_owner;
    if (lane.value >= lanes_.size() || lanes_[lane.value].occupied ||
        (uses_resident && (!resident_ || resident_->generation != plan.resident_generation)) ||
        (!uses_resident && resident_)) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    transaction_.emplace(OpenMaterialization{
        .lane      = lane,
        .admission = plan,
        .prompt    = std::move(prompt),
    });
    bump_revision();
    return runtime::ContextTransactionReserveStatus::Reserved;
}

ContextTransactionProgress
ProgramImpl::progress_context_transaction(runtime::CancellationFlagView cancellation) {
    if (!transaction_ || transaction_->published) {
        throw std::logic_error("Inkling 2 Program has no progressable transaction");
    }
    OpenMaterialization& open          = *transaction_;
    const AdmissionCandidateImpl& plan = open.admission;
    MaterializationResult result;
    if (plan.source != AdmissionSource::Root) {
        result.source = MaterializationSourceResult{.mode = runtime::PrivateSourceMode::Retain};
    }
    if (plan.evicted_owner) { result.victims.push_back({.owner = *plan.evicted_owner}); }
    if (cancellation.requested()) {
        result.status = runtime::ContextTransactionStatus::Aborted;
        return result;
    }
    const std::uint32_t index  = open.lane.value;
    const std::uint32_t reused = plan.summary.reusable_prompt_tokens;
    switch (plan.source) {
    case AdmissionSource::Root:
        execution_->reset(index);
        break;
    case AdmissionSource::Endpoint:
        break;
    case AdmissionSource::Anchor:
        execution_->restore_image(index);
        break;
    }
    resident_.reset();
    Lane& lane                            = lanes_.at(index);
    const std::span<const TokenId> prompt = open.prompt.token_ids();
    lane                                  = Lane{};
    lane.occupied                         = true;
    lane.epoch                            = next_epoch_++;
    lane.tokens.assign(prompt.begin(), prompt.end());
    lane.tokens.reserve(prompt.size() + plan.summary.effective_output_tokens);
    lane.prompt_tokens = static_cast<std::uint32_t>(prompt.size());
    lane.prefilled     = reused;
    lane.executed      = reused;
    lane.reused        = reused;
    lane.path          = plan.summary.prefix_reuse_path;
    lane.summary       = plan.summary;
    lane.sampling      = plan.sampling;
    if (const std::uint32_t K = execution_->draft_tokens(); K > 0) {
        lane.speculative =
            SpeculativeStats{.backend               = SpeculativeBackend::Mtp,
                             .enabled               = true,
                             .draft_window          = K,
                             .accepted_per_position = std::vector<std::uint64_t>(K, 0)};
    }
    lane.started     = Clock::now();
    open.published   = true;
    result.status    = runtime::ContextTransactionStatus::Published;
    result.published = StartResult{.sequence = handle_of(index)};
    if (plan.source != AdmissionSource::Root) {
        result.source =
            MaterializationSourceResult{.mode = runtime::PrivateSourceMode::ConsumeToActive};
    }
    if (plan.evicted_owner) {
        result.victims.front().disposition        = runtime::VictimDisposition::Evicted;
        result.victims.front().pressure_committed = true;
    }
    bump_revision();
    return result;
}

void ProgramImpl::finalize_context_transaction() noexcept { transaction_.reset(); }

bool ProgramImpl::has_context_transaction() const noexcept { return transaction_.has_value(); }

PrefillProgress ProgramImpl::advance_prefill(SequenceHandle sequence,
                                             runtime::ExecutionTiming* failed_timing,
                                             runtime::TokenMaskProvider* masks) {
    require_unconstrained(masks, 1);
    Lane* lane = lane_of(sequence);
    if (lane == nullptr || lane->poisoned || lane->prefilled >= lane->prompt_tokens) {
        throw std::logic_error("prefill sequence is not prefilling");
    }
    const std::uint32_t index = Access::lane(sequence).value;
    const std::uint32_t chunk =
        std::min(plan_.layout.prefill_chunk, lane->prompt_tokens - lane->prefilled);
    const bool last  = lane->prefilled + chunk == lane->prompt_tokens;
    const auto begin = Clock::now();
    std::optional<TokenId> selected;
    try {
        selected = execution_->prefill(
            index, std::span<const TokenId>(lane->tokens).subspan(lane->prefilled, chunk),
            lane->prefilled, last ? &lane->sampling : nullptr);
    } catch (...) {
        if (failed_timing != nullptr) {
            failed_timing->device_wait_ns = nanoseconds_between(begin, Clock::now());
        }
        throw;
    }
    const auto end = Clock::now();
    lane->prefilled += chunk;
    lane->executed = lane->prefilled;
    if (last && lane->summary.publish_continuation) {
        // The prompt anchor: the state after the prompt, before any generated token.
        execution_->save_image(index);
        lane->anchor = lane->prompt_tokens;
    }
    lane->timings.prefill_seconds += seconds_between(begin, end);

    PrefillProgress progress;
    progress.summary                 = runtime::BeginSummary{.prompt_tokens        = lane->prompt_tokens,
                                                             .reused_prompt_tokens = lane->reused,
                                                             .prefix_reuse_path    = lane->path};
    progress.processed_prompt_tokens = chunk;
    progress.complete                = last;
    progress.timing.device_wait_ns   = nanoseconds_between(begin, end);
    if (last) {
        if (!selected) { throw std::logic_error("Inkling 2 prefill selected no token"); }
        pending_                 = Pending{};
        pending_.transaction     = next_transaction_++;
        pending_.tokens[0]       = *selected;
        pending_.row_counts[0]   = 1;
        pending_.prefill[0]      = true;
        const SequenceHandle row = sequence;
        progress.pending.emplace(Access::make_pending(
            this, pending_.transaction, std::span<const SequenceHandle>(&row, 1),
            std::span<const TokenId>(pending_.tokens.data(), 1),
            std::span<const std::int32_t>(pending_.row_counts.data(), 1), 1, progress.timing));
    }
    return progress;
}

// The frontier from which speculative rounds stop paying: every verified column rereads the
// global visits' keys, and the drafter's own global attention grows with the context. Measured
// from the server (2x RTX 5070 Ti, K = 3, temperature 0, 256-token summaries): decode with /
// without the drafter 126.8 / 100.3 tok/s at 31k, 102.2 / 88.3 at 64k, 77.8 / 77.3 at 98k, 62.4
// / 72.1 at 130k, 43.7 / 60.3 at 194k. Beyond it rows take ordinary steps, which do not feed the
// drafter's context: a sequence's frontier only grows, and a resume below the limit restarts at a
// prompt anchor or an endpoint whose context the drafter already holds. The choice depends on the
// position alone, and both kinds of step give the target's tokens.
constexpr std::uint32_t kSpeculativeContextLimit = 98304;

std::optional<std::uint32_t>
ProgramImpl::speculative_extent(const Lane& lane,
                                const runtime::RoundBudget& budget) const noexcept {
    // A row with forced tokens to execute takes an ordinary step (like the Qwen backends' rows
    // they cannot speculate), and so does a row whose frontier has reached
    // kSpeculativeContextLimit. Any other row, greedy or sampled, verifies up to K drafts within
    // its token budget (the round licenses at most extent + 1) and with the round's K + 1 positions
    // inside the context.
    const std::uint32_t K = execution_->draft_tokens();
    if (K == 0 || lane.tokens.size() != std::size_t(lane.executed) + 1U ||
        lane.executed >= kSpeculativeContextLimit ||
        lane.executed + K + 1U > plan_.layout.capacity) {
        return std::nullopt;
    }
    return std::min<std::uint32_t>(K, budget.generated_tokens_remaining - 1U);
}

PendingBatch ProgramImpl::decode(std::span<const SequenceHandle> sequences,
                                 std::span<const runtime::RoundBudget> budgets,
                                 runtime::ExecutionTiming* failed_timing,
                                 runtime::TokenMaskProvider* masks) {
    if (sequences.empty() || sequences.size() != budgets.size() ||
        sequences.size() > kMaximumConcurrency) {
        throw std::logic_error("decode round membership is invalid");
    }
    require_unconstrained(masks, sequences.size());
    pending_              = Pending{};
    pending_.transaction  = next_transaction_++;
    const std::uint32_t K = execution_->draft_tokens();
    pending_.row_stride   = K + 1U;
    const auto begin      = Clock::now();
    for (std::size_t row = 0; row < sequences.size(); ++row) {
        Lane* lane = lane_of(sequences[row]);
        if (lane == nullptr || lane->poisoned || lane->prefilled != lane->prompt_tokens ||
            lane->tokens.size() <= lane->prompt_tokens ||
            budgets[row].generated_tokens_remaining == 0) {
            throw std::logic_error("decode sequence is not decodable");
        }
        if (lane->tokens.size() >= plan_.layout.capacity) {
            throw std::logic_error("decode sequence reached its context capacity");
        }
        const auto row_begin = Clock::now();
        TokenId* out         = pending_.tokens.data() + row * pending_.row_stride;
        try {
            if (const auto extent = speculative_extent(*lane, budgets[row])) {
                const std::vector<TokenId> licensed = execution_->speculate(
                    Access::lane(sequences[row]).value, lane->tokens.back(), lane->executed,
                    *extent, lane->sampling,
                    std::span<const TokenId>(lane->tokens).subspan(lane->prompt_tokens));
                std::copy(licensed.begin(), licensed.end(), out);
                pending_.row_counts[row]  = static_cast<std::int32_t>(licensed.size());
                pending_.speculative[row] = true;
                pending_.frontier[row]    = lane->executed;
                lane->executed += static_cast<std::uint32_t>(licensed.size());
                SpeculativeStats& stats = lane->speculative;
                const auto accepted     = static_cast<std::uint32_t>(licensed.size()) - 1U;
                stats.rounds += 1;
                stats.drafted_tokens += *extent;
                stats.accepted_tokens += accepted;
                for (std::uint32_t i = 0; i < accepted; ++i) {
                    stats.accepted_per_position[i] += 1;
                }
            } else {
                out[0] = execution_->decode(
                    Access::lane(sequences[row]).value,
                    std::span<const TokenId>(lane->tokens).subspan(lane->executed), lane->executed,
                    lane->sampling,
                    std::span<const TokenId>(lane->tokens).subspan(lane->prompt_tokens));
                lane->executed           = static_cast<std::uint32_t>(lane->tokens.size());
                pending_.row_counts[row] = 1;
                if (K > 0) { lane->speculative.fallback_steps += 1; }
            }
        } catch (...) {
            lane->poisoned = true;
            if (failed_timing != nullptr) {
                failed_timing->device_wait_ns = nanoseconds_between(begin, Clock::now());
            }
            throw;
        }
        lane->timings.decode_seconds += seconds_between(row_begin, Clock::now());
    }
    runtime::ExecutionTiming timing;
    timing.device_wait_ns = nanoseconds_between(begin, Clock::now());
    return Access::make_pending(
        this, pending_.transaction, sequences,
        std::span<const TokenId>(pending_.tokens.data(), sequences.size() * pending_.row_stride),
        std::span<const std::int32_t>(pending_.row_counts.data(), sequences.size()),
        pending_.row_stride, timing);
}

runtime::ExecutionTiming ProgramImpl::append_forced_tokens(
    std::span<const SequenceHandle> sequences, std::span<const TokenId> row_major_tokens,
    std::uint32_t row_stride,
    std::span<const std::optional<std::uint32_t>> prefix_execution_splits) {
    if (sequences.empty() || row_stride == 0 ||
        row_major_tokens.size() != sequences.size() * std::size_t(row_stride) ||
        prefix_execution_splits.size() != sequences.size()) {
        throw std::logic_error("forced token rows are invalid");
    }
    for (std::size_t row = 0; row < sequences.size(); ++row) {
        const Lane* lane = lane_of(sequences[row]);
        if (lane == nullptr || lane->poisoned || lane->prefilled != lane->prompt_tokens ||
            lane->tokens.size() + row_stride > plan_.layout.capacity ||
            prefix_execution_splits[row]) {
            throw std::logic_error("forced tokens do not fit their sequence");
        }
    }
    // The forced tokens join the committed sequence; the next decode round executes them with the
    // last generated token (tokens[executed..) in one forward call).
    for (std::size_t row = 0; row < sequences.size(); ++row) {
        Lane& lane       = *lane_of(sequences[row]);
        const auto first = row_major_tokens.begin() + std::ptrdiff_t(row * row_stride);
        lane.tokens.insert(lane.tokens.end(), first, first + std::ptrdiff_t(row_stride));
    }
    return {};
}

CommitResult ProgramImpl::commit(PendingBatch&& pending,
                                 std::span<const runtime::CommitDecision> decisions,
                                 runtime::CommitObservation) {
    if (Access::owner(pending) != this || Access::transaction(pending) != pending_.transaction ||
        pending_.transaction == 0) {
        throw std::logic_error("pending batch does not belong to this Program round");
    }
    const std::span<const SequenceHandle> rows = Access::rows(pending);
    if (decisions.size() != rows.size()) {
        throw std::logic_error("commit decisions are not row aligned");
    }
    for (std::size_t row = 0; row < rows.size(); ++row) {
        const Lane* lane             = lane_of(rows[row]);
        const std::uint32_t produced = static_cast<std::uint32_t>(pending_.row_counts[row]);
        if (lane == nullptr || decisions[row].accepted_tokens > produced ||
            (decisions[row].cancelled && !decisions[row].terminal) ||
            (!decisions[row].terminal && decisions[row].accepted_tokens != produced) ||
            decisions[row].prefix_execution_split_after) {
            throw std::logic_error("commit decision is invalid for the round");
        }
    }
    CommitResult result;
    result.row_count = rows.size();
    const auto now   = Clock::now();
    for (std::size_t row = 0; row < rows.size(); ++row) {
        Lane& lane                             = *lane_of(rows[row]);
        const runtime::CommitDecision decision = decisions[row];
        const std::uint32_t produced = static_cast<std::uint32_t>(pending_.row_counts[row]);
        if (decision.accepted_tokens > 0U && !decision.cancelled) {
            if (lane.tokens.size() == lane.prompt_tokens) {
                lane.first_token                 = now;
                lane.timings.first_token_seconds = seconds_between(lane.started, now);
                lane.timings.prompt_wall_seconds = lane.timings.first_token_seconds;
            }
            const TokenId* licensed = pending_.tokens.data() + row * pending_.row_stride;
            lane.tokens.insert(lane.tokens.end(), licensed, licensed + decision.accepted_tokens);
            lane.timings.generation_wall_seconds = seconds_between(lane.first_token, now);
            // A terminal row that keeps fewer licensed tokens than the round executed: the state
            // (and the resident endpoint) holds only the kept ones.
            if (pending_.speculative[row] && decision.accepted_tokens < produced) {
                try {
                    execution_->settle(Access::lane(rows[row]).value, decision.accepted_tokens);
                    lane.executed = pending_.frontier[row] + decision.accepted_tokens;
                } catch (...) { lane.poisoned = true; }
            }
        }
        // A round that keeps none of its tokens leaves state the lane cannot describe: the
        // sequence is not catalogued (its output is unaffected).
        if (pending_.speculative[row] && decision.accepted_tokens == 0U) { lane.poisoned = true; }
        lane.timings.total_seconds   = seconds_between(lane.started, now);
        result.rows[row].timings     = lane.timings;
        result.rows[row].speculative = lane.speculative;
        result.rows[row].disposition = decision.cancelled
                                           ? runtime::CommitDisposition::CancelledReleased
                                       : decision.terminal ? runtime::CommitDisposition::Finishable
                                                           : runtime::CommitDisposition::Active;
        if (decision.cancelled) { release_lane(lane); }
    }
    Access::consume(pending);
    pending_ = Pending{};
    return result;
}

DiscardResult ProgramImpl::abort_pending(PendingBatch&& pending) noexcept {
    if (Access::owner(pending) != this || Access::transaction(pending) != pending_.transaction ||
        pending_.transaction == 0) {
        return {};
    }
    const std::span<const SequenceHandle> rows = Access::rows(pending);
    for (const SequenceHandle& row : rows) {
        if (Lane* lane = lane_of(row); lane != nullptr) { release_lane(*lane); }
    }
    DiscardResult result{.status = runtime::ConsumeStatus::Consumed, .row_count = rows.size()};
    Access::consume(pending);
    pending_ = Pending{};
    return result;
}

FinishResult ProgramImpl::finish(SequenceHandle sequence) noexcept {
    Lane* lane = lane_of(sequence);
    if (lane == nullptr) { return {}; }
    FinishResult result;
    result.status      = runtime::ConsumeStatus::Consumed;
    result.disposition = runtime::FinishDisposition::Released;
    result.timings     = lane->timings;
    result.speculative = lane->speculative;
    if (lane->summary.publish_continuation && !lane->poisoned && lane->executed != 0) {
        try {
            resident_      = Resident{.generation = next_generation_++,
                                      .tokens     = lane->tokens,
                                      .executed   = lane->executed,
                                      .anchor     = lane->anchor};
            result.summary = resident_summary();
            result.continuation.emplace(Access::make_continuation(this, 0, resident_->generation));
            result.disposition = runtime::FinishDisposition::Catalogued;
        } catch (...) {
            resident_.reset();
            result.summary = {};
            result.continuation.reset();
        }
    }
    release_lane(*lane);
    return result;
}

AbortResult ProgramImpl::abort(SequenceHandle sequence) noexcept {
    Lane* lane = lane_of(sequence);
    if (lane == nullptr) { return {}; }
    AbortResult result;
    result.status      = runtime::ConsumeStatus::Consumed;
    result.timings     = lane->timings;
    result.speculative = lane->speculative;
    release_lane(*lane);
    return result;
}

std::optional<std::uint32_t>
ProgramImpl::device_kv_lease_settlement_tokens(SequenceHandle) const noexcept {
    // Every lane holds max_context positions from admission: the lease never settles early.
    return std::nullopt;
}

void ProgramImpl::fail_all_cleanup() noexcept {
    for (Lane& lane : lanes_) {
        if (lane.occupied) { release_lane(lane); }
    }
    transaction_.reset();
    resident_.reset();
    pending_ = Pending{};
}

bool ProgramImpl::isolated_request_feasible(const RequestBasePlan& base) const noexcept {
    const runtime::RequestPlanSummary& summary = base.summary();
    return base.impl_ != nullptr && summary.prompt_tokens != 0 &&
           summary.prompt_tokens <= plan_.layout.capacity && summary.effective_output_tokens != 0;
}

PhysicalUsageSnapshot ProgramImpl::physical_usage() const noexcept {
    PhysicalUsageSnapshot out;
    out.resource_revision = revision_;
    const std::uint32_t lane_pages =
        (plan_.layout.capacity + plan_.layout.page_tokens - 1U) / plan_.layout.page_tokens;
    for (const Lane& lane : lanes_) {
        if (!lane.occupied) { continue; }
        ++out.device_state_slots;
        out.device_main_kv_pages += lane_pages;
    }
    return out;
}

MemorySummary ProgramImpl::memory_summary() const noexcept {
    MemorySummary out;
    try {
        out = execution_->memory_summary();
    } catch (...) {}
    out.max_context             = plan_.layout.capacity;
    out.kv_capacity             = plan_.kv_capacity;
    out.kv_capacity_page_groups = plan_.main_page_groups;
    return out;
}

void ProgramImpl::reset_memory_peaks() noexcept { execution_->reset_memory_peaks(); }

DeviceKVPages ProgramImpl::device_kv_available_pages() const noexcept {
    const std::uint32_t lane_pages =
        (plan_.layout.capacity + plan_.layout.page_tokens - 1U) / plan_.layout.page_tokens;
    std::uint32_t used = 0;
    for (const Lane& lane : lanes_) { used += lane.occupied ? lane_pages : 0; }
    return {.main = plan_.main_page_groups > used ? plan_.main_page_groups - used : 0};
}

ProgramImpl::Lane* ProgramImpl::lane_of(SequenceHandle sequence) noexcept {
    return const_cast<Lane*>(std::as_const(*this).lane_of(sequence));
}

const ProgramImpl::Lane* ProgramImpl::lane_of(SequenceHandle sequence) const noexcept {
    if (Access::owner(sequence) != this) { return nullptr; }
    const std::uint32_t index = Access::lane(sequence).value;
    if (index >= lanes_.size()) { return nullptr; }
    const Lane& lane = lanes_[index];
    return lane.occupied && lane.epoch == Access::epoch(sequence) ? &lane : nullptr;
}

SequenceHandle ProgramImpl::handle_of(std::uint32_t lane) const noexcept {
    return Access::make_sequence(this, runtime::LaneId{lane}, lanes_[lane].epoch);
}

void ProgramImpl::bump_revision() noexcept { ++revision_.value; }

void ProgramImpl::release_lane(Lane& lane) noexcept {
    lane = Lane{};
    bump_revision();
}

} // namespace ninfer::models::inkling2::detail
