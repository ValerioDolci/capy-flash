#include "models/inkling2/program/program.h"

#include "core/device.h"
#include "models/inkling2/execution/parameters.h"
#include "models/inkling2/program/program_impl.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::inkling2 {
namespace {

[[noreturn]] void unsupported(const char* what) {
    throw std::logic_error(std::string("Inkling 2 Program: ") + what +
                           " is not supported (one request at a time; the context cache keeps "
                           "one resident sequence with its prompt anchor on the device)");
}

} // namespace

CaptureAssessment::CaptureAssessment() = default;

SequencePlan::SequencePlan(std::unique_ptr<detail::SequencePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}

SequencePlan::SequencePlan(SequencePlan&&) noexcept            = default;
SequencePlan& SequencePlan::operator=(SequencePlan&&) noexcept = default;
SequencePlan::~SequencePlan()                                  = default;

std::uint32_t SequencePlan::capacity() const noexcept {
    return impl_ != nullptr ? impl_->layout.capacity : 0;
}

std::uint32_t SequencePlan::kv_capacity() const noexcept {
    return impl_ != nullptr ? impl_->kv_capacity : 0;
}

std::uint32_t SequencePlan::max_concurrency() const noexcept {
    return impl_ != nullptr ? impl_->layout.max_concurrency : 0;
}

std::size_t SequencePlan::device_reservation_bytes() const noexcept {
    return impl_ != nullptr ? impl_->device_reservation_bytes : 0;
}

std::size_t SequencePlan::workspace_capacity_bytes() const noexcept {
    return impl_ != nullptr ? impl_->layout.workspace_bytes : 0;
}

SequencePlanner::SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl> impl) noexcept
    : impl_(std::move(impl)) {}

SequencePlanner::SequencePlanner(SequencePlanner&&) noexcept            = default;
SequencePlanner& SequencePlanner::operator=(SequencePlanner&&) noexcept = default;
SequencePlanner::~SequencePlanner()                                     = default;

const runtime::SequenceCapacityCurve& SequencePlanner::capacity_curve() const noexcept {
    static const runtime::SequenceCapacityCurve empty;
    return impl_ != nullptr ? impl_->curve : empty;
}

std::size_t SequencePlanner::kv_pool_bytes(std::uint32_t main_page_groups) const noexcept {
    return impl_ != nullptr ? impl_->layout.page_group_bytes * main_page_groups : 0;
}

SequencePlan SequencePlanner::finalize(std::uint32_t main_page_groups) && {
    if (impl_ == nullptr) { throw std::logic_error("sequence planner is empty"); }
    auto plan                      = std::make_unique<detail::SequencePlanImpl>();
    plan->layout                   = impl_->layout;
    plan->main_page_groups         = main_page_groups;
    plan->kv_capacity              = impl_->curve.resolved_tokens(main_page_groups);
    plan->device_reservation_bytes = impl_->curve.reservation_bytes(main_page_groups);
    impl_.reset();
    return SequencePlan(std::move(plan));
}

RequestBasePlan::RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}

RequestBasePlan::RequestBasePlan(RequestBasePlan&&) noexcept            = default;
RequestBasePlan& RequestBasePlan::operator=(RequestBasePlan&&) noexcept = default;
RequestBasePlan::~RequestBasePlan()                                     = default;

const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const PreparedContextCache& RequestBasePlan::context_cache() const noexcept {
    static const PreparedContextCache empty;
    return impl_ != nullptr ? impl_->context_cache : empty;
}

std::optional<PrefixShortlistKey>
RequestBasePlan::prefix_shortlist_key(std::uint32_t frontier) const noexcept {
    if (impl_ == nullptr || frontier == 0 || frontier >= impl_->digests.size()) {
        return std::nullopt;
    }
    return PrefixShortlistKey{.digests      = impl_->digests[frontier],
                              .frontier     = frontier,
                              .identity_tag = detail::kPrefixIdentityTag};
}

std::optional<runtime::PrefillWork>
RequestBasePlan::shared_candidate_rebuild_work(std::uint32_t) const noexcept {
    return std::nullopt;
}

AdmissionCandidate::AdmissionCandidate(
    std::unique_ptr<detail::AdmissionCandidateImpl> impl) noexcept
    : impl_(std::move(impl)) {}

AdmissionCandidate::AdmissionCandidate(AdmissionCandidate&&) noexcept            = default;
AdmissionCandidate& AdmissionCandidate::operator=(AdmissionCandidate&&) noexcept = default;
AdmissionCandidate::~AdmissionCandidate()                                        = default;

const runtime::RequestPlanSummary& AdmissionCandidate::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const runtime::IdentityMaterializationAssessment&
AdmissionCandidate::identity_assessment() const noexcept {
    static const runtime::IdentityMaterializationAssessment empty;
    return impl_ != nullptr ? impl_->identity : empty;
}

CapturePressureCandidate::CapturePressureCandidate(
    std::unique_ptr<detail::CapturePressureCandidateImpl> impl) noexcept
    : impl_(std::move(impl)) {}

CapturePressureCandidate::CapturePressureCandidate(CapturePressureCandidate&&) noexcept = default;
CapturePressureCandidate&
CapturePressureCandidate::operator=(CapturePressureCandidate&&) noexcept = default;
CapturePressureCandidate::~CapturePressureCandidate()                    = default;

PressurePlanningSession::PressurePlanningSession(
    std::unique_ptr<detail::PressurePlanningSessionImpl> impl) noexcept
    : impl_(std::move(impl)) {}

PressurePlanningSession::PressurePlanningSession(PressurePlanningSession&&) noexcept = default;
PressurePlanningSession&
PressurePlanningSession::operator=(PressurePlanningSession&&) noexcept = default;
PressurePlanningSession::~PressurePlanningSession()                    = default;

PressureTargetHandle PressurePlanningSession::identity_target(runtime::PlanningCandidateId) const {
    unsupported("admission pressure");
}

// The one pressure target: the root with every catalogued owner evicted.
PressureTargetHandle
PressurePlanningSession::root_maximal_target(runtime::PlanningCandidateId root_candidate) {
    if (impl_ == nullptr || !impl_->root || impl_->root_id != root_candidate) {
        throw std::logic_error("Inkling 2 pressure: the root candidate is unknown");
    }
    PressureTargetHandle target;
    target.session_    = impl_.get();
    target.generation_ = 1;
    target.index_      = 0;
    return target;
}

PressureTargetHandle PressurePlanningSession::maximal_target(runtime::PlanningCandidateId) {
    unsupported("admission pressure");
}

PressureConstructionCursor PressurePlanningSession::begin_construction(PressureTargetHandle, bool) {
    unsupported("admission pressure");
}

runtime::PressureConstructionStep
PressurePlanningSession::next_construction_option(PressureConstructionCursor&) {
    unsupported("admission pressure");
}

void PressurePlanningSession::choose_construction(PressureConstructionCursor&,
                                                  runtime::PressureConstructionOptionId) {
    unsupported("admission pressure");
}

std::optional<PressureTargetHandle>
PressurePlanningSession::construction_target(const PressureConstructionCursor&) {
    unsupported("admission pressure");
}

runtime::PressureTargetGuidance PressurePlanningSession::guidance(PressureTargetHandle) {
    unsupported("admission pressure");
}

AssessedPressureTarget PressurePlanningSession::assess(PressureTargetHandle target) {
    if (impl_ == nullptr || !impl_->root || target.session_ != impl_.get() ||
        target.generation_ != 1) {
        throw std::logic_error("Inkling 2 pressure: unknown target");
    }
    const detail::AdmissionCandidateImpl& root = *impl_->root;
    runtime::PressureTargetAssessment assessment;
    assessment.physical_status     = root.identity.physical_status;
    assessment.source_mode         = runtime::PrivateSourceMode::ConsumeToActive;
    assessment.machine_work        = root.identity.machine_work;
    assessment.owner_outcomes      = impl_->outcomes;
    assessment.checkpoint_impacts  = impl_->impacts;
    assessment.dropped_checkpoints = static_cast<std::uint32_t>(impl_->impacts.size());
    assessment.candidate           = impl_->root_id;
    assessment.root_maximal        = true;
    return AssessedPressureTarget(
        impl_.get(), 1, 0, assessment, 0, 0, nullptr,
        AdmissionCandidate(std::make_unique<detail::AdmissionCandidateImpl>(root)), std::nullopt);
}

PreparedPressureExpansion PressurePlanningSession::prepare_expansion(PressureTargetHandle,
                                                                     std::uint32_t) {
    unsupported("admission pressure");
}

PressureExpansionView PressurePlanningSession::commit_expansion(PreparedPressureExpansion&&) {
    unsupported("admission pressure");
}

void PressurePlanningSession::discard_expansion(PreparedPressureExpansion&&) noexcept {}

runtime::PrefillWork PressurePlanningSession::shared_capture_split_prefill_work(
    const AssessedPressureTarget&, const PreparedPrompt&, std::span<const std::uint32_t>) const {
    unsupported("admission pressure");
}

std::optional<ResourcePlan> PressurePlanningSession::seal(AssessedPressureTarget&& assessed,
                                                          const PreparedPrompt&,
                                                          runtime::FinalScheduleIntent intent) {
    if (impl_ == nullptr || assessed.session_ != impl_.get() || !assessed.executable_ ||
        !intent.shared_capture_frontiers.empty()) {
        throw std::logic_error("Inkling 2 pressure: the target cannot be sealed");
    }
    if (assessed.assessment_.physical_status != runtime::MaterializationPhysicalStatus::Feasible) {
        return std::nullopt;
    }
    return ResourcePlan(std::move(*assessed.executable_), impl_->revision, false);
}

std::optional<CapturePressurePlan> PressurePlanningSession::seal_capture(AssessedPressureTarget&&) {
    unsupported("capture pressure");
}

CapturePressurePlanningSession::CapturePressurePlanningSession(
    CapturePressurePlanningSession&&) noexcept = default;
CapturePressurePlanningSession&
CapturePressurePlanningSession::operator=(CapturePressurePlanningSession&&) noexcept = default;
CapturePressurePlanningSession::~CapturePressurePlanningSession()                    = default;

PressureTargetHandle CapturePressurePlanningSession::identity_target() const {
    unsupported("capture pressure");
}

runtime::PressureTargetGuidance CapturePressurePlanningSession::guidance(PressureTargetHandle) {
    unsupported("capture pressure");
}

AssessedPressureTarget CapturePressurePlanningSession::assess(PressureTargetHandle) {
    unsupported("capture pressure");
}

PreparedPressureExpansion CapturePressurePlanningSession::prepare_expansion(PressureTargetHandle) {
    unsupported("capture pressure");
}

PressureExpansionView
CapturePressurePlanningSession::commit_expansion(PreparedPressureExpansion&&) {
    unsupported("capture pressure");
}

void CapturePressurePlanningSession::discard_expansion(PreparedPressureExpansion&&) noexcept {}

std::optional<CapturePressurePlan> CapturePressurePlanningSession::seal(AssessedPressureTarget&&) {
    unsupported("capture pressure");
}

Program::Program(std::unique_ptr<detail::ProgramImpl> impl) noexcept : impl_(std::move(impl)) {}

Program::~Program() noexcept = default;

RequestBasePlan Program::plan_request(const PreparedPrompt& prompt,
                                      const runtime::ResolvedExecutionOptions& options) {
    return impl_->plan_request(prompt, options);
}

std::vector<float> Program::causal_score(PreparedPrompt&& prompt, std::uint32_t first_target) {
    return impl_->causal_score(std::move(prompt), first_target);
}

std::uint32_t Program::disk_resume_tokens(const PreparedPrompt&) { return 0; }

std::optional<AdmissionCandidate> Program::inspect_admission(
    const PreparedPrompt& prompt, const RequestBasePlan& base, runtime::LaneId destination,
    const ContinuationHandle* source, const SharedPrefixHandle* shared_source,
    std::optional<runtime::CheckpointRef> checkpoint, bool must_retain_private_source) {
    return impl_->inspect_admission(prompt, base, destination, source, checkpoint,
                                    shared_source != nullptr, must_retain_private_source);
}

std::optional<ResourcePlan> Program::seal_identity(const AdmissionCandidate& candidate,
                                                   const PreparedPrompt&,
                                                   runtime::FinalScheduleIntent intent) {
    std::optional<AdmissionCandidate> sealed = impl_->seal_identity(candidate, intent);
    if (!sealed) { return std::nullopt; }
    return ResourcePlan(std::move(*sealed), impl_->resource_revision(), false);
}

PressurePlanningSession
Program::begin_pressure_planning(std::span<const AdmissionCandidate* const> candidates,
                                 std::span<const runtime::PlanningCandidateId> candidate_ids,
                                 std::span<const ContinuationHandle* const> private_owners,
                                 std::span<const runtime::PlanningOwnerId> private_owner_ids,
                                 std::span<const SharedPrefixHandle* const> shared_owners,
                                 std::span<const runtime::PlanningOwnerId>) {
    return PressurePlanningSession(impl_->begin_pressure_planning(
        candidates, candidate_ids, private_owners, private_owner_ids, shared_owners));
}

runtime::PrefillWork
Program::shared_capture_split_prefill_work(const AdmissionCandidate& candidate,
                                           const PreparedPrompt&,
                                           std::span<const std::uint32_t> frontiers) {
    if (!frontiers.empty()) { unsupported("shared prefix capture"); }
    return candidate.identity_assessment().machine_work.remaining_prefill_work;
}

runtime::ContextTransactionReserveStatus
Program::start_resource_transaction(ResourcePlan&& plan, PreparedPrompt&& prompt,
                                    runtime::CancellationFlagView cancellation) {
    if (plan.resource_revision() != impl_->resource_revision()) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    return impl_->start_resource_transaction(std::move(plan.admission_), std::move(prompt),
                                             cancellation);
}

std::optional<PersistentBackfillProof>
Program::prove_persistent_backfill(const RequestBasePlan&, const ResourcePlan&,
                                   std::span<const SequenceHandle>) const {
    return std::nullopt;
}

ContextTransactionProgress
Program::progress_context_transaction(runtime::CancellationFlagView cancellation) {
    return impl_->progress_context_transaction(cancellation);
}

void Program::finalize_context_transaction() noexcept { impl_->finalize_context_transaction(); }

bool Program::has_context_transaction() const noexcept { return impl_->has_context_transaction(); }

PrefillProgress Program::advance_prefill(SequenceHandle sequence,
                                         runtime::ExecutionTiming* failed_timing,
                                         runtime::TokenMaskProvider* masks) {
    return impl_->advance_prefill(sequence, failed_timing, masks);
}

CaptureAssessment Program::inspect_capture(const CaptureOffer&, const SharedPrefixHandle*,
                                           const SharedPrefixHandle*,
                                           std::optional<runtime::CheckpointRef>, bool) const {
    unsupported("prefix capture");
}

std::vector<runtime::CheckpointRecoveryAlternativeWork>
Program::checkpoint_recovery_work(const ContinuationHandle& owner,
                                  runtime::CheckpointRef checkpoint) const {
    return impl_->checkpoint_recovery_work(owner, checkpoint);
}

std::vector<runtime::CheckpointRecoveryAlternativeWork>
Program::checkpoint_recovery_work(const SharedPrefixHandle&, runtime::CheckpointRef) const {
    unsupported("checkpoint recovery");
}

CapturePressurePlanningSession Program::begin_capture_pressure_planning(
    const CaptureAssessment&, std::span<const ContinuationHandle* const>,
    std::span<const runtime::PlanningOwnerId>, std::span<const SharedPrefixHandle* const>,
    std::span<const runtime::PlanningOwnerId>) {
    unsupported("capture pressure");
}

bool Program::shared_capture_matches(const CaptureOffer&, const SharedPrefixHandle&) const {
    return false;
}

void Program::skip_capture(CaptureOffer&&) { unsupported("prefix capture"); }

runtime::ContextTransactionReserveStatus
Program::reserve_active_capture(CaptureOffer&&, const SharedPrefixHandle*,
                                const SharedPrefixHandle*, std::optional<runtime::CheckpointRef>,
                                bool, runtime::CancellationFlagView) {
    unsupported("prefix capture");
}

runtime::ContextTransactionReserveStatus Program::reserve_active_capture_with_pressure(
    CaptureOffer&&, const SharedPrefixHandle*, const SharedPrefixHandle*,
    std::optional<runtime::CheckpointRef>, bool, CapturePressurePlan&&,
    runtime::CancellationFlagView) {
    unsupported("prefix capture");
}

PendingBatch Program::decode(std::span<const SequenceHandle> sequences,
                             std::span<const runtime::RoundBudget> budgets,
                             runtime::ExecutionTiming* failed_timing,
                             runtime::TokenMaskProvider* masks) {
    return impl_->decode(sequences, budgets, failed_timing, masks);
}

bool Program::sampling_bypass_hooks() const noexcept { return false; }

runtime::ExecutionTiming
Program::append_forced_tokens(std::span<const SequenceHandle> sequences,
                              std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                              std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
                              runtime::ExecutionTiming*) {
    return impl_->append_forced_tokens(sequences, row_major_tokens, row_stride,
                                       prefix_execution_splits);
}

CommitResult Program::commit(PendingBatch&& pending,
                             std::span<const runtime::CommitDecision> decisions,
                             runtime::CommitObservation observation, runtime::ExecutionTiming*) {
    return impl_->commit(std::move(pending), decisions, observation);
}

DiscardResult Program::abort_pending(PendingBatch&& pending) noexcept {
    return impl_->abort_pending(std::move(pending));
}

FinishResult Program::finish(SequenceHandle sequence) noexcept { return impl_->finish(sequence); }

FinishResult Program::finish_cancelled(SequenceHandle sequence) noexcept {
    return impl_->finish(sequence);
}

AbortResult Program::abort(SequenceHandle sequence) noexcept { return impl_->abort(sequence); }

std::optional<std::uint32_t>
Program::device_kv_lease_settlement_tokens(SequenceHandle sequence, std::uint32_t) const noexcept {
    return impl_->device_kv_lease_settlement_tokens(sequence);
}

std::optional<DeviceKVLeaseShortfall>
Program::device_kv_lease_shortfall(SequenceHandle) const noexcept {
    return std::nullopt;
}

bool Program::resume_device_kv_lease(SequenceHandle) noexcept { return true; }

DeviceKVPages Program::retained_device_kv_pages(const ContinuationHandle&) const noexcept {
    return {};
}

DeviceKVPages Program::retained_device_kv_pages(const SharedPrefixHandle&) const noexcept {
    return {};
}

DeviceKVPages Program::idle_device_kv_pages(const ContinuationHandle&) const {
    unsupported("retained context");
}

DeviceKVPages Program::idle_device_kv_pages(const SharedPrefixHandle&) const {
    unsupported("retained context");
}

std::vector<DeviceKVPages>
Program::releasable_device_kv_pages(std::span<const ContinuationHandle* const> continuations,
                                    std::span<const SharedPrefixHandle* const> shared) const {
    if (!continuations.empty() || !shared.empty()) { unsupported("retained context"); }
    return {};
}

DeviceKVPages Program::device_kv_available_pages() const noexcept {
    return impl_->device_kv_available_pages();
}

std::optional<ContinuationSummary> Program::demote_retained_kv_for_lease(const ContinuationHandle&,
                                                                         DeviceKVPages) noexcept {
    return std::nullopt;
}

std::optional<SharedPrefixSummary>
Program::demote_retained_shared_kv_for_lease(const SharedPrefixHandle&, DeviceKVPages) noexcept {
    return std::nullopt;
}

bool Program::idle_device_kv_offload() const noexcept { return false; }

std::optional<std::size_t>
Program::idle_device_kv_host_bytes(const ContinuationHandle&) const noexcept {
    return std::nullopt;
}

std::size_t Program::host_kv_free_bytes() const noexcept { return 0; }

DeviceKVCompaction Program::compact_device_kv() { return {}; }

bool Program::disk_cache_enabled() const noexcept { return false; }

std::uint32_t Program::disk_cache_idle_ms() const noexcept { return 0; }

DiskCacheSave Program::disk_cache_save_one(const std::function<bool()>&) {
    return DiskCacheSave::NothingToSave;
}

std::uint32_t Program::disk_cache_flush_seconds() const noexcept { return 0; }

bool Program::disk_cache_drain(std::chrono::steady_clock::time_point) { return true; }

std::string Program::disk_cache_status() const { return {}; }

ReleaseResult Program::release_continuation(ContinuationHandle&& continuation) noexcept {
    return impl_->release_continuation(std::move(continuation));
}

std::optional<ContinuationSummary>
Program::drop_continuation_anchor(const ContinuationHandle&, runtime::CheckpointRef) noexcept {
    return std::nullopt;
}

ReleaseResult Program::release_shared_prefix(SharedPrefixHandle&&) noexcept { return {}; }

void Program::fail_all_cleanup() noexcept { impl_->fail_all_cleanup(); }

bool Program::isolated_request_feasible(const RequestBasePlan& base) const noexcept {
    return impl_->isolated_request_feasible(base);
}

runtime::ProgramResourceRevision Program::resource_revision() const noexcept {
    return impl_->resource_revision();
}

PhysicalUsageSnapshot Program::physical_usage() const noexcept { return impl_->physical_usage(); }

SpeculativeTotals Program::speculative_totals() const noexcept {
    return impl_->speculative_totals();
}

MemorySummary Program::memory_summary() const noexcept { return impl_->memory_summary(); }

void Program::reset_memory_peaks() noexcept { impl_->reset_memory_peaks(); }

const TpTransportStatus& Program::tp_transport() const noexcept { return impl_->tp_transport(); }

bool Program::peer_exchange_timed_out() const noexcept { return false; }

std::string Program::recover_peer_transport() { unsupported("peer transport recovery"); }

SequencePlanner make_sequence_planner(const execution::Parameters& parameters,
                                      const execution::Parameters* peer_parameters,
                                      ExecutionContext& execution, const EngineOptions& options) {
    if ((peer_parameters != nullptr) != (execution.tp == 2) || options.tp != execution.tp) {
        throw std::logic_error(
            "Inkling 2 sequence planner does not match the tensor-parallel width");
    }
    auto planner    = std::make_unique<detail::SequencePlannerImpl>();
    planner->layout = detail::plan_sequence_layout(parameters, peer_parameters, options);
    const detail::SequenceLayout& layout = planner->layout;
    // Every lane addresses max_context positions of global KV; the pool is that and no more.
    const std::uint32_t lane_pages =
        (layout.capacity + layout.page_tokens - 1U) / layout.page_tokens;
    runtime::SequenceCapacityCurve& curve = planner->curve;
    curve.main_page_tokens                = layout.page_tokens;
    curve.minimum_main_page_groups        = lane_pages;
    curve.maximum_main_page_groups        = lane_pages * layout.max_concurrency;
    curve.minimum_device_reservation_bytes =
        layout.fixed_bytes + layout.page_group_bytes * lane_pages;
    curve.bytes_per_additional_main_page_group = layout.page_group_bytes;
    return SequencePlanner(std::move(planner));
}

std::unique_ptr<Program> create_program(const execution::Parameters& parameters,
                                        const execution::Parameters* peer_parameters,
                                        SequencePlan&& plan, ExecutionContext& execution,
                                        const StartupObserver&) {
    if (plan.impl_ == nullptr) { throw std::logic_error("Inkling 2 sequence plan is empty"); }
    detail::SequencePlanImpl finalized = *plan.impl_;
    plan.impl_.reset();
    auto runtime =
        detail::make_program_execution(parameters, peer_parameters, finalized.layout, execution);
    return std::unique_ptr<Program>(
        new Program(std::make_unique<detail::ProgramImpl>(finalized, std::move(runtime))));
}

} // namespace ninfer::models::inkling2
