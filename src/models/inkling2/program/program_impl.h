#pragma once

#include "models/inkling2/program/drafter_choice.h"
#include "models/inkling2/program/execution.h"
#include "models/inkling2/frontend/frontend.h"
#include "models/inkling2/program/program.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace ninfer::models::inkling2::detail {

// The widest decode round: an anchor and the drafter's most drafts.
inline constexpr std::size_t kMaxRoundWidth = 8;

// Per-rank layout chosen at startup. One Main KV page group holds `page_tokens` positions of every
// global visit's K/V; the local visits' rings and the convolution tails are fixed per lane.
struct SequenceLayout {
    std::uint32_t capacity            = 0; // max_context
    std::uint32_t max_concurrency     = 0;
    std::uint32_t prefill_chunk       = 0;
    std::uint32_t page_tokens         = 0;
    std::size_t fixed_bytes           = 0; // per rank, independent of the KV page count
    std::size_t page_group_bytes      = 0; // per rank, one Main KV page group
    std::size_t workspace_bytes       = 0; // per rank, included in fixed_bytes
    std::size_t image_bytes           = 0; // per rank, the anchor image; included in fixed_bytes
    std::size_t graph_allowance_bytes = 0; // per rank, warmup and decode graph; in fixed_bytes
    bool context_cache                = false;
    bool cuda_graph                   = false; // single-token decode replays one captured graph
    // Drafts per speculative round (--spec mtp|dflash2 --draft-tokens K with the artifact's
    // drafter of that backend); 0: no drafter, every decode step is ordinary.
    std::uint32_t draft_tokens       = 0;
    SpeculativeBackend draft_backend = SpeculativeBackend::None;
    // The forward's planned bounds: columns of a call (the prefill chunk, or a speculative
    // round's K + 1 when wider) and logit columns (scoring blocks, or K + 1 when wider).
    std::uint32_t forward_tokens = 0;
    std::uint32_t logit_columns  = 0;
    std::size_t draft_bytes = 0; // per rank, the drafter's state, workspace and round; in fixed
};

struct SequencePlannerImpl {
    SequenceLayout layout;
    runtime::SequenceCapacityCurve curve;
};

struct SequencePlanImpl {
    SequenceLayout layout;
    std::uint32_t main_page_groups       = 0;
    std::uint32_t kv_capacity            = 0;
    std::size_t device_reservation_bytes = 0;
};

// Prefix identity of a token sequence: two independent 64-bit polynomial hashes per frontier.
using PrefixDigest                                = std::array<std::uint64_t, 2>;
inline constexpr std::uint32_t kPrefixIdentityTag = 1;

// digests[f] = digest of tokens[0..f).
[[nodiscard]] std::vector<PrefixDigest> prefix_digests(std::span<const TokenId> tokens);

struct RequestBasePlanImpl {
    runtime::RequestPlanSummary summary;
    PreparedContextCache context_cache;
    ResolvedSamplingParameters sampling;
    std::vector<PrefixDigest> digests; // empty when the request may not reuse a prefix
};

// Where an admission starts from: the root, or the resident sequence's endpoint or prompt anchor.
enum class AdmissionSource : std::uint8_t { Root, Endpoint, Anchor };

struct AdmissionCandidateImpl {
    runtime::RequestPlanSummary summary;
    runtime::IdentityMaterializationAssessment identity;
    runtime::LaneId destination;
    ResolvedSamplingParameters sampling;
    AdmissionSource source            = AdmissionSource::Root;
    std::uint64_t resident_generation = 0; // the source (or evicted) resident sequence
    // Root admission that evicts the catalogued resident sequence: its planning owner.
    std::optional<runtime::PlanningOwnerId> evicted_owner;
};

struct CapturePressureCandidateImpl {};

// The root-maximal pressure target only: the root admission with every catalogued owner evicted.
struct PressurePlanningSessionImpl {
    std::optional<AdmissionCandidateImpl> root;
    runtime::PlanningCandidateId root_id;
    std::vector<runtime::PressureOwnerOutcome> outcomes;
    // Every checkpoint of an evicted owner is lost; recovery = prefill from the root.
    std::vector<std::vector<runtime::CheckpointRecoveryAlternativeWork>> recovery;
    std::vector<runtime::PressureCheckpointRecoveryImpact> impacts;
    runtime::ProgramResourceRevision revision;
};

struct CaptureAssessmentImpl {};

class ProgramImpl {
public:
    ProgramImpl(SequencePlanImpl plan, std::unique_ptr<ProgramExecution> execution);

    [[nodiscard]] RequestBasePlan plan_request(const PreparedPrompt& prompt,
                                               const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] std::vector<float> causal_score(PreparedPrompt&& prompt,
                                                  std::uint32_t first_target);
    [[nodiscard]] std::optional<AdmissionCandidate>
    inspect_admission(const PreparedPrompt& prompt, const RequestBasePlan& base,
                      runtime::LaneId destination, const ContinuationHandle* source,
                      std::optional<runtime::CheckpointRef> checkpoint, bool from_shared,
                      bool retain_source);
    [[nodiscard]] std::unique_ptr<PressurePlanningSessionImpl>
    begin_pressure_planning(std::span<const AdmissionCandidate* const> candidates,
                            std::span<const runtime::PlanningCandidateId> candidate_ids,
                            std::span<const ContinuationHandle* const> private_owners,
                            std::span<const runtime::PlanningOwnerId> private_owner_ids,
                            std::span<const SharedPrefixHandle* const> shared_owners);
    [[nodiscard]] std::vector<runtime::CheckpointRecoveryAlternativeWork>
    checkpoint_recovery_work(const ContinuationHandle& owner,
                             runtime::CheckpointRef checkpoint) const;
    [[nodiscard]] ReleaseResult release_continuation(ContinuationHandle&& continuation) noexcept;
    // The sealed copy of `candidate`; Program binds it to the current revision.
    [[nodiscard]] std::optional<AdmissionCandidate>
    seal_identity(const AdmissionCandidate& candidate, runtime::FinalScheduleIntent intent);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    start_resource_transaction(AdmissionCandidate&& admission, PreparedPrompt&& prompt,
                               runtime::CancellationFlagView cancellation);
    [[nodiscard]] ContextTransactionProgress
    progress_context_transaction(runtime::CancellationFlagView cancellation);
    void finalize_context_transaction() noexcept;
    [[nodiscard]] bool has_context_transaction() const noexcept;
    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence,
                                                  runtime::ExecutionTiming* failed_timing,
                                                  runtime::TokenMaskProvider* masks);
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                      std::span<const runtime::RoundBudget> budgets,
                                      runtime::ExecutionTiming* failed_timing,
                                      runtime::TokenMaskProvider* masks);
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle> sequences,
                         std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                         std::span<const std::optional<std::uint32_t>> prefix_execution_splits);
    [[nodiscard]] CommitResult commit(PendingBatch&& pending,
                                      std::span<const runtime::CommitDecision> decisions,
                                      runtime::CommitObservation observation);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&& pending) noexcept;
    // Catalogues the sequence (its state stays resident) when the request publishes a continuation.
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;
    [[nodiscard]] std::optional<std::uint32_t>
    device_kv_lease_settlement_tokens(SequenceHandle sequence) const noexcept;
    void fail_all_cleanup() noexcept;
    [[nodiscard]] bool isolated_request_feasible(const RequestBasePlan& base) const noexcept;

    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept {
        return revision_;
    }

    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;

    [[nodiscard]] SpeculativeTotals speculative_totals() const noexcept {
        return speculative_totals_;
    }

    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;

    [[nodiscard]] const TpTransportStatus& tp_transport() const noexcept { return transport_; }

    [[nodiscard]] DeviceKVPages device_kv_available_pages() const noexcept;

private:
    using Clock = std::chrono::steady_clock;

    struct Lane {
        bool occupied       = false;
        bool poisoned       = false; // a discarded round left execution state ahead of tokens
        std::uint64_t epoch = 0;
        std::vector<TokenId> tokens; // prompt, then every committed generated token
        std::uint32_t prompt_tokens = 0;
        std::uint32_t prefilled     = 0;
        std::uint32_t executed      = 0; // tokens[executed..) have not run through the forward
        std::uint32_t reused        = 0; // prompt tokens taken from the resident sequence
        PrefixReusePath path        = PrefixReusePath::Root;
        std::uint32_t anchor        = 0; // frontier of the lane's anchor image; 0 = none
        runtime::RequestPlanSummary summary;
        ResolvedSamplingParameters sampling;
        GenerationTimings timings;
        SpeculativeStats speculative;
        DrafterChoice drafters;   // with paired drafters (--spec auto): which proposes
        bool after_steps = false; // the last decode step was an ordinary one
        Clock::time_point started;
        Clock::time_point first_token;
    };

    // The sequence whose state stays on the device after its request finished: catalogued as one
    // continuation with an endpoint (tokens[0..executed)) and the prompt anchor (the image).
    struct Resident {
        std::uint64_t generation = 0;
        std::vector<TokenId> tokens;
        std::uint32_t executed = 0;
        std::uint32_t anchor   = 0;
    };

    struct OpenMaterialization {
        runtime::LaneId lane;
        AdmissionCandidateImpl admission;
        PreparedPrompt prompt;
        bool published = false;
    };

    // The licensed tokens of the round, row-major with `row_stride` columns per row: one for an
    // ordinary step, draft_tokens + 1 for a speculative round (`speculative` rows), whose state
    // holds `frontier` + row_counts[row] positions until the commit keeps fewer.
    struct Pending {
        std::uint64_t transaction = 0;
        std::uint32_t row_stride  = 1;
        std::array<TokenId, kMaximumConcurrency * kMaxRoundWidth> tokens{};
        std::array<std::int32_t, kMaximumConcurrency> row_counts{};
        std::array<bool, kMaximumConcurrency> prefill{};
        std::array<bool, kMaximumConcurrency> speculative{};
        std::array<std::uint32_t, kMaximumConcurrency> frontier{};
    };

    // Whether lane `lane`'s next decode step is a speculative round, and how many drafts it
    // verifies within the budget and the context.
    [[nodiscard]] std::optional<std::uint32_t>
    speculative_extent(const Lane& lane, const runtime::RoundBudget& budget) const noexcept;

    [[nodiscard]] Lane* lane_of(SequenceHandle sequence) noexcept;
    [[nodiscard]] const Lane* lane_of(SequenceHandle sequence) const noexcept;
    [[nodiscard]] SequenceHandle handle_of(std::uint32_t lane) const noexcept;
    void bump_revision() noexcept;
    void release_lane(Lane& lane) noexcept;
    [[nodiscard]] runtime::IdentityMaterializationAssessment
    assessment(const runtime::RequestPlanSummary& summary) const;
    [[nodiscard]] bool resident_matches(const ContinuationHandle& handle) const noexcept;
    [[nodiscard]] CheckpointSummary checkpoint(runtime::CheckpointKind kind, std::uint32_t ordinal,
                                               std::uint32_t frontier) const;
    [[nodiscard]] ContinuationSummary resident_summary() const;
    [[nodiscard]] std::uint64_t service_work(const runtime::RequestPlanSummary& summary) const;

    SequencePlanImpl plan_;
    std::unique_ptr<ProgramExecution> execution_;
    // With paired drafters: the measured cost of each kind of decode step, across requests.
    StepCosts costs_;
    std::vector<Lane> lanes_;
    std::optional<OpenMaterialization> transaction_;
    std::optional<Resident> resident_;
    std::uint64_t next_generation_ = 1;
    Pending pending_;
    std::uint64_t next_transaction_ = 1;
    std::uint64_t next_epoch_       = 1;
    runtime::ProgramResourceRevision revision_{.value = 1};
    SpeculativeTotals speculative_totals_;
    TpTransportStatus transport_;
};

} // namespace ninfer::models::inkling2::detail
