#pragma once
#include "models/inkling2/frontend/frontend.h"
#include "models/inkling2/program/program.h"

namespace ninfer::models::inkling2 {

// The common request controller is instantiated once for this concrete model implementation.
struct RuntimeTypes {
    using Frontend                   = inkling2::Frontend;
    using PreparedPrompt             = inkling2::PreparedPrompt;
    using OutputSession              = inkling2::OutputSession;
    using PublishedOutput            = inkling2::PublishedOutput;
    using SequencePlanner            = inkling2::SequencePlanner;
    using SequencePlan               = inkling2::SequencePlan;
    using RequestBasePlan            = inkling2::RequestBasePlan;
    using AdmissionCandidate         = inkling2::AdmissionCandidate;
    using ResourcePlan               = inkling2::ResourcePlan;
    using PersistentBackfillProof    = inkling2::PersistentBackfillProof;
    using SequenceHandle             = inkling2::SequenceHandle;
    using ContinuationHandle         = inkling2::ContinuationHandle;
    using SharedPrefixHandle         = inkling2::SharedPrefixHandle;
    using CaptureOffer               = inkling2::CaptureOffer;
    using CacheSessionKey            = inkling2::PreparedSessionKey;
    using ContinuationSummary        = inkling2::ContinuationSummary;
    using SharedPrefixSummary        = inkling2::SharedPrefixSummary;
    using PressurePlanningSession    = inkling2::PressurePlanningSession;
    using PressureTargetHandle       = inkling2::PressureTargetHandle;
    using AssessedPressureTarget     = inkling2::AssessedPressureTarget;
    using CapturePressurePlan        = inkling2::CapturePressurePlan;
    using MaterializationResult      = inkling2::MaterializationResult;
    using ContextTransactionProgress = inkling2::ContextTransactionProgress;
    using CaptureAssessment          = inkling2::CaptureAssessment;
    using ActiveCaptureResult        = inkling2::ActiveCaptureResult;
    using PendingBatch               = inkling2::PendingBatch;
    using StartResult                = inkling2::StartResult;
    using PrefillProgress            = inkling2::PrefillProgress;
    using CommitResult               = inkling2::CommitResult;
    using DiscardResult              = inkling2::DiscardResult;
    using FinishResult               = inkling2::FinishResult;
    using AbortResult                = inkling2::AbortResult;
    using ReleaseResult              = inkling2::ReleaseResult;
    using Program                    = inkling2::Program;
};

} // namespace ninfer::models::inkling2
