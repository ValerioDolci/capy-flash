#include "models/inkling2/instance.h"

#include "artifact/formats.h"
#include "artifact/reader.h"
#include "core/startup.h"
#include "models/inkling2/load.h"
#include "models/inkling2/program/program.h"

#include <algorithm>
#include <chrono>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::inkling2 {
namespace {

using Clock = std::chrono::steady_clock;

// What this Program implements: one request at a time, text only, speculative decoding with the
// artifact's drafter only (--spec mtp), a device-only context cache.
void validate_options(const EngineOptions& options, const ExecutionContext& execution) {
    if (options.tp != 1 && options.tp != 2) {
        throw std::invalid_argument("Inkling 2 runs on one or two GPUs: pass --tp 1 or --tp 2");
    }
    if (execution.tp != options.tp || !execution.dev[0]) {
        throw std::invalid_argument("Engine execution context does not match options.tp");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    if (options.max_concurrency != 1) {
        throw std::invalid_argument("Inkling 2 serves one request at a time: pass "
                                    "--max-concurrency 1");
    }
    // The context cache keeps the last sequence of the one lane on the device (endpoint and prompt
    // anchor); it has no Host or disk tier.
    if (!options.context_cache.disk_directory.empty()) {
        throw std::invalid_argument("Inkling 2 context cache has no disk tier: remove "
                                    "--disk-cache-dir");
    }
    // --spec mtp: the artifact's DSpark drafter (component mtp), K drafts per round; an artifact
    // without the component decodes without it.
    if (options.speculative.backend == SpeculativeBackend::DFlash2 ||
        (options.speculative.backend == SpeculativeBackend::Mtp &&
         (options.speculative.draft_tokens == 0 || options.speculative.draft_tokens > 7 ||
          options.speculative.proposal_head != ProposalHead::Full))) {
        throw std::invalid_argument(
            "Inkling 2 speculative decoding is --spec mtp --draft-tokens 1..7 (the artifact's "
            "drafter), without --lm-head-draft");
    }
    if (options.enable_vision) {
        throw std::invalid_argument("Inkling 2 is text only: start without --vision");
    }
    if (options.kv_cache != KvCacheStorage::BFloat16) {
        throw std::invalid_argument("Inkling 2 keeps its KV cache in BF16: pass --kv-dtype bf16");
    }
}

// The artifact's own limits: the context the checkpoint declares (model_max_length) and the
// stacked attention inputs (one parent per input, tools/convert/inkling2_recipes.py) that the
// forward reads as one Linear.
void validate_artifact(const EngineOptions& options, const Config& config) {
    if (options.max_context > config.text.max_position_embeddings) {
        throw std::invalid_argument("Inkling 2 max_context " + std::to_string(options.max_context) +
                                    " exceeds the model's model_max_length " +
                                    std::to_string(config.text.max_position_embeddings) +
                                    ": lower --max-context");
    }
}

void validate_layout(const Model& model) {
    const AttentionWeights& attention = model.weights().text.layers.front().attention;
    const auto parent                 = [&](WeightId id) {
        const auto& parts = model.weight(id).view.parts;
        return parts.empty() ? nullptr : parts.front().parent;
    };
    if (parent(attention.query) != parent(attention.gate)) {
        throw std::invalid_argument(
            "Inkling 2 artifact has the old layout (separate attention input parents): reconvert "
            "the checkpoint with tools/convert/inkling2.py");
    }
}

std::size_t free_device_bytes(int device) {
    int previous = 0;
    CUDA_CHECK(cudaGetDevice(&previous));
    CUDA_CHECK(cudaSetDevice(device));
    std::size_t free_bytes   = 0;
    std::size_t total_bytes  = 0;
    const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    CUDA_CHECK(cudaSetDevice(previous));
    CUDA_CHECK(status);
    return free_bytes;
}

std::vector<std::size_t> rank_free_bytes(const ExecutionContext& execution) {
    std::vector<std::size_t> out;
    for (int rank = 0; rank < execution.tp; ++rank) {
        out.push_back(free_device_bytes(execution.dev[static_cast<std::size_t>(rank)]->device));
    }
    return out;
}

void synchronize_ranks(const ExecutionContext& execution) {
    for (int rank = 0; rank < execution.tp; ++rank) {
        execution.dev[static_cast<std::size_t>(rank)]->synchronize();
    }
}

} // namespace

ModelInstance::ModelInstance(std::unique_ptr<Model> source, const EngineOptions& options)
    : model(std::move(source)), parameters(*model),
      peer_parameters(model->device_count() == 2
                          ? std::make_unique<const execution::Parameters>(*model, 1)
                          : nullptr),
      frontend(make_frontend(
          {.tokenizer_json         = model->resources().tokenizer_json,
           .tokenizer_config_json  = model->resources().tokenizer_config_json,
           .chat_template_jinja    = model->resources().chat_template_jinja,
           .generation_config_json = model->resources().generation_config_json},
          {.chat_template_path = options.chat_template_path, .max_context = options.max_context})),
      capacity(options.max_context) {}

ModelInstance::~ModelInstance() = default;

bool is_inkling2_artifact(const std::filesystem::path& path) noexcept {
    try {
        const artifact::Reader reader(path);
        return is_inkling2(reader.directory());
    } catch (...) { return false; }
}

ConstructedModel construct_model(const EngineOptions& options, ExecutionContext& execution) {
    validate_options(options, execution);
    DeviceContext& device = execution.primary();
    const auto start      = Clock::now();
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect.complete();
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    validate_artifact(options, parse_config(reader.directory(), models::load_options(options)));
    auto plan = plan_load(reader, models::load_options(options));
    binding.complete();
    auto model = materialize_model(std::move(plan), execution, &options.startup_observer);
    synchronize_ranks(execution);
    validate_layout(*model);
    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<ModelInstance>(std::move(model), options);
    if ((execution.tp == 2) != (instance->peer_parameters != nullptr)) {
        throw std::logic_error("loaded Model device count does not match the Engine tp");
    }
    frontend.complete();

    StartupPhaseScope planning(options.startup_observer, StartupPhase::TargetFinalize);
    const std::string signature = "inkling2-text";
    auto context_cost           = runtime::resolve_context_machine_cost(
        {.hardware_class = runtime::context_cost_hardware_class(
             device.props.name, device.props.major, device.props.minor),
                   .prefill_signature = signature},
        options.context_cost.preset_path);
    auto planner = make_sequence_planner(instance->parameters, instance->peer_parameters.get(),
                                         execution, options);
    const auto budgets = rank_free_bytes(execution);
    auto resolution    = runtime::resolve_kv_capacity_symmetric(options.kv_capacity,
                                                                planner.capacity_curve(), budgets);
    auto sequence      = std::move(planner).finalize(resolution.main_page_groups);
    if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
        sequence.kv_capacity() != resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized Program plan");
    }
    instance->kv_capacity_resolution = resolution;
    planning.complete();

    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    instance->program = create_program(instance->parameters, instance->peer_parameters.get(),
                                       std::move(sequence), execution, options.startup_observer);
    synchronize_ranks(execution);
    program.complete();
    const auto after = rank_free_bytes(execution);
    instance->kv_capacity_resolution.available_after_startup_bytes =
        *std::min_element(after.begin(), after.end());

    const auto& stats = instance->model->storage_stats();
    LoadSummary summary;
    summary.architecture      = std::string(kTextArchitecture);
    summary.turn_anchors      = 0;
    summary.model_name        = instance->model->info().name;
    summary.prefill_signature = signature;
    std::set<std::string> formats;
    for (int rank = 0; rank < instance->model->device_count(); ++rank) {
        for (const auto& weight : instance->model->weight_data(rank)) {
            for (const auto& part : weight.view.parts) {
                formats.emplace(artifact::format_name(part.parent->geometry.format));
            }
        }
    }
    summary.weight_formats.assign(formats.begin(), formats.end());
    summary.load_seconds             = std::chrono::duration<double>(Clock::now() - start).count();
    summary.upload_seconds           = stats.upload_seconds;
    summary.artifact_bytes_read      = stats.read_bytes;
    summary.host_to_device_bytes     = stats.h2d_bytes;
    summary.peak_staging_bytes       = stats.peak_staging_bytes;
    summary.device_object_count      = stats.device_object_count;
    summary.host_object_count        = stats.host_object_count;
    summary.host_mapped_bytes        = stats.host_mapped_bytes;
    summary.host_pageable_bytes      = stats.host_pageable_bytes;
    summary.host_pageable_objects    = stats.host_pageable_objects;
    summary.host_mapped_objects      = stats.host_mapped_objects;
    summary.host_pinned_limit_bytes  = stats.host_pinned_limit_bytes;
    summary.host_pinned_alloc_failed = stats.host_pinned_alloc_failed;
    for (int i = 0; i < stats.device_count; ++i) {
        const auto rank = static_cast<std::size_t>(i);
        summary.devices.push_back({.device               = execution.dev[rank]->device,
                                   .capacity_bytes       = stats.per_device_capacity_bytes[rank],
                                   .host_to_device_bytes = stats.per_device_h2d_bytes[rank],
                                   .sharded_bytes        = stats.sharded_bytes[rank],
                                   .replicated_bytes     = stats.replicated_bytes[rank],
                                   .local_bytes          = stats.local_bytes[rank],
                                   .host_mapped_bytes = stats.per_device_host_mapped_bytes[rank]});
    }
    summary.context_cost       = std::move(context_cost.summary);
    summary.tp_transport       = instance->program->tp_transport().transport;
    summary.tp_eager_transport = instance->program->tp_transport().eager_transport;
    return {std::move(instance), std::move(summary), std::move(context_cost.model)};
}

} // namespace ninfer::models::inkling2
