#pragma once

#include "core/device.h"
#include "models/inkling2/execution/parameters.h"
#include "models/inkling2/model.h"
#include "models/inkling2/program/runtime_types.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/kv_capacity.h"

#include <memory>

namespace ninfer::artifact {
class Reader;
}

namespace ninfer::models::inkling2 {

// The Inkling 2 model instance the common Engine cores run (runtime::EngineCore,
// runtime::CausalScoreCore): the same members as runtime::ModelInstance over this family's
// contract.
struct ModelInstance {
    using ModelContract = RuntimeTypes;

    std::unique_ptr<Model> model;
    const execution::Parameters parameters;
    // Rank 1's Parameters of a two-device Model; null at tensor-parallel width 1.
    const std::unique_ptr<const execution::Parameters> peer_parameters;
    Frontend frontend;
    runtime::KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<Program> program;

    ModelInstance(std::unique_ptr<Model> model, const EngineOptions& options);
    ~ModelInstance();
    ModelInstance(const ModelInstance&)            = delete;
    ModelInstance& operator=(const ModelInstance&) = delete;
};

struct ConstructedModel {
    std::unique_ptr<ModelInstance> instance;
    LoadSummary load;
    runtime::ContextMachineCostModel context_cost;
};

// Whether `path` opens as an artifact whose Text component is Inkling 2 (config.h is_inkling2).
// False for anything else, including a path that does not open: the other family then reports it.
[[nodiscard]] bool is_inkling2_artifact(const std::filesystem::path& path) noexcept;

// Construction over every rank of `execution` from options normalized by
// runtime::normalize_engine_options; execution.tp equals options.tp.
[[nodiscard]] ConstructedModel construct_model(const EngineOptions& options,
                                               ExecutionContext& execution);

} // namespace ninfer::models::inkling2
