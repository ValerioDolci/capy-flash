#include "models/inkling2/load.h"

#include "artifact/reader.h"
#include "models/inkling2/load/bindings.h"
#include "models/inkling2/load/expert_banks.h"
#include "models/inkling2/load/sharding.h"

#include <array>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::inkling2 {

struct LoadPlan::Impl {
    Config config;
    LoadOptions options;
    ModelWeights weights;
    std::vector<loading::PendingWeight> pending;
    artifact::MaterializationPlan materialization;
    Resources resources;
    InstanceInfo info;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const Config& LoadPlan::config() const { return impl_->config; }

const ModelWeights& LoadPlan::weights() const { return impl_->weights; }

const Resources& LoadPlan::resources() const { return impl_->resources; }

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    return impl_->materialization;
}

std::size_t LoadPlan::parameter_count() const { return impl_->pending.size(); }

const artifact::ParameterReference& LoadPlan::parameter(WeightId id) const {
    return impl_->pending.at(id.index).reference;
}

std::span<const WeightUse> LoadPlan::uses(WeightId id) const {
    return impl_->pending.at(id.index).uses;
}

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options) {
    if (options.tp < 1 || options.tp > static_cast<int>(artifact::kMaximumDevices)) {
        throw std::invalid_argument("tensor parallelism must be 1 or 2");
    }
    auto out     = std::make_unique<LoadPlan::Impl>();
    out->options = options;
    out->config  = parse_config(reader.directory(), options);
    artifact::Binder binder(reader, options.tp);
    out->resources = loading::bind_resources(binder);
    loading::Bindings bindings(binder);
    out->weights.text = loading::bind_text(bindings, out->config.text);
    if (out->config.draft) {
        out->weights.draft = loading::bind_draft(bindings, *out->config.draft, out->config.text);
    }
    if (out->config.dflash2) {
        out->weights.dflash2 =
            loading::bind_dflash2(bindings, *out->config.dflash2, out->config.text);
    }
    loading::install_shard_resolver(binder, bindings.weights, out->config.text, options,
                                    out->config.draft ? &*out->config.draft : nullptr,
                                    out->config.dflash2 ? &*out->config.dflash2 : nullptr);
    out->pending              = std::move(bindings.weights);
    out->materialization      = std::move(binder).finish();
    out->info.name            = reader.directory().metadata.value("name", "inkling2");
    out->info.metadata_json   = reader.directory().metadata.dump();
    out->info.provenance_json = reader.directory().provenance.dump();
    out->info.artifact_id     = reader.artifact_id();
    return LoadPlan(std::move(out));
}

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                         const StartupObserver* observer) {
    if (!plan.impl_) { throw artifact::ArtifactError("load plan was already consumed"); }
    if (plan.impl_->options.tp != 1) {
        throw std::invalid_argument("a one-device materialization needs a tp 1 plan");
    }
    auto data    = std::move(plan.impl_);
    auto backing = artifact::materialize(*data->materialization.source,
                                         std::move(data->materialization), device, observer);
    const std::array<const DeviceContext*, 1> devices{&device};
    auto banks = loading::cut_expert_banks(data->pending, data->weights.text, data->config.text,
                                           backing, devices);
    Model::DeviceWeights bound;
    bound[0]   = loading::resolve_weights(std::move(data->pending), backing);
    auto model = std::unique_ptr<Model>(new Model(
        std::move(data->config), data->options, std::move(data->weights), std::move(bound),
        std::move(banks), 1, data->resources, std::move(data->info), std::move(backing)));
    model->narrow_message_gates();
    return model;
}

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, ExecutionContext& execution,
                                         const StartupObserver* observer) {
    if (!plan.impl_) { throw artifact::ArtifactError("load plan was already consumed"); }
    const int tp = plan.impl_->options.tp;
    if (execution.tp != tp) {
        throw std::invalid_argument(
            "ExecutionContext tensor parallelism differs from the load plan");
    }
    if (tp == 1) { return materialize_model(std::move(plan), execution.primary(), observer); }
    auto data    = std::move(plan.impl_);
    auto backing = artifact::materialize(*data->materialization.source,
                                         std::move(data->materialization), execution, observer);
    std::array<const DeviceContext*, artifact::kMaximumDevices> devices{};
    Model::DeviceWeights bound;
    for (int device = 0; device < tp; ++device) {
        devices[static_cast<std::size_t>(device)] =
            &*execution.dev[static_cast<std::size_t>(device)];
        bound[static_cast<std::size_t>(device)] =
            loading::resolve_weights(data->pending, backing, device);
    }
    auto banks = loading::cut_expert_banks(
        data->pending, data->weights.text, data->config.text, backing,
        std::span<const DeviceContext* const>(devices.data(), static_cast<std::size_t>(tp)));
    auto model = std::unique_ptr<Model>(new Model(
        std::move(data->config), data->options, std::move(data->weights), std::move(bound),
        std::move(banks), tp, data->resources, std::move(data->info), std::move(backing)));
    model->narrow_message_gates();
    return model;
}

} // namespace ninfer::models::inkling2
