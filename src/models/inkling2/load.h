#pragma once

#include "models/inkling2/model.h"

#include <memory>
#include <span>

namespace ninfer::artifact {
class Reader;
struct ParameterReference;
} // namespace ninfer::artifact

namespace ninfer::models::inkling2 {

// Cold load plan; it borrows its Reader until materialization. No device allocation happens in
// plan_load.
class LoadPlan {
public:
    ~LoadPlan();
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const Config& config() const;
    [[nodiscard]] const ModelWeights& weights() const;
    [[nodiscard]] const Resources& resources() const;
    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;
    [[nodiscard]] std::size_t parameter_count() const;
    [[nodiscard]] const artifact::ParameterReference& parameter(WeightId id) const;
    [[nodiscard]] std::span<const WeightUse> uses(WeightId id) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit LoadPlan(std::unique_ptr<Impl> impl);
    friend LoadPlan plan_load(const artifact::Reader&, LoadOptions);
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, ExecutionContext&,
                                                    const StartupObserver*);
};

// Plans the Text backbone of an Inkling 2 artifact (models/inkling2/config.h) for options.tp
// devices with the placement of load/sharding.h. Throws artifact::ArtifactError for an artifact
// whose Text component is not Inkling 2 and for options other than the Text backbone.
[[nodiscard]] LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options = {});
// Requires a plan for one device.
[[nodiscard]] std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                                       const StartupObserver* observer = nullptr);
// Requires execution.tp == options.tp. Device i of the Model is execution.dev[i]; at tp 1 this is
// the DeviceContext overload on execution.primary().
[[nodiscard]] std::unique_ptr<Model> materialize_model(LoadPlan&& plan, ExecutionContext& execution,
                                                       const StartupObserver* observer = nullptr);

} // namespace ninfer::models::inkling2
