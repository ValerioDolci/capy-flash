#pragma once

#include "artifact/binder.h"
#include "models/inkling2/config.h"
#include "models/inkling2/draft/config.h"
#include "models/inkling2/model.h"
#include "models/inkling2/weights.h"

#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::inkling2::loading {

[[nodiscard]] Resources bind_resources(artifact::Binder& binder);

struct PendingWeight {
    artifact::ParameterReference reference;
    std::vector<WeightUse> uses;
    std::vector<std::string> source_objects;
};

class Bindings {
public:
    explicit Bindings(artifact::Binder& binder) : binder(binder) {}

    // A parameter with its linear Uses. Every Use must be A16Only without auxiliaries: Inkling 2
    // never quantizes activations.
    [[nodiscard]] WeightId parameter(std::string name, artifact::Shape shape,
                                     std::vector<std::string> inputs   = {},
                                     std::optional<QType> exact_format = {});
    [[nodiscard]] WeightId direct(std::string name, artifact::Shape shape,
                                  QType format = QType::BF16);

    [[nodiscard]] const PendingWeight& at(WeightId id) const { return weights.at(id.index); }

    [[nodiscard]] WeightUseId use(WeightId id, std::string_view input) const;

    artifact::Binder& binder;
    std::vector<PendingWeight> weights;

private:
    std::map<std::string, WeightId, std::less<>> parameters_;
};

[[nodiscard]] TextWeights bind_text(Bindings& bindings, const TextConfig& config);
// The drafter of the `mtp` component (weights.h, mtp/ names).
[[nodiscard]] DraftWeights bind_draft(Bindings& bindings, const draft::DraftConfig& config,
                                      const TextConfig& text);

[[nodiscard]] std::vector<BoundWeight>
resolve_weights(std::vector<PendingWeight>&& pending,
                const artifact::MaterializedArtifact& materialized);
// One tensor-parallel rank's views. A parameter whose parents `device` does not hold (an expert
// bank placed SingleDevice on the other rank) keeps its name and uses with an empty view.
[[nodiscard]] std::vector<BoundWeight>
resolve_weights(std::span<const PendingWeight> pending,
                const artifact::MaterializedArtifact& materialized, int device);

} // namespace ninfer::models::inkling2::loading
