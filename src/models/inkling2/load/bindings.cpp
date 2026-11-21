#include "models/inkling2/load/bindings.h"

#include "artifact/views.h"

#include <string>

namespace ninfer::models::inkling2::loading {

Resources bind_resources(artifact::Binder& binder) {
    const auto resource = [&](std::string_view role) {
        const auto bytes = binder.host_object(binder.resource("text", role));
        return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    Resources out;
    out.tokenizer_json         = resource("tokenizer.json");
    out.tokenizer_config_json  = resource("tokenizer_config.json");
    out.chat_template_jinja    = resource("chat_template.jinja");
    out.generation_config_json = resource("generation_config.json");
    return out;
}

WeightUseId Bindings::use(WeightId id, std::string_view input) const {
    const auto& parameter = at(id);
    for (std::size_t i = 0; i < parameter.uses.size(); ++i) {
        if (parameter.uses[i].input == input) { return {id, i}; }
    }
    throw artifact::ArtifactError(parameter.reference.name + ": unresolved Use " +
                                  std::string(input));
}

WeightId Bindings::parameter(std::string name, artifact::Shape shape,
                             std::vector<std::string> inputs, std::optional<QType> exact_format) {
    if (parameters_.contains(name)) {
        throw artifact::ArtifactError(name + ": duplicate model parameter declaration");
    }
    PendingWeight pending;
    pending.reference =
        binder.parameter(name, std::move(shape), artifact::Residency::Device, exact_format);
    for (auto& input : inputs) {
        const auto& use = binder.use(name, input);
        if (use.activation_policy != artifact::ActivationPolicy::A16Only ||
            !use.auxiliaries.empty()) {
            throw artifact::ArtifactError(name + "@" + input +
                                          ": Inkling 2 Uses are A16Only without auxiliaries");
        }
        pending.uses.push_back({std::move(input), ops::LinearPolicy::A16Only});
    }
    for (const auto& part : pending.reference.binding.parts) {
        pending.source_objects.push_back(
            artifact::object_id(binder.reader().directory().object(part.object)));
    }
    const WeightId id{weights.size()};
    parameters_.emplace(std::move(name), id);
    weights.push_back(std::move(pending));
    return id;
}

WeightId Bindings::direct(std::string name, artifact::Shape shape, QType format) {
    return parameter(std::move(name), std::move(shape), {}, format);
}

std::vector<BoundWeight> resolve_weights(std::vector<PendingWeight>&& pending,
                                         const artifact::MaterializedArtifact& materialized) {
    std::vector<BoundWeight> out;
    out.reserve(pending.size());
    for (auto& item : pending) {
        auto view = artifact::bind_view(item.reference, materialized);
        out.push_back({std::move(item.reference.name), std::move(item.source_objects),
                       std::move(view), std::move(item.uses)});
    }
    return out;
}

std::vector<BoundWeight> resolve_weights(std::span<const PendingWeight> pending,
                                         const artifact::MaterializedArtifact& materialized,
                                         int device) {
    std::vector<BoundWeight> out;
    out.reserve(pending.size());
    for (const auto& item : pending) {
        const auto& reference = item.reference;
        std::size_t held      = 0;
        for (const auto& part : reference.binding.parts) {
            held += materialized.has_device(part.object, device) ? 1 : 0;
        }
        WeightView view;
        if (held == reference.binding.parts.size()) {
            view = artifact::bind_view(reference, materialized, device);
        } else if (held != 0) {
            throw artifact::ArtifactError(reference.name + ": device " + std::to_string(device) +
                                          " holds only some of its parents");
        }
        out.push_back({reference.name, item.source_objects, std::move(view), item.uses});
    }
    return out;
}

} // namespace ninfer::models::inkling2::loading
