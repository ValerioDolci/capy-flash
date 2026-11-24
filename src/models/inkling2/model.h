#pragma once

#include "artifact/framing.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "artifact/materializer.h"
#include "models/inkling2/config.h"
#include "models/inkling2/expert_bank.h"
#include "models/inkling2/weights.h"
#include "ninfer/ops/weight_input.h"

#include <array>
#include <memory>
#include <vector>
#include <span>
#include <string>
#include <string_view>

namespace ninfer::models::inkling2 {

// Frontend resources of the Text component; the views borrow the model's Host backing.
struct Resources {
    std::string_view tokenizer_json;
    std::string_view tokenizer_config_json;
    std::string_view chat_template_jinja;
    std::string_view generation_config_json;
};

struct InstanceInfo {
    std::string name;
    std::string metadata_json;
    std::string provenance_json;
    artifact::ArtifactId artifact_id{};
};

class LoadPlan;

class Model {
public:
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&)                 = delete;
    Model& operator=(Model&&)      = delete;

    [[nodiscard]] const Config& config() const noexcept { return config_; }

    [[nodiscard]] const LoadOptions& options() const noexcept { return options_; }

    [[nodiscard]] const ModelWeights& weights() const noexcept { return weights_; }

    // Tensor-parallel degree of the backing: 1, or 2 with one view set per rank.
    [[nodiscard]] int device_count() const noexcept { return device_count_; }

    // False for a parameter without device views: a routed expert (HostFile, served by
    // expert_banks).
    [[nodiscard]] bool resident(WeightId id, int device = 0) const;

    // The routed-expert banks `device` computes for `layer`, in call order: every expert at half
    // the intermediate width, rank r's half at tp 2, both halves at tp 1 (expert_bank.h).
    [[nodiscard]] std::span<const std::unique_ptr<ExpertBank>> expert_banks(std::uint32_t layer,
                                                                            int device = 0) const;
    // Device bytes of those banks over every layer (not part of storage_stats()).
    [[nodiscard]] std::uint64_t expert_bank_bytes(int device = 0) const;

    // The view on `device`: a complete parent, or that rank's Rows/Columns shard with the split
    // axis of its logical shape narrowed (load/sharding.h). Throws when it is not resident there.
    [[nodiscard]] const BoundWeight& weight(WeightId id, int device = 0) const;

    [[nodiscard]] ops::WeightInput input(WeightUseId id, int device = 0) const;
    // The parameter's one Use.
    [[nodiscard]] ops::WeightInput input(WeightId id, int device = 0) const;

    // Indexed by WeightId; entries not resident on `device` have an empty view.
    [[nodiscard]] std::span<const BoundWeight> weight_data(int device = 0) const {
        return bound_.at(checked_device(device));
    }

    // BF16 copy on `device` of a message gate's forget or gain matrix (the checkpoint stores them
    // as BF16; the artifact widens them exactly to FP32): ne {r, H} like the FP32 view. Empty for
    // any other parameter.
    [[nodiscard]] Tensor narrowed(WeightId id, int device = 0) const;

    [[nodiscard]] const Resources& resources() const noexcept { return resources_; }

    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }

    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }

private:
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, ExecutionContext&,
                                                    const StartupObserver*);
    using DeviceWeights = std::array<std::vector<BoundWeight>, artifact::kMaximumDevices>;

    Model(Config config, LoadOptions options, ModelWeights weights, DeviceWeights bound,
          ExpertBanks banks, int device_count, Resources resources, InstanceInfo info,
          artifact::MaterializedArtifact backing);

    [[nodiscard]] std::size_t checked_device(int device) const;

    // Fills narrowed_ for every message gate's forget and gain on every device (at load).
    void narrow_message_gates();

    // Destroy all borrowers before backing. The caller keeps its DeviceContext or ExecutionContext
    // alive through cleanup.
    artifact::MaterializedArtifact backing_;
    Config config_;
    LoadOptions options_;
    ModelWeights weights_;
    DeviceWeights bound_;
    ExpertBanks banks_;
    // Indexed by WeightId per device; empty buffers for every parameter but forget and gain.
    std::array<std::vector<DeviceBuffer>, artifact::kMaximumDevices> narrowed_;
    int device_count_ = 1;
    Resources resources_;
    InstanceInfo info_;
};

} // namespace ninfer::models::inkling2
