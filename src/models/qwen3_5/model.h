#pragma once

#include "artifact/framing.h"
#include "artifact/materializer.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/weights.h"
#include "ninfer/ops/weight_input.h"

#include <array>
#include <memory>
#include <span>
#include <string>

namespace ninfer::models::qwen3_5 {

struct InstanceInfo {
    std::string name;
    std::string metadata_json;
    std::string provenance_json;
    artifact::ArtifactId artifact_id{};
};

class LoadPlan;

// Where one routed expert of a Qwen4Exp block lives in its layer's stacked NVFP4 banks: for the
// gate|up bank (its gate rows, then its up rows) and for the down bank, the parent and the byte
// ranges from the parent's first byte that hold the expert's codes, swizzled block scales and its
// FP32 divisor words. An expert cache slot is these spans copied back to back; every span is a
// contiguous range of the host-mapped bank (3 per bank for Qwen3.8-Flash-Next).
struct ExpertBankSpans {
    const WeightParent* parent = nullptr;
    std::uint64_t row_begin    = 0;
    std::uint64_t row_count    = 0;
    std::vector<ByteSpan> spans;
};

struct RoutedExpertSpans {
    ExpertBankSpans gate_up, down;

    [[nodiscard]] std::uint64_t bytes() const noexcept;
};

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

    // False when `device` holds none of the parameter's parents (a PrimaryOnly or SingleDevice
    // placement on the other rank).
    [[nodiscard]] bool resident(WeightId id, int device = 0) const;

    // The view on `device`: a complete parent, or that rank's Rows/Columns shard with the split
    // axis of its logical shape narrowed. Throws when the parameter is not resident there.
    [[nodiscard]] const BoundWeight& weight(WeightId id, int device = 0) const;

    // A Residency::HostFile weight (the Qwen4Exp n-gram table): its mapped object segments and
    // geometry, shared by every device. Throws for any other weight.
    [[nodiscard]] const BoundWeight& file_weight(WeightId id) const;

    [[nodiscard]] ops::WeightInput input(WeightUseId id, int device = 0) const;
    [[nodiscard]] ops::WeightInput input(WeightId id, int device = 0) const;

    // Indexed by WeightId; entries not resident on `device` have an empty view.
    [[nodiscard]] std::span<const BoundWeight> weight_data(int device = 0) const {
        return bound_.at(checked_device(device));
    }

    [[nodiscard]] const FrontendResources& resources() const noexcept { return resources_; }

    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }

    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }

    // Exclusive expert tier: the expert banks' start-up tails (artifact::HostTails), or null.
    [[nodiscard]] artifact::HostTails* host_tails() const noexcept { return backing_.host_tails(); }

private:
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, ExecutionContext&,
                                                    const StartupObserver*);
    using DeviceWeights = std::array<std::vector<BoundWeight>, artifact::kMaximumDevices>;

    Model(Config config, LoadOptions options, ModelWeights weights, DeviceWeights bound,
          int device_count, FrontendResources resources, InstanceInfo info,
          artifact::MaterializedArtifact backing);

    [[nodiscard]] std::size_t checked_device(int device) const;

    // Destroy all borrowers before backing. The caller keeps its DeviceContext or ExecutionContext
    // alive through cleanup.
    artifact::MaterializedArtifact backing_;
    Config config_;
    LoadOptions options_;
    ModelWeights weights_;
    DeviceWeights bound_;
    int device_count_ = 1;
    FrontendResources resources_;
    InstanceInfo info_;
};

// Throws unless `expert` of `moe` binds a gate and an up matrix that are adjacent rows of one
// parent and a down matrix that is a row block of one parent, with NVFP4 rows on 128-row tiles.
[[nodiscard]] RoutedExpertSpans routed_expert_spans(const Model& model, const MoeWeights& moe,
                                                    std::uint32_t expert, int device = 0);

} // namespace ninfer::models::qwen3_5
