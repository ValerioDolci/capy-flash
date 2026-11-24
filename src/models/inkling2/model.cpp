#include "models/inkling2/model.h"

#include "core/device.h"
#include "core/device_scope.h"
#include "core/weight_view.h"

#include <bit>
#include <cstdint>
#include <vector>

#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::inkling2 {

Model::Model(Config config, LoadOptions options, ModelWeights weights, DeviceWeights bound,
             ExpertBanks banks, int device_count, Resources resources, InstanceInfo info,
             artifact::MaterializedArtifact backing)
    : backing_(std::move(backing)), config_(std::move(config)), options_(options),
      weights_(std::move(weights)), bound_(std::move(bound)), banks_(std::move(banks)),
      device_count_(device_count), resources_(resources), info_(std::move(info)) {}

Model::~Model() = default;

std::size_t Model::checked_device(int device) const {
    if (device < 0 || device >= device_count_) {
        throw std::out_of_range("model device " + std::to_string(device) + " is outside the " +
                                std::to_string(device_count_) + "-device backing");
    }
    return static_cast<std::size_t>(device);
}

bool Model::resident(WeightId id, int device) const {
    return !bound_[checked_device(device)].at(id.index).view.parts.empty();
}

std::span<const std::unique_ptr<ExpertBank>> Model::expert_banks(std::uint32_t layer,
                                                                 int device) const {
    return banks_.layers[checked_device(device)].at(layer);
}

std::uint64_t Model::expert_bank_bytes(int device) const {
    return banks_.storage[checked_device(device)].bytes;
}

Tensor Model::narrowed(WeightId id, int device) const {
    const auto& copies = narrowed_[checked_device(device)];
    if (id.index >= copies.size() || copies[id.index].p == nullptr) { return Tensor{}; }
    const auto& shape = weight(id, device).view.shape;
    return Tensor(copies[id.index].p, DType::BF16,
                  {static_cast<std::int32_t>(shape.at(1)), static_cast<std::int32_t>(shape.at(0))});
}

void Model::narrow_message_gates() {
    for (int device = 0; device < device_count_; ++device) {
        auto& copies = narrowed_[checked_device(device)];
        copies.resize(bound_[checked_device(device)].size());
        for (const auto& layer : weights_.text.layers) {
            if (!layer.core) { continue; }
            for (const auto& gate : layer.core->gates) {
                for (const WeightId id : {gate.forget, gate.gain}) {
                    const auto& view = weight(id, device).view;
                    if (view.shape.size() != 2) {
                        throw std::logic_error("Inkling 2: a message gate matrix is not 2-D");
                    }
                    const Tensor wide =
                        weight_tensor(view, {static_cast<std::int32_t>(view.shape[1]),
                                             static_cast<std::int32_t>(view.shape[0])});
                    // The copy lives on the device of the FP32 view.
                    cudaPointerAttributes where{};
                    CUDA_CHECK(cudaPointerGetAttributes(&where, wide.data));
                    const ScopedCurrentDevice scope(where.device);
                    const auto n = static_cast<std::size_t>(wide.numel());
                    std::vector<float> host(n);
                    CUDA_CHECK(cudaMemcpy(host.data(), wide.data, n * sizeof(float),
                                          cudaMemcpyDeviceToHost));
                    std::vector<std::uint16_t> narrow(n);
                    for (std::size_t i = 0; i < n; ++i) {
                        const auto bits = std::bit_cast<std::uint32_t>(host[i]);
                        if ((bits & 0xFFFFU) != 0U) {
                            throw std::runtime_error(
                                "Inkling 2: a message gate weight is not a BF16 value");
                        }
                        narrow[i] = static_cast<std::uint16_t>(bits >> 16);
                    }
                    copies[id.index] = DeviceBuffer(n * sizeof(std::uint16_t));
                    copies[id.index].copy_from_host(narrow.data(), n * sizeof(std::uint16_t));
                }
            }
        }
    }
}

const BoundWeight& Model::weight(WeightId id, int device) const {
    const auto& bound = bound_[checked_device(device)].at(id.index);
    if (bound.view.parts.empty()) {
        throw std::invalid_argument(bound.name + " is not resident on device " +
                                    std::to_string(device));
    }
    return bound;
}

ops::WeightInput Model::input(WeightUseId id, int device) const {
    const auto& parameter = weight(id.parameter, device);
    return {parameter.view, parameter.uses.at(id.use_index).policy, std::nullopt};
}

ops::WeightInput Model::input(WeightId id, int device) const {
    if (weight(id, device).uses.size() != 1) {
        throw std::invalid_argument("weight input requires an explicit mathematical use");
    }
    return input(WeightUseId{id, 0}, device);
}

} // namespace ninfer::models::inkling2
