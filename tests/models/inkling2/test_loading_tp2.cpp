// Inkling 2 loading at tp 1 and tp 2: plan and materialize an artifact (the synthetic reference
// checkpoint's, or the released model's) with the placement of load/sharding.h.
//
// Plan (host only): the Qwen path rejects the artifact; at tp 2 every parameter's parent carries
// the placement of expected_placement.h (heads split, output head split by vocabulary rows, expert
// banks whole on rank (ID >= E/2), the rest replicated) and the per-rank device bytes are printed
// by class.
// Materialize (tp CUDA devices): the device memory each rank's weights take, every device parent
// byte-identical to the slice of its artifact parent the host applies itself (reader bytes +
// PlaneCopy list), each rank's views with the narrowed logical shapes, and each rank's
// Parameters: its expert banks (both at tp 1, one at tp 2 with first ID r E/2), its projection
// shards.

#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/reader.h"
#include "artifact/slices.h"
#include "core/device.h"
#include "models/inkling2/execution/parameters.h"
#include "models/inkling2/expected_placement.h"
#include "models/inkling2/load.h"
#include "models/qwen3_5/load.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace ink = ninfer::models::inkling2;
using artifact::ShardAxis;
using ninfer::test::artifact_fixture::require;
using ninfer::test::inkling2::expected_placement;
using ninfer::test::inkling2::rank_shape;

const char* axis_name(ShardAxis axis) {
    switch (axis) {
    case ShardAxis::Replicated:
        return "replicated";
    case ShardAxis::Rows:
        return "rows";
    case ShardAxis::Columns:
        return "columns";
    case ShardAxis::PrimaryOnly:
        return "primary";
    case ShardAxis::SingleDevice:
        return "single-device";
    case ShardAxis::HostMapped:
        return "host-mapped";
    }
    return "?";
}

std::uint64_t free_bytes(int device) {
    std::size_t free = 0, total = 0;
    if (cudaSetDevice(device) != cudaSuccess || cudaMemGetInfo(&free, &total) != cudaSuccess) {
        throw std::runtime_error("cudaMemGetInfo failed");
    }
    return free;
}

int check_parameters(const ink::Model& model, int device, int tp) {
    const ink::execution::Parameters parameters(model, device);
    const auto& c   = model.config().text;
    const auto& t   = parameters.text;
    const auto half = static_cast<std::int32_t>(c.num_experts / 2);
    int failures    = 0;
    for (std::size_t layer = 0; layer < t.layers.size(); ++layer) {
        const auto& p     = t.layers[layer];
        const auto& banks = p.moe.banks;
        const bool ok     = tp == 1 ? banks.size() == 2 && banks[0].first_expert == 0 &&
                                      banks[1].first_expert == half && banks[0].experts == half &&
                                      banks[1].experts == half
                                    : banks.size() == 1 && banks[0].first_expert == device * half &&
                                      banks[0].experts == half;
        if (!ok) {
            std::cerr << "layer " << layer << " rank " << device << ": unexpected expert banks\n";
            ++failures;
        }
        for (const auto& bank : banks) {
            const std::uint64_t rows = std::uint64_t(bank.experts) * 2 * c.expert_intermediate;
            if (bank.gate_up.shape != std::vector<std::uint64_t>{rows, c.hidden_size} ||
                bank.down.shape !=
                    std::vector<std::uint64_t>{std::uint64_t(bank.experts) * c.hidden_size,
                                               c.expert_intermediate}) {
                std::cerr << "layer " << layer << ": bank view shapes differ\n";
                ++failures;
            }
        }
        const auto q  = static_cast<std::int32_t>(c.query_width() / tp);
        const auto kv = static_cast<std::int32_t>(c.key_width() / tp);
        if (p.attention.query.weight.n != q || p.attention.key.weight.n != kv ||
            p.attention.output.weight.k != q ||
            p.attention.sink.weight.n != static_cast<std::int32_t>(c.num_attention_heads / tp) ||
            p.attention.query_scale.ne[1] !=
                static_cast<std::int32_t>(c.num_attention_heads / tp) ||
            p.input_norms.size() != (c.global(static_cast<std::uint32_t>(layer)) ? 3U : 1U) ||
            p.core.has_value() != c.core(static_cast<std::uint32_t>(layer))) {
            std::cerr << "layer " << layer << " rank " << device << ": shard extents differ\n";
            ++failures;
        }
    }
    if (t.output_head.weight.n != static_cast<std::int32_t>(c.unpadded_vocab_size / tp) ||
        t.token_embedding.ne[1] != static_cast<std::int32_t>(c.vocab_size)) {
        std::cerr << "rank " << device << ": vocabulary extents differ\n";
        ++failures;
    }
    return failures;
}

int run(const std::filesystem::path& path, int tp, bool plan_only) {
    artifact::Reader reader(path);
    try {
        (void)qwen3_5::plan_load(reader, {});
        std::cerr << "the Qwen path accepted an Inkling 2 artifact\n";
        return 1;
    } catch (const artifact::ArtifactError& error) {
        std::printf("Qwen path rejects it: %s\n", error.what());
    }
    auto plan          = ink::plan_load(reader, LoadOptions{.tp = tp});
    const auto& config = plan.config().text;
    const auto& mplan  = plan.materialization();
    require(mplan.device_count == tp, "the plan is not for tp devices");

    std::map<std::size_t, std::array<const artifact::DevicePlacement*, 2>> placed;
    for (const auto& item : mplan.device_objects) {
        placed[item.object.index][static_cast<std::size_t>(item.device)] = &item;
    }
    int failures = 0;
    std::map<std::string, std::array<std::uint64_t, 2>> by_kind;
    std::array<std::uint64_t, 2> total{};
    std::set<std::size_t> counted;
    std::map<std::string, std::vector<std::uint64_t>> shapes; // parameter -> rank shape
    for (std::size_t index = 0; index < plan.parameter_count(); ++index) {
        const auto& reference = plan.parameter(ink::WeightId{index});
        const auto& name      = reference.name;
        auto expect           = expected_placement(name, config.num_experts);
        if (tp == 1) { expect = {ShardAxis::Replicated, 0, expect.kind}; }
        for (const auto& part : reference.binding.parts) {
            const auto object = part.object.index;
            const auto found  = placed.find(object);
            if (found == placed.end()) {
                std::cerr << name << ": parent has no device placement\n";
                ++failures;
                continue;
            }
            for (int device = 0; device < tp; ++device) {
                const auto* item = found->second[static_cast<std::size_t>(device)];
                const bool held = expect.axis != ShardAxis::SingleDevice || expect.holder == device;
                if ((item != nullptr) != held) {
                    std::cerr << name << ": device " << device << (held ? " lacks" : " holds")
                              << " the parent\n";
                    ++failures;
                    continue;
                }
                if (item == nullptr) { continue; }
                if (item->axis != expect.axis ||
                    (expect.axis == ShardAxis::SingleDevice && item->device != expect.holder)) {
                    std::cerr << name << ": device " << device << " placement "
                              << axis_name(item->axis) << ", expected " << axis_name(expect.axis)
                              << '\n';
                    ++failures;
                }
                if (counted.insert(object * 2 + static_cast<std::size_t>(device)).second) {
                    by_kind[expect.kind][static_cast<std::size_t>(device)] += item->bytes;
                    total[static_cast<std::size_t>(device)] += item->bytes;
                }
            }
        }
        if (artifact::is_sharded(expect.axis)) {
            shapes[name] = rank_shape(reference.shape, expect.axis);
        }
    }
    std::printf("%s: tp %d, %zu parameters, %zu device parents\n", path.filename().c_str(), tp,
                plan.parameter_count(), mplan.device_objects.size());
    for (const auto& [kind, bytes] : by_kind) {
        std::printf("  %-22s rank0 %13llu B  rank1 %13llu B\n", kind.c_str(),
                    static_cast<unsigned long long>(bytes[0]),
                    static_cast<unsigned long long>(bytes[1]));
    }
    std::printf("  %-22s rank0 %13llu B  rank1 %13llu B\n", "total",
                static_cast<unsigned long long>(total[0]),
                static_cast<unsigned long long>(total[1]));
    if (failures != 0 || plan_only) { return failures; }

    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < tp) {
        std::cout << "SKIP: materialization needs " << tp << " CUDA devices\n";
        return 77;
    }
    const std::vector<artifact::DevicePlacement> placements = mplan.device_objects;
    std::vector<std::optional<std::size_t>> parameter_object(plan.parameter_count());
    for (std::size_t index = 0; index < plan.parameter_count(); ++index) {
        const auto& parts = plan.parameter(ink::WeightId{index}).binding.parts;
        bool one          = !parts.empty();
        for (const auto& part : parts) { one = one && part.object == parts.front().object; }
        if (one) { parameter_object[index] = parts.front().object.index; }
    }
    std::vector<int> ids;
    for (int device = 0; device < tp; ++device) { ids.push_back(device); }
    ExecutionContext execution(ids);
    std::array<std::uint64_t, 2> before{};
    for (int device = 0; device < tp; ++device) {
        before[static_cast<std::size_t>(device)] = free_bytes(device);
    }
    auto model = ink::materialize_model(std::move(plan), execution);
    for (int device = 0; device < tp; ++device) {
        const auto used = before[static_cast<std::size_t>(device)] - free_bytes(device);
        std::printf("  device memory taken by the weights, rank %d: %llu B (%.2f GiB)\n", device,
                    static_cast<unsigned long long>(used), double(used) / (1ULL << 30));
    }

    std::map<std::pair<std::size_t, int>, const WeightParent*> parents;
    std::size_t views_checked = 0;
    for (int device = 0; device < tp; ++device) {
        const auto data = model->weight_data(device);
        for (std::size_t index = 0; index < data.size(); ++index) {
            const auto& bound = data[index];
            if (bound.view.parts.empty()) { continue; }
            const auto shape = shapes.find(bound.name);
            if (shape != shapes.end()) {
                ++views_checked;
                if (bound.view.shape != shape->second) {
                    std::cerr << bound.name << ": rank " << device << " view shape differs\n";
                    ++failures;
                }
            }
            if (parameter_object[index]) {
                parents[{*parameter_object[index], device}] = bound.view.parts.front().parent;
            }
        }
    }
    std::uint64_t compared       = 0;
    std::size_t parents_compared = 0;
    for (const auto& placement : placements) {
        const auto found = parents.find({placement.object.index, placement.device});
        if (found == parents.end()) {
            std::cerr << "object " << placement.object.index << " device " << placement.device
                      << ": no bound parameter reaches it\n";
            ++failures;
            continue;
        }
        const WeightParent& parent = *found->second;
        auto source                = reader.read_object(placement.object);
        std::vector<std::byte> want;
        if (artifact::is_sharded(placement.axis)) {
            want.assign(placement.bytes, std::byte{0});
            for (const auto& copy : placement.copies) {
                std::memcpy(want.data() + copy.dest_offset, source.data() + copy.source_offset,
                            copy.bytes);
            }
        } else {
            want = std::move(source);
        }
        require(parent.geometry.bytes == want.size(), "device parent size");
        std::vector<std::byte> got(want.size());
        if (cudaSetDevice(placement.device) != cudaSuccess ||
            cudaMemcpy(got.data(), parent.data, got.size(), cudaMemcpyDeviceToHost) !=
                cudaSuccess) {
            throw std::runtime_error("download of a device parent failed");
        }
        if (got != want) {
            std::cerr << "object " << placement.object.index << " device " << placement.device
                      << " (" << axis_name(placement.axis) << "): bytes differ from the slice\n";
            ++failures;
        }
        compared += got.size();
        ++parents_compared;
    }
    std::printf("materialized: %zu device parents compared byte for byte (%llu B), %zu split views "
                "checked\n",
                parents_compared, static_cast<unsigned long long>(compared), views_checked);
    for (int device = 0; device < tp; ++device) {
        failures += check_parameters(*model, device, tp);
    }
    std::printf("Parameters built and checked on %d rank(s)\n", tp);
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path path;
        int tp         = 2;
        bool plan_only = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--artifact" && i + 1 < argc) {
                path = argv[++i];
            } else if (arg == "--tp" && i + 1 < argc) {
                tp = std::stoi(argv[++i]);
            } else if (arg == "--plan-only") {
                plan_only = true;
            } else {
                std::cerr << "usage: --artifact PATH [--tp 1|2] [--plan-only]\n";
                return 2;
            }
        }
        if (path.empty()) {
            std::cout << "SKIP: supply --artifact (an Inkling 2 artifact)\n";
            return 77;
        }
        const int failures = run(path, tp, plan_only);
        if (failures == 77) { return 77; }
        std::cout << (failures ? "FAIL" : "OK") << " inkling2 loading tp " << tp << '\n';
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
