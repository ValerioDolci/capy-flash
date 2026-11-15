// Qwen4Exp (Qwen3.8-Flash-Next) at tp 2: plan and materialize a real-geometry artifact with the
// tensor-parallel placement of load/sharding.h, on two devices.
//
// Plan (host only, also the real 125B artifact with --plan-only): every parameter's parent carries
// the placement an independent name table of this test expects (heads split, indexer / router /
// HC / PLE replicated, shared expert on rank block % 2, routed experts host-mapped, the n-gram
// table file-mapped), every rank shard of a split parameter has the per-rank shape the split Ops
// consume, and the per-rank device bytes are printed by class.
// Materialize (two CUDA devices): every device parent is byte-identical to the slices of the
// artifact parent the host applies itself (reader bytes + PlaneCopy list), the host-mapped expert
// banks are one copy both ranks address, and each rank's views have the narrowed logical shapes.
#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/formats.h"
#include "artifact/reader.h"
#include "artifact/slices.h"
#include "core/device.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/model.h"

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
namespace qwen = ninfer::models::qwen3_5;
using artifact::ShardAxis;
using ninfer::test::artifact_fixture::require;

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

struct Expectation {
    ShardAxis axis = ShardAxis::Replicated;
    int holder     = 0; // SingleDevice
    std::string kind;   // inventory class
};

bool contains(const std::string& name, const char* part) {
    return name.find(part) != std::string::npos;
}

bool ends_with(const std::string& name, const char* suffix) {
    const std::string s(suffix);
    return name.size() >= s.size() && name.compare(name.size() - s.size(), s.size(), s) == 0;
}

std::uint32_t block_index(const std::string& name, std::uint32_t layers) {
    for (const auto& [prefix, base] :
         {std::pair<std::string, std::uint32_t>{"text/layers/", 0},
          std::pair<std::string, std::uint32_t>{"mtp/layers/", layers}}) {
        if (name.rfind(prefix, 0) == 0) {
            return base + static_cast<std::uint32_t>(std::stoul(name.substr(prefix.size())));
        }
    }
    throw std::runtime_error(name + ": not a block parameter");
}

// The placement this test expects for one Qwen4Exp logical parameter (independent of
// load/sharding.cpp: its own table over the parameter names).
std::optional<Expectation> expected_placement(const std::string& name, std::uint32_t layers,
                                              bool embedding_host) {
    if (contains(name, "/moe/experts/")) {
        return Expectation{ShardAxis::HostMapped, 0, "experts"};
    }
    if (ends_with(name, "/ple/table")) { return std::nullopt; } // HostFile
    if (ends_with(name, "/ple/table_global_scale")) {
        return Expectation{ShardAxis::Replicated, 0, "ple"};
    }
    if (name.rfind("vision/", 0) == 0) { return Expectation{ShardAxis::SingleDevice, 0, "vision"}; }
    if (name == "text/token_embedding") {
        return Expectation{embedding_host ? ShardAxis::HostMapped : ShardAxis::Replicated, 0,
                           "embedding"};
    }
    if (name == "text/output_head") { return Expectation{ShardAxis::Rows, 0, "output head"}; }
    const bool hc = contains(name, "attn_hc/") || contains(name, "ffn_hc/") ||
                    contains(name, "hc_head/") || name == "mtp/hidden_norm";
    if (hc) { return Expectation{ShardAxis::Replicated, 0, "hc"}; }
    if (name.rfind("mtp/", 0) == 0 && !contains(name, "/layers/")) {
        return Expectation{ShardAxis::Replicated, 0, "mtp stem"};
    }
    if (contains(name, "/attention/indexer_")) {
        return Expectation{ShardAxis::Replicated, 0, "indexer"};
    }
    if (contains(name, "/attention/")) {
        if (ends_with(name, "_norm")) { return Expectation{ShardAxis::Replicated, 0, "attention"}; }
        if (ends_with(name, "/output")) { return Expectation{ShardAxis::Columns, 0, "attention"}; }
        return Expectation{ShardAxis::Rows, 0, "attention"};
    }
    if (contains(name, "/gdn/")) {
        if (ends_with(name, "/norm")) { return Expectation{ShardAxis::Replicated, 0, "gdn"}; }
        if (ends_with(name, "/output") || ends_with(name, "/convolution")) {
            return Expectation{ShardAxis::Columns, 0, "gdn"};
        }
        return Expectation{ShardAxis::Rows, 0, "gdn"};
    }
    if (ends_with(name, "/moe/router") || ends_with(name, "/moe/shared_score")) {
        return Expectation{ShardAxis::Replicated, 0, "router"};
    }
    if (contains(name, "/moe/shared/")) {
        return Expectation{ShardAxis::SingleDevice, static_cast<int>(block_index(name, layers) % 2),
                           "shared expert"};
    }
    if (contains(name, "/ple/")) { return Expectation{ShardAxis::Replicated, 0, "ple"}; }
    throw std::runtime_error(name + ": the test has no expectation for this parameter");
}

// The per-rank logical shape of a split parameter (Rows / Columns), as the split Ops read it.
std::vector<std::uint64_t> rank_shape(const std::string& name, std::vector<std::uint64_t> shape,
                                      ShardAxis axis) {
    if (axis == ShardAxis::Rows) { shape.front() /= 2; }
    if (axis == ShardAxis::Columns) { shape.back() /= 2; }
    (void)name;
    return shape;
}

struct Totals {
    std::map<std::string, std::array<std::uint64_t, 2>> by_kind;
    std::array<std::uint64_t, 2> total{};
};

int run(const std::filesystem::path& path, const LoadOptions& options, bool plan_only) {
    artifact::Reader reader(path);
    auto plan          = qwen::plan_load(reader, options);
    const auto& config = plan.config().text;
    require(config.qwen4(), "not a Qwen4Exp artifact");
    const auto layers = config.num_hidden_layers;
    const auto& mplan = plan.materialization();
    require(mplan.device_count == 2, "the plan is not for two devices");

    // object -> placement per device
    std::map<std::size_t, std::array<const artifact::DevicePlacement*, 2>> placed;
    for (const auto& item : mplan.device_objects) {
        placed[item.object.index][static_cast<std::size_t>(item.device)] = &item;
    }
    std::set<std::size_t> mapped;
    for (const auto& item : mplan.host_mapped_objects) { mapped.insert(item.object.index); }

    int failures = 0;
    Totals totals;
    std::set<std::size_t> counted;
    std::map<std::string, std::vector<std::uint64_t>> shapes; // parameter -> rank shape
    for (std::size_t index = 0; index < plan.parameter_count(); ++index) {
        const auto& reference = plan.parameter(qwen::WeightId{index});
        const auto& name      = reference.name;
        if (reference.residency != artifact::Residency::Device) { continue; }
        const auto expect = expected_placement(name, layers, options.embedding_host);
        if (!expect) { continue; }
        for (const auto& part : reference.binding.parts) {
            const auto object = part.object.index;
            if (expect->axis == ShardAxis::HostMapped) {
                if (!mapped.contains(object) || placed.contains(object)) {
                    std::cerr << name << ": expected host-mapped\n";
                    ++failures;
                }
                continue;
            }
            const auto found = placed.find(object);
            if (found == placed.end()) {
                std::cerr << name << ": parent has no device placement\n";
                ++failures;
                continue;
            }
            for (int device = 0; device < 2; ++device) {
                const auto* item = found->second[static_cast<std::size_t>(device)];
                const bool held =
                    expect->axis != ShardAxis::SingleDevice || expect->holder == device;
                if ((item != nullptr) != held) {
                    std::cerr << name << ": device " << device << (held ? " lacks" : " holds")
                              << " the parent\n";
                    ++failures;
                    continue;
                }
                if (item == nullptr) { continue; }
                if (item->axis != expect->axis ||
                    (expect->axis == ShardAxis::SingleDevice && item->device != expect->holder)) {
                    std::cerr << name << ": device " << device << " placement "
                              << axis_name(item->axis) << ", expected " << axis_name(expect->axis)
                              << '\n';
                    ++failures;
                }
                if (counted.insert(object * 2 + static_cast<std::size_t>(device)).second) {
                    totals.by_kind[expect->kind][static_cast<std::size_t>(device)] += item->bytes;
                    totals.total[static_cast<std::size_t>(device)] += item->bytes;
                }
            }
        }
        if (artifact::is_sharded(expect->axis)) {
            shapes[name] = rank_shape(name, reference.shape, expect->axis);
        }
    }
    std::printf("%s: tp 2, HC replicated%s, %zu parameters\n", path.filename().c_str(),
                options.embedding_host ? ", embedding host-mapped" : "", plan.parameter_count());
    for (const auto& [kind, bytes] : totals.by_kind) {
        std::printf("  %-14s rank0 %13llu B  rank1 %13llu B\n", kind.c_str(),
                    static_cast<unsigned long long>(bytes[0]),
                    static_cast<unsigned long long>(bytes[1]));
    }
    std::printf("  %-14s rank0 %13llu B  rank1 %13llu B (arena capacity %llu / %llu)\n", "total",
                static_cast<unsigned long long>(totals.total[0]),
                static_cast<unsigned long long>(totals.total[1]),
                static_cast<unsigned long long>(mplan.per_device_capacity_bytes[0]),
                static_cast<unsigned long long>(mplan.per_device_capacity_bytes[1]));
    std::printf("  host-mapped %llu B, file-mapped objects %zu\n",
                static_cast<unsigned long long>(mplan.host_mapped_bytes),
                mplan.file_objects.size());
    if (failures != 0 || plan_only) { return failures; }

    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "SKIP: materialization at tp 2 needs two CUDA devices\n";
        return failures == 0 ? 77 : failures;
    }
    // Keep the placements and each parameter's parent object: materialize consumes the plan.
    const std::vector<artifact::DevicePlacement> placements = mplan.device_objects;
    std::vector<std::optional<std::size_t>> parameter_object(plan.parameter_count());
    for (std::size_t index = 0; index < plan.parameter_count(); ++index) {
        const auto& parts = plan.parameter(qwen::WeightId{index}).binding.parts;
        if (parts.empty()) { continue; }
        bool one = true;
        for (const auto& part : parts) { one = one && part.object == parts.front().object; }
        if (one) { parameter_object[index] = parts.front().object.index; }
    }
    ExecutionContext execution({0, 1});
    auto model = qwen::materialize_model(std::move(plan), execution);

    // Each rank's views: narrowed logical shapes, and the device parent of every placed object.
    std::map<std::pair<std::size_t, int>, const WeightParent*> parents;
    std::size_t views_checked = 0;
    for (int device = 0; device < 2; ++device) {
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
    // Every device parent is byte-identical to the host-applied slice of its artifact parent.
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
    // Host-mapped banks: both ranks address the same copy.
    std::size_t banks = 0;
    for (std::size_t index = 0; index < model->weight_data(0).size(); ++index) {
        const auto& a = model->weight_data(0)[index];
        const auto& b = model->weight_data(1)[index];
        if (!contains(a.name, "/moe/experts/") || a.view.parts.empty()) { continue; }
        if (b.view.parts.empty() ||
            a.view.parts.front().parent->data != b.view.parts.front().parent->data) {
            std::cerr << a.name << ": ranks do not share the host-mapped bank\n";
            ++failures;
        }
        ++banks;
    }
    std::printf("materialized: %zu device parents compared byte for byte (%llu B), %zu split views "
                "checked, %zu expert views on the shared host-mapped banks\n",
                parents_compared, static_cast<unsigned long long>(compared), views_checked, banks);
    const auto& stats = model->storage_stats();
    std::printf("  device capacity rank0 %llu B rank1 %llu B; sharded %llu / %llu, replicated %llu "
                "/ %llu, rank-local %llu / %llu; host-mapped %llu B\n",
                static_cast<unsigned long long>(stats.per_device_capacity_bytes[0]),
                static_cast<unsigned long long>(stats.per_device_capacity_bytes[1]),
                static_cast<unsigned long long>(stats.sharded_bytes[0]),
                static_cast<unsigned long long>(stats.sharded_bytes[1]),
                static_cast<unsigned long long>(stats.replicated_bytes[0]),
                static_cast<unsigned long long>(stats.replicated_bytes[1]),
                static_cast<unsigned long long>(stats.local_bytes[0]),
                static_cast<unsigned long long>(stats.local_bytes[1]),
                static_cast<unsigned long long>(stats.host_mapped_bytes));
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path path;
        LoadOptions options{.speculative = SpeculativeBackend::None, .tp = 2};
        bool plan_only = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--artifact" && i + 1 < argc) {
                path = argv[++i];
            } else if (arg == "--mtp") {
                options.speculative = SpeculativeBackend::Mtp;
            } else if (arg == "--embedding-host") {
                options.embedding_host = true;
            } else if (arg == "--plan-only") {
                plan_only = true;
            } else {
                std::cerr << "usage: --artifact PATH [--mtp] [--embedding-host] [--plan-only]\n";
                return 2;
            }
        }
        if (path.empty()) {
            std::cout << "SKIP: supply --artifact (a Qwen4Exp artifact)\n";
            return 77;
        }
        const int failures = run(path, options, plan_only);
        if (failures == 77) { return 77; }
        std::cout << (failures ? "FAIL" : "OK") << " qwen4 tp2 loading\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
