// Inkling 2 loading at tp 1 and tp 2: plan and materialize an artifact (the synthetic reference
// checkpoint's, or the released model's) with the placement of load/sharding.h.
//
// Plan (host only): the Qwen path rejects the artifact; at tp 2 every parameter's parent carries
// the placement of expected_placement.h (heads split, output head split by vocabulary rows, the
// rest replicated) and the per-rank device bytes are printed by class; the routed experts are
// HostFile, with no device parent.
// Materialize (tp CUDA devices): the device memory each rank's weights take, every device parent
// byte-identical to the slice of its artifact parent the host applies itself (reader bytes +
// PlaneCopy list), each rank's views with the narrowed logical shapes, each rank's expert banks
// (load/expert_banks.h) element by element against the artifact's banks (every layer of a small
// artifact; layers 0 and 20, experts 0, E/2-1, E/2, E-1 of a large one: codes, scales through
// weight_scale_offset, divisors), and each rank's Parameters: its expert banks (both halves of F
// at tp 1, half r at tp 2), its projection shards.
// --embedding-host: text/token_embedding is the plan's one HostMapped parent, on no device; once
// materialized every rank addresses the same page-locked host copy, byte-identical to the
// artifact's parent.

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

// One expert matrix in its artifact bank: the bank object and the expert's first bank row.
struct ExpertSource {
    artifact::ObjectHandle object;
    std::uint64_t row = 0;
};

struct LayerExperts {
    std::vector<ExpertSource> gate_up, down;
};

// The code of element (row, column) of a parent: an E2M1 nibble, an E4M3 byte or a BF16 word.
std::uint32_t code_of(const WeightGeometry& g, const std::byte* bytes, std::uint64_t row,
                      std::uint64_t column) {
    if (g.format == QType::NVFP4) {
        const auto b =
            std::to_integer<std::uint32_t>(bytes[row * g.code_bytes_per_row + column / 2]);
        return column % 2 ? b >> 4 : b & 15U;
    }
    if (g.format == QType::MXFP8_E4M3FN_G32_UE8M0) {
        return std::to_integer<std::uint32_t>(bytes[row * g.code_bytes_per_row + column]);
    }
    std::uint16_t word = 0;
    std::memcpy(&word, bytes + (row * g.shape[1] + column) * 2, 2);
    return word;
}

// The scale byte of element (row, column) (none for BF16) and its NVFP4 divisor word.
std::uint32_t scale_of(const WeightGeometry& g, const std::byte* bytes, std::uint64_t row,
                       std::uint64_t column) {
    if (g.format == QType::BF16) { return 0; }
    return std::to_integer<std::uint32_t>(
        bytes[weight_scale_offset(g, row, column / g.group_size)]);
}

std::uint32_t divisor_of(const WeightGeometry& g, const std::byte* bytes, std::uint64_t row) {
    if (g.format != QType::NVFP4) { return 0; }
    std::uint32_t word = 0;
    std::memcpy(&word, bytes + g.divisor_offset + row / g.rows_per_divisor() * 4, 4);
    return word;
}

// Every element of the checked experts of each rank's banks against the artifact's banks: half h
// of F (rank h at tp 2, bank h at tp 1) holds, per expert, gate rows [h F/2, (h+1) F/2), the
// matching up rows, and those down columns.
int check_expert_banks(const ink::Model& model, const artifact::Reader& reader,
                       const std::vector<LayerExperts>& sources, int tp) {
    const auto& c         = model.config().text;
    const std::uint64_t E = c.num_experts, H = c.hidden_size, F = c.expert_intermediate, f = F / 2;
    const bool large = E * H * F > (1ULL << 26);
    std::vector<std::uint64_t> experts;
    for (std::uint64_t e = 0; e < E; ++e) {
        if (!large || e == 0 || e == E / 2 - 1 || e == E / 2 || e == E - 1) {
            experts.push_back(e);
        }
    }
    int failures          = 0;
    std::uint64_t checked = 0;
    for (std::uint32_t layer = 0; layer < c.num_hidden_layers; ++layer) {
        if (large && layer != 0 && layer != 20) { continue; }
        std::map<std::size_t, std::vector<std::byte>> objects;
        const auto bytes_of = [&](artifact::ObjectHandle object) -> const std::byte* {
            auto& out = objects[object.index];
            if (out.empty()) { out = reader.read_object(object); }
            return out.data();
        };
        for (int device = 0; device < tp; ++device) {
            const auto banks = model.expert_banks(layer, device);
            for (std::size_t i = 0; i < banks.size(); ++i) {
                const auto half =
                    static_cast<std::uint64_t>(tp == 2 ? device : static_cast<int>(i));
                const auto& bank = *banks[i];
                const auto& G    = bank.gate_up_parent.geometry;
                const auto& D    = bank.down_parent.geometry;
                std::vector<std::byte> gate_up(G.bytes), down(D.bytes);
                if (cudaSetDevice(device) != cudaSuccess ||
                    cudaMemcpy(gate_up.data(), bank.gate_up_parent.data, G.bytes,
                               cudaMemcpyDeviceToHost) != cudaSuccess ||
                    cudaMemcpy(down.data(), bank.down_parent.data, D.bytes,
                               cudaMemcpyDeviceToHost) != cudaSuccess) {
                    throw std::runtime_error("download of an expert bank failed");
                }
                bool same = true;
                for (const auto e : experts) {
                    const auto& gs  = sources[layer].gate_up[e];
                    const auto& S   = reader.geometry(gs.object);
                    const auto* src = bytes_of(gs.object);
                    for (std::uint64_t r = 0; r < F; ++r) {
                        const auto n = e * F + r;
                        const auto m = gs.row + (r < f ? half * f + r : F + half * f + r - f);
                        same = same && divisor_of(G, gate_up.data(), n) == divisor_of(S, src, m);
                        for (std::uint64_t col = 0; col < H; ++col) {
                            same = same &&
                                   code_of(G, gate_up.data(), n, col) == code_of(S, src, m, col) &&
                                   scale_of(G, gate_up.data(), n, col) == scale_of(S, src, m, col);
                        }
                    }
                    const auto& ds   = sources[layer].down[e];
                    const auto& DS   = reader.geometry(ds.object);
                    const auto* dsrc = bytes_of(ds.object);
                    for (std::uint64_t r = 0; r < H; ++r) {
                        const auto n = e * H + r, m = ds.row + r;
                        same = same && divisor_of(D, down.data(), n) == divisor_of(DS, dsrc, m);
                        for (std::uint64_t col = 0; col < f; ++col) {
                            same = same &&
                                   code_of(D, down.data(), n, col) ==
                                       code_of(DS, dsrc, m, half * f + col) &&
                                   scale_of(D, down.data(), n, col) ==
                                       scale_of(DS, dsrc, m, half * f + col);
                        }
                    }
                    checked += F * H + H * f;
                }
                if (!same) {
                    std::cerr << "layer " << layer << " rank " << device << " bank " << i
                              << ": expert bank differs from the artifact's banks\n";
                    ++failures;
                }
            }
        }
    }
    std::printf("expert banks: %llu elements compared with the artifact's banks (%zu experts a "
                "layer)\n",
                static_cast<unsigned long long>(checked), experts.size());
    return failures;
}

int check_parameters(const ink::Model& model, int device, int tp) {
    const ink::execution::Parameters parameters(model, device);
    const auto& c      = model.config().text;
    const auto& t      = parameters.text;
    const auto experts = static_cast<std::int32_t>(c.num_experts);
    int failures       = 0;
    for (std::size_t layer = 0; layer < t.layers.size(); ++layer) {
        const auto& p     = t.layers[layer];
        const auto& banks = p.moe.banks;
        bool ok           = banks.size() == (tp == 1 ? 2U : 1U);
        for (const auto& bank : banks) {
            ok = ok && bank.first_expert == 0 && bank.experts == experts &&
                 bank.gate_up.shape ==
                     std::vector<std::uint64_t>{std::uint64_t(experts) * c.expert_intermediate,
                                                c.hidden_size} &&
                 bank.down.shape ==
                     std::vector<std::uint64_t>{std::uint64_t(experts) * c.hidden_size,
                                                c.expert_intermediate / 2};
        }
        if (!ok) {
            std::cerr << "layer " << layer << " rank " << device << ": unexpected expert banks\n";
            ++failures;
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

int run(const std::filesystem::path& path, int tp, bool plan_only, bool embedding_host) {
    artifact::Reader reader(path);
    try {
        (void)qwen3_5::plan_load(reader, {});
        std::cerr << "the Qwen path accepted an Inkling 2 artifact\n";
        return 1;
    } catch (const artifact::ArtifactError& error) {
        std::printf("Qwen path rejects it: %s\n", error.what());
    }
    auto plan = ink::plan_load(reader, LoadOptions{.tp = tp, .embedding_host = embedding_host});
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
    std::size_t mapped = 0;
    std::set<std::size_t> host_objects; // the HostMapped parents (embedding_host)
    for (std::size_t index = 0; index < plan.parameter_count(); ++index) {
        const auto& reference = plan.parameter(ink::WeightId{index});
        const auto& name      = reference.name;
        auto expect           = expected_placement(name, config.num_experts);
        if (reference.residency == artifact::Residency::HostFile) {
            if (expect.kind != "experts") {
                std::cerr << name << ": only the routed experts are file-mapped\n";
                ++failures;
            }
            ++mapped;
            continue;
        }
        if (tp == 1) { expect = {ShardAxis::Replicated, 0, expect.kind}; }
        if (embedding_host && name == "text/token_embedding") {
            for (const auto& part : reference.binding.parts) {
                host_objects.insert(part.object.index);
                if (placed.contains(part.object.index)) {
                    std::cerr << name << ": a device holds the host-mapped parent\n";
                    ++failures;
                }
            }
            continue;
        }
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
    std::printf("%s: tp %d, %zu parameters (%zu file-mapped experts), %zu device parents\n",
                path.filename().c_str(), tp, plan.parameter_count(), mapped,
                mplan.device_objects.size());
    if (mapped != 2ULL * config.num_experts * config.num_hidden_layers) {
        std::cerr << "the routed experts are not all file-mapped\n";
        ++failures;
    }
    std::set<std::size_t> planned_host;
    for (const auto& item : mplan.host_mapped_objects) { planned_host.insert(item.object.index); }
    if (planned_host != host_objects || host_objects.size() != (embedding_host ? 1U : 0U)) {
        std::cerr << "the host-mapped parents are not the token embedding's\n";
        ++failures;
    }
    if (embedding_host) {
        std::printf("  %-22s %llu B in mapped page-locked host memory\n", "embedding",
                    static_cast<unsigned long long>(mplan.host_mapped_bytes));
    }
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
    std::vector<LayerExperts> sources;
    for (const auto& layer : plan.weights().text.layers) {
        LayerExperts out;
        for (const auto& expert : layer.moe.experts) {
            for (const auto& [id, list] :
                 {std::pair{expert.gate_up, &out.gate_up}, std::pair{expert.down, &out.down}}) {
                const auto& part = plan.parameter(id).binding.parts.at(0);
                list->push_back({part.object, part.begin / reader.geometry(part.object).shape[1]});
            }
        }
        sources.push_back(std::move(out));
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
    for (const auto object : host_objects) {
        const auto want        = reader.read_object(artifact::ObjectHandle{object});
        const std::byte* first = nullptr;
        for (int device = 0; device < tp; ++device) {
            const auto found = parents.find({object, device});
            if (found == parents.end()) {
                std::cerr << "rank " << device << ": no view of the host-mapped embedding\n";
                ++failures;
                continue;
            }
            const WeightParent& parent = *found->second;
            cudaPointerAttributes attributes{};
            std::vector<std::byte> got(want.size());
            if (cudaSetDevice(device) != cudaSuccess ||
                cudaPointerGetAttributes(&attributes, parent.data) != cudaSuccess ||
                cudaMemcpy(got.data(), parent.data, got.size(), cudaMemcpyDefault) != cudaSuccess) {
                throw std::runtime_error("reading the host-mapped embedding failed");
            }
            if (first == nullptr) { first = parent.data; }
            if (attributes.type != cudaMemoryTypeHost || parent.data != first || got != want) {
                std::cerr << "rank " << device
                          << ": the embedding is not the one host copy of the artifact's bytes\n";
                ++failures;
            }
        }
        std::printf("host-mapped embedding: one page-locked copy for %d rank(s), %zu B equal to "
                    "the artifact\n",
                    tp, want.size());
    }
    failures += check_expert_banks(*model, reader, sources, tp);
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
        int tp              = 2;
        bool plan_only      = false;
        bool embedding_host = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--artifact" && i + 1 < argc) {
                path = argv[++i];
            } else if (arg == "--tp" && i + 1 < argc) {
                tp = std::stoi(argv[++i]);
            } else if (arg == "--plan-only") {
                plan_only = true;
            } else if (arg == "--embedding-host") {
                embedding_host = true;
            } else {
                std::cerr << "usage: --artifact PATH [--tp 1|2] [--plan-only] [--embedding-host]\n";
                return 2;
            }
        }
        if (path.empty()) {
            std::cout << "SKIP: supply --artifact (an Inkling 2 artifact)\n";
            return 77;
        }
        const int failures = run(path, tp, plan_only, embedding_host);
        if (failures == 77) { return 77; }
        std::cout << (failures ? "FAIL" : "OK") << " inkling2 loading tp " << tp << '\n';
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
