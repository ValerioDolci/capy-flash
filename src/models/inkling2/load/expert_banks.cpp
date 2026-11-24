#include "models/inkling2/load/expert_banks.h"

#include <cuda_runtime.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::models::inkling2::loading {
namespace {

constexpr int kHalves = 2;

// The bytes of one mapped artifact bank: in place when one segment maps the whole object,
// otherwise a host copy of its segments.
struct MappedBank {
    const std::byte* data = nullptr;
    std::uint64_t bytes   = 0;
    std::vector<std::byte> copy;

    // Once the bank is cut, its file pages leave the process (they stay in the page cache): the
    // Model keeps the mapping for its lifetime but never reads the experts through it again.
    ~MappedBank() {
        if (data == nullptr || !copy.empty()) { return; }
        const auto page  = static_cast<std::uintptr_t>(::sysconf(_SC_PAGESIZE));
        const auto begin = reinterpret_cast<std::uintptr_t>(data) / page * page;
        const auto end   = reinterpret_cast<std::uintptr_t>(data) + bytes;
        (void)::madvise(reinterpret_cast<void*>(begin), end - begin, MADV_DONTNEED);
    }
};

// Reads a layer's mapped banks into the page cache in the background, from several threads that
// each touch a run of pages in order. The artifact maps its files for scattered lookups
// (MADV_RANDOM: no readahead), and the cut reads each expert's half rows and half columns, so a
// file not yet in the page cache would arrive one page at a time; the banks' ranges are marked
// sequential first, so each thread's run is read ahead in large requests.
class ReadAhead {
public:
    ReadAhead() = default;
    explicit ReadAhead(const std::map<std::size_t, MappedBank>& banks) {
        constexpr int kThreads         = 8;
        constexpr std::uint64_t kChunk = std::uint64_t{64} << 20;
        const auto page = static_cast<std::uint64_t>(::sysconf(_SC_PAGESIZE));
        std::vector<std::pair<const std::byte*, std::uint64_t>> chunks;
        for (const auto& [index, bank] : banks) {
            if (bank.data == nullptr || !bank.copy.empty()) { continue; }
            const auto begin = reinterpret_cast<std::uintptr_t>(bank.data) / page * page;
            const auto end   = reinterpret_cast<std::uintptr_t>(bank.data) + bank.bytes;
            (void)::madvise(reinterpret_cast<void*>(begin), end - begin, MADV_SEQUENTIAL);
            for (std::uint64_t at = 0; at < bank.bytes; at += kChunk) {
                chunks.emplace_back(bank.data + at, std::min(kChunk, bank.bytes - at));
            }
        }
        for (int t = 0; t < kThreads && static_cast<std::size_t>(t) < chunks.size(); ++t) {
            threads_.emplace_back([chunks, page, t] {
                unsigned sum = 0;
                for (std::size_t c = static_cast<std::size_t>(t); c < chunks.size(); c += kThreads) {
                    const auto* p = reinterpret_cast<const volatile unsigned char*>(chunks[c].first);
                    for (std::uint64_t at = 0; at < chunks[c].second; at += page) { sum += p[at]; }
                }
                (void)sum;
            });
        }
    }
    ReadAhead(const ReadAhead&)            = delete;
    ReadAhead& operator=(const ReadAhead&) = delete;
    ~ReadAhead() {
        for (auto& thread : threads_) { thread.join(); }
    }

private:
    std::vector<std::thread> threads_;
};

const std::byte* mapped_bank(std::map<std::size_t, MappedBank>& banks,
                             artifact::ObjectHandle object,
                             const artifact::MaterializedArtifact& materialized) {
    auto& bank = banks[object.index];
    if (bank.data != nullptr) { return bank.data; }
    const auto segments = materialized.file_segments(object);
    const auto bytes    = materialized.file_geometry(object).bytes;
    if (segments.size() == 1 && segments[0].object_offset == 0 && segments[0].bytes >= bytes) {
        bank.data  = segments[0].data;
        bank.bytes = bytes;
        return bank.data;
    }
    bank.copy.resize(bytes);
    for (const auto& segment : segments) {
        if (segment.object_offset >= bytes) { continue; }
        std::memcpy(bank.copy.data() + segment.object_offset, segment.data,
                    std::min(segment.bytes, bytes - segment.object_offset));
    }
    bank.data = bank.copy.data();
    return bank.data;
}

// One expert matrix in its artifact bank: the bank's bytes and the expert's first bank row.
struct Source {
    const std::byte* data          = nullptr;
    const WeightGeometry* geometry = nullptr;
    std::uint64_t row              = 0;
};

// Without `banks` only the geometry and row (no bytes).
Source source(const PendingWeight& weight, const artifact::MaterializedArtifact& materialized,
              std::map<std::size_t, MappedBank>* banks) {
    const auto& reference = weight.reference;
    const auto& parts     = reference.binding.parts;
    if (reference.residency != artifact::Residency::HostFile || parts.size() != 1) {
        throw std::invalid_argument(reference.name +
                                    ": an expert matrix is one file-mapped bank region");
    }
    Source out;
    out.geometry = &materialized.file_geometry(parts[0].object);
    if (out.geometry->shape.size() != 2 || out.geometry->shape[1] != reference.shape.at(1) ||
        parts[0].begin % out.geometry->shape[1]) {
        throw std::invalid_argument(reference.name + ": an expert is whole rows of its bank");
    }
    out.data = banks != nullptr ? mapped_bank(*banks, parts[0].object, materialized) : nullptr;
    out.row  = parts[0].begin / out.geometry->shape[1];
    return out;
}

// The parent offset of the scale word (four E4M3 scales: groups 4 kt .. 4 kt + 3) of `row` in
// 64-column tile `kt` of a block_scale_k16_m128x4 plane.
std::uint64_t scale_word(const WeightGeometry& g, std::uint64_t row, std::uint64_t kt) {
    const auto inner = row % 128;
    return g.scale_offset + (row / 128 * (g.shape[1] / 64) + kt) * 512 + inner % 32 * 16 +
           inner / 32 * 4;
}

struct Plane {
    const WeightGeometry& g;
    std::vector<std::byte>& bytes;
};

// Copies `rows` consecutive rows from source row `from` to plane row `to`, keeping columns
// [first, first + width) of the source (all of the destination's columns).
void copy_rows(const Source& s, std::uint64_t from, Plane out, std::uint64_t to, std::uint64_t rows,
               std::uint64_t first) {
    const auto& src  = *s.geometry;
    const auto& dst  = out.g;
    const auto width = dst.shape[1];
    std::byte* bytes = out.bytes.data();
    if (src.format == QType::BF16 || src.format == QType::MXFP8_E4M3FN_G32_UE8M0) {
        const std::uint64_t word = src.format == QType::BF16 ? 2 : 1;
        for (std::uint64_t r = 0; r < rows; ++r) {
            std::memcpy(bytes + (to + r) * dst.code_bytes_per_row,
                        s.data + (from + r) * src.code_bytes_per_row + first * word, width * word);
        }
        if (src.format == QType::MXFP8_E4M3FN_G32_UE8M0) {
            for (std::uint64_t r = 0; r < rows; ++r) {
                std::memcpy(bytes + dst.scale_offset + (to + r) * dst.scale_bytes_per_row,
                            s.data + src.scale_offset + (from + r) * src.scale_bytes_per_row +
                                first / 32,
                            width / 32);
            }
        }
        return;
    }
    // NVFP4: two codes a byte, one scale word per row and 64-column tile.
    for (std::uint64_t r = 0; r < rows; ++r) {
        std::memcpy(bytes + (to + r) * dst.code_bytes_per_row,
                    s.data + (from + r) * src.code_bytes_per_row + first / 2, width / 2);
        for (std::uint64_t kt = 0; kt < width / 64; ++kt) {
            std::memcpy(bytes + scale_word(dst, to + r, kt),
                        s.data + scale_word(src, from + r, first / 64 + kt), 4);
        }
    }
}

// The NVFP4 divisor word of source row `row` into word `index` of the plane.
void copy_divisor(const Source& s, std::uint64_t row, Plane out, std::uint64_t index) {
    const auto& src = *s.geometry;
    std::memcpy(out.bytes.data() + out.g.divisor_offset + index * 4,
                s.data + src.divisor_offset + row / src.rows_per_divisor() * 4, 4);
}

// The geometry of one half of a layer's experts, checked against the artifact's banks.
std::unique_ptr<ExpertBank> plan_bank(const std::vector<Source>& gate_up,
                                      const std::vector<Source>& down, const TextConfig& c) {
    const std::uint64_t E = gate_up.size(), H = c.hidden_size, F = c.expert_intermediate,
                        f = F / kHalves;
    const auto& first     = *gate_up.front().geometry;
    const bool nvfp4      = first.format == QType::NVFP4;
    const bool supported  = (nvfp4 && first.layout == QuantLayout::BlockScaleK16M128x4) ||
                           (first.format == QType::MXFP8_E4M3FN_G32_UE8M0 &&
                            first.layout == QuantLayout::BlockScaleK32) ||
                           (first.format == QType::BF16 && first.layout == QuantLayout::Contiguous);
    if (!supported || F % (kHalves * 64)) {
        throw std::invalid_argument("expert banks are NVFP4, MXFP8 or BF16 with F/2 a multiple "
                                    "of 64");
    }
    for (std::uint64_t e = 0; e < E; ++e) {
        for (const auto* s : {&gate_up[e], &down[e]}) {
            if (s->geometry->format != first.format || s->geometry->layout != first.layout) {
                throw std::invalid_argument("a layer's experts share one format");
            }
        }
        if (nvfp4 && (gate_up[e].geometry->rows_per_divisor() != 2 * F ||
                      down[e].geometry->rows_per_divisor() != H)) {
            throw std::invalid_argument("NVFP4 expert banks keep one divisor per expert");
        }
    }
    const auto divisors = nvfp4 ? E : 1;
    auto bank           = std::make_unique<ExpertBank>();
    bank->experts       = static_cast<std::int32_t>(E);
    const std::array<std::uint64_t, 2> gu_shape{E * F, H}, d_shape{E * H, f};
    bank->gate_up_parent.geometry = weight_geometry(first.format, first.layout, gu_shape, divisors);
    bank->down_parent.geometry    = weight_geometry(first.format, first.layout, d_shape, divisors);
    return bank;
}

// Fills the bank's parents of half `half` (at their device addresses) from the artifact's banks.
void fill_bank(ExpertBank& bank, const std::vector<Source>& gate_up,
               const std::vector<Source>& down, const TextConfig& c, int half,
               std::vector<std::byte>& bytes) {
    const std::uint64_t E = gate_up.size(), H = c.hidden_size, F = c.expert_intermediate,
                        f = F / kHalves;
    const bool nvfp4      = bank.gate_up_parent.geometry.format == QType::NVFP4;
    for (auto* parent : {&bank.gate_up_parent, &bank.down_parent}) {
        bytes.assign(parent->geometry.bytes, std::byte{0});
        Plane plane{parent->geometry, bytes};
        for (std::uint64_t e = 0; e < E; ++e) {
            if (parent == &bank.gate_up_parent) {
                const auto& s = gate_up[e];
                copy_rows(s, s.row + half * f, plane, e * F, f, 0);         // gate
                copy_rows(s, s.row + F + half * f, plane, e * F + f, f, 0); // up
                if (nvfp4) { copy_divisor(s, s.row, plane, e); }
            } else {
                const auto& s = down[e];
                copy_rows(s, s.row, plane, e * H, H, half * f);
                if (nvfp4) { copy_divisor(s, s.row, plane, e); }
            }
        }
        const auto status = cudaMemcpy(const_cast<std::byte*>(parent->data), bytes.data(),
                                       bytes.size(), cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("expert bank upload failed: ") +
                                     cudaGetErrorString(status));
        }
    }
    for (auto* view : {&bank.gate_up, &bank.down}) {
        const auto& parent = view == &bank.gate_up ? bank.gate_up_parent : bank.down_parent;
        *view = WeightView{parent.geometry.shape, {{&parent, 0, parent.geometry.elements}}};
    }
}

constexpr std::uint64_t align256(std::uint64_t value) { return (value + 255) / 256 * 256; }

} // namespace

ExpertBanks cut_expert_banks(std::span<const PendingWeight> pending, const TextWeights& weights,
                             const TextConfig& config,
                             const artifact::MaterializedArtifact& materialized,
                             std::span<const DeviceContext* const> devices) {
    if (devices.empty() || devices.size() > artifact::kMaximumDevices) {
        throw std::invalid_argument("expert banks are cut for one or two devices");
    }
    const auto device_of = [&](int half) {
        return devices.size() == 1 ? std::size_t{0} : static_cast<std::size_t>(half);
    };
    // The sources of one layer; with `banks`, their bytes too.
    const auto layer_sources = [&](std::size_t layer, std::map<std::size_t, MappedBank>* banks) {
        std::array<std::vector<Source>, 2> out;
        for (const auto& expert : weights.layers[layer].moe.experts) {
            out[0].push_back(source(pending[expert.gate_up.index], materialized, banks));
            out[1].push_back(source(pending[expert.down.index], materialized, banks));
        }
        return out;
    };
    ExpertBanks out;
    std::array<std::uint64_t, artifact::kMaximumDevices> bytes{};
    std::vector<std::array<std::uint64_t, kHalves * 2>> offsets(weights.layers.size());
    for (std::size_t layer = 0; layer < weights.layers.size(); ++layer) {
        const auto s = layer_sources(layer, nullptr);
        for (int half = 0; half < kHalves; ++half) {
            auto bank             = plan_bank(s[0], s[1], config);
            const auto d          = device_of(half);
            const auto index      = static_cast<std::size_t>(half) * 2;
            offsets[layer][index] = bytes[d];
            bytes[d] += align256(bank->gate_up_parent.geometry.bytes);
            offsets[layer][index + 1] = bytes[d];
            bytes[d] += align256(bank->down_parent.geometry.bytes);
            if (out.layers[d].empty()) { out.layers[d].resize(weights.layers.size()); }
            out.layers[d][layer].push_back(std::move(bank));
        }
    }
    for (std::size_t d = 0; d < devices.size(); ++d) {
        devices[d]->bind_to_current_thread();
        out.storage[d] = DeviceBuffer(bytes[d]);
    }
    // Each layer's banks are read ahead while the previous layer is cut.
    std::vector<std::byte> staging;
    std::map<std::size_t, MappedBank> banks;
    auto s = layer_sources(0, &banks);
    { ReadAhead first(banks); }
    for (std::size_t layer = 0; layer < weights.layers.size(); ++layer) {
        std::map<std::size_t, MappedBank> next_banks;
        std::array<std::vector<Source>, 2> next;
        std::optional<ReadAhead> ahead;
        if (layer + 1 < weights.layers.size()) {
            next = layer_sources(layer + 1, &next_banks);
            ahead.emplace(next_banks);
        }
        for (int half = 0; half < kHalves; ++half) {
            const auto d     = device_of(half);
            const auto index = static_cast<std::size_t>(half) * 2;
            auto& bank =
                *out.layers[d][layer][devices.size() == 1 ? static_cast<std::size_t>(half) : 0];
            auto* base               = static_cast<std::byte*>(out.storage[d].p);
            bank.gate_up_parent.data = base + offsets[layer][index];
            bank.down_parent.data    = base + offsets[layer][index + 1];
            devices[d]->bind_to_current_thread();
            fill_bank(bank, s[0], s[1], config, half, staging);
        }
        ahead.reset();
        banks = std::move(next_banks);
        s     = std::move(next);
    }
    devices[0]->bind_to_current_thread();
    return out;
}

} // namespace ninfer::models::inkling2::loading
