#include "models/inkling2/load/sharding.h"

#include "artifact/reader.h"

#include <algorithm>
#include <charconv>
#include <stdexcept>
#include <string>

namespace ninfer::models::inkling2::loading {
namespace {

using artifact::ShardAxis;
using artifact::SliceRange;

[[noreturn]] void reject(std::string_view name, const std::string& reason) {
    throw std::invalid_argument(std::string(name) + ": tensor parallelism " + reason);
}

LogicalShard whole(ShardAxis axis, int device = 0) { return {.axis = axis, .device = device}; }

// Splits `total` into `tp` equal contiguous blocks aligned to `units` (heads or rows).
LogicalShard even(std::string_view name, ShardAxis axis, std::uint64_t total, std::uint64_t units,
                  int tp) {
    const auto parts = static_cast<std::uint64_t>(tp);
    if (!units || total % units || units % parts) {
        reject(name, "requires " + std::to_string(units) + " units of " + std::to_string(total) +
                         " to divide by " + std::to_string(tp));
    }
    LogicalShard out{.axis = axis};
    const auto count = total / parts;
    for (int device = 0; device < tp; ++device) {
        out.ranges[static_cast<std::size_t>(device)].push_back(
            {static_cast<std::uint64_t>(device) * count, count});
    }
    return out;
}

std::uint64_t leading(std::string_view name, const artifact::Shape& shape) {
    if (shape.empty()) { reject(name, "requires a shaped parameter"); }
    return shape.front();
}

std::uint64_t trailing(std::string_view name, const artifact::Shape& shape) {
    if (shape.size() != 2) { reject(name, "splits the second axis of a matrix only"); }
    return shape.back();
}

std::uint32_t number(std::string_view name, std::string_view digits) {
    std::uint32_t value     = 0;
    const auto [end, error] = std::from_chars(digits.data(), digits.data() + digits.size(), value);
    if (error != std::errc{} || end != digits.data() + digits.size()) {
        reject(name, "has a malformed index");
    }
    return value;
}

// The rank holding an expert's bank: IDs [r E/tp, (r+1) E/tp) on rank r.
int expert_rank(std::string_view name, std::string_view leaf, const TextConfig& c, int tp) {
    const auto rest  = leaf.substr(std::string_view("moe/experts/").size());
    const auto slash = rest.find('/');
    if (slash == std::string_view::npos) { reject(name, "has a malformed expert name"); }
    const auto expert = number(name, rest.substr(0, slash));
    if (c.num_experts % static_cast<std::uint32_t>(tp) || expert >= c.num_experts) {
        reject(name, "requires the experts to divide into one bank per rank");
    }
    return static_cast<int>(expert / (c.num_experts / static_cast<std::uint32_t>(tp)));
}

LogicalShard layer_shard(std::string_view name, std::string_view leaf, const artifact::Shape& shape,
                         const TextConfig& c, int tp) {
    if (leaf.starts_with("moe/experts/")) {
        return whole(ShardAxis::SingleDevice, expert_rank(name, leaf, c, tp));
    }
    if (leaf == "attention/query" || leaf == "attention/gate" || leaf == "attention/relative" ||
        leaf == "attention/sink" || leaf == "attention/query_scale" ||
        leaf == "attention/relative_scale") {
        return even(name, ShardAxis::Rows, leading(name, shape), c.num_attention_heads, tp);
    }
    if (leaf == "attention/key" || leaf == "attention/value" || leaf == "attention/key_conv" ||
        leaf == "attention/value_conv" || leaf == "attention/key_conv_projection" ||
        leaf == "attention/value_conv_projection") {
        return even(name, ShardAxis::Rows, leading(name, shape), c.num_key_value_heads, tp);
    }
    if (leaf == "attention/output") {
        return even(name, ShardAxis::Columns, trailing(name, shape), c.num_attention_heads, tp);
    }
    if (leaf == "attention/key_conv_static" || leaf == "attention/value_conv_static") {
        return even(name, ShardAxis::Columns, trailing(name, shape), c.num_key_value_heads, tp);
    }
    if (leaf == "attention_queries" || leaf == "input_norm" || leaf == "query_input_norm" ||
        leaf == "key_input_norm" || leaf == "value_input_norm" ||
        leaf == "attention/relative_profile" || leaf == "post_attention_conv" ||
        leaf == "ffn_queries" || leaf == "ffn_norm" || leaf == "moe/router" ||
        leaf == "moe/router_bias" || leaf == "post_ffn_conv" || leaf == "message_queries" ||
        leaf.starts_with("message_gate/")) {
        return whole(ShardAxis::Replicated);
    }
    reject(name, "has no placement for this parameter");
}

LogicalShard draft_shard(std::string_view name, const artifact::Shape& shape,
                         const draft::DraftConfig* d, int tp) {
    if (d == nullptr) { reject(name, "names a drafter parameter without the drafter's config"); }
    constexpr std::string_view layers = "mtp/layers/";
    if (name.starts_with(layers)) {
        const auto slash = name.find('/', layers.size());
        if (slash == std::string_view::npos ||
            number(name, name.substr(layers.size(), slash - layers.size())) >= d->num_layers) {
            reject(name, "names no drafter layer");
        }
        const auto leaf = name.substr(slash + 1);
        if (leaf == "attention/query") {
            return even(name, ShardAxis::Rows, leading(name, shape), d->num_attention_heads, tp);
        }
        if (leaf == "attention/key" || leaf == "attention/value" ||
            leaf == "attention/context_key" || leaf == "attention/context_value") {
            return even(name, ShardAxis::Rows, leading(name, shape), d->num_key_value_heads, tp);
        }
        if (leaf == "attention/output") {
            return even(name, ShardAxis::Columns, trailing(name, shape), d->num_attention_heads,
                        tp);
        }
        if (leaf == "mlp/gate" || leaf == "mlp/up") {
            const auto rows = leading(name, shape);
            return even(name, ShardAxis::Rows, rows, rows, tp);
        }
        if (leaf == "mlp/down") {
            const auto columns = trailing(name, shape);
            return even(name, ShardAxis::Columns, columns, columns, tp);
        }
        if (leaf == "context_norm" || leaf == "input_norm" || leaf == "post_attention_norm" ||
            leaf == "attention/query_norm" || leaf == "attention/key_norm" ||
            leaf == "attention/context_key_norm") {
            return whole(ShardAxis::Replicated);
        }
        reject(name, "has no placement for this parameter");
    }
    if (name == "mtp/markov/folded_output" || name == "mtp/markov/folded_bias") {
        const auto rows = leading(name, shape);
        return even(name, ShardAxis::Rows, rows, rows, tp);
    }
    if (name == "mtp/fuse_weight" || name == "mtp/mask_embedding" || name == "mtp/final_norm" ||
        name == "mtp/markov/context_norm" || name == "mtp/markov/token_embedding" ||
        name == "mtp/markov/recurrence" || name == "mtp/markov/recurrence_bias") {
        return whole(ShardAxis::Replicated);
    }
    reject(name, "has no placement for this parameter");
}

LogicalShard shard_rule(std::string_view name, const artifact::Shape& shape, const TextConfig& c,
                        int tp, const draft::DraftConfig* draft) {
    if (tp == 1) { return {}; }
    if (name.starts_with("mtp/")) { return draft_shard(name, shape, draft, tp); }
    if (name == "text/token_embedding" || name == "text/embedding_norm" ||
        name == "text/final_queries" || name == "text/final_norm") {
        return whole(ShardAxis::Replicated);
    }
    if (name == "text/output_head") {
        const auto rows = leading(name, shape);
        return even(name, ShardAxis::Rows, rows, rows, tp);
    }
    constexpr std::string_view layers = "text/layers/";
    if (name.starts_with(layers)) {
        const auto slash = name.find('/', layers.size());
        if (slash != std::string_view::npos) {
            const auto layer = number(name, name.substr(layers.size(), slash - layers.size()));
            if (layer >= c.num_hidden_layers) { reject(name, "names no layer"); }
            return layer_shard(name, name.substr(slash + 1), shape, c, tp);
        }
    }
    reject(name, "has no placement for this parameter");
}

void validate_tp(const LoadOptions& options) {
    if (options.tp < 1 || options.tp > static_cast<int>(artifact::kMaximumDevices)) {
        throw std::invalid_argument("tensor parallelism must be 1 or 2");
    }
}

std::uint64_t row_elements(const artifact::Shape& shape) {
    return shape.size() > 1 ? weight_element_count(std::span(shape).subspan(1)) : 1;
}

// Sorts, rejects overlaps and joins adjacent ranges.
void normalize(std::vector<SliceRange>& ranges, const std::string& id) {
    std::sort(ranges.begin(), ranges.end(), [](const SliceRange& a, const SliceRange& b) {
        return a.begin != b.begin ? a.begin < b.begin : a.count < b.count;
    });
    std::vector<SliceRange> out;
    for (const auto& range : ranges) {
        if (!out.empty() && out.back().begin + out.back().count > range.begin) {
            throw std::invalid_argument(id + ": tensor-parallel ranges overlap");
        }
        if (!out.empty() && out.back().begin + out.back().count == range.begin) {
            out.back().count += range.count;
        } else {
            out.push_back(range);
        }
    }
    ranges = std::move(out);
}

// Every index of [0, extent) belongs to exactly one device.
void require_partition(const artifact::ShardPlacement& placement, int tp, std::uint64_t extent,
                       const std::string& id) {
    std::vector<SliceRange> all;
    for (int device = 0; device < tp; ++device) {
        const auto& ranges = placement.device_ranges[static_cast<std::size_t>(device)];
        if (ranges.empty()) {
            throw std::invalid_argument(id + ": tensor-parallel placement leaves device " +
                                        std::to_string(device) + " empty");
        }
        all.insert(all.end(), ranges.begin(), ranges.end());
    }
    normalize(all, id);
    if (all.size() != 1 || all.front().begin != 0 || all.front().count != extent) {
        throw std::invalid_argument(id + ": tensor-parallel ranges do not cover the parent");
    }
}

struct ParentState {
    std::optional<artifact::ShardPlacement> placement;
    std::string first; // First logical parameter, for diagnostics.
};

void merge_axis(ParentState& state, const LogicalShard& shard, const std::string& name,
                const std::string& id) {
    if (!state.placement) {
        state.placement = artifact::ShardPlacement{.axis = shard.axis, .device = shard.device};
        state.first     = name;
        return;
    }
    if (state.placement->axis != shard.axis || state.placement->device != shard.device) {
        throw std::invalid_argument(id + ": " + state.first + " and " + name +
                                    " share a parent with different tensor-parallel placements");
    }
}

} // namespace

LogicalShard logical_shard(std::string_view name, const artifact::Shape& shape,
                           const TextConfig& config, const LoadOptions& options,
                           const draft::DraftConfig* draft) {
    validate_tp(options);
    return shard_rule(name, shape, config, options.tp, draft);
}

std::vector<std::optional<artifact::ShardPlacement>>
parent_placements(std::span<const PendingWeight> weights, std::size_t object_count,
                  const GeometryLookup& geometry, const TextConfig& config,
                  const LoadOptions& options, const draft::DraftConfig* draft) {
    validate_tp(options);
    std::vector<ParentState> states(object_count);
    std::vector<std::string> ids(object_count);
    for (const auto& weight : weights) {
        const auto& reference = weight.reference;
        const auto& name      = reference.name;
        const auto shard      = shard_rule(name, reference.shape, config, options.tp, draft);
        const auto row        = row_elements(reference.shape);
        std::uint64_t logical = 0; // Logical element offset of the current part.
        for (std::size_t i = 0; i < reference.binding.parts.size(); ++i) {
            const auto& part   = reference.binding.parts[i];
            const auto& parent = geometry(part.object);
            auto& state        = states.at(part.object.index);
            auto& id           = ids[part.object.index];
            if (id.empty()) {
                id = i < weight.source_objects.size() ? weight.source_objects[i] : name;
            }
            merge_axis(state, shard, name, id);
            if (shard.axis == ShardAxis::Rows) {
                // Logical row ranges intersect this part's element range as whole parent rows.
                const auto parent_row = row_elements(parent.shape);
                const auto length     = part.end - part.begin;
                for (std::size_t device = 0; device < shard.ranges.size(); ++device) {
                    for (const auto& range : shard.ranges[device]) {
                        const auto first = std::max(range.begin * row, logical);
                        const auto last =
                            std::min((range.begin + range.count) * row, logical + length);
                        if (first >= last) { continue; }
                        const auto begin = part.begin + (first - logical);
                        const auto end   = part.begin + (last - logical);
                        if (begin % parent_row || end % parent_row) {
                            throw std::invalid_argument(
                                name + ": a row split does not fall on rows of parent " + id);
                        }
                        state.placement->device_ranges[device].push_back(
                            {begin / parent_row, (end - begin) / parent_row});
                    }
                }
            } else if (shard.axis == ShardAxis::Columns) {
                // A column split narrows every parent row: the parent must have the logical row
                // width and this parameter must bind whole parent rows.
                const auto columns = reference.shape.back();
                if (parent.shape.size() != 2 || parent.shape[1] != columns ||
                    part.begin % columns || part.end % columns) {
                    throw std::invalid_argument(name + ": a column split requires whole rows of " +
                                                id);
                }
                auto& ranges = state.placement->device_ranges;
                if (std::all_of(ranges.begin(), ranges.end(),
                                [](const auto& list) { return list.empty(); })) {
                    ranges = shard.ranges;
                } else if (ranges != shard.ranges) {
                    throw std::invalid_argument(id + ": " + state.first + " and " + name +
                                                " split the columns of one parent differently");
                }
            }
            logical += part.end - part.begin;
        }
    }
    std::vector<std::optional<artifact::ShardPlacement>> out(object_count);
    for (std::size_t object = 0; object < object_count; ++object) {
        auto& placement = states[object].placement;
        if (!placement) { continue; }
        if (artifact::is_sharded(placement->axis)) {
            const auto& parent = geometry(artifact::ObjectHandle{object});
            for (auto& ranges : placement->device_ranges) { normalize(ranges, ids[object]); }
            const auto extent = placement->axis == ShardAxis::Rows
                                    ? parent.elements / row_elements(parent.shape)
                                    : parent.shape.at(1);
            require_partition(*placement, options.tp, extent, ids[object]);
        }
        out[object] = std::move(placement);
    }
    return out;
}

void install_shard_resolver(artifact::Binder& binder, std::span<const PendingWeight> weights,
                            const TextConfig& config, const LoadOptions& options,
                            const draft::DraftConfig* draft) {
    if (options.tp == 1) { return; }
    if (binder.device_count() != options.tp) {
        throw std::logic_error("Binder device count differs from tensor parallelism");
    }
    const auto& reader = binder.reader();
    auto placements    = parent_placements(
        weights, reader.directory().objects.size(),
        [&](artifact::ObjectHandle object) -> const WeightGeometry& {
            return reader.geometry(object);
        },
        config, options, draft);
    binder.set_shard_resolver([placements = std::move(placements),
                               &reader](artifact::ObjectHandle object, const WeightGeometry&) {
        const auto& placement = placements.at(object.index);
        if (!placement) {
            throw artifact::ArtifactError(
                artifact::object_id(reader.directory().object(object)) +
                ": device parent has no logical tensor-parallel placement");
        }
        return *placement;
    });
}

} // namespace ninfer::models::inkling2::loading
