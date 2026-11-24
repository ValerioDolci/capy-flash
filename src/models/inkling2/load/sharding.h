#pragma once

#include "artifact/binder.h"
#include "models/inkling2/config.h"
#include "models/inkling2/load/bindings.h"

#include <array>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::models::inkling2::loading {

// Tensor-parallel placement of one logical parameter, in the parameter's own coordinates:
// `ranges[d]` are the rows (Rows) or columns (Columns) device d computes with. At tp 1 every
// parameter is whole on device 0. At tp 2, heads split into contiguous per-rank blocks (24/8 ->
// 12/4 for the released model), so a rank's query heads keep their KV heads:
//
//   Rows        attention/{query,gate,relative,sink,query_scale,relative_scale} by query heads;
//               attention/{key,value,key_conv,value_conv,key_conv_projection,value_conv_projection}
//               by KV heads; text/output_head by vocabulary rows (the unpadded rows halved);
//               moe/experts/<e>/gate_up by intermediate channels: rank r's gate rows
//               [r F/2, (r+1) F/2) and the matching up rows [F + r F/2, F + (r+1) F/2).
//   Columns     attention/output over its query-head input; attention/{key,value}_conv_static
//               over its KV-head channels; moe/experts/<e>/down over the same channels.
//               Every rank thus holds every expert at half width. The expert parameters are
//               HostFile: their artifact banks are not sliced by the binder (a stacked NVFP4 plane
//               is not, and half an expert's gate is not whole 128-row scale tiles); the model
//               cuts them (load/expert_banks.h) and parent_placements skips them.
//   Replicated  everything else: embedding and norms, block-residual and message queries, message
//               gates, router and its bias, post-attention and post-FFN convolutions,
//               attention/relative_profile, final norm and queries.
//   HostMapped  text/token_embedding with options.embedding_host, at tp 1 too: one copy in mapped
//               page-locked host memory that every rank's gather (the prefill, the decode and the
//               drafter's anchor) reads in place over PCIe.
//
// The drafter (mtp/, given its config) splits like a Text layer: Rows attention/query by query
// heads, attention/{key,value,context_key,context_value} by KV heads, mlp/{gate,up} in halves,
// markov/{folded_output,folded_bias} by the output head's vocabulary rows; Columns
// attention/output by query heads, mlp/down in halves; every norm, fuse_weight, mask_embedding,
// markov/{token_embedding,recurrence,recurrence_bias} Replicated.
struct LogicalShard {
    artifact::ShardAxis axis = artifact::ShardAxis::Replicated;
    int device               = 0; // Holder under SingleDevice.
    std::array<std::vector<artifact::SliceRange>, artifact::kMaximumDevices> ranges;
};

// Throws std::invalid_argument unless options.tp is 1 or 2, for a parameter without a rule, and
// for a head count or expert count that does not divide by options.tp.
[[nodiscard]] LogicalShard logical_shard(std::string_view name, const artifact::Shape& shape,
                                         const TextConfig& config, const LoadOptions& options,
                                         const draft::DraftConfig* draft = nullptr);

using GeometryLookup = std::function<const WeightGeometry&(artifact::ObjectHandle)>;

// Combines the logical placements of every parameter bound to one parent. Row ranges map
// through the Binding parts to whole parent rows and must partition the parent; column ranges
// must agree across parameters and partition its columns; complete-parent kinds must agree.
// Indexed by object; parents that no parameter demands stay empty.
[[nodiscard]] std::vector<std::optional<artifact::ShardPlacement>>
parent_placements(std::span<const PendingWeight> weights, std::size_t object_count,
                  const GeometryLookup& geometry, const TextConfig& config,
                  const LoadOptions& options, const draft::DraftConfig* draft = nullptr);

// Installs the combined placements on a Binder planned for options.tp devices; no-op at tp 1
// unless options.embedding_host, whose HostMapped parent needs the resolver there too.
void install_shard_resolver(artifact::Binder& binder, std::span<const PendingWeight> weights,
                            const TextConfig& config, const LoadOptions& options,
                            const draft::DraftConfig* draft = nullptr);

} // namespace ninfer::models::inkling2::loading
