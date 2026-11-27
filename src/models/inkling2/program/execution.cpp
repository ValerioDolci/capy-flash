#include "models/inkling2/program/execution.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/device_scope.h"
#include "core/layout.h"
#include "models/inkling2/draft/dflash2.h"
#include "models/inkling2/draft/draft.h"
#include "models/inkling2/draft/paired.h"
#include "models/inkling2/draft/parameters.h"
#include "models/inkling2/execution/forward.h"
#include "models/inkling2/execution/parameters.h"
#include "models/inkling2/execution/prefetch_lane.h"
#include "models/inkling2/execution/tp.h"
#include "models/inkling2/program/program_impl.h"
#include "models/inkling2/program/speculative/round.h"
#include "models/inkling2/state/visit_state.h"
#include "ninfer/ops/allreduce.h"
#include "core/decode_graph.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/peer_mailbox.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/scatter.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ninfer::models::inkling2::detail {
namespace {

namespace exec = execution;

// Logit columns of one scoring call: the logits come back to the Host in FP32 after the soft cap,
// so a column costs 4 bytes per vocabulary row of Host traffic and of device buffer.
constexpr std::int32_t kScoreColumns = 64;

// The mailbox slot of the captured decode step's all-reduces (one token's partial sums).
constexpr std::size_t kMailboxSlotBytes = std::size_t{16} << 10;

// Device memory planned per rank for the startup warmup and the decode graph: the kernels' code
// loaded on first launch, the instantiated graph and its upload.
constexpr std::size_t kGraphAllowanceBytes = std::size_t{16} << 20;

// Device memory planned per rank for the speculative round beside the drafter's own: its logits,
// convolution record and buffers, and its graph.
constexpr std::size_t kRoundAllowanceBytes = std::size_t{32} << 20;

[[nodiscard]] std::size_t at(int rank) { return static_cast<std::size_t>(rank); }

// Counter-based uniform draw in [0, 1) keyed by (seed, position): no state, the same draw for the
// same request position.
[[nodiscard]] double uniform(std::uint64_t seed, std::uint32_t position) noexcept {
    std::uint64_t z = seed ^ (0x9e3779b97f4a7c15ULL * (std::uint64_t(position) + 1U));
    z               = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z               = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    z ^= z >> 31U;
    return double(z >> 11U) * (1.0 / 9007199254740992.0);
}

// The sampling contract of ops::sample (ninfer/ops/sampling.h) over Host FP32 logits: penalties,
// then the top_k best (lower id first on ties), weights exp((v - max) / temperature), min_p, top_p,
// one draw.
[[nodiscard]] TokenId sample_host(std::span<const float> logits,
                                  const ResolvedSamplingParameters& sampling,
                                  std::span<const TokenId> generated, std::uint32_t position) {
    std::unordered_map<TokenId, std::uint32_t> counts;
    if (sampling.presence_penalty != 0.0F || sampling.frequency_penalty != 0.0F) {
        for (const TokenId id : generated) { ++counts[id]; }
    }
    const auto adjusted = [&](std::size_t v) {
        float value = logits[v];
        if (!counts.empty()) {
            if (const auto found = counts.find(static_cast<TokenId>(v)); found != counts.end()) {
                value -=
                    sampling.presence_penalty + sampling.frequency_penalty * float(found->second);
            }
        }
        return value;
    };
    const std::size_t k = std::size_t(std::clamp(sampling.top_k <= 0 ? 20 : sampling.top_k, 1, 20));
    std::vector<std::pair<float, TokenId>> best;
    best.reserve(k + 1);
    for (std::size_t v = 0; v < logits.size(); ++v) {
        const float value = adjusted(v);
        if (!std::isfinite(value)) { continue; }
        if (best.size() == k && value <= best.back().first) { continue; }
        auto where = std::upper_bound(best.begin(), best.end(), value,
                                      [](float x, const auto& item) { return x > item.first; });
        best.insert(where, {value, static_cast<TokenId>(v)});
        if (best.size() > k) { best.pop_back(); }
    }
    if (best.empty()) { throw std::runtime_error("sampling found no finite logit"); }
    if (sampling.temperature <= 0.0F) { return best.front().second; }
    std::vector<double> weights(best.size());
    for (std::size_t i = 0; i < best.size(); ++i) {
        weights[i] = std::exp(double(best[i].first - best.front().first) / sampling.temperature);
    }
    const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    std::size_t keep   = weights.size();
    if (sampling.min_p > 0.0F) {
        while (keep > 1 && weights[keep - 1] < sampling.min_p * weights[0]) { --keep; }
    }
    if (sampling.top_p < 1.0F) {
        double cumulative = 0.0;
        for (std::size_t i = 0; i < keep; ++i) {
            cumulative += weights[i];
            if (cumulative >= sampling.top_p * total) {
                keep = i + 1;
                break;
            }
        }
    }
    const double kept =
        std::accumulate(weights.begin(), weights.begin() + std::ptrdiff_t(keep), 0.0);
    double draw = uniform(sampling.seed, position) * kept;
    for (std::size_t i = 0; i < keep; ++i) {
        draw -= weights[i];
        if (draw < 0.0) { return best[i].second; }
    }
    return best[keep - 1].second;
}

// The byte ranges of a rank's visit state that an anchor image keeps: the local rings (one
// contiguous range, planned first) and the convolution tails with the batch row (the reset range).
struct ImageRange {
    std::size_t offset = 0;
    std::size_t bytes  = 0;
};

[[nodiscard]] std::vector<ImageRange> image_ranges(const VisitStateLayout& layout) {
    std::vector<ImageRange> out;
    if (!layout.ring.k.empty()) {
        std::size_t begin = std::numeric_limits<std::size_t>::max(), end = 0;
        for (const auto* regions : {&layout.ring.k, &layout.ring.v}) {
            for (const TensorRegion& region : *regions) {
                begin = std::min(begin, region.region.offset);
                end   = std::max(end, region.region.offset + region.region.bytes);
            }
        }
        out.push_back({begin, end - begin});
    }
    out.push_back({layout.zero_begin, layout.zero_end - layout.zero_begin});
    return out;
}

[[nodiscard]] std::size_t image_bytes(const VisitStateLayout& layout) {
    std::size_t bytes = 0;
    for (const ImageRange& range : image_ranges(layout)) { bytes += range.bytes; }
    return bytes;
}

// A row whose token needs the soft-capped FP32 logits on the Host: sampling or penalties.
[[nodiscard]] bool host_choice(const ResolvedSamplingParameters& sampling) noexcept {
    return sampling.temperature > 0.0F || sampling.presence_penalty != 0.0F ||
           sampling.frequency_penalty != 0.0F;
}

[[nodiscard]] std::size_t free_device_bytes() {
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

// The Inkling 2 forward over one sequence per rank (B = 1: one lane): prompt chunks run eagerly,
// a single-token decode replays one captured graph (eagerly without CUDA Graphs).
class ForwardExecution final : public ProgramExecution {
public:
    ForwardExecution(const exec::Parameters& parameters, const exec::Parameters* peer,
                     const SequenceLayout& layout, ExecutionContext& execution)
        : execution_(execution), tp_(execution.tp), layout_(layout),
          config_(parameters.model.config().text),
          rank_config_(tp_ == 2 ? exec::shard_text_config(config_, 2) : config_),
          parameters_{&parameters, peer} {
        if (layout_.max_concurrency != 1) {
            throw std::logic_error("Inkling 2 execution runs one lane");
        }
        if (tp_ == 2) {
            const bool peer_access = ops::enable_peer_access(execution_);
            events_.emplace(execution_);
            if (!peer_access) {
                // Chunks of 1 MiB: each chunk's host-to-device leg overlaps the next chunk's
                // device-to-host leg (the prefill exchanges are 4.7-9.4 MB per visit).
                exchange_.emplace(execution_, std::size_t(1) << 20);
                events_->attach_host_exchange(&*exchange_);
            }
        }
        max_logits_ = std::int32_t(layout_.logit_columns);
        for (int r = 0; r < tp_; ++r) {
            const ScopedCurrentDevice scope(execution_.dev[at(r)]->device);
            LayoutBuilder builder;
            state_layout_[at(r)] = plan_visit_state(builder, rank_config_, layout_.capacity);
            backing_[at(r)]      = std::make_unique<DeviceBuffer>(state_layout_[at(r)].bytes);
            // Positions the sequence has not reached hold zeros, not stale device memory.
            CUDA_CHECK(cudaMemset(backing_[at(r)]->p, 0, backing_[at(r)]->bytes));
            state_[at(r)] = std::make_unique<VisitState>(
                DeviceSpan{backing_[at(r)]->p, backing_[at(r)]->bytes}, state_layout_[at(r)]);
            work_[at(r)] = std::make_unique<WorkspaceArena>(layout_.workspace_bytes);
            ids_[at(r)]  = std::make_unique<DeviceBuffer>(std::size_t(layout_.prefill_chunk) * 4);
            positions_[at(r)] =
                std::make_unique<DeviceBuffer>(std::size_t(layout_.prefill_chunk) * 4);
            if (layout_.context_cache) {
                image_[at(r)] = std::make_unique<DeviceBuffer>(image_bytes(state_layout_[at(r)]));
            }
            prefetch_[at(r)] = std::make_unique<exec::PrefetchLane>(execution_.dev[at(r)]->device);
        }
        const ScopedCurrentDevice scope(execution_.dev[0]->device);
        const std::size_t V = config_.unpadded_vocab_size;
        logits_mem_         = std::make_unique<DeviceBuffer>(V * std::size_t(max_logits_) * 2);
        capped_mem_         = std::make_unique<DeviceBuffer>(V * std::size_t(max_logits_) * 4);
        argmax_mem_         = std::make_unique<DeviceBuffer>(4);
        if (layout_.draft_tokens > 0) { create_drafter(); }
        prepare();
    }

    void reset(std::uint32_t lane) override {
        require_lane(lane);
        for (int r = 0; r < tp_; ++r) {
            const ScopedCurrentDevice scope(execution_.dev[at(r)]->device);
            state_[at(r)]->reset(stream(r));
            CUDA_CHECK(cudaStreamSynchronize(stream(r)));
        }
        if (drafter_) { drafter_->clear_pending(); }
        forget_pending();
    }

    void save_image(std::uint32_t lane) override {
        copy_image(lane, true);
        if (drafter_) { drafter_->save_image(); }
    }

    void restore_image(std::uint32_t lane) override {
        copy_image(lane, false);
        if (drafter_) { drafter_->restore_image(); }
        forget_pending();
    }

    std::uint32_t draft_tokens() const noexcept override {
        return drafter_ ? layout_.draft_tokens : 0U;
    }

    bool paired_drafters() const noexcept override { return pairs_[1] != nullptr; }

    std::optional<StartupCosts> startup_costs() const override { return startup_costs_; }

    std::vector<TokenId> speculate(std::uint32_t lane, TokenId anchor, std::uint32_t frontier,
                                   std::uint32_t extent, const ResolvedSamplingParameters& select,
                                   std::span<const TokenId> generated,
                                   SpeculativeBackend drafter) override {
        require_lane(lane);
        if (!drafter_ || extent > layout_.draft_tokens ||
            frontier + layout_.draft_tokens + 1U > layout_.capacity) {
            throw std::logic_error("Inkling 2 speculative round outside its planned extent");
        }
        // Paired drafters: round 0 is DSpark's (pairs_[0], also drafter_), round 1 DFlash2's. The
        // round materializes at most its own width of pending columns; ordinary steps may have
        // left more.
        const std::size_t which =
            paired_drafters() && drafter == SpeculativeBackend::DFlash2 ? 1 : 0;
        if (pending_columns_ > std::int32_t(layout_.draft_tokens) + 1) { catch_up(frontier); }
        auto& round = rounds_[which];
        if (!round) {
            transport();
            std::array<speculative::RoundRank, 2> ranks{};
            for (int r = 0; r < tp_; ++r) {
                ranks[at(r)] = {.config     = &rank_config_,
                                .parameters = parameters_[at(r)],
                                .work       = work_[at(r)].get(),
                                .stream     = stream(r),
                                .device     = execution_.dev[at(r)]->device,
                                .state      = state_[at(r)].get(),
                                .prefetch   = prefetch_[at(r)].get()};
            }
            round = std::make_unique<speculative::VerifyRound>(
                config_, ranks, tp_,
                exec::ForwardCollectives{&execution_, events_ ? &*events_ : nullptr},
                bridge_ ? &*bridge_ : nullptr, layout_.draft_tokens,
                std::int32_t(layout_.forward_tokens), max_logits_,
                which == 1 ? pairs_[1].get() : drafter_.get());
        }
        last_round_                    = round.get();
        speculative::VerifyIngress& in = round->ingress();
        in.anchor                      = anchor;
        in.frontier                    = static_cast<std::int32_t>(frontier);
        in.extent                      = static_cast<std::int32_t>(extent);
        round->launch(layout_.cuda_graph);
        pending_width_                       = std::int32_t(layout_.draft_tokens) + 1;
        fed_synced_                          = false;
        const speculative::VerifyEgress& out = round->egress();
        pending_columns_                     = out.count;
        if (!host_choice(select)) {
            return {out.licensed.begin(), out.licensed.begin() + out.count};
        }
        // The Host sampler, column by column, on the soft-capped logits of the columns verified.
        const std::size_t V = config_.unpadded_vocab_size;
        if (!sampled_) {
            sampled_ = std::make_unique<PinnedHostBuffer>(
                V * (std::size_t(layout_.draft_tokens) + 1U) * sizeof(float));
        }
        const std::span<const float> host(static_cast<const float*>(sampled_->data()),
                                          V * (std::size_t(extent) + 1U));
        {
            const ScopedCurrentDevice scope(execution_.dev[0]->device);
            CUDA_CHECK(cudaMemcpy(sampled_->data(), round->capped().data, host.size() * 4,
                                  cudaMemcpyDeviceToHost));
        }
        std::vector<TokenId> history(generated.begin(), generated.end());
        std::vector<TokenId> taken;
        for (std::uint32_t j = 0; j <= extent; ++j) {
            const TokenId token =
                sample_host(host.subspan(std::size_t(j) * V, V), select, history, frontier + j);
            taken.push_back(token);
            if (j == extent || token != out.drafts[j]) { break; }
            history.push_back(token);
        }
        if (static_cast<std::int32_t>(taken.size()) != out.count) {
            round->recommit(static_cast<std::int32_t>(taken.size()));
            pending_columns_ = static_cast<std::int32_t>(taken.size());
        }
        return taken;
    }

    void settle(std::uint32_t lane, std::uint32_t committed) override {
        require_lane(lane);
        if (last_round_ == nullptr) {
            throw std::logic_error("Inkling 2 settle without a speculative round");
        }
        last_round_->recommit(static_cast<std::int32_t>(committed));
        pending_columns_ = static_cast<std::int32_t>(committed);
    }

    std::optional<TokenId> prefill(std::uint32_t lane, std::span<const TokenId> tokens,
                                   std::uint32_t position,
                                   const ResolvedSamplingParameters* select) override {
        require_lane(lane);
        run(tokens, position, select != nullptr ? 1 : 0, select != nullptr && host_choice(*select),
            true);
        if (select == nullptr) { return std::nullopt; }
        return choose(*select, {}, position + std::uint32_t(tokens.size()) - 1U);
    }

    TokenId decode(std::uint32_t lane, std::span<const TokenId> tokens, std::uint32_t position,
                   const ResolvedSamplingParameters& select,
                   std::span<const TokenId> generated) override {
        require_lane(lane);
        if (tokens.size() == 1 && (layout_.cuda_graph || paired_drafters())) {
            replay(tokens.front(), position);
            return choose(select, generated, position, true);
        }
        run(tokens, position, 1, host_choice(select), true);
        return choose(select, generated, position + std::uint32_t(tokens.size()) - 1U);
    }

    std::vector<float> score(std::span<const TokenId> tokens, std::uint32_t first_target) override {
        reset(0);
        std::vector<float> out;
        out.reserve(tokens.size() - first_target);
        const std::size_t V = config_.unpadded_vocab_size;
        std::vector<float> host;
        // Column t of a call at `begin` predicts tokens[begin + t + 1]; the last token predicts
        // nothing and is not run.
        const std::uint32_t inputs = static_cast<std::uint32_t>(tokens.size()) - 1U;
        for (std::uint32_t begin = 0; begin < inputs;) {
            const std::uint32_t width =
                std::min({std::uint32_t(max_logits_), layout_.prefill_chunk, inputs - begin});
            const bool wanted = begin + width >= first_target;
            run(tokens.subspan(begin, width), begin, wanted ? std::int32_t(width) : 0, wanted);
            if (wanted) {
                host.resize(V * width);
                const ScopedCurrentDevice scope(execution_.dev[0]->device);
                CUDA_CHECK(cudaMemcpy(host.data(), capped_mem_->p, host.size() * 4,
                                      cudaMemcpyDeviceToHost));
                for (std::uint32_t t = 0; t < width; ++t) {
                    const std::uint32_t target = begin + t + 1U;
                    if (target < first_target) { continue; }
                    const float* column = host.data() + std::size_t(t) * V;
                    const float peak    = *std::max_element(column, column + V);
                    double sum          = 0.0;
                    for (std::size_t v = 0; v < V; ++v) {
                        sum += std::exp(double(column[v] - peak));
                    }
                    out.push_back(float(double(column[tokens[target]] - peak) - std::log(sum)));
                }
            }
            begin += width;
        }
        return out;
    }

    // Rank 0's device (tp 2 ranks hold the same shapes): weights, the fixed sequence memory (the
    // visit state with its global K/V at max_context, the anchor image, the token and logit
    // buffers), the workspace, and the startup warmup and graph preparation per rank.
    MemorySummary memory_summary() const override {
        MemorySummary out;
        out.device          = execution_.dev[0]->device;
        out.max_context     = layout_.capacity;
        out.kv_cache        = KvCacheStorage::BFloat16;
        const auto& model   = parameters_[0]->model;
        const auto& weights = model.storage_stats();
        const std::size_t weight_bytes =
            (tp_ == 2 ? weights.per_device_capacity_bytes[0] : weights.device_capacity_bytes) +
            model.expert_bank_bytes(0);
        out.weights                = ArenaMemorySummary{weight_bytes, weight_bytes, weight_bytes};
        std::size_t sequence_bytes = backing_[0]->bytes + ids_[0]->bytes + positions_[0]->bytes +
                                     logits_mem_->bytes + capped_mem_->bytes + argmax_mem_->bytes;
        if (image_[0] != nullptr) { sequence_bytes += image_[0]->bytes; }
        if (drafter_) { sequence_bytes += drafter_->state_bytes() + drafter_->workspace_bytes(); }
        out.sequence = ArenaMemorySummary{sequence_bytes, sequence_bytes, sequence_bytes};
        out.workspace =
            ArenaMemorySummary{work_[0]->capacity(), work_[0]->used(), work_[0]->peak_used()};
        out.workspace_logical_peak_bytes = work_[0]->peak_used();
        out.cuda_graph_allowance_bytes   = layout_.graph_allowance_bytes;
        out.cuda_graph_observed_bytes    = graph_observed_bytes_;
        out.kv_payload_bytes             = layout_.page_group_bytes * state_layout_[0].pages;
        out.prefill_chunk                = layout_.prefill_chunk;
        return out;
    }

    void reset_memory_peaks() noexcept override {
        for (int r = 0; r < tp_; ++r) { work_[at(r)]->reset_peak(); }
    }

    std::vector<std::uint8_t> snapshot(std::uint32_t lane) const override {
        require_lane(lane);
        std::vector<std::uint8_t> out;
        const auto append = [&](const void* device, std::size_t bytes) {
            const std::size_t offset = out.size();
            out.resize(offset + bytes);
            CUDA_CHECK(cudaMemcpy(out.data() + offset, device, bytes, cudaMemcpyDeviceToHost));
        };
        for (int r = 0; r < tp_; ++r) {
            const ScopedCurrentDevice scope(execution_.dev[at(r)]->device);
            CUDA_CHECK(cudaStreamSynchronize(stream(r)));
            append(backing_[at(r)]->p, backing_[at(r)]->bytes);
        }
        const ScopedCurrentDevice scope(execution_.dev[0]->device);
        const std::size_t V      = config_.unpadded_vocab_size;
        const std::size_t column = std::size_t(std::max<std::int64_t>(capped_.ne[1] - 1, 0)) * V;
        append(static_cast<const float*>(capped_mem_->p) + column, V * 4);
        return out;
    }

    TpTransportStatus tp_transport() const override {
        TpTransportStatus out;
        if (drafter_) {
            out.drafter = std::string(layout_.draft_backend == SpeculativeBackend::Auto
                                          ? "DSpark (mtp) and DFlash2 (dflash2), chosen per round "
                                            "(or neither), "
                                      : layout_.draft_backend == SpeculativeBackend::DFlash2
                                          ? "DFlash2 (dflash2), "
                                          : "DSpark (mtp), ") +
                          std::to_string(layout_.draft_tokens) + " drafts per round" +
                          (tp_ == 2 ? ", split by heads and vocabulary over 2 ranks" : "");
        }
        if (tp_ != 2) { return out; }
        out.eager_transport = exchange_ ? "pinned host exchange" : "direct peer copies";
        if (layout_.cuda_graph) {
            out.transport  = "mailbox";
            out.slot_bytes = kMailboxSlotBytes;
        } else {
            out.transport = out.eager_transport;
        }
        return out;
    }

private:
    void require_lane(std::uint32_t lane) const {
        if (lane != 0) { throw std::logic_error("Inkling 2 execution runs lane 0 only"); }
    }

    [[nodiscard]] cudaStream_t stream(int r) const { return execution_.dev[at(r)]->stream; }

    // Saves (or restores) the image ranges of every rank's state, stream ordered, then waits.
    void copy_image(std::uint32_t lane, bool save) {
        require_lane(lane);
        if (!layout_.context_cache) {
            throw std::logic_error("Inkling 2 execution has no anchor image without context cache");
        }
        for (int r = 0; r < tp_; ++r) {
            const ScopedCurrentDevice scope(execution_.dev[at(r)]->device);
            auto* state        = static_cast<unsigned char*>(backing_[at(r)]->p);
            auto* image        = static_cast<unsigned char*>(image_[at(r)]->p);
            std::size_t offset = 0;
            for (const ImageRange& range : image_ranges(state_layout_[at(r)])) {
                unsigned char* live = state + range.offset;
                CUDA_CHECK(cudaMemcpyAsync(save ? image + offset : live,
                                           save ? live : image + offset, range.bytes,
                                           cudaMemcpyDeviceToDevice, stream(r)));
                offset += range.bytes;
            }
        }
        for (int r = 0; r < tp_; ++r) {
            const ScopedCurrentDevice scope(execution_.dev[at(r)]->device);
            CUDA_CHECK(cudaStreamSynchronize(stream(r)));
        }
    }

    // The drafter of the artifact's mtp component (--spec mtp) or dflash2 component (--spec
    // dflash2), split like the target at tp 2; with --spec auto both, the DFlash2 drafter on the
    // DSpark drafter's pending features, and one PairedProposer per proposing drafter.
    void create_drafter() {
        if (layout_.draft_backend == SpeculativeBackend::Auto) {
            create_dspark();
            std::array<Tensor, 2> features;
            for (int r = 0; r < tp_; ++r) {
                features[at(r)] = parts_[0]->sink(r, max_draft_width())->features;
            }
            create_dflash2(features);
            pairs_[0] = std::make_unique<draft::PairedProposer>(*parts_[0], *parts_[1]);
            pairs_[1] = std::make_unique<draft::PairedProposer>(*parts_[1], *parts_[0]);
            drafter_  = std::move(pairs_[0]);
            for (int r = 0; r < tp_; ++r) {
                const ScopedCurrentDevice scope(execution_.dev[at(r)]->device);
                const Tensor& pending = features[at(r)];
                features_[at(r)]      = pending;
                staging_[at(r)] =
                    std::make_unique<DeviceBuffer>(std::size_t(pending.ne[0]) * 4 * 2);
                fed_[at(r)] = std::make_unique<DeviceBuffer>(8);
                const std::array<std::int32_t, 2> words{0, 4};
                fed_[at(r)]->copy_from_host(words.data(), sizeof(words));
                fed_sinks_[at(r)] = {
                    parts_[0]->sink(r, 1)->visits,
                    Tensor(staging_[at(r)]->p, DType::BF16, {pending.ne[0], 1, 4})};
            }
            return;
        }
        if (layout_.draft_backend == SpeculativeBackend::DFlash2) {
            create_dflash2({});
            drafter_ = std::move(parts_[1]);
            return;
        }
        create_dspark();
        drafter_ = std::move(parts_[0]);
    }

    [[nodiscard]] std::int32_t max_draft_width() const {
        return std::max<std::int32_t>(std::int32_t(layout_.prefill_chunk), 8);
    }

    void create_dflash2(std::array<Tensor, 2> features) {
        const auto& dflash2 = parameters_[0]->model.config().dflash2;
        if (!dflash2) { throw std::logic_error("Inkling 2 draft tokens without a drafter"); }
        dflash2_config_ = draft::shard_dflash2_config(*dflash2, tp_);
        std::array<draft::DFlash2Rank, 2> ranks{};
        for (int r = 0; r < tp_; ++r) {
            const ScopedCurrentDevice scope(execution_.dev[at(r)]->device);
            dflash2_parameters_[at(r)] = std::make_unique<draft::DFlash2Parameters>(
                parameters_[at(r)]->model, r, dflash2_config_);
            ranks[at(r)] = {&dflash2_config_, dflash2_parameters_[at(r)].get(), parameters_[at(r)],
                            stream(r), execution_.dev[at(r)]->device};
        }
        parts_[1] = std::make_unique<draft::DFlash2Drafter>(
            *dflash2, ranks, tp_,
            exec::ForwardCollectives{&execution_, events_ ? &*events_ : nullptr},
            std::max<std::int32_t>(std::int32_t(layout_.prefill_chunk),
                                   std::int32_t(dflash2->block_size)),
            layout_.context_cache, features);
    }

    void create_dspark() {
        const auto& draft = parameters_[0]->model.config().draft;
        if (!draft) { throw std::logic_error("Inkling 2 draft tokens without a drafter"); }
        draft_config_ = draft::shard_draft_config(*draft, tp_);
        std::array<draft::DraftRank, 2> ranks{};
        for (int r = 0; r < tp_; ++r) {
            const ScopedCurrentDevice scope(execution_.dev[at(r)]->device);
            draft_parameters_[at(r)] =
                std::make_unique<draft::Parameters>(parameters_[at(r)]->model, r, draft_config_);
            ranks[at(r)] = {&draft_config_, draft_parameters_[at(r)].get(), parameters_[at(r)],
                            stream(r), execution_.dev[at(r)]->device};
        }
        parts_[0] = std::make_unique<draft::Drafter>(
            *draft, config_, ranks, tp_,
            exec::ForwardCollectives{&execution_, events_ ? &*events_ : nullptr},
            std::max<std::int32_t>(std::int32_t(layout_.prefill_chunk),
                                   std::int32_t(draft->block_size)),
            layout_.capacity, layout_.context_cache);
    }

    // The two-device transport of the captured graphs, created once: the mailbox (with wide slots
    // for the speculative round's exchanges when there is a drafter) and the bridge.
    void transport() {
        if (tp_ != 2 || mailbox_) { return; }
        mailbox_.emplace(
            execution_, kMailboxSlotBytes, 2,
            drafter_ ? speculative::round_exchange_bytes(config_, speculative::kMaxDrafts) : 0);
        // Every kernel that follows an exchange of the forward waits for its dependency before it
        // reads the stream's data, so an exchange may release it at entry: it launches while the
        // exchange waits on the peer.
        mailbox_->set_release_dependents(true);
        events_->attach_mailbox(&*mailbox_);
        bridge_.emplace(execution_.dev[0]->device, execution_.dev[1]->device);
    }

    // Runs `tokens` at `position` on every rank; the head writes the last `logit_columns` columns
    // (BF16, and FP32 after the soft cap when `capped`). With `feed` and a drafter, the call's
    // states become the drafter's context.
    void run(std::span<const TokenId> tokens, std::uint32_t position, std::int32_t logit_columns,
             bool capped, bool feed = false) {
        const auto T = static_cast<std::int32_t>(tokens.size());
        if (T == 0 || T > std::int32_t(layout_.prefill_chunk) || logit_columns > max_logits_ ||
            position + tokens.size() > layout_.capacity) {
            throw std::logic_error("Inkling 2 forward call exceeds its planned extent");
        }
        std::vector<std::int32_t> positions(tokens.size());
        std::iota(positions.begin(), positions.end(), std::int32_t(position));
        std::array<exec::ForwardRank<WorkspaceArena>, 2> ranks{};
        std::array<Tensor, 2> ids;
        std::array<Tensor, 2> position_tensors;
        for (int r = 0; r < tp_; ++r) {
            const ScopedCurrentDevice scope(execution_.dev[at(r)]->device);
            CUDA_CHECK(cudaStreamSynchronize(stream(r)));
            ids_[at(r)]->copy_from_host(tokens.data(), tokens.size() * 4);
            positions_[at(r)]->copy_from_host(positions.data(), positions.size() * 4);
            ids[at(r)]              = Tensor(ids_[at(r)]->p, DType::I32, {T});
            position_tensors[at(r)] = Tensor(positions_[at(r)]->p, DType::I32, {T});
            ranks[at(r)]            = {.config     = &rank_config_,
                                       .parameters = parameters_[at(r)],
                                       .work       = work_[at(r)].get(),
                                       .stream     = stream(r),
                                       .device     = execution_.dev[at(r)]->device,
                                       .state      = state_[at(r)].get(),
                                       .positions  = &position_tensors[at(r)],
                                       .prefetch   = prefetch_[at(r)].get()};
        }
        // The drafter's context of the last calls first: this call's features replace them.
        if (feed && drafter_ && pending_width_ > 0) { catch_up(position); }
        for (int r = 0; feed && drafter_ && r < tp_; ++r) {
            ranks[at(r)].features = drafter_->sink(r, T);
        }
        exec::Forward<WorkspaceArena> forward(config_, ranks, tp_,
                                              {&execution_, events_ ? &*events_ : nullptr},
                                              std::int32_t(layout_.forward_tokens), max_logits_);
        const auto V = static_cast<std::int32_t>(config_.unpadded_vocab_size);
        logits_      = Tensor(logits_mem_->p, DType::BF16, {V, std::max(logit_columns, 1)});
        capped_      = Tensor(capped_mem_->p, DType::FP32, {V, std::max(logit_columns, 1)});
        const ScopedCurrentDevice scope(execution_.dev[0]->device);
        forward.run(ids, exec::ForwardColumns{T, std::int32_t(position), logit_columns},
                    exec::ForwardLogits{&logits_, capped ? &capped_ : nullptr});
        for (int r = 0; r < tp_; ++r) {
            const ScopedCurrentDevice rank(execution_.dev[at(r)]->device);
            CUDA_CHECK(cudaStreamSynchronize(stream(r)));
        }
        if (feed && drafter_) {
            drafter_->append(std::int32_t(position), T);
            forget_pending();
        }
    }

    // Eager: every pending column (the last round's licensed ones, then the ordinary steps')
    // becomes context of the drafters, ending at `frontier`.
    void catch_up(std::uint32_t frontier) {
        if (std::uint32_t(pending_columns_) > frontier) {
            throw std::logic_error("Inkling 2 pending drafter features beyond their frontier");
        }
        if (pending_width_ > 0) { drafter_->catch_up(std::int32_t(frontier), pending_width_); }
        forget_pending();
    }

    void forget_pending() {
        pending_width_   = 0;
        pending_columns_ = 0;
        fed_synced_      = false;
    }

    // Startup, before the first request: one eager prompt block as wide as a prefill chunk with
    // the score path's logit columns, then the decode step (captured and replayed with CUDA Graphs,
    // eager otherwise), so the first request finds every kernel's code loaded and the graph
    // uploaded. The state is cleared afterwards; with CUDA Graphs the device memory this took on
    // each rank is the graph preparation's observation.
    void prepare() {
        if (layout_.capacity < 2) { return; }
        std::array<std::size_t, 2> free_before{};
        for (int r = 0; r < tp_; ++r) {
            const ScopedCurrentDevice scope(execution_.dev[at(r)]->device);
            CUDA_CHECK(cudaDeviceSynchronize());
            free_before[at(r)] = free_device_bytes();
        }
        const ResolvedSamplingParameters greedy;
        const std::uint32_t width = std::min(layout_.prefill_chunk, layout_.capacity - 1U);
        const std::vector<TokenId> block(width, 0);
        run(block, 0, std::min<std::int32_t>(max_logits_, std::int32_t(width)), true, true);
        (void)choose(greedy, {}, width - 1U);
        const std::array<TokenId, 1> next{0};
        (void)decode(0, next, width, greedy, {});
        // The speculative round's graph, captured here like the decode step's.
        if (drafter_ && width + 1U + layout_.draft_tokens + 1U <= layout_.capacity) {
            (void)speculate(0, 0, width + 1U, 0, greedy, {}, SpeculativeBackend::Mtp);
            if (paired_drafters()) {
                (void)speculate(0, 0, width + 1U, 0, greedy, {}, SpeculativeBackend::DFlash2);
                if (layout_.cuda_graph) { time_steps(width + 1U); }
            }
        }
        reset(0);
        if (!layout_.cuda_graph) { return; }
        for (int r = 0; r < tp_; ++r) {
            const ScopedCurrentDevice scope(execution_.dev[at(r)]->device);
            CUDA_CHECK(cudaDeviceSynchronize());
            const std::size_t free_after = free_device_bytes();
            graph_observed_bytes_[at(r)] =
                free_before[at(r)] > free_after ? free_before[at(r)] - free_after : 0;
        }
    }

    // The startup's timing of the paired execution's steps at `position` (startup_costs()): eight
    // fed ordinary steps, then eight rounds of each drafter, each block's median after its first
    // two. Every call runs at the same position, so nothing is left pending between two of them
    // (the features of repeated calls at one position would end there and be caught up before it).
    // The state is reset afterwards.
    void time_steps(std::uint32_t position) {
        using Clock = std::chrono::steady_clock;
        const ResolvedSamplingParameters greedy;
        const std::array<TokenId, 1> token{0};
        const auto median = [&](const auto& call) {
            std::vector<double> ms;
            for (int i = 0; i < 8; ++i) {
                drafter_->clear_pending();
                forget_pending();
                const auto begin = Clock::now();
                call();
                if (i >= 2) {
                    ms.push_back(
                        std::chrono::duration<double, std::milli>(Clock::now() - begin).count());
                }
            }
            std::nth_element(ms.begin(), ms.begin() + std::ptrdiff_t(ms.size() / 2), ms.end());
            return ms[ms.size() / 2];
        };
        const std::uint32_t K = layout_.draft_tokens;
        StartupCosts out;
        out.position    = position;
        out.step_ms     = median([&] { (void)decode(0, token, position, greedy, {}); });
        out.round_ms[0] = median(
            [&] { (void)speculate(0, 0, position, K, greedy, {}, SpeculativeBackend::Mtp); });
        out.round_ms[1] = median(
            [&] { (void)speculate(0, 0, position, K, greedy, {}, SpeculativeBackend::DFlash2); });
        startup_costs_ = out;
    }

    // One decode step from the captured graph: the token and its position go to the fixed device
    // buffers the graph reads; the state advances inside it.
    // With paired drafters (--spec auto) the step feeds both drafters (decode_body), also without
    // CUDA Graphs, where the same body runs eagerly.
    void replay(TokenId token, std::uint32_t position) {
        if (position + 1U > layout_.capacity) {
            throw std::logic_error("Inkling 2 decode exceeds its planned extent");
        }
        const bool fed = paired_drafters();
        if (fed && pending_columns_ >= max_draft_width()) { catch_up(position); }
        if (layout_.cuda_graph && !graph_.ready()) { capture_decode(position); }
        const auto at_position = static_cast<std::int32_t>(position);
        for (int r = 0; r < tp_; ++r) {
            const ScopedCurrentDevice scope(execution_.dev[at(r)]->device);
            CUDA_CHECK(cudaStreamSynchronize(stream(r)));
            ids_[at(r)]->copy_from_host(&token, 4);
            positions_[at(r)]->copy_from_host(&at_position, 4);
            if (fed && !fed_synced_) { fed_[at(r)]->copy_from_host(&pending_columns_, 4); }
        }
        const ScopedCurrentDevice scope(execution_.dev[0]->device);
        if (!layout_.cuda_graph) {
            decode_body(position);
        } else {
            if (tp_ == 2) { bridge_->gate_launch(stream(1), stream(0)); }
            graph_.launch(stream(0));
            if (tp_ == 2) { bridge_->gate_peer_after_launch(stream(1), stream(0)); }
        }
        for (int r = 0; r < tp_; ++r) {
            const ScopedCurrentDevice rank(execution_.dev[at(r)]->device);
            CUDA_CHECK(cudaStreamSynchronize(stream(r)));
        }
        if (fed) {
            fed_synced_ = true;
            ++pending_columns_;
            pending_width_ = std::max(pending_width_, pending_columns_);
        }
    }

    // The decode step of the forward driver (10-forward.md, step V2): T = 1 with its logits, their
    // soft cap and the argmax on rank 0, one graph over both devices whose all-reduces go through
    // the mailbox. Captured once, at the lane's first single-token decode, and replayed for every
    // later step of every request: the state, the token, the position and the workspace keep their
    // addresses (the arena is deterministic at T = 1). `position` only bounds the call; the graph
    // reads the position from its device buffer.
    void capture_decode(std::uint32_t position) {
        transport();
        for (int r = 0; r < tp_; ++r) {
            const ScopedCurrentDevice scope(execution_.dev[at(r)]->device);
            CUDA_CHECK(cudaStreamSynchronize(stream(r)));
        }
        const ScopedCurrentDevice scope(execution_.dev[0]->device);
        const auto body = [&] { decode_body(position); };
        if (tp_ == 2) {
            definition_.capture(stream(0), body, DecodeGraphPeerCapture{&*bridge_, stream(1)});
        } else {
            definition_.capture(stream(0), body);
        }
        graph_.instantiate(definition_);
        graph_.upload(stream(0));
        for (int r = 0; r < tp_; ++r) {
            const ScopedCurrentDevice rank(execution_.dev[at(r)]->device);
            CUDA_CHECK(cudaStreamSynchronize(stream(r)));
        }
    }

    // The decode step's work (stream ordered, captured by capture_decode): the forward of the
    // token in ids_ at the position in positions_ (`position` only bounds the call), its logits,
    // soft cap and argmax. With paired drafters the forward writes the token's four states into
    // the staging column, which joins the shared pending features after the pending ones (at the
    // count in fed_ word 0); both drafters' pending count grows by one.
    void decode_body(std::uint32_t position) {
        const bool fed = paired_drafters();
        std::array<exec::ForwardRank<WorkspaceArena>, 2> ranks{};
        std::array<Tensor, 2> ids;
        std::array<Tensor, 2> position_tensors;
        for (int r = 0; r < tp_; ++r) {
            ids[at(r)]              = Tensor(ids_[at(r)]->p, DType::I32, {1});
            position_tensors[at(r)] = Tensor(positions_[at(r)]->p, DType::I32, {1});
            ranks[at(r)]            = {.config     = &rank_config_,
                                       .parameters = parameters_[at(r)],
                                       .work       = work_[at(r)].get(),
                                       .stream     = stream(r),
                                       .device     = execution_.dev[at(r)]->device,
                                       .state      = state_[at(r)].get(),
                                       .positions  = &position_tensors[at(r)],
                                       .features   = fed ? &fed_sinks_[at(r)] : nullptr,
                                       .prefetch   = prefetch_[at(r)].get()};
        }
        exec::Forward<WorkspaceArena> forward(config_, ranks, tp_,
                                              {&execution_, events_ ? &*events_ : nullptr},
                                              std::int32_t(layout_.forward_tokens), max_logits_);
        const auto V = static_cast<std::int32_t>(config_.unpadded_vocab_size);
        logits_      = Tensor(logits_mem_->p, DType::BF16, {V, 1});
        capped_      = Tensor(capped_mem_->p, DType::FP32, {V, 1});
        forward.run(ids, exec::ForwardColumns{1, std::int32_t(position), 1},
                    exec::ForwardLogits{&logits_, &capped_});
        Tensor out(argmax_mem_->p, DType::I32, {1});
        ops::argmax(logits_, out, V, stream(0));
        if (!fed) { return; }
        std::array<Tensor, 2> counts;
        for (int r = 0; r < tp_; ++r) {
            const ScopedCurrentDevice rank(execution_.dev[at(r)]->device);
            const Tensor& staged = fed_sinks_[at(r)].features;
            const Tensor source(staged.data, DType::BF16, {staged.ne[0], 4, 1});
            Tensor destination = features_[at(r)].permute({0, 2, 1, 3});
            counts[at(r)]      = Tensor(fed_[at(r)]->p, DType::I32, {1});
            const Tensor four(static_cast<std::int32_t*>(fed_[at(r)]->p) + 1, DType::I32, {1});
            ops::scatter_bf16_batch(source, counts[at(r)], four, destination, stream(r));
            ops::increment_i32_scalar(counts[at(r)], stream(r));
        }
        drafter_->commit(counts);
    }

    // The token of the last logit column: argmax of the BF16 logits for a greedy row without
    // penalties (the soft cap is monotone), otherwise sample_host over the soft-capped FP32 logits.
    // `replayed`: the captured step already wrote the argmax and the soft-capped logits.
    [[nodiscard]] TokenId choose(const ResolvedSamplingParameters& sampling,
                                 std::span<const TokenId> generated, std::uint32_t position,
                                 bool replayed = false) {
        const ScopedCurrentDevice scope(execution_.dev[0]->device);
        const auto V = static_cast<std::int32_t>(config_.unpadded_vocab_size);
        if (replayed) {
            logits_ = Tensor(logits_mem_->p, DType::BF16, {V, 1});
            capped_ = Tensor(capped_mem_->p, DType::FP32, {V, 1});
        }
        if (!host_choice(sampling) && replayed) {
            TokenId id = 0;
            CUDA_CHECK(cudaMemcpy(&id, argmax_mem_->p, 4, cudaMemcpyDeviceToHost));
            return id;
        }
        if (!host_choice(sampling)) {
            Tensor last = logits_.slice(1, logits_.ne[1] - 1, 1).view({V, 1});
            Tensor out(argmax_mem_->p, DType::I32, {1});
            ops::argmax(last, out, V, stream(0));
            TokenId id = 0;
            CUDA_CHECK(cudaStreamSynchronize(stream(0)));
            CUDA_CHECK(cudaMemcpy(&id, argmax_mem_->p, 4, cudaMemcpyDeviceToHost));
            return id;
        }
        std::vector<float> host(static_cast<std::size_t>(V));
        const std::size_t column = std::size_t(capped_.ne[1] - 1) * std::size_t(V);
        CUDA_CHECK(cudaMemcpy(host.data(), static_cast<const float*>(capped_mem_->p) + column,
                              host.size() * 4, cudaMemcpyDeviceToHost));
        return sample_host(host, sampling, generated, position);
    }

    ExecutionContext& execution_;
    int tp_ = 1;
    SequenceLayout layout_;
    const TextConfig& config_;
    TextConfig rank_config_;
    std::array<const exec::Parameters*, 2> parameters_{};
    std::optional<ops::PeerEvents> events_;
    std::optional<ops::HostExchange> exchange_;
    std::optional<ops::PeerMailbox> mailbox_;
    std::optional<DecodeGraphPeerBridge> bridge_;
    DecodeGraphDefinition definition_;
    DecodeGraphExecutable graph_;
    std::int32_t max_logits_ = 1;
    std::array<std::size_t, 2> graph_observed_bytes_{};
    std::array<VisitStateLayout, 2> state_layout_;
    std::array<std::unique_ptr<DeviceBuffer>, 2> backing_, ids_, positions_, image_;
    std::array<std::unique_ptr<VisitState>, 2> state_;
    std::array<std::unique_ptr<WorkspaceArena>, 2> work_;
    std::array<std::unique_ptr<exec::PrefetchLane>, 2> prefetch_;
    std::unique_ptr<DeviceBuffer> logits_mem_, capped_mem_, argmax_mem_;
    Tensor logits_, capped_;
    // The drafter and its round (--spec mtp or dflash2 with the artifact's component).
    draft::DraftConfig draft_config_;
    std::array<std::unique_ptr<draft::Parameters>, 2> draft_parameters_;
    draft::DFlash2Config dflash2_config_;
    std::array<std::unique_ptr<draft::DFlash2Parameters>, 2> dflash2_parameters_;
    // The drafters (DSpark, DFlash2) the artifact's components give: with one, it is drafter_;
    // with both (--spec auto) drafter_ is the pair DSpark proposes in and pairs_[1] the one
    // DFlash2 proposes in, each with its verification round (rounds_[0], rounds_[1]).
    std::array<std::unique_ptr<draft::Proposer>, 2> parts_;
    std::array<std::unique_ptr<draft::PairedProposer>, 2> pairs_;
    std::unique_ptr<draft::Proposer> drafter_;
    std::array<std::unique_ptr<speculative::VerifyRound>, 2> rounds_;
    speculative::VerifyRound* last_round_ = nullptr;
    // The soft-capped logits of a sampled round's columns on the Host.
    std::unique_ptr<PinnedHostBuffer> sampled_;
    // The width of the calls whose features the drafter has not consumed (0: none), and how many
    // of their columns are pending (the drafters' device count).
    std::int32_t pending_width_   = 0;
    std::int32_t pending_columns_ = 0;
    // Paired drafters: the ordinary step's staging column of target states (BF16 [H, 1, 4]) and
    // its sink, the shared pending features (BF16 [H, max width, 4]), and per rank the pending
    // count the step appends at (I32, then the constant 4); fed_synced_: fed_ holds
    // pending_columns_.
    std::array<std::unique_ptr<DeviceBuffer>, 2> staging_, fed_;
    std::array<exec::FeatureSink, 2> fed_sinks_{};
    std::array<Tensor, 2> features_;
    bool fed_synced_ = false;
    std::optional<StartupCosts> startup_costs_;
};

[[nodiscard]] const std::optional<draft::DraftConfig>& logical_draft(const exec::Parameters& p) {
    return p.model.config().draft;
}

} // namespace

SequenceLayout plan_sequence_layout(const exec::Parameters& parameters,
                                    const exec::Parameters* peer_parameters,
                                    const EngineOptions& options) {
    const TextConfig& logical = parameters.model.config().text;
    const int tp              = options.tp;
    const TextConfig config   = tp == 2 ? exec::shard_text_config(logical, 2) : logical;
    SequenceLayout layout;
    layout.capacity        = options.max_context;
    layout.max_concurrency = options.max_concurrency;
    layout.prefill_chunk   = options.prefill_chunk;
    layout.page_tokens     = std::uint32_t(kVisitStatePageTokens);
    LayoutBuilder builder;
    const VisitStateLayout state = plan_visit_state(builder, config, options.max_context);
    std::uint32_t global_visits  = 0;
    for (const Visit& visit : config.visits) {
        if (visit.kind == LayerKind::Global) { ++global_visits; }
    }
    // BF16 K and FP16 V of the rank's KV heads at every global visit.
    layout.page_group_bytes =
        std::size_t(kVisitStatePageTokens) * 2 * 2 * config.key_width() * global_visits;
    std::array<exec::ForwardRank<WorkspaceArena>, 2> ranks{};
    const std::array<const exec::Parameters*, 2> rank_parameters{&parameters, peer_parameters};
    for (int r = 0; r < tp; ++r) {
        ranks[at(r)].config     = &config;
        ranks[at(r)].parameters = rank_parameters[at(r)];
        ranks[at(r)].device     = r;
    }
    const auto& draft         = logical_draft(parameters);
    const auto& dflash2       = parameters.model.config().dflash2;
    const auto backend        = options.speculative.backend;
    const bool paired         = backend == SpeculativeBackend::Auto && draft && dflash2;
    const bool dspark         = (backend == SpeculativeBackend::Mtp || paired) && draft;
    const bool blocked        = (backend == SpeculativeBackend::DFlash2 || paired) && dflash2;
    const bool drafted        = dspark || blocked;
    const std::uint32_t round = drafted ? options.speculative.draft_tokens + 1U : 0U;
    layout.forward_tokens     = std::max(options.prefill_chunk, round);
    layout.logit_columns =
        std::max(std::min<std::uint32_t>(kScoreColumns, options.prefill_chunk), round);
    const auto max_logits  = static_cast<std::int32_t>(layout.logit_columns);
    layout.workspace_bytes = exec::forward_workspace_bytes(
        logical, ranks, tp, std::int32_t(layout.forward_tokens), max_logits);
    const std::size_t V = logical.unpadded_vocab_size;
    const std::size_t io =
        std::size_t(options.prefill_chunk) * 8 + V * std::size_t(max_logits) * 6 + 4;
    // The state carries its global K/V at max_context; the curve's page count adds none on top.
    const std::size_t global_kv  = layout.page_group_bytes * state.pages;
    layout.context_cache         = options.context_cache.enabled;
    layout.cuda_graph            = options.use_cuda_graph;
    layout.image_bytes           = layout.context_cache ? image_bytes(state) : 0;
    layout.graph_allowance_bytes = layout.cuda_graph ? kGraphAllowanceBytes : 0;
    // Paired (--spec auto): both drafters, the DFlash2 one on the DSpark one's pending features,
    // and a verification round each.
    if (blocked) {
        layout.draft_tokens  = options.speculative.draft_tokens;
        layout.draft_backend = paired ? SpeculativeBackend::Auto : SpeculativeBackend::DFlash2;
        layout.draft_bytes += draft::DFlash2Drafter::planned_bytes(
                                  *dflash2, draft::shard_dflash2_config(*dflash2, tp),
                                  static_cast<std::int32_t>(V / std::size_t(tp)),
                                  std::max<std::int32_t>(std::int32_t(options.prefill_chunk),
                                                         std::int32_t(dflash2->block_size)),
                                  options.context_cache.enabled, !paired) +
                              kRoundAllowanceBytes;
    }
    if (dspark) {
        const draft::DraftConfig rank_draft = draft::shard_draft_config(*draft, tp);
        layout.draft_tokens                 = options.speculative.draft_tokens;
        layout.draft_backend = paired ? SpeculativeBackend::Auto : SpeculativeBackend::Mtp;
        layout.draft_bytes +=
            draft::Drafter::planned_bytes(
                *draft, rank_draft, static_cast<std::int32_t>(V / std::size_t(tp)),
                static_cast<std::int32_t>(
                    rank_draft.head_dim *
                    (rank_draft.num_attention_heads + 2 * rank_draft.num_key_value_heads)),
                std::max<std::int32_t>(std::int32_t(options.prefill_chunk),
                                       std::int32_t(draft->block_size)),
                options.max_context, layout.context_cache) +
            kRoundAllowanceBytes;
    }
    layout.fixed_bytes = state.bytes - std::min(state.bytes, global_kv) + layout.workspace_bytes +
                         io + layout.image_bytes + layout.graph_allowance_bytes +
                         layout.draft_bytes;
    return layout;
}

std::unique_ptr<ProgramExecution> make_program_execution(const exec::Parameters& parameters,
                                                         const exec::Parameters* peer_parameters,
                                                         const SequenceLayout& layout,
                                                         ExecutionContext& execution) {
    return std::make_unique<ForwardExecution>(parameters, peer_parameters, layout, execution);
}

} // namespace ninfer::models::inkling2::detail
