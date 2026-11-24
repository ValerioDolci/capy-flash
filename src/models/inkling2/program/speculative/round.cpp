#include "models/inkling2/program/speculative/round.h"

#include "core/device.h"
#include "core/device_scope.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/speculative_round.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::models::inkling2::speculative {
namespace {

static_assert(std::is_trivially_copyable_v<VerifyIngress> &&
              std::is_trivially_copyable_v<VerifyEgress>);
static_assert(sizeof(TokenId) == 4);

[[nodiscard]] std::size_t at(int rank) { return static_cast<std::size_t>(rank); }

// An I32 view of `count` words at byte `offset` of `buffer`.
[[nodiscard]] Tensor words(const DeviceBuffer& buffer, std::size_t offset,
                           std::initializer_list<std::int32_t> shape) {
    return Tensor(static_cast<std::byte*>(buffer.p) + offset, DType::I32, shape);
}

} // namespace

// Rank r's device side of the round. Rank 0 also holds the egress record, the column argmax and
// the sampling configuration of the acceptance; rank 1 the copy of the licensed count.
struct VerifyRound::Rank {
    DeviceBuffer ingress{sizeof(VerifyIngress)};
    DeviceBuffer physical{4}; // I32 [1] = K: the extent of the verify inputs
    DeviceBuffer ids{kMaxWidth * 4};
    DeviceBuffer positions{kMaxWidth * 4};
    DeviceBuffer count{4};
    DeviceBuffer egress{sizeof(VerifyEgress)};
    DeviceBuffer targets{kMaxWidth * 4};
    DeviceBuffer sampling{sizeof(ops::SamplingConfig)};
    std::optional<ConvRecord> record;
    Tensor positions_view;
};

std::size_t round_exchange_bytes(const TextConfig& config, std::uint32_t drafts) {
    return std::size_t(drafts + 1U) * config.hidden_size * sizeof(float);
}

VerifyRound::VerifyRound(const TextConfig& config, std::array<RoundRank, 2> ranks, int width,
                         execution::ForwardCollectives collectives,
                         const DecodeGraphPeerBridge* bridge, std::uint32_t drafts,
                         std::int32_t max_tokens, std::int32_t max_logit_columns,
                         draft::Drafter* drafter)
    : config_(config), ranks_(ranks), width_(width), collectives_(collectives), bridge_(bridge),
      drafts_(drafts), max_tokens_(max_tokens), max_logit_columns_(max_logit_columns),
      drafter_(drafter) {
    if (drafter != nullptr && drafter->proposals() != kMaxDrafts) {
        throw std::invalid_argument("Inkling 2 verification round: the drafter proposes " +
                                    std::to_string(kMaxDrafts) + " drafts");
    }
    const auto T = static_cast<std::int32_t>(drafts + 1U);
    if (drafts == 0 || drafts > kMaxDrafts) {
        throw std::invalid_argument("Inkling 2 verification round: drafts must be 1.." +
                                    std::to_string(kMaxDrafts));
    }
    if ((width != 1 && width != 2) || (width == 2 && bridge == nullptr) || max_tokens < T ||
        max_logit_columns < T) {
        throw std::invalid_argument("Inkling 2 verification round: invalid width or bounds");
    }
    for (int r = 0; r < width_; ++r) {
        const RoundRank& rank = ranks_[at(r)];
        if (rank.config == nullptr || rank.parameters == nullptr || rank.work == nullptr ||
            rank.state == nullptr) {
            throw std::invalid_argument("Inkling 2 verification round: incomplete rank " +
                                        std::to_string(r));
        }
        const ScopedCurrentDevice scope(rank.device);
        rank_[at(r)] = std::make_unique<Rank>();
        Rank& own    = *rank_[at(r)];
        const auto K = static_cast<std::int32_t>(drafts);
        own.physical.copy_from_host(&K, 4);
        own.record.emplace(*rank.state, static_cast<std::uint32_t>(rank.config->visits.size()), T);
        own.positions_view = Tensor(own.positions.p, DType::I32, {T});
        if (r == 0) {
            const ops::SamplingConfig greedy{};
            own.sampling.copy_from_host(&greedy, sizeof(greedy));
        }
    }
    const ScopedCurrentDevice scope(ranks_[0].device);
    const auto V = static_cast<std::int32_t>(config_.unpadded_vocab_size);
    logits_mem_  = std::make_unique<DeviceBuffer>(std::size_t(V) * std::size_t(T) * 2);
    logits_      = Tensor(logits_mem_->p, DType::BF16, {V, T});
    capped_mem_  = std::make_unique<DeviceBuffer>(std::size_t(V) * std::size_t(T) * 4);
    capped_      = Tensor(capped_mem_->p, DType::FP32, {V, T});
    accept_work_ = std::make_unique<WorkspaceArena>(std::max<std::size_t>(
        256, ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                 V, static_cast<std::int32_t>(drafts), static_cast<std::int32_t>(drafts), 1, 1)));
    host_        = std::make_unique<PinnedHostBuffer>(sizeof(VerifyIngress) + sizeof(VerifyEgress));
    ingress()    = VerifyIngress{};
}

VerifyRound::~VerifyRound() = default;

VerifyIngress& VerifyRound::ingress() noexcept {
    return *static_cast<VerifyIngress*>(host_->data());
}

const VerifyEgress& VerifyRound::egress() const noexcept {
    return *reinterpret_cast<const VerifyEgress*>(static_cast<const std::byte*>(host_->data()) +
                                                  sizeof(VerifyIngress));
}

std::size_t VerifyRound::graph_nodes() const {
    return definition_.ready() ? definition_.node_count() : 0;
}

void VerifyRound::synchronize() const {
    for (int r = 0; r < width_; ++r) {
        const ScopedCurrentDevice scope(ranks_[at(r)].device);
        CUDA_CHECK(cudaStreamSynchronize(stream(r)));
    }
}

void VerifyRound::body() {
    const auto K                    = static_cast<std::int32_t>(drafts_);
    const auto T                    = K + 1;
    const auto V                    = static_cast<std::int32_t>(config_.unpadded_vocab_size);
    constexpr std::size_t kAnchor   = offsetof(VerifyIngress, anchor);
    constexpr std::size_t kFrontier = offsetof(VerifyIngress, frontier);
    constexpr std::size_t kExtent   = offsetof(VerifyIngress, extent);
    constexpr std::size_t kDrafts   = offsetof(VerifyIngress, drafts);
    constexpr std::size_t kLicensed = offsetof(VerifyEgress, licensed);
    constexpr std::size_t kCount    = offsetof(VerifyEgress, count);
    constexpr std::size_t kAccepted = offsetof(VerifyEgress, accepted);

    // 1. Every rank uploads the record; a drafter writes its proposals over the record's drafts.
    for (int r = 0; r < width_; ++r) {
        const ScopedCurrentDevice scope(ranks_[at(r)].device);
        CUDA_CHECK(cudaMemcpyAsync(rank_[at(r)]->ingress.p, host_->data(), sizeof(VerifyIngress),
                                   cudaMemcpyHostToDevice, stream(r)));
    }
    if (drafter_ != nullptr) {
        std::array<Tensor, 2> anchors, frontiers, proposals;
        for (int r = 0; r < width_; ++r) {
            anchors[at(r)]   = words(rank_[at(r)]->ingress, kAnchor, {1});
            frontiers[at(r)] = words(rank_[at(r)]->ingress, kFrontier, {1});
            proposals[at(r)] = words(rank_[at(r)]->ingress, kDrafts, {std::int32_t(kMaxDrafts)});
        }
        drafter_->propose(anchors, frontiers, proposals, T, K);
    }

    // 2. Every rank derives its own verify inputs.
    std::array<Tensor, 2> ids;
    std::array<execution::ForwardRank<WorkspaceArena>, 2> forward_ranks{};
    for (int r = 0; r < width_; ++r) {
        const RoundRank& rank = ranks_[at(r)];
        Rank& own             = *rank_[at(r)];
        const ScopedCurrentDevice scope(rank.device);
        Tensor verify_ids = words(own.ids, 0, {T, 1});
        Tensor positions  = words(own.positions, 0, {T, 1});
        ops::speculative_prepare_verify_inputs(
            words(own.ingress, kAnchor, {1}), words(own.ingress, kDrafts, {K, 1}),
            words(own.ingress, kFrontier, {1}), words(own.physical, 0, {1}), verify_ids, positions,
            stream(r));
        ids[at(r)]           = words(own.ids, 0, {T});
        forward_ranks[at(r)] = {.config     = rank.config,
                                .parameters = rank.parameters,
                                .work       = rank.work,
                                .stream     = rank.stream,
                                .device     = rank.device,
                                .state      = rank.state,
                                .positions  = &own.positions_view,
                                .record     = &*own.record,
                                .features   = drafter_ ? drafter_->sink(r, T) : nullptr,
                                .prefetch   = rank.prefetch};
    }

    // 3. The forward at T = K + 1 with every column's logits; the convolutions record.
    const ScopedCurrentDevice scope(ranks_[0].device);
    execution::Forward<WorkspaceArena> forward(config_, forward_ranks, width_, collectives_,
                                               max_tokens_, max_logit_columns_);
    forward.run(ids, execution::ForwardColumns{T, ingress().frontier, T},
                execution::ForwardLogits{&logits_, &capped_});

    // 4. Rank 0: argmax per column, greedy acceptance over the true extent.
    Rank& zero      = *rank_[0];
    Tensor targets  = words(zero.targets, 0, {T});
    Tensor targets2 = words(zero.targets, 0, {T, 1});
    Tensor anchor   = words(zero.ingress, kAnchor, {1});
    Tensor frontier = words(zero.ingress, kFrontier, {1});
    Tensor licensed = words(zero.egress, kLicensed, {T, 1});
    Tensor count    = words(zero.egress, kCount, {1});
    Tensor accepted = words(zero.egress, kAccepted, {1});
    ops::argmax(logits_, targets, V, stream(0));
    ops::speculative_accept_greedy_drafts(
        targets2, logits_.view({V, T, 1}), words(zero.ingress, kDrafts, {K, 1}),
        words(zero.ingress, kExtent, {1}), frontier, anchor, licensed, count, accepted, V,
        static_cast<const ops::SamplingConfig*>(zero.sampling.p), *accept_work_, stream(0));

    // 5. Rank 1 pulls the licensed count once the acceptance is complete; every rank folds.
    if (width_ == 2) {
        const ops::PeerEvents& events = *collectives_.events;
        Rank& one                     = *rank_[1];
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(0), stream(0)));
        const ScopedCurrentDevice peer(ranks_[1].device);
        CUDA_CHECK(cudaStreamWaitEvent(stream(1), events.inputs_ready(0), 0));
        CUDA_CHECK(::ninfer::memcpy_device_async(one.count.p, count.data, 4, stream(1)));
        one.record->fold(words(one.count, 0, {1}), T, stream(1));
    }
    zero.record->fold(count, T, stream(0));
    if (drafter_ != nullptr) {
        drafter_->commit({count, width_ == 2 ? words(rank_[1]->count, 0, {1}) : Tensor{}});
    }

    // 6. The egress record (with the round's drafts) to the host.
    CUDA_CHECK(::ninfer::memcpy_device_async(
        static_cast<std::byte*>(zero.egress.p) + offsetof(VerifyEgress, drafts),
        static_cast<const std::byte*>(zero.ingress.p) + kDrafts, sizeof(VerifyEgress::drafts),
        stream(0)));
    CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(host_->data()) + sizeof(VerifyIngress),
                               zero.egress.p, sizeof(VerifyEgress), cudaMemcpyDeviceToHost,
                               stream(0)));
}

void VerifyRound::recommit(std::int32_t count) {
    if (count < 1 || count > static_cast<std::int32_t>(drafts_) + 1) {
        throw std::invalid_argument("Inkling 2 verification round: recommit count out of range");
    }
    synchronize();
    const auto T = static_cast<std::int32_t>(drafts_ + 1U);
    std::array<Tensor, 2> counts;
    for (int r = 0; r < width_; ++r) {
        const ScopedCurrentDevice scope(ranks_[at(r)].device);
        rank_[at(r)]->count.copy_from_host(&count, 4);
        counts[at(r)] = words(rank_[at(r)]->count, 0, {1});
        rank_[at(r)]->record->fold(counts[at(r)], T, stream(r));
    }
    if (drafter_ != nullptr) { drafter_->commit(counts); }
    synchronize();
}

void VerifyRound::capture() {
    synchronize();
    const ScopedCurrentDevice scope(ranks_[0].device);
    if (width_ == 2) {
        definition_.capture(
            stream(0), [this] { body(); }, DecodeGraphPeerCapture{bridge_, stream(1)});
    } else {
        definition_.capture(stream(0), [this] { body(); });
    }
    graph_.instantiate(definition_);
    graph_.upload(stream(0));
    synchronize();
}

void VerifyRound::launch(bool graph) {
    const VerifyIngress& in = ingress();
    if (in.extent < 0 || in.extent > static_cast<std::int32_t>(drafts_) || in.frontier < 0 ||
        std::uint64_t(in.frontier) + drafts_ + 1U > ranks_[0].state->max_context()) {
        throw std::invalid_argument(
            "Inkling 2 verification round: extent or frontier out of range");
    }
    synchronize();
    if (!graph) {
        body();
        synchronize();
        return;
    }
    if (!graph_.ready()) { capture(); }
    const ScopedCurrentDevice scope(ranks_[0].device);
    if (width_ == 2) { bridge_->gate_launch(stream(1), stream(0)); }
    graph_.launch(stream(0));
    if (width_ == 2) { bridge_->gate_peer_after_launch(stream(1), stream(0)); }
    synchronize();
}

} // namespace ninfer::models::inkling2::speculative
