// Translation-unit fragment of src/models/qwen3_5/program/speculative/target_verification.cpp,
// included once at its end (after its closing namespace), so the two-device code below shares that
// file's private helpers without copying them. Not a header.
//
// Two-device (tp 2) overload of target_verify_accept() (declared in program/context.h).

#include "models/qwen3_5/program/program_impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ninfer::models::qwen3_5::execution {

namespace {
// Process-wide probe state; staging allocated once at Program setup (never inside a capture).
struct TpRankProbe {
    std::FILE* log           = nullptr;
    void* host[2]            = {nullptr, nullptr};
    std::size_t capacity     = 0;
    std::size_t bytes        = 0;
    std::size_t column_bytes = 0;
    std::int32_t width       = 0;
    long long rounds         = 0;
    long long mismatches     = 0;
    bool oversize_reported   = false;
};

TpRankProbe& tp_rank_probe() {
    static TpRankProbe probe = [] {
        TpRankProbe p;
        const char* path = std::getenv("NINFER_TP_RANK_PROBE");
        if (path == nullptr || *path == '\0') { return p; }
        p.log      = std::fopen(path, "a");
        p.capacity = std::size_t{32} << 20;
        for (void*& h : p.host) {
            if (cudaHostAlloc(&h, p.capacity, cudaHostAllocPortable) != cudaSuccess) {
                p.log = nullptr;
            }
        }
        if (p.log != nullptr) {
            std::fprintf(p.log, "probe start capacity %zu\n", p.capacity);
            std::fflush(p.log);
        }
        return p;
    }();
    return probe;
}
} // namespace

bool tp_rank_probe_enabled() noexcept { return tp_rank_probe().log != nullptr; }

void tp_rank_probe_check(std::uint32_t frontier, std::int32_t row, std::int32_t count,
                         const std::int32_t* tokens) noexcept {
    TpRankProbe& p = tp_rank_probe();
    if (p.log == nullptr || p.column_bytes == 0) { return; }
    ++p.rounds;
    for (std::int32_t j = 0; j < p.width; ++j) {
        const std::size_t off = (static_cast<std::size_t>(row) * p.width + j) * p.column_bytes;
        if (off + p.column_bytes > p.bytes) { break; }
        const auto* a = static_cast<const unsigned char*>(p.host[0]) + off;
        const auto* b = static_cast<const unsigned char*>(p.host[1]) + off;
        if (std::memcmp(a, b, p.column_bytes) == 0) { continue; }
        std::size_t diff = 0, first = p.column_bytes;
        for (std::size_t k = 0; k < p.column_bytes; ++k) {
            if (a[k] != b[k]) {
                ++diff;
                if (first == p.column_bytes) { first = k; }
            }
        }
        ++p.mismatches;
        std::fprintf(p.log,
                     "MISMATCH round %lld frontier %u row %d col %d/%d valid %d bytes_diff %zu "
                     "first_byte %zu tokens",
                     p.rounds, frontier, row, j, p.width, count, diff, first);
        for (std::int32_t t = 0; t < count; ++t) { std::fprintf(p.log, " %d", tokens[t]); }
        std::fputc('\n', p.log);
        std::fflush(p.log);
    }
    if (p.rounds % 5000 == 0) {
        std::fprintf(p.log, "alive rounds %lld mismatches %lld\n", p.rounds, p.mismatches);
        std::fflush(p.log);
    }
}

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          TargetVerifyFrameView peer, Tensor& peer_continuation_hidden_store,
                          ops::CausalAttentionExecutionEnvelope envelope) {
    if (execution.tp == nullptr) {
        throw std::logic_error("tensor-parallel target verify requires rank 1");
    }
    if (frame.replay_records == nullptr || peer.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    // Proposals, the gathered logits and acceptance are rank 0's alone. The split DFlash2
    // drafter captures rank 1's own features beside rank 0's (peer.feature_sink).
    if (peer.proposal_q.data != nullptr ||
        (peer.feature_sink != nullptr && frame.feature_sink == nullptr)) {
        throw std::logic_error("tensor-parallel target verify accepts on rank 0 only, and captures "
                               "rank 1's features only beside rank 0's");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    card.set_peer_feature_sink(peer.feature_sink);
    const TextContext::RankTensors hidden{frame.target_hidden, peer.target_hidden};
    card.target_verify_batch({frame.ids, peer.ids}, {frame.cache_positions, peer.cache_positions},
                             {frame.rope_positions, peer.rope_positions},
                             {frame.valid_columns, peer.valid_columns},
                             {frame.kv_table_rows, peer.kv_table_rows},
                             {frame.state_source_slots, peer.state_source_slots}, envelope, hidden,
                             frame.target_logits, frame.target_tokens, frame.feature_sink);
    card.set_peer_feature_sink(nullptr);
    accept_verified_drafts(execution, continuation_hidden_store, frame);

    // Rank 1 reads the accepted counts across devices once rank 0's acceptance is complete. The
    // copy is a pull on rank 1's stream over unified addressing, as the collectives' transfers.
    const DeviceContext& rank1    = *execution.tp->execution->dev[1];
    const ops::PeerEvents& events = *execution.tp->events;
    CUDA_CHECK(cudaEventRecord(events.inputs_ready(0), execution.device.stream));
    const ScopedCurrentDevice scope(rank1.device);
    CUDA_CHECK(cudaStreamWaitEvent(rank1.stream, events.inputs_ready(0), 0));
    CUDA_CHECK(::ninfer::memcpy_device_async(peer.accepted_drafts.data, frame.accepted_drafts.data,
                                             frame.accepted_drafts.bytes(), rank1.stream));
    ops::speculative_select_accepted_hidden(peer.target_hidden, peer.accepted_drafts,
                                            peer.selected_hidden, rank1.stream);
    ops::scatter(peer.selected_hidden, peer.state_destination_slots, peer_continuation_hidden_store,
                 rank1.stream);

    // The two ranks' verification hidden must be bit-identical (replicated residual).
    if (TpRankProbe& probe = tp_rank_probe(); probe.log != nullptr) {
        const std::size_t bytes = frame.target_hidden.bytes();
        if (bytes <= probe.capacity && bytes == peer.target_hidden.bytes() &&
            frame.target_hidden.ne[1] > 0) {
            probe.bytes = bytes;
            probe.width = frame.target_hidden.ne[1];
            probe.column_bytes =
                bytes /
                (static_cast<std::size_t>(frame.target_hidden.ne[1]) *
                 static_cast<std::size_t>(std::max<std::int64_t>(1, frame.target_hidden.ne[2])));
            CUDA_CHECK(cudaMemcpyAsync(probe.host[1], peer.target_hidden.data, bytes,
                                       cudaMemcpyDeviceToHost, rank1.stream));
            const ScopedCurrentDevice rank0(execution.device.device);
            CUDA_CHECK(cudaMemcpyAsync(probe.host[0], frame.target_hidden.data, bytes,
                                       cudaMemcpyDeviceToHost, execution.device.stream));
        } else if (!probe.oversize_reported) {
            probe.oversize_reported = true;
            std::fprintf(probe.log, "probe skip: hidden bytes %zu peer %zu capacity %zu\n", bytes,
                         peer.target_hidden.bytes(), probe.capacity);
            std::fflush(probe.log);
        }
    }
}

} // namespace ninfer::models::qwen3_5::execution
