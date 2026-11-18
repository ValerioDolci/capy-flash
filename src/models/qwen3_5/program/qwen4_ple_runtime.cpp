#include "models/qwen3_5/program/qwen4_ple_runtime.h"
#include "models/qwen3_5/program/program_impl.h"
#include "ninfer/ops/ple.h"
#include "core/device_scope.h"
#include "ops/host_moe/host_moe_executor.h"
#include "core/nvtx.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

void Qwen4PleRuntime::construct(ProgramImpl& program, const SequencePlanImpl& plan,
                                DeviceSpan backing, execution::ForwardRuntime& runtime,
                                execution::ForwardRuntime& peer_runtime) {
    const auto& config = program.parameters.model.config().text;
    const auto& q      = *plan.persistent.qwen4;
    const auto& block  = program.parameters.qwen4->layers.at(config.ple->layer);
    if (!block.ple) { throw std::logic_error("Qwen4Exp PLE layer has no PLE parameters"); }
    gather = std::make_unique<execution::PleGather>(block.ple->table, block.ple->hash, true);
    if (gather->width() != config.ple->embed_dim) {
        throw std::logic_error("Qwen4Exp PLE gather width does not match the configuration");
    }
    host.emplace(static_cast<std::size_t>(config.ple->embed_dim) * program.prefill_chunk *
                 sizeof(std::uint16_t));
    runtime.ple_gather = gather.get();
    runtime.ple_host   = static_cast<std::uint16_t*>(host->data());
    runtime.ple_states = program.state_images->ple_states(0);
    if (q.ple_record) { runtime.ple_record = q.ple_record->bind(backing); }
    if (q.decode_ple_rows) {
        runtime.decode_ple_rows = q.decode_ple_rows->bind(backing);
        decode_buffer.emplace(runtime.decode_ple_rows.bytes());
        decode_host = static_cast<std::uint16_t*>(decode_buffer->data());
        std::fill_n(decode_host, runtime.decode_ple_rows.numel(), std::uint16_t{0});
        runtime.decode_ple_host = decode_host;
    }
    const auto& peer = program.peer;
    if (peer) {
        peer_runtime.ple_states = peer->state_images->ple_states(0);
        if (q.decode_ple_rows) {
            if (!peer->qwen4_decode_ple_rows.data) {
                throw std::logic_error("Qwen4Exp rank 1 has no decode PLE rows");
            }
            peer_runtime.decode_ple_rows = peer->qwen4_decode_ple_rows;
            peer_runtime.decode_ple_host = decode_host;
            peer_runtime.ple_record      = peer->qwen4_ple_record;
        }
    }
    if (q.decode_ple_rows) {
        // Deferred decode rows: one mapped word per rank (own cache line), consumed by the
        // rank's PLE layer; the rounds raise them after gathering the rows post-launch.
        // Allocated before any capture (the graphs bake in the addresses).
        constexpr std::size_t kLine = 64;
        staged_words.emplace(2 * kLine, HostMapping::Yes);
        auto* base = static_cast<std::byte*>(staged_words->data());
        if (staged_words->device_data() != staged_words->data()) {
            throw std::logic_error("Qwen4Exp deferred PLE rows need unified addressing");
        }
        std::fill_n(base, 2 * kLine, std::byte{0});
        staged[0]                 = reinterpret_cast<std::uint32_t*>(base);
        staged[1]                 = reinterpret_cast<std::uint32_t*>(base + kLine);
        runtime.decode_ple_staged = staged[0];
        if (peer) { peer_runtime.decode_ple_staged = staged[1]; }
        stager = std::make_unique<Qwen4PleStager>([this, &program] { stage_round(program); });
    }
}

void Qwen4PleRuntime::prefetch_next_round(const ProgramImpl& program,
                                          std::span<const std::uint32_t> lanes, bool mtp) {
    if (!gather || (mtp ? !program.mtp_host_egress : !program.ordinary_host_egress)) { return; }
    const std::size_t context = program.parameters.model.config().text.ple->ngram_size - 1U;
    if (!mtp) {
        // The next round's column is the sampled token; its context the ledger's last tokens.
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const SequenceState& sequence = program.active_sequence(lanes[row]);
            const std::size_t ledger      = sequence.ledger.size();
            const std::size_t before      = std::min(context, ledger);
            std::array<std::int32_t, 8> window{};
            if (before + 1U > window.size()) { return; }
            std::size_t n = 0;
            for (std::size_t i = ledger - before; i < ledger; ++i) {
                window[n++] = sequence.ledger[i];
            }
            window[n++] = program.ordinary_host_egress->sampled_tokens[row];
            gather->prefetch(std::span<const std::int32_t>(window.data(), n), before, 1);
        }
        return;
    }
    const std::uint32_t draft_window    = program.draft_window;
    const std::uint32_t max_concurrency = program.max_concurrency;
    const auto* mtp_host_egress         = program.mtp_host_egress;
    const std::uint32_t width           = draft_window + 1U;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const SequenceState& sequence = program.active_sequence(lanes[row]);
        const std::int32_t count      = mtp_host_egress->licensed_counts[row];
        if (count <= 0 || static_cast<std::uint32_t>(count) > width) { continue; }
        // After the commit the ledger is the current one + the licensed tokens; the next round
        // reads the `context` tokens before its anchor (the last licensed token), the anchor,
        // then the drafts (padding columns repeat the anchor), as stage_rows does.
        const TokenId* licensed  = mtp_host_egress->licensed_tokens.data() + row * width;
        const std::size_t ledger = sequence.ledger.size();
        const std::size_t total =
            ledger + static_cast<std::size_t>(count) - 1U; // before the anchor
        const std::size_t before = std::min(context, total);
        std::array<std::int32_t, 32> window{};
        if (before + width > window.size()) { return; }
        std::size_t n = 0;
        for (std::size_t i = total - before; i < total; ++i) {
            window[n++] = i < ledger ? sequence.ledger[i] : licensed[i - ledger];
        }
        const TokenId anchor    = licensed[count - 1];
        window[n++]             = anchor;
        const std::int32_t next = std::clamp(mtp_host_egress->next_extents[row], 0,
                                             static_cast<std::int32_t>(draft_window));
        for (std::uint32_t j = 0; j < draft_window; ++j) {
            window[n++] = static_cast<std::int32_t>(j) < next
                              ? mtp_host_egress->next_drafts[j * max_concurrency + row]
                              : anchor;
        }
        gather->prefetch(std::span<const std::int32_t>(window.data(), n), before, width);
    }
}

void Qwen4PleRuntime::raise_staged() noexcept {
    for (std::uint32_t* word : staged) {
        if (word != nullptr) { ops::host::raise_host_flag(word); }
    }
}

void Qwen4PleRuntime::queue_rows(std::uint32_t lane, std::uint32_t frontier,
                                 std::span<const TokenId> tokens, std::size_t first_column) {
    if (tokens.size() > Stage{}.tokens.size()) {
        throw std::logic_error("Qwen4Exp PLE round window exceeds its staging");
    }
    Stage stage;
    stage.lane         = lane;
    stage.frontier     = frontier;
    stage.count        = static_cast<std::uint32_t>(tokens.size());
    stage.first_column = first_column;
    std::copy(tokens.begin(), tokens.end(), stage.tokens.begin());
    queue.push_back(stage);
}

void Qwen4PleRuntime::begin_round() {
    queue.clear();
    // No stream waits on the words now (the previous round has synchronized): a raise that no
    // round consumed (a failed round) must not release this one before its rows are written.
    for (std::uint32_t* word : staged) {
        if (word != nullptr) { ops::host::lower_host_flag(word); }
    }
}

void Qwen4PleRuntime::stage_round(const ProgramImpl& program) {
    // Raised whatever happens: a launched round waits for it at its PLE layer (a failed gather
    // fails the round; the device must still be able to finish it).
    struct Raise {
        Qwen4PleRuntime* ple;

        ~Raise() { ple->raise_staged(); }
    } raise{this};

    for (const Stage& stage : queue) {
        stage_rows(program, program.active_sequence(stage.lane).ledger, stage.frontier,
                   std::span<const TokenId>(stage.tokens.data(), stage.count), stage.first_column);
    }
    queue.clear();
}

void Qwen4PleRuntime::stage_rows(const ProgramImpl& program, std::span<const TokenId> ledger,
                                 std::uint32_t frontier, std::span<const TokenId> tokens,
                                 std::size_t first_column) {
    nvtx::ScopedRange range(nvtx::Name::Qwen4PleGather, nvtx::Category::Control,
                            static_cast<std::uint64_t>(tokens.size()));
    const auto& forward_runtime_record = program.forward_runtime_record;
    if (!gather || decode_host == nullptr || tokens.empty() || ledger.size() < frontier ||
        !forward_runtime_record ||
        first_column + tokens.size() >
            static_cast<std::size_t>(forward_runtime_record->decode_ple_rows.ne[1])) {
        throw std::logic_error("Qwen4Exp PLE round staging is invalid");
    }
    // The n-gram rows of position p read the ngram_size-1 tokens before p; the window's own
    // tokens come after the committed ones. A shorter history is the sequence start.
    const std::size_t context = program.parameters.model.config().text.ple->ngram_size - 1U;
    const std::size_t before  = std::min<std::size_t>(context, frontier);
    std::array<std::int32_t, 32> window{};
    if (before + tokens.size() > window.size()) {
        throw std::logic_error("Qwen4Exp PLE round window exceeds its staging");
    }
    std::size_t count = 0;
    for (std::size_t i = frontier - before; i < frontier; ++i) { window[count++] = ledger[i]; }
    for (const TokenId token : tokens) { window[count++] = token; }
    const std::size_t width = gather->width();
    gather->gather(
        std::span<const std::int32_t>(window.data(), count), before, tokens.size(),
        std::span<std::uint16_t>(decode_host + first_column * width, tokens.size() * width));
}

void Qwen4PleRuntime::fold(ProgramImpl& program, std::span<const ops::GdnReplayFoldRow> rows) {
    fold_rank(program, rows, 0);
    if (program.peer && program.qwen4_peer_runtime_record) {
        const ScopedCurrentDevice rank1(program.peer->device.device);
        fold_rank(program, rows, 1);
    }
}

void Qwen4PleRuntime::fold_rank(ProgramImpl& program, std::span<const ops::GdnReplayFoldRow> rows,
                                int rank) {
    const std::optional<execution::ForwardRuntime>& runtime =
        rank == 0 ? program.forward_runtime_record : program.qwen4_peer_runtime_record;
    if (!runtime || runtime->ple_record.data == nullptr) { return; }
    if (rows.empty() || rows.size() > kMaximumConcurrency) {
        throw std::logic_error("Qwen4Exp PLE fold rows are invalid");
    }
    WorkspaceArena& arena = rank == 0 ? program.work : program.peer->work;
    cudaStream_t stream   = rank == 0 ? program.device.stream : program.peer->device.stream;
    const auto count      = static_cast<std::int32_t>(rows.size());
    auto scope            = arena.scope();
    Tensor control        = arena.alloc(DType::I32, {count, 3});
    std::array<std::int32_t, 3 * kMaximumConcurrency> host{};
    for (std::int32_t row = 0; row < count; ++row) {
        host[static_cast<std::size_t>(row)]             = rows[row].source_state_slot;
        host[static_cast<std::size_t>(count + row)]     = rows[row].destination_state_slot;
        host[static_cast<std::size_t>(2 * count + row)] = rows[row].commit_columns;
    }
    // The transfer reads the pageable array before returning, so it may go out of scope.
    CUDA_CHECK(cudaMemcpyAsync(control.data, host.data(), control.bytes(), cudaMemcpyHostToDevice,
                               stream));
    const Tensor& record   = runtime->ple_record;
    const Tensor inputs    = record.slice(2, 0, count);
    Tensor states          = runtime->ple_states;
    const Tensor sources   = control.slice(1, 0, 1).view({count});
    const Tensor targets   = control.slice(1, 1, 1).view({count});
    const Tensor committed = control.slice(1, 2, 1).view({count});
    ops::ple_conv_advance(
        inputs, &committed,
        static_cast<std::int32_t>(program.parameters.model.config().text.ple->conv_history()),
        states, sources, targets, stream);
}

} // namespace ninfer::models::qwen3_5::detail
