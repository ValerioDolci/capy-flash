#include "models/qwen3_5/program/validation_hooks.h"
#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context.h"
#include "core/device.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::models::qwen3_5::detail {
namespace {

// Validation token sequences: one whole sequence (prompt + continuation) per line.
std::vector<std::vector<TokenId>> read_sequences(const char* path, const char* variable) {
    std::vector<std::vector<TokenId>> sequences;
    std::FILE* file = std::fopen(path, "r");
    if (file == nullptr) { throw std::runtime_error(std::string("cannot open ") + variable); }
    std::vector<TokenId> line;
    long long value = 0;
    int c           = 0;
    bool in_number  = false;
    while ((c = std::fgetc(file)) != EOF) {
        if (c >= '0' && c <= '9') {
            value     = (in_number ? value * 10 : 0) + (c - '0');
            in_number = true;
            continue;
        }
        if (in_number) { line.push_back(static_cast<TokenId>(value)); }
        in_number = false;
        if (c == '\n' && !line.empty()) {
            sequences.push_back(std::move(line));
            line = {};
        }
    }
    if (in_number) { line.push_back(static_cast<TokenId>(value)); }
    if (!line.empty()) { sequences.push_back(std::move(line)); }
    std::fclose(file);
    return sequences;
}

// NINFER_QWEN4_REPLAY_MODEL_DRAFTS=1 keeps the model's own drafts (and extent) where the replay
// has no recorded window: forced targets with realistic rejections.
bool replay_model_drafts() {
    static const bool value = [] {
        const char* v = std::getenv("NINFER_QWEN4_REPLAY_MODEL_DRAFTS");
        return v != nullptr && std::string_view(v) == "1";
    }();
    return value;
}

} // namespace

std::unique_ptr<ValidationHooks> ValidationHooks::from_environment(bool mtp,
                                                                   std::int32_t text_layers,
                                                                   std::size_t public_token_count,
                                                                   const char* meta_engine,
                                                                   const std::string& meta_model) {
    auto hooks = std::make_unique<ValidationHooks>();
    execution::require_qwen4_validation_opt_in();
    if (const char* oracle = std::getenv("NINFER_QWEN4_MTP_ORACLE");
        oracle != nullptr && *oracle != '\0' && mtp) {
        hooks->mtp_oracle = read_sequences(oracle, "NINFER_QWEN4_MTP_ORACLE");
    }
    if (const char* forced = std::getenv("NINFER_QWEN4_FORCE_TOKENS");
        forced != nullptr && *forced != '\0') {
        hooks->forced_tokens = read_sequences(forced, "NINFER_QWEN4_FORCE_TOKENS");
    }
    if (const char* none = std::getenv("NINFER_QWEN4_MTP_NO_DRAFTS")) {
        hooks->mtp_no_drafts = std::string_view(none) == "1";
    }
    if (const char* cap = std::getenv("NINFER_QWEN4_MTP_EXTENT_CAP"); cap != nullptr && *cap != 0) {
        hooks->mtp_extent_cap = static_cast<std::uint32_t>(std::stoul(cap));
    }
    if (const char* calib = std::getenv("NINFER_QWEN4_MTP_CALIB");
        calib != nullptr && *calib != '\0' && mtp) {
        hooks->calib_file = std::fopen(calib, "a");
        if (hooks->calib_file == nullptr) {
            throw std::runtime_error(std::string("NINFER_QWEN4_MTP_CALIB: cannot open ") + calib);
        }
    }
    hooks->replay = execution::Qwen4Replay::from_environment();
    (void)execution::tp_rank_probe_enabled(); // allocate the probe staging before any capture
    if (const char* rounds = std::getenv("NINFER_QWEN4_ROUND_LOG");
        rounds != nullptr && *rounds != '\0') {
        hooks->round_log = std::fopen(rounds, "a");
        if (hooks->round_log == nullptr) {
            throw std::runtime_error(std::string("NINFER_QWEN4_ROUND_LOG: cannot open ") + rounds);
        }
    }
    if (const char* record = std::getenv("NINFER_QWEN4_REPLAY_RECORD");
        record != nullptr && *record != '\0') {
        hooks->replay_record = std::fopen(record, "a");
        if (hooks->replay_record == nullptr) {
            throw std::runtime_error(std::string("NINFER_QWEN4_REPLAY_RECORD: cannot open ") +
                                     record);
        }
    }
    if (const char* top2 = std::getenv("NINFER_QWEN4_TOP2"); top2 != nullptr && *top2 != '\0') {
        hooks->top2_file = std::fopen(top2, "a");
        if (hooks->top2_file == nullptr) {
            throw std::runtime_error("cannot open NINFER_QWEN4_TOP2");
        }
        hooks->top2_domain = public_token_count;
    }
    // Validation dumps (NINFER_QWEN4_TFDUMP): hidden taps only where execution is eager.
    hooks->tfdump = execution::Qwen4TfDump::from_environment(text_layers);
    if (hooks->tfdump) { hooks->tfdump->write_meta(meta_engine, meta_model); }

    const bool any =
        !hooks->mtp_oracle.empty() || !hooks->forced_tokens.empty() || hooks->mtp_no_drafts ||
        hooks->mtp_extent_cap != qwen3_5::kMtpMaximumProposalDrafts ||
        hooks->calib_file != nullptr || hooks->replay != nullptr || hooks->round_log != nullptr ||
        hooks->replay_record != nullptr || hooks->top2_file != nullptr || hooks->tfdump != nullptr;
    if (!any) { hooks.reset(); }
    return hooks;
}

ValidationHooks::~ValidationHooks() {
    if (top2_file != nullptr) { std::fclose(top2_file); }
    if (calib_file != nullptr) { std::fclose(calib_file); }
    if (replay_record != nullptr) { std::fclose(replay_record); }
    if (round_log != nullptr) { std::fclose(round_log); }
}

bool ValidationHooks::bypasses_sampling() const noexcept {
    return !forced_tokens.empty() || !mtp_oracle.empty() || replay != nullptr || tfdump != nullptr;
}

void ValidationHooks::on_first_token(std::span<const TokenId> ledger, std::uint32_t prompt_tokens,
                                     TokenId& token) {
    if (tfdump) {
        // Validation: teacher-forced first token (NINFER_QWEN4_TFDUMP_FORCE).
        if (const std::int32_t forced = tfdump->forced_token(prompt_tokens); forced >= 0) {
            token = forced;
        }
    }
    if (replay) {
        // Replay: the reference's first token after the prompt.
        if (const execution::Qwen4ReplaySequence* ref = replay->match(ledger);
            ref != nullptr && ledger.size() == prompt_tokens) {
            token = ref->ids[prompt_tokens];
            ++replay->forced_tokens;
        }
    }
    if (replay_record != nullptr) {
        std::fprintf(replay_record, "req %u\n", static_cast<unsigned>(prompt_tokens));
    }
    round_follows = false; // round log: the next round starts a request
}

void ValidationHooks::after_ordinary_round(const ProgramImpl& program,
                                           std::span<const std::uint32_t> lanes,
                                           TokenId* sampled_tokens, const Tensor& logits) {
    if (!forced_tokens.empty()) {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            force_tokens(program.active_sequence(lanes[row]).ledger, &sampled_tokens[row], 1);
        }
    }
    if (replay) {
        // Replay: the reference token (teacher forcing through the ordinary route).
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const SequenceState& sequence = program.active_sequence(lanes[row]);
            if (const execution::Qwen4ReplaySequence* ref = replay->match(sequence.ledger)) {
                if (sampled_tokens[row] != ref->ids[sequence.ledger.size()]) {
                    ++replay->argmax_disagreements;
                }
                sampled_tokens[row] = ref->ids[sequence.ledger.size()];
                ++replay->forced_tokens;
            } else {
                ++replay->rounds_unmatched;
            }
        }
    }
    if (top2_file) {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            write_top2(logits.slice(1, static_cast<std::int32_t>(row), 1), lanes[row],
                       program.active_sequence(lanes[row]).execution_frontier, 0);
        }
    }
    if (tfdump && tfdump->logits_enabled()) {
        // Validation dump of the round's logits (position = the consumed token's position).
        const Tensor round = logits.slice(1, 0, static_cast<std::int32_t>(lanes.size()));
        std::vector<std::uint16_t> host(static_cast<std::size_t>(round.numel()));
        CUDA_CHECK(cudaMemcpy(host.data(), round.data, round.bytes(), cudaMemcpyDeviceToHost));
        std::vector<std::int32_t> positions;
        for (const std::uint32_t lane : lanes) {
            positions.push_back(
                static_cast<std::int32_t>(program.active_sequence(lane).execution_frontier));
        }
        tfdump->write_logits(host, round.ne[0], positions);
    }
}

void ValidationHooks::on_ordinary_commit(const ProgramImpl& program, std::uint32_t base_E,
                                         double seconds, std::size_t ledger_size, TokenId& token,
                                         TokenId& sampled_token) {
    log_round(program, 'o', base_E, 0, 0, seconds);
    if (tfdump) {
        // Validation: teacher-forced decode (NINFER_QWEN4_TFDUMP_FORCE).
        if (const std::int32_t forced = tfdump->forced_token(ledger_size); forced >= 0) {
            token         = forced;
            sampled_token = token;
        }
    }
}

std::uint32_t ValidationHooks::plan_mtp_row(std::span<const TokenId> ledger, std::uint32_t frontier,
                                            std::uint32_t draft_window, std::uint32_t model_extent,
                                            MtpRowReplay& row) {
    // Replay: the recorded window of this frontier (extent, drafts) and the reference.
    row.ref    = replay ? replay->match(ledger) : nullptr;
    row.window = nullptr;
    if (row.ref != nullptr) {
        const auto it = row.ref->windows.find(frontier);
        if (it != row.ref->windows.end()) { row.window = &it->second; }
        ++(row.window != nullptr ? replay->windows_replayed : replay->windows_unlisted);
    } else if (replay) {
        ++replay->rounds_unmatched;
    }
    if (row.ref == nullptr) { return model_extent; }
    return row.window != nullptr    ? row.window->extent
           : !replay_model_drafts() ? draft_window
                                    : model_extent;
}

std::uint32_t ValidationHooks::cap_mtp_extent(std::uint32_t proposed,
                                              std::uint32_t limit) const noexcept {
    return mtp_no_drafts ? 0U : std::min({proposed, limit, mtp_extent_cap});
}

void ValidationHooks::override_mtp_drafts(std::span<const TokenId> ledger, std::uint32_t frontier,
                                          std::uint32_t extent, std::uint32_t width,
                                          const MtpRowReplay& row, TokenId* drafts,
                                          TokenId* targets) const {
    if (!mtp_oracle.empty()) {
        // Validation oracle drafts: the reference continuation of the matching sequence.
        for (const std::vector<TokenId>& line : mtp_oracle) {
            if (line.size() <= ledger.size() ||
                !std::equal(ledger.begin(), ledger.end(), line.begin())) {
                continue;
            }
            for (std::uint32_t j = 0; j < extent && frontier + 1U + j < line.size(); ++j) {
                drafts[j] = line[frontier + 1U + j];
            }
            break;
        }
    }
    if (row.ref != nullptr) {
        // Replay: recorded drafts (or the reference itself where no window was
        // recorded), and the reference continuation as every column's target, so the
        // device accepts the recorded count and commits the reference tokens.
        const auto& ids = row.ref->ids;
        for (std::uint32_t j = 0; j < extent; ++j) {
            const std::size_t at = frontier + 1U + j;
            if (row.window == nullptr && replay_model_drafts()) { continue; }
            drafts[j] = row.window != nullptr && j < row.window->drafts.size()
                            ? row.window->drafts[j]
                            : ids[std::min(at, ids.size() - 1)];
        }
        for (std::uint32_t j = 0; j < width; ++j) {
            const std::size_t at = frontier + 1U + j;
            targets[j]           = ids[std::min(at, ids.size() - 1)];
        }
    }
}

void ValidationHooks::after_mtp_round(const ProgramImpl& program,
                                      std::span<const std::uint32_t> lanes, std::uint32_t width,
                                      const std::int32_t* licensed_counts, TokenId* licensed_tokens,
                                      const Tensor& target_logits) {
    if (!forced_tokens.empty()) {
        // Only the round's last committed token is not consumed by the round itself, so only
        // single-token commits (NINFER_QWEN4_MTP_NO_DRAFTS) can be forced.
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            if (licensed_counts[row] != 1) {
                throw std::logic_error("NINFER_QWEN4_FORCE_TOKENS requires one token per round");
            }
            force_tokens(program.active_sequence(lanes[row]).ledger, licensed_tokens + row * width,
                         1);
        }
    }
    if (tfdump && tfdump->logits_enabled()) {
        // Validation dump of the target logits of every committed verification column
        // (position = the consumed token's position, as for ordinary rounds).
        const Tensor& logits = target_logits;
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::int32_t count = licensed_counts[row];
            const Tensor columns =
                logits.slice(2, static_cast<std::int32_t>(row), 1).slice(1, 0, count);
            std::vector<std::uint16_t> host(static_cast<std::size_t>(columns.numel()));
            CUDA_CHECK(
                cudaMemcpy(host.data(), columns.data, columns.bytes(), cudaMemcpyDeviceToHost));
            std::vector<std::int32_t> positions;
            for (std::int32_t j = 0; j < count; ++j) {
                positions.push_back(static_cast<std::int32_t>(
                    program.active_sequence(lanes[row]).execution_frontier +
                    static_cast<std::uint32_t>(j)));
            }
            tfdump->write_logits(host, logits.ne[0], positions);
        }
    }
    if (top2_file) {
        // Validation: the target's top-2 of every committed verification column.
        const Tensor& logits = target_logits;
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::int32_t count = licensed_counts[row];
            for (std::int32_t j = 0; j < count; ++j) {
                write_top2(logits.slice(2, static_cast<std::int32_t>(row), 1)
                               .slice(1, j, 1)
                               .view({logits.ne[0], 1}),
                           lanes[row],
                           program.active_sequence(lanes[row]).execution_frontier +
                               static_cast<std::uint32_t>(j),
                           1);
            }
        }
    }
}

void ValidationHooks::on_mtp_commit(const ProgramImpl& program, std::uint32_t base_E,
                                    std::uint32_t extent, std::int32_t accepted, double seconds,
                                    std::span<const TokenId> drafts,
                                    std::span<const TokenId> ledger,
                                    std::span<const TokenId> row_tokens) {
    log_round(program, 'm', base_E, extent, accepted, seconds);
    if (replay_record != nullptr) {
        std::fprintf(replay_record, "win %u %u %d", base_E, extent, accepted);
        for (const TokenId draft : drafts) { std::fprintf(replay_record, " %d", draft); }
        std::fputc('\n', replay_record);
    }
    if (replay) {
        if (const execution::Qwen4ReplaySequence* ref = replay->match(ledger)) {
            for (std::size_t j = 0; j < row_tokens.size(); ++j) {
                const std::size_t at = ledger.size() + j;
                if (at >= ref->ids.size() || ref->ids[at] != row_tokens[j]) {
                    ++replay->commit_mismatches;
                    break;
                }
            }
        }
    }
}

void ValidationHooks::write_mtp_calib(std::int32_t extent, std::uint32_t committed,
                                      std::span<const float> draft_logprobs) {
    if (extent > 0) {
        std::fprintf(calib_file, "%d %u", extent, committed - 1U);
        for (const float logprob : draft_logprobs) { std::fprintf(calib_file, " %.5g", logprob); }
        std::fputc('\n', calib_file);
    }
}

void ValidationHooks::on_score_prompt(std::span<const std::int32_t> tokens) {
    if (tfdump) { tfdump->write_tokens(tokens); }
}

void ValidationHooks::on_score_logits(const Tensor& logits,
                                      const std::vector<std::int32_t>& positions) {
    if (tfdump && tfdump->logits_enabled()) {
        std::vector<std::uint16_t> host(static_cast<std::size_t>(logits.numel()));
        CUDA_CHECK(cudaMemcpy(host.data(), logits.data, logits.bytes(), cudaMemcpyDeviceToHost));
        tfdump->write_logits(host, logits.ne[0], positions);
    }
}

void ValidationHooks::log_round(const ProgramImpl& program, char kind, std::uint32_t frontier,
                                std::uint32_t extent, std::int32_t accepted, double round_seconds) {
    if (round_log == nullptr) { return; }
    const auto now    = std::chrono::steady_clock::now();
    const double gap  = round_follows
                            ? std::chrono::duration<double>(now - round_end).count() - round_seconds
                            : -1e-6;
    long long missing = 0, cpu = 0, gpu = 0;
    double wall = 0, fill_wait = 0;
    if (const auto* offload = dynamic_cast<const execution::Qwen4OffloadExperts*>(
            program.qwen4_routed.experts.get())) {
        const execution::Qwen4OffloadWindowStats& w = offload->last_window();
        missing                                     = static_cast<long long>(w.missing);
        cpu                                         = static_cast<long long>(w.host_experts);
        gpu                                         = static_cast<long long>(w.gpu_read_experts);
        wall                                        = w.wall_us;
        fill_wait                                   = w.fill_wait_us;
    }
    const int columns =
        program.qwen4_routed.elastic ? program.qwen4_routed.elastic->stats().columns : -1;
    std::fprintf(round_log, "%c %u %u %d %.1f %.1f %lld %lld %lld %.1f %.1f %d\n", kind, frontier,
                 extent, accepted, 1e6 * round_seconds, 1e6 * gap, missing, cpu, gpu, wall,
                 fill_wait, columns);
    round_end     = std::chrono::steady_clock::now();
    round_follows = true;
}

void ValidationHooks::force_tokens(std::span<const TokenId> ledger, TokenId* tokens,
                                   std::int32_t count) const {
    for (const std::vector<TokenId>& line : forced_tokens) {
        if (line.size() <= ledger.size() ||
            !std::equal(ledger.begin(), ledger.end(), line.begin())) {
            continue;
        }
        for (std::int32_t j = 0; j < count && ledger.size() + j < line.size(); ++j) {
            tokens[j] = line[ledger.size() + static_cast<std::size_t>(j)];
        }
        return;
    }
}

void ValidationHooks::write_top2(const Tensor& logits_column, std::uint32_t lane,
                                 std::uint32_t position, int source) {
    std::vector<std::uint16_t> host(static_cast<std::size_t>(logits_column.ne[0]));
    CUDA_CHECK(cudaMemcpy(host.data(), logits_column.data, host.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));
    const auto value = [](std::uint16_t bits) {
        const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
        float out;
        std::memcpy(&out, &word, sizeof(out));
        return out;
    };
    const std::size_t domain = top2_domain;
    std::size_t best = 0, second = 1;
    if (value(host[second]) > value(host[best])) { std::swap(best, second); }
    for (std::size_t i = 2; i < domain; ++i) {
        const float v = value(host[i]);
        if (v > value(host[best])) {
            second = best;
            best   = i;
        } else if (v > value(host[second])) {
            second = i;
        }
    }
    std::fprintf(top2_file, "%u %u %d %zu %.6g %zu %.6g\n", lane, position, source, best,
                 static_cast<double>(value(host[best])), second,
                 static_cast<double>(value(host[second])));
    std::fflush(top2_file);
}

} // namespace ninfer::models::qwen3_5::detail
