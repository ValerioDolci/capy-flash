// Inkling 2 Text forward driver below the Engine: loads an artifact at tp 1 or tp 2 (plan_load,
// materialize_model, Parameters per rank), sizes the forward's workspace and the per-sequence
// visit state, and drives models/inkling2/execution/forward.h directly.
//
// Modes:
//   score     teacher forcing over the given ids, prefilled in blocks of --block columns; writes a
//             tfdump/1 directory (tools/ref_q4/tfdump.py) with tokens, logits after the soft cap
//             (rows --logit-rows), hidden/embed, hidden/V<vv> (the output of visit vv in execution
//             order, the reference dump's names), hidden/final and, with --parts, the visits'
//             internal points (attn, attn_conv, ain, msg, min, moe_in, moe), for rows --rows.
//   generate  prefill of the prompt in blocks, then greedy decoding one token at a time up to
//             --max-new tokens or --eos; prints the ids and the prefill and decode rates and, with
//             a reference JSON (tools/ref_ink/greedy.py), the first divergence and the reference's
//             top-1/top-2 margin there. With --graph the decode steps replay one CUDA graph of
//             the T = 1 forward and its argmax (both devices, all-reduces through the mailbox),
//             captured after the prefill (its capture time is printed, not counted). With --warmup
//             an untimed prefill of the same prompt runs first, so the timed prefill is a warm one.
//   tp-parity the same ids scored at tp 1 (device 0) and tp 2. The two differ by the BF16
//             rounding of each rank's attention partial before its all-reduce (one per visit),
//             which the 48 visits grow to the size of the BF16 reference's own noise, so the bound
//             is that noise: KL(tp1 || tp2) of the soft-capped logits, mean and max, within
//             --kl-mean / --kl-max (twice the synthetic reference's bf16-to-fp64 floor), and every
//             replicated value of every visit bit-identical on the two ranks. The per-column
//             distance in BF16 ulps (the 2-ulp criterion of test_text_context_tp2.cpp for two
//             layers) is printed for information.
//   blocks    the same ids scored token by token and in blocks of --block: per column the largest
//             logit difference and whether the bits agree; fails beyond two BF16 ulps, or on any
//             differing bit with --identical (blocks of up to 16 columns keep every Op on the
//             route of one column: the BF16 Linear's GEMV below 17 columns).
//   spec      speculative verification rounds (program/speculative/round.h) with fake drafts, or
//             with --drafter the artifact's drafter proposing in the round (acceptance printed):
//             first the greedy of generate (decode graph) keeping every step's BF16 logits; its
//             ids are the truth, or --drafts names a JSON whose generated_ids are (and the greedy
//             must equal them). Then for every K of --draft-tokens and rule of --draft-rule the
//             prompt is prefilled again and decoded by rounds of K drafts from the truth: right
//             (the truth shifted by one: every draft accepted), wrong (every draft rejected) or
//             random (right up to a random j, then one wrong, then right). Fails unless the ids
//             equal the truth, every round licenses the expected count (P + 1, 1, j + 1) and every
//             committed column's logits equal the greedy step's at its position, bit for bit.
//             Prints the rounds' wall time (ms per round, the first round's capture excluded);
//             with --profile-steps N rounds [2, 2 + N) of the first K and rule run eagerly
//             inside the profiler window. --eager runs every round without the graph.
//   drafts    the drafter's seven proposals at every anchor of --ids (prompt + continuation),
//             committed token by token (teacher forcing); --out JSON as tools/ref_ink/draft_ref.py.
//   features  the drafter's feature sinks (forward.h FeatureSink) over the ids in blocks: each
//             rank's features equal its tapped visit inputs and head input bit for bit, the ranks
//             agree and the logits are those of the same blocks without sinks; --out writes them.
// Ids: a whitespace-separated text file, or a reference JSON whose prompt_ids (+ generated_ids
// when scoring) are used. --embedding-host loads every mode's model with the token embedding in
// one mapped page-locked host copy (LoadOptions::embedding_host). Exit 77 without the artifact or
// the CUDA devices.

#include "artifact/reader.h"
#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/device_scope.h"
#include "core/layout.h"
#include "models/inkling2/draft/draft.h"
#include "models/inkling2/draft/parameters.h"
#include "models/inkling2/execution/forward.h"
#include "models/inkling2/execution/parameters.h"
#include "models/inkling2/execution/prefetch_lane.h"
#include "models/inkling2/execution/tp.h"
#include "models/inkling2/load.h"
#include "models/inkling2/program/speculative/round.h"
#include "models/inkling2/state/visit_state.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/peer_mailbox.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace ink  = ninfer::models::inkling2;
namespace exec = ninfer::models::inkling2::execution;
namespace fs   = std::filesystem;

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t word = std::uint32_t(bits) << 16;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

std::vector<float> download(const Tensor& t, cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto n = static_cast<std::size_t>(t.numel());
    std::vector<float> out(n);
    if (t.dtype == DType::FP32) {
        CUDA_CHECK(cudaMemcpy(out.data(), t.data, n * 4, cudaMemcpyDeviceToHost));
    } else if (t.dtype == DType::I32) {
        std::vector<std::int32_t> ids(n);
        CUDA_CHECK(cudaMemcpy(ids.data(), t.data, n * 4, cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < n; ++i) { out[i] = static_cast<float>(ids[i]); }
    } else if (t.dtype == DType::BF16) {
        std::vector<std::uint16_t> bits(n);
        CUDA_CHECK(cudaMemcpy(bits.data(), t.data, n * 2, cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < n; ++i) { out[i] = bf16_to_float(bits[i]); }
    } else {
        throw std::invalid_argument("tap of an unexpected dtype");
    }
    return out;
}

std::vector<std::uint8_t> download_bytes(const Tensor& t, cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::uint8_t> out(t.bytes());
    CUDA_CHECK(cudaMemcpy(out.data(), t.data, out.size(), cudaMemcpyDeviceToHost));
    return out;
}

// .npy v1.0, little-endian, C order.
void write_npy(const fs::path& path, const char* descr, const void* data, std::size_t item,
               const std::vector<std::size_t>& shape) {
    std::string dims;
    std::size_t count = 1;
    for (const auto d : shape) {
        dims += std::to_string(d) + ",";
        count *= d;
    }
    if (shape.size() > 1) { dims.pop_back(); }
    std::string header = "{'descr': '" + std::string(descr) +
                         "', 'fortran_order': False, 'shape': (" + dims + "), }";
    const std::size_t total = 10 + header.size() + 1;
    header.append((64 - total % 64) % 64, ' ');
    header.push_back('\n');
    std::ofstream out(path, std::ios::binary);
    out.write("\x93NUMPY\x01\x00", 8);
    const auto len = static_cast<std::uint16_t>(header.size());
    out.write(reinterpret_cast<const char*>(&len), 2);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(count * item));
    if (!out) { throw std::runtime_error("cannot write " + path.string()); }
}

void write_rows(const fs::path& path, const std::vector<std::int64_t>& rows) {
    write_npy(path, "<i8", rows.data(), 8, {rows.size()});
}

std::vector<std::int32_t> int_array(const std::string& text, const std::string& key) {
    const auto at = text.find("\"" + key + "\"");
    if (at == std::string::npos) { return {}; }
    const auto open  = text.find('[', at);
    const auto close = text.find(']', open);
    std::vector<std::int32_t> out;
    std::stringstream in(text.substr(open + 1, close - open - 1));
    std::string item;
    while (std::getline(in, item, ',')) {
        if (item.find_first_not_of(" \n\t\r") != std::string::npos) {
            out.push_back(std::stoi(item));
        }
    }
    return out;
}

// The reference's per-step margins (tools/ref_ink/greedy.py "steps": top1, logit, margin).
std::vector<double> step_margins(const std::string& text) {
    std::vector<double> out;
    auto at = text.find("\"steps\"");
    if (at == std::string::npos) { return out; }
    const auto end = text.find(']', at);
    while ((at = text.find("\"margin\"", at)) != std::string::npos && at < end) {
        at = text.find(':', at);
        out.push_back(std::stod(text.substr(at + 1)));
    }
    return out;
}

struct Ids {
    std::vector<std::int32_t> prompt, continuation;
    std::vector<double> margins;
};

Ids read_ids(const fs::path& path) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error("cannot read " + path.string()); }
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();
    Ids ids;
    if (path.extension() == ".json") {
        ids.prompt       = int_array(text, "prompt_ids");
        ids.continuation = int_array(text, "generated_ids");
        ids.margins      = step_margins(text);
    } else {
        std::stringstream words(text);
        std::int32_t id = 0;
        while (words >> id) { ids.prompt.push_back(id); }
    }
    if (ids.prompt.empty()) { throw std::runtime_error(path.string() + ": no ids"); }
    return ids;
}

// Positions (sequence indices) a spec selects: "all" or comma-separated a, a-b, negative from
// the end ("0-3,-28--1").
std::set<std::int64_t> select_rows(const std::string& spec, std::int64_t length) {
    std::set<std::int64_t> out;
    if (spec == "all") {
        for (std::int64_t i = 0; i < length; ++i) { out.insert(i); }
        return out;
    }
    std::stringstream in(spec);
    std::string item;
    auto resolve = [&](std::int64_t v) { return v < 0 ? length + v : v; };
    while (std::getline(in, item, ',')) {
        const auto dash = item.find('-', 1);
        std::int64_t a = std::stoll(item.substr(0, dash)), b = a;
        if (dash != std::string::npos) { b = std::stoll(item.substr(dash + 1)); }
        for (std::int64_t i = resolve(a); i <= resolve(b); ++i) {
            if (i >= 0 && i < length) { out.insert(i); }
        }
    }
    return out;
}

std::string point_name(std::int32_t visit, std::string_view part) {
    if (visit == exec::ForwardTap::kHeadTap) { return std::string(part); }
    char name[16];
    std::snprintf(name, sizeof(name), "V%02d", visit);
    return part.empty() ? std::string(name) : std::string(name) + "." + std::string(part);
}

// Collects the wanted rows of the wanted points on the host (rank 0's copy); at tp 2 counts the
// replicated points whose two rank copies differ.
class DumpTap final : public exec::ForwardTap {
public:
    DumpTap(std::set<std::string> parts, std::set<std::int64_t> rows, bool compare)
        : parts_(std::move(parts)), rows_(std::move(rows)), compare_(compare) {}

    bool wants(std::int32_t visit, std::string_view part) const override {
        if (visit == kHeadTap) { return true; }
        return parts_.count(std::string(part)) != 0;
    }

    void capture(std::int32_t visit, std::string_view part, const Tensor& value,
                 std::int32_t first_position, cudaStream_t stream) override {
        const auto host = download(value, stream);
        const auto H    = static_cast<std::size_t>(value.ne[0]);
        const auto T    = static_cast<std::size_t>(value.ne[1]);
        const auto Q    = static_cast<std::size_t>(value.ne[2]);
        auto& store     = points_[point_name(visit, part)];
        for (std::size_t t = 0; t < T; ++t) {
            const std::int64_t position = first_position + static_cast<std::int64_t>(t);
            if (rows_.count(position) == 0) { continue; }
            std::vector<float> row(H * Q);
            for (std::size_t q = 0; q < Q; ++q) {
                std::memcpy(&row[q * H], &host[(q * T + t) * H], H * sizeof(float));
            }
            store[position] = std::move(row);
        }
    }

    void compare_ranks(std::int32_t visit, std::string_view part, const Tensor& rank0,
                       const Tensor& rank1, cudaStream_t stream0, cudaStream_t stream1) override {
        if (!compare_) { return; }
        const auto a = download_bytes(rank0, stream0);
        std::vector<std::uint8_t> b;
        {
            int device0 = 0;
            CUDA_CHECK(cudaGetDevice(&device0));
            CUDA_CHECK(cudaSetDevice(1));
            b = download_bytes(rank1, stream1);
            CUDA_CHECK(cudaSetDevice(device0));
        }
        ++compared_;
        if (a != b) {
            ++differing_;
            if (differing_ <= 8) {
                std::printf("rank copies differ at %s\n", point_name(visit, part).c_str());
            }
        }
    }

    void save(const fs::path& directory) const {
        fs::create_directories(directory / "hidden");
        for (const auto& [name, store] : points_) {
            if (store.empty()) { continue; }
            if (name.ends_with(".route_ids")) {
                save_route(directory, name.substr(0, name.size() - 10), store);
                continue;
            }
            if (name.ends_with(".route_weights")) { continue; }
            std::vector<std::int64_t> rows;
            std::vector<float> data;
            for (const auto& [position, row] : store) {
                rows.push_back(position);
                data.insert(data.end(), row.begin(), row.end());
            }
            const auto width = store.begin()->second.size();
            write_npy(directory / "hidden" / (name + ".npy"), "<f4", data.data(), 4,
                      {rows.size(), width});
            write_rows(directory / "hidden" / (name + ".rows.npy"), rows);
        }
    }

    // route/V<vv>.idx.npy (I32 [R,k], ascending ids) and .w.npy (FP32 [R,k], the same order), the
    // reference dump's routing files.
    void save_route(const fs::path& directory, const std::string& visit,
                    const std::map<std::int64_t, std::vector<float>>& ids) const {
        const auto& weights = points_.at(visit + ".route_weights");
        fs::create_directories(directory / "route");
        std::vector<std::int64_t> rows;
        std::vector<std::int32_t> idx;
        std::vector<float> w;
        for (const auto& [position, row] : ids) {
            rows.push_back(position);
            const auto& wrow = weights.at(position);
            std::vector<std::size_t> order(row.size());
            for (std::size_t i = 0; i < order.size(); ++i) { order[i] = i; }
            std::sort(order.begin(), order.end(),
                      [&](std::size_t a, std::size_t b) { return row[a] < row[b]; });
            for (const auto i : order) {
                idx.push_back(static_cast<std::int32_t>(row[i]));
                w.push_back(wrow[i]);
            }
        }
        const auto k = ids.begin()->second.size();
        write_npy(directory / "route" / (visit + ".idx.npy"), "<i4", idx.data(), 4,
                  {rows.size(), k});
        write_npy(directory / "route" / (visit + ".w.npy"), "<f4", w.data(), 4, {rows.size(), k});
        write_rows(directory / "route" / "rows.npy", rows);
    }

    [[nodiscard]] std::size_t compared() const { return compared_; }

    [[nodiscard]] std::size_t differing() const { return differing_; }

private:
    std::set<std::string> parts_;
    std::set<std::int64_t> rows_;
    bool compare_;
    std::map<std::string, std::map<std::int64_t, std::vector<float>>> points_;
    std::size_t compared_ = 0, differing_ = 0;
};

// --embedding-host, for every Session of the run.
bool embedding_host = false;

std::uint64_t used_bytes(int device) {
    const ScopedCurrentDevice scope(device);
    std::size_t free = 0, total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return total - free;
}

// One loaded model with its forward, workspace and the state of one sequence.
class Session {
public:
    Session(const fs::path& artifact, int tp, std::int32_t max_block, std::int32_t max_logits,
            std::uint32_t max_context, bool drafter = false)
        : tp_(tp), max_block_(max_block), max_logits_(max_logits) {
        std::vector<int> devices;
        for (int d = 0; d < tp; ++d) { devices.push_back(d); }
        execution_.emplace(devices);
        if (tp == 2) {
            peer_ = ops::enable_peer_access(*execution_);
            events_.emplace(*execution_);
            if (!peer_) {
                exchange_.emplace(*execution_, std::size_t(1) << 20);
                events_->attach_host_exchange(&*exchange_);
            }
        }
        reader_.emplace(artifact);
        models::LoadOptions options{};
        options.tp             = tp;
        options.speculative    = drafter ? SpeculativeBackend::Mtp : SpeculativeBackend::None;
        options.embedding_host = embedding_host;
        auto plan              = ink::plan_load(*reader_, options);
        model_                 = ink::materialize_model(std::move(plan), *execution_);
        config_                = model_->config().text;
        rank_config_           = tp == 2 ? exec::shard_text_config(config_, 2) : config_;
        for (int r = 0; r < tp; ++r) {
            const ScopedCurrentDevice scope(r);
            parameters_[r] = std::make_unique<exec::Parameters>(*model_, r);
        }
        std::array<exec::ForwardRank<WorkspaceArena>, 2> plan_ranks{};
        for (int r = 0; r < tp; ++r) {
            plan_ranks[r].config     = &rank_config_;
            plan_ranks[r].parameters = parameters_[r].get();
            plan_ranks[r].device     = r;
        }
        workspace_bytes_ =
            exec::forward_workspace_bytes(config_, plan_ranks, tp, max_block, max_logits);
        for (int r = 0; r < tp; ++r) {
            const ScopedCurrentDevice scope(r);
            LayoutBuilder builder;
            layout_[r]  = ink::plan_visit_state(builder, rank_config_, max_context);
            backing_[r] = std::make_unique<DeviceBuffer>(layout_[r].bytes);
            state_[r]   = std::make_unique<ink::VisitState>(
                DeviceSpan{backing_[r]->p, backing_[r]->bytes}, layout_[r]);
            work_[r]      = std::make_unique<WorkspaceArena>(workspace_bytes_);
            ids_[r]       = std::make_unique<DeviceBuffer>(std::size_t(max_block) * 4);
            positions_[r] = std::make_unique<DeviceBuffer>(std::size_t(max_block) * 4);
            prefetch_[r]  = std::make_unique<exec::PrefetchLane>(r);
        }
        {
            const ScopedCurrentDevice scope(0);
            const auto V = static_cast<std::int32_t>(config_.unpadded_vocab_size);
            logits_mem_  = std::make_unique<DeviceBuffer>(std::size_t(V) * max_logits * 2);
            capped_mem_  = std::make_unique<DeviceBuffer>(std::size_t(V) * max_logits * 4);
            argmax_mem_  = std::make_unique<DeviceBuffer>(4);
        }
        if (drafter) {
            const auto& draft = *model_->config().draft;
            draft_config_     = ink::draft::shard_draft_config(draft, tp);
            std::array<ink::draft::DraftRank, 2> ranks{};
            for (int r = 0; r < tp; ++r) {
                const ScopedCurrentDevice scope(r);
                draft_parameters_[r] =
                    std::make_unique<ink::draft::Parameters>(*model_, r, draft_config_);
                ranks[r] = {&draft_config_, draft_parameters_[r].get(), parameters_[r].get(),
                            stream(r), r};
            }
            if (tp == 2) { transport(); }
            drafter_ = std::make_unique<ink::draft::Drafter>(
                draft, config_, ranks, tp,
                exec::ForwardCollectives{&*execution_, events_ ? &*events_ : nullptr},
                std::max<std::int32_t>(max_block, std::int32_t(draft.block_size)), max_context);
        }
    }

    [[nodiscard]] ink::draft::Drafter* drafter() const { return drafter_.get(); }

    // A prefill block that also feeds the drafter's context (its feature sinks, then append).
    void step_drafted(const std::vector<std::int32_t>& ids) {
        const auto T = static_cast<std::int32_t>(ids.size());
        std::array<exec::FeatureSink, 2> sinks{};
        for (int r = 0; r < tp_; ++r) { sinks[r] = *drafter_->sink(r, T); }
        const std::int32_t first = position_;
        step(ids, 1, false, nullptr, &sinks);
        drafter_->append(first, T);
    }

    [[nodiscard]] const ink::TextConfig& config() const { return config_; }

    [[nodiscard]] std::size_t workspace_bytes() const { return workspace_bytes_; }

    [[nodiscard]] std::size_t state_bytes() const { return layout_[0].bytes; }

    [[nodiscard]] std::size_t workspace_peak(int r) const { return work_[r]->peak_used(); }

    [[nodiscard]] bool peer() const { return peer_; }

    void reset() {
        for (int r = 0; r < tp_; ++r) {
            const ScopedCurrentDevice scope(r);
            state_[r]->reset(stream(r));
            CUDA_CHECK(cudaStreamSynchronize(stream(r)));
        }
        position_ = 0;
    }

    // Runs `ids` (at most max_block) at the current position; the logits of the last `logits`
    // columns land in logits() / capped() on rank 0.
    void step(const std::vector<std::int32_t>& ids, std::int32_t logits, bool capped,
              exec::ForwardTap* tap, const std::array<exec::FeatureSink, 2>* sinks = nullptr) {
        const auto T = static_cast<std::int32_t>(ids.size());
        upload(ids);
        auto ranks = forward_ranks(T);
        for (int r = 0; sinks != nullptr && r < tp_; ++r) { ranks[r].features = &(*sinks)[r]; }
        exec::Forward<WorkspaceArena> forward(config_, ranks, tp_,
                                              {&*execution_, events_ ? &*events_ : nullptr},
                                              max_block_, max_logits_);
        bind_logits(logits);
        const ScopedCurrentDevice scope(0);
        forward.run(ids_tensors(T), exec::ForwardColumns{T, position_, logits},
                    exec::ForwardLogits{&logits_, capped ? &capped_ : nullptr}, tap);
        synchronize();
        position_ += T;
    }

    // Captures one decode step (T = 1, its logits and the argmax on rank 0) into one graph over
    // both devices, with the mailbox as the transport of its all-reduces. The forward reads the
    // token and its position from the fixed device buffers, so the graph replays at any position.
    void capture_decode() {
        transport();
        auto ranks = forward_ranks(1);
        exec::Forward<WorkspaceArena> forward(config_, ranks, tp_,
                                              {&*execution_, events_ ? &*events_ : nullptr},
                                              max_block_, max_logits_);
        bind_logits(1);
        synchronize();
        const ScopedCurrentDevice scope(0);
        const auto ids = ids_tensors(1);
        auto body      = [&] {
            forward.run(ids, exec::ForwardColumns{1, position_, 1},
                             exec::ForwardLogits{&logits_, nullptr}, nullptr);
            Tensor out(argmax_mem_->p, DType::I32, {1});
            ops::argmax(logits_, out, logits_.ne[0], stream(0));
        };
        if (tp_ == 2) {
            definition_.capture(stream(0), body, DecodeGraphPeerCapture{&*bridge_, stream(1)});
        } else {
            definition_.capture(stream(0), body);
        }
        graph_.instantiate(definition_);
        graph_.upload(stream(0));
        synchronize();
        std::printf("decode graph: %zu nodes over %d device(s)\n", definition_.node_count(), tp_);
    }

    // The two-device transport of the captured graphs, created once: the mailbox (ordinary slots
    // for the decode step's exchanges, wide ones for a verification round of up to kMaxDrafts
    // drafts) and the bridge.
    void transport() {
        if (tp_ != 2 || mailbox_) { return; }
        mailbox_.emplace(
            *execution_, std::size_t{16} << 10, 2,
            ink::speculative::round_exchange_bytes(config_, ink::speculative::kMaxDrafts));
        // Every kernel that follows an exchange of the forward waits for its dependency before it
        // reads the stream's data, so an exchange may release it at entry: it launches while the
        // exchange waits on the peer.
        mailbox_->set_release_dependents(true);
        events_->attach_mailbox(&*mailbox_);
        bridge_.emplace(0, 1);
    }

    // A verification round of `drafts` drafts over this session's state (round.h).
    [[nodiscard]] std::unique_ptr<ink::speculative::VerifyRound> round(std::uint32_t drafts,
                                                                       bool drafted = true) {
        transport();
        std::array<ink::speculative::RoundRank, 2> ranks{};
        for (int r = 0; r < tp_; ++r) {
            ranks[r] = {.config     = &rank_config_,
                        .parameters = parameters_[r].get(),
                        .work       = work_[r].get(),
                        .stream     = stream(r),
                        .device     = r,
                        .state      = state_[r].get(),
                        .prefetch   = prefetch_[r].get()};
        }
        return std::make_unique<ink::speculative::VerifyRound>(
            config_, ranks, tp_,
            exec::ForwardCollectives{&*execution_, events_ ? &*events_ : nullptr},
            bridge_ ? &*bridge_ : nullptr, drafts, max_block_, max_logits_,
            drafted ? drafter_.get() : nullptr);
    }

    [[nodiscard]] std::int32_t position() const { return position_; }

    // A round committed `count` tokens.
    void advance(std::int32_t count) { position_ += count; }

    // Rank 0's BF16 logits of the last step, as stored bits.
    [[nodiscard]] std::vector<std::uint8_t> logits_bits() const {
        const ScopedCurrentDevice scope(0);
        return download_bytes(logits_, stream(0));
    }

    // One captured decode step: token `id` at the current position; returns the argmax.
    [[nodiscard]] std::int32_t replay(std::int32_t id) {
        upload({id});
        const ScopedCurrentDevice scope(0);
        if (tp_ == 2) { bridge_->gate_launch(stream(1), stream(0)); }
        graph_.launch(stream(0));
        if (tp_ == 2) { bridge_->gate_peer_after_launch(stream(1), stream(0)); }
        std::int32_t next = 0;
        CUDA_CHECK(cudaStreamSynchronize(stream(0)));
        CUDA_CHECK(cudaMemcpy(&next, argmax_mem_->p, 4, cudaMemcpyDeviceToHost));
        synchronize();
        ++position_;
        return next;
    }

    [[nodiscard]] std::int32_t argmax_last() {
        const ScopedCurrentDevice scope(0);
        Tensor last = logits_.slice(1, logits_.ne[1] - 1, 1).view({logits_.ne[0]});
        Tensor out(argmax_mem_->p, DType::I32, {1});
        ops::argmax(last.view({logits_.ne[0], 1}), out, logits_.ne[0], stream(0));
        std::int32_t id = 0;
        CUDA_CHECK(cudaStreamSynchronize(stream(0)));
        CUDA_CHECK(cudaMemcpy(&id, argmax_mem_->p, 4, cudaMemcpyDeviceToHost));
        return id;
    }

    [[nodiscard]] std::vector<float> logits_host() const {
        const ScopedCurrentDevice scope(0);
        return download(logits_, stream(0));
    }

    [[nodiscard]] std::vector<float> capped_host() const {
        const ScopedCurrentDevice scope(0);
        return download(capped_, stream(0));
    }

    [[nodiscard]] cudaStream_t stream(int r) const { return execution_->dev[r]->stream; }

private:
    // Uploads the call's ids and positions into every rank's fixed buffers (streams idle first).
    void upload(const std::vector<std::int32_t>& ids) {
        std::vector<std::int32_t> positions(ids.size());
        for (std::size_t t = 0; t < ids.size(); ++t) {
            positions[t] = position_ + static_cast<std::int32_t>(t);
        }
        for (int r = 0; r < tp_; ++r) {
            const ScopedCurrentDevice scope(r);
            CUDA_CHECK(cudaStreamSynchronize(stream(r)));
            ids_[r]->copy_from_host(ids.data(), ids.size() * 4);
            positions_[r]->copy_from_host(positions.data(), positions.size() * 4);
        }
    }

    [[nodiscard]] std::array<Tensor, 2> ids_tensors(std::int32_t T) const {
        std::array<Tensor, 2> out;
        for (int r = 0; r < tp_; ++r) { out[r] = Tensor(ids_[r]->p, DType::I32, {T}); }
        return out;
    }

    std::array<exec::ForwardRank<WorkspaceArena>, 2> forward_ranks(std::int32_t T) {
        std::array<exec::ForwardRank<WorkspaceArena>, 2> ranks{};
        for (int r = 0; r < tp_; ++r) {
            position_tensors_[r] = Tensor(positions_[r]->p, DType::I32, {T});
            ranks[r].config      = &rank_config_;
            ranks[r].parameters  = parameters_[r].get();
            ranks[r].work        = work_[r].get();
            ranks[r].stream      = stream(r);
            ranks[r].device      = r;
            ranks[r].state       = state_[r].get();
            ranks[r].positions   = &position_tensors_[r];
            ranks[r].prefetch    = prefetch_[r].get();
        }
        return ranks;
    }

    void bind_logits(std::int32_t logits) {
        const auto V = static_cast<std::int32_t>(config_.unpadded_vocab_size);
        logits_      = Tensor(logits_mem_->p, DType::BF16, {V, std::max(logits, 1)});
        capped_      = Tensor(capped_mem_->p, DType::FP32, {V, std::max(logits, 1)});
    }

    void synchronize() const {
        for (int r = 0; r < tp_; ++r) {
            const ScopedCurrentDevice scope(r);
            CUDA_CHECK(cudaStreamSynchronize(stream(r)));
        }
    }

    int tp_;
    std::int32_t max_block_, max_logits_;
    bool peer_ = false;
    std::optional<ExecutionContext> execution_;
    std::optional<ops::PeerEvents> events_;
    std::optional<ops::HostExchange> exchange_;
    std::optional<artifact::Reader> reader_;
    std::unique_ptr<ink::Model> model_;
    ink::TextConfig config_, rank_config_;
    std::array<std::unique_ptr<exec::Parameters>, 2> parameters_;
    std::size_t workspace_bytes_ = 0;
    std::array<ink::VisitStateLayout, 2> layout_;
    std::array<std::unique_ptr<DeviceBuffer>, 2> backing_, ids_, positions_;
    std::array<std::unique_ptr<ink::VisitState>, 2> state_;
    std::array<std::unique_ptr<WorkspaceArena>, 2> work_;
    std::array<std::unique_ptr<exec::PrefetchLane>, 2> prefetch_;
    std::unique_ptr<DeviceBuffer> logits_mem_, capped_mem_, argmax_mem_;
    Tensor logits_, capped_;
    std::array<Tensor, 2> position_tensors_;
    std::int32_t position_ = 0;
    // The captured decode step (capture_decode).
    std::optional<ops::PeerMailbox> mailbox_;
    std::optional<DecodeGraphPeerBridge> bridge_;
    DecodeGraphDefinition definition_;
    DecodeGraphExecutable graph_;
    // The drafter (--drafter).
    ink::draft::DraftConfig draft_config_;
    std::array<std::unique_ptr<ink::draft::Parameters>, 2> draft_parameters_;
    std::unique_ptr<ink::draft::Drafter> drafter_;
};

struct Options {
    fs::path artifact, ids, out;
    std::string mode   = "score";
    int tp             = 2;
    std::int32_t block = 64;
    std::string rows = "all", logit_rows = "all", parts;
    std::int32_t max_new     = 128;
    std::int32_t eos         = 200006;
    std::uint32_t context    = 0;
    std::int32_t first_block = 0;   // score: width of the first block (then --block)
    std::int32_t dump_step   = -1;  // generate, eager: BF16 logits of this step to <out>.step.npy
    double kl_mean = 0, kl_max = 0; // tp-parity bounds
    bool identical = false;         // blocks: every column bit-identical
    // generate: the profiler window (cudaProfilerStart/Stop, for nsys
    // --capture-range=cudaProfilerApi) around prefill block `profile_block` or decode steps [2, 2 +
    // profile_steps).
    std::int32_t profile_block = -1;
    std::int32_t profile_steps = 0;
    bool graph                 = false; // generate: decode steps replay one captured graph
    bool warmup                = false; // generate: one untimed prefill before the timed one
    // spec: the truth's JSON (generated_ids), the draft counts and rules, rounds without graph.
    fs::path drafts;
    std::string draft_tokens = "3";
    std::string draft_rules  = "right";
    bool eager               = false;
    bool drafter             = false; // spec: the artifact's drafter proposes the drafts
};

using Clock = std::chrono::steady_clock;

double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// Scores `sequence` in blocks; returns the BF16 logits (as floats) of every column, [T][V].
std::vector<std::vector<float>> score_bf16(Session& session, const std::vector<std::int32_t>& seq,
                                           std::int32_t block) {
    session.reset();
    std::vector<std::vector<float>> out;
    const auto V = session.config().unpadded_vocab_size;
    for (std::size_t first = 0; first < seq.size(); first += std::size_t(block)) {
        const std::vector<std::int32_t> ids(
            seq.begin() + static_cast<std::ptrdiff_t>(first),
            seq.begin() + static_cast<std::ptrdiff_t>(std::min(seq.size(), first + block)));
        const auto n = static_cast<std::int32_t>(ids.size());
        session.step(ids, n, false, nullptr);
        const auto host = session.logits_host();
        for (std::int32_t t = 0; t < n; ++t) {
            out.emplace_back(host.begin() + std::ptrdiff_t(t) * V,
                             host.begin() + std::ptrdiff_t(t + 1) * V);
        }
    }
    return out;
}

float bf16_ulp(float magnitude) {
    if (magnitude <= 0 || !std::isfinite(magnitude)) { return 0; }
    return std::ldexp(1.0F, std::ilogb(magnitude) - 7);
}

// Per column: max |a - b| within two BF16 ulps of the column's largest |a|; argmax equal up to it.
int compare_columns(const std::vector<std::vector<float>>& a,
                    const std::vector<std::vector<float>>& b, const char* label) {
    int failures          = 0;
    std::size_t identical = 0;
    float worst_ratio     = 0;
    for (std::size_t t = 0; t < a.size(); ++t) {
        float largest = 0, worst = 0;
        bool same = true;
        for (std::size_t i = 0; i < a[t].size(); ++i) {
            if (!std::isfinite(a[t][i]) || !std::isfinite(b[t][i])) {
                ++failures;
                break;
            }
            largest = std::max(largest, std::abs(a[t][i]));
            worst   = std::max(worst, std::abs(a[t][i] - b[t][i]));
            same    = same && a[t][i] == b[t][i];
        }
        identical += same;
        const float tolerance = 2.0F * bf16_ulp(largest);
        worst_ratio           = std::max(worst_ratio, tolerance > 0 ? worst / tolerance : 0.0F);
        const auto ia         = std::max_element(a[t].begin(), a[t].end()) - a[t].begin();
        const auto ib         = std::max_element(b[t].begin(), b[t].end()) - b[t].begin();
        if (worst > tolerance ||
            (ia != ib && a[t][std::size_t(ia)] - a[t][std::size_t(ib)] > tolerance)) {
            if (failures < 8) {
                std::printf("%s: column %zu max |diff| %g > tolerance %g (argmax %td / %td)\n",
                            label, t, double(worst), double(tolerance), ia, ib);
            }
            ++failures;
        }
    }
    std::printf("%s: %zu columns, %zu bit-identical, worst |diff| / (2 BF16 ulps) = %.3f, %d "
                "failing\n",
                label, a.size(), identical, double(worst_ratio), failures);
    return failures;
}

// KL(a || b) of the soft-capped distributions of every column, in FP64: mean and max.
std::pair<double, double> kl_columns(const std::vector<std::vector<float>>& a,
                                     const std::vector<std::vector<float>>& b, float cap) {
    double sum = 0, worst = 0;
    for (std::size_t t = 0; t < a.size(); ++t) {
        const auto n = a[t].size();
        std::vector<double> la(n), lb(n);
        double ma = -1e300, mb = -1e300;
        for (std::size_t i = 0; i < n; ++i) {
            la[i] = cap * std::tanh(double(a[t][i]) / cap);
            lb[i] = cap * std::tanh(double(b[t][i]) / cap);
            ma    = std::max(ma, la[i]);
            mb    = std::max(mb, lb[i]);
        }
        double za = 0, zb = 0;
        for (std::size_t i = 0; i < n; ++i) {
            za += std::exp(la[i] - ma);
            zb += std::exp(lb[i] - mb);
        }
        double kl = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const double pa = std::exp(la[i] - ma) / za;
            kl += pa * ((la[i] - ma - std::log(za)) - (lb[i] - mb - std::log(zb)));
        }
        sum += kl;
        worst = std::max(worst, kl);
    }
    return {sum / double(a.size()), worst};
}

int run_score(const Options& o) {
    const Ids ids                 = read_ids(o.ids);
    std::vector<std::int32_t> seq = ids.prompt;
    seq.insert(seq.end(), ids.continuation.begin(), ids.continuation.end());
    const auto length = static_cast<std::int64_t>(seq.size());
    const auto block  = std::min<std::int32_t>(o.block, static_cast<std::int32_t>(length));
    const auto widest = std::max<std::int32_t>(block, o.first_block);
    Session session(o.artifact, o.tp, widest, widest,
                    o.context ? o.context : static_cast<std::uint32_t>(length));
    std::set<std::string> parts{""};
    std::stringstream list(o.parts);
    std::string part;
    while (std::getline(list, part, ',')) {
        if (!part.empty()) { parts.insert(part); }
    }
    DumpTap tap(parts, select_rows(o.rows, length), o.tp == 2);
    const auto logit_rows = select_rows(o.logit_rows, length);
    const auto V          = session.config().unpadded_vocab_size;
    std::vector<std::int64_t> rows;
    std::vector<float> logits;
    session.reset();
    double busy = 0;
    for (std::int64_t first = 0, width = o.first_block > 0 ? o.first_block : block; first < length;
         first += width, width         = block) {
        const std::vector<std::int32_t> chunk(
            seq.begin() + first, seq.begin() + std::min<std::int64_t>(length, first + width));
        const auto n  = static_cast<std::int32_t>(chunk.size());
        const auto t0 = Clock::now();
        session.step(chunk, n, true, &tap);
        busy += seconds(t0, Clock::now());
        const auto host = session.capped_host();
        for (std::int32_t t = 0; t < n; ++t) {
            if (logit_rows.count(first + t) == 0) { continue; }
            rows.push_back(first + t);
            logits.insert(logits.end(), host.begin() + std::ptrdiff_t(t) * V,
                          host.begin() + std::ptrdiff_t(t + 1) * V);
        }
    }
    fs::create_directories(o.out);
    tap.save(o.out);
    std::vector<std::int64_t> tokens(seq.begin(), seq.end());
    write_npy(o.out / "tokens.npy", "<i8", tokens.data(), 8, {tokens.size()});
    write_npy(o.out / "logits.npy", "<f4", logits.data(), 4, {rows.size(), std::size_t(V)});
    write_rows(o.out / "rows.npy", rows);
    std::ofstream meta(o.out / "meta.json");
    meta << "{\"format\": \"tfdump/1\", \"producer\": \"ninfer_inkling2_tf_driver\", \"tp\": "
         << o.tp << ", \"block\": " << block << ", \"T\": " << length << ", \"artifact\": \""
         << o.artifact.string() << "\"}\n";
    std::printf("score: %lld tokens, blocks of %d, tp %d, %.2f s (with taps)\n",
                static_cast<long long>(length), block, o.tp, busy);
    std::printf("workspace %.1f MiB per rank (peak used %.1f MiB), state %.1f MiB per rank\n",
                double(session.workspace_bytes()) / (1 << 20),
                double(session.workspace_peak(0)) / (1 << 20),
                double(session.state_bytes()) / (1 << 20));
    if (o.tp == 2) {
        std::printf("rank parity: %zu replicated points compared, %zu differing\n", tap.compared(),
                    tap.differing());
        return tap.differing() == 0 ? 0 : 1;
    }
    return 0;
}

int run_generate(const Options& o) {
    const Ids ids     = read_ids(o.ids);
    const auto prompt = static_cast<std::int32_t>(ids.prompt.size());
    const auto block  = std::min(o.block, prompt);
    Session session(o.artifact, o.tp, block, 1,
                    o.context ? o.context : static_cast<std::uint32_t>(prompt + o.max_new));
    if (o.warmup) {
        // One untimed prefill of the same prompt (the process's first kernel launches load their
        // modules), then a fresh sequence: the timed prefill is a warm one, as llama-bench
        // measures.
        session.reset();
        for (std::int32_t first = 0; first < prompt; first += block) {
            session.step(
                {ids.prompt.begin() + first, ids.prompt.begin() + std::min(prompt, first + block)},
                1, false, nullptr);
        }
    }
    session.reset();
    const auto t0 = Clock::now();
    for (std::int32_t first = 0, index = 0; first < prompt; first += block, ++index) {
        const std::vector<std::int32_t> chunk(ids.prompt.begin() + first,
                                              ids.prompt.begin() + std::min(prompt, first + block));
        const bool profiled = index == o.profile_block;
        if (profiled) { CUDA_CHECK(cudaProfilerStart()); }
        session.step(chunk, 1, false, nullptr);
        if (profiled) { CUDA_CHECK(cudaProfilerStop()); }
    }
    std::vector<std::int32_t> generated{session.argmax_last()};
    const auto prefilled = Clock::now();
    if (o.graph) { session.capture_decode(); }
    const auto t1 = Clock::now();
    while (static_cast<std::int32_t>(generated.size()) < o.max_new && generated.back() != o.eos) {
        // The profiled decode window: steps [2, 2 + profile_steps) (after a warm step).
        const auto step = static_cast<std::int32_t>(generated.size());
        if (o.profile_steps > 0 && step == 2) { CUDA_CHECK(cudaProfilerStart()); }
        if (o.graph) {
            generated.push_back(session.replay(generated.back()));
        } else {
            session.step({generated.back()}, 1, false, nullptr);
            if (step == o.dump_step && !o.out.empty()) {
                const auto row = session.logits_host();
                write_npy(fs::path(o.out).replace_extension(".step.npy"), "<f4", row.data(), 4,
                          {row.size()});
            }
            generated.push_back(session.argmax_last());
        }
        if (o.profile_steps > 0 && step + 1 == 2 + o.profile_steps) {
            CUDA_CHECK(cudaProfilerStop());
        }
    }
    const auto t2           = Clock::now();
    const double prefill_s  = seconds(t0, prefilled);
    const double decode_s   = seconds(t1, t2);
    const auto decode_steps = static_cast<double>(generated.size() - 1);
    std::printf("generated:");
    for (const auto id : generated) { std::printf(" %d", id); }
    std::printf("\nprefill %d tokens in %.3f s = %.1f tok/s (blocks of %d); decode %zu tokens in "
                "%.3f s = %.2f tok/s\n",
                prompt, prefill_s, prompt / prefill_s, block, generated.size() - 1, decode_s,
                decode_steps > 0 ? decode_steps / decode_s : 0.0);
    for (int r = 0; r < o.tp; ++r) {
        std::printf("rank %d: device memory in use %.2f GiB (workspace %.1f MiB, peak used %.1f "
                    "MiB; state %.1f MiB)\n",
                    r, double(used_bytes(r)) / (1ULL << 30),
                    double(session.workspace_bytes()) / (1 << 20),
                    double(session.workspace_peak(r)) / (1 << 20),
                    double(session.state_bytes()) / (1 << 20));
    }
    if (!ids.continuation.empty()) {
        std::size_t same = 0;
        while (same < generated.size() && same < ids.continuation.size() &&
               generated[same] == ids.continuation[same]) {
            ++same;
        }
        if (same == std::min(generated.size(), ids.continuation.size())) {
            std::printf("reference: identical over %zu tokens\n", same);
        } else {
            std::printf("reference: first divergence at step %zu (engine %d, reference %d), "
                        "reference margin there %.4f\n",
                        same, generated[same], ids.continuation[same],
                        same < ids.margins.size() ? ids.margins[same] : -1.0);
        }
    }
    if (!o.out.empty()) {
        std::ofstream out(o.out);
        out << "{\"generated_ids\": [";
        for (std::size_t i = 0; i < generated.size(); ++i) {
            out << (i ? ", " : "") << generated[i];
        }
        out << "], \"prefill_s\": " << prefill_s << ", \"decode_s\": " << decode_s
            << ", \"prompt\": " << prompt << ", \"block\": " << block << ", \"tp\": " << o.tp
            << "}\n";
    }
    return 0;
}

std::vector<std::string> split_list(const std::string& text) {
    std::vector<std::string> out;
    std::stringstream list(text);
    std::string item;
    while (std::getline(list, item, ',')) {
        if (!item.empty()) { out.push_back(item); }
    }
    return out;
}

// Prefills the prompt in blocks of `block` (the same structure for every pass) and returns the
// first generated token.
std::int32_t prefill_prompt(Session& session, const std::vector<std::int32_t>& prompt,
                            std::int32_t block, bool drafted = false) {
    session.reset();
    const auto length = static_cast<std::int32_t>(prompt.size());
    for (std::int32_t first = 0; first < length; first += block) {
        const std::vector<std::int32_t> ids(prompt.begin() + first,
                                            prompt.begin() + std::min(length, first + block));
        if (drafted) {
            session.step_drafted(ids);
        } else {
            session.step(ids, 1, false, nullptr);
        }
    }
    return session.argmax_last();
}

double median(std::vector<double> values) {
    if (values.empty()) { return 0; }
    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
}

int run_spec(const Options& o) {
    namespace spec    = ink::speculative;
    const Ids ids     = read_ids(o.ids);
    const auto prompt = static_cast<std::int32_t>(ids.prompt.size());
    const auto block  = std::min(o.block, prompt);
    const auto width  = static_cast<std::int32_t>(spec::kMaxWidth);
    Session session(o.artifact, o.tp, std::max(block, width), width,
                    o.context ? o.context : static_cast<std::uint32_t>(prompt + o.max_new + width),
                    o.drafter);
    const std::size_t row_bytes = std::size_t(session.config().unpadded_vocab_size) * 2;
    if (o.drafter) {
        std::printf("drafter: state %.1f MiB, workspace %.1f MiB per rank; device memory in use "
                    "%.2f GiB on rank 0 (context %u)\n",
                    double(session.drafter()->state_bytes()) / (1 << 20),
                    double(session.drafter()->workspace_bytes()) / (1 << 20),
                    double(used_bytes(0)) / (1 << 30),
                    o.context ? o.context : static_cast<std::uint32_t>(prompt + o.max_new + width));
    }

    // The greedy without drafts (decode graph), keeping the BF16 logits of every step.
    std::vector<std::int32_t> greedy{prefill_prompt(session, ids.prompt, block)};
    std::vector<std::vector<std::uint8_t>> step_logits;
    session.capture_decode();
    std::vector<double> step_ms;
    while (static_cast<std::int32_t>(greedy.size()) < o.max_new && greedy.back() != o.eos) {
        const auto t0 = Clock::now();
        greedy.push_back(session.replay(greedy.back()));
        step_ms.push_back(1e3 * seconds(t0, Clock::now()));
        step_logits.push_back(session.logits_bits());
    }
    std::vector<std::int32_t> truth = greedy;
    int failures                    = 0;
    if (!o.drafts.empty()) {
        std::ifstream in(o.drafts);
        std::stringstream text;
        text << in.rdbuf();
        truth           = int_array(text.str(), "generated_ids");
        const bool same = truth == greedy;
        failures += same ? 0 : 1;
        std::printf("greedy against %s: %s\n", o.drafts.filename().c_str(),
                    same ? "identical" : "DIFFERENT");
    }
    const double step_median =
        median({step_ms.begin() + std::min<std::size_t>(1, step_ms.size()), step_ms.end()});
    std::printf("greedy: %zu tokens, decode step %.3f ms (median, graph)\n", greedy.size(),
                step_median);

    const auto V  = static_cast<std::int32_t>(session.config().unpadded_vocab_size);
    const auto N  = static_cast<std::int32_t>(truth.size());
    bool profiled = false;
    for (const std::string& k_text : split_list(o.draft_tokens)) {
        const auto K = static_cast<std::uint32_t>(std::stoul(k_text));
        auto round   = session.round(K, false);
        std::unique_ptr<spec::VerifyRound> drafted;
        if (o.drafter) { drafted = session.round(K); }
        std::vector<std::string> rules = split_list(o.draft_rules);
        if (o.drafter && std::find(rules.begin(), rules.end(), "drafter") == rules.end()) {
            rules = {"drafter"};
        }
        for (const std::string& rule : rules) {
            if (rule == "drafter" && !o.drafter) {
                throw std::invalid_argument("the drafter rule needs --drafter");
            }
            spec::VerifyRound* round_of_rule = rule == "drafter" ? drafted.get() : round.get();
            if (rule != "right" && rule != "wrong" && rule != "random" && rule != "drafter") {
                throw std::invalid_argument("unknown draft rule " + rule);
            }
            std::mt19937 random(1234U + K);
            std::vector<std::int32_t> out{
                prefill_prompt(session, ids.prompt, block, rule == "drafter")};
            std::vector<std::size_t> accepted_at(K + 1, 0); // rounds by accepted drafts
            std::vector<double> drafter_ms;
            std::vector<double> round_ms;
            std::size_t rounds = 0, wrong_counts = 0, columns = 0, identical_columns = 0;
            const bool profile = o.profile_steps > 0 && !profiled;
            while (static_cast<std::int32_t>(out.size()) < N && out.back() != o.eos) {
                // out[m] is the anchor; a round licenses at most P + 1 of the N - m - 1 left.
                const auto m   = static_cast<std::int32_t>(out.size()) - 1;
                auto& in       = round_of_rule->ingress();
                in.anchor      = out.back();
                in.frontier    = session.position();
                in.extent      = std::min<std::int32_t>(std::int32_t(K), N - m - 2);
                std::int32_t j = in.extent; // the accepted drafts the rule makes
                if (rule == "wrong") { j = 0; }
                if (rule == "random") {
                    j = std::uniform_int_distribution<std::int32_t>(0, in.extent)(random);
                }
                for (std::int32_t i = 1; i <= std::int32_t(K); ++i) {
                    const std::int32_t right = m + i < N ? truth[std::size_t(m + i)] : truth.back();
                    in.drafts[std::size_t(i - 1)] = i == j + 1 ? (right + 1) % V : right;
                }
                const bool eager_round = o.eager || (profile && rounds >= 2 &&
                                                     rounds < 2 + std::size_t(o.profile_steps));
                if (profile && rounds == 2) { CUDA_CHECK(cudaProfilerStart()); }
                const auto t0 = Clock::now();
                round_of_rule->launch(!eager_round);
                const auto& eg = round_of_rule->egress();
                if (rounds > 0) { round_ms.push_back(1e3 * seconds(t0, Clock::now())); }
                if (profile && rounds + 1 == 2 + std::size_t(o.profile_steps)) {
                    CUDA_CHECK(cudaProfilerStop());
                    profiled = true;
                }
                ++rounds;
                // Every committed column's logits against the greedy step at its position.
                std::vector<std::uint8_t> bits;
                {
                    const ScopedCurrentDevice scope(0);
                    bits = download_bytes(round_of_rule->logits(), session.stream(0));
                }
                for (std::int32_t c = 0; c < eg.count; ++c) {
                    const auto step = static_cast<std::size_t>(m + c);
                    if (step >= step_logits.size()) { continue; }
                    ++columns;
                    identical_columns +=
                        std::equal(step_logits[step].begin(), step_logits[step].end(),
                                   bits.begin() + std::ptrdiff_t(c * row_bytes));
                }
                if (rule == "drafter") {
                    ++accepted_at[std::size_t(eg.accepted)];
                } else if (eg.count != std::min(j, in.extent) + 1) {
                    ++wrong_counts;
                }
                for (std::int32_t c = 0; c < eg.count; ++c) {
                    out.push_back(eg.licensed[std::size_t(c)]);
                    if (out.back() == o.eos) { break; }
                }
                session.advance(eg.count);
            }
            const bool same = out == truth;
            const bool ok   = same && wrong_counts == 0 && identical_columns == columns;
            if (rule == "drafter") {
                // Acceptance by position: rounds whose drafts reached position i, accepted there.
                std::string line;
                std::size_t reached = rounds;
                for (std::uint32_t i = 0; i < K; ++i) {
                    const std::size_t accepted =
                        reached - accepted_at[i]; // rounds that accepted draft i + 1
                    line += " " + std::to_string(accepted) + "/" + std::to_string(reached);
                    reached = accepted;
                }
                std::printf("drafter K=%u accepted per round:", K);
                for (std::uint32_t a = 0; a <= K; ++a) { std::printf(" %zu", accepted_at[a]); }
                std::printf("; accepted/reached by position:%s\n", line.c_str());
            }
            failures += ok ? 0 : 1;
            std::printf("spec K=%u %s: %zu rounds, %zu tokens, %.3f tokens/round, ids %s, counts "
                        "%s (%zu off), logits %zu/%zu columns identical, round %.3f ms (median, "
                        "%s), nodes %zu %s\n",
                        K, rule.c_str(), rounds, out.size() - 1,
                        double(out.size() - 1) / double(std::max<std::size_t>(rounds, 1)),
                        same ? "identical" : "DIFFERENT", wrong_counts ? "WRONG" : "as expected",
                        wrong_counts, identical_columns, columns, median(round_ms),
                        o.eager ? "eager" : "graph", round_of_rule->graph_nodes(),
                        ok ? "OK" : "FAIL");
        }
    }
    return failures == 0 ? 0 : 1;
}

// Every rank's copy, for one call, of the values the drafter's features must equal: the output of
// the visit before each captured visit (its input) and the head input.
class FeatureTap final : public exec::ForwardTap {
public:
    explicit FeatureTap(std::array<std::uint32_t, 3> visits) : visits_(visits) {}

    bool wants(std::int32_t visit, std::string_view part) const override {
        return visit == kHeadTap ? part == "final" : part.empty() && block(visit) >= 0;
    }

    void capture(std::int32_t visit, std::string_view part, const Tensor& value,
                 std::int32_t first_position, cudaStream_t stream) override {
        (void)part, (void)first_position;
        values[0][std::size_t(block(visit))] = download_bytes(value, stream);
    }

    void compare_ranks(std::int32_t visit, std::string_view part, const Tensor& rank0,
                       const Tensor& rank1, cudaStream_t stream0, cudaStream_t stream1) override {
        (void)part, (void)rank0, (void)stream0;
        const ScopedCurrentDevice scope(1);
        values[1][std::size_t(block(visit))] = download_bytes(rank1, stream1);
    }

    std::array<std::array<std::vector<std::uint8_t>, 4>, 2> values;

private:
    [[nodiscard]] int block(std::int32_t visit) const {
        if (visit == kHeadTap) { return 3; }
        for (std::size_t j = 0; j < visits_.size(); ++j) {
            if (std::int32_t(visits_[j]) - 1 == visit) { return static_cast<int>(j); }
        }
        return -1;
    }

    std::array<std::uint32_t, 3> visits_;
};

// The drafter's features (forward.h FeatureSink) over the ids in blocks of --block: every rank's
// four feature blocks must equal, bit for bit, the tapped visit outputs and head input of the same
// call, the ranks' copies must agree, and the logits must equal those of the same blocks without
// sinks. With --out, writes features.npy (FP32 [N, 4, H]: inputs of the captured visits, head
// input) and features.rows.npy.
int run_features(const Options& o) {
    const Ids ids                 = read_ids(o.ids);
    std::vector<std::int32_t> seq = ids.prompt;
    seq.insert(seq.end(), ids.continuation.begin(), ids.continuation.end());
    const auto N     = static_cast<std::int32_t>(seq.size());
    const auto block = std::min(o.block, N);
    Session session(o.artifact, o.tp, block, block, static_cast<std::uint32_t>(N));
    const auto H = static_cast<std::int32_t>(session.config().hidden_size);
    const std::array<std::uint32_t, 3> visits{5, 23, 41};
    std::array<std::unique_ptr<DeviceBuffer>, 2> storage;
    for (int r = 0; r < o.tp; ++r) {
        const ScopedCurrentDevice scope(r);
        storage[r] = std::make_unique<DeviceBuffer>(std::size_t(H) * block * 4 * 2);
    }
    std::vector<float> features(std::size_t(N) * 4 * H);
    std::vector<std::vector<std::uint8_t>> logits;
    int failures = 0;
    session.reset();
    for (std::int32_t first = 0; first < N; first += block) {
        const std::int32_t T = std::min(block, N - first);
        std::array<exec::FeatureSink, 2> sinks{};
        for (int r = 0; r < o.tp; ++r) {
            sinks[r] = {visits, Tensor(storage[r]->p, DType::BF16, {H, T, 4})};
        }
        FeatureTap tap(visits);
        session.step({seq.begin() + first, seq.begin() + first + T}, T, false, &tap, &sinks);
        logits.push_back(session.logits_bits());
        std::array<std::vector<std::uint8_t>, 2> bytes;
        for (int r = 0; r < o.tp; ++r) {
            const ScopedCurrentDevice scope(r);
            bytes[r]                      = download_bytes(sinks[r].features, session.stream(r));
            const std::size_t block_bytes = std::size_t(H) * T * 2;
            for (std::size_t j = 0; j < 4; ++j) {
                const auto& tapped = tap.values[r][j];
                if (tapped.size() != block_bytes ||
                    !std::equal(tapped.begin(), tapped.end(),
                                bytes[r].begin() + std::ptrdiff_t(j * block_bytes))) {
                    std::printf("FAIL rank %d block %zu at [%d, %d): sink differs from the tap\n",
                                r, j, first, first + T);
                    ++failures;
                }
            }
        }
        if (o.tp == 2 && bytes[0] != bytes[1]) {
            std::printf("FAIL rank copies of the features differ at [%d, %d)\n", first, first + T);
            ++failures;
        }
        for (std::int32_t t = 0; t < T; ++t) {
            for (std::int32_t j = 0; j < 4; ++j) {
                for (std::int32_t h = 0; h < H; ++h) {
                    std::uint16_t bits = 0;
                    std::memcpy(&bits, &bytes[0][((std::size_t(j) * T + t) * H + h) * 2], 2);
                    features[(std::size_t(first + t) * 4 + j) * H + h] = bf16_to_float(bits);
                }
            }
        }
    }
    session.reset();
    std::size_t blocks = 0, same = 0;
    for (std::int32_t first = 0; first < N; first += block, ++blocks) {
        const std::int32_t T = std::min(block, N - first);
        session.step({seq.begin() + first, seq.begin() + first + T}, T, false, nullptr);
        same += session.logits_bits() == logits[blocks];
    }
    failures += same == blocks ? 0 : 1;
    if (!o.out.empty()) {
        fs::create_directories(o.out);
        write_npy(o.out / "features.npy", "<f4", features.data(), 4,
                  {std::size_t(N), 4, std::size_t(H)});
        std::vector<std::int64_t> rows(static_cast<std::size_t>(N));
        for (std::int32_t i = 0; i < N; ++i) { rows[std::size_t(i)] = i; }
        write_rows(o.out / "features.rows.npy", rows);
    }
    std::printf("features: %d positions in blocks of %d, visits %u %u %u + head input, logits "
                "with and without sinks %zu/%zu blocks identical: %s\n",
                N, block, visits[0], visits[1], visits[2], same, blocks, failures ? "FAIL" : "OK");
    return failures == 0 ? 0 : 1;
}

// The drafter's proposals at every anchor of a fixed sequence (prompt_ids + generated_ids of --ids,
// e.g. a reference greedy): the prompt is prefilled with the drafter's context, then every position
// m of the continuation is one round of K = 1 with extent 0 at anchor = the sequence's token there,
// which commits exactly that token (teacher forcing) and records the 7 proposals. Writes --out as
// JSON {anchors, drafts, tokens, n_prompt} (the reference's drafts files,
// tools/ref_ink/draft_ref.py).
int run_drafts(const Options& o) {
    namespace spec                = ink::speculative;
    const Ids ids                 = read_ids(o.ids);
    std::vector<std::int32_t> seq = ids.prompt;
    seq.insert(seq.end(), ids.continuation.begin(), ids.continuation.end());
    const auto prompt = static_cast<std::int32_t>(ids.prompt.size());
    const auto N      = static_cast<std::int32_t>(seq.size());
    const auto block  = std::min(o.block, prompt);
    const auto width  = static_cast<std::int32_t>(spec::kMaxWidth);
    Session session(o.artifact, o.tp, std::max(block, width), width,
                    static_cast<std::uint32_t>(N + width), true);
    (void)prefill_prompt(session, ids.prompt, block, true);
    auto round = session.round(1);
    std::string anchors, drafts;
    for (std::int32_t p = prompt; p + 1 < N; ++p) {
        auto& in    = round->ingress();
        in.anchor   = seq[std::size_t(p)];
        in.frontier = session.position();
        in.extent   = 0;
        round->launch(!o.eager);
        const auto& eg = round->egress();
        session.advance(eg.count);
        anchors += (anchors.empty() ? "" : ",") + std::to_string(p);
        drafts += std::string(drafts.empty() ? "" : ",") + "[";
        for (std::size_t i = 0; i < eg.drafts.size(); ++i) {
            drafts += (i ? "," : "") + std::to_string(eg.drafts[i]);
        }
        drafts += "]";
    }
    std::string tokens;
    for (const auto id : seq) { tokens += (tokens.empty() ? "" : ",") + std::to_string(id); }
    std::ofstream out(o.out);
    out << "{\"anchors\": [" << anchors << "], \"drafts\": [" << drafts << "], \"tokens\": ["
        << tokens << "], \"n_prompt\": " << prompt << "}\n";
    std::printf("drafts: %d anchors written to %s\n", N - 1 - prompt, o.out.c_str());
    return out ? 0 : 1;
}

int run_tp_parity(const Options& o) {
    const Ids ids = read_ids(o.ids);
    const auto block =
        std::min<std::int32_t>(o.block, static_cast<std::int32_t>(ids.prompt.size()));
    const auto context = static_cast<std::uint32_t>(ids.prompt.size());
    std::vector<std::vector<float>> one, two;
    float cap = 0;
    {
        Session session(o.artifact, 1, block, block, context);
        cap = session.config().final_logit_softcap;
        one = score_bf16(session, ids.prompt, block);
    }
    std::size_t differing = 0, compared = 0;
    {
        Session session(o.artifact, 2, block, block, context);
        two = score_bf16(session, ids.prompt, block);
        // The replicated values must stay bit-identical on the two ranks.
        DumpTap tap({"", "attn", "attn_conv", "ain", "msg", "min", "moe_in", "moe"}, {}, true);
        session.reset();
        for (std::size_t first = 0; first < ids.prompt.size(); first += std::size_t(block)) {
            const std::vector<std::int32_t> chunk(
                ids.prompt.begin() + static_cast<std::ptrdiff_t>(first),
                ids.prompt.begin() +
                    static_cast<std::ptrdiff_t>(std::min(ids.prompt.size(), first + block)));
            session.step(chunk, 1, false, &tap);
        }
        differing = tap.differing();
        compared  = tap.compared();
    }
    (void)compare_columns(one, two, "tp 2 against tp 1 (informative)");
    const auto [kl_mean, kl_max] = kl_columns(one, two, cap);
    std::printf("tp 2 against tp 1: KL mean %.3e (bound %.3e), max %.3e (bound %.3e); rank "
                "parity %zu points, %zu differing\n",
                kl_mean, o.kl_mean, kl_max, o.kl_max, compared, differing);
    return kl_mean <= o.kl_mean && kl_max <= o.kl_max && differing == 0 && compared > 0 ? 0 : 1;
}

int run_blocks(const Options& o) {
    const Ids ids = read_ids(o.ids);
    const auto block =
        std::min<std::int32_t>(o.block, static_cast<std::int32_t>(ids.prompt.size()));
    Session session(o.artifact, o.tp, block, block, static_cast<std::uint32_t>(ids.prompt.size()));
    const auto tokenwise    = score_bf16(session, ids.prompt, 1);
    const auto blocks       = score_bf16(session, ids.prompt, block);
    const std::string label = "blocks of " + std::to_string(block) + " against token by token";
    const int failures      = compare_columns(tokenwise, blocks, label.c_str());
    if (o.identical) {
        for (std::size_t t = 0; t < tokenwise.size(); ++t) {
            if (tokenwise[t] != blocks[t]) {
                std::printf("column %zu: bits differ (identity required)\n", t);
                return 1;
            }
        }
    }
    return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    try {
        Options o;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) { throw std::invalid_argument(arg + " needs a value"); }
                return argv[++i];
            };
            if (arg == "--artifact") {
                o.artifact = next();
            } else if (arg == "--mode") {
                o.mode = next();
            } else if (arg == "--tp") {
                o.tp = std::stoi(next());
            } else if (arg == "--ids") {
                o.ids = next();
            } else if (arg == "--block") {
                o.block = std::stoi(next());
            } else if (arg == "--out") {
                o.out = next();
            } else if (arg == "--rows") {
                o.rows = next();
            } else if (arg == "--logit-rows") {
                o.logit_rows = next();
            } else if (arg == "--parts") {
                o.parts = next();
            } else if (arg == "--max-new") {
                o.max_new = std::stoi(next());
            } else if (arg == "--eos") {
                o.eos = std::stoi(next());
            } else if (arg == "--kl-mean") {
                o.kl_mean = std::stod(next());
            } else if (arg == "--kl-max") {
                o.kl_max = std::stod(next());
            } else if (arg == "--profile-block") {
                o.profile_block = std::stoi(next());
            } else if (arg == "--profile-steps") {
                o.profile_steps = std::stoi(next());
            } else if (arg == "--warmup") {
                o.warmup = true;
            } else if (arg == "--graph") {
                o.graph = true;
            } else if (arg == "--identical") {
                o.identical = true;
            } else if (arg == "--first-block") {
                o.first_block = std::stoi(next());
            } else if (arg == "--dump-step") {
                o.dump_step = std::stoi(next());
            } else if (arg == "--drafts") {
                o.drafts = next();
            } else if (arg == "--draft-tokens") {
                o.draft_tokens = next();
            } else if (arg == "--draft-rule") {
                o.draft_rules = next();
            } else if (arg == "--drafter") {
                o.drafter = true;
            } else if (arg == "--eager") {
                o.eager = true;
            } else if (arg == "--context") {
                o.context = static_cast<std::uint32_t>(std::stoul(next()));
            } else if (arg == "--embedding-host") {
                embedding_host = true;
            } else {
                std::cerr
                    << "usage: --artifact PATH --ids FILE [--mode score|generate|tp-parity|"
                       "blocks|spec|features|drafts] [--drafter] [--tp 1|2] [--block N] [--out "
                       "PATH] [--rows SPEC] "
                       "[--logit-rows SPEC] [--parts LIST] [--max-new N] [--eos ID] "
                       "[--context N] [--kl-mean X --kl-max X] [--identical] [--profile-block K] "
                       "[--profile-steps N] [--graph] [--drafts JSON] [--draft-tokens K,...] "
                       "[--draft-rule right|wrong|random,...] [--eager] [--embedding-host]\n";
                return 2;
            }
        }
        if (o.artifact.empty() || o.ids.empty()) {
            std::cout << "SKIP: supply --artifact and --ids\n";
            return 77;
        }
        int devices      = 0;
        const int needed = o.mode == "tp-parity" ? 2 : o.tp;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < needed) {
            std::cout << "SKIP: needs " << needed << " CUDA devices\n";
            return 77;
        }
        int rc = 2;
        if (o.mode == "score") {
            if (o.out.empty()) { throw std::invalid_argument("score needs --out"); }
            rc = run_score(o);
        } else if (o.mode == "generate") {
            rc = run_generate(o);
        } else if (o.mode == "tp-parity") {
            rc = run_tp_parity(o);
        } else if (o.mode == "blocks") {
            rc = run_blocks(o);
        } else if (o.mode == "spec") {
            rc = run_spec(o);
        } else if (o.mode == "features") {
            rc = run_features(o);
        } else if (o.mode == "drafts") {
            if (o.out.empty()) { throw std::invalid_argument("drafts needs --out"); }
            rc = run_drafts(o);
        } else {
            throw std::invalid_argument("unknown mode " + o.mode);
        }
        std::cout << (rc == 0 ? "OK" : "FAIL") << " inkling2 driver " << o.mode << '\n';
        return rc;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
