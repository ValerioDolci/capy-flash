// Replays a llama.cpp routing trace (routing.bin of the F0 expert statistics) through the
// expert-cache policy with the window semantics of the offline LRU simulator, and prints one line
// per counted window: "hits accesses missing inserts" (missing = distinct experts absent at window
// start, inserts = slot insertions). These lines were compared with the simulators window by
// window (the simulators are not part of this repository).
//
// routing.bin: int32 header {48, 10, 512}, then packed records {u8 dom, u8 phase, i32 token,
// i32 pos, u16 experts[48][10]} (0xFFFF = not recorded). Layers 0..46 are used (layer 47 is only
// valid in generation).
//
// Streams (lru_sim.streams): concat = records stably sorted by (dom, phase); gen_cold = dom 3
// decode records; gen_after_prefill = dom 3 records in file order, cut where the phase changes,
// windows inside each segment (the last one may be short), only decode windows counted. Other
// streams use whole windows only. Per window and layer, a record contributes its 10 ids unless
// one of them is 0xFFFF.

#include "core/expert_cache_policy.h"
#include "ninfer/types.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer::expert_cache;

constexpr int kLayersStored        = 48;
constexpr int kTopK                = 10;
constexpr int kExperts             = 512;
constexpr int kLayers              = 47;
constexpr std::size_t kRecordBytes = 2 + 4 + 4 + kLayersStored * kTopK * 2;

struct Record {
    std::uint8_t dom   = 0;
    std::uint8_t phase = 0;
    std::uint16_t experts[kLayersStored][kTopK]{};
};

std::vector<Record> load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot open " + path); }
    std::int32_t header[3]{};
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    if (header[0] != kLayersStored || header[1] != kTopK || header[2] != kExperts) {
        throw std::runtime_error("unexpected routing.bin header");
    }
    std::vector<char> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (raw.size() % kRecordBytes != 0) { throw std::runtime_error("truncated routing.bin"); }
    std::vector<Record> records(raw.size() / kRecordBytes);
    for (std::size_t i = 0; i < records.size(); ++i) {
        const char* r    = raw.data() + i * kRecordBytes;
        records[i].dom   = static_cast<std::uint8_t>(r[0]);
        records[i].phase = static_cast<std::uint8_t>(r[1]);
        std::memcpy(records[i].experts, r + 10, sizeof(records[i].experts)); // little-endian host
    }
    return records;
}

struct Window {
    std::vector<std::size_t> records;
    bool counted = true;
};

std::vector<Window> make_windows(const std::vector<Record>& R, const std::string& stream, int W) {
    std::vector<std::size_t> order;
    bool segment = false;
    if (stream == "concat") {
        order.resize(R.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return R[a].dom * 10 + R[a].phase < R[b].dom * 10 + R[b].phase;
        });
    } else if (stream == "gen_cold") {
        for (std::size_t i = 0; i < R.size(); ++i) {
            if (R[i].dom == 3 && R[i].phase == 1) { order.push_back(i); }
        }
    } else if (stream == "gen_after_prefill") {
        for (std::size_t i = 0; i < R.size(); ++i) {
            if (R[i].dom == 3) { order.push_back(i); }
        }
        segment = true;
    } else {
        throw std::runtime_error("unknown stream " + stream);
    }
    std::vector<Window> windows;
    if (segment) {
        std::size_t start = 0;
        while (start < order.size()) {
            std::size_t end = start;
            while (end < order.size() && R[order[end]].phase == R[order[start]].phase) { ++end; }
            for (std::size_t i = start; i < end; i += static_cast<std::size_t>(W)) {
                Window w;
                w.records.assign(order.begin() + static_cast<std::ptrdiff_t>(i),
                                 order.begin() + static_cast<std::ptrdiff_t>(std::min(end, i + W)));
                w.counted = R[order[start]].phase == 1;
                windows.push_back(std::move(w));
            }
            start = end;
        }
    } else {
        for (std::size_t i = 0; i + static_cast<std::size_t>(W) <= order.size(); i += W) {
            Window w;
            w.records.assign(order.begin() + static_cast<std::ptrdiff_t>(i),
                             order.begin() + static_cast<std::ptrdiff_t>(i + W));
            windows.push_back(std::move(w));
        }
    }
    return windows;
}

[[noreturn]] void usage() {
    std::cerr
        << "usage: ninfer_expert_cache_replay ROUTING_BIN --stream S --K N [--W 4]\n"
           "  [--partitions P] [--rule engine|f0k_single_pass|f0m_window_start_free]\n"
           "  [--admit THRESHOLD HALF_LIFE_TOKENS] [--evict-tau WINDOWS] [--cap N] [--out FILE]\n"
           "  [--publish-lag N]  also report device hits when a fill becomes usable N windows\n"
           "                     after the window that decided it (0 = the next window)\n";
    std::exit(2);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(); }
    std::string path = argv[1], stream = "gen_after_prefill", out_path;
    int W = 4, lag = -1;
    PolicyConfig c;
    c.layers  = kLayers;
    c.experts = kExperts;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next           = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); }
            return argv[++i];
        };
        if (a == "--stream") {
            stream = next();
        } else if (a == "--K") {
            c.slots_per_layer = std::stoi(next());
        } else if (a == "--W") {
            W = std::stoi(next());
        } else if (a == "--partitions") {
            c.partitions = std::stoi(next());
        } else if (a == "--cap") {
            c.insert_cap = std::stoi(next());
        } else if (a == "--admit") {
            c.admission        = true;
            c.admit_threshold  = std::stod(next());
            c.half_life_tokens = std::stod(next());
        } else if (a == "--evict-tau") {
            c.evict_tau_windows = std::stod(next());
            if (c.half_life_tokens <= 0.0) {
                c.half_life_tokens = ninfer::RoutedExpertOptions{}.half_life_tokens;
            }
        } else if (a == "--rule") {
            const std::string r = next();
            if (r == "engine") {
                c.rule = Rule::kEngine;
            } else if (r == "f0k_single_pass") {
                c.rule = Rule::kF0KSinglePass;
            } else if (r == "f0m_window_start_free") {
                c.rule = Rule::kF0MWindowStartFree;
            } else {
                usage();
            }
        } else if (a == "--publish-lag") {
            lag = std::stoi(next());
        } else if (a == "--out") {
            out_path = next();
        } else {
            usage();
        }
    }
    try {
        const std::vector<Record> R       = load(path);
        const std::vector<Window> windows = make_windows(R, stream, W);
        ExpertCachePolicy policy(c);
        LayerDecision d;
        std::vector<std::int32_t> ids;
        std::vector<Routed> routed;
        std::FILE* out = out_path.empty() ? stdout : std::fopen(out_path.c_str(), "w");
        if (out == nullptr) { throw std::runtime_error("cannot write " + out_path); }
        std::int64_t H = 0, A = 0, M = 0, F = 0, counted = 0, DH = 0;
        // Window index of each (layer, expert) insertion, for --publish-lag.
        std::vector<std::int64_t> inserted_at(static_cast<std::size_t>(kLayers) * kExperts, -1);
        std::int64_t wi       = -1;
        double decide_seconds = 0.0;
        for (const Window& w : windows) {
            ++wi;
            std::int64_t h = 0, a = 0, m = 0, f = 0, dh = 0;
            for (int l = 0; l < kLayers; ++l) {
                ids.clear();
                for (const std::size_t r : w.records) {
                    const std::uint16_t* e = R[r].experts[l];
                    if (std::any_of(e, e + kTopK, [](std::uint16_t x) { return x == 0xFFFF; })) {
                        continue;
                    }
                    ids.insert(ids.end(), e, e + kTopK);
                }
                count_routed(ids, kExperts, routed);
                for (const Routed& x : routed) {
                    const auto at = inserted_at[static_cast<std::size_t>(l * kExperts + x.expert)];
                    if (policy.slot_of(l, x.expert) >= 0 && at <= wi - 1 - std::max(lag, 0)) {
                        dh += x.count;
                    }
                }
                const auto t0 = std::chrono::steady_clock::now();
                policy.decide(l, routed, static_cast<std::int32_t>(w.records.size()), d);
                decide_seconds +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                h += d.hits;
                a += d.accesses;
                m += static_cast<std::int64_t>(d.missing.size());
                f += static_cast<std::int64_t>(d.inserts.size());
                for (const Insert& ins : d.inserts) {
                    inserted_at[static_cast<std::size_t>(l * kExperts + ins.expert)] = wi;
                }
            }
            if (!w.counted) { continue; }
            ++counted;
            DH += dh;
            H += h;
            A += a;
            M += m;
            F += f;
            std::fprintf(out, "%lld %lld %lld %lld\n", static_cast<long long>(h),
                         static_cast<long long>(a), static_cast<long long>(m),
                         static_cast<long long>(f));
        }
        if (out != stdout) { std::fclose(out); }
        std::fprintf(
            stderr,
            "stream %s K %d P %d windows %lld (all %zu): hit %.4f missing/w %.2f "
            "inserts/w %.2f, decide %.1f us/window (47 layers)\n",
            stream.c_str(), c.slots_per_layer, c.partitions, static_cast<long long>(counted),
            windows.size(),
            static_cast<double>(H) / static_cast<double>(std::max<std::int64_t>(A, 1)),
            static_cast<double>(M) / static_cast<double>(std::max<std::int64_t>(counted, 1)),
            static_cast<double>(F) / static_cast<double>(std::max<std::int64_t>(counted, 1)),
            decide_seconds * 1e6 / static_cast<double>(windows.size()));
        if (lag >= 0) {
            std::fprintf(stderr, "publish lag %d window(s): device hit %.4f\n", lag,
                         static_cast<double>(DH) /
                             static_cast<double>(std::max<std::int64_t>(A, 1)));
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
