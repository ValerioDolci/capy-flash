// Per-layer slot demand curves of Qwen3.8-Flash-Next at tp 2 from NINFER_QWEN4_ROUTING_DUMP
// files: every dump is replayed through the engine's expert-cache policy (ExpertCachePolicy, the
// engine rule with the RoutedExpertOptions defaults) once per slot count of a grid, every layer
// holding that many slots per GPU, and for every layer and slot count the tool prints the
// missing experts summed over the decode windows and the modelled GPU wait summed over the same
// windows. tools/qwen4_slot_curves.py turns these tables into
// src/models/qwen3_5/qwen4_slot_curves.h (layer_slot_profile).
//
// Window semantics follow Qwen4OffloadExperts with keep_decode_residency (the default):
// - prompt windows leave residency and frequencies alone (count_prompt);
// - a request that starts a new conversation (--new-prompts) has its prompt's routing counts
//   applied to the frequencies when its first decode window begins (apply_prompt, cap -1);
// - the first `boost_windows` decode windows after a prompt insert up to `boost_cap` per GPU;
// - a decode window decides each layer once on its whole routing (the MTP layer's calls merged).
// Not modelled: lending (the lent layers refill with what they held, so it is only the frequency
// update of a new conversation's lent layers that differs), fill latency and the elastic columns'
// shrinking with KV growth.
//
// Dump: records {i32 window, i32 kind (1 prompt, 0 decode), i32 layer, i32 tokens} followed by
// tokens x 10 i32 expert ids (-1: a column that routed nothing).
//
// Wait model (per layer and decode window, m = experts missing on both GPUs): the GPU waits
// max(0, a * m - b) us for the CPU partial (a: CPU time per expert, b: the GPU work that hides
// it). The defaults were measured on a Ryzen 9 / RTX 5080 + RTX 5070 Ti rig with the GSQ edition.

#include "core/expert_cache_policy.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ninfer::expert_cache;

constexpr std::int32_t kExperts = 512;
constexpr std::int32_t kTopK    = 10;

struct LayerWindow {
    std::int32_t layer  = 0;
    std::int32_t tokens = 0; // routed columns
    std::vector<Routed> routed;
};

struct Window {
    bool prompt        = false;
    bool request_start = false; // first prompt window after a decode window (or the first)
    std::vector<LayerWindow> layers;
};

struct Options {
    std::vector<std::int32_t> grid;
    std::int32_t layers        = 49;
    std::int32_t partitions    = 2;
    std::int32_t insert_cap    = 2;
    double admit_threshold     = 2.0;
    double half_life_tokens    = 256.0;
    double evict_tau_windows   = 4.0;
    std::int32_t boost_windows = 24;
    std::int32_t boost_cap     = 8;
    std::string new_prompts    = "all"; // all | first | none
    double wait_per_expert_us  = 36.0;
    double wait_hidden_us      = 43.0;
    unsigned threads           = 0;
};

std::vector<Window> load(const std::string& path, std::int32_t layers) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot open " + path); }
    std::vector<Window> windows;
    std::map<std::int32_t, std::vector<std::int32_t>> ids; // layer -> ids of the open window
    std::map<std::int32_t, std::int32_t> tokens;
    std::int32_t open = -1;
    bool open_prompt  = false;
    bool last_decode  = true;
    auto close        = [&]() {
        if (open < 0) { return; }
        Window w;
        w.prompt        = open_prompt;
        w.request_start = open_prompt && last_decode;
        for (auto& [l, v] : ids) {
            LayerWindow lw;
            lw.layer  = l;
            lw.tokens = tokens[l];
            count_routed(v, kExperts, lw.routed);
            if (!lw.routed.empty()) { w.layers.push_back(std::move(lw)); }
        }
        last_decode = !open_prompt;
        windows.push_back(std::move(w));
        ids.clear();
        tokens.clear();
    };
    std::vector<std::int32_t> row;
    for (;;) {
        std::int32_t h[4];
        if (!in.read(reinterpret_cast<char*>(h), sizeof(h))) { break; }
        const std::int32_t win = h[0], kind = h[1], layer = h[2], t = h[3];
        if (layer < 0 || layer >= layers || t < 0 || (kind != 0 && kind != 1)) {
            throw std::runtime_error("bad record in " + path);
        }
        row.resize(static_cast<std::size_t>(t) * kTopK);
        if (!in.read(reinterpret_cast<char*>(row.data()),
                     static_cast<std::streamsize>(row.size() * sizeof(std::int32_t)))) {
            throw std::runtime_error("truncated " + path);
        }
        if (win != open) {
            close();
            open        = win;
            open_prompt = kind == 1;
        }
        auto& v = ids[layer];
        for (std::int32_t c = 0; c < t; ++c) {
            const std::int32_t* r = row.data() + static_cast<std::size_t>(c) * kTopK;
            if (r[0] < 0) { continue; }
            ++tokens[layer];
            v.insert(v.end(), r, r + kTopK);
        }
    }
    close();
    return windows;
}

struct Curve {
    std::vector<double> missing; // [layer]
    std::vector<double> wait;    // [layer]
};

Curve replay(const std::vector<Window>& windows, const Options& o, std::int32_t slots) {
    PolicyConfig c;
    c.layers                   = o.layers;
    c.experts                  = kExperts;
    c.partitions               = o.partitions;
    c.slots_per_layer          = slots * o.partitions;
    c.admission                = o.admit_threshold > 0.0;
    c.admit_threshold          = o.admit_threshold;
    c.half_life_tokens         = o.half_life_tokens;
    c.evict_tau_windows        = o.evict_tau_windows;
    c.recency_weighted_prompts = true;
    c.insert_cap               = o.insert_cap * o.partitions;
    c.rule                     = Rule::kEngine;
    ExpertCachePolicy policy(c);
    Curve out;
    out.missing.assign(static_cast<std::size_t>(o.layers), 0.0);
    out.wait.assign(static_cast<std::size_t>(o.layers), 0.0);
    LayerDecision d;
    std::vector<std::vector<std::int64_t>> pending(static_cast<std::size_t>(o.layers),
                                                   std::vector<std::int64_t>(kExperts, 0));
    std::vector<std::int64_t> pending_tokens(static_cast<std::size_t>(o.layers), 0);
    bool pending_prompt = false, prompt_new = false, after_prompt = false;
    std::int32_t boost_left = 0, requests = 0;
    std::vector<Routed> routed;
    for (const Window& w : windows) {
        if (w.prompt) {
            if (w.request_start) {
                ++requests;
                prompt_new = o.new_prompts == "all" || (o.new_prompts == "first" && requests == 1);
            }
            if (boost_left > 0) {
                boost_left = 0;
                policy.set_insert_cap(o.insert_cap * o.partitions);
            }
            for (const LayerWindow& lw : w.layers) {
                if (prompt_new) {
                    auto& p = pending[static_cast<std::size_t>(lw.layer)];
                    for (const Routed& r : lw.routed) {
                        p[static_cast<std::size_t>(r.expert)] += r.count;
                    }
                    pending_tokens[static_cast<std::size_t>(lw.layer)] += lw.tokens;
                    pending_prompt = true;
                }
                policy.count_prompt(lw.layer, lw.routed, lw.tokens, d);
            }
            after_prompt = true;
            continue;
        }
        if (pending_prompt) {
            for (std::int32_t l = 0; l < o.layers; ++l) {
                auto& t = pending_tokens[static_cast<std::size_t>(l)];
                auto& p = pending[static_cast<std::size_t>(l)];
                if (t > 0) {
                    routed.clear();
                    for (std::int32_t e = 0; e < kExperts; ++e) {
                        if (p[static_cast<std::size_t>(e)] > 0) {
                            routed.push_back(Routed{
                                e, static_cast<std::int32_t>(p[static_cast<std::size_t>(e)])});
                        }
                    }
                    policy.count_prompt(l, routed, static_cast<std::int32_t>(t), d,
                                        /*frequencies=*/true);
                }
                t = 0;
                std::fill(p.begin(), p.end(), 0);
            }
            pending_prompt = false;
        }
        if (after_prompt && o.boost_windows > 0 && o.insert_cap > 0) {
            boost_left = o.boost_windows;
            policy.set_insert_cap(o.boost_cap * o.partitions);
        } else if (boost_left > 0 && --boost_left == 0) {
            policy.set_insert_cap(o.insert_cap * o.partitions);
        }
        after_prompt = false;
        prompt_new   = false;
        for (const LayerWindow& lw : w.layers) {
            policy.decide(lw.layer, lw.routed, lw.tokens, d);
            const auto m = static_cast<double>(d.missing.size());
            out.missing[static_cast<std::size_t>(lw.layer)] += m;
            out.wait[static_cast<std::size_t>(lw.layer)] +=
                std::max(0.0, o.wait_per_expert_us * m - o.wait_hidden_us);
        }
    }
    return out;
}

std::vector<std::int32_t> parse_list(const std::string& s) {
    std::vector<std::int32_t> v;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ',')) { v.push_back(std::stoi(x)); }
    return v;
}

[[noreturn]] void usage() {
    std::cerr << "usage: ninfer_qwen4_slot_curves ROUTING_DUMP [--grid 12,16,...] [--new-prompts "
                 "all|first|none]\n"
                 "  [--half-life T] [--admit F] [--evict-tau W] [--cap N] [--boost W N]\n"
                 "  [--wait A B] [--threads N]\n"
                 "prints: '# windows D' then 'slots layer missing wait_us' (sums over the D "
                 "decode windows), slots per GPU and layer\n";
    std::exit(2);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(); }
    Options o;
    for (std::int32_t s = 12; s <= 256; s += 4) { o.grid.push_back(s); }
    const std::string path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next           = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); }
            return argv[++i];
        };
        if (a == "--grid") {
            o.grid = parse_list(next());
        } else if (a == "--new-prompts") {
            o.new_prompts = next();
            if (o.new_prompts != "all" && o.new_prompts != "first" && o.new_prompts != "none") {
                usage();
            }
        } else if (a == "--half-life") {
            o.half_life_tokens = std::stod(next());
        } else if (a == "--admit") {
            o.admit_threshold = std::stod(next());
        } else if (a == "--evict-tau") {
            o.evict_tau_windows = std::stod(next());
        } else if (a == "--cap") {
            o.insert_cap = std::stoi(next());
        } else if (a == "--boost") {
            o.boost_windows = std::stoi(next());
            o.boost_cap     = std::stoi(next());
        } else if (a == "--wait") {
            o.wait_per_expert_us = std::stod(next());
            o.wait_hidden_us     = std::stod(next());
        } else if (a == "--threads") {
            o.threads = static_cast<unsigned>(std::stoi(next()));
        } else {
            usage();
        }
    }
    try {
        const std::vector<Window> windows = load(path, o.layers);
        std::int64_t decode               = 0;
        for (const Window& w : windows) { decode += w.prompt ? 0 : 1; }
        std::vector<Curve> curves(o.grid.size());
        const unsigned n =
            o.threads > 0 ? o.threads : std::max(1U, std::thread::hardware_concurrency() / 2);
        std::vector<std::thread> pool;
        std::size_t next = 0;
        std::mutex mu;
        for (unsigned t = 0; t < n; ++t) {
            pool.emplace_back([&]() {
                for (;;) {
                    std::size_t k;
                    {
                        std::lock_guard<std::mutex> lock(mu);
                        if (next >= o.grid.size()) { return; }
                        k = next++;
                    }
                    curves[k] = replay(windows, o, o.grid[k]);
                }
            });
        }
        for (auto& t : pool) { t.join(); }
        std::printf("# windows %lld\n", static_cast<long long>(decode));
        for (std::size_t k = 0; k < o.grid.size(); ++k) {
            for (std::int32_t l = 0; l < o.layers; ++l) {
                std::printf("%d %d %.0f %.0f\n", o.grid[k], l,
                            curves[k].missing[static_cast<std::size_t>(l)],
                            curves[k].wait[static_cast<std::size_t>(l)]);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
